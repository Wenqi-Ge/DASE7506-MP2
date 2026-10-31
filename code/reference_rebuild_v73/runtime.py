"""NumPy-only inference for the v73 candidate-selection ensemble."""
import json
from pathlib import Path
import numpy as np
from common import PROTOCOL
from .features import groups,rich_and_compact,quantitative,text_scores
from .stack import VIEWS,margin_scores,meta_matrix,text_columns

METHOD='v73_candidate_attention_ranker_ensemble_cpu'
NETWORKS={
 'compact':dict(width=128,layers=2,activation='gelu',final_norm=True,skip=False,dropout=.2,epochs=15,lr=.0001,decay=.001,balanced=False),
 'compact_balanced':dict(width=128,layers=2,activation='gelu',final_norm=True,skip=False,dropout=.2,epochs=15,lr=.0001,decay=.001,balanced=True),
 'rich':dict(width=96,layers=1,activation='relu',final_norm=False,skip=True,dropout=.3,epochs=1,lr=.0003,decay=.08,balanced=True)}


def read_npz(path):
    with np.load(path,allow_pickle=False) as f:return {k:f[k] for k in f.files}


def gelu(x):return .5*x*(1+np.tanh(np.float32(np.sqrt(2/np.pi))*(x+np.float32(.044715)*x**3)))
def layer_norm(x,w,b):return (x-x.mean(-1,keepdims=True))/np.sqrt(x.var(-1,keepdims=True)+np.float32(1e-5))*w+b


def neural_inputs(arrays,rich,compact,repair,family):
    e=np.asarray(arrays['embeddings'],np.float32)
    if family.startswith('compact'):return np.concatenate((e,compact),-1)
    return np.concatenate((e,e-e.mean(1,keepdims=True,dtype=np.float64).astype(np.float32),rich,repair),-1)


def neural_logits(x,state,config):
    def linear(value,name):return value@state[name+'.weight'].T+state[name+'.bias']
    h=linear(x,'stem');n,t,width=h.shape;part=width//4
    for layer in range(config['layers']):
        prefix=f'layers.{layer}.';a=layer_norm(h,state[prefix+'norm_attention.weight'],state[prefix+'norm_attention.bias'])
        q,k,v=np.split(linear(a,prefix+'qkv'),3,-1)
        q,k,v=[item.reshape(n,t,4,part).transpose(0,2,1,3) for item in (q,k,v)]
        score=(q@k.swapaxes(-1,-2))/np.float32(np.sqrt(part));weight=np.exp(score-score.max(-1,keepdims=True));weight/=weight.sum(-1,keepdims=True)
        attended=(weight@v).transpose(0,2,1,3).reshape(n,t,width);h=h+linear(attended,prefix+'attended')
        a=layer_norm(h,state[prefix+'norm_feedforward.weight'],state[prefix+'norm_feedforward.bias'])
        a=linear(a,prefix+'expand');a=np.maximum(a,0) if config['activation']=='relu' else gelu(a)
        h=h+linear(a,prefix+'contract')
    if config['final_norm']:h=layer_norm(h,state['final_norm.weight'],state['final_norm.bias'])
    output=linear(h,'head')[...,0]
    if config['skip']:output+=(x@state['direct.weight'].T)[...,0]
    return output


def neural_score(x,archive,family):
    result=np.zeros(x.shape[:2],np.float64);config=NETWORKS[family]
    for member in range(len(archive['mean'])):
        parameters={k[3:]:v[member].astype(np.float32) for k,v in archive.items() if k.startswith('p::')}
        for start in range(0,len(x),128):
            sl=slice(start,start+128);z=np.clip((x[sl]-archive['mean'][member])/archive['scale'][member],-8,8).astype(np.float32)
            logit=neural_logits(z,parameters,config);result[sl]+=1/(1+np.exp(-np.clip(logit,-30,30)))
    return result/len(archive['mean'])


def tree_score(x,model):
    value=np.zeros(len(x),np.float64);powers=(1<<np.arange(model['features'].shape[1],dtype=np.int64))[None,None,:]
    for start in range(0,len(x),128):
        code=((x[start:start+128,model['features']]>model['borders'][None])*powers).sum(-1)
        value[start:start+128]=model['leaves'][model['offsets'][None]+code].sum(-1)
    value=value*model['scale'].item()+model['bias'].item()
    return (1/(1+np.exp(-np.clip(value,-30,30))))


def pool_choices(probability,bank,amount):
    _,same,valid=groups(bank);average=(same@probability[...,None])[...,0]/np.maximum(same.sum(-1),1)
    score=(1-amount)*probability+amount*average
    return [int(j) for j in np.where(valid,score,-1e20).argmax(-1)]


def components(bank,arrays,checkpoint,names):
    root=Path(checkpoint);rich,compact,repair=rich_and_compact(bank);result={}
    trees=[name for name in names if name.startswith('tree')]
    if trees:
        stored=read_npz(root/'stack.npz');models=[]
        for i,(mode,_,_) in enumerate(VIEWS):models.append(dict(mode=mode,mean=stored[f'base{i}_mean'],scale=stored[f'base{i}_scale'],weights=stored[f'base{i}_weights']))
        base=np.stack([margin_scores(m,bank,arrays,rich) for m in models],-1).astype(np.float32)
        quantity,schemas=quantitative(bank);projection=dict(mean=stored['projection_mean'],components=stored['projection_components'])
        basic=meta_matrix(bank,arrays,rich,base,projection,quantity)
        extended=None
        for name in trees:
            if name=='tree_log':
                if extended is None:
                    text=text_scores(bank,stored['text_idf'],stored['text_coef']).astype(np.float32)
                    extended=np.concatenate((basic,schemas.reshape(len(bank)*4,-1),text_columns(text).reshape(len(bank)*4,-1),repair.reshape(len(bank)*4,-1)),1)
                matrix=extended
            else:matrix=basic
            result[name]=tree_score(matrix,read_npz(root/(name+'.npz'))).reshape(len(bank),4)
    for name in names:
        if name in NETWORKS:result[name]=neural_score(neural_inputs(arrays,rich,compact,repair,name),read_npz(root/(name+'.npz')),name)
    return result


def predict(bank,arrays,checkpoint):
    if not bank:return []
    if checkpoint is None:raise ValueError('A fitted checkpoint is required')
    root=Path(checkpoint);config=json.loads((root/'config.json').read_text())
    if config['method']!=METHOD or config['protocol']!=PROTOCOL:raise ValueError('Wrong rebuilt model')
    weights=config['weights']
    if any(w<0 or not np.isfinite(w) for w in weights.values()) or not np.isclose(sum(weights.values()),1):raise ValueError('Invalid weights')
    values=components(bank,arrays,root,list(weights))
    return pool_choices(sum(w*values[k] for k,w in weights.items()),bank,config['pooling'])
