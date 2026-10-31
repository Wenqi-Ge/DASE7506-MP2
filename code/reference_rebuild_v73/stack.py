"""Multi-view linear ranking and tree-input construction."""
import numpy as np
from .features import groups,token_counts,HASH_DIM,text_scores

VIEWS=(('raw',False,.001),('context',False,.0001),('cluster',False,.0003),('raw',True,.001),('cluster',True,.0001))


def balanced_copies(bank,arrays,labels):
    """Construct 1-positive/3-negative groups and their inverse views."""
    rng=np.random.default_rng(7506);extra=[];vectors=[];targets=[];origins=list(range(len(bank)))
    parsed,_,valid=groups(bank)
    for i,row in enumerate(bank):
        positive=np.flatnonzero(labels[i]);negative=np.flatnonzero(valid[i]&~labels[i])
        if not len(negative):continue
        for positives in (1,3):
            selected=np.r_[rng.choice(positive,positives,replace=True),rng.choice(negative,4-positives,replace=True)];rng.shuffle(selected)
            extra.append(dict(row,candidates=[dict(row['candidates'][int(k)],index=j) for j,k in enumerate(selected)]))
            vectors.append(arrays['embeddings'][i,selected]);targets.append(labels[i,selected]);origins.append(i)
    e=np.concatenate((arrays['embeddings'],np.asarray(vectors,np.float32)),0) if vectors else arrays['embeddings'].copy()
    y=np.concatenate((labels,np.asarray(targets,bool)),0) if targets else labels.copy()
    return bank+extra,{'embeddings':e},y,np.asarray(origins,np.int64)


def view_matrix(bank,arrays,rich,mode):
    e=np.asarray(arrays['embeddings'],np.float32);mean=e.mean(1,keepdims=True,dtype=np.float64).astype(np.float32)
    if mode=='raw':parts=(e,rich)
    elif mode=='context':parts=(e,e-mean,e*mean,rich)
    elif mode=='cluster':
        _,same,_=groups(bank);center=((same.astype(np.float64)@e.astype(np.float64))/np.maximum(same.sum(-1,keepdims=True),1)).astype(np.float32)
        parts=(e,e-mean,e*mean,center,center-mean,e*center,rich)
    else:raise ValueError('Unknown view')
    return np.concatenate(parts,-1).reshape(len(bank)*4,-1)


def training_pairs(bank,labels):
    parsed,_,valid=groups(bank);positive=[];negative=[];weights=[]
    for i,row in enumerate(parsed):
        yes=np.flatnonzero(labels[i]);no=np.flatnonzero(valid[i]&~labels[i])
        if not len(no):continue
        counts={v:row.count(v) for v in row if v is not None};local=[]
        for a in yes:
            for b in no:local.append((4*i+a,4*i+b,1/(counts[row[a]]*counts[row[b]])))
        total=sum(w for _,_,w in local)
        for a,b,w in local:positive.append(a);negative.append(b);weights.append(w/total)
    return np.asarray(positive),np.asarray(negative),np.asarray(weights,np.float64)


def fit_margin(bank,arrays,labels,rich,mode,c_value):
    from scipy.optimize import minimize
    from scipy.special import expit
    x=view_matrix(bank,arrays,rich,mode);mean=x.mean(0,dtype=np.float64).astype(np.float32)
    scale=x.std(0,dtype=np.float64);scale=np.where(scale<1e-10,1.,scale).astype(np.float32)
    z=np.clip((x-mean)/scale,-8,8).astype(np.float32);yes,no,weight=training_pairs(bank,labels);weight/=weight.sum()
    differences=(z[yes]-z[no]).astype(np.float64)
    # Reference C uses the SUM of weights on both signed copies. Our data term
    # is a weighted mean, so retain the exact equivalent regularization scale.
    penalty=1/(c_value*2*len(yes))
    def loss(w):
        margin=differences@w
        return float(weight@np.logaddexp(0,-margin)+.5*penalty*(w@w)), -(differences.T@(weight*expit(-margin)))+penalty*w
    result=minimize(loss,np.zeros(z.shape[-1]),jac=True,method='L-BFGS-B',options=dict(maxiter=300,ftol=1e-9,gtol=1e-6))
    if not result.success:raise RuntimeError('Base optimizer: '+str(result.message))
    return dict(mean=mean,scale=scale,weights=result.x.astype(np.float32),mode=mode,iterations=int(result.nit),C=c_value,penalty=penalty,pairs=len(yes))


def margin_scores(model,bank,arrays,rich):
    matrix=view_matrix(bank,arrays,rich,model['mode']);z=np.clip((matrix-model['mean'])/model['scale'],-8,8)
    logits=np.einsum('nd,d->n',z,model['weights'],dtype=np.float64,optimize=False)
    return (1/(1+np.exp(-np.clip(logits,-30,30)))).reshape(len(bank),4)


def meta_matrix(bank,arrays,rich,probability,projection,quantity):
    p=np.clip(probability,1e-5,1-1e-5).astype(np.float32);logit=np.log(p/(1-p))
    rank=((p[:,:,None,:]>p[:,None,:,:]).sum(2)+.5*(p[:,:,None,:]==p[:,None,:,:]).sum(2))/4
    _,same,_=groups(bank);same/=np.maximum(same.sum(-1,keepdims=True),1)
    evidence=np.concatenate((p,logit,logit-logit.mean(1,keepdims=True),p-p.max(1,keepdims=True),rank,same@p),-1)
    e=arrays['embeddings'];z=(e-projection['mean'])@projection['components'].T;mean=z.mean(1,keepdims=True,dtype=np.float64).astype(np.float32)
    semantic=np.concatenate((z,z-mean,z*mean,np.broadcast_to(mean,z.shape)),-1)
    return np.concatenate((rich,evidence,semantic,quantity),-1).reshape(len(bank)*4,-1).astype(np.float32)


def text_matrix(bank):
    from scipy import sparse
    indices=[];data=[];offset=[0]
    for row in bank:
        for candidate in row['candidates']:
            counts=token_counts(candidate['text']);indices.extend(counts);data.extend(counts.values());offset.append(len(indices))
    return sparse.csr_matrix((np.asarray(data,np.float64),np.asarray(indices,np.int32),np.asarray(offset,np.int64)),shape=(len(bank)*4,HASH_DIM))


def fit_text(counts,bank,labels,fit):
    from scipy import sparse
    from sklearn.linear_model import LogisticRegression
    rows=(fit[:,None]*4+np.arange(4)).ravel();chosen=counts[rows]
    df=np.asarray((chosen!=0).sum(0)).ravel();idf=(np.log((len(rows)+1)/(df+1))+1).astype(np.float32)
    matrix=counts.copy();matrix.data=1+np.log(matrix.data);matrix=matrix.multiply(idf).tocsr()
    norm=np.sqrt(np.asarray(matrix.multiply(matrix).sum(1)).ravel()).clip(1e-12)
    matrix=sparse.diags(1/norm)@matrix
    subset=[bank[i] for i in fit];yes,no,w=training_pairs(subset,labels[fit]);source=rows
    delta=matrix[source[yes]]-matrix[source[no]];w/=w.mean()
    estimator=LogisticRegression(C=1.,fit_intercept=False,solver='liblinear',dual=True,max_iter=1000,random_state=7536)
    estimator.fit(sparse.vstack((delta,-delta),format='csr'),np.r_[np.ones(len(yes)),np.zeros(len(yes))],sample_weight=np.r_[w,w])
    if estimator.n_iter_.max()>=1000:raise RuntimeError('Lexical optimizer did not converge')
    return idf,estimator.coef_[0].astype(np.float32)


def text_columns(probability):
    p=np.clip(probability,1e-5,1-1e-5)
    return np.stack((p,np.log(p/(1-p)),p-p.mean(1,keepdims=True),p-p.max(1,keepdims=True)),-1)
