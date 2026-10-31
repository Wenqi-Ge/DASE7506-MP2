"""Fit the v73 candidate-selection ensemble on supplied TRAIN. Validation selects the fusion configuration; TEST is excluded from fitting."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='4'
os.environ['OPENBLAS_CORETYPE']='Haswell';os.environ['NPY_DISABLE_CPU_FEATURES']='X86_V4,AVX512_ICL,AVX512_SPR'
os.environ.setdefault('CUDA_VISIBLE_DEVICES','0');os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'

import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.decomposition import PCA
from catboost import CatBoostClassifier,CatBoostRanker,Pool
import torch
from torch.nn import functional as F
from common import DATA,PROTOCOL,load_inputs,parse_answer,read_jsonl,check_data
from .features import rich_and_compact,quantitative,groups,text_scores
from .stack import VIEWS,balanced_copies,fit_margin,margin_scores,meta_matrix,text_matrix,fit_text,text_columns
from .net import SetVerifier
from .runtime import METHOD,NETWORKS,neural_inputs,neural_logits,neural_score,tree_score,pool_choices


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def emit(value):print(json.dumps(value),flush=True)


def labels(bank,split):
    if split not in ('train','validation'):raise ValueError('TEST/audit references forbidden during fitting')
    check_data(DATA,(split,));refs=read_jsonl(DATA/f'gsm8k_{split}.jsonl');y=np.zeros((len(bank),4),bool)
    for i,(row,ref) in enumerate(zip(bank,refs,strict=True)):
        if (row['id'],row['question'])!=(ref['id'],ref['question']):raise ValueError('Alignment')
        answer=parse_answer(ref['answer'])
        if answer is None:raise ValueError('Invalid reference')
        y[i]=[parse_answer(c['text'])==answer for c in row['candidates']]
    if not y.any(-1).all():raise ValueError('No correct candidate')
    return y


def export_tree(model):
    with tempfile.NamedTemporaryFile(suffix='.json') as file:
        model.save_model(file.name,format='json');document=json.loads(Path(file.name).read_text())
    mapping={item['feature_index']:item['flat_feature_index'] for item in document['features_info']['float_features']}
    trees=document['oblivious_trees'];depth=max(len(tree.get('splits') or []) for tree in trees)
    features=np.zeros((len(trees),depth),np.int32);borders=np.full((len(trees),depth),np.inf,np.float32);offsets=[];leaves=[]
    for i,tree in enumerate(trees):
        offsets.append(len(leaves));leaves.extend(tree['leaf_values'])
        for j,split in enumerate(tree.get('splits') or []):
            if split['split_type']!='FloatFeature':raise ValueError('Unexpected split')
            features[i,j]=mapping[split['float_feature_index']];borders[i,j]=split['border']
    scale,bias=document['scale_and_bias']
    return dict(features=features,borders=borders,offsets=np.asarray(offsets,np.int32),leaves=np.asarray(leaves,np.float64),scale=np.asarray(scale),bias=np.asarray(bias[0]))


def full_base_fit(bank,arrays,y,rich):
    models=[];aug=None
    for mode,balanced,penalty in VIEWS:
        if balanced and aug is None:
            ab,aa,ay,origin=balanced_copies(bank,arrays,y);ar,_,_=rich_and_compact(ab);aug=(ab,aa,ay,ar)
        b,a,t,r=aug if balanced else (bank,arrays,y,rich)
        models.append(fit_margin(b,a,t,r,mode,penalty))
    return models


def candidate_count(prob,bank,y,pool):
    choice=pool_choices(prob,bank,pool);return int(y[np.arange(len(y)),choice].sum())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    run=args.output;run.mkdir(parents=True,exist_ok=False);(run/'search').mkdir();(run/'checkpoint').mkdir();code=Path(__file__).resolve().parents[1]
    source=['submission_v73.py']+[str(p.relative_to(code)) for p in sorted((code/'reference_rebuild_v73').glob('*.py'))]
    sources={name:sha(code/name) for name in source}
    plan=dict(at_utc=datetime.now(timezone.utc).isoformat(),method=METHOD,source_sha256=sources,
      network_configurations=NETWORKS,base_views=VIEWS,base_folds=5,regularization_definition_url='https://scikit-learn.org/1.8/modules/linear_model.html#logistic-regression',
      fusion='Five components in1/6 steps;tree total1/3..2/3;compact-family total>=1/6;pool0/.6/1. Highest validation correct then trap correct then candidate Brier.',
      test_accessed=False,official_sha256={p:sha(code/p) for p in ('common.py','predict.py','score_predictions.py')},manifest_sha256=sha(DATA/'manifest.json'),
      limitations=['Benchmark reproduction is not an independent blind evaluation.','Validation is adaptive development.','The fixed-base OOF meta holdout is not full nested pipeline CV.'],
      ai_assistance='OpenAI Codex substantially assisted implementation, experiments and documentation.')
    write(run/'plan.json',plan)
    for name in source:
        target=run/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(code/name,target)
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    start=time.perf_counter();bank,arrays=load_inputs('train');vb,va=load_inputs('validation');y,vy=labels(bank,'train'),labels(vb,'validation')
    rich,compact,repair=rich_and_compact(bank);vr,vc,vrepair=rich_and_compact(vb)
    quantity,schemas=quantitative(bank);vq,vschemas=quantitative(vb)
    np.savez(run/'search/features.npz',rich=rich,compact=compact,repair=repair,quantity=quantity,schemas=schemas,
             validation_rich=vr,validation_compact=vc,validation_repair=vrepair,validation_quantity=vq,validation_schemas=vschemas)
    emit(dict(stage='features',seconds=time.perf_counter()-start,rich_dim=rich.shape[-1],compact_dim=compact.shape[-1]))
    folds=list(StratifiedKFold(5,shuffle=True,random_state=7526).split(np.arange(len(bank)),y.sum(-1)))
    oof=np.zeros((len(bank),4,len(VIEWS)),np.float32);text_oof=np.zeros((len(bank),4),np.float32);records=[];counts=text_matrix(bank)
    for fold,(fit,held) in enumerate(folds):
        tick=time.perf_counter();fb=[bank[i] for i in fit];fa={'embeddings':arrays['embeddings'][fit]};fy=y[fit]
        hb=[bank[i] for i in held];ha={'embeddings':arrays['embeddings'][held]}
        fitted=full_base_fit(fb,fa,fy,rich[fit]);fold_archive={}
        for j,model in enumerate(fitted):
            oof[held,:,j]=margin_scores(model,hb,ha,rich[held])
            for key in ('mean','scale','weights'):fold_archive[f'base{j}_{key}']=model[key]
        np.savez(run/'search'/f'fold{fold}-bases.npz',**fold_archive)
        idf,coef=fit_text(counts,bank,y,fit);text_oof[held]=text_scores(hb,idf,coef)
        row=dict(fold=fold,fit_questions=len(fit),held_questions=len(held),fit_hash=hashlib.sha256(fit.astype(np.int64).tobytes()).hexdigest(),
                 held_hash=hashlib.sha256(held.astype(np.int64).tobytes()).hexdigest(),seconds=time.perf_counter()-tick)
        records.append(row);emit(dict(stage='base_oof',**row))
    bases=full_base_fit(bank,arrays,y,rich);base_val=np.stack([margin_scores(m,vb,va,vr) for m in bases],-1).astype(np.float32)
    idf,coef=fit_text(counts,bank,y,np.arange(len(bank)));text_val=text_scores(vb,idf,coef).astype(np.float32)
    pca=PCA(n_components=64,svd_solver='randomized',random_state=7526).fit(arrays['embeddings'].reshape(-1,1024))
    projection=dict(mean=pca.mean_.astype(np.float32),components=pca.components_.astype(np.float32))
    geometry={f'base{i}_{key}':m[key] for i,m in enumerate(bases) for key in ('mean','scale','weights')}
    geometry.update(projection_mean=projection['mean'],projection_components=projection['components'],text_idf=idf,text_coef=coef)
    np.savez(run/'search/stack.npz',**geometry);np.savez(run/'search/base-evidence.npz',oof=oof,validation=base_val,text_oof=text_oof,text_validation=text_val)
    basic=meta_matrix(bank,arrays,rich,oof,projection,quantity);vbasic=meta_matrix(vb,va,vr,base_val,projection,vq)
    extended=np.concatenate((basic,schemas.reshape(len(bank)*4,-1),text_columns(text_oof).reshape(len(bank)*4,-1),repair.reshape(len(bank)*4,-1)),1)
    vextended=np.concatenate((vbasic,vschemas.reshape(len(vb)*4,-1),text_columns(text_val).reshape(len(vb)*4,-1),vrepair.reshape(len(vb)*4,-1)),1)
    parsed,same,valid=groups(bank);vparsed,vsame,vvalid=groups(vb)
    informative=np.flatnonzero([len(set(a for a in row if a is not None))>=2 for row in parsed])
    fit,held=train_test_split(informative,test_size=.2,random_state=7546,stratify=y[informative].sum(-1))
    rows=lambda q:(q[:,None]*4+np.arange(4)).ravel();fi,hi,allrows=rows(fit),rows(held),rows(informative)
    predictions={};tree_records=[]
    for name,depth,penalty,matrix,vmatrix in (('tree_pair',4,20.,basic,vbasic),('tree_log',6,40.,extended,vextended)):
        tick=time.perf_counter();options=dict(iterations=1200,depth=depth,learning_rate=.035,l2_leaf_reg=penalty,random_seed=7546,thread_count=4,verbose=False,allow_writing_files=False)
        if name=='tree_pair':
            pilot=CatBoostRanker(**options,loss_function='PairLogit')
            pilot.fit(Pool(matrix[fi],y[fit].ravel(),group_id=np.repeat(np.arange(len(fit)),4)),eval_set=Pool(matrix[hi],y[held].ravel(),group_id=np.repeat(np.arange(len(held)),4)),early_stopping_rounds=100)
            options['iterations']=max(100,pilot.best_iteration_+1);model=CatBoostRanker(**options,loss_function='PairLogit')
            model.fit(Pool(matrix[allrows],y[informative].ravel(),group_id=np.repeat(np.arange(len(informative)),4)))
            native=model.predict(vmatrix)
        else:
            pilot=CatBoostClassifier(**options,loss_function='Logloss');pilot.fit(matrix[fi],y[fit].ravel(),eval_set=(matrix[hi],y[held].ravel()),early_stopping_rounds=100)
            options['iterations']=max(100,pilot.best_iteration_+1);model=CatBoostClassifier(**options,loss_function='Logloss');model.fit(matrix[allrows],y[informative].ravel());native=model.predict(vmatrix,prediction_type='RawFormulaVal')
        packed=export_tree(model);prob=tree_score(vmatrix,packed);expected=1/(1+np.exp(-np.clip(native,-30,30)))
        error=float(np.max(np.abs(prob-expected)))
        if error>1e-10:raise RuntimeError('Tree export mismatch')
        np.savez(run/'search'/(name+'.npz'),**packed);predictions[name]=prob.reshape(len(vb),4)
        row=dict(name=name,dimensions=matrix.shape[1],iterations=options['iterations'],validation_correct=candidate_count(predictions[name],vb,vy,1.),seconds=time.perf_counter()-tick,export_error=error)
        tree_records.append(row);emit(dict(stage='tree',**row))
    ab,aa,ay,origins=balanced_copies(bank,arrays,y);extra_r,extra_c,extra_p=rich_and_compact(ab[len(bank):])
    ar=np.concatenate((rich,extra_r));ac=np.concatenate((compact,extra_c));ap=np.concatenate((repair,extra_p))
    _,_,av=groups(ab);permutation=np.random.default_rng(13).permutation(len(bank));neural_folds=np.split(permutation,5);neural_records=[]
    for family,config in NETWORKS.items():
        raw=neural_inputs(arrays,rich,compact,repair,family);augraw=neural_inputs(aa,ar,ac,ap,family) if config['balanced'] else None
        members=[];means=[];scales=[];held_scores=np.zeros((len(bank),4),np.float32)
        fits=[(i,np.setdiff1d(np.arange(len(bank)),h),h,37) for i,h in enumerate(neural_folds)] if family!='rich' else [(i,np.arange(len(bank)),None,seed) for i,seed in enumerate((7506,7507,7508))]
        for member,fit,h,seed in fits:
            tick=time.perf_counter();torch.manual_seed(seed);rng=np.random.default_rng(seed)
            x=augraw if config['balanced'] else raw;target=ay if config['balanced'] else y;mask=av if config['balanced'] else valid
            selected=np.flatnonzero(np.isin(origins,fit)) if config['balanced'] else fit
            mean=x[selected].reshape(-1,x.shape[-1]).mean(0);scale=x[selected].reshape(-1,x.shape[-1]).std(0)+1e-6
            normalized=np.clip((x-mean)/scale,-8,8).astype(np.float32)
            tx,ty,tv=[torch.as_tensor(v,device='cuda') for v in (normalized,target.astype(np.float32),mask)]
            hard=(target.sum(-1)<mask.sum(-1)).astype(np.float32)
            net=SetVerifier(x.shape[-1],config).cuda();optimizer=torch.optim.AdamW(net.parameters(),lr=config['lr'],weight_decay=config['decay'])
            losses=[]
            for epoch in range(config['epochs']):
                net.train();total=0.
                for batch in np.array_split(rng.permutation(selected),max(1,int(np.ceil(len(selected)/64)))):
                    logit=net(tx[batch]);label=ty[batch];vv=tv[batch]
                    mass=torch.logsumexp(logit.masked_fill(~vv,-1e4),-1)-torch.logsumexp(logit.masked_fill(label==0,-1e4),-1)
                    binary=F.binary_cross_entropy_with_logits(logit,label,reduction='none').mean(-1)
                    if family=='rich':
                        pair=label[:,:,None]*(1-label[:,None,:])*vv[:,None,:]
                        rank=(F.softplus(-(logit[:,:,None]-logit[:,None,:]))*pair).sum((1,2))/pair.sum((1,2)).clamp_min(1)
                        loss=.5*rank.mean()+.5*mass.mean()+.15*binary.mean()
                    else:loss=((binary+.5*mass)*torch.as_tensor(1+hard[batch],device='cuda')).mean()
                    optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);optimizer.step();total+=float(loss.detach())*len(batch)
                losses.append(total/len(selected))
            parameters={k:v.detach().cpu().numpy().astype(np.float16) for k,v in net.state_dict().items()}
            net.load_state_dict({k:torch.as_tensor(v.astype(np.float32),device='cuda') for k,v in parameters.items()});net.eval()
            probe=np.clip((raw[:32]-mean)/scale,-8,8).astype(np.float32)
            with torch.no_grad():native=net(torch.as_tensor(probe,device='cuda')).cpu().numpy()
            deployed=neural_logits(probe,{k:v.astype(np.float32) for k,v in parameters.items()},config);error=float(np.max(np.abs(native-deployed)))
            if error>.001:raise RuntimeError('Neural export mismatch')
            if h is not None:
                values=neural_logits(np.clip((raw[h]-mean)/scale,-8,8),{k:v.astype(np.float32) for k,v in parameters.items()},config)
                held_scores[h]=1/(1+np.exp(-np.clip(values,-30,30)))
            members.append(parameters);means.append(mean);scales.append(scale)
            row=dict(family=family,member=member,seed=seed,fit_original_questions=len(fit),fit_derived_questions=len(selected),held_questions=0 if h is None else len(h),
                     fit_hash=hashlib.sha256(fit.astype(np.int64).tobytes()).hexdigest(),epochs=config['epochs'],losses=losses,export_error=error,seconds=time.perf_counter()-tick)
            neural_records.append(row);emit(dict(stage='neural',**row));del tx,ty,tv,net,optimizer;torch.cuda.empty_cache()
        archive=dict(mean=np.stack(means),scale=np.stack(scales),**{'p::'+k:np.stack([m[k] for m in members]) for k in members[0]})
        np.savez(run/'search'/(family+'.npz'),**archive);np.savez(run/'search'/('oof-'+family+'.npz'),scores=held_scores)
        vx=neural_inputs(va,vr,vc,vrepair,family);predictions[family]=neural_score(vx,archive,family)
        emit(dict(stage='neural_family',name=family,validation_correct=candidate_count(predictions[family],vb,vy,1.)))
    names=('tree_pair','tree_log','compact','compact_balanced','rich');trials=[];best=None
    votes=vsame.sum(-1);trap=np.array([int(vy[i].sum())==1 and max(Counter(v for v in row if v is not None).values(),default=0)>=2 and not bool(vy[i,np.argmax(votes[i])]) for i,row in enumerate(vparsed)])
    for a in range(7):
     for b in range(7-a):
      if a+b not in (2,3,4):continue
      for c in range(7-a-b):
       for d in range(7-a-b-c):
        e=6-a-b-c-d
        if c+d<1:continue
        weights={k:u/6 for k,u in zip(names,(a,b,c,d,e)) if u}
        probability=sum(w*predictions[k] for k,w in weights.items());hard_valid=vvalid&(vvalid&~vy).any(-1)[:,None]
        brier=float(np.mean((probability[hard_valid]-vy[hard_valid])**2))
        for pooling in (0.,.6,1.):
            chosen=pool_choices(probability,vb,pooling);correct=vy[np.arange(len(vy)),chosen]
            row=dict(weights=weights,pooling=pooling,correct=int(correct.sum()),trap_correct=int(correct[trap].sum()),candidate_brier=brier);trials.append(row)
            if best is None or (row['correct'],row['trap_correct'],-brier)>(best['correct'],best['trap_correct'],-best['candidate_brier']):best=row
    for name in best['weights']:shutil.copyfile(run/'search'/(name+'.npz'),run/'checkpoint'/(name+'.npz'))
    if any(n.startswith('tree') for n in best['weights']):shutil.copyfile(run/'search/stack.npz',run/'checkpoint/stack.npz')
    write(run/'checkpoint/config.json',dict(method=METHOD,protocol=PROTOCOL,weights=best['weights'],pooling=best['pooling'],selected=best))
    np.savez(run/'validation-components.npz',**predictions)
    write(run/'experiments.json',dict(plan=plan,base_oof=records,trees=tree_records,neural=neural_records,trials=trials,selected=best,
          seconds=time.perf_counter()-start,test_accessed=False,gpu=torch.cuda.get_device_name(0),peak_gpu_bytes=torch.cuda.max_memory_allocated(),versions={n:importlib.metadata.version(n) for n in ('numpy','scipy','scikit-learn','catboost','torch')}))
    assets={p.name:sha(p) for p in (run/'checkpoint').iterdir()};size=sum(p.stat().st_size for p in (run/'checkpoint').iterdir())
    if size>16*1024*1024:raise RuntimeError('Asset budget')
    write(run/'freeze.json',dict(at_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sources,asset_sha256=assets,inference_asset_bytes=size,experiments_sha256=sha(run/'experiments.json'),frozen_before_test=True))
    emit(dict(complete=True,selected=best,asset_bytes=size,seconds=time.perf_counter()-start))


if __name__=='__main__':main()
