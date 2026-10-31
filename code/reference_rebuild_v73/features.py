"""Arithmetic, textual, quantity and candidate-agreement features."""
from collections import Counter
from fractions import Fraction
import math
import re
import zlib
import numpy as np
from common import parse_answer
from .maths import mentions,equations,chain_summary,repair_summary,near,normalize

PUNCTUATION=('\n','=','+','-','*','/',r'\frac',r'\boxed','####','$','%','?','**','→',':')
CUES=tuple('therefore|however|wait|correct|mistake|assum|approximately|remaining|total|each|per|half|double|percent|let |check|actually|alternatively'.split('|'))
CONTEXT_WORDS=('more','less','total','left','profit','percent','average','how many')
TOPICS=('each','per','total','left','remaining','more','less','twice','double','percent','profit','cost','pay','spent','week','hour','minute','day','average','speed','rate','before','after','together','difference','every')
UNITS=np.array([60,1/60,100,.01,7,1/7,12,1/12],np.float64)
HASH_DIM=32768
TOKENS=re.compile(r'[a-z]+|[-+]?(?:\d+(?:\.\d+)?|\.\d+)')


def groups(bank):
    parsed=[[parse_answer(c['text']) for c in row['candidates']] for row in bank]
    same=np.array([[[float(a is not None and a==b) for b in row] for a in row] for row in parsed],np.float32).reshape(len(bank),4,4)
    valid=np.array([[a is not None for a in row] for row in parsed],bool).reshape(len(bank),4)
    return parsed,same,valid


def relative(x):return np.concatenate((x,x-x.mean(1,keepdims=True,dtype=np.float64).astype(np.float32),x-x.max(1,keepdims=True)),-1)


def real(answer):
    if answer is None:return 0.
    return float(max(Fraction(-10**18),min(Fraction(10**18),Fraction(answer))))


def literal_numbers(text):
    """Literal number tokens for format and coverage descriptors.

    These intentionally differ from canonical numeric equality used by voting.
    The scanner retains the spelling of numeric literals.
    """
    text=text.replace(',','');tokens=[];i=0
    while i<len(text):
        start=i
        if text[i] in '+-' and i+1<len(text) and text[i+1].isdigit():i+=1
        if not text[i].isdigit():i+=1;continue
        while i<len(text) and text[i].isdigit():i+=1
        if i+1<len(text) and text[i]=='.' and text[i+1].isdigit():
            i+=1
            while i<len(text) and text[i].isdigit():i+=1
        if i+1<len(text) and text[i]=='/' and text[i+1].isdigit():
            i+=1
            while i<len(text) and text[i].isdigit():i+=1
        tokens.append(text[start:i])
    return tokens


def rich_and_compact(bank):
    parsed,same,valid=groups(bank);basic=[];extra=[];compact=[];repair=[];repair_targets=[]
    for i,row in enumerate(bank):
        q=row['question'];qvalues={v for _,v in mentions(q)};qliterals=set(literal_numbers(q));qtext=q.lower()
        local=[];added=[];small=[];fixed=[];targets=[]
        for j,candidate in enumerate(row['candidates']):
            text=candidate['text'];lower=text.lower();values=[v for _,v in mentions(text)];unique=set(values)
            literals=literal_numbers(text);literal_set=set(literals)
            answer=Fraction(parsed[i][j]) if valid[i,j] else None;a=real(parsed[i][j]);votes=same[i,j].sum()
            prefix=[math.log1p(len(text)),math.log1p(len(text.split())),math.log1p(len(literals)),math.log1p(len(literal_set)),float(valid[i,j]),votes,
                    len(set(v for v in parsed[i] if v is not None)),math.log1p(abs(a)),float(a<0),float(a==0),float(a==int(a)),float(parsed[i][j] in qliterals),
                    len(literal_set&qliterals)/max(len(qliterals),1),len(literal_set-qliterals)/max(len(literal_set),1)]
            lexical=[math.log1p(text.count(token)) for token in PUNCTUATION]+[math.log1p(lower.count(word)) for word in CUES]+[math.log1p(len(q))]
            local.append(prefix+lexical)
            normalized=equations(text,latex=True)
            residual=np.array([abs(float(e.computed-e.reported))/max(1.,abs(float(e.computed)),abs(float(e.reported))) for e in normalized])
            errors=residual>.005;checked=len(residual)
            arithmetic=[math.log1p(checked),math.log1p(int(errors.sum())),float(errors.mean()) if checked else 0,
                        float(residual.max()) if checked else 0,float(residual.mean()) if checked else 0,
                        math.log1p(int((~errors).sum())),float(np.mean([int(bool(re.search(r'[+*/]|\d\s*-',e.lhs)) and bool(re.search(r'[+*/]|\d\s*-',e.rhs))) for e in normalized])) if checked else 0]
            matching=sum(abs(float(v)-a)<=1e-6+1e-5*abs(a) for v in values)
            tail=text.rsplit('####',1)[-1] if '####' in text else text[-200:]
            added.append(arithmetic+[math.log1p(matching),float(bool(values) and abs(float(values[-1])-a)<=1e-6+1e-5*abs(a)),math.log1p(len(tail)),
                         float('####' in text),float(r'\boxed' in text),math.log1p(len(re.findall(r'\\frac|/|%',q))),
                         *[float(word in qtext) for word in CONTEXT_WORDS]])
            raw=equations(text,latex=False);n=len(raw);correct=sum(e.computed==e.reported for e in raw)
            boxed=re.findall(r'\\boxed\{([^{}]+)\}',text);hash_tail=text.rsplit('####',1)[-1] if '####' in text else None
            agreement=bool(boxed and hash_tail is not None and parse_answer('#### '+boxed[-1])==parse_answer('#### '+hash_tail))
            small.append([float(valid[i,j]),math.log1p(votes),len(set(v for v in parsed[i] if v is not None)),math.log1p(max(Counter(v for v in parsed[i] if v is not None).values(),default=0)),
                          math.log1p(len(text)),sum(ch.isdigit() for ch in text)/max(len(text),1),float(bool(boxed) or hash_tail is not None),float(agreement),
                          math.log1p(sum(ch in '+-*/=÷×' for ch in text)),float(n>0),n/10,correct/max(n,1),(n-correct)/max(n,1),0.,
                          float(bool(raw) and answer is not None and raw[-1].reported==answer),len(unique&qvalues)/max(len(qvalues),1),
                          float(answer is not None and answer.denominator==1),math.log1p(abs(a)),*chain_summary(raw,answer)])
            summary,target=repair_summary(normalized,answer);fixed.append(summary);targets.append(target)
        for j,answer in enumerate(parsed[i]):
            value=Fraction(answer) if answer is not None else None
            supporters=[k for k,target in enumerate(targets) if value is not None and target is not None and near(target,value)]
            target=targets[j];hits=sum(target is not None and other is not None and near(target,Fraction(other)) for other in parsed[i])
            fixed[j].extend([len(supporters),sum(k!=j for k in supporters),hits,float(target is not None and hits>0)])
        basic.append(local);extra.append(added);compact.append(small);repair.append(fixed);repair_targets.append(targets)
    if not bank:return np.empty((0,4,207),np.float32),np.empty((0,4,22),np.float32),np.empty((0,4,39),np.float32)
    rich=np.concatenate((relative(np.asarray(basic,np.float32)),relative(np.asarray(extra,np.float32))),-1)
    return np.nan_to_num(rich).astype(np.float32),np.nan_to_num(np.asarray(compact,np.float32)),np.nan_to_num(relative(np.asarray(repair,np.float32)))


_ONES='zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen'.split()
_TENS='twenty thirty forty fifty sixty seventy eighty ninety'.split()
WORDS={word:float(i) for i,word in enumerate(_ONES)}
WORDS.update({word:float(10*(i+2)) for i,word in enumerate(_TENS)})
WORDS.update(dozen=12.,half=.5,quarter=.25,double=2.,doubles=2.,doubling=2.,twice=2.,triple=3.,triples=3.)


def quantities(text,ordinals=False,limit=12):
    if ordinals:
        lookup=dict(zip(('first','second','third','fourth'),('1','2','3','4')))
        text=re.sub(r'\b(?:first|second|third|fourth)\b',lambda m:lookup[m[0]],text.lower())
    positions=[(pos,float(value)) for pos,value in mentions(text)]
    positions.extend((m.start(),WORDS[m[0]]) for m in re.finditer(r'[a-z]+',text.lower()) if m[0] in WORDS)
    return np.array([v for _,v in sorted(positions) if np.isfinite(v) and abs(v)<1e12][:limit],np.float64)


def pair_space(q):
    n=len(q)
    if not n:return np.zeros(0),np.zeros(0,np.uint16)
    left,right=np.meshgrid(np.arange(n),np.arange(n),indexing='ij');keep=left!=right
    a,b=q[left[keep]],q[right[keep]];bits=(1<<left[keep])|(1<<right[keep])
    with np.errstate(divide='ignore',invalid='ignore',over='ignore'):values=np.r_[a+b,a-b,a*b,a/b]
    masks=np.tile(bits,4);good=np.isfinite(values)&(np.abs(values)<1e12)
    return values[good],masks[good].astype(np.uint16)


def isclose(values,target):return np.abs(values-target)<=1e-5*max(1.,abs(float(target)))


def query_sorted(sorted_values,target):
    if not len(sorted_values):return np.zeros(np.shape(target),bool)
    at=np.searchsorted(sorted_values,target);left=np.clip(at-1,0,len(sorted_values)-1);right=np.clip(at,0,len(sorted_values)-1)
    tolerance=1e-5*np.maximum(1,np.abs(target))
    return np.isfinite(target)&((np.abs(sorted_values[left]-target)<=tolerance)|(np.abs(sorted_values[right]-target)<=tolerance))


def reachability(q,pairs,masks,target):
    hit=isclose(pairs,target);one=bool(hit.any());coverage=2 if one else 0;two=False;three=False
    if len(pairs) and len(q):
        x,z=pairs[:,None],q[None,:]
        with np.errstate(divide='ignore',invalid='ignore',over='ignore'):options=(x+z,x-z,z-x,x*z,x/z,z/x)
        for option in options:
            where=np.argwhere(np.isfinite(option)&isclose(option,target))
            if len(where):
                two=True;coverage=max(coverage,max((int(masks[i])|(1<<int(j))).bit_count() for i,j in where[:512]))
    if len(pairs):
        order=np.argsort(pairs);sorted_pairs=pairs[order]
        with np.errstate(divide='ignore',invalid='ignore',over='ignore'):needs=(target-pairs,pairs-target,target+pairs,target/pairs,pairs/target,target*pairs)
        for need in needs:
            hit=query_sorted(sorted_pairs,need)
            if hit.any():
                three=True;ix=np.flatnonzero(hit)[:512];at=np.searchsorted(sorted_pairs,need[ix])
                for shift in (0,-1):
                    at2=np.clip(at+shift,0,len(pairs)-1);ok=np.abs(sorted_pairs[at2]-need[ix])<=1e-5*np.maximum(1,np.abs(need[ix]))
                    if ok.any():coverage=max(coverage,max((int(masks[i])|int(masks[order[j]])).bit_count() for i,j in zip(ix[ok],at2[ok])))
    return [float(one),float(two),float(three),coverage/max(len(q),1)]


def ordered_schemas(q):
    a,b,c,d,e=np.pad(q,(0,max(0,5-len(q))))[:5]
    with np.errstate(divide='ignore',invalid='ignore',over='ignore'):
        affine=[a+s*b*c+t*d*e for s,t in ((1,-1),(-1,1),(-1,-1),(1,1))]
        products=[a*b+c*d-e,a*b-c*d+e,a*b-c*d,a*b+c*d]
        grouped=[(a-b)*c*d,a*(b-c)*d,a*b*c*d,(a-b-c)*d,(a+b+c)*d,a*(b+c+d)]
        rates=[a*b/(c*d),a/b*c*d]
    all_forms=np.asarray(affine+products+grouped+rates,np.float64)
    return all_forms[[0,1,4,5,2,3,8,9,10,6,7,11,14,15,12,13]]


def quantitative(bank):
    parsed,_,_=groups(bank);basic=[];advanced=[]
    for i,row in enumerate(bank):
        q=quantities(row['question']);pairs,masks=pair_space(q);ql=row['question'].lower()
        oq=quantities(row['question'],True);opairs,_=pair_space(oq)
        if len(oq) and len(opairs):
            x,z=opairs[:,None],oq[None,:]
            with np.errstate(divide='ignore',invalid='ignore',over='ignore'):derived=np.concatenate([a.ravel() for a in (x+z,x-z,z-x,x*z,x/z,z/x)])
            derived=np.unique(derived[np.isfinite(derived)&(np.abs(derived)<1e12)])
        else:derived=np.zeros(0)
        forms=ordered_schemas(oq);whole=[q.sum(),q.prod() if len(q) and len(q)<=8 else 0,q.max() if len(q) else 0,q.min() if len(q) else 0]
        local=[];extra=[]
        for j,candidate in enumerate(row['candidates']):
            target=real(parsed[i][j]);text=candidate['text'];rows=equations(text,True)
            eqtext=' '.join(line for line in normalize(text).splitlines() if '=' in line)
            eqvals=np.array([float(v) for _,v in mentions(eqtext)],np.float64);allvalues=quantities(text,limit=None)
            cover=sum(isclose(eqvals,x).any() for x in q)/max(len(q),1);mentioned=sum(isclose(allvalues,x).any() for x in q)/max(len(q),1)
            distance=[math.log1p(min(abs(target-v)/max(1,abs(v)),1e6)) for v in whole]
            exact=[float(abs(target-v)<=1e-5*max(1,abs(v))) for v in whole]
            series=False
            if len(q)>=3:
                start,step,count=q[:,None,None],q[None,:,None],q[None,None,:]
                series=bool(isclose(count*start+step*count*(count-1)/2,target).any())
            local.append([math.log1p(len(q)),cover,mentioned,*distance,*exact,*reachability(q,pairs,masks,target),float(series),*[float(word in ql) for word in TOPICS]])
            hits=query_sorted(derived,target/UNITS).astype(float);form_hits=(np.isfinite(forms)&isclose(forms,target)).astype(float)
            with np.errstate(divide='ignore',invalid='ignore'):error=np.abs(forms-target)/np.maximum(1,np.abs(forms))
            error=np.nan_to_num(error,nan=1e6,posinf=1e6,neginf=1e6)
            extra.append([*hits,*form_hits,*np.log1p(np.minimum(error,1e6)),float(len(oq)==4),float(len(oq)==5)])
        basic.append(local);advanced.append(extra)
    return relative(np.asarray(basic,np.float32)),relative(np.asarray(advanced,np.float32))


def token_counts(text):
    words=TOKENS.findall(text.lower());counts=Counter(words);counts.update(a+' '+b for a,b in zip(words,words[1:]))
    hashed=Counter()
    for token,count in counts.items():hashed[zlib.crc32(token.encode('utf-8'))%HASH_DIM]+=count
    return hashed


def text_scores(bank,idf,coef):
    values=[]
    for row in bank:
        for candidate in row['candidates']:
            counts=token_counts(candidate['text'])
            if not counts:values.append(.5);continue
            indices=np.fromiter(counts.keys(),np.int64);amount=np.fromiter(counts.values(),np.float64)
            vector=(1+np.log(amount))*idf[indices];norm=np.linalg.norm(vector)
            logit=float(vector@coef[indices]/max(norm,1e-12));values.append(1/(1+math.exp(-max(-30,min(30,logit)))))
    return np.asarray(values).reshape(len(bank),4)
