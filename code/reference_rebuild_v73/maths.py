"""Bounded exact arithmetic and numeric-expression analysis."""
import ast
from dataclasses import dataclass
from fractions import Fraction
import math
import re
from functools import lru_cache

NUMBER=re.compile(r'(?<![\w.])[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:/\d+)?')
ARITHMETIC=re.compile(r'[\d.\s()+*/\-]+')


@lru_cache(maxsize=65536)
def mentions(text):
    clean=re.sub(r'(?<=\d),(?=\d{3}(?:\D|$))','',str(text))
    result=[]
    for token in NUMBER.finditer(clean):
        try:
            value=Fraction(token.group())
            if abs(value)<10**15:result.append((token.start(),value))
        except (ValueError,ZeroDivisionError):pass
    return result


def near(a,b):return abs(float(a-b))<=1e-6+.003*max(abs(float(a)),abs(float(b)))


def evaluate(text,repairs=()):
    """Return original/repaired values and literal operands, never execute code."""
    text=text.strip()
    if len(text)>200:raise ValueError('Expression budget')
    try:tree=ast.parse(text,mode='eval')
    except (SyntaxError,ValueError):raise ValueError('Arithmetic syntax') from None
    if sum(1 for _ in ast.walk(tree))>64:raise ValueError('Node budget')
    literals=[];operations=0
    def walk(node):
        nonlocal operations
        if isinstance(node,ast.Constant) and type(node.value) in (int,float):
            original=Fraction(ast.get_source_segment(text,node));patched=original;literals.append(original)
        elif isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.UAdd,ast.USub)):
            original,patched=walk(node.operand)
            if isinstance(node.op,ast.USub):original,patched=-original,-patched
        elif isinstance(node,ast.BinOp) and isinstance(node.op,(ast.Add,ast.Sub,ast.Mult,ast.Div,ast.Pow)):
            operations+=1;a,pa=walk(node.left);b,pb=walk(node.right)
            if isinstance(node.op,ast.Add):original,patched=a+b,pa+pb
            elif isinstance(node.op,ast.Sub):original,patched=a-b,pa-pb
            elif isinstance(node.op,ast.Mult):original,patched=a*b,pa*pb
            elif isinstance(node.op,ast.Div):original,patched=a/b,pa/pb
            else:
                if b.denominator!=1 or pb.denominator!=1 or abs(b)>8 or abs(pb)>8:raise ValueError('Exponent budget')
                original,patched=a**int(b),pa**int(pb)
        else:raise ValueError('Numeric nodes only')
        for value in (original,patched):
            if abs(value)>10**18 or value.numerator.bit_length()>256 or value.denominator.bit_length()>256:raise ValueError('Magnitude budget')
        for wrong,right in reversed(repairs):
            if near(original,wrong):patched=right;break
        return original,patched
    try:original,patched=walk(tree.body)
    except (ArithmeticError,OverflowError,RecursionError):raise ValueError('Undefined arithmetic') from None
    return original,patched,literals,operations


def normalize(text):
    text=str(text).replace('−','-').replace('×','*').replace('÷','/')
    text=re.sub(r'(?<=\d),(?=\d{3}(?:\D|$))','',text)
    text=re.sub(r'\\(?:text|mathrm|operatorname)\{[^{}]*\}',' ',text)
    text=re.sub(r'\\(?:left|right|displaystyle)|\\[,;!]','',text)
    for _ in range(8):
        changed=re.sub(r'\\(?:frac|dfrac|tfrac)\s*\{([^{}]*)\}\s*\{([^{}]*)\}',lambda m:'(('+m[1]+')/('+m[2]+'))',text)
        changed=re.sub(r'\\(?:boxed|mathbf|mathit)\s*\{([^{}]*)\}',lambda m:'('+m[1]+')',changed)
        if changed==text:break
        text=changed
    text=text.replace(r'\times','*').replace(r'\cdot','*').replace(r'\div','/')
    text=re.sub(r'(?<=\d)\s*\\?%','/100',text)
    text=re.sub(r'\*\*(?=[A-Za-z])|(?<=[A-Za-z])\*\*','',text)
    return text.replace('^','**').replace('{','(').replace('}',')')


def boundary(fragment,left):
    spans=[m.group() for m in ARITHMETIC.finditer(fragment) if any(c.isdigit() for c in m.group())]
    if not spans:return None
    token=spans[-1 if left else 0].strip().strip('*').strip()
    proposals=[token]
    if left:
        proposals.extend(token[i:] for i,c in enumerate(token) if c.isdigit())
    else:
        for n in range(1,min(10,len(token))):
            if not token[-n:].strip(' .)*'):proposals.append(token[:-n])
    for candidate in proposals[:12]:
        try:
            original,_,leaves,count=evaluate(candidate)
            return candidate,original,leaves,count
        except ValueError:pass
    return None


@dataclass
class Equation:
    lhs:str
    rhs:str
    computed:Fraction
    reported:Fraction
    operands:list
    operations:int


@lru_cache(maxsize=32768)
def equations(text,latex=True):
    cleaned=normalize(text) if latex else re.sub(r'(?<=\d),(?=\d{3}(?:\D|$))','',str(text)).replace('×','*').replace('÷','/').replace('−','-')
    pieces=[(i,line) for i,line in enumerate(cleaned.splitlines())]
    if latex:
        for block in re.finditer(r'\$\$(.*?)\$\$|\\\[(.*?)\\\]',cleaned,re.S):
            value=block[1] if block[1] is not None else block[2]
            pieces.append((cleaned[:block.start()].count('\n'),value.replace('\n',' ')))
    found=[];seen=set()
    for _,piece in sorted(pieces,key=lambda item:item[0]):
        parts=piece.split('=')
        for i in range(len(parts)-1):
            if not latex and '%' in parts[i][-100:]+parts[i+1][:100]:continue
            a=boundary(parts[i][-220:],True);b=boundary(parts[i+1][:220],False)
            if a is None or b is None or not(a[3]+b[3]):continue
            key=(re.sub(r'\s','',a[0]),re.sub(r'\s','',b[0]))
            if key in seen:continue
            seen.add(key);found.append(Equation(a[0],b[0],a[1],b[1],a[2],a[3]+b[3]))
            if len(found)==100:return found
    return found


def chain_summary(rows,answer):
    if not rows:return [1.,0.,.5,0.]
    correct=[row.computed==row.reported for row in rows];parents=[];used=set()
    for k,row in enumerate(rows):
        dependencies={i for i in range(k) if rows[i].reported in row.operands};parents.append(dependencies);used.update(dependencies)
    first=next((i/len(rows) for i,value in enumerate(correct) if not value),1.)
    clean=.5;roots=[i for i,row in enumerate(rows) if answer is not None and row.reported==answer]
    if roots:
        stack=[roots[-1]];visited=set()
        while stack:
            at=stack.pop()
            if at not in visited:visited.add(at);stack.extend(parents[at])
        clean=float(all(correct[i] for i in visited))
    orphan=sum(ok and i not in used for i,ok in enumerate(correct))/max(sum(correct),1)
    return [first,len(used)/max(len(rows)-1,1),clean,orphan]


def repair_summary(rows,answer):
    history=[];propagated=0;errors=[]
    for row in rows:
        errors.append(abs(float(row.computed-row.reported))/max(1e-6,abs(float(row.computed)),abs(float(row.reported))))
        try:_,patched,_,_=evaluate(row.lhs,history)
        except ValueError:continue
        if not near(patched,row.reported) and len(history)<12:
            propagated+=near(row.computed,row.reported);history.append((row.reported,patched))
    target=None
    if answer is not None:
        for wrong,right in reversed(history):
            if near(answer,wrong) and not near(answer,right):target=right;break
    bad=sum(e>.005 for e in errors)
    stats=[math.log1p(len(rows)),math.log1p(bad),bad/max(len(rows),1),max(errors,default=0),sum(errors)/max(len(errors),1),
           math.log1p(len(history)),math.log1p(propagated),float(target is not None),math.log1p(abs(float(target-answer))) if target is not None else 0.]
    return stats,target
