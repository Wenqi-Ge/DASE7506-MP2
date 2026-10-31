"""Feature contracts and faithful objective/export checks for the rebuilt recipe."""
from fractions import Fraction
from pathlib import Path
import ast
import unittest
from unittest.mock import patch
import numpy as np
import torch
from scipy import sparse
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from common import load_inputs,parse_answer
from reference_rebuild_v73.maths import evaluate,equations,chain_summary,repair_summary
from reference_rebuild_v73.features import rich_and_compact,quantitative,text_scores,groups
from reference_rebuild_v73.stack import fit_margin,view_matrix,training_pairs,text_matrix,fit_text,meta_matrix,text_columns
from reference_rebuild_v73.runtime import NETWORKS,neural_inputs,neural_logits,pool_choices,predict
from reference_rebuild_v73.net import SetVerifier


class ReferenceRebuildTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        bank,arrays=load_inputs('train');cls.bank=bank[:4];cls.arrays={'embeddings':arrays['embeddings'][:4]}

    def test_feature_dimensions(self):
        rich,compact,repair=rich_and_compact(self.bank);quantity,schemas=quantitative(self.bank)
        self.assertEqual(rich.shape,(4,4,207));self.assertEqual(compact.shape,(4,4,22));self.assertEqual(repair.shape,(4,4,39))
        rng=np.random.default_rng(72);projection={'mean':np.zeros(1024,np.float32),'components':rng.normal(size=(64,1024)).astype(np.float32)}
        basic=meta_matrix(self.bank,self.arrays,rich,np.full((4,4,5),.5),projection,quantity)
        extended=np.concatenate((basic,schemas.reshape(16,-1),text_columns(np.full((4,4),.5)).reshape(16,-1),repair.reshape(16,-1)),1)
        self.assertEqual(basic.shape,(16,619));self.assertEqual(extended.shape,(16,788));self.assertTrue(np.isfinite(extended).all())

    def test_exact_arithmetic_and_repair_propagation(self):
        self.assertEqual(evaluate('(3+1)/8')[0],Fraction(1,2))
        rows=equations('12/3=8;8*5=40',True);stats,target=repair_summary(rows,Fraction(40))
        self.assertEqual(target,Fraction(20));self.assertEqual(stats[7],1)
        self.assertEqual(chain_summary(rows,Fraction(40))[2],0)

    def test_thousands_and_latex(self):
        raw=equations('1,200 / 3 = 400',False);self.assertEqual(len(raw),1);self.assertEqual(raw[0].computed,Fraction(400))
        latex=equations(r'\frac{12}{3}=4',True);self.assertEqual(latex[0].computed,Fraction(4))

    def test_unsafe_expressions_never_execute(self):
        for text in ('__import__("os").system("true")','x+1','1/0','2**100','(1).__class__'):
            with self.assertRaises(ValueError):evaluate(text)

    def test_handwritten_regularization_matches_sklearn_C(self):
        rng=np.random.default_rng(17)
        bank=[dict(id=str(i),question='1 or2',candidates=[dict(text='#### '+str(a)) for a in (1,2,1,3)]) for i in range(6)]
        arrays={'embeddings':rng.normal(size=(6,4,1024)).astype(np.float32)};rich=np.zeros((6,4,207),np.float32);y=np.array([[1,0,1,0]]*6,bool)
        c=.002;model=fit_margin(bank,arrays,y,rich,'raw',c)
        x=view_matrix(bank,arrays,rich,'raw');z=np.clip((x-model['mean'])/model['scale'],-8,8)
        yes,no,w=training_pairs(bank,y);d=(z[yes]-z[no]).astype(np.float64);w/=w.mean()
        reference=LogisticRegression(C=c,fit_intercept=False,solver='lbfgs',tol=1e-9,max_iter=1000)
        reference.fit(np.concatenate((d,-d)),np.r_[np.ones(len(d)),np.zeros(len(d))],sample_weight=np.r_[w,w])
        np.testing.assert_allclose(model['weights'],reference.coef_[0],rtol=.003,atol=3e-6)

    def test_lexical_export_matches_sparse_calculation(self):
        bank=[dict(id=str(i),question='q',candidates=[dict(text=t) for t in ('correct solution #### 1','incorrect solution #### 2','correct again #### 1','wrong answer #### 3')]) for i in range(5)]
        y=np.array([[1,0,1,0]]*5,bool);counts=text_matrix(bank);idf,coef=fit_text(counts,bank,y,np.arange(5))
        x=counts.copy();x.data=1+np.log(x.data);x=x.multiply(idf).tocsr();norm=np.sqrt(np.asarray(x.multiply(x).sum(1)).ravel()).clip(1e-12)
        expected=expit(np.asarray(x@coef).ravel()/norm).reshape(5,4)
        np.testing.assert_allclose(text_scores(bank,idf,coef),expected,rtol=1e-7,atol=1e-7)

    def test_all_neural_exports_and_permutation(self):
        rng=np.random.default_rng(73);rich,compact,repair=rich_and_compact(self.bank)
        for family,config in NETWORKS.items():
            x=neural_inputs(self.arrays,rich,compact,repair,family);x=(x-x.mean(0))/np.maximum(x.std(0),1)
            torch.manual_seed(77);net=SetVerifier(x.shape[-1],config).eval();state={k:v.detach().numpy() for k,v in net.state_dict().items()}
            with torch.no_grad():native=net(torch.as_tensor(x)).numpy()
            deployed=neural_logits(x,state,config);np.testing.assert_allclose(deployed,native,rtol=2e-5,atol=2e-6)
            order=[2,0,3,1];np.testing.assert_allclose(neural_logits(x[:,order],state,config),deployed[:,order],rtol=2e-5,atol=2e-6)

    def test_pooling_numeric_equivalence_and_reference_isolation(self):
        bank=[dict(question='q',candidates=[dict(text=t) for t in ('#### 0.5','#### 7','#### 1/2','invalid')])]
        with patch('common.read_jsonl',side_effect=AssertionError('reference read')):
            rich_and_compact(bank)
            for amount in (0.,.6,1.):
                selected=pool_choices(np.array([[.7,.6,.8,.99]]),bank,amount)[0]
                self.assertEqual(parse_answer(bank[0]['candidates'][selected]['text']),'1/2')
        self.assertEqual(predict([],{},None),[])

    def test_only_current_package_modules_imported(self):
        code=Path(__file__).parent
        for path in list((code/'reference_rebuild_v73').glob('*.py'))+[code/'submission_v73.py']:
            for node in ast.walk(ast.parse(path.read_text())):
                names=[x.name for x in node.names] if isinstance(node,ast.Import) else ([node.module] if isinstance(node,ast.ImportFrom) and node.module else [])
                if isinstance(node,ast.ImportFrom) and node.level:names=['reference_rebuild_v73.'+name for name in names]
                for name in names:
                    root=name.split('.')[0]
                    if (code/root).with_suffix('.py').exists() or (code/root).is_dir():self.assertIn(root,('common','reference_rebuild_v73'),str(path))


if __name__=='__main__':unittest.main()
