"""Synthetic-only integration and selection tests, no price or label files."""
import copy,importlib.util,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

P=Path(__file__).resolve().parent

def mod(name,path):
    sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
s=mod('tested_honest117',P/'study.py')
fixture_source=mod('honest117_fixture',P.parent/'monthly_garch70/test_study.py')

def fixture():
    a,m=fixture_source.fixture()
    r,_=s.local.ranked_forecasts({x:a['raw_'+x] for x in s.local.ARMS},a['model_available'],a,4)
    a['rank_local_linear']=r['local_linear']
    a['raw_matched_in_sample']=a['raw_local_linear'].copy()
    a['raw_honest_local']=a['raw_local_linear'][:,::-1].copy()
    a['honest_model_available']=a['model_available'].copy()
    return a,m

def row(name,cagr=.5,sharpe=1.5,dd=-.3,early=.5,recent=.5):
    return dict(name=name,training=dict(cagr=cagr,sharpe=sharpe,max_drawdown=dd),early=dict(cagr=early),recent=dict(cagr=recent))

class StudyTests(unittest.TestCase):
    def test_fixed_three_arms(self):
        self.assertEqual(s.NAMES,('incumbent_local67','matched_in_sample','honest_local'))
    def test_control_exact(self):
        a,m=fixture();p=dict(s.local.POLICY_FIELDS)
        got=s.choices(s.NAMES[0],a,m,4,p);want=s.local.choices(s.local.configs()[4],a,m,4,policy=p)
        self.assertEqual(got[0],want[0]);np.testing.assert_array_equal(got[1],want[1])
    def test_equal_raw_is_exact_control(self):
        a,m=fixture();p=dict(s.local.POLICY_FIELDS)
        got=s.choices(s.NAMES[1],a,m,4,p);want=s.choices(s.NAMES[0],a,m,4,p)
        self.assertEqual(got[0],want[0]);self.assertEqual(got[1].tobytes(),want[1].tobytes())
    def test_delta_budget_and_single_episode(self):
        a,m=fixture();p=dict(s.local.POLICY_FIELDS);pre=s.local.modified_pre_score(s.local.configs()[4],a,4)
        ranks,ready=s.ranked_pair(a,4)
        coeff=.25*(.05*a['state_weak_narrow'][:,None]+.01*(a['state_calm10']&~a['state_weak_narrow'])[:,None]*a['extension_available'])
        want=coeff*(ranks['honest_local']-a['rank_local_linear'])
        np.testing.assert_array_equal(want,s.score_delta('honest_local',a,4))
        with patch.object(s.local.ep,'select_boundary',wraps=s.local.ep.select_boundary) as spy:s.choices('honest_local',a,m,4,p)
        self.assertEqual(spy.call_count,1);np.testing.assert_array_equal(spy.call_args.args[0],pre+want)
        for k in ('core','baseline'):np.testing.assert_array_equal(spy.call_args.args[4][k],a[k])
    def test_both_arms_whole_month_fallback(self):
        a,m=fixture();p=dict(s.local.POLICY_FIELDS)
        a['raw_matched_in_sample'][2,1]=np.nan
        ranks,ready=s.ranked_pair(a,4);self.assertFalse(ready[2])
        for x in s.ARMS:np.testing.assert_array_equal(ranks[x][2],a['rank_local_linear'][2])
        a['honest_model_available'][:]=False
        want=s.choices(s.NAMES[0],a,m,4,p)
        for x in s.ARMS:
            got=s.choices(x,a,m,4,p);self.assertEqual(got[0],want[0]);self.assertEqual(got[1].tobytes(),want[1].tobytes())
    def test_ranking_uses_original_U_not_ownhistory(self):
        a,m=fixture();a['honest_local_available']=np.zeros((4,6),bool)
        a['eligible'][:,0]=False;a['raw_honest_local'][:,0]=999.
        ranks,ready=s.ranked_pair(a,4)
        self.assertTrue(ready.all());self.assertTrue(np.all(ranks['honest_local'][:,0]==a['rank_local_linear'][:,0]))
        np.testing.assert_array_equal(ranks['honest_local'][:,1:],np.tile(np.arange(5,0,-1,dtype=float)/5.-.5,(4,1)))
    def test_prefix_and_no_mutation(self):
        a,m=fixture();before=copy.deepcopy(a);p=dict(s.local.POLICY_FIELDS)
        for name in s.NAMES:
            ch,sc=s.choices(name,a,m,4,p)
            for n in (0,1,2,4):
                pc,ps=s.choices(name,a,m,n,p);self.assertEqual(pc,ch[:n]);self.assertEqual(ps.tobytes(),sc[:n].tobytes())
            poison=copy.deepcopy(a)
            for v in poison.values():
                if v.dtype.kind=='f':v[2:]=np.nan
            pc,ps=s.choices(name,poison,m,2,p);self.assertEqual(pc,ch[:2]);self.assertEqual(ps.tobytes(),sc[:2].tobytes())
        for k in a:self.assertEqual(a[k].tobytes(),before[k].tobytes())
    def test_less_than_three_eligible(self):
        a,m=fixture();p=dict(s.local.POLICY_FIELDS)
        for count in (0,1,2):
            a['eligible'][:]=False;a['eligible'][:,:count]=True
            for name in s.NAMES:self.assertEqual([len(x) for x in s.choices(name,a,m,4,p)[0]],[2 if count==2 else 0]*4)
    def test_selection_compare_both_not_diagnostic_promotion(self):
        rows=[row(x) for x in s.NAMES];self.assertEqual(s.select(rows)['primary'],s.NAMES[0])
        rows[1]=row(s.NAMES[1],.9);self.assertEqual(s.select(rows)['primary'],s.NAMES[0])
        rows[2]=row(s.NAMES[2],1.);self.assertEqual(s.select(rows)['primary'],s.NAMES[2])
        rows[2]=row(s.NAMES[2],.7);self.assertEqual(s.select(rows)['primary'],s.NAMES[0])
    def test_all_gates(self):
        for k,v in [('cagr',.49),('sharpe',1.44),('dd',-.331),('early',.474),('recent',.474)]:
            self.assertFalse(s.admitted(row('x',**{k:v}),row('y')))
        with self.assertRaises(ValueError):s.admitted(row('x',cagr=float('nan')),row('y'))
    def test_infinite_forecast_and_unregistered(self):
        a,m=fixture();a['raw_honest_local'][0,0]=np.inf
        with self.assertRaises(FloatingPointError):s.ranked_pair(a,4)
        with self.assertRaises(ValueError):s.choices('unknown',{}, {},0,{})

if __name__=='__main__':unittest.main()
