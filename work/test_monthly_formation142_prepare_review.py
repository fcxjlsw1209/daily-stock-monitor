"""Synthetic-only independent142 audit tests. No actual market/order cache reads."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import monthly_formation142_prepare_review as r
from test_monthly_price_delay140_prepare_review import score_fixture


def synthetic_prices(width=5):
    time=np.arange(430)[:,None];columns=np.arange(width)[None,:]
    returns=.0007+.00008*columns+.006*np.sin(time*(.14+.023*columns))+.001*np.cos(time*.07+columns)
    return 100*np.cumprod(1.+returns,axis=0)


def feature_fixture(high=False):
    prices=synthetic_prices(49);days=pd.bdate_range('2015-01-02',periods=len(prices))
    names=['S%02d'%i for i in range(32)]+['E%02d'%i for i in range(17)]
    frame=pd.DataFrame(prices,index=days,columns=names)
    t=np.arange(len(days));frame['SPY']=100*np.cumprod(1.+(.020 if high else .002)*np.sin(t*.31)+.0002)
    meta=dict(dates=[str(days[i].date())for i in (350,410)],symbols=[names[:],names[:]],etfs=names[-17:])
    calm=np.full(2,not high,bool);weights=np.where(calm,.5,.52);cores=[]
    for day in meta['dates']:
        p=frame.loc[:day,names];raw=p.iloc[-43]/p.iloc[-253 if not high else -295]-1.
        if high:raw=raw/p.pct_change(fill_method=None).rolling(63).std().iloc[-1].replace(0,np.nan)
        cores.append((raw.rank(pct=True).fillna(.5).to_numpy()-.5)*(.52 if high else .5))
    old=dict(core=np.array(cores),core_weight=weights,state_calm10=calm,eligible=np.ones((2,49),bool))
    return dict(adj=frame),meta,old,days


def synthetic_features(a):
    n,width=a['core'].shape
    out=dict(formation_rank=np.full((n,12,width),np.nan),formation_count=np.zeros(n,np.int64),
        formation_ready=np.ones(n,bool),formation_valid=np.ones((n,width),bool))
    for i in range(n):
        count=10 if a['state_calm10'][i]else 12;out['formation_count'][i]=count
        for b in range(count):out['formation_rank'][i,b]=r.rank_oracle(np.sin(np.arange(width)+b*.23+i*.1))
    return out


class FeatureTests(unittest.TestCase):
    def test_rank_pct_and_missing(self):np.testing.assert_array_equal(r.rank_oracle([1.,1.,3.,np.nan]),[0.,0.,.5,0.])
    def test_rank_allmissing_neutral(self):np.testing.assert_array_equal(r.rank_oracle([np.nan,np.nan]),[0.,0.])
    def test_rank_infinity_rejected(self):
        with self.assertRaises(ValueError):r.rank_oracle([1.,np.inf])
    def test_calm_ten_common_blocks(self):
        p=synthetic_prices();x=r.scenario_oracle(p,350,calm=True)
        self.assertTrue(x['ready']);self.assertEqual(x['scenario_core_ranks'].shape,(10,5));self.assertEqual(x['blocks'][0],dict(start_price_index=98,end_price_index=119,deleted_returns=21));self.assertEqual(x['blocks'][-1]['end_price_index'],308)
    def test_other_twelve_common_blocks(self):
        x=r.scenario_oracle(synthetic_prices(),350,calm=False)
        self.assertEqual(len(x['blocks']),12);self.assertEqual(x['blocks'][0]['start_price_index'],56)
    def test_removed_gross_product_manual(self):
        p=synthetic_prices();x=r.scenario_oracle(p,350,calm=True);raw=(p[308]/p[98])/(p[119]/p[98])-1
        np.testing.assert_array_equal(x['scenario_core_ranks'][0],r.rank_oracle(raw))
    def test_sample_std_manual(self):
        p=100*np.cumprod(1.+np.linspace(-.02,.02,64));self.assertAlmostEqual(r.sigma_oracle(p),np.std(p[1:]/p[:-1]-1,ddof=1),14)
    def test_formation_skips_latest42_calm(self):
        p=synthetic_prices();x=r.scenario_oracle(p,350,calm=True);p[309:351]*=2
        y=r.scenario_oracle(p,350,calm=True);np.testing.assert_array_equal(x['scenario_core_ranks'],y['scenario_core_ranks'])
    def test_future_strings_not_interpreted(self):
        p=synthetic_prices().astype(object);x=r.scenario_oracle(p,350,calm=False);p[351:]='future'
        y=r.scenario_oracle(p,350,calm=False);np.testing.assert_array_equal(x['scenario_core_ranks'],y['scenario_core_ranks'])
    def test_original_missing_mask_retained(self):
        p=synthetic_prices();p[98,2]=np.nan;x=r.scenario_oracle(p,350,calm=True)
        self.assertFalse(x['original_finite_mask'][2]);self.assertTrue(np.all(x['scenario_core_ranks'][:,2]==0.))
    def test_internal_missing_wholemonth_fallback(self):
        p=synthetic_prices();p[160,2]=np.nan;x=r.scenario_oracle(p,350,calm=True)
        self.assertFalse(x['ready']);self.assertTrue(x['original_finite_mask'][2]);self.assertEqual(x['blocks'],[])
    def test_insufficient_history(self):
        x=r.scenario_oracle(synthetic_prices(),200,calm=True);self.assertFalse(x['ready']);self.assertIsNone(x['original_finite_mask'])
    def test_unit_scaling(self):
        p=synthetic_prices();x=r.scenario_oracle(p,350,calm=False);y=r.scenario_oracle(p*np.array([1.,10.,.01,100.,.5]),350,calm=False)
        np.testing.assert_array_equal(x['scenario_core_ranks'],y['scenario_core_ranks'])
    def test_formation_gross_overflow(self):
        p=synthetic_prices();p[98,0]=1e-300;p[308,0]=1e300
        with self.assertRaises(FloatingPointError):r.scenario_oracle(p,350,calm=True)
    def test_formation_gross_underflow(self):
        p=synthetic_prices();p[98,0]=1e300;p[308,0]=1e-300
        with self.assertRaises(FloatingPointError):r.scenario_oracle(p,350,calm=True)
    def test_block_gross_underflow(self):
        p=synthetic_prices();p[119,0]=1e300;p[140,0]=1e-300
        with self.assertRaises(FloatingPointError):r.scenario_oracle(p,350,calm=True)
    def test_block_gross_overflow(self):
        p=synthetic_prices();p[119,0]=1e-300;p[140,0]=1e300
        with self.assertRaises(FloatingPointError):r.scenario_oracle(p,350,calm=True)
    def test_feature_contract_calm_and_noncalm(self):
        for high in (False,True):
            p,m,a,s=feature_fixture(high);x=r.features_oracle(p,m,a,s)
            self.assertEqual(x['arrays']['formation_rank'].shape,(2,12,49));self.assertEqual(x['arrays']['formation_count'].tolist(),[12 if high else 10]*2);self.assertEqual(x['arrays']['formation_ready'].tolist(),[True,True])
    def test_bad_original_core_stops(self):
        p,m,a,s=feature_fixture();a['core'][0,0]+=.01
        with self.assertRaises(AssertionError):r.features_oracle(p,m,a,s)
    def test_bad_original_state_stops(self):
        p,m,a,s=feature_fixture();a['state_calm10'][0]=False
        with self.assertRaisesRegex(ValueError,'state parity'):r.features_oracle(p,m,a,s)
    def test_bad_original_weight_stops(self):
        p,m,a,s=feature_fixture();a['core_weight'][0]=.52
        with self.assertRaisesRegex(ValueError,'weight parity'):r.features_oracle(p,m,a,s)
    def test_missing_official_session_stops(self):
        p,m,a,s=feature_fixture();p['adj']=p['adj'].drop(s[100])
        with self.assertRaisesRegex(ValueError,'official formation'):r.features_oracle(p,m,a,s)
    def test_feature_prefix_and_future_poison(self):
        p,m,a,s=feature_fixture();full=r.features_oracle(p,m,a,s)
        for n in (0,1,2):
            sub=dict(m,dates=m['dates'][:n],symbols=m['symbols'][:n]);cut=p['adj'].loc[:m['dates'][n-1]]if n else p['adj'].iloc[:0]
            x=r.features_oracle(dict(adj=cut),sub,a,s)
            for k in r.EXTRA:np.testing.assert_array_equal(x['arrays'][k],full['arrays'][k][:n])
        p['adj']=p['adj'].astype(object);p['adj'].loc[s[351]:]='future'
        sub=dict(m,dates=m['dates'][:1],symbols=m['symbols'][:1]);x=r.features_oracle(p,sub,a,s)
        for k in r.EXTRA:np.testing.assert_array_equal(x['arrays'][k],full['arrays'][k][:1])


class SelectorTests(unittest.TestCase):
    def choose(self,scenarios,**kwargs):
        args=dict(pre=np.array([.3,.2,.1,.09]),chosen=[0,1,2],core=np.zeros(4),weight=.5,scenarios=None if scenarios is None else np.asarray(scenarios),eligible=np.ones(4,bool),keys=list('ABCD'),arm='worst_scenario');args.update(kwargs);return r.selector_oracle(**args)
    def test_joint_min_not_individual_min(self):
        x=self.choose([[.4,-.4,-.4,.1],[-.4,.4,-.4,.1]])
        self.assertEqual(x['selected_keys'],['A','B','D'])
    def test_mean_and_worst_can_differ(self):
        data=[[0.,0.,.5,0.],[0.,0.,-.3,0.]]
        self.assertEqual(self.choose(data,arm='mean_scenario')['selected_keys'],['A','B','C'])
        self.assertEqual(self.choose(data)['selected_keys'],['A','B','D'])
    def test_original_tie_preferred(self):
        x=self.choose(np.full((2,4),.1),pre=np.zeros(4),chosen=[3,1,2]);self.assertEqual(x['selected_keys'],['B','C','D'])
    def test_missing_preserves_order(self):
        x=self.choose(None,chosen=[2,1,0]);self.assertEqual(x['selected_indices'],[2,1,0]);self.assertTrue(x['fallback'])
    def test_unchanged_core_preserves_order(self):
        x=self.choose(np.zeros((2,4)),chosen=[2,1,0]);self.assertEqual(x['reason'],'unchanged_weighted_core');self.assertEqual(x['selected_indices'],[2,1,0])
    def test_candidate_bad_history_fallback(self):
        x=self.choose(np.ones((2,4))*.1,history_available=np.array([True,True,True,False]));self.assertEqual(x['reason'],'candidate_history_unavailable')
    def test_eligibility_respected(self):
        x=self.choose(np.ones((2,4))*.4,eligible=np.array([True,True,True,False]));self.assertEqual(x['selected_indices'],[0,1,2])
    def test_fewer_than_three_fallback(self):self.assertEqual(self.choose(np.ones((2,4))*.1,chosen=[1,0])['selected_indices'],[1,0])
    def test_common_scenarios_missing_nan_fallback(self):self.assertEqual(self.choose(np.array([[np.nan,0.,0.,0.]]))['reason'],'nan_scenario_values')
    def test_infinite_scenario_rejected(self):
        with self.assertRaises(ValueError):self.choose([[np.inf,0.,0.,0.]])
    def test_out_of_range_rank_rejected(self):
        with self.assertRaises(ValueError):self.choose([[.51,0.,0.,0.]])
    def test_original_selected_outside_top8_included(self):
        x=r.selector_oracle(np.linspace(.5,0.,10),[0,1,9],np.zeros(10),.5,np.tile(np.linspace(-.5,.5,10),(2,1)),np.ones(10,bool),[str(j)for j in range(10)],arm='mean_scenario')
        self.assertIn(9,x['candidate_indices']);self.assertEqual(len(x['candidate_indices']),9)


class ScoreTests(unittest.TestCase):
    def test_baseline_episode_once_newarms_bypass(self):
        a,m=score_fixture();f=synthetic_features(a);h=r.old.original_helper()
        with patch.object(r.old,'original_helper',return_value=h),patch.object(h,'episode',wraps=h.episode)as spy:
            s,c,d=r.score_oracle(a,m,f);self.assertEqual(spy.call_count,1)
        for name in r.NAMES[1:]:
            for i,item in enumerate(d[name]):
                if not item['fallback']:np.testing.assert_array_equal(s[name][i],h.original_prefix(a,m)['local'][i])
    def test_missing_exact_finalscore_fallback(self):
        a,m=score_fixture();f=synthetic_features(a);f['formation_rank'][:]=np.nan;f['formation_ready'][:]=False;f['formation_count'][:]=0;s,c,_=r.score_oracle(a,m,f)
        for name in r.NAMES[1:]:self.assertEqual(s[name].tobytes(),s[r.NAMES[0]].tobytes());self.assertEqual(c[name],c[r.NAMES[0]])
    def test_noop_exact_finalscore_fallback(self):
        a,m=score_fixture();f=synthetic_features(a)
        for i in range(2):f['formation_rank'][i,:f['formation_count'][i]]=a['core'][i]/a['core_weight'][i]
        s,c,_=r.score_oracle(a,m,f)
        for name in r.NAMES[1:]:self.assertEqual(s[name].tobytes(),s[r.NAMES[0]].tobytes());self.assertEqual(c[name],c[r.NAMES[0]])
    def test_bad_count_rejected(self):
        a,m=score_fixture();f=synthetic_features(a);f['formation_count'][0]=11
        with self.assertRaises(ValueError):r.score_oracle(a,m,f)
    def test_finite_padding_rejected(self):
        a,m=score_fixture();f=synthetic_features(a);f['formation_rank'][0,11]=0.
        with self.assertRaises(ValueError):r.score_oracle(a,m,f)
    def test_dtype_mismatch_rejected(self):
        a,m=score_fixture();f=synthetic_features(a);f['formation_count']=f['formation_count'].astype(float)
        with self.assertRaises(ValueError):r.score_oracle(a,m,f)
    def test_original_inputs_unchanged(self):
        a,m=score_fixture();f=synthetic_features(a);before={k:v.tobytes()for k,v in a.items()};r.score_oracle(a,m,f);self.assertEqual(before,{k:v.tobytes()for k,v in a.items()})
    def test_all123_score_prefixes(self):
        a,m=score_fixture();a={k:np.tile(v,(61,1))if v.ndim==2 else np.tile(v,61)for k,v in a.items()};m=dict(symbols=m['symbols']*61,etfs=m['etfs']);f=synthetic_features(a);s,c,d=r.score_oracle(a,m,f)
        for n in range(123):
            ss,cc,dd=r.score_oracle(a,m,f,n)
            for name in r.NAMES:self.assertEqual(ss[name].tobytes(),s[name][:n].tobytes());self.assertEqual(cc[name],c[name][:n])
    def test_future_feature_not_interpreted(self):
        a,m=score_fixture();f=synthetic_features(a);want=r.score_oracle(a,m,f,1);f['formation_rank'][1]=np.inf;got=r.score_oracle(a,m,f,1)
        for name in r.NAMES:self.assertEqual(want[0][name].tobytes(),got[0][name].tobytes())


class IsolationTests(unittest.TestCase):
    def test_no_producer_import(self):
        tree=ast.parse(Path(r.__file__).read_text());bad=('monthly_formation142_runner','monthly_formation142_features','monthly_formation142_selector')
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):self.assertFalse(any(alias.name in bad for alias in node.names))
            if isinstance(node,ast.ImportFrom):self.assertNotIn(node.module,bad)
    def test_no_actual_audit_on_import(self):self.assertEqual(r.STUDY,'outputs/monthly_formation142')
    def test_invalid_plan_digest_fails_before_file_access(self):
        with self.assertRaises(ValueError):r.audit('/nonexistent','PLAN.json','bad')
    def test_only_registered_round_can_open_cache(self):
        with tempfile.TemporaryDirectory()as directory:
            p=Path(directory)/r.STUDY;p.mkdir(parents=True);f=p/'PLAN.json';f.write_text(json.dumps(dict(round=142,status='DRAFT',arms=list(r.NAMES))))
            with self.assertRaisesRegex(ValueError,'Round142'):r.audit(directory,f,hashlib.sha256(f.read_bytes()).hexdigest())


if __name__=='__main__':unittest.main()
