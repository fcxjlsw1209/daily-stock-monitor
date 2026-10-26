"""Synthetic independent Fraction/original-score checks; no real data."""
import copy
import ast
from pathlib import Path
import unittest
from unittest.mock import patch
from datetime import timedelta
import numpy as np
import pandas as pd
import monthly_profitability141_prepare_review as r


def event(year,gross='100',assets='1000'):
    return dict(event_id=str(year),available_from=f'{year+1}-02-03',admitted=True,
      original_annual=True,measurement_eligible=True,issuer_id='issuer',security_id='stock',currency_key='USD',
      identity_valid_from=f'{year+1}-02-03',identity_valid_through=f'{year+3}-12-31',
      period_start=f'{year}-01-01',period_end=f'{year}-12-31',
      gross_profit=dict(value=gross,scale='1000000'),assets=dict(value=assets,scale='1000000'))


def fixture():
    stocks=['S%02d'%i for i in range(32)];etfs=['E%02d'%i for i in range(17)]
    m=dict(dates=['2022-03-31','2022-04-29'],symbols=[stocks+etfs,stocks+etfs],etfs=etfs)
    return m,dict(schema_version=1,events={'S00':[event(2020),event(2021,'130','1600')]})


class FactTests(unittest.TestCase):
    def test_growth_and_profitability_distinct(self):
        m,d=fixture();x=r.features_oracle(m,d)['arrays'];self.assertEqual(x[r.EXTRA[0]][0,0],.5);self.assertEqual(x[r.EXTRA[1]][0,0],-.5)
    def test_normalizes_independent_scales(self):
        m,d=fixture();e=d['events']['S00'][1];e['gross_profit']=dict(value='130000',scale='1000');self.assertEqual(r.features_oracle(m,d)['arrays'][r.EXTRA[0]][0,0],.5)
    def test_negative_gross_valid(self):
        m,d=fixture();d['events']['S00'][0]['gross_profit']['value']='-200';d['events']['S00'][1]['gross_profit']['value']='-10';self.assertEqual(r.features_oracle(m,d)['arrays'][r.EXTRA[1]][0,0],.5)
    def test_nonpositive_assets_neutral(self):
        m,d=fixture();d['events']['S00'][1]['assets']['value']='0';self.assertFalse(r.features_oracle(m,d)['arrays']['common_available'].any())
    def test_currency_change_neutral(self):
        m,d=fixture();d['events']['S00'][1]['currency_key']='EUR';self.assertFalse(r.features_oracle(m,d)['arrays']['common_available'].any())
    def test_literal_currency_allowed(self):
        m,d=fixture()
        for e in d['events']['S00']:e['currency_key']='issuer:$'
        self.assertTrue(r.features_oracle(m,d)['arrays']['common_available'][0,0])
    def test_latest_invalid_blocks_fallback(self):
        m,d=fixture();d['events']['S00'].append(dict(event(2021),event_id='bad',available_from='2022-03-01',admitted=False));self.assertFalse(r.features_oracle(m,d)['arrays']['common_available'].any())
    def test_future_poison_cannot_change_past(self):
        m,d=fixture();a=r.features_oracle(m,d);d['events']['S00'].append(dict(event_id='future',available_from='2030-01-01',gross_profit='bad'));b=r.features_oracle(m,d)
        for k in r.EXTRA:np.testing.assert_array_equal(a['arrays'][k],b['arrays'][k])
    def test_unresolved_intervening_blocks(self):
        m,d=fixture();d['events']['S00'].append(dict(event_id='unknown',available_from='2021-03-01'));self.assertFalse(r.features_oracle(m,d)['arrays']['common_available'].any())
    def test_same_day_conflict_neutral(self):
        m,d=fixture();d['events']['S00'].append(dict(d['events']['S00'][-1],event_id='other'));self.assertFalse(r.features_oracle(m,d)['arrays']['common_available'].any())
    def test_noncontiguous_period_neutral(self):
        m,d=fixture();d['events']['S00'][-1]['period_start']='2021-01-02';self.assertFalse(r.features_oracle(m,d)['arrays']['common_available'].any())
    def test_identity_interval(self):
        m,d=fixture();d['events']['S00'][-1]['identity_valid_through']='2022-03-01';self.assertFalse(r.features_oracle(m,d)['arrays']['common_available'].any())
    def test_same_issuer_and_security(self):
        m,d=fixture();d['events']['S00'][-1]['issuer_id']='other';self.assertFalse(r.features_oracle(m,d)['arrays']['common_available'].any())
    def test_stale_report(self):
        m,d=fixture();m['dates']=['2024-03-29','2024-04-30'];self.assertFalse(r.features_oracle(m,d)['arrays']['common_available'].any())
    def test_unchanged_inputs_and_matched_mask(self):
        m,d=fixture();before=copy.deepcopy((m,d));a=r.features_oracle(m,d);self.assertEqual((m,d),before)
        for k in r.EXTRA[:-1]:np.testing.assert_array_equal(np.isfinite(a['arrays'][k]),a['arrays']['common_available'])
    def test_original_axes_etf_and_noncovered_neutral(self):
        m,d=fixture();d['events']['E00']=d['events']['S00'];a=r.features_oracle(m,d);self.assertEqual(a['arrays']['common_available'].shape,(2,49));self.assertEqual(a['arrays']['common_available'].sum(),2)
    def test_fraction_keeps_tiny_difference(self):
        m,d=fixture();d['events']['S00'][0]['period_start']='2020-01-02';d['events']['S00'][1]['gross_profit']['value']='100.00000000000000000000000000000001';d['events']['S00'][1]['assets']['value']='1000';self.assertEqual(r.features_oracle(m,d)['arrays'][r.EXTRA[1]][0,0],.5)
    def test_empty_prefix(self):
        m,d=fixture();m.update(dates=[],symbols=[]);self.assertEqual(r.features_oracle(m,d)['arrays']['common_available'].shape,(0,49))
    def test_synthetic_independent_matches_producer(self):
        import monthly_profitability141_factor as producer
        m,d=fixture();a=r.features_oracle(m,d);b=producer.build_features(m,d);r.verify_audit(b['audit'],a)
        for key in r.EXTRA:np.testing.assert_array_equal(a['arrays'][key],b['arrays'][key])
    def test_tampered_audit_rejected(self):
        import monthly_profitability141_factor as producer
        m,d=fixture();a=r.features_oracle(m,d);b=producer.build_features(m,d);b['audit']['cells'][0]['available']=False
        with self.assertRaises(ValueError):r.verify_audit(b['audit'],a)
    def test_all_chronology_audit_fields_reconstructed(self):
        import monthly_profitability141_factor as producer
        m,d=fixture();expected=r.features_oracle(m,d);saved=producer.build_features(m,d)['audit']
        for key in ('current_available_from','prior_available_from','current_period_end',
                    'prior_period_end','current_days','prior_days','currency_key'):
            with self.subTest(key=key):
                bad=copy.deepcopy(saved);bad['cells'][0][key]=None
                with self.assertRaises(ValueError):r.verify_audit(bad,expected)
    def test_rejected_pairs_chronology_matches(self):
        import monthly_profitability141_factor as producer
        for mutation in ('currency','identity','invalid_latest','prior_invalid','intervening','stale'):
            with self.subTest(mutation=mutation):
                m,d=fixture();current=d['events']['S00'][-1]
                if mutation=='currency':current['currency_key']='EUR'
                elif mutation=='identity':current['issuer_id']='other'
                elif mutation=='invalid_latest':current['admitted']=False
                elif mutation=='prior_invalid':d['events']['S00'][0]['admitted']=False
                elif mutation=='intervening':d['events']['S00'].append(dict(event_id='unknown',available_from='2021-03-01'))
                else:m['dates']=['2024-03-29','2024-04-30']
                r.verify_audit(producer.build_features(m,d)['audit'],r.features_oracle(m,d))
    def test_audit_boolean_fit_count_rejected(self):
        import monthly_profitability141_factor as producer
        m,d=fixture();saved=producer.build_features(m,d)['audit'];saved['new_model_fits']=False
        with self.assertRaises(ValueError):r.verify_audit(saved,r.features_oracle(m,d))
    def test_audit_source_provenance_bound(self):
        import monthly_profitability141_factor as producer
        m,d=fixture();expected=r.features_oracle(m,d);saved=producer.build_features(m,d)['audit']
        saved.update(admitted_dataset_sha256='a'*64,source_admission_sha256='b'*64)
        r.verify_audit(saved,expected,dataset_sha='a'*64,admission_sha='b'*64)
        for kwargs in (dict(dataset_sha='c'*64,admission_sha='b'*64),
                       dict(dataset_sha='a'*64,admission_sha='c'*64),dict(dataset_sha='a'*64)):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):r.verify_audit(saved,expected,**kwargs)
    def test_no_producer_import(self):
        tree=ast.parse(Path(r.__file__).read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):self.assertFalse(any('profitability141_factor' in a.name or 'profitability141_runner' in a.name for a in node.names))
            if isinstance(node,ast.ImportFrom):self.assertNotIn(node.module,('monthly_profitability141_factor','monthly_profitability141_runner'))


def score_fixture():
    shape=(2,5);baseline=np.array([[.16,.12,.08,.07,-.10],[.15,.13,.10,.09,-.10]])
    core=np.array([[.10,.08,.05,.04,-.10],[.10,.08,.06,.05,-.10]])
    a=dict(baseline=baseline,core=core,eligible=np.ones(shape,bool),core_weight=np.array([.5,.52]),
       state_weak_narrow=np.array([False,True]),state_calm10=np.array([True,False]),
       pre_score=baseline.copy(),effects=np.zeros(shape),forecast_available=np.zeros(shape,bool),
       rank_log_excess=np.zeros(shape),old_ml4=np.zeros(shape),extension_available=np.zeros(shape,bool),
       model_available=np.zeros(shape,bool),raw_local_constant=np.zeros(shape),
       raw_local_linear=np.zeros(shape),raw_source_log=np.zeros(shape))
    return a,dict(symbols=[list('ABCDE'),list('ABCDE')],etfs=['E'])


class ScoreTests(unittest.TestCase):
    def test_common_missing_exact_baseline(self):
        a,m=score_fixture();hand=np.full((2,5),np.nan);s,c,d=r.score_oracle(a,m,hand,hand)
        for name in r.NAMES[1:]:self.assertEqual(s[name].tobytes(),s[r.NAMES[0]].tobytes());self.assertEqual(c[name],c[r.NAMES[0]])
    def test_zero_hand_exact_baseline(self):
        a,m=score_fixture();hand=np.zeros((2,5));hand[:,-1]=np.nan;s,c,d=r.score_oracle(a,m,hand,hand)
        self.assertEqual(np.count_nonzero(d[r.NAMES[2]]),0);self.assertEqual(s[r.NAMES[2]].tobytes(),s[r.NAMES[0]].tobytes())
    def test_bounded_corrected_support(self):
        a,m=score_fixture();hand=np.full((2,5),.5);hand[:,-1]=np.nan;_,_,d=r.score_oracle(a,m,hand,hand)
        support=(a['baseline']-a['core'])/(1-a['core_weight'][:,None]);expected=.025*(np.minimum(.5,support+.5)-support);expected[:,-1]=0.
        np.testing.assert_allclose(d[r.NAMES[2]],expected,rtol=0,atol=1e-16)
    def test_ineligible_keeps_delta_zero(self):
        a,m=score_fixture();a['eligible'][0,0]=False;hand=np.full((2,5),.4);hand[:,-1]=np.nan;_,_,d=r.score_oracle(a,m,hand,hand);self.assertEqual(d[r.NAMES[2]][0,0],0.)
    def test_etf_finite_rejected(self):
        a,m=score_fixture();hand=np.full((2,5),.1)
        with self.assertRaises(ValueError):r.score_oracle(a,m,hand,hand)
    def test_unmatched_masks_rejected(self):
        a,m=score_fixture();hand=np.full((2,5),np.nan);other=hand.copy();other[0,0]=.2
        with self.assertRaises(ValueError):r.score_oracle(a,m,hand,other)
    def test_future_not_interpreted(self):
        a,m=score_fixture();hand=np.zeros((2,5));hand[:,-1]=np.nan;expected=r.score_oracle(a,m,hand,hand,1);hand[1]=np.inf;actual=r.score_oracle(a,m,hand,hand,1)
        for name in r.NAMES:self.assertEqual(expected[0][name].tobytes(),actual[0][name].tobytes());self.assertEqual(expected[1][name],actual[1][name])
    def test_one_episode_per_arm(self):
        a,m=score_fixture();hand=np.full((2,5),np.nan);h=r.old.original_helper()
        with patch.object(r.old,'original_helper',return_value=h),patch.object(h,'episode',wraps=h.episode)as spy:
            r.score_oracle(a,m,hand,hand);self.assertEqual(spy.call_count,3)
    def test_source_arrays_unchanged(self):
        a,m=score_fixture();before={k:v.tobytes()for k,v in a.items()};hand=np.full((2,5),np.nan);r.score_oracle(a,m,hand,hand);self.assertEqual(before,{k:v.tobytes()for k,v in a.items()})


def full_score_fixture():
    a,_=score_fixture();a={k:np.tile(v,(61,10))[:,:49] if v.ndim==2 else np.tile(v,61) for k,v in a.items()}
    names=['S%02d'%j for j in range(32)]+['E%02d'%j for j in range(17)]
    m=dict(symbols=[names[:] for _ in range(122)],etfs=names[32:])
    raw=np.tile(np.linspace(-.48,.48,49),(122,1));cw=a['core_weight'][:,None];a['core']=cw*raw
    a['baseline']=a['core']+np.tile(np.sin(np.arange(49))*.015,(122,1));a['pre_score']=a['baseline'].copy()
    a['effects']=np.tile(np.cos(np.arange(49))*.05,(122,1))
    beta=np.tile(np.sign(np.linspace(-1.,1.,32))*.5,(122,1));rho=beta[:,::-1].copy()
    beta=np.c_[beta,np.full((122,17),np.nan)];rho=np.c_[rho,np.full((122,17),np.nan)]
    beta[5,:]=np.nan;rho[5,:]=np.nan;a.update(gross_profit_growth_sign=beta,gross_profitability_improvement_sign=rho)
    return a,m,beta,rho


class FullScoreIntegrationTests(unittest.TestCase):
    def backend(self,m):
        import monthly_profitability141_runner as producer
        h=r.old.original_helper()
        class Backend:
            adapter=producer.module('independent141_test_adapter',r.ROOT/'outputs/monthly_path70_research/incumbent_adapter.py')
            @staticmethod
            def baseline(a,meta,n,policy):
                x=h.original_prefix(a,meta,n);c,s,_=h.episode(x['local'],x,policy=policy);return c,s
            @staticmethod
            def pre(a,n):
                x=h.original_prefix(a,m,n);full=np.zeros_like(a['baseline']);full[:n]=x['local'];return full
            @staticmethod
            def selector(score,eligible,names,effects,original,policy):
                x=dict(baseline=original['baseline'],core=original['core'],cw=original['core_weight'],
                    support=(original['baseline']-original['core'])/(1-original['core_weight']),
                    eligible=eligible,names=names,effects=effects,calm=original['state_calm10'],weak=original['state_weak_narrow'])
                c,s,_=h.episode(score,x,policy=policy);return c,s
        return Backend()
    def test_all366_synthetic_choices_and_three_scores(self):
        import monthly_profitability141_runner as producer
        a,m,beta,rho=full_score_fixture();ss,cc,_=r.score_oracle(a,m,beta,rho);backend=self.backend(m)
        for name in r.NAMES:
            c,s=producer.choose(name,a,m,122,backend,r.old.original_helper().POLICY)
            np.testing.assert_allclose(s,ss[name],rtol=0,atol=r.ATOL);self.assertEqual(c,cc[name])
    def test_prefixes_with_original_episode(self):
        import monthly_profitability141_runner as producer
        a,m,beta,rho=full_score_fixture();backend=self.backend(m)
        for n in range(123):
            ss,cc,_=r.score_oracle(a,m,beta,rho,n)
            for name in r.NAMES:
                c,s=producer.choose(name,a,m,n,backend,r.old.original_helper().POLICY)
                np.testing.assert_allclose(s,ss[name],rtol=0,atol=r.ATOL);self.assertEqual(c,cc[name])
    def test_interface_exact(self):
        import monthly_profitability141_runner as producer
        for key in ('INPUTS','CACHE_FILES','NAMES','CHECKS','EXTRA'):self.assertEqual(getattr(r,key),getattr(producer,key))
    def test_independent_scores_do_not_call_new_producer(self):
        import monthly_profitability141_runner as producer
        a,m,beta,rho=full_score_fixture()
        with patch.object(producer,'choose',side_effect=AssertionError('not an oracle')):
            scores,choices,_=r.score_oracle(a,m,beta,rho);self.assertEqual(len(choices[r.NAMES[2]]),122)


if __name__=='__main__':unittest.main()
