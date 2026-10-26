"""Synthetic independent Fraction/original-score checks; no real data."""
import copy
import ast
from pathlib import Path
import unittest
from unittest.mock import patch
from datetime import timedelta
import numpy as np
import pandas as pd
import monthly_same_report144_prepare_review as r


def event(year=2021,gross='130',assets='1600'):
    end=r.date(year,12,31); prior=r.date(year-1,12,31);available=r.date(year+1,2,3)
    def period(y,g,a):
        start=r.date(y,1,1);end=r.date(y,12,31)
        return dict(period_start=str(start),period_end=str(end),days_inclusive=(end-start).days+1,
                    gross_profit=dict(value=g,scale='1000000',currency_literal='$'),assets=dict(value=a,scale='1000000',currency_literal='$'))
    eid='S00:'+str(year)
    return dict(event_id=eid,symbol='S00',security_id='S00',issuer_id='issuer',currency_key='same-report-dollar:'+eid,
        method='same_current_release_annual_comparison',admitted=True,release_date=str(available-r.timedelta(days=1)),available_from=str(available),
        identity_valid_from=str(available),identity_valid_through=str(end+r.timedelta(days=550)),
        same_report=dict(current=period(year,gross,assets),comparison=period(year-1,'100','1000')))


def fixture():
    stocks=['S%02d'%i for i in range(32)];etfs=['E%02d'%i for i in range(17)]
    m=dict(dates=['2022-03-31','2022-04-29'],symbols=[stocks+etfs,stocks+etfs],etfs=etfs)
    return m,dict(schema_version=1,method='same_current_release_annual_comparison',events=[event()])


class FactTests(unittest.TestCase):
    def check(self,m,d):
        import monthly_same_report144_factor as p
        a=r.features_oracle(m,d);b=p.build_features(m,d);r.verify_audit(b['audit'],a)
        for k in r.EXTRA:np.testing.assert_array_equal(a['arrays'][k],b['arrays'][k])
        return a
    def test_growth_profitability_diverge(self):
        m,d=fixture();a=self.check(m,d)['arrays'];self.assertEqual(a[r.EXTRA[0]][0,0],.5);self.assertEqual(a[r.EXTRA[1]][0,0],-.5)
    def test_current_release_controls_both_columns(self):
        m,d=fixture();m['dates']=['2021-12-31','2022-02-02'];self.assertFalse(self.check(m,d)['arrays']['common_available'].any())
    def test_first_current_available_session(self):
        m,d=fixture();m['dates']=['2022-02-02','2022-02-03'];np.testing.assert_array_equal(self.check(m,d)['arrays']['common_available'][:,0],[False,True])
    def test_expired_latest_blocks_valid_older(self):
        m,d=fixture();d['events'].append(dict(event(),event_id='expired',available_from='2022-03-01',identity_valid_through='2022-03-02'));self.assertFalse(self.check(m,d)['arrays']['common_available'].any())
    def test_rejected_latest_blocks_old(self):
        m,d=fixture();d['events'].append(dict(event(),event_id='bad',available_from='2022-03-01',admitted=False));self.assertFalse(self.check(m,d)['arrays']['common_available'].any())
    def test_future_poison_ignored(self):
        m,d=fixture();before=self.check(m,d);d['events'].append(dict(event_id='future',symbol='S00',available_from='2030-01-01',same_report='bad'));after=self.check(m,d)
        for k in r.EXTRA:np.testing.assert_array_equal(before['arrays'][k],after['arrays'][k])
    def test_same_day_conflicts_unavailable(self):
        m,d=fixture();d['events'].append(dict(event(),event_id='other'));self.assertFalse(self.check(m,d)['arrays']['common_available'].any())
    def test_duplicate_id_conflicts_unavailable(self):
        m,d=fixture();d['events'].append(dict(event(),admitted=False));self.assertFalse(self.check(m,d)['arrays']['common_available'].any())
    def test_exact_duplicates_idempotent(self):
        m,d=fixture();d['events'].append(copy.deepcopy(d['events'][0]));self.assertTrue(self.check(m,d)['arrays']['common_available'][0,0])
    def test_all_invalid_monetary_and_date_inputs(self):
        for role in ('current','comparison'):
            for field,value in [('days_inclusive',False),('period_start','2021-01-02')]:
                m,d=fixture();d['events'][0]['same_report'][role][field]=value
                with self.subTest(role=role,field=field):self.assertFalse(self.check(m,d)['arrays']['common_available'].any())
            for key,field,value in [('assets','value','0'),('assets','value','-1'),('assets','scale','0'),('gross_profit','value','NaN'),('gross_profit','currency_literal','EUR')]:
                m,d=fixture();d['events'][0]['same_report'][role][key][field]=value
                with self.subTest(role=role,key=key,field=field):self.assertFalse(self.check(m,d)['arrays']['common_available'].any())
    def test_identity_bounds_and_currency(self):
        for key,value in [('currency_key','USD'),('security_id','other'),('identity_valid_from','2022-02-02'),('identity_valid_through','2029-12-31'),('release_date','2022-02-04')]:
            m,d=fixture();d['events'][0][key]=value
            with self.subTest(key=key):self.assertFalse(self.check(m,d)['arrays']['common_available'].any())
    def test_staleness_550_inclusive(self):
        m,d=fixture();end=r.date(2021,12,31);m['dates']=[str(end+r.timedelta(days=550)),str(end+r.timedelta(days=551))]
        np.testing.assert_array_equal(self.check(m,d)['arrays']['common_available'][:,0],[True,False])
    def test_negative_gp_valid(self):
        m,d=fixture();d['events'][0]['same_report']['comparison']['gross_profit']['value']='-200';d['events'][0]['same_report']['current']['gross_profit']['value']='-100';self.assertEqual(self.check(m,d)['arrays'][r.EXTRA[0]][0,0],.5)
    def test_exact_decimal_near_tie(self):
        m,d=fixture();e=d['events'][0];e['same_report']['comparison']['gross_profit']['value']='366';e['same_report']['current']['gross_profit']['value']='365.00000000000000000000000000000000000001';e['same_report']['current']['assets']['value']='1000';self.assertEqual(self.check(m,d)['arrays'][r.EXTRA[1]][0,0],.5)
    def test_scales_normalize_independently(self):
        m,d=fixture();e=d['events'][0]['same_report']['current']['gross_profit'];e.update(value='130000',scale='1000');self.assertEqual(self.check(m,d)['arrays'][r.EXTRA[0]][0,0],.5)
    def test_no_earlier_monetry_event_needed(self):
        m,d=fixture();self.assertEqual(len(d['events']),1);self.assertTrue(self.check(m,d)['arrays']['common_available'][0,0])
    def test_empty_prefix(self):
        m,d=fixture();m.update(dates=[],symbols=[]);self.assertEqual(self.check(m,d)['arrays']['common_available'].shape,(0,49))
    def test_inputs_unchanged_and_mask_exact(self):
        m,d=fixture();saved=copy.deepcopy((m,d));a=self.check(m,d)['arrays'];self.assertEqual((m,d),saved)
        for k in r.EXTRA[:-1]:np.testing.assert_array_equal(np.isfinite(a[k]),a['common_available'])
    def test_every_audit_field_checked(self):
        import monthly_same_report144_factor as p
        m,d=fixture();want=r.features_oracle(m,d);saved=p.build_features(m,d)['audit']
        for key in want['cells'][0]:
            bad=copy.deepcopy(saved);bad['cells'][0][key]='tampered'
            with self.subTest(key=key),self.assertRaises(ValueError):r.verify_audit(bad,want)
    def test_audit_sources_and_counts(self):
        import monthly_same_report144_factor as p
        m,d=fixture();want=r.features_oracle(m,d);saved=p.build_features(m,d)['audit'];saved.update(admitted_dataset_sha256='a'*64,source_admission_sha256='b'*64)
        r.verify_audit(saved,want,dataset_sha='a'*64,admission_sha='b'*64)
        with self.assertRaises(ValueError):r.verify_audit(saved,want,dataset_sha='b'*64,admission_sha='b'*64)
        saved['new_model_fits']=False
        with self.assertRaises(ValueError):r.verify_audit(saved,want)
    def test_no_producer_imports_in_oracle(self):
        tree=ast.parse(Path(r.__file__).read_text())
        for n in ast.walk(tree):
            if isinstance(n,ast.Import):self.assertFalse(any(a.name.endswith(('_factor','_runner'))for a in n.names))
            if isinstance(n,ast.ImportFrom):self.assertFalse((n.module or '').endswith(('_factor','_runner')))


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
        import monthly_same_report144_runner as producer
        h=r.old.original_helper()
        class Backend:
            adapter=producer.module('independent144_test_adapter',r.ROOT/'outputs/monthly_path70_research/incumbent_adapter.py')
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
        import monthly_same_report144_runner as producer
        a,m,beta,rho=full_score_fixture();ss,cc,_=r.score_oracle(a,m,beta,rho);backend=self.backend(m)
        for name in r.NAMES:
            c,s=producer.choose(name,a,m,122,backend,r.old.original_helper().POLICY)
            np.testing.assert_allclose(s,ss[name],rtol=0,atol=r.ATOL);self.assertEqual(c,cc[name])
    def test_prefixes_with_original_episode(self):
        import monthly_same_report144_runner as producer
        a,m,beta,rho=full_score_fixture();backend=self.backend(m)
        for n in range(123):
            ss,cc,_=r.score_oracle(a,m,beta,rho,n)
            for name in r.NAMES:
                c,s=producer.choose(name,a,m,n,backend,r.old.original_helper().POLICY)
                np.testing.assert_allclose(s,ss[name],rtol=0,atol=r.ATOL);self.assertEqual(c,cc[name])
    def test_interface_exact(self):
        import monthly_same_report144_runner as producer
        for key in ('INPUTS','CACHE_FILES','NAMES','CHECKS','EXTRA'):self.assertEqual(getattr(r,key),getattr(producer,key))
    def test_independent_scores_do_not_call_new_producer(self):
        import monthly_same_report144_runner as producer
        a,m,beta,rho=full_score_fixture()
        with patch.object(producer,'choose',side_effect=AssertionError('not an oracle')):
            scores,choices,_=r.score_oracle(a,m,beta,rho);self.assertEqual(len(choices[r.NAMES[2]]),122)



class AdditionalTests(unittest.TestCase):
    def test_every_financial_source_prefix(self):
        m,d=fixture();tester=FactTests();full=tester.check(m,d)
        for n in range(len(m['dates'])+1):
            sub=dict(m,dates=m['dates'][:n],symbols=m['symbols'][:n]);data=dict(d,events=[e for e in d['events']if n and e['available_from']<=m['dates'][n-1]])
            part=tester.check(sub,data)
            for k in r.EXTRA:np.testing.assert_array_equal(part['arrays'][k],full['arrays'][k][:n])
    def test_near_tie_rank_order_independent(self):
        import monthly_same_report144_runner as producer
        a,m,beta,rho=full_score_fixture();test=FullScoreIntegrationTests();backend=test.backend(m)
        for delta in (0.,-2e-13,2e-13,-2e-11,2e-11):
            a['core'][0,:3]=.05;a['baseline'][0,:3]=[.1,.1+delta,.1-delta];a['pre_score'][0,:3]=a['baseline'][0,:3]
            ss,cc,_=r.score_oracle(a,m,beta,rho)
            for name in r.NAMES:
                actual,scores=producer.choose(name,a,m,122,backend,r.old.original_helper().POLICY)
                self.assertEqual(actual,cc[name]);np.testing.assert_allclose(scores,ss[name],rtol=0,atol=r.ATOL)
    def test_invalid_latest_same_period_does_not_resurrect_previous(self):
        m,d=fixture();d['events'].insert(0,event(2020,'100','1000'));d['events'][-1]['admitted']=False
        self.assertFalse(FactTests().check(m,d)['arrays']['common_available'].any())


def write_prereview():
    """Synthetic-only numerical/lifecycle review; no historical arrays loaded."""
    import io,sys,json,hashlib
    root=r.ROOT
    source_paths=r.source_files()
    implementations=[root/name for name in ['work/monthly_same_report144_factor.py','work/monthly_same_report144_runner.py','work/test_monthly_same_report144_factor.py','work/test_monthly_same_report144_runner.py','work/register_monthly_same_report144.py','work/test_register_monthly_same_report144.py']]
    digest=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    before={str(p):digest(p)for p in source_paths+implementations}
    proposal=root/'work/monthly_same_report144_proposal.json'
    if digest(proposal)!=r.PROPOSAL_SHA:raise ValueError('Prospective proposal mismatch')
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
    if not result.wasSuccessful():raise SystemExit(stream.getvalue())
    if before!={p:digest(p)for p in before}:raise ValueError('Reviewed source changed during tests')
    log=root/'outputs/monthly_same_report144/PREPARE_REVIEW_TESTS.log'
    with log.open('x')as f:f.write(stream.getvalue())
    record=dict(round=144,stage='preregistration_prepare_review',proposal_sha256=r.PROPOSAL_SHA,
        synthetic_only=True,real_144_factor_or_return_calculations=0,passed=True,
        source_hashes={str(p):digest(p)for p in source_paths},implementation_hashes={str(p):digest(p)for p in implementations},
        input_hashes={str(proposal):digest(proposal)},tests_run=result.testsRun,
        synthetic_suites=[dict(name='prepare_review',tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),log=stream.getvalue())],
        log_path=str(log),log_sha256=digest(log),material_remaining_issues=[],
        scope=['Independent same-current-report comparison; all four values first available at current release next session.',
               'Exact Fraction division independently checks producer Decimal cross-product financial signs; latest-invalid blocks fallback.',
               'Original 27 arrays and unchanged 2.5% correction/one episode; all 123 prefixes and near-tie order checks.',
               'Independent nine-path accounting, dual first88 gates, fixed units, cost equation and source/frozen hash tampering covered.'],
        limitations=['Synthetic and source review only, no actual round144 historical factors/orders/model fits/returns evaluated.',
                     'Conditional AAPL/AMD current-report source assumptions and inherited universe/data-vintage caveats remain.'])
    destination=root/'work/monthly_same_report144_prepare_prereview.json'
    with destination.open('x')as f:json.dump(record,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print(json.dumps(dict(passed=True,tests=result.testsRun,path=str(destination),sha256=digest(destination),log=str(log))))

if __name__=='__main__':
    import sys
    if sys.argv[1:]==['--write-prereview']:write_prereview()
    else:unittest.main(verbosity=2)
