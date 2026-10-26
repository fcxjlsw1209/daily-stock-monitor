"""Synthetic-only independent round134 checks; never opens market caches."""
import copy
import itertools
import math
import unittest
import numpy as np
import monthly_latent_state134_prepare_review as r


def parameters():
    return dict(pi=[.6,.4], transition=[[.88,.12],[.2,.8]], means=[-.2,.5], variances=[.3,1.4])


def brute(z,p):
    paths=list(itertools.product(range(2),repeat=len(z))); weights=[]
    e=r.emissions(z,p)
    for path in paths:
        w=p['pi'][path[0]]*math.exp(e[0][path[0]])
        for t in range(1,len(z)):w*=p['transition'][path[t-1]][path[t]]*math.exp(e[t][path[t]])
        weights.append(w)
    total=math.fsum(weights)
    gamma=[[math.fsum(w for s,w in zip(paths,weights) if s[t]==k)/total for k in range(2)] for t in range(len(z))]
    xi=[[math.fsum(w*sum(s[t]==j and s[t+1]==k for t in range(len(z)-1)) for s,w in zip(paths,weights))/total for k in range(2)] for j in range(2)]
    return math.log(total),gamma,xi


def fixture():
    shape=(2,5)
    baseline=np.array([[.16,.12,.08,.07,-.10],[.15,.13,.10,.09,-.10]])
    core=np.array([[.10,.08,.05,.04,-.10],[.10,.08,.06,.05,-.10]])
    a=dict(baseline=baseline,core=core,eligible=np.ones(shape,bool),core_weight=np.array([.5,.52]),
       state_weak_narrow=np.array([False,True]),state_calm10=np.array([True,False]),
       pre_score=baseline.copy(),effects=np.zeros(shape),forecast_available=np.zeros(shape,bool),
       rank_log_excess=np.zeros(shape),old_ml4=np.zeros(shape),extension_available=np.zeros(shape,bool),
       model_available=np.zeros(shape,bool),raw_local_constant=np.zeros(shape),
       raw_local_linear=np.zeros(shape),raw_source_log=np.zeros(shape))
    return a,dict(symbols=[list('ABCDE'),list('ABCDE')])


class ProbabilityTests(unittest.TestCase):
    def test_likelihood_full_path_enumeration(self):
        z=[-.3,.1,1.8,-.4,.2];p=parameters()
        self.assertAlmostEqual(r.sufficient_statistics(z,p)['log_likelihood'],brute(z,p)[0],13)

    def test_smoothed_gamma_enumeration(self):
        z=[-.3,.1,1.8,-.4,.2];p=parameters()
        np.testing.assert_allclose(r.sufficient_statistics(z,p)['gamma'],brute(z,p)[1],rtol=0,atol=2e-15)

    def test_transition_counts_enumeration(self):
        z=[-.3,.1,1.8,-.4,.2];p=parameters()
        np.testing.assert_allclose(r.sufficient_statistics(z,p)['xi_sum'],brute(z,p)[2],rtol=0,atol=2e-15)

    def test_each_filtered_prefix_enumeration(self):
        z=[-.3,.1,1.8,-.4,.2];p=parameters();s=r.sufficient_statistics(z,p)
        for t in range(1,len(z)):
            np.testing.assert_allclose(s['filtered'][t],brute(z[:t+1],p)[1][-1],atol=2e-15,rtol=0)

    def test_gamma_sum_and_xi_total(self):
        z=[-.1,.2,3.,.4,-2.];s=r.sufficient_statistics(z,parameters())
        self.assertAlmostEqual(sum(s['occupancy']),len(z),13)
        self.assertAlmostEqual(sum(sum(row) for row in s['xi_sum']),len(z)-1,13)

    def test_backward_information_is_not_trading_filter(self):
        s=r.sufficient_statistics([0.,0.,0.,5.,5.],parameters())
        self.assertGreater(abs(s['gamma'][1][0]-s['filtered'][1][0]),1e-4)
        np.testing.assert_allclose(s['gamma'][-1],s['filtered'][-1],rtol=0,atol=1e-15)

    def test_future_prefix_isolation(self):
        p=parameters();z=[0.,-.1,.2,.5]
        a=r.sufficient_statistics(z+[1.,-1.],p)['filtered']
        b=r.sufficient_statistics(z+[100.,-100.],p)['filtered']
        self.assertEqual(a[:len(z)],b[:len(z)])

    def test_state_permutation(self):
        p=parameters();swap=dict(pi=p['pi'][::-1],transition=[row[::-1] for row in p['transition'][::-1]],means=p['means'][::-1],variances=p['variances'][::-1])
        a=r.sufficient_statistics([0.,1.,-.3],p);b=r.sufficient_statistics([0.,1.,-.3],swap)
        self.assertAlmostEqual(a['log_likelihood'],b['log_likelihood'],14)
        np.testing.assert_allclose(a['gamma'],[row[::-1] for row in b['gamma']],atol=1e-15,rtol=0)
        self.assertEqual(r.canonical(p),r.canonical(swap))

    def test_stationary(self):
        p=parameters();s=r.stationary(p)
        for k in range(2):self.assertAlmostEqual(s[k],sum(s[j]*p['transition'][j][k] for j in range(2)),14)

    def test_equal_transition_rows_remove_memory(self):
        p=parameters();p['transition']=[[.6,.4],[.6,.4]]
        x=r.step_filter(.3,[.99,.01],p);y=r.step_filter(.3,r.stationary(p),p,predict=False)
        np.testing.assert_allclose(x,y,rtol=0,atol=1e-15)

    def test_persistent_transition_retains_memory(self):
        p=parameters()
        a=r.step_filter(.3,[.99,.01],p);b=r.step_filter(.3,[.01,.99],p)
        self.assertGreater(abs(a[0]-b[0]),.1)

    def test_mstep_weighted_mean_variance(self):
        z=[-1.,0.,2.];s=r.sufficient_statistics(z,parameters());p=r.em_update(z,s)
        for k in range(2):
            w=np.array(s['gamma'])[:,k]
            self.assertAlmostEqual(p['means'][k],np.average(z,weights=w),14)
            self.assertAlmostEqual(p['variances'][k],max(r.VAR_FLOOR,np.average((np.array(z)-p['means'][k])**2,weights=w)),14)

    def test_floor_applied_to_probability_not_raw_count(self):
        s=dict(occupancy=[2.,2.],gamma=[[.5,.5]]*4,xi_sum=[[1e9,1e-20],[1e-20,1e9]])
        p=r.em_update([0.,0.,0.,0.],s)
        self.assertAlmostEqual(p['transition'][0][1],1e-8/(1+1e-8),20)
        self.assertEqual(p['variances'],[r.VAR_FLOOR]*2)

    def test_underflow_stable_log_domain(self):
        s=r.sufficient_statistics([100.,-100.,200.],parameters())
        self.assertTrue(math.isfinite(s['log_likelihood']))
        for row in s['filtered']:self.assertAlmostEqual(sum(row),1.,12)

    def test_invalid_numeric_and_normalization(self):
        for v in [float('nan'),float('inf'),True]:
            with self.assertRaises(ValueError):r.sufficient_statistics([0.,v],parameters())
        p=parameters();p['pi']=[.5,.6]
        with self.assertRaises(ValueError):r.sufficient_statistics([0.,1.],p)

    def test_unavailable_short_and_constant(self):
        self.assertFalse(r.fit_oracle([.01]*755)['available'])
        x=r.fit_oracle([.01]*756)
        self.assertEqual(x['reason'],'scale_unavailable')

    def test_past_only_query_recursion(self):
        fitted=dict(available=True,parameters=parameters(),mean=.001,sd=.02,terminal_filtered=[.7,.3])
        a=r.query_oracle([.01,.03,-.02],fitted);b=r.query_oracle([.01,.03,2.],fitted)
        self.assertEqual(a[:2],b[:2])
        first=r.step_filter((.01-.001)/.02,[.7,.3],parameters())[0]
        self.assertAlmostEqual(a[0]['filtered'],first,14)


class ScoreTests(unittest.TestCase):
    def test_missing_is_exact_baseline(self):
        a,m=fixture();raw=np.zeros((2,5));risk=np.ones((2,5))*.1
        s,c,d=r.score_oracle(a,m,raw,risk,[None,None],[None,None])
        for name in r.NAMES[1:]:
            self.assertEqual(s[name].tobytes(),s[r.NAMES[0]].tobytes());self.assertEqual(c[name],c[r.NAMES[0]])
            self.assertEqual(np.count_nonzero(d[name]),0)

    def test_zero_probability_means_risk_not_missing(self):
        a,m=fixture();raw=np.ones((2,5))*.3;risk=np.ones((2,5))*.2
        _,_,d=r.score_oracle(a,m,raw,risk,[0.,0.],[1.,1.])
        cw=a['core_weight'][:,None]
        np.testing.assert_allclose(d[r.NAMES[2]],cw*risk-a['core'],atol=1e-16)
        np.testing.assert_allclose(d[r.NAMES[1]],cw*raw-a['core'],atol=1e-16)

    def test_both_flavors_required_same_cell(self):
        a,m=fixture();raw=np.ones((2,5))*.1;risk=np.ones((2,5))*.2;raw[0,1]=np.nan;risk[1,2]=np.nan
        _,_,d=r.score_oracle(a,m,raw,risk,[.4,.6],[.6,.4])
        for name in r.NAMES[1:]:self.assertEqual(d[name][0,1],0.);self.assertEqual(d[name][1,2],0.)

    def test_support_identity_preserved(self):
        a,m=fixture();raw=np.zeros((2,5));risk=np.ones((2,5))*.1
        _,_,d=r.score_oracle(a,m,raw,risk,[.4,.6],[.6,.4]);cw=a['core_weight'][:,None]
        original=(a['baseline']-a['core'])/(1-cw)
        for name in r.NAMES[1:]:
            amended=(a['baseline']+d[name]-a['core']-d[name])/(1-cw)
            np.testing.assert_allclose(original,amended,rtol=0,atol=1e-16)

    def test_prefix_does_not_inspect_later_probabilities(self):
        a,m=fixture();rank=np.ones((2,5))*.1
        x=r.score_oracle(a,m,rank,rank,[.3,float('inf')],[.6,float('inf')],1)
        y=r.score_oracle(a,m,rank,rank,[.3,.1],[.6,.2],1)
        for name in r.NAMES:np.testing.assert_array_equal(x[0][name],y[0][name]);self.assertEqual(x[1][name],y[1][name])

    def test_invalid_probability_rejected(self):
        a,m=fixture();rank=np.ones((2,5))*.1
        for q in [-.01,1.01,True,float('inf')]:
            with self.assertRaises(ValueError):r.score_oracle(a,m,rank,rank,[q,.1],[.6,.2])




class ModelParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import monthly_latent_state134_model as producer
        cls.producer=producer
        rng=np.random.default_rng(840134)
        cls.raw=np.concatenate([rng.normal(0,.003,270),rng.normal(.001,.03,270),rng.normal(-.001,.004,270),rng.normal(0,.025,270)])
        cls.expected=r.fit_record(cls.raw)
        cls.actual=producer.fit(cls.raw)

    def test_complete_synthetic_fit_schema_and_values(self):
        r.compare(self.actual,self.expected,'synthetic_fit')

    def test_final_sufficient_statistics_independent(self):
        f=self.actual;p=f['parameters'];z=(self.raw-f['scaler']['mean'])/f['scaler']['sd']
        stats=r.sufficient_statistics(z,dict(pi=p['initial'],transition=p['transition'],means=p['means'],variances=p['variances']))
        self.assertAlmostEqual(stats['log_likelihood'],f['log_likelihood'],8)
        np.testing.assert_allclose(stats['occupancy'],f['occupancy'],atol=1e-9,rtol=1e-10)
        np.testing.assert_allclose(stats['filtered'][-1],f['filtered_posterior'],atol=1e-12,rtol=0)

    def test_continuing_filter_and_memoryless(self):
        f=self.actual;p=f['parameters'];our=dict(available=True,parameters=dict(pi=p['initial'],transition=p['transition'],means=p['means'],variances=p['variances']),mean=f['scaler']['mean'],sd=f['scaler']['sd'],terminal_filtered=f['filtered_posterior'])
        x=[-.02,.003,.04,.002];wanted=r.query_oracle(x,our)
        fp=self.producer.filter_path(x,f,f['filtered_posterior'],True);mp=self.producer.memoryless(x,f)
        np.testing.assert_allclose(fp[:,0],[v['filtered'] for v in wanted],atol=1e-12,rtol=0)
        np.testing.assert_allclose(mp[:,0],[v['memoryless'] for v in wanted],atol=1e-12,rtol=0)


class AnnualClockTests(unittest.TestCase):
    def fixture(self):
        import pandas as pd
        sessions=pd.bdate_range('2010-01-01','2017-12-29')
        price=pd.Series(100*np.exp(np.arange(len(sessions))*.0002),index=sessions)
        def fitted(x):
            p=parameters()
            return dict(available=True,status='available',reason=None,fit_attempted=True,n_observations=len(x),
                scaler=dict(mean=0.,sd=.01),parameters=dict(initial=p['pi'],transition=p['transition'],means=p['means'],variances=p['variances']),filtered_posterior=[.7,.3])
        return sessions,price,fitted

    def test_five_year_bounds_and_preceding_price(self):
        sessions,price,fitted=self.fixture();out=r.annual_oracle(price,sessions,['2016-06-30'],fitted)
        row=out['audit']['years'][0]
        self.assertEqual(row['lower_bound_exclusive'],'2010-12-31');self.assertEqual(row['cutoff'],'2015-12-31')
        self.assertEqual(row['training_start'],'2011-01-03');self.assertEqual(row['preceding_price_date'],'2010-12-31')
        self.assertEqual(row['training_observations'],sum('2010-12-31'<str(d.date())<='2015-12-31' for d in sessions))

    def test_missing_preceding_price_prevents_fit(self):
        sessions,price,fitted=self.fixture();price.loc['2010-12-31']=np.nan
        out=r.annual_oracle(price,sessions,['2016-06-30'],lambda x:self.fail('must not fit'))
        self.assertEqual(out['audit']['years'][0]['reason'],'incomplete_training_prices')
        self.assertTrue(np.isnan(out['filtered'][0]))

    def test_late_missing_does_not_contaminate_earlier_evidence(self):
        sessions,price,fitted=self.fixture();price.loc['2016-09-01']=np.nan
        full=r.annual_oracle(price,sessions,['2016-06-30','2016-10-31'],fitted)
        pre=r.annual_oracle(price.loc[:'2016-06-30'],sessions,['2016-06-30'],fitted)
        self.assertEqual(full['audit']['signals'][0],pre['audit']['signals'][0])
        self.assertEqual(full['audit']['signals'][1]['first_invalid_query_date'],'2016-09-01')
        self.assertTrue(np.isnan(full['memoryless'][1]));self.assertTrue(np.isnan(full['filtered'][1]))

    def test_new_year_fits_new_window(self):
        sessions,price,fitted=self.fixture();out=r.annual_oracle(price,sessions,['2016-12-30','2017-01-31'],fitted)
        self.assertEqual([x['cutoff'] for x in out['audit']['years']],['2015-12-31','2016-12-31'])
        self.assertEqual(out['audit']['new_model_fits'],2)

    def test_calendar_shortage_does_not_silently_shorten_fit(self):
        sessions,price,fitted=self.fixture();sessions=sessions[sessions>='2011-01-03']
        out=r.annual_oracle(price,sessions,['2016-06-30'],lambda x:self.fail('must not fit'))
        self.assertEqual(out['audit']['years'][0]['reason'],'incomplete_training_calendar')

    def test_future_value_never_coerced(self):
        sessions,price,fitted=self.fixture();price=price.astype(object);price.loc['2017-01-02']=object()
        x=r.annual_oracle(price,sessions,['2016-06-30'],fitted)
        self.assertTrue(np.isfinite(x['filtered'][0]))

    def test_producer_year_and_signal_parity_without_real_fit(self):
        import monthly_latent_state134_model as producer
        from unittest.mock import patch
        sessions,price,fitted=self.fixture();price.loc['2016-09-01']=np.nan
        sig=['2016-06-30','2016-10-31','2017-03-31']
        expected=r.annual_oracle(price,sessions,sig,fitted)
        with patch.object(producer,'fit',fitted):actual=producer.annual_probabilities(price,sessions,sig)
        r.compare(actual['audit'],expected['audit'],'annual_fixture')
        np.testing.assert_allclose(actual['filtered'],expected['filtered'],atol=1e-12,rtol=0,equal_nan=True)
        np.testing.assert_allclose(actual['memoryless'],expected['memoryless'],atol=1e-12,rtol=0,equal_nan=True)


class CoreFormulaTests(unittest.TestCase):
    def test_rank_ties_and_original_missing_neutrality(self):
        np.testing.assert_array_equal(r.centered_rank([1.,1.,3.,float('nan')]),[0.,0.,.5,0.])

    def test_core_endpoints_and_unchanged_weights(self):
        import pandas as pd
        dates=pd.bdate_range('2020-01-01',periods=330)
        x=np.arange(len(dates));p=pd.DataFrame({name:100*np.exp(.001*k*x+.0001*np.sin(x*(k+1))) for k,name in enumerate(['SPY','A','B','C'],1)},index=dates)
        signal=str(dates[-1].date());meta=dict(dates=[signal],symbols=[['A','B','C','SPY']])
        raw,risk,cw,eligible=r.reconstruct_core({'adj':p},meta)
        t=len(dates)-1;vals=[p[s].iloc[t-42]/p[s].iloc[t-252]-1 for s in meta['symbols'][0]]
        np.testing.assert_array_equal(raw[0],r.centered_rank(vals))
        self.assertEqual(cw[0],.5);self.assertTrue(eligible.all())
        poisoned=p.copy();poisoned.iloc[-41:]=poisoned.iloc[-41:]*1.03
        rr,_,_,_=r.reconstruct_core({'adj':poisoned},meta)
        np.testing.assert_array_equal(raw,rr)

    def test_weighted_rank_roundtrip_preserves_cached_core(self):
        rank=np.array([(j+1)/49-.5 for j in range(49)])
        np.testing.assert_array_equal(.52*(.52*rank/.52),.52*rank)




class FinalBoundaryTests(unittest.TestCase):
    def test_likelihood_tolerance_has_no_relative_extension(self):
        with self.assertRaises(ValueError):r.compare(10000.000001,10000.,'log_likelihood')

    def test_one_original_episode_call_and_complete_score_parity(self):
        import importlib.util
        path=r.ROOT/'outputs/monthly_core_timing70_research/core_adapter.py'
        spec=importlib.util.spec_from_file_location('latent134_test_old_core',path)
        adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
        a,m=fixture();raw=np.array([[.4,.1,.3,-.2,-.3],[.1,.4,-.1,.3,-.2]])
        risk=raw[:,::-1].copy();q=np.array([.3,.8]);rank=q[:,None]*raw+(1-q[:,None])*risk
        h=r.original_helper();x=h.original_prefix(a,m);calls=[]
        def selector(pre,eligible,names,effects,original,policy):
            calls.append(1)
            state=dict(x,eligible=eligible,names=names,effects=effects,baseline=original['baseline'],core=original['core'],cw=original['core_weight'])
            state['support']=(state['baseline']-state['core'])/(1-state['cw'])
            chosen,scores,_=h.episode(pre,state,policy=policy)
            return chosen,scores
        chosen,score=adapter.score_and_choose('incumbent66',1.,rank,a,m,2,pre_score=x['local'],effects=a['effects'],episode_policy=h.POLICY,boundary_selector=selector)
        expected,c,_=r.score_oracle(a,m,raw,risk,q,q)
        self.assertEqual(calls,[1]);self.assertEqual(chosen,c[r.NAMES[2]])
        np.testing.assert_array_equal(score,expected[r.NAMES[2]])

if __name__=='__main__':unittest.main()
