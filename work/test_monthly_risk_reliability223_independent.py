"""Synthetic-only independent annual risk-reliability and funded-slot tests."""
import copy
import inspect
import math
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import monthly_risk_reliability223_independent as independent


def fixture():
    dates=pd.bdate_range('2010-08-02','2018-02-02')
    x=np.arange(len(dates),dtype=float)
    a=np.exp(.00012*x+.05*np.sin(x/17.)+.03*np.cos(x/71.))
    b=np.exp(-.0001*x+.08*np.sin(x/29.)+.02*np.cos(x/7.))
    frame=pd.DataFrame({'A':a,'B':b,'FUND':a.copy(),'FLAT':np.ones(len(dates))},index=dates)
    days=['2016-08-17','2016-08-31','2016-12-30','2017-01-31']
    names=[['A','B','FUND','FLAT'],['B','A','FUND','FLAT'],['A','FUND','B','FLAT'],['A','B','FUND','FLAT']]
    return {'adj':frame},dict(dates=days,symbols=names,etfs=['FUND']),dates.strftime('%Y-%m-%d').tolist()


def physical_prefix(args,n):
    panel,meta,calendar=args
    meta=copy.deepcopy(meta)
    for key in ('dates','ends','symbols'):
        if key in meta:meta[key]=meta[key][:n]
    through=pd.Timestamp.min if not n else pd.Timestamp(meta['dates'][-1])
    return {'adj':panel['adj'].loc[:through].copy()},meta,calendar


def features():
    raw=np.array([[-.2,-.4,-.2,np.nan],[-.1,-.6,-.8,np.nan],[-.2,-.4,-.8,np.nan]])
    mask=np.isfinite(raw)
    other=raw[:,::-1].copy();other[:,0]=raw[:,0];other[:,-1]=np.nan
    result=dict(available=mask,raw_candidate=raw,raw_control=other)
    for key in ('candidate','control'):
        result[key]=independent.original_pct_ranks(result['raw_'+key],mask)
    return result


def original(rows=3,width=4,shape='column'):
    calm=np.array([i%2==0 for i in range(rows)],bool)
    cw=np.where(calm,.5,.52)
    core=np.full((rows,width),.04)
    support=np.array([[.2,-.2,.1,-.1]]*rows)[:,:width]
    baseline=core+(1-cw[:,None])*support
    if shape=='column':cw=cw[:,None]
    elif shape=='matrix':cw=np.broadcast_to(cw[:,None],(rows,width)).copy()
    return dict(core=core,baseline=baseline,core_weight=cw,state_calm10=calm,
                eligible=np.zeros((rows,width),bool))


def targets(values,full=None):
    n,width=values['available'].shape
    result={}
    for arm,key in independent.RAW.items():
        x=np.full((n if full is None else full,width),np.nan);x[:n]=values[key]
        if n:x[0]=np.nan
        result[arm]=x
    return result


class ArithmeticTests(unittest.TestCase):
    def test_handworked_noncentered_pair_and_equal_month_means(self):
        f=np.ones(64);f[1]=2.
        p=[1.,2.,1.];r=[1.,2.,1.,2.]
        out=independent.measure_oracle(f,p,r)
        fm=math.fsum([1.,.25])/63.
        pm=1.25/2.;rm=2.25/3.
        self.assertTrue(out['available'])
        self.assertEqual(out['forecast_second_moment'],fm)
        self.assertEqual(out['prior_second_moment'],pm)
        self.assertEqual(out['realized_second_moment'],rm)
        self.assertEqual(out['candidate_error'],((rm-fm)/(rm+fm))*((rm-fm)/(rm+fm)))
        self.assertEqual(out['control_error'],((rm-pm)/(rm+pm))*((rm-pm)/(rm+pm)))
        self.assertNotEqual(pm,np.var([1.,-.5]))

    def test_common_zero_denominator_excludes_both(self):
        f=np.ones(64);f[-1]=2.
        a=independent.measure_oracle(f,[1.,1.],[1.,1.])
        self.assertTrue(a['source_available']);self.assertFalse(a['available'])
        self.assertEqual(a['reason'],'zero_pair_denominator')
        self.assertIsNone(a['candidate_error']);self.assertIsNone(a['control_error'])
        b=independent.measure_oracle(np.ones(64),[1.,2.],[1.,2.])
        self.assertTrue(b['available']);self.assertEqual(b['candidate_error'],1.)

    def test_missing_and_invalid_price_priority(self):
        f=np.ones(64,dtype=object);f[0]=True
        a=independent.measure_oracle(f,[1.,np.nan],[1.,-2.])
        self.assertEqual(a['reason'],'nonnumeric_forecast_price')
        f[0]=float('nan');a=independent.measure_oracle(f,[1.,np.nan],[1.,-2.])
        self.assertEqual(a['reason'],'invalid_forecast_price')
        a=independent.measure_oracle(np.ones(64),['x',1.],[1.,-2.])
        self.assertEqual(a['reason'],'nonnumeric_prior_price')
        self.assertTrue(all(a[key] is None for key in independent.MEASURES))

    def test_shapes_checked_before_values(self):
        with self.assertRaises(ValueError):independent.measure_oracle(np.full(64,np.nan),[1.],[1.,2.])
        with self.assertRaises(ValueError):independent.measure_oracle(np.ones(63),[1.,2.],[1.,2.])
        with self.assertRaises(ValueError):independent.measure_oracle(np.ones(64),[[1.,2.]],[1.,2.])

    def test_overflow_and_nonzero_underflow_hard_error(self):
        f=np.ones(64);f[1]=1e308
        with self.assertRaises(FloatingPointError):independent.measure_oracle(f,[1.,2.],[1.,2.])
        with self.assertRaises(FloatingPointError):independent.square(1e-200)
        with self.assertRaises(FloatingPointError):independent.positive_divide(np.nextafter(0.,1.),60.)
        with self.assertRaises(FloatingPointError):independent.error_score(1e308,1e308)
        self.assertEqual(independent.square(0.),0.)

    def test_zero_error_signed_mean_and_rank_ties(self):
        self.assertEqual(independent.error_score(.1,.1),0.)
        x=np.array([[-0.,-.25,-.25,np.nan]])
        a=independent.original_pct_ranks(x,np.isfinite(x))
        np.testing.assert_array_equal(a,np.array([[.5,0.,0.,np.nan]]))
        b=independent.original_pct_ranks(np.array([[-0.]]),np.ones((1,1),bool))
        self.assertEqual(b[0,0],.5)


class CalendarAndGroupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.args=fixture();cls.values,cls.audit=independent.independent_features(*cls.args)

    def test_sixty_months_prior_cutoff_and_unique_year_id_reuse(self):
        a=self.audit
        self.assertEqual(a['year_security_groups'],8)
        self.assertEqual(a['historical_month_pairs'],480)
        self.assertEqual(a['supported_groups'],6);self.assertEqual(a['cold_start_groups'],2)
        self.assertEqual(a['new_model_fits'],12);self.assertEqual(a['predictive_model_fits'],12)
        self.assertEqual(a['new_ridge_fits'],0);self.assertEqual(a['new_tree_fits'],0)
        self.assertEqual([g['group_id'] for g in a['group_history']],sorted(g['group_id'] for g in a['group_history']))
        for group in a['group_history']:
            self.assertEqual(len(group['pairs']),60)
            self.assertEqual(group['history_start_month'],str(group['year']-5)+'-01')
            self.assertEqual(group['history_end_month'],str(group['year']-1)+'-12')
            self.assertTrue(all(p['realized_price_end']<=group['cutoff'] for p in group['pairs']))
        self.assertEqual(a['event_history'][0]['cells'][0]['raw_candidate'],a['event_history'][1]['cells'][1]['raw_candidate'])

    def test_pair_calendar_anchors_and_63_daily_forecast(self):
        sessions=[pd.Timestamp(s) for s in self.args[2]]
        f,p,r=independent.calendar_windows(sessions,'2012-02')
        self.assertEqual(len(f),64);self.assertEqual(p[-1],r[0]);self.assertEqual(f[-1],p[-1])
        self.assertEqual(str(r[0].date()),'2012-01-31');self.assertEqual(str(p[0].date()),'2011-12-30')
        self.assertEqual(len(r)-1,len(pd.bdate_range('2012-02-01','2012-02-29')))

    def test_incomplete_official_calendar_hard_error_even_missing_symbol(self):
        panel,meta,cal=fixture();meta['symbols']=[['UNKNOWN']]*len(meta['dates'])
        bad=[x for x in cal if not x.startswith('2011-02')]
        with self.assertRaises(ValueError):independent.independent_features(panel,meta,bad,limit=1)
        with self.assertRaises(ValueError):independent.calendar_windows([pd.Timestamp(s) for s in cal if s>='2011-01-01'],'2011-01')

    def test_common_valid_pair_min18_exact_without_filling_old_months(self):
        p,m,c=fixture();m['dates']=m['dates'][:1];m['symbols']=[['A']]
        good=dict(source_available=True,available=True,reason='available',forecast_second_moment=.1,
            prior_second_moment=.2,realized_second_moment=.3,candidate_error=.25,control_error=.04)
        bad=independent.empty('invalid_forecast_price')
        for n in (17,18):
            with patch.object(independent,'measure_oracle',side_effect=[good.copy() for _ in range(n)]+[bad.copy() for _ in range(60-n)]):
                values,audit=independent.independent_features(p,m,c)
            group=audit['group_history'][0]
            self.assertEqual(group['available_pairs'],n);self.assertEqual(len(group['pairs']),60)
            self.assertEqual(bool(values['available'][0,0]),n==18)
            self.assertEqual(audit['calibration_estimates'],2 if n==18 else 0)
            if n==18:self.assertEqual(group['raw_candidate'],-.25)

    def test_annual_error_means_equal_month_weight_and_negative_zero(self):
        p,m,c=fixture();m['dates']=m['dates'][:1];m['symbols']=[['A']]
        for zero in (False,True):
            responses=[]
            for k in range(60):
                error=0. if zero else (k%5)/4.
                responses.append(dict(source_available=True,available=True,reason='available',
                    forecast_second_moment=.1,prior_second_moment=.2,realized_second_moment=.3,
                    candidate_error=error,control_error=0.))
            with patch.object(independent,'measure_oracle',side_effect=responses):
                v,a=independent.independent_features(p,m,c)
            expected=-math.fsum(x['candidate_error'] for x in responses)/60.
            self.assertEqual(v['raw_candidate'][0,0],expected)
            self.assertTrue(np.signbit(v['raw_control'][0,0]))
            self.assertEqual(a['calibration_estimates'],2)
            self.assertTrue(v['available'][0,0])
            if zero:self.assertTrue(np.signbit(v['raw_candidate'][0,0]))

    def test_source_missing_priority_and_no_current_price_requirement(self):
        p,m,c=fixture();m['dates']=m['dates'][:1];m['symbols']=[['UNKNOWN']]
        _,audit=independent.independent_features(p,m,c)
        self.assertEqual(audit['pair_reason_counts'],{'missing_symbol':60})
        p,m,c=fixture();p['adj']=p['adj'].loc[:'2015-12-31']
        v,a=independent.independent_features(p,m,c,limit=1)
        np.testing.assert_array_equal(v['available'],self.values['available'][:1])
        self.assertEqual(a['group_history'],self.audit['group_history'][:4])

    def test_counts_consumption_once_per_group_and_zero_pairs_counted(self):
        a=self.audit
        pairs=[p for g in a['group_history'] for p in g['pairs']]
        source=[p for p in pairs if p['source_available']];valid=[p for p in pairs if p['available']]
        self.assertEqual(a['source_available_pairs'],len(source));self.assertEqual(a['available_pairs'],len(valid))
        self.assertEqual(a['valid_price_observations'],sum(64+p['prior_return_count']+1+p['realized_return_count']+1 for p in source))
        self.assertEqual(a['simple_return_observations'],sum(63+p['prior_return_count']+p['realized_return_count'] for p in source))
        self.assertEqual(a['second_moment_estimates'],3*len(source));self.assertEqual(a['error_evaluations'],2*len(valid))
        self.assertEqual(a['ranked_observations'],2*int(self.values['available'].sum()))
        self.assertEqual(a['available_by_event'],self.values['available'].sum(axis=1).tolist())

    def test_common_all_id_domain_not_eligibility_and_exact_column_identity(self):
        p,m,c=fixture();m['eligible']=[[False]*4]*4
        v,a=independent.independent_features(p,m,c)
        for key in v:np.testing.assert_array_equal(v[key],self.values[key])
        self.assertEqual(a,self.audit)
        first=a['event_history'][0]['cells']
        self.assertEqual(first[0]['raw_candidate'],first[2]['raw_candidate'])
        self.assertNotEqual(first[0]['group_id'],first[2]['group_id'])
        self.assertEqual(first[0]['candidate'],first[2]['candidate'])

    def test_current_year_and_future_price_poison_not_read(self):
        p,m,c=fixture();m['dates']=m['dates'][:3];m['symbols']=m['symbols'][:3]
        frame=p['adj'].astype(object);frame.loc[frame.index>='2016-01-01']='not_read'
        v,a=independent.independent_features({'adj':frame},m,c)
        for key in v:np.testing.assert_array_equal(v[key],self.values[key][:3])
        self.assertEqual(a['group_history'],self.audit['group_history'][:4])

    def test_exact_full_audit_validation_and_scope(self):
        check=independent.verify_features(*self.args,self.values,self.audit)
        self.assertTrue(all(check[k] for k in independent.FEATURE_FLAGS))
        self.assertFalse(check['computes_returns']);self.assertFalse(check['producer_modules_imported'])
        self.assertEqual(check['numerical_tolerance'],0.)
        for key in ('available_by_event','new_model_fits','predictive_model_fits',*independent.COUNT_FIELDS):
            self.assertEqual(check[key],self.audit[key])
        a=copy.deepcopy(self.audit);a['group_history'][0]['pairs'][0]['forecast_date']='2011-01-03'
        with self.assertRaises(ValueError):independent.verify_features(*self.args,self.values,a)
        v={k:x.copy() for k,x in self.values.items()};v['raw_candidate'][0,0]+=1e-12
        with self.assertRaises(ValueError):independent.verify_features(*self.args,v,self.audit)

    def test_metadata_validation_and_physical_empty_prefix(self):
        p,m,c=fixture();m['dates'][1]='2016-08-30'
        with self.assertRaises(ValueError):independent.independent_features(p,m,c)
        p,m,c=physical_prefix(self.args,0);v,a=independent.independent_features(p,m,c)
        self.assertEqual(v['available'].shape,(0,0));self.assertEqual(a['new_model_fits'],0)
        self.assertEqual(a['group_history'],[]);self.assertEqual(a['available_by_event'],[])
        p,m,c=fixture();m['symbols'][0][1]='A'
        with self.assertRaises(ValueError):independent.independent_features(p,m,c)


class DeltaTests(unittest.TestCase):
    def test_support_funded_formula_all_states_and_no_eligible_multiplier(self):
        v=features();a=original();t=targets(v);d=independent.independent_deltas(v,a,t)
        support=(a['baseline']-a['core'])/(1-a['core_weight'])
        for arm in d:
            good=np.isfinite(t[arm]);expected=np.zeros(good.shape)
            expected[good]=.01*(t[arm][good]-support[good])
            np.testing.assert_array_equal(d[arm],expected)
            self.assertTrue(np.any(d[arm][1]!=0));self.assertTrue(np.any(d[arm][2]!=0))
        receipt=independent.verify_deltas(v,a,t,d)
        self.assertTrue(all(receipt[k] for k in independent.DELTA_FLAGS))
        self.assertEqual(receipt['budget'],.01);self.assertEqual(receipt['max_absolute_delta'],.01000000000001)

    def test_initial_missing_and_zero_rank_are_distinct(self):
        v=features();a=original();t=targets(v);d=independent.independent_deltas(v,a,t)
        for arm,key in independent.RAW.items():
            self.assertTrue(np.all(d[arm][0]==0.));self.assertTrue(np.all(d[arm][:,-1]==0.))
        t[independent.ARMS[1]][0,0]=v['candidate'][0,0]
        with self.assertRaises(ValueError):independent.independent_deltas(v,a,t)
        raw=np.array([[-.1,-.2,-.3],[-.1,-.2,-.2]])
        mask=np.ones(raw.shape,bool);r=independent.original_pct_ranks(raw,mask)
        vv=dict(available=mask,raw_candidate=raw,raw_control=raw.copy(),candidate=r,control=r.copy())
        aa=original(2,3);dd=independent.independent_deltas(vv,aa,targets(vv))
        self.assertEqual(r[1,1],0.);self.assertNotEqual(dd[independent.ARMS[1]][1,1],0.)

    def test_fixed_budget_support_bounds_and_bad_weights(self):
        v=features();a=original();t=targets(v)
        for bad in (.005,.02,True,float('nan')):
            with self.assertRaises(ValueError):independent.independent_deltas(v,a,t,bad)
        a['baseline'][1,0]=1.
        with self.assertRaises(ValueError):independent.independent_deltas(v,a,t)
        a=original();a['core_weight'][1]=.51
        with self.assertRaises(ValueError):independent.independent_deltas(v,a,t)

    def test_weight_shapes_suffix_poison_and_no_second_ranking(self):
        v=features();reference=None
        for shape in ('vector','column','matrix'):
            a=original(shape=shape);d=independent.independent_deltas(v,a,targets(v))
            if reference is None:reference=d
            else:
                for arm in d:np.testing.assert_array_equal(d[arm],reference[arm])
        part={key:value[:2].copy() for key,value in v.items()};a=original()
        a['baseline']=a['baseline'].astype(object);a['core']=a['core'].astype(object);a['core_weight']=a['core_weight'].astype(object)
        a['baseline'][2]='POISON';a['core'][2]='POISON';a['core_weight'][2]='POISON'
        d=independent.independent_deltas(part,a,targets(part,3))
        for arm in d:np.testing.assert_array_equal(d[arm],reference[arm][:2])
        a=original();t=targets(v);t[independent.ARMS[1]][1,:3]=t[independent.ARMS[1]][1,:3][::-1]
        with self.assertRaises(ValueError):independent.independent_deltas(v,a,t)


class ProducerCrossTests(unittest.TestCase):
    def test_unique_two_fixtures_all_logical_physical_prefixes_and_full_audit(self):
        import monthly_risk_reliability223_kernel as producer
        from test_monthly_risk_reliability223_kernel import fixture as producer_fixture
        for args in (fixture(),producer_fixture()):
            full_values,full_audit=producer.build_features(*args)
            independent.verify_features(*args,full_values,full_audit)
            for n in range(len(args[1]['dates'])+1):
                logical,audit=producer.build_features(*args,limit=n)
                independent.verify_features(*args,logical,audit,limit=n)
                pa=physical_prefix(args,n);pv,pr=producer.build_features(*pa)
                independent.verify_features(*pa,pv,pr)
                for key in pv:
                    if n:np.testing.assert_array_equal(pv[key],full_values[key][:n])
                    else:self.assertEqual(pv[key].shape,(0,0))
                self.assertEqual(pr['event_history'],full_audit['event_history'][:n])
                group_map={g['group_id']:g for g in full_audit['group_history']}
                for g in pr['group_history']:self.assertEqual(g,group_map[g['group_id']])
        source=inspect.getsource(independent)
        self.assertNotIn('import monthly_risk_reliability223_kernel',source)


if __name__=='__main__':unittest.main()
