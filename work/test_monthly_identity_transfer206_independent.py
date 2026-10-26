"""Independent round206 synthetic audit tests; never loads research input arrays."""
import copy
import math
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import monthly_identity_transfer206_independent as oracle


def fixture():
    months=pd.date_range('2014-01-31','2022-12-31',freq='ME')
    dates=np.repeat(months.values.astype('datetime64[D]'),3)
    ends=np.repeat((months+pd.offsets.MonthEnd()).values.astype('datetime64[D]'),3)
    names=np.tile(['ONE','TWO','FUND'],len(months))
    rng=np.random.default_rng(91206)
    X=rng.uniform(-.5,.5,(len(dates),10))
    y=.23*X[:,1]+.14*X[:,4]**2+np.tile([.027,-.043,.016],len(months))
    y=(y.reshape(-1,3)-y.reshape(-1,3).mean(axis=1,keepdims=True)).reshape(-1)
    targets={key:y*(1.+.3*i) for i,key in enumerate(oracle.TARGETS)}
    event_dates=np.repeat(np.array(['2015-02-27','2018-01-31','2021-03-31'],dtype='datetime64[D]'),3)
    return [dates,ends,names,X,targets,np.ones(len(dates),bool),event_dates,
        np.tile(['ONE','FUND','NEW'],3),rng.uniform(-.5,.5,(9,10)),np.ones(9,bool)]


def prefix(args,n,physical=True):
    a=copy.deepcopy(args)
    for i in (6,7,8,9):a[i]=a[i][:n]
    if physical:
        if n:
            year=int(a[6].max().astype('datetime64[Y]').astype(int))+1970
            keep=a[1]<=np.datetime64('%04d-12-31'%(year-1))
        else:keep=np.zeros(len(a[0]),bool)
        for i in (0,1,2,3,5):a[i]=a[i][keep]
        a[4]={k:y[keep] for k,y in a[4].items()}
    return a


def packed(args,built):
    dates,ends,symbols,X,targets,good,queries,qnames,qX,qgood=args
    def grouped(days,names,ends=None):
        unique=list(dict.fromkeys(str(d) for d in days))
        item=dict(dates=unique,symbols=[[str(s) for s in names[days==np.datetime64(d)]] for d in unique])
        if ends is not None:item['ends']=[str(ends[np.flatnonzero(days==np.datetime64(d))[0]]) for d in unique]
        return item
    training=grouped(dates,symbols,ends)
    # One synthetic query per metadata row also permits every flattened prefix.
    monthly=dict(dates=[str(d) for d in queries],symbols=[[str(s)] for s in qnames])
    shape=(len(queries),1)
    original={'eligible':qgood.reshape(shape).copy(),
        'model_available':built['available'].reshape(shape).copy(),
        'local_available':built['local_available'].reshape(shape).copy()}
    for key in ('source_log','local_constant','local_linear'):
        original['raw_'+key]=built['raw'][key].reshape(shape).copy()
    train=dict(dates=dates,ends=ends,train_plain=X,common_good=good,
               event_dates=queries,event_plain=qX)
    train.update({'target_'+key:y for key,y in targets.items()})
    return train,dict(monthly=monthly,training=training),original


def compare(args,built):
    expected=oracle.reconstruct(*args)
    for key in oracle.RAW_KEYS:oracle.close(built['raw'][key],expected['raw'][key],key)
    for key in ('available','local_available'):oracle.same(built[key],expected[key],key)
    oracle.record_check(built['audit'],expected['audit'],'audit')
    train,meta,original=packed(args,built)
    return oracle.check_prepare(train,meta,original,built)


def rank_fixture():
    source=np.array([[0.,.1,.2,.3], [.2,.1,0.,-.1], [0.,.2,.1,.3], [0.,.1,.2,.3]])
    linear=source[:,::-1].copy()
    shape=source.shape
    a=dict(baseline=np.zeros(shape),eligible=np.ones(shape,bool),forecast_available=np.ones(shape,bool),
        model_available=np.ones(shape,bool),local_available=np.zeros(shape,bool),
        raw_source_log=source.copy(),raw_local_constant=source+.02,
        raw_local_linear=linear,rank_log_excess=np.vstack([oracle.centered_rank(x) for x in source]),
        state_calm10=np.array([True,True,False,True]),
        state_weak_narrow=np.array([False,False,True,False]),
        extension_available=np.ones(shape,bool))
    raw={key:source.ravel().copy() for key in oracle.RAW_KEYS}
    raw['candidate']=np.array([[.1,.1,.2,.2],[.3,.1,.1,.2],[.1,.2,.1,.2],[.2,.3,.1,0.]]).ravel()
    raw['control']=np.array([[0.,.3,.2,.1],[.2,.3,.1,0.],[.3,.2,.1,0.],[0.,.2,.1,.3]]).ravel()
    f=dict(raw=raw,available=np.ones(source.size,bool),local_available=np.zeros(source.size,bool))
    return a,f


class ScalarTests(unittest.TestCase):
    def test_handworked_matched_month_peer_average_excludes_self_and_extra_month(self):
        dates=np.array(['2020-01-31']*3+['2020-02-28']*2+['2020-03-31'],dtype='datetime64[D]')
        symbols=np.array(['A','B','C','A','B','B']);X=np.zeros((6,10))
        residual=np.array([2.,10.,30.,4.,8.,1e6]);weights=np.array([.2,.8])
        number,audit=oracle.matched_peer_months(dates,symbols,X,residual,[0,3],np.zeros(10),weights,np.arange(6))
        self.assertAlmostEqual(number,10.4,places=14)
        self.assertEqual([a['peer_mean'] for a in audit],[20.,8.])
        self.assertEqual([a['peer_row_ids'] for a in audit],[[1,2],[4]])
        self.assertEqual([a['own_weight'] for a in audit],[.2,.8])
        self.assertNotEqual(number,math.fsum(weights*residual[[0,3]]))

    def test_gaussian_original_rank_units_and_no_peer_count_mass(self):
        days=np.array(['2020-01-31']*3,dtype='datetime64[D]');X=np.zeros((3,10));X[2]=.5
        residual=np.array([0.,1.,-1.]);q=np.zeros(10)
        num,audit=oracle.matched_peer_months(days,['A','B','C'],X,residual,[0],q,[.3],range(3))
        expected=(1-math.exp(-.5))/(1+math.exp(-.5))
        self.assertAlmostEqual(num,.3*expected,places=15)
        X=np.vstack([X,X[1:]]);r=np.r_[residual,residual[1:]]
        expanded=np.repeat(days[0],5)
        n,b=oracle.matched_peer_months(expanded,['A','B','C','D','E'],X,r,[0],q,[.3],range(5))
        self.assertAlmostEqual(num,n,places=15)
        self.assertAlmostEqual(b[0]['peer_weight_mass'],2*audit[0]['peer_weight_mass'],places=15)

    def test_missing_same_month_peer_is_hard_error(self):
        with self.assertRaisesRegex(ValueError,'Missing same-month'):
            oracle.matched_peer_months(np.array(['2020-01-31']),['A'],np.zeros((1,10)),np.ones(1),[0],np.zeros(10),[1.],[0])

    def test_augmented_penalized_intercept_and_unnormalized_mass(self):
        X=np.zeros((3,10));w=np.array([.1,.2,.3]);y=np.array([1.,2.,3.])
        c,l,_=oracle.local_fit(X,np.zeros(10),np.ones(10),y,w)
        self.assertAlmostEqual(c,1.4/12.6,places=15);self.assertAlmostEqual(c,l,places=15)
        scaled=oracle.local_fit(X,np.zeros(10),np.ones(10),y,w*10)[0]
        self.assertNotEqual(c,scaled)

    def test_original_percentile_ties_singleton(self):
        np.testing.assert_array_equal(oracle.centered_rank([3,1,3,2]),[.375,-.25,.375,0.])
        np.testing.assert_array_equal(oracle.centered_rank([1]),[.5])
        np.testing.assert_array_equal(oracle.centered_rank([1,1]),[.25,.25])


class AnnualTests(unittest.TestCase):
    def test_annual_window_cohort_and_counts(self):
        z=oracle.reconstruct(*fixture())
        self.assertEqual(z['audit']['annual'][0]['status'],'cold_start')
        for annual in z['audit']['annual']:
            self.assertLessEqual(annual['latest_label_end'],annual['cutoff'])
            self.assertGreater(annual['first_training_signal'],annual['lookback_exclusive_start'])
        fits=sum(row['prediction_rows']>0 for row in z['audit']['annual'])
        c=z['audit']['counts']
        self.assertEqual(c['source_target_fits'],4*fits)
        self.assertEqual(c['actual_regression_solves'],6*fits+c['reference_local_solves'])
        self.assertEqual(c['peer_month_evaluations'],sum(len(x['peer_months']) for x in z['audit']['local']))
        self.assertEqual(c['peer_row_observations'],sum(x['peer_count'] for row in z['audit']['local'] for x in row['peer_months']))
        self.assertEqual(set(z['audit']['arms']),set(oracle.ARMS[1:]))

    def test_own18_threshold_no_own_exact_pooled_and_same_support(self):
        a=fixture();a[6:]=[v[3:4] for v in a[6:]]
        old=np.flatnonzero((a[2]=='ONE')&(a[1]<=np.datetime64('2017-12-31')))
        a[5][old[:-18]]=False
        z=oracle.reconstruct(*a);self.assertTrue(z['local_available'][0])
        self.assertEqual(z['audit']['local'][0]['own_months'],18)
        a[5][old[-18]]=False;z=oracle.reconstruct(*a)
        self.assertTrue(z['available'][0]);self.assertFalse(z['local_available'][0])
        self.assertEqual(z['audit']['local'][0]['own_months'],17)
        for key in ('candidate','control'):self.assertEqual(z['raw'][key][0],z['raw']['source_log'][0])

    def test_four_target_nonfinite_and_singleton_month_excluded(self):
        a=fixture();a[4]['loss_utility'][0:2]=np.nan;z=oracle.reconstruct(*a)
        for annual in z['audit']['annual']:
            self.assertTrue(set([0,1,2]).isdisjoint(annual['training_row_ids']))

    def test_future_object_poison_ignored_and_empty_prefix(self):
        a=prefix(fixture(),4,False);expected=oracle.reconstruct(*a)
        future=a[1]>np.datetime64('2017-12-31')
        for i in (3,5):a[i]=a[i].astype(object);a[i][future]='unconsumed poison'
        for key in a[4]:a[4][key]=a[4][key].astype(object);a[4][key][future]='unconsumed poison'
        actual=oracle.reconstruct(*a)
        oracle.record_check(actual['audit'],expected['audit'],'future')
        for key in oracle.RAW_KEYS:np.testing.assert_array_equal(actual['raw'][key],expected['raw'][key])
        empty=oracle.reconstruct(*prefix(a,0,False))
        self.assertEqual(empty['audit']['annual'],[]);self.assertTrue(all(v==0 for v in empty['audit']['counts'].values()))

    def test_zero_residual_still_evaluates_peer_months(self):
        a=fixture()
        for y in a[4].values():y[:]=0.
        z=oracle.reconstruct(*a);c=z['audit']['counts']
        self.assertEqual(c['reference_local_solves'],0);self.assertGreater(c['peer_query_evaluations'],0)
        for key in oracle.RAW_KEYS:np.testing.assert_array_equal(z['raw'][key][z['available']],0.)

    def test_invalid_shapes_masks_identities_targets_and_rank_bounds(self):
        for change in ('shape','bool','duplicate','target','bounds','end'):
            a=fixture()
            if change=='shape':a[3]=a[3][:,:9]
            if change=='bool':a[5]=a[5].astype(int)
            if change=='duplicate':a[2][1]=a[2][0]
            if change=='target':a[4].pop('vol_scaled')
            if change=='bounds':a[3][0,0]=.6
            if change=='end':a[1][0]=a[0][0]
            with self.subTest(change=change),self.assertRaises(ValueError):oracle.reconstruct(*a)

    def test_both_producer_fixtures_full_audit_all_physical_prefixes(self):
        import monthly_identity_transfer206_model as model
        import test_monthly_identity_transfer206_model as production_tests
        for make in (fixture,production_tests.fixture):
            a=make();before=copy.deepcopy(a);full=model.annual_forecasts(*a)
            receipt=compare(a,full)
            self.assertTrue(all(receipt[k] is True for k in oracle.PREPARE_FLAGS))
            for n in range(len(a[6])+1):
                p=prefix(a,n);z=model.annual_forecasts(*p)
                compare(p,z)
                for key in oracle.RAW_KEYS:np.testing.assert_array_equal(z['raw'][key],full['raw'][key][:n])
            for i,v in enumerate(a):
                if isinstance(v,dict):
                    for k in v:np.testing.assert_array_equal(v[k],before[i][k])
                else:np.testing.assert_array_equal(v,before[i])

    def test_corrupt_raw_mask_peers_weights_cutoff_and_denominator_rejected(self):
        import monthly_identity_transfer206_model as model
        a=fixture();built=model.annual_forecasts(*a);train,meta,arrays=packed(a,built)
        localidx=next(i for i,row in enumerate(built['audit']['local']) if row['status']=='fit')
        for change in ('raw','mask','peer_id','peer_mean','own_weight','denominator','cutoff','counter'):
            z=copy.deepcopy(built);row=z['audit']['local'][localidx]
            if change=='raw':z['raw']['candidate'][-1]+=.001
            if change=='mask':z['available'][-1]=False
            if change=='peer_id':row['peer_months'][0]['peer_row_ids'][0]=row['own_row_ids'][0]
            if change=='peer_mean':row['peer_months'][0]['peer_mean']+=.001
            if change=='own_weight':row['own_weights'][0]*=2
            if change=='denominator':row['correction_denominator']+=1
            if change=='cutoff':row['cutoff']='2022-12-31'
            if change=='counter':z['audit']['counts']['peer_month_evaluations']-=1
            with self.subTest(change=change),self.assertRaises(ValueError):oracle.check_prepare(train,meta,arrays,z)


class RankTests(unittest.TestCase):
    def test_actual_frozen_rank_host_pair_integration_all_prefixes(self):
        import monthly_identity_transfer206_host as host
        study=host.local()
        a,f=rank_fixture();f['raw']['control'][5]=np.nan
        for n in range(5):
            ranks,deltas=host.rank_and_delta(study,a,f,n)
            small={k:v[:n].copy() for k,v in a.items()}
            partial={k:({s:t[:n*4].copy() for s,t in v.items()} if k=='raw' else v[:n*4].copy()) for k,v in f.items()}
            receipt=oracle.verify_rank_slot(small,partial,ranks,deltas)
            self.assertEqual(tuple(host.SLOT_FLAGS),oracle.RANK_FLAGS)
            self.assertTrue(all(receipt[k] for k in host.SLOT_FLAGS))

    def test_pairwise_whole_row_readiness_uses_old_linear_not_pooled(self):
        a,f=rank_fixture();f['raw']['control'][5]=np.nan
        ranks,deltas,ready=oracle.rank_oracle(a,f)
        for arm in oracle.NEW_RAW:
            self.assertFalse(ready[arm][1]);np.testing.assert_array_equal(deltas[arm][1],0.)
            np.testing.assert_array_equal(ranks[arm][1],oracle.centered_rank(a['raw_local_linear'][1]))
        self.assertFalse(np.array_equal(ranks[oracle.ARMS[1]][1],a['rank_log_excess'][1]))

    def test_source_identity_exact_shortcut_full_domain_and_initial(self):
        a,f=rank_fixture();f['raw']['candidate']=a['raw_source_log'].ravel().copy()
        ranks,deltas,ready=oracle.rank_oracle(a,f)
        np.testing.assert_array_equal(ranks[oracle.ARMS[1]],a['rank_log_excess'])
        self.assertTrue(ready[oracle.ARMS[1]].all())
        for arm in oracle.NEW_RAW:np.testing.assert_array_equal(deltas[arm][0],0.)
        self.assertGreater(np.count_nonzero(deltas[oracle.ARMS[1]][1:]),0)
        self.assertFalse(a['local_available'].any()) # no own-history rank filter

    def test_exact_slot_formula_masks_and_corruption(self):
        a,f=rank_fixture();a['eligible'][2,3]=False;a['extension_available'][1,0]=False
        ranks,deltas,ready=oracle.rank_oracle(a,f)
        for arm in oracle.NEW_RAW:
            self.assertEqual(deltas[arm][2,3],0.);self.assertEqual(deltas[arm][1,0],0.)
        receipt=oracle.verify_rank_slot(a,f,ranks,deltas)
        self.assertTrue(all(receipt[k] for k in oracle.RANK_FLAGS))
        wrong=copy.deepcopy(deltas);wrong[oracle.ARMS[1]][2,0]+=1e-15
        with self.assertRaises(ValueError):oracle.verify_rank_slot(a,f,ranks,wrong)

    def test_rank_all_prefixes_and_no_mutation(self):
        a,f=rank_fixture();before=copy.deepcopy(a);full=oracle.rank_oracle(a,f)
        for n in range(len(a['eligible'])+1):
            aa={k:v[:n].copy() for k,v in a.items()}
            ff={k:({s:t[:n*4].copy() for s,t in v.items()} if k=='raw' else v[:n*4].copy()) for k,v in f.items()}
            r,d,ready=oracle.rank_oracle(aa,ff)
            oracle.verify_rank_slot(aa,ff,r,d)
            for arm in oracle.NEW_RAW:
                np.testing.assert_array_equal(r[arm],full[0][arm][:n]);np.testing.assert_array_equal(d[arm],full[1][arm][:n])
        for key in a:np.testing.assert_array_equal(a[key],before[key])

    def test_invalid_rank_shapes_and_bool_masks(self):
        a,f=rank_fixture();a['eligible']=a['eligible'].astype(int)
        with self.assertRaises(ValueError):oracle.rank_oracle(a,f)
        a,f=rank_fixture();f['raw']['candidate'][0]=np.inf
        with self.assertRaises(ValueError):oracle.rank_oracle(a,f)

    def test_only_pinned_accounting_dependency_and_no_producer_import(self):
        source=Path(oracle.__file__).read_text()
        self.assertNotIn('import monthly_identity_transfer206_model',source)
        self.assertEqual(oracle.sha(oracle.ACCOUNTING),oracle.ACCOUNTING_SHA)
        helper=oracle.accounting()
        self.assertTrue(callable(helper.audit_path));self.assertTrue(callable(helper.statistics_by_split))
        self.assertEqual(len(oracle.ARMS)*len(oracle.PATHS),9)


if __name__=='__main__':unittest.main(verbosity=2)
