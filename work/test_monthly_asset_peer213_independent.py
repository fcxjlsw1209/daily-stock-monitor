"""Synthetic-only independent213 tests: no historical caches, prices, or outcomes."""
import copy
import math
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import monthly_asset_peer213_independent as oracle

ETFS=['FUND','SECOND_FUND']


def fixture():
    months=pd.date_range('2014-01-31','2022-12-31',freq='ME')
    dates=np.repeat(months.values.astype('datetime64[D]'),4)
    ends=np.repeat((months+pd.offsets.MonthEnd()).values.astype('datetime64[D]'),4)
    names=np.tile(['ONE','TWO','FUND','SECOND_FUND'],len(months))
    rng=np.random.default_rng(91213);X=rng.uniform(-.5,.5,(len(dates),10))
    y=.23*X[:,1]+.14*X[:,4]**2+np.tile([.027,-.043,.016,.08],len(months))
    y=(y.reshape(-1,4)-y.reshape(-1,4).mean(axis=1,keepdims=True)).reshape(-1)
    targets={key:y*(1.+.3*i) for i,key in enumerate(oracle.TARGETS)}
    queries=np.repeat(np.array(['2015-02-27','2018-01-31','2021-03-31'],dtype='datetime64[D]'),4)
    return [dates,ends,names,X,targets,np.ones(len(dates),bool),queries,
        np.tile(['ONE','FUND','NEW','SECOND_FUND'],3),rng.uniform(-.5,.5,(12,10)),np.ones(12,bool)]


def cached(args):
    # Frozen206 is used solely to manufacture synthetic predecessor-vintage inputs.
    import monthly_identity_transfer206_model as prior
    built=prior.annual_forecasts(*args)
    return dict(**{'raw_'+k:v.copy() for k,v in built['raw'].items()},
                available=built['available'].copy(),local_available=built['local_available'].copy())


def build(args,reference=None,etfs=ETFS):
    return oracle.reconstruct(*args,etfs=etfs,cached_reference=cached(args) if reference is None else reference)


def prefix(args,n,physical=True):
    a=copy.deepcopy(args)
    for i in (6,7,8,9):a[i]=a[i][:n]
    if physical:
        year=int(a[6].max().astype('datetime64[Y]').astype(int))+1970 if n else None
        keep=a[1]<=np.datetime64('%04d-12-31'%(year-1)) if n else np.zeros(len(a[0]),bool)
        for i in (0,1,2,3,5):a[i]=a[i][keep]
        a[4]={k:y[keep] for k,y in a[4].items()}
    return a


def packed(args,reference):
    dates,ends,symbols,X,targets,good,queries,qnames,qX,qgood=args
    unique=list(dict.fromkeys(str(d) for d in dates))
    training=dict(dates=unique,symbols=[[str(s) for s in symbols[dates==np.datetime64(d)]] for d in unique],
        ends=[str(ends[np.flatnonzero(dates==np.datetime64(d))[0]]) for d in unique])
    monthly=dict(dates=[str(d) for d in queries],symbols=[[str(s)] for s in qnames],etfs=ETFS.copy())
    shape=(len(queries),1)
    arrays=dict(eligible=qgood.reshape(shape),model_available=reference['available'].reshape(shape),
        local_available=reference['local_available'].reshape(shape))
    arrays.update({'raw_'+k:reference['raw_'+k].reshape(shape) for k in oracle.RAW_KEYS[:3]})
    train=dict(dates=dates,ends=ends,train_plain=X,common_good=good,event_dates=queries,event_plain=qX)
    train.update({'target_'+k:v for k,v in targets.items()})
    return train,dict(monthly=monthly,training=training),arrays


def checked(args,forecast,reference):
    return oracle.check_prepare(*packed(args,reference),forecast,reference)


class PeerTests(unittest.TestCase):
    def test_handworked_same_type_and_all_peer_means(self):
        dates=np.array(['2020-01-31']*4+['2020-02-28']*4,dtype='datetime64[D]')
        names=np.tile(['A','B','F','G'],2);X=np.zeros((8,10));r=np.array([2.,10.,30.,50.,4.,8.,40.,60.])
        allnum,rows,samenum,missing=oracle.asset_peer_months(dates,names,X,r,[0,4],np.zeros(10),[.2,.8],range(8),['F','G'])
        self.assertEqual(missing,[]);self.assertAlmostEqual(samenum,8.4)
        self.assertAlmostEqual(allnum,.2*30.+.8*36.)
        self.assertEqual([v['same_peer_row_ids'] for v in rows],[[1],[5]])
        self.assertEqual([v['peer_row_ids'] for v in rows],[[1,2,3],[5,6,7]])
        _,fundrows,fundnum,_=oracle.asset_peer_months(dates,names,X,r,[2,6],np.zeros(10),[.2,.8],range(8),['F','G'])
        self.assertEqual(fundnum,58.);self.assertEqual([v['same_peer_row_ids'] for v in fundrows],[[3],[7]])

    def test_any_missing_month_discards_entire_same_type_estimate(self):
        ds=np.array(['2020-01-31']*3+['2020-02-28']*2,dtype='datetime64[D]')
        names=['A','B','F','A','F'];X=np.zeros((5,10));r=np.array([0.,4.,8.,0.,9.])
        number,rows,same,missing=oracle.asset_peer_months(ds,names,X,r,[0,3],np.zeros(10),[.25,.75],range(5),['F'])
        self.assertEqual(missing,['2020-02']);self.assertIsNone(same)
        self.assertEqual(number,.25*6.+.75*9.)
        self.assertEqual([v['same_peer_count'] for v in rows],[1,0])
        for row in rows:
            for field in ('same_peer_weight_mass','same_peer_weighted_residual','same_peer_mean'):self.assertIsNone(row[field])

    def test_same_type_selection_includes_all_same_peers_without_age_or_count_mass(self):
        ds=np.array(['2020-01-31']*4,dtype='datetime64[D]');X=np.zeros((4,10));X[2]=.5
        number,rows,same,_=oracle.asset_peer_months(ds,['A','B','C','F'],X,[0.,1.,-1.,99.],[0],np.zeros(10),[.3],range(4),['F'])
        expected=.3*(1-math.exp(-.5))/(1+math.exp(-.5))
        self.assertAlmostEqual(same,expected,places=14);self.assertEqual(rows[0]['same_peer_count'],2)
        self.assertNotEqual(number,same)

    def test_nonfinite_arithmetic_is_error_not_fallback(self):
        ds=np.array(['2020-01-31']*3,dtype='datetime64[D]')
        with self.assertRaises((ValueError,FloatingPointError)):
            oracle.asset_peer_months(ds,['A','B','F'],np.zeros((3,10)),[0.,np.nan,0.],[0],np.zeros(10),[1.],range(3),['F'])


class AnnualTests(unittest.TestCase):
    def test_counts_cohorts_cache_exact_and_only_pooled_fits(self):
        args=fixture();cache=cached(args);before=copy.deepcopy(cache);z=build(args,cache)
        self.assertEqual(z['audit']['annual'][0]['status'],'cold_start')
        for row in z['audit']['annual']:
            self.assertLessEqual(row['latest_label_end'],row['cutoff'])
            self.assertGreater(row['first_training_signal'],row['lookback_exclusive_start'])
        c=z['audit']['counts'];fits=sum(r['prediction_rows']>0 for r in z['audit']['annual'])
        self.assertEqual(c['actual_regression_solves'],fits);self.assertEqual(c['annual_pooled_log_refits'],fits)
        self.assertEqual(c['own_weight_evaluations'],c['all_peer_query_evaluations'])
        self.assertEqual(c['own_weight_evaluations'],c['same_type_supported_queries']+c['same_type_fallback_queries'])
        self.assertEqual(c['all_peer_month_evaluations'],sum(len(r['peer_months']) for r in z['audit']['local']))
        self.assertEqual(c['same_type_month_evaluations'],sum(len(r['peer_months']) for r in z['audit']['local'] if r['status']=='same_type_supported'))
        for k in oracle.RAW_KEYS[:3]:self.assertEqual(z['raw'][k].tobytes(),cache['raw_'+k].tobytes())
        self.assertEqual(z['raw']['control'].tobytes(),cache['raw_candidate'].tobytes())
        for k in cache:self.assertEqual(cache[k].tobytes(),before[k].tobytes())
        receipt=checked(args,z,cache);self.assertTrue(all(receipt[k] for k in oracle.PREPARE_FLAGS))
        self.assertEqual(receipt['independent_pooled_fits'],fits);self.assertFalse(receipt['computes_returns'])

    def test_query_fallback_uses_all_peer_not_pooled_and_does_not_filter(self):
        args=fixture();cache=cached(args);z=build(args,cache,etfs=['FUND'])
        indices=[r['event_row'] for r in z['audit']['local'] if r['status']=='same_type_missing_month']
        self.assertTrue(indices)
        np.testing.assert_array_equal(z['raw']['candidate'][indices],cache['raw_candidate'][indices])
        self.assertTrue(np.any(z['raw']['candidate'][indices]!=cache['raw_source_log'][indices]))
        np.testing.assert_array_equal(z['available'],cache['available'])
        for row in z['audit']['local']:
            if row['status']=='same_type_missing_month':
                self.assertIsNone(row['same_peer_numerator']);self.assertIsNone(row['same_peer_correction'])

    def test_own_history_threshold18_and_insufficient_pooled(self):
        args=fixture();args[6:]=[v[4:5] for v in args[6:]]
        own=np.flatnonzero((args[2]=='ONE')&(args[1]<=np.datetime64('2017-12-31')))
        args[5][own[:-18]]=False;z=build(args)
        self.assertEqual(z['audit']['local'][0]['own_months'],18);self.assertTrue(z['local_available'][0])
        args[5][own[-18]]=False;z=build(args)
        self.assertEqual(z['audit']['local'][0]['own_months'],17);self.assertFalse(z['local_available'][0])
        for k in ('candidate','control'):self.assertEqual(z['raw'][k][0],z['raw']['source_log'][0])

    def test_four_target_finite_cohort_and_singleton_month_exclusion(self):
        args=fixture();args[4]['loss_utility'][:3]=np.nan;z=build(args)
        for row in z['audit']['annual']:self.assertTrue(set(range(4)).isdisjoint(row['training_row_ids']))

    def test_zero_residual_still_counted_and_identity(self):
        args=fixture()
        for y in args[4].values():y[:]=0.
        z=build(args);self.assertGreater(z['audit']['counts']['own_weight_evaluations'],0)
        for k in oracle.RAW_KEYS:np.testing.assert_array_equal(z['raw'][k][z['available']],0.)

    def test_future_poison_physical_prefixes_and_no_mutation(self):
        args=fixture();cache=cached(args);full=build(args,cache);original=copy.deepcopy(args)
        for n in range(len(args[6])+1):
            smaller=prefix(args,n);ref={k:v[:n].copy() for k,v in cache.items()};z=build(smaller,ref)
            for k in oracle.RAW_KEYS:oracle.close(z['raw'][k],full['raw'][k][:n],k,atol=0.)
            oracle.same(z['available'],full['available'][:n],'prefix support')
        a=prefix(args,5,False);ref={k:v[:5].copy() for k,v in cache.items()};expected=build(a,ref)
        future=a[1]>np.datetime64('2017-12-31')
        for i in (3,5):a[i]=a[i].astype(object);a[i][future]='unconsumed poison'
        for k in a[4]:a[4][k]=a[4][k].astype(object);a[4][k][future]='unconsumed poison'
        actual=build(a,ref);oracle.record_check(actual['audit'],expected['audit'],'future')
        for i,value in enumerate(args):
            if isinstance(value,dict):
                for k in value:np.testing.assert_array_equal(value[k],original[i][k])
            else:np.testing.assert_array_equal(value,original[i])

    def test_invalid_cache_shapes_masks_and_forecasts_fail(self):
        args=fixture();cache=cached(args)
        for kind in ('extra','missing','axis','bool','nan','source','local_constant','old_control','old_candidate'):
            c=copy.deepcopy(cache)
            if kind=='extra':c['extra']=np.zeros(len(args[6]))
            if kind=='missing':c.pop('raw_control')
            if kind=='axis':c['raw_candidate']=c['raw_candidate'][:-1]
            if kind=='bool':c['available']=c['available'].astype(int)
            if kind=='nan':c['raw_candidate'][-1]=np.nan
            if kind=='source':c['raw_source_log'][-1]+=.01
            if kind=='local_constant':c['raw_local_constant'][-1]+=.01;c['raw_control'][-1]+=.01
            if kind=='old_control':c['raw_control'][-1]+=.01
            if kind=='old_candidate':c['raw_candidate'][-1]+=.01
            with self.subTest(kind=kind),self.assertRaises((ValueError,FloatingPointError)):build(args,c)

    def test_invalid_inputs_and_etf_metadata(self):
        for kind in ('shape','bool','duplicate','targets','bounds','end','etfs','duplicate_etf'):
            args=fixture();cache=cached(args);etfs=ETFS
            if kind=='shape':args[3]=args[3][:,:9]
            if kind=='bool':args[5]=args[5].astype(int)
            if kind=='duplicate':args[2][1]=args[2][0]
            if kind=='targets':args[4].pop('vol_scaled')
            if kind=='bounds':args[3][0,0]=.6
            if kind=='end':args[1][0]=args[0][0]
            if kind=='etfs':etfs=[' FUND']
            if kind=='duplicate_etf':etfs=['FUND','FUND']
            with self.subTest(kind=kind),self.assertRaises(ValueError):build(args,cache,etfs)

    def test_producer_crosscheck_complete_audit_and_prefixes(self):
        import monthly_asset_peer213_model as producer
        args=fixture();cache=cached(args)
        for n in (0,1,5,8,12):
            small=prefix(args,n);ref={k:v[:n].copy() for k,v in cache.items()}
            got=producer.annual_forecasts(*small,etfs=ETFS,cached_reference=ref)
            receipt=checked(small,got,ref);self.assertTrue(all(receipt[k] for k in oracle.PREPARE_FLAGS))
        self.assertEqual(tuple(producer.COUNT_FIELDS),oracle.COUNT_FIELDS)

    def test_corrupt_forecast_audit_peer_domains_and_fallback_fail(self):
        import monthly_asset_peer213_model as producer
        args=fixture();cache=cached(args);built=producer.annual_forecasts(*args,etfs=ETFS,cached_reference=cache)
        loc=next(i for i,r in enumerate(built['audit']['local']) if r['status']=='same_type_supported')
        for change in ('raw','mask','peer_id','same_id','same_mean','own_weight','denominator','cutoff','counter','type'):
            z=copy.deepcopy(built);row=z['audit']['local'][loc]
            if change=='raw':z['raw']['candidate'][-1]+=.001
            if change=='mask':z['available'][-1]=False
            if change=='peer_id':row['peer_months'][0]['peer_row_ids'][0]=row['own_row_ids'][0]
            if change=='same_id':row['peer_months'][0]['same_peer_row_ids'][0]=row['own_row_ids'][0]
            if change=='same_mean':row['peer_months'][0]['same_peer_mean']+=.001
            if change=='own_weight':row['own_weights'][0]*=2
            if change=='denominator':row['correction_denominator']+=1
            if change=='cutoff':row['cutoff']='2022-12-31'
            if change=='counter':z['audit']['counts']['same_type_month_evaluations']-=1
            if change=='type':row['asset_type']='etf' if row['asset_type']=='stock' else 'stock'
            with self.subTest(change=change),self.assertRaises(ValueError):checked(args,z,cache)


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


class RankTests(unittest.TestCase):
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
        self.assertNotIn('import monthly_asset_peer213_model',source)
        self.assertEqual(oracle.sha(oracle.ACCOUNTING),oracle.ACCOUNTING_SHA)
        helper=oracle.accounting()
        self.assertTrue(callable(helper.audit_path));self.assertTrue(callable(helper.statistics_by_split))
        self.assertEqual(len(oracle.ARMS)*len(oracle.PATHS),9)

    def test_raw_fallback_does_not_imply_identical_cross_section_rank(self):
        a,f=rank_fixture();f['raw']['candidate'][4:8]=[.4,.3,.2,.1];f['raw']['control'][4:8]=[.4,.5,.6,.7]
        r,d,_=oracle.rank_oracle(a,f)
        self.assertEqual(f['raw']['candidate'][4],f['raw']['control'][4])
        self.assertNotEqual(r[oracle.ARMS[1]][1,0],r[oracle.ARMS[2]][1,0])


if __name__=='__main__':unittest.main(verbosity=2)
