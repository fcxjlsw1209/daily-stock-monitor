"""Synthetic source227 oracle checks; no real caches or research phases."""
import copy
import importlib.util
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
import monthly_ineligible_source227_independent as m


def module(name,path):
    sp=importlib.util.spec_from_file_location(name,path);obj=importlib.util.module_from_spec(sp);sp.loader.exec_module(obj);return obj


def fixture():
    calendar=pd.bdate_range('2013-01-01','2018-03-30')
    t=np.arange(len(calendar),dtype=float)
    adj=pd.DataFrame({s:100*np.exp(.0002*t+a*np.sin(t/11)+b*np.cos(t/23)) for s,a,b in
        [('A',.012,.008),('B',.009,-.01),('C',.018,.014),('SPY',.004,.003)]},index=calendar)
    signal=[g.iloc[-1] for _,g in pd.Series(calendar,index=calendar).groupby(calendar.to_period('M')) if str(g.iloc[0].to_period('M'))>='2014-08' and str(g.iloc[0].to_period('M'))<='2016-11']
    ds=np.array(signal,dtype='datetime64[D]');ends=[]
    for d in ds:
        nextmonth=pd.Timestamp(d).to_period('M')+1;ends.append(calendar[calendar.to_period('M')==nextmonth][-1])
    es=np.array(ends,dtype='datetime64[D]');N=len(ds);rng=np.random.default_rng(227)
    X=rng.uniform(-.5,.5,(N*3,10));el=np.tile([True,True,False],N)
    labels=module('source227_fixture_labels',m.ROOT/'outputs/monthly_target_definition70/labels.py')
    original=labels.build_targets(adj,np.repeat(ds,3),np.repeat(es,3),np.tile(['A','B','C'],N),el,el,np.ones(N*3,bool))
    train=dict(dates=np.repeat(ds,3),ends=np.repeat(es,3),offsets=np.arange(N+1)*3,train_plain=X,
        train_eligible=el,train_good=el.copy(),common_good=original['good'],vol63=original['vol63'])
    train.update({'target_'+k:v.copy() for k,v in original['targets'].items()})
    meta=dict(dates=ds.astype(str).tolist(),ends=es.astype(str).tolist(),symbols=[['A','B','C'] for _ in ds])
    return adj,train,meta,np.asarray(calendar.strftime('%Y-%m-%d').tolist(),dtype=object)


def model_fixture():
    adj,tr,meta,cal=fixture();N=len(meta['dates']);rng=np.random.default_rng(28)
    qd=np.repeat(np.array(['2017-01-31','2017-02-28','2017-03-31'],dtype='datetime64[D]'),3)
    args=[tr['dates'],tr['ends'],np.tile(['A','B','C'],N),tr['train_plain'],{k:tr['target_'+k] for k in m.TARGETS},tr['common_good'],
        qd,np.tile(['A','B','C'],3),rng.uniform(-.5,.5,(9,10)),np.tile([True,True,False],3)]
    old=m.frozen().annual_forecasts(*args)
    source={'raw_'+k:old['raw'][k].copy() for k in m.RAW_KEYS}
    source.update(available=old['available'].copy(),local_available=old['local_available'].copy(),
        raw_candidate=old['raw']['local_linear'].copy(),raw_control=old['raw']['local_constant'].copy())
    return args,source,np.ones(9,bool)


def prefix(args,source,allowed,n):
    a=copy.deepcopy(args)
    for i in (6,7,8,9):a[i]=a[i][:n]
    return a,{k:v[:n].copy() for k,v in source.items()},allowed[:n].copy()


class LabelTests(unittest.TestCase):
    def test_extra_rows_reuse_original_means_and_copy_original_targets(self):
        adj,tr,meta,cal=fixture();out=m.labels_oracle(adj,tr,meta,cal)
        self.assertEqual(int(out['added'].sum()),len(meta['dates']))
        for k in m.TARGETS:
            np.testing.assert_array_equal(out['targets'][k][tr['common_good']],tr['target_'+k][tr['common_good']])
        self.assertTrue(m.check_labels(adj,tr,meta,cal,out)['passed'])
        self.assertTrue(np.all(out['added']<=~tr['train_eligible']))

    def test_added_prices_cannot_recenter_existing_rows(self):
        args=list(fixture());a=m.labels_oracle(*args)
        args[0]=args[0].copy();args[0].loc[args[0].index>='2015-01-01','C']*=4
        b=m.labels_oracle(*args)
        self.assertEqual([z['centers'] for z in a['audit']['months']],[z['centers'] for z in b['audit']['months']])
        for k in m.TARGETS:np.testing.assert_array_equal(a['targets'][k][args[1]['common_good']],b['targets'][k][args[1]['common_good']])

    def test_original_clips_before_mean_and_no_second_clip(self):
        adj,tr,meta,cal=fixture();out=m.labels_oracle(adj,tr,meta,cal)
        month=out['audit']['months'][0];row=month['rows'][-1]
        for k in m.TARGETS:
            raw=out['audit']['rows'][row]['raw_targets'][k]
            self.assertEqual(out['targets'][k][row],np.clip(raw,-m.LIMITS[k],m.LIMITS[k])-month['centers'][k])

    def test_missing_official_future_row_drops_added_not_old(self):
        adj,tr,meta,cal=fixture();end=pd.Timestamp(meta['ends'][0]);signal=pd.Timestamp(meta['dates'][0])
        absent=adj.index[(adj.index>signal)&(adj.index<end)][2];adj=adj.drop(absent)
        tr={k:(v[:2] if k=='offsets' else v[:3]) for k,v in tr.items()};meta={k:v[:1] for k,v in meta.items()}
        out=m.labels_oracle(adj,tr,meta,cal)
        self.assertFalse(out['added'][2]);self.assertTrue(out['good'][0]);self.assertTrue(out['audit']['rows'][0]['original_strict_exception'])

    def test_invalid_interior_future_asset_and_spy_block_added(self):
        for s,reason in [('C','invalid_future_asset_price'),('SPY','invalid_future_spy_price')]:
            adj,tr,meta,cal=fixture();d=pd.Timestamp(meta['dates'][0]);adj.loc[adj.index[adj.index>d][2],s]=np.nan
            out=m.labels_oracle(adj,tr,meta,cal);self.assertEqual(out['audit']['rows'][2]['reason'],reason)

    def test_full295_warmup_not_just64(self):
        adj,tr,meta,cal=fixture();d=pd.Timestamp(meta['dates'][0]);p=adj.index.get_loc(d);adj.iloc[p-200,adj.columns.get_loc('C')]=0.
        out=m.labels_oracle(adj,tr,meta,cal);self.assertEqual(out['audit']['rows'][2]['reason'],'invalid_history_price')

    def test_no_center_does_not_form_new_training_month(self):
        adj,tr,meta,cal=fixture();tr=copy.deepcopy(tr);tr['common_good'][:3]=False
        for k in m.TARGETS:tr['target_'+k][:3]=np.nan
        out=m.labels_oracle(adj,tr,meta,cal)
        self.assertEqual(out['audit']['rows'][2]['reason'],'fewer_than_two_original_center_rows');self.assertFalse(out['good'][:3].any())
        self.assertIsNotNone(out['audit']['rows'][2]['raw_targets'])

    def test_eligible_rejected_rows_are_not_revived(self):
        adj,tr,meta,cal=fixture();tr=copy.deepcopy(tr);tr['common_good'][0:3]=False
        for k in m.TARGETS:tr['target_'+k][0:3]=np.nan
        out=m.labels_oracle(adj,tr,meta,cal)
        self.assertEqual(out['audit']['rows'][0]['reason'],'original_eligible_not_admitted');self.assertIsNone(out['audit']['rows'][0]['strict_source_valid'])

    def test_zero_vol_finite_feature_or_id_failure(self):
        for mode,reason in [('flat','zero_volatility'),('X','nonfinite_features'),('missing','missing_symbol')]:
            adj,tr,meta,cal=fixture();tr=copy.deepcopy(tr)
            if mode=='flat':adj['C']=100.
            elif mode=='X':tr['train_plain'][2,0]=np.nan
            else:adj=adj.drop(columns='C')
            out=m.labels_oracle(adj,tr,meta,cal);self.assertEqual(out['audit']['rows'][2]['reason'],reason)

    def test_calendar_monthend_and_exact_metadata(self):
        for mode in ('signal','end','offset','duplicate'):
            adj,tr,meta,cal=fixture();meta=copy.deepcopy(meta);tr=copy.deepcopy(tr)
            if mode=='signal':cal=cal[cal!=meta['dates'][0]]
            if mode=='end':cal=cal[cal<=meta['ends'][-1]]
            if mode=='offset':tr['offsets'][1]+=1
            if mode=='duplicate':meta['symbols'][0][2]='A'
            with self.assertRaises(ValueError):m.labels_oracle(adj,tr,meta,cal)

    def test_old_target_or_vol_mismatch_hardfail(self):
        for field in ('target_log_excess','vol63'):
            args=list(fixture());args[1]=copy.deepcopy(args[1]);args[1][field][0]+=.00001
            with self.assertRaises(ValueError):m.labels_oracle(*args)

    def test_label_physical_prefix_and_future_poison(self):
        adj,tr,meta,cal=fixture();full=m.labels_oracle(adj,tr,meta,cal)
        for n in (0,1,12,len(meta['dates'])):
            count=n*3;t={k:(v[:n+1].copy() if k=='offsets' else v[:count].copy()) for k,v in tr.items()}
            mt={k:v[:n] for k,v in meta.items()};end=meta['ends'][n-1] if n else '2013-01-01'
            px=adj.loc[:end];out=m.labels_oracle(px,t,mt,cal)
            for k in m.TARGETS:np.testing.assert_array_equal(out['targets'][k],full['targets'][k][:count])
        poisoned=adj.copy();poisoned.loc[poisoned.index>pd.Timestamp(meta['ends'][-1]),:]=np.nan
        changed=m.labels_oracle(poisoned,tr,meta,cal)
        m.record_check(changed['audit'],full['audit'])

    def test_derived_overflow_hardfail(self):
        adj,tr,meta,cal=fixture();d=pd.Timestamp(meta['dates'][0]);end=pd.Timestamp(meta['ends'][0]);adj.loc[d,'C']=1e-300;adj.loc[end,'C']=1e300
        with self.assertRaises(FloatingPointError):m.labels_oracle(adj,tr,meta,cal)


class ForecastTests(unittest.TestCase):
    def test_independent_controller_matches_frozen_full_audit(self):
        args,source,allowed=model_fixture();expect=m.frozen().annual_forecasts(*args);got=m.replay_oracle(*args)
        for k in m.RAW_KEYS:m.close(got['raw'][k],expect['raw'][k],k)
        m.record_check(got['audit'],expect['audit'])

    def test_all_three_new_raws_cold_own_fallback_old_exact(self):
        args,source,allowed=model_fixture();got=m.forecasts_oracle(*args,cached_source=source,expansion_allowed=allowed)
        self.assertEqual(got['audit']['counts']['added_available_rows'],3)
        for k in m.RAW_KEYS:
            m.close(got['raw'][k][source['available']],source['raw_'+k][source['available']],k)
            m.close(got['raw'][k][~source['available']],got['raw']['source_log'][~source['available']],'own cold fallback')
        self.assertTrue(m.check_forecasts(*args,cached_source=source,expansion_allowed=allowed,result=got)['passed'])

    def test_warmup_allowance_is_required_and_no_target_mask(self):
        args,source,allowed=model_fixture();allowed[2]=False
        got=m.forecasts_oracle(*args,cached_source=source,expansion_allowed=allowed)
        self.assertFalse(got['available'][2]);self.assertTrue(got['available'][5]);self.assertEqual(got['audit']['counts']['expansion_requested_rows'],2)

    def test_nonfinite_new_query_is_unavailable(self):
        args,source,allowed=model_fixture();args[8][2,0]=np.nan
        got=m.forecasts_oracle(*args,cached_source=source,expansion_allowed=allowed);self.assertFalse(got['available'][2])

    def test_old_cache_mismatch_or_extra_mask_rejected(self):
        for mode in ('value','mask'):
            args,source,allowed=model_fixture()
            if mode=='value':source['raw_local_linear'][0]+=.01
            else:source['available'][2]=True
            with self.assertRaises(ValueError):m.forecasts_oracle(*args,cached_source=source,expansion_allowed=allowed)

    def test_every_physical_query_prefix(self):
        args,source,allowed=model_fixture();full=m.forecasts_oracle(*args,cached_source=source,expansion_allowed=allowed)
        for n in range(len(allowed)+1):
            a,s,w=prefix(args,source,allowed,n);out=m.forecasts_oracle(*a,cached_source=s,expansion_allowed=w)
            for k in m.RAW_KEYS:m.close(out['raw'][k],full['raw'][k][:n],'prefix '+k)

    def test_future_training_poison_unused_and_maturity_boundaries(self):
        args,source,allowed=model_fixture();base=m.forecasts_oracle(*args,cached_source=source,expansion_allowed=allowed)
        a=copy.deepcopy(args);a[0]=np.r_[a[0],np.datetime64('2017-01-31')];a[1]=np.r_[a[1],np.datetime64('2017-02-28')]
        a[2]=np.concatenate([a[2],['FUTURE']]);a[3]=np.vstack([a[3],np.full(10,np.nan)]);a[5]=np.r_[a[5],True]
        for k in m.TARGETS:a[4][k]=np.r_[a[4][k],np.nan]
        other=m.forecasts_oracle(*a,cached_source=source,expansion_allowed=allowed)
        for k in m.RAW_KEYS:m.close(other['raw'][k],base['raw'][k],k)
        m.record_check(other['audit'],base['audit'])

    def test_singleton_four_target_and18month_rules(self):
        args,_,allowed=model_fixture();args=copy.deepcopy(args)
        args[5][51:]=False
        ref=m.frozen().annual_forecasts(*args);out=m.replay_oracle(*args)
        self.assertFalse(out['available'].any());m.record_check(out['audit'],ref['audit'])

    def test_no_mutation_and_cached_unused_fields_ignored(self):
        args,source,allowed=model_fixture();before=copy.deepcopy((args,source,allowed));source['raw_candidate'][:]=np.nan
        m.forecasts_oracle(*args,cached_source=source,expansion_allowed=allowed)
        for i in (0,1,2,3,5,6,7,8,9):np.testing.assert_array_equal(args[i],before[0][i])
        np.testing.assert_array_equal(allowed,before[2])


class BilateralTests(unittest.TestCase):
    def test_both_label_fixtures_exact_arrays_audit(self):
        producer=module('source227_cross_model',m.ROOT/'work/monthly_ineligible_source227_model.py')
        fixtures=module('source227_cross_fixtures',m.ROOT/'work/test_monthly_ineligible_source227_model.py')
        for args in (fixture(),fixtures.fixture_labels()):
            got=producer.build_labels(*args);m.check_labels(*args,got)
            adj,tr,meta,cal=args
            for n in range(len(meta['dates'])+1):
                count=int(tr['offsets'][n]);t={k:(v[:n+1].copy() if k=='offsets' else v[:count].copy()) for k,v in tr.items()}
                mt={k:(v[:n] if isinstance(v,list) and len(v)==len(meta['dates']) else v) for k,v in meta.items()}
                end=meta['ends'][n-1] if n else '1900-01-01';px=adj.loc[:end]
                small=producer.build_labels(px,t,mt,cal);m.check_labels(px,t,mt,cal,small)
                for k in m.TARGETS:m.close(small['targets'][k],got['targets'][k][:count],'label prefix '+k)

    def test_forecast_all_prefix_full_audit(self):
        producer=module('source227_cross_forecasts',m.ROOT/'work/monthly_ineligible_source227_model.py')
        fixtures=module('source227_cross_forecast_fixtures',m.ROOT/'work/test_monthly_ineligible_source227_model.py')
        for fixture_value in (model_fixture(),fixtures.fixture_forecasts()):
            for n in range(len(fixture_value[2])+1):
                a,s,w=fixtures.physical_prefix(fixture_value,n);out=producer.forecasts(*a,cached_source=s,expansion_allowed=w)
                m.check_forecasts(*a,cached_source=s,expansion_allowed=w,result=out)


if __name__=='__main__':unittest.main()
