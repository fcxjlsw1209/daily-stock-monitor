"""Invented prices/features only; never open research caches or run a phase."""
import copy
from pathlib import Path
import unittest
import importlib.util
import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location('tested_ineligible227', Path(__file__).with_name('monthly_ineligible_source227_model.py'))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def fixture_forecasts():
    rng = np.random.default_rng(227)
    days = pd.date_range('2013-12-31','2019-12-31',freq='ME')
    dates = np.repeat(days.to_numpy(dtype='datetime64[D]'),4)
    ends = np.repeat((days+pd.offsets.MonthEnd(1)).to_numpy(dtype='datetime64[D]'),4)
    symbols = np.tile(['A','B','C','D'],len(days))
    X = rng.uniform(-.5,.5,(len(dates),10))
    y = .1*X[:,0]+.2*X[:,1]**2+np.tile([-.04,.01,.05,-.02],len(days))
    y = (y.reshape(-1,4)-y.reshape(-1,4).mean(axis=1,keepdims=True)).ravel()
    targets = {key:y*(j+1) for j,key in enumerate(m.TARGETS)}
    ed = np.array(['2016-08-17','2016-08-31','2017-01-31','2017-11-30','2018-01-31','2018-11-30'],dtype='datetime64[D]')
    args = [dates,ends,symbols,X,targets,np.ones(len(dates),bool),ed,
            np.array(['A','B','C','D','A','B']),rng.uniform(-.5,.5,(len(ed),10)),
            np.array([True,False,True,False,True,False])]
    old = m.local.annual_forecasts(*args)
    cache = {'raw_'+key:old['raw'][key].copy() for key in m.RAW_KEYS}
    cache.update(raw_candidate=old['raw']['local_linear'].copy(),raw_control=old['raw']['local_constant'].copy(),
                 available=old['available'].copy(),local_available=old['local_available'].copy())
    return args,cache,np.array([True,True,True,False,True,True])


def physical_prefix(fixture,n):
    args,cache,allowed = copy.deepcopy(fixture)
    for k in (6,7,8,9): args[k] = args[k][:n]
    cache = {key:value[:n] for key,value in cache.items()};allowed=allowed[:n]
    cutoff=np.datetime64('%04d-12-31'%(int(str(args[6][-1])[:4])-1),'D') if n else None
    use=args[1]<=cutoff if n else np.zeros(len(args[0]),bool)
    for k in (0,1,2,3,5):args[k]=args[k][use]
    args[4]={key:value[use] for key,value in args[4].items()}
    return args,cache,allowed


def run_forecast(fixture):
    args,cache,allowed=fixture
    return m.forecasts(*args,cached_source=cache,expansion_allowed=allowed)


def fixture_labels(first='2015-05-29',second='2015-06-30'):
    sessions=pd.bdate_range('2012-01-02','2018-12-31')
    t=np.arange(len(sessions),dtype=float)
    adj=pd.DataFrame({
        'A':100*np.exp(.0005*t+.015*np.sin(t*.23)),
        'B':100*np.exp(.0002*t+.012*np.cos(t*.19)),
        'C':100*np.exp(-.0001*t+.017*np.sin(t*.31)),
        'D':100*np.exp(.0003*t+.014*np.cos(t*.29)),
        'SPY':100*np.exp(.0002*t+.009*np.sin(t*.17))},index=sessions)
    days=[first,second]
    ends=[str(sessions[sessions.to_period('M')==pd.Timestamp(d).to_period('M')+1][-1].date()) for d in days]
    tm=dict(dates=days,ends=ends,symbols=[['A','B','C','D'] for _ in days],feature_names=['legacy_not_plain10']*21)
    dates=np.repeat(np.array(days,dtype='datetime64[D]'),4)
    end=np.repeat(np.array(ends,dtype='datetime64[D]'),4)
    symbols=np.tile(['A','B','C','D'],2)
    eligible=np.tile([True,True,False,True],2)
    cached_good=np.tile([True,True,False,False],2)
    X=np.random.default_rng(7227).uniform(-.5,.5,(8,10))
    old=m.old_labels.build_targets(adj,dates,end,symbols,eligible,cached_good,np.isfinite(X).all(axis=1))
    train=dict(dates=dates,ends=end,offsets=np.array([0,4,8]),train_plain=X,
               train_eligible=eligible,train_good=cached_good,common_good=old['good'],vol63=old['vol63'])
    train.update({'target_'+key:value for key,value in old['targets'].items()})
    return adj,train,tm,[str(x.date()) for x in sessions]


class ForecastTests(unittest.TestCase):
    def test_exact_old_copy_and_new_mask_only(self):
        f=fixture_forecasts();z=run_forecast(f);args,c,allowed=f
        self.assertEqual(set(z),{'raw','available','local_available','audit'})
        self.assertEqual(set(z['raw']),set(m.RAW_KEYS))
        new=allowed&~args[9]
        self.assertEqual(z['audit']['new_query_mask'],new.tolist())
        self.assertTrue(z['available'][new].all())
        self.assertFalse(z['available'][3])
        for key in m.RAW_KEYS:
            self.assertEqual(z['raw'][key][c['available']].tobytes(),c['raw_'+key][c['available']].tobytes())

    def test_missing_warmup_does_not_remove_original(self):
        f=fixture_forecasts();f[2][:]=False;z=run_forecast(f)
        for key in m.RAW_KEYS:np.testing.assert_array_equal(z['raw'][key],f[1]['raw_'+key])
        self.assertEqual(z['audit']['counts']['added_available_rows'],0)

    def test_original_mismatch_aborts_instead_of_stitching(self):
        f=fixture_forecasts();f[1]['raw_local_linear'][0]+=1e-15
        with self.assertRaises(AssertionError):run_forecast(f)

    def test_original_mask_mismatch_fails(self):
        f=fixture_forecasts();f[1]['available'][1]=True
        with self.assertRaises(ValueError):run_forecast(f)

    def test_counts_include_both_replay_domains(self):
        z=run_forecast(fixture_forecasts());a=z['audit'];c=a['counts']
        self.assertEqual(set(c),set(m.COUNT_FIELDS))
        self.assertEqual(c['available_rows'],c['original_available_rows']+c['added_available_rows'])
        for label in ('old','new'):
            r=a[label+'_replay'];n=sum(x['prediction_rows']>0 for x in r['annual'])
            self.assertEqual(c[label+'_source_target_fits'],4*n)
            self.assertEqual(c[label+'_pooled_log_refits'],n)
            self.assertEqual(c[label+'_local_solves'],sum(x['status']=='fit' for x in r['local']))
        self.assertEqual(c['actual_regression_solves'],sum(c[k] for k in ('old_source_target_fits','new_source_target_fits','old_pooled_log_refits','new_pooled_log_refits','old_local_solves','new_local_solves')))

    def test_all_physical_prefixes_exact(self):
        f=fixture_forecasts();z=run_forecast(f)
        for n in range(len(f[0][6])+1):
            short=run_forecast(physical_prefix(f,n))
            for key in m.RAW_KEYS:np.testing.assert_array_equal(short['raw'][key],z['raw'][key][:n])
            for key in ('available','local_available'):np.testing.assert_array_equal(short[key],z[key][:n])

    def test_future_unused_training_poison(self):
        f=fixture_forecasts();short=physical_prefix(f,2);z=run_forecast(short)
        f[0][3][f[0][1]>np.datetime64('2015-12-31')]=np.inf
        for k in f[0][4]:f[0][4][k][f[0][1]>np.datetime64('2015-12-31')]=np.nan
        zz=run_forecast(physical_prefix(f,2))
        for key in m.RAW_KEYS:np.testing.assert_array_equal(zz['raw'][key],z['raw'][key])

    def test_no_own_history_keeps_pooled(self):
        f=fixture_forecasts();f[0][7][1]='Z';z=run_forecast(f)
        self.assertTrue(z['available'][1]);self.assertFalse(z['local_available'][1])
        self.assertEqual(z['raw']['local_linear'][1],z['raw']['source_log'][1])

    def test_nonfinite_new_feature_stays_missing(self):
        f=fixture_forecasts();f[0][8][1,0]=np.nan;z=run_forecast(f)
        self.assertFalse(z['available'][1])

    def test_malformed_bool_and_bounds_fail(self):
        f=fixture_forecasts()
        with self.assertRaises(ValueError):m.forecasts(*f[0],cached_source=f[1],expansion_allowed=f[2].astype(int))
        f[0][8][1,0]=.6
        with self.assertRaises(ValueError):run_forecast(f)


class LabelTests(unittest.TestCase):
    def test_old_targets_exact_added_only_ineligible(self):
        f=fixture_labels();z=m.build_labels(*f);old=f[1]
        self.assertEqual(set(z),{'targets','good','added','vol63','audit'})
        self.assertEqual(z['added'].tolist(),[False,False,True,False]*2)
        self.assertTrue(np.array_equal(z['good'],old['common_good']|z['added']))
        for k in m.TARGETS:
            use=~z['added'];self.assertEqual(z['targets'][k][use].tobytes(),old['target_'+k][use].tobytes())
        self.assertEqual(set(z['audit']['counts']),set(m.LABEL_COUNT_FIELDS))

    def test_new_labels_do_not_change_centers_or_old_rows(self):
        f=fixture_labels();before=m.build_labels(*f)
        date=pd.Timestamp(f[2]['ends'][-1]);f[0].at[date,'C']*=50
        z=m.build_labels(*f)
        self.assertEqual([r['centers'] for r in z['audit']['months']],[r['centers'] for r in before['audit']['months']])
        for key in m.TARGETS:np.testing.assert_array_equal(z['targets'][key][f[1]['common_good']],before['targets'][key][f[1]['common_good']])
        self.assertNotEqual(z['targets']['log_excess'][6],before['targets']['log_excess'][6])

    def test_future_asset_hole_blocks_added(self):
        f=fixture_labels();f[0].at[pd.Timestamp('2015-07-13'),'C']=np.nan
        z=m.build_labels(*f);self.assertFalse(z['added'][6])
        self.assertEqual(z['audit']['rows'][6]['reason'],'invalid_future_asset_price')

    def test_old_future_exception_retained_not_dropped(self):
        f=fixture_labels();f[0].at[pd.Timestamp('2015-06-12'),'SPY']=np.nan
        z=m.build_labels(*f)
        self.assertTrue(z['good'][0] and z['good'][1]);self.assertFalse(z['added'][2])
        self.assertEqual(z['audit']['counts']['original_strict_exceptions'],2)
        self.assertEqual(z['audit']['rows'][0]['reason'],'original_retained_exception:invalid_future_spy_price')

    def test_missing_official_future_row_not_shortened(self):
        f=list(fixture_labels());f[0]=f[0].drop(pd.Timestamp('2015-07-13'))
        z=m.build_labels(*f);self.assertFalse(z['added'][6])
        self.assertEqual(z['audit']['rows'][6]['reason'],'missing_official_price_session')

    def test_mutated_original_history_is_not_silently_repaired(self):
        f=fixture_labels();f[0].at[pd.Timestamp('2015-06-12'),'C']=np.nan
        with self.assertRaises(AssertionError):m.build_labels(*f)

    def test_295_history_required_even_when_vol64_exists(self):
        f=fixture_labels('2012-05-31','2012-06-29');z=m.build_labels(*f)
        self.assertFalse(z['added'].any());self.assertTrue(z['good'][0])
        self.assertEqual(z['audit']['rows'][2]['reason'],'insufficient_history')

    def test_eligible_common_bad_not_revived(self):
        z=m.build_labels(*fixture_labels())
        self.assertFalse(z['good'][3]);self.assertEqual(z['audit']['rows'][3]['reason'],'original_eligible_not_admitted')

    def test_no_new_center_from_added_rows(self):
        f=fixture_labels();f[1]['train_good'][:4]=False
        old=m.old_labels.build_targets(f[0],f[1]['dates'],f[1]['ends'],np.tile(['A','B','C','D'],2),f[1]['train_eligible'],f[1]['train_good'],np.isfinite(f[1]['train_plain']).all(axis=1))
        f[1]['common_good']=old['good']
        for k in m.TARGETS:f[1]['target_'+k]=old['targets'][k]
        z=m.build_labels(*f);self.assertFalse(z['added'][2])
        self.assertEqual(z['audit']['rows'][2]['reason'],'fewer_than_two_original_center_rows')

    def test_label_parity_tolerance_then_exact_cache_copy(self):
        f=fixture_labels();f[1]['target_log_excess'][0]+=5e-13;z=m.build_labels(*f)
        self.assertEqual(z['targets']['log_excess'][0],f[1]['target_log_excess'][0])
        f[1]['target_log_excess'][0]+=1e-7
        with self.assertRaises(AssertionError):m.build_labels(*f)

    def test_wrong_offsets_or_identity_blocked(self):
        f=fixture_labels();f[1]['offsets'][-1]+=1
        with self.assertRaises(ValueError):m.build_labels(*f)
        f=fixture_labels();f[2]['dates'][0]='2015-05-28'
        with self.assertRaises(ValueError):m.build_labels(*f)

    def test_calendar_boundary_must_be_proved(self):
        f=list(fixture_labels());f[3]=[d for d in f[3] if d<=f[2]['ends'][-1]]
        with self.assertRaises(ValueError):m.build_labels(*f)

    def test_future_prefix_and_zero_rows(self):
        f=fixture_labels();z=m.build_labels(*f)
        for n in (0,1,2):
            tm={k:(v[:n] if k in ('dates','ends','symbols') else v) for k,v in f[2].items()}
            t={k:(v[:n+1] if k=='offsets' else v[:n*4]) for k,v in f[1].items()}
            last=pd.Timestamp(tm['ends'][-1]) if n else f[0].index[0]
            # The next official session proves the last month boundary; prices themselves end at label-end.
            adj=f[0].loc[:last].copy() if n else f[0].iloc[:0].copy()
            zz=m.build_labels(adj,t,tm,f[3])
            for key in m.TARGETS:np.testing.assert_array_equal(zz['targets'][key],z['targets'][key][:n*4])
            np.testing.assert_array_equal(zz['good'],z['good'][:n*4])


if __name__=='__main__': unittest.main(verbosity=2)
