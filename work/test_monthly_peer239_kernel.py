"""Synthetic-only peer239 numerical tests; no historical files or replay."""
from copy import deepcopy
from datetime import date,timedelta
import json
import unittest
from unittest import mock
import numpy as np
import pandas as pd
import monthly_peer239_kernel as kernel
import peer239_sources as source
import peer239_maturity_sources as maturity


def repair_policy():
    rules=[('AZN','XLV','2023-03-11',None),('BUD','XLP','2016-03-15',None),('CHL','VOX','2016-04-28',None),('GOOGL','XLK','2016-02-12','2018-09-24'),('PDD','XLY','2020-04-25',None),('TSLA','XLY','2020-02-14',None)]
    return dict(schema='ownsector238_repair_source_policy_v1',missing_mapping_rules=[dict(symbol=s,benchmark=b,public_date=str(date.fromisoformat(start)-timedelta(days=1)),valid_from=start,valid_until_exclusive=end,source_file='synthetic.json',source_ids=['synthetic_'+s]) for s,b,start,end in rules],communication_benchmark=dict(old='XLC',new='VOX',price_field='adjClose',fill_or_splice=False))


def history_policy():
    return dict(schema='peer239_history_bound_policy_v1',securities=[dict(symbol='SPCX',issuer_key='https://companiesmarketcap.com/spacex/marketcap/',not_before='2026-06-08',available_from='2026-06-09',kind='exchange_new_listing_lower_bound',evidence_id='spcx_nasdaq_new_issue'),dict(symbol='SKHY',issuer_key='https://companiesmarketcap.com/sk-hynix/marketcap/',not_before='2026-07-10',available_from='2026-07-11',kind='issuer_actual_adr_start',evidence_id='skhy_adr_listing')])


def make_report(panel,meta,calendar,arrays=None,sectors=None,issuers=None):
    # Source declarations only: no momentum, median, feature or rank arithmetic.
    if sectors is None:sectors=pd.DataFrame('Technology',index=panel['adj'].index,columns=panel['adj'].columns)
    if issuers is None:issuers={s:'issuer:'+s for names in meta['symbols'] for s in names if s not in meta['etfs']}
    original=source.inspect(dict(adj=panel['adj'],sectors=sectors),meta,calendar,issuers,repair_policy())
    return maturity.project(original,meta,history_policy())


def fixture():
    axis=pd.bdate_range('2015-01-01','2017-03-03');calendar=[str(d.date()) for d in axis]
    dates=['2016-08-17','2016-08-31','2016-09-30','2016-10-31'];names=['AAA','BBB','CCC','DDD','QQQ']
    meta=dict(dates=dates,symbols=[names.copy() for _ in dates],etfs=['QQQ'])
    frame=pd.DataFrame(100.,index=axis,columns=names)
    for day in dates[1:]:
        t=calendar.index(day)
        for s,gross in zip(names[:4],(1.125,1.25,1.5,.875)):frame.at[axis[t-42],s]=100*gross
    panel={'adj':frame};arrays={'eligible':np.ones((4,5),dtype=bool)};arrays['eligible'][:,3]=False
    return panel,meta,calendar,arrays,make_report(panel,meta,calendar,arrays)


def physical(args,n):
    panel,meta,calendar,arrays,report=args
    mm=deepcopy(meta);mm['dates']=mm['dates'][:n];mm['symbols']=mm['symbols'][:n]
    if n:
        end=pd.Timestamp(mm['dates'][-1]);pos=calendar.index(mm['dates'][-1]);cc=calendar[:pos+2]
        pp={'adj':panel['adj'].loc[:end].copy()}
    else:cc=[];pp={'adj':panel['adj'].iloc[:0].copy()}
    rr=deepcopy(report);rr['events']=rr['events'][:n];rr['records']=[r for r in rr['records'] if r['query_index']<n]
    return pp,mm,cc,{'eligible':arrays['eligible'][:n].copy()},rr


def run(args,limit=None):return kernel.build_features(*args,limit=limit)


def arrays_equal(case,a,b):
    case.assertEqual(set(a),set(b))
    for k in a:np.testing.assert_array_equal(a[k],b[k])


class KernelTests(unittest.TestCase):
    def test_schema_counts_odd_median_and_rank(self):
        values,audit=run(fixture())
        self.assertEqual(tuple(values),kernel.VALUE_KEYS);self.assertEqual(set(audit),{'schema','counts','events'})
        self.assertEqual(audit['counts'],dict(events=4,stock_cells=16,mature_cells=12,supported_cells=12,endpoint_observations=24,new_model_fits=0))
        np.testing.assert_array_equal(values['momentum'][1,:4],[.125,.25,.5,-.125])
        np.testing.assert_array_equal(values['peer_median'][1,:4],[.25,.125,.125,.25])
        np.testing.assert_array_equal(values['peer_target'][1,:4],[0.,.25,.5,-.25])
        np.testing.assert_array_equal(values['absolute_target'][1,:4],[0.,.25,.5,-.25])
        self.assertEqual(tuple(audit['events'][1]),kernel.EVENT_KEYS);self.assertEqual(tuple(audit['events'][1]['cells'][0]),kernel.CELL_KEYS)
        json.dumps(audit,allow_nan=False)

    def test_even_median_canonical_two_peers(self):
        args=list(fixture());sectors=pd.DataFrame('Technology',index=args[0]['adj'].index,columns=args[0]['adj'].columns);sectors['DDD']='Healthcare'
        args[4]=make_report(*args[:4],sectors=sectors);v,a=run(args)
        np.testing.assert_array_equal(v['peer_median'][1,:3],[.375,.3125,.1875])
        self.assertTrue(v['mature'][1,3]);self.assertFalse(v['supported'][1,3]);self.assertTrue(np.isnan(v['peer_median'][1,3]))
        self.assertEqual(a['counts']['supported_cells'],9)

    def test_all_ties_and_zero_peer_residual_valid(self):
        args=list(fixture());args[0]['adj'].loc[:,:]=100.;args[4]=make_report(*args[:4]);v,a=run(args)
        np.testing.assert_array_equal(v['peer_target'][1,:4],[.125]*4);np.testing.assert_array_equal(v['absolute_target'][1,:4],[.125]*4)
        self.assertTrue(all(c['peer_difference']==0. for c in a['events'][1]['cells']))

    def test_tied_average_percentile_exact(self):
        args=list(fixture());args[0]['adj']['BBB']=args[0]['adj']['AAA'];args[4]=make_report(*args[:4]);v,_=run(args)
        self.assertEqual(v['absolute_target'][1,0],.125);self.assertEqual(v['absolute_target'][1,1],.125)
        self.assertEqual(v['peer_target'][1,0],v['peer_target'][1,1])

    def test_negative_Efalse_peer_participates_and_E_toggle_invariant(self):
        args=fixture();v,a=run(args)
        self.assertFalse(args[3]['eligible'][1,3]);self.assertEqual(v['momentum'][1,3],-.125)
        self.assertIn(3,a['events'][1]['cells'][0]['peer_indices']);args[3]['eligible']=~args[3]['eligible']
        vv,aa=run(args);arrays_equal(self,v,vv);self.assertEqual(a,aa)

    def test_rank_domain_supported_only_common_control(self):
        args=list(fixture());sectors=pd.DataFrame('Technology',index=args[0]['adj'].index,columns=args[0]['adj'].columns);sectors['DDD']='Healthcare';args[4]=make_report(*args[:4],sectors=sectors)
        v,_=run(args);self.assertAlmostEqual(v['absolute_target'][1,0],1/3-.5);self.assertEqual(v['absolute_target'][1,2],.5)
        self.assertTrue(np.isnan(v['absolute_target'][1,3]));self.assertTrue(np.isnan(v['peer_target'][1,3]))

    def test_price_rescaling_by_security_invariance(self):
        args=list(fixture());v,a=run(args)
        for s,scale in zip(args[0]['adj'].columns,(2.,4.,.5,.25,8.)):args[0]['adj'][s]*=scale
        vv,aa=run(args);arrays_equal(self,v,vv);self.assertEqual(a,aa)

    def test_partial_and_ETF_no_numeric_reads(self):
        args=fixture();args[0]['adj']['QQQ']='poison';v,a=run(args)
        self.assertFalse(v['mature'][0].any());self.assertFalse(v['supported'][:,4].any())
        for k in kernel.VALUE_KEYS[2:]:self.assertTrue(np.isnan(v[k][0]).all());self.assertTrue(np.isnan(v[k][:,4]).all())
        prefix=physical(args,1);prefix[0]['adj']=prefix[0]['adj'].iloc[:0]
        pv,pa=run(prefix);arrays_equal(self,pv,{k:x[:1] for k,x in v.items()})

    def test_insufficient_peers_not_a_source_gap(self):
        args=list(fixture());sectors=pd.DataFrame('Technology',index=args[0]['adj'].index,columns=args[0]['adj'].columns);sectors['CCC']='Healthcare';sectors['DDD']='Energy';args[4]=make_report(*args[:4],sectors=sectors)
        v,a=run(args);self.assertEqual(a['counts']['mature_cells'],12);self.assertEqual(a['counts']['supported_cells'],0)
        self.assertTrue(np.isnan(v['peer_target']).all());self.assertTrue(np.isfinite(v['momentum'][1:,:4]).all())

    def test_all_logical_physical_prefixes_exact_and_immutable(self):
        args=fixture();before=deepcopy(args);v,a=run(args)
        for n in range(5):
            vv,aa=run(args,n);pv,pa=run(physical(args,n));arrays_equal(self,vv,pv);self.assertEqual(aa,pa)
            arrays_equal(self,vv,{k:x[:n] for k,x in v.items()});self.assertEqual(aa['events'],a['events'][:n])
        pd.testing.assert_frame_equal(args[0]['adj'],before[0]['adj']);self.assertEqual(args[1:3],before[1:3]);np.testing.assert_array_equal(args[3]['eligible'],before[3]['eligible']);self.assertEqual(args[4],before[4])

    def test_empty_physical_zero_width(self):
        args=list(physical(fixture(),0));args[3]['eligible']=np.zeros((0,0),dtype=bool);v,a=run(args)
        self.assertTrue(all(x.shape==(0,0) for x in v.values()));self.assertEqual(a['events'],[])

    def test_future_price_report_and_meta_poison_not_consumed(self):
        args=fixture();v,a=run(args,2);cutoff=pd.Timestamp(args[1]['dates'][1]);args[0]['adj']=args[0]['adj'].astype(object);args[0]['adj'].loc[args[0]['adj'].index>cutoff,:]='poison'
        args[1]['dates'][2]='badfuture';args[1]['symbols'][2]=['bad'];args[4]['events'][2]={'poison':True}
        for r in args[4]['records']:
            if r['query_index']>=2:r.clear();r.update(query_index=2,poison=True)
        vv,aa=run(args,2);arrays_equal(self,v,vv);self.assertEqual(a,aa)

    def test_last42_prices_excluded(self):
        args=list(physical(fixture(),2));v,a=run(args);pos=args[2].index(args[1]['dates'][-1]);cutoff=pd.Timestamp(args[2][pos-42]);args[0]['adj']=args[0]['adj'].astype(object);args[0]['adj'].loc[args[0]['adj'].index>cutoff,:]='poison'
        vv,aa=run(args);arrays_equal(self,v,vv);self.assertEqual(a,aa)

    def test_no_silent_missing_peer(self):
        for mutate in ('omit','reorder','duplicate','wrong_count','wrong_support','extra'):
            args=fixture();r=args[4]['records'][4]
            if mutate=='omit':r['peer_indices'].pop()
            elif mutate=='reorder':r['peer_indices'].reverse()
            elif mutate=='duplicate':r['peer_indices'].append(r['peer_indices'][0])
            elif mutate=='wrong_count':r['peer_issuer_count']=1
            elif mutate=='wrong_support':r['structural_peer_support']=False
            else:r['peer_indices'].append(4)
            with self.subTest(mutate=mutate),self.assertRaises(ValueError):run(args)

    def test_missing_duplicate_extra_records_rejected(self):
        for mutate in ('missing','duplicate','ETF'):
            args=fixture()
            if mutate=='missing':args[4]['records'].pop(4)
            elif mutate=='duplicate':args[4]['records'].insert(4,deepcopy(args[4]['records'][4]))
            else:r=deepcopy(args[4]['records'][4]);r['column_index']=4;r['symbol']='QQQ';args[4]['records'].insert(8,r)
            with self.subTest(mutate=mutate),self.assertRaises(ValueError):run(args)

    def test_duplicate_issuer_and_missing_sector_rejected(self):
        for mutate in ('issuer_duplicate','issuer_missing','sector_missing'):
            args=fixture();r=args[4]['records'][4]
            if mutate=='issuer_duplicate':r['issuer_key']=args[4]['records'][5]['issuer_key']
            elif mutate=='issuer_missing':r['issuer_key']=None
            else:r['sector']=None
            with self.subTest(mutate=mutate),self.assertRaises(ValueError):run(args)

    def test_report_endpoint_date_role_status_binding(self):
        for key,value in (('role','benchmark_start'),('symbol','BBB'),('date','2015-01-01'),('status','missing'),('reason','bad')):
            args=fixture();args[4]['records'][4]['endpoints'][0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):run(args)

    def test_report_event_and_current_gap_validation(self):
        changes=[lambda r:r.__setitem__('admitted',False),lambda r:r.__setitem__('status','SOURCE_GAPS'),lambda r:r.__setitem__('gaps',[{}]),lambda r:r['records'][4].__setitem__('gap_reasons',['bad']),lambda r:r['events'][1].__setitem__('mature_indices',[]),lambda r:r['events'][1].__setitem__('supported_indices',[]),lambda r:r['events'][1].__setitem__('source_positions',[0,1])]
        for change in changes:
            args=fixture();change(args[4])
            with self.assertRaises(ValueError):run(args)

    def test_actual_numeric_invalid_required_hard_failure(self):
        for value in (None,pd.NA,np.nan,True,np.bool_(False),'100',1+0j,0.,-1.,float('inf'),10**400):
            args=fixture();args[0]['adj']=args[0]['adj'].astype(object);day=args[4]['records'][4]['source_dates'][0];args[0]['adj'].at[pd.Timestamp(day),'AAA']=value
            with self.subTest(value=repr(value)),self.assertRaises(ValueError):run(args)

    def test_absent_required_source_rows_or_columns_hard_failure(self):
        args=fixture();args[0]['adj']=args[0]['adj'].drop(columns='AAA')
        with self.assertRaises(ValueError):run(args)
        args=fixture();day=args[4]['records'][4]['source_dates'][0];args[0]['adj']=args[0]['adj'].drop(pd.Timestamp(day))
        with self.assertRaises(ValueError):run(args)

    def test_ratio_overflow_aborts(self):
        args=fixture();r=args[4]['records'][4];args[0]['adj'].at[pd.Timestamp(r['source_dates'][0]),'AAA']=1e-300;args[0]['adj'].at[pd.Timestamp(r['source_dates'][1]),'AAA']=1e308
        with self.assertRaises(FloatingPointError):run(args)

    def test_even_median_overflow_aborts(self):
        args=list(fixture());sectors=pd.DataFrame('Technology',index=args[0]['adj'].index,columns=args[0]['adj'].columns);sectors['DDD']='Healthcare'
        for r in args[4]['records']:
            if r['query_index']==1 and r['symbol'] in ('BBB','CCC'):
                args[0]['adj'].at[pd.Timestamp(r['source_dates'][0]),r['symbol']]=1.;args[0]['adj'].at[pd.Timestamp(r['source_dates'][1]),r['symbol']]=1e308
        args[4]=make_report(*args[:4],sectors=sectors)
        with self.assertRaises(FloatingPointError):run(args)

    def test_first_regular_row_applies_math(self):
        args=list(fixture());args[1]=dict(args[1],dates=args[1]['dates'][1:],symbols=args[1]['symbols'][1:]);args[3]={'eligible':args[3]['eligible'][1:]};args[4]=make_report(*args[:4]);v,_=run(args)
        self.assertTrue(v['mature'][0,:4].all())

    def test_canonical_dates_calendar_shapes_and_bool(self):
        changes=[lambda a:a[1]['dates'].__setitem__(1,'2016-8-31'),lambda a:a[1]['dates'].__setitem__(1,'2016-08-30'),lambda a:a[1]['symbols'][1].__setitem__(1,'AAA'),lambda a:a[1]['etfs'].append('QQQ'),lambda a:a[3].__setitem__('eligible',a[3]['eligible'].astype(int)),lambda a:a[2].insert(0,a[2][0])]
        for change in changes:
            args=fixture();change(args)
            with self.assertRaises(ValueError):run(args)
        for limit in (-1,True,5,1.):
            with self.assertRaises(ValueError):run(fixture(),limit)

    def test_price_axis_schema_rejected(self):
        for mode in ('duplicate','reverse','timezone','nonmidnight'):
            args=fixture();f=args[0]['adj']
            if mode=='duplicate':f=pd.concat([f.iloc[:1],f])
            elif mode=='reverse':f=f.iloc[::-1]
            elif mode=='timezone':f=f.tz_localize('UTC')
            else:f=f.copy();f.index=f.index+pd.Timedelta(hours=1)
            args[0]['adj']=f
            with self.subTest(mode=mode),self.assertRaises(ValueError):run(args)

    def test_visible_official_warmup_retains_stock_without_numeric_reads(self):
        args=warm_fixture();v,a=run(args);self.assertFalse(v['mature'][0,3]);self.assertFalse(v['supported'][0,3]);self.assertTrue(np.isnan(v['momentum'][0,3]));self.assertEqual(a['counts']['stock_cells'],4);self.assertEqual(a['counts']['mature_cells'],3)
        self.assertNotIn(3,a['events'][0]['cells'][0]['peer_indices']);args[0]['adj']['SPCX']='numeric poison';vv,aa=run(args);arrays_equal(self,v,vv);self.assertEqual(a,aa)

    def test_bad_warmup_bound_or_evidence_not_allowed(self):
        for mode in ('future','wrong_issuer','wrong_bound','pretend_mature','gap_dropped'):
            args=warm_fixture();r=args[4]['records'][3]
            if mode=='future':r['history_bound']['available_from']='2026-09-01'
            elif mode=='wrong_issuer':r['history_bound']['issuer_key']='other'
            elif mode=='wrong_bound':r['history_bound']['not_before']='2026-07-01'
            elif mode=='pretend_mature':r['history_status']='observed_full_window'
            else:r['original_gap_reasons']=[]
            with self.subTest(mode=mode),self.assertRaises(ValueError):run(args)

    def test_warmup_does_not_erase_postbound_missing_or_prebound_price(self):
        for mode in ('post_missing','pre_valid'):
            args=warm_fixture();r=args[4]['records'][3]
            if mode=='post_missing':r['endpoints'][1].update(status='missing',reason='missing_value')
            else:r['endpoints'][0].update(status='valid',reason=None)
            with self.subTest(mode=mode),self.assertRaises(ValueError):run(args)

    def test_boolean_source_identity_indices_rejected(self):
        changes=[lambda r:r['events'][1].__setitem__('query_index',True),
            lambda r:r['events'][1]['stock_indices'].__setitem__(0,False),
            lambda r:r['events'][1]['mature_indices'].__setitem__(0,False),
            lambda r:r['events'][1]['supported_indices'].__setitem__(0,False),
            lambda r:r['records'][5]['peer_indices'].__setitem__(0,False)]
        for change in changes:
            args=fixture();change(args[4])
            with self.assertRaises(ValueError):run(args)

    def test_no_numeric_validation_on_unconsumed_future_E(self):
        args=fixture();v,a=run(args,2);args[3]['eligible']=args[3]['eligible'].astype(object);args[3]['eligible'][2:,:]='bad'
        vv,aa=run(args,2);arrays_equal(self,v,vv);self.assertEqual(a,aa)


def warm_fixture():
    axis=pd.bdate_range('2025-01-01','2026-09-03');calendar=[str(d.date()) for d in axis];names=['AAA','BBB','CCC','SPCX','QQQ'];meta=dict(dates=['2026-08-31'],symbols=[names],etfs=['QQQ']);frame=pd.DataFrame(100.,index=axis,columns=names);frame.loc[frame.index<pd.Timestamp('2026-06-08'),'SPCX']=np.nan
    sectors=pd.DataFrame('Technology',index=axis,columns=names);sectors['SPCX']='Unknown';issuers={s:'issuer:'+s for s in names};issuers['SPCX']='https://companiesmarketcap.com/spacex/marketcap/'
    panel={'adj':frame};arrays={'eligible':np.ones((1,5),dtype=bool)};report=make_report(panel,meta,calendar,arrays,sectors=sectors,issuers=issuers)
    return panel,meta,calendar,arrays,report


if __name__=='__main__':unittest.main(verbosity=2)
