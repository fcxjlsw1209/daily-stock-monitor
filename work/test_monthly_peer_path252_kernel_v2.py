"""Synthetic source/path/arithmetic tests; no historical data or replay."""
from copy import deepcopy
from fractions import Fraction
import json
import unittest
from unittest import mock
import numpy as np
import pandas as pd
import monthly_peer_path252_kernel_v2 as k


def pair(p):
    return Fraction(int(p['numerator']), int(p['denominator']))


def make_report(panel, meta, calendar, arrays=None, sectors=None, issuers=None):
    """Synthetic239 metadata declarations; not a source loader or new oracle."""
    sectors = {} if sectors is None else sectors
    issuers = {} if issuers is None else issuers
    events, records = [], []
    for i, (day, names) in enumerate(zip(meta['dates'],meta['symbols'])):
        t=calendar.index(day);partial=day=='2016-08-17'
        stock=[j for j,s in enumerate(names) if s not in meta['etfs']]
        positions=[] if partial else [t-252,t-42]
        dates=[calendar[p] for p in positions]
        rr=[]
        for j in stock:
            symbol=names[j];issuer=issuers.get(symbol,'issuer:'+symbol);sector=sectors.get(symbol,'XLK')
            bound=deepcopy(k.prior.BOUNDS.get(symbol))
            if partial or not bound or bound['issuer_key']!=issuer or bound['available_from']>day:bound=None
            status='initial_partial' if partial else 'official_pre_inception' if bound and dates[0]<bound['not_before'] else 'observed_full_window'
            points=[];deferred=[]
            for date,role in zip(dates,('security_start','security_end')):
                reason=None
                if symbol not in panel['adj'].columns:reason='missing_column'
                elif pd.Timestamp(date) not in panel['adj'].index:reason='missing_row'
                elif pd.isna(panel['adj'].at[pd.Timestamp(date),symbol]):reason='missing_value'
                points.append(dict(role=role,symbol=symbol,date=date,status='missing' if reason else 'valid',reason=reason))
                if reason:deferred.append('endpoint:'+role+':'+reason)
            if status=='official_pre_inception' and sector is None:deferred.append('missing_sector')
            deferred=sorted(deferred)
            rr.append(dict(query_index=i,column_index=j,date=day,symbol=symbol,issuer_key=issuer,
                original_sector=sector,sector=sector,mapping_origin='query_date_sector_panel',
                mapping_repair='unchanged' if sector is not None else 'unresolved',mapping_evidence=None,
                peer_indices=[],peer_issuer_count=0,structural_peer_support=False,initial_partial=partial,
                source_positions=positions.copy(),source_dates=dates.copy(),endpoints=points,gap_reasons=[],
                original_gap_reasons=deferred,history_status=status,history_bound=bound,deferred_gap_reasons=deferred.copy()))
        mature=[r['column_index'] for r in rr if r['history_status']=='observed_full_window']
        for r in rr:
            peers=[p for p in rr if r['column_index'] in mature and p['column_index'] in mature
                and r['sector']==p['sector'] and r['issuer_key']!=p['issuer_key']]
            peers.sort(key=lambda p:(p['issuer_key'],p['symbol'],p['column_index']))
            r.update(peer_indices=[p['column_index'] for p in peers],peer_issuer_count=len({p['issuer_key'] for p in peers}),
                     structural_peer_support=r['column_index'] in mature and len(peers)>=2)
        supported=[r['column_index'] for r in rr if r['structural_peer_support']]
        events.append(dict(query_index=i,date=day,initial_partial=partial,stock_indices=stock,
                           source_positions=positions,source_dates=dates,mature_indices=mature,supported_indices=supported))
        records.extend(rr)
    counts=dict(events=len(events),stock_cells=len(records),mature_cells=sum(r['history_status']=='observed_full_window' for r in records),
        warmup_cells=sum(r['history_status']=='official_pre_inception' for r in records),partial_cells=sum(r['initial_partial'] for r in records),
        supported_cells=sum(r['structural_peer_support'] for r in records),deferred_sector_cells=sum('missing_sector' in r['deferred_gap_reasons'] for r in records),
        deferred_endpoints=sum(sum(x.startswith('endpoint:') for x in r['deferred_gap_reasons']) for r in records),gap_cells=0)
    return dict(schema='peer239_mature_source_report_v1',status='ADMITTED',admitted=True,events=events,records=records,gaps=[],counts=counts,
        source_only=True,original_inventory_preserved=True,strict_pit=False,momentum_computed=False,rankings_computed=False,orders_computed=False,
        returns_computed=False,model_fits=0,original_failed_audit_preserved=True)


def fixture():
    axis=pd.bdate_range('2015-01-01','2017-03-03');calendar=[str(d.date()) for d in axis]
    dates=['2016-08-17','2016-08-31','2016-09-30','2016-10-31'];names=['AAA','BBB','CCC','DDD','QQQ']
    meta=dict(dates=dates,symbols=[names.copy() for _ in dates],etfs=['QQQ'])
    frame=pd.DataFrame(100.,index=axis,columns=names)
    for j,s in enumerate(names[:4]):
        frame[s]=[64.+4*((p//7+3*j)%5) for p in range(len(axis))]
    panel={'adj':frame};arrays={'eligible':np.ones((4,5),dtype=bool)};arrays['eligible'][:,3]=False
    return panel,meta,calendar,arrays,make_report(panel,meta,calendar,arrays)


def warm_fixture():
    axis=pd.bdate_range('2025-01-01','2026-09-03');calendar=[str(d.date()) for d in axis]
    names=['AAA','BBB','CCC','SPCX','QQQ'];meta=dict(dates=['2026-08-31'],symbols=[names],etfs=['QQQ'])
    frame=pd.DataFrame(100.,index=axis,columns=names);frame.loc[frame.index<pd.Timestamp('2026-06-12'),'SPCX']=np.nan
    panel={'adj':frame};arrays={'eligible':np.ones((1,5),dtype=bool)}
    report=make_report(panel,meta,calendar,arrays,sectors={'SPCX':None},issuers={'SPCX':k.prior.BOUNDS['SPCX']['issuer_key']})
    return panel,meta,calendar,arrays,report


def physical(args,n):
    panel,meta,calendar,arrays,report=args
    mm=deepcopy(meta);mm['dates']=mm['dates'][:n];mm['symbols']=mm['symbols'][:n]
    cc=calendar[:calendar.index(mm['dates'][-1])+2] if n else []
    pp={'adj':panel['adj'].loc[:pd.Timestamp(mm['dates'][-1])].copy() if n else panel['adj'].iloc[:0].copy()}
    rr=deepcopy(report);rr['events']=rr['events'][:n];rr['records']=[r for r in rr['records'] if r['query_index']<n]
    return pp,mm,cc,{'eligible':arrays['eligible'][:n].copy()},rr


def run_source(args,limit=None):return k.build_source(*args,limit=limit)

def run(args,limit=None):
    src=run_source(args,limit)
    return k.build_features(*args[:4],src,limit=limit)


def same_values(a,b):
    assert tuple(a)==tuple(b)
    for key in a:np.testing.assert_array_equal(a[key],b[key])


class Tests(unittest.TestCase):
    def test_source_feature_schemas_and_counts(self):
        args=fixture();src=run_source(args);v,a=run(args)
        self.assertEqual(tuple(src),k.SOURCE_KEYS);self.assertEqual(tuple(src['counts']),k.COUNT_FIELDS)
        self.assertEqual(src['counts'],dict(events=4,stock_cells=16,mature_cells=12,supported_cells=12,price_observations=2532,new_model_fits=0))
        self.assertEqual(a['counts'],src['counts']);self.assertEqual(tuple(v),k.VALUE_KEYS)
        self.assertEqual(tuple(a['events'][1]),k.EVENT_KEYS);self.assertEqual(tuple(a['events'][1]['cells'][0]),k.CELL_KEYS)
        self.assertEqual(len(src['events'][1]['path_dates']),211);self.assertEqual(len(src['records'][4]['path_observations']),211)
        self.assertTrue(all(tuple(o)==k.OBSERVATION_KEYS for o in src['records'][4]['path_observations']))
        json.dumps(src,allow_nan=False);json.dumps(a,allow_nan=False)

    def test_exact_two_peer_daily_reset_vs_endpoint(self):
        args=list(physical(fixture(),2));frame=args[0]['adj'];frame.loc[:,:]=100.
        t=args[2].index(args[1]['dates'][1]);day=pd.Timestamp(args[2][t-251])
        frame.at[day,'BBB']=200.;frame.at[day,'CCC']=50.
        args[4]=make_report(*args[:4],sectors={'DDD':'XLV'})
        v,a=run(args);c=a['events'][1]['cells'][0]
        self.assertEqual(pair(c['daily_peer_gross']),Fraction(25,16));self.assertEqual(pair(c['endpoint_peer_gross']),1)
        self.assertEqual(pair(c['path_raw']),Fraction(-9,25));self.assertEqual(pair(c['endpoint_raw']),0)
        self.assertFalse(v['supported'][1,3]);self.assertTrue(v['mature'][1,3]);self.assertIsNone(a['events'][1]['cells'][3]['path_raw'])

    def test_odd_even_median_and_exact_raw_rank(self):
        self.assertEqual(k._median([Fraction(2),Fraction(8),Fraction(4)]),4)
        self.assertEqual(k._median([Fraction(2),Fraction(8)]),5)
        big=Fraction(10**400);vals=[big,big+1,big+2]
        self.assertEqual(k._rank(big,vals),float(Fraction(-1,6)));self.assertEqual(k._rank(big+1,vals),float(Fraction(1,6)))
        self.assertEqual(k._rank(big+2,vals),.5)

    def test_all_ties_zero_raw_is_available_nonzero_rank(self):
        args=fixture();args[0]['adj'].loc[:,:]=100.
        v,a=run(args)
        np.testing.assert_array_equal(v['path_target'][1:,:4],np.full((3,4),.125))
        for ev in a['events'][1:]:
            for c in ev['cells']:
                self.assertEqual(pair(c['path_raw']),0);self.assertEqual(pair(c['endpoint_raw']),0)
                self.assertEqual(pair(c['daily_peer_gross']),1)

    def test_identical_peer_paths_references_equal(self):
        args=fixture()
        for s in ('BBB','CCC','DDD'):args[0]['adj'][s]=args[0]['adj']['AAA']
        v,a=run(args)
        for ev in a['events'][1:]:
            for c in ev['cells']:self.assertEqual(c['daily_peer_gross'],c['endpoint_peer_gross'])
        np.testing.assert_array_equal(v['path_target'],v['endpoint_target'])

    def test_representable_per_security_rescaling(self):
        args=fixture();v,a=run(args)
        for s,scale in zip(args[0]['adj'].columns,(2.,4.,.5,.25,8.)):args[0]['adj'][s]*=scale
        vv,aa=run(args);same_values(v,vv);self.assertEqual(a,aa)

    def test_security_column_permutation_preserves_named_evidence(self):
        args=fixture();v,a=run(args);perm=[3,1,4,0,2]
        args[1]['symbols'][:]=[[ss[j] for j in perm] for ss in args[1]['symbols']]
        args[3]['eligible']=args[3]['eligible'][:,perm];args=list(args);args[4]=make_report(*args[:4])
        vv,aa=run(args)
        for key in k.VALUE_KEYS:np.testing.assert_array_equal(vv[key],v[key][:,perm])
        for first,second in zip(a['events'],aa['events']):
            by={c['symbol']:c for c in first['cells']}
            for c in second['cells']:
                for key in k.CELL_KEYS[7:]:self.assertEqual(c[key],by[c['symbol']][key])

    def test_Efalse_peer_and_eligibility_toggle(self):
        args=fixture();src=run_source(args);v,a=run(args);args[3]['eligible']=~args[3]['eligible']
        self.assertEqual(src,run_source(args));vv,aa=run(args);same_values(v,vv);self.assertEqual(a,aa)
        self.assertIn(3,a['events'][1]['cells'][0]['peer_indices'])

    def test_ETF_and_partial_never_consume_numeric_prices(self):
        args=fixture();args[0]['adj']=args[0]['adj'].astype(object);args[0]['adj']['QQQ']='poison'
        v,a=run(args);self.assertFalse(v['mature'][0].any());self.assertTrue(np.isnan(v['path_target'][:,4]).all())
        first=list(physical(args,1));first[0]['adj']=first[0]['adj'].iloc[:0]
        vv,aa=run(first);same_values(vv,{key:value[:1] for key,value in v.items()});self.assertEqual(aa['events'],a['events'][:1])

    def test_no_supported_domain_zero_denominator_not_evaluated(self):
        args=list(fixture());args[4]=make_report(*args[:4],sectors={'CCC':'XLV','DDD':'XLE'})
        with mock.patch.object(k,'_rank',side_effect=AssertionError('empty rank must not run')):
            v,a=run(args)
        self.assertEqual(a['counts']['mature_cells'],12);self.assertEqual(a['counts']['supported_cells'],0)
        self.assertTrue(np.isnan(v['path_target']).all())

    def test_all_logical_physical_prefixes_immutability(self):
        args=fixture();before=deepcopy(args);src=run_source(args);v,a=run(args)
        for n in range(5):
            ss=run_source(args,n);physical_src=run_source(physical(args,n));self.assertEqual(ss,physical_src)
            vv,aa=k.build_features(*args[:4],src,limit=n);pv,pa=run(physical(args,n))
            same_values(vv,pv);same_values(vv,{key:x[:n] for key,x in v.items()});self.assertEqual(aa,pa);self.assertEqual(aa['events'],a['events'][:n])
        pd.testing.assert_frame_equal(args[0]['adj'],before[0]['adj']);self.assertEqual(args[1:3],before[1:3]);np.testing.assert_array_equal(args[3]['eligible'],before[3]['eligible']);self.assertEqual(args[4],before[4])

    def test_empty_prefix_width_and_no_feature_arithmetic(self):
        args=physical(fixture(),0)
        with mock.patch.object(k,'_median',side_effect=AssertionError('empty math')):
            v,a=run(args)
        self.assertTrue(all(x.shape==(0,5) for x in v.values()));self.assertEqual(a['events'],[])
        self.assertEqual(a['counts']['price_observations'],0)

    def test_future_source_meta_price_E_not_consumed(self):
        args=fixture();src=run_source(args);v,a=k.build_features(*args[:4],src,limit=2)
        cutoff=pd.Timestamp(args[1]['dates'][1]);args[0]['adj']=args[0]['adj'].astype(object);args[0]['adj'].loc[args[0]['adj'].index>cutoff,:]='bad'
        args[1]['dates'][2]='futurebad';args[1]['symbols'][2]=['bad'];args[3]['eligible']=args[3]['eligible'].astype(object);args[3]['eligible'][2:,:]='bad'
        src['events'][2]={'bad':True}
        for r in src['records']:
            if r['query_index']>=2:r.clear();r.update(query_index=2,bad=True)
        vv,aa=k.build_features(*args[:4],src,limit=2);same_values(v,vv);self.assertEqual(a,aa)

    def test_recent42_ignored_by_new_factor(self):
        args=physical(fixture(),2);v,a=run(args);t=args[2].index(args[1]['dates'][-1]);cutoff=pd.Timestamp(args[2][t-42])
        args[0]['adj']=args[0]['adj'].astype(object);args[0]['adj'].loc[args[0]['adj'].index>cutoff,:]='poison'
        vv,aa=run(args);same_values(v,vv);self.assertEqual(a,aa)

    def test_middle_missing_on_ineligible_or_unsupported_stock_aborts(self):
        for sectors in ({},{'DDD':'XLV'}):
            args=list(physical(fixture(),2));args[4]=make_report(*args[:4],sectors=sectors)
            t=args[2].index(args[1]['dates'][1]);args[0]['adj'].at[pd.Timestamp(args[2][t-100]),'DDD']=np.nan
            with self.assertRaisesRegex(ValueError,'mature_path_missing'):run_source(args)

    def test_missing_row_column_and_all_invalid_source_taxonomy(self):
        bad=[(None,'mature_path_missing'),(pd.NA,'mature_path_missing'),(np.nan,'mature_path_missing'),
             (True,'invalid_boolean'),(np.bool_(False),'invalid_boolean'),('100','invalid_nonreal'),
             (1+0j,'invalid_nonreal'),(0.,'nonpositive'),(-1.,'nonpositive'),(float('inf'),'nonfinite'),
             (float('-inf'),'nonfinite'),(10**400,'invalid_numeric_conversion'),(Fraction(1,10**1000),'nonpositive')]
        for value,reason in bad:
            args=physical(fixture(),2);t=args[2].index(args[1]['dates'][1]);day=pd.Timestamp(args[2][t-100]);args[0]['adj']=args[0]['adj'].astype(object);args[0]['adj'].at[day,'AAA']=value
            with self.subTest(value=repr(value)),self.assertRaisesRegex(ValueError,reason):run_source(args)
        for mode in ('row','column'):
            args=physical(fixture(),2);t=args[2].index(args[1]['dates'][1]);day=pd.Timestamp(args[2][t-100])
            args[0]['adj']=args[0]['adj'].drop(index=day) if mode=='row' else args[0]['adj'].drop(columns='AAA')
            with self.assertRaisesRegex(ValueError,'missing_'+mode):run_source(args)

    def test_exact_huge_raw_need_not_fit_float(self):
        args=list(physical(fixture(),2));args[0]['adj'].loc[:,:]=1.
        t=args[2].index(args[1]['dates'][1]);start=pd.Timestamp(args[2][t-252]);end=pd.Timestamp(args[2][t-42])
        args[0]['adj'].at[start,'AAA']=1e-300;args[0]['adj'].at[end,'AAA']=1e308
        v,a=run(args);raw=pair(a['events'][1]['cells'][0]['endpoint_raw'])
        self.assertGreater(raw,10**600);self.assertTrue(np.isfinite(v['path_target'][1,:4]).all());json.dumps(a,allow_nan=False)

    def test_warmup_all211_inspected_including_missing_not_recast(self):
        args=warm_fixture();src=run_source(args);v,a=run(args)
        self.assertEqual(src['counts']['price_observations'],844);self.assertEqual(src['counts']['mature_cells'],3)
        record=src['records'][3];self.assertEqual(len(record['path_observations']),211)
        missing=[x for x in record['path_observations'] if x['status']=='missing'];self.assertTrue(missing)
        self.assertTrue(all(x['date']<'2026-06-12' for x in missing));self.assertFalse(v['mature'][0,3]);self.assertNotIn(3,a['events'][0]['cells'][0]['peer_indices'])

    def test_prebound_positive_internal_quote_hard_failure(self):
        args=warm_fixture();t=args[2].index(args[1]['dates'][0]);day=pd.Timestamp(args[2][t-100]);self.assertLess(str(day.date()),'2026-06-08')
        args[0]['adj'].at[day,'SPCX']=1.
        with self.assertRaisesRegex(ValueError,'positive_prebound_identity_conflict'):run_source(args)

    def test_warmup_invalid_not_hidden_and_postbound_missing_aborts(self):
        for value in (True,'bad',0.,-1.,float('inf')):
            args=warm_fixture();t=args[2].index(args[1]['dates'][0]);day=pd.Timestamp(args[2][t-100]);args[0]['adj']=args[0]['adj'].astype(object);args[0]['adj'].at[day,'SPCX']=value
            with self.subTest(value=repr(value)),self.assertRaises(ValueError):run_source(args)
        args=warm_fixture();args[0]['adj'].at[pd.Timestamp('2026-06-12'),'SPCX']=np.nan
        with self.assertRaisesRegex(ValueError,'post_bound_missing'):run_source(args)

    def test_prebound_missing_row_can_be_deferred_when_no_mature_column_uses_it(self):
        args=warm_fixture();args=list(args);args[1]['symbols']=[['SPCX','QQQ']];args[3]={'eligible':np.ones((1,2),dtype=bool)}
        args[4]=make_report(*args[:4],sectors={'SPCX':None},issuers={'SPCX':k.prior.BOUNDS['SPCX']['issuer_key']})
        t=args[2].index(args[1]['dates'][0]);day=pd.Timestamp(args[2][t-100]);args[0]['adj']=args[0]['adj'].drop(index=day)
        src=run_source(args);self.assertIn(dict(date=str(day.date()),status='missing',reason='missing_row'),src['records'][0]['path_observations'])

    def test_listing_bound_identity_available_date_and_equality(self):
        for field,value in [('available_from','2027-01-01'),('issuer_key','other'),('not_before','2026-07-01')]:
            args=warm_fixture();args[4]['records'][3]['history_bound'][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):run_source(args)
        args=warm_fixture();args[0]['adj'].at[pd.Timestamp('2026-06-12'),'SPCX']=123.;run_source(args)
        args[0]['adj'].at[pd.Timestamp('2026-06-12'),'SPCX']=np.nan
        with self.assertRaisesRegex(ValueError,'post_bound_missing'):run_source(args)

    def test_duplicate_issuer_missing_sector_and_missing_inventory(self):
        for mode in ('duplicate','issuer_missing','sector','missing_record','extraETF'):
            args=fixture();r=args[4]['records'][4]
            if mode=='duplicate':r['issuer_key']=args[4]['records'][5]['issuer_key']
            elif mode=='issuer_missing':r['issuer_key']=None
            elif mode=='sector':r['sector']=None
            elif mode=='missing_record':args[4]['records'].pop(4)
            else:r=deepcopy(r);r['column_index']=4;r['symbol']='QQQ';args[4]['records'].insert(8,r)
            with self.subTest(mode=mode),self.assertRaises(ValueError):run_source(args)

    def test_peer_tamper_E_filter_self_and_cohort_drop_rejected(self):
        for mode in ('drop','self','reverse','count','support','boolindex'):
            args=fixture();r=args[4]['records'][4]
            if mode=='drop':r['peer_indices'].pop()
            elif mode=='self':r['peer_indices'].append(0)
            elif mode=='reverse':r['peer_indices'].reverse()
            elif mode=='count':r['peer_issuer_count']=2
            elif mode=='support':r['structural_peer_support']=False
            else:r['peer_indices'][0]=True
            with self.subTest(mode=mode),self.assertRaises(ValueError):run_source(args)

    def test_daily_report_proof_tampering_rejected(self):
        changes=[lambda s:s.__setitem__('daily_paths_checked',False),lambda s:s['counts'].__setitem__('price_observations',0),
            lambda s:s['counts'].__setitem__('new_model_fits',False),lambda s:s['events'][1]['path_dates'].__setitem__(100,'2000-01-01'),
            lambda s:s['records'][4]['path_observations'][100].__setitem__('status','missing'),
            lambda s:s['records'][4]['path_observations'][100].__setitem__('reason','missing_value'),
            lambda s:s['records'][4]['path_observations'].pop(),lambda s:s.__setitem__('admitted',False)]
        for change in changes:
            args=fixture();src=run_source(args);change(src)
            with self.assertRaises(ValueError):k.build_features(*args[:4],src)

    def test_old_endpoint_report_cannot_be_passed_as_daily_admission(self):
        args=fixture()
        with self.assertRaisesRegex(ValueError,'daily source report'):k.build_features(*args)
        with mock.patch.object(k.prior,'build_features',side_effect=AssertionError('old feature forbidden')),mock.patch.object(k.prior,'_momentum',side_effect=AssertionError('old endpoint feature forbidden')):
            run(args)

    def test_canonical_axes_E_type_and_regular_monthend(self):
        for mode in ('date','midmonth','duplicatesymbol','ETFduplicate','Eint','calendarrepeat','priceaxisreverse','priceaxisduplicate','pricezone'):
            args=fixture()
            if mode=='date':args[1]['dates'][1]='2016-8-31'
            elif mode=='midmonth':args[1]['dates'][1]='2016-08-30'
            elif mode=='duplicatesymbol':args[1]['symbols'][1][1]='AAA'
            elif mode=='ETFduplicate':args[1]['etfs'].append('QQQ')
            elif mode=='Eint':args[3]['eligible']=args[3]['eligible'].astype(int)
            elif mode=='calendarrepeat':args[2].insert(0,args[2][0])
            elif mode=='priceaxisreverse':args[0]['adj']=args[0]['adj'].iloc[::-1]
            elif mode=='priceaxisduplicate':args[0]['adj']=pd.concat([args[0]['adj'].iloc[:1],args[0]['adj']])
            else:args[0]['adj']=args[0]['adj'].tz_localize('UTC')
            with self.subTest(mode=mode),self.assertRaises((ValueError,TypeError)):run_source(args)
        for n in (-1,True,5,1.):
            with self.assertRaises(ValueError):run_source(fixture(),n)

    def test_source_builder_has_no_feature_math(self):
        with mock.patch.object(k,'_rank',side_effect=AssertionError('rank')),mock.patch.object(k,'_median',side_effect=AssertionError('median')),mock.patch.object(k,'Fraction',side_effect=AssertionError('ratio')):
            report=run_source(fixture())
        self.assertFalse(report['momentum_computed']);self.assertFalse(report['rankings_computed']);self.assertEqual(report['model_fits'],0)

    def test_missing_reference_endpoint_and_full_path_both_fail(self):
        for offset in (252,100,42):
            args=physical(fixture(),2);t=args[2].index(args[1]['dates'][1]);args[0]['adj'].at[pd.Timestamp(args[2][t-offset]),'AAA']=np.nan
            with self.subTest(offset=offset),self.assertRaisesRegex(ValueError,'mature_path_missing'):run_source(args)

    def test_one_regular_query_without_initial_partial_runs(self):
        args=list(physical(fixture(),2));args[1]['dates']=args[1]['dates'][1:];args[1]['symbols']=args[1]['symbols'][1:];args[3]['eligible']=args[3]['eligible'][1:]
        args[4]=make_report(*args[:4]);v,a=run(args);self.assertTrue(v['mature'][0,:4].all());self.assertEqual(a['counts']['price_observations'],844)


    def test_v2_actual_trade_bound_preserves_all_original_metadata(self):
        args=warm_fixture();before=deepcopy(args[4]);src=run_source(args)
        rec=src['records'][3]
        self.assertEqual(rec['history_bound'],k.LISTING_POLICY['original_history_bound'])
        self.assertEqual(rec['path_history_bound'],k.LISTING_POLICY['path_bound_override'])
        self.assertEqual(rec['history_bound']['not_before'],'2026-06-08')
        self.assertEqual(rec['path_history_bound']['not_before'],'2026-06-12')
        legacy=k._legacy_view(src,1)
        self.assertEqual(legacy['events'],before['events']);self.assertEqual(legacy['records'],before['records'])
        self.assertEqual(args[4],before)
        self.assertFalse(rec['structural_peer_support']);self.assertEqual(rec['history_status'],'official_pre_inception')
        for old,new in zip(args[4]['records'],src['records']):
            for key,value in old.items():self.assertEqual(new[key],value)

    def test_actual_confirmation_publication_and_next_day_clock(self):
        rec=deepcopy(warm_fixture()[4]['records'][3]);old=deepcopy(rec['history_bound'])
        for day in ('2026-06-30','2026-07-31','2026-08-04'):
            rec['date']=day;self.assertEqual(k._path_history_bound(rec),old)
        for day in ('2026-08-05','2026-08-31'):
            rec['date']=day;self.assertEqual(k._path_history_bound(rec),k.LISTING_POLICY['path_bound_override'])
        self.assertEqual(rec['history_bound'],old)

    def test_old_issuer_or_other_symbol_never_receives_override(self):
        original=deepcopy(warm_fixture()[4]['records'][3])
        for field,value in [('issuer_key','issuer:old_SPCX_fund'),('symbol','OTHER')]:
            r=deepcopy(original);r[field]=value
            self.assertEqual(k._path_history_bound(r),r['history_bound'])
        args=list(warm_fixture());args[4]=make_report(*args[:4],issuers={'SPCX':'issuer:old_SPCX_fund'})
        with self.assertRaisesRegex(ValueError,'Ordinary identity/sector gaps cannot disappear'):run_source(args)

    def test_visible_override_four_pretrade_days_are_missing_only(self):
        for day in ('2026-06-08','2026-06-09','2026-06-10','2026-06-11'):
            args=warm_fixture();src=run_source(args)
            self.assertIn(dict(date=day,status='missing',reason='missing_value'),src['records'][3]['path_observations'])
            args[0]['adj'].at[pd.Timestamp(day),'SPCX']=1.
            with self.assertRaisesRegex(ValueError,'positive_prebound_identity_conflict'):run_source(args)

    def test_new_bound_tamper_cache_rejected_before_exact_feature_math(self):
        for field,value in [('not_before','2026-06-13'),('available_from','2026-06-09'),('issuer_key','old')]:
            args=warm_fixture();src=run_source(args);src['records'][3]['path_history_bound'][field]=value
            with mock.patch.object(k,'_median',side_effect=AssertionError('feature math before proof')):
                with self.assertRaisesRegex(ValueError,'Evidence differs'):k.build_features(*args[:4],src)
        args=warm_fixture();src=run_source(args);del src['records'][3]['path_history_bound']
        with self.assertRaisesRegex(ValueError,'record keys'):k.build_features(*args[:4],src)

    def test_bound_override_never_creates_mature_partial_or_eligible_peer(self):
        args=warm_fixture();src=run_source(args);v,a=run(args)
        self.assertFalse(v['mature'][0,3]);self.assertFalse(v['supported'][0,3]);self.assertTrue(np.isnan(v['path_target'][0,3]))
        self.assertEqual(src['events'][0]['mature_indices'],args[4]['events'][0]['mature_indices'])
        partial=run_source(physical(fixture(),1))
        self.assertTrue(all(r['path_history_bound'] is None for r in partial['records']))
        self.assertEqual(partial['counts']['price_observations'],0)

    def test_legacy_mature_with_new_pretrade_missing_is_source_hard_failure(self):
        args=list(warm_fixture());calendar=args[2];t=calendar.index(args[1]['dates'][0])
        start=t-252;self.assertLess(calendar[start],'2026-06-08')
        # Isolate the future-contract edge without changing frozen239 validator:
        # even if validated metadata called it mature, no None enters Fraction.
        real=k._metadata(*args,None)
        frame,sessions,rows,width=real
        row=list(rows[0]);row[6]=[0,1,2,3]
        with mock.patch.object(k,'_metadata',return_value=(frame,sessions,[tuple(row)],width)):
            with self.assertRaisesRegex(ValueError,'mature_path_bound_conflict'):run_source(args)

    def test_v2_warm_all_logical_physical_prefixes_and_cache(self):
        args=list(warm_fixture());args[1]['dates']=['2026-06-30','2026-07-31','2026-08-31']
        args[1]['symbols']=[args[1]['symbols'][0].copy() for _ in range(3)]
        args[3]={'eligible':np.ones((3,5),dtype=bool)}
        args[4]=make_report(*args[:4],sectors={'SPCX':None},issuers={'SPCX':k.prior.BOUNDS['SPCX']['issuer_key']})
        src=run_source(args);v,a=run(args)
        for i in range(3):
            rec=src['records'][4*i+3]
            expected=k.LISTING_POLICY['path_bound_override'] if i==2 else rec['history_bound']
            self.assertEqual(rec['path_history_bound'],expected)
        for n in range(4):
            ss=run_source(args,n);pp=run_source(physical(args,n));self.assertEqual(ss,pp)
            vv,aa=k.build_features(*args[:4],src,limit=n);pv,pa=run(physical(args,n))
            same_values(vv,pv);self.assertEqual(aa,pa);self.assertEqual(aa['events'],a['events'][:n])


if __name__=='__main__':unittest.main(verbosity=2)
