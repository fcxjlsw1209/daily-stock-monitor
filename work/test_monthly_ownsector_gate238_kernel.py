"""Synthetic-only238 gate tests. make_report is a fixture, never source admission."""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import monthly_ownsector_gate238_kernel as k


def point(role, symbol, day):
    return dict(role=role, symbol=symbol, date=day, status='valid', reason=None)


def make_report(panel, meta, calendar, arrays, benchmark_by_symbol=None):
    """Construct declared-admitted SYNTHETIC metadata without reading price values.

    The caller must supply synthetic frames. Default stock benchmark is XLK;
    explicit ETF names always use SPY. VOX assigns original-XLC repair metadata.
    Aggregate counts describe this fixture and may remain full in prefix tests.
    """
    del panel
    mapping = benchmark_by_symbol or {}
    events, records = [], []
    width = arrays['eligible'].shape[1]
    for i, (d, names) in enumerate(zip(meta['dates'], meta['symbols'])):
        partial = d == k.INITIAL_PARTIAL
        pos = calendar.index(d)
        ids = np.flatnonzero(arrays['eligible'][i]).tolist()
        required = [] if partial else ids.copy()
        positions = [] if partial else [pos - 252, pos - 42]
        dates = [calendar[q] for q in positions]
        market = [point(role, 'SPY', dd) for role, dd in zip(('market_start', 'market_end'), dates)]
        events.append(dict(query_index=i, date=d, initial_partial=partial, active=not partial,
            original_eligible_indices=ids, required_indices=required, source_positions=positions,
            source_dates=dates, market_endpoints=market, required_count=len(required),
            admitted_count=len(required), gap_count=0))
        for j in required:
            s = names[j]; etf = s in meta['etfs']; b = 'SPY' if etf else mapping.get(s, 'XLK')
            old = 'XLC' if b == 'VOX' else b
            origin = 'original_META.etfs_SPY' if etf else 'query_date_sector_panel'
            repair = 'full_period_communication_proxy' if old == 'XLC' else 'unchanged'
            source = origin if repair == 'unchanged' else repair
            points = [point(role, sym, dd) for role, sym, dd in (
                ('security_start', s, dates[0]), ('security_end', s, dates[1]),
                ('benchmark_start', b, dates[0]), ('benchmark_end', b, dates[1]))]
            records.append(dict(query_index=i, column_index=j, date=d, symbol=s, is_etf=etf,
                benchmark=b, mapping_source=source, mapping_status='available', endpoints=points,
                admitted=True, gap_reasons=[], original_benchmark=old, mapping_repair=repair,
                mapping_evidence=None, original_mapping_source=origin))
    r = len(records); active = sum(e['active'] for e in events); funds = sum(r['is_etf'] for r in records)
    counts = dict(events=len(events), symbols_per_event=width, checked_cells=len(events)*width,
        initial_partial_events=sum(e['initial_partial'] for e in events), regular_events=active,
        original_eligible_cells=sum(len(e['original_eligible_indices']) for e in events), required_cells=r,
        stock_required_cells=r-funds, etf_required_cells=funds, mapping_available_cells=r,
        mapping_missing_cells=0, security_benchmark_endpoint_checks=4*r, market_endpoint_checks=2*active,
        valid_endpoints=4*r+2*active, missing_endpoints=0, invalid_endpoints=0,
        admitted_cells=r, gap_cells=0, market_gap_events=0)
    return dict(schema='ownsector238_repaired_source_report_v1', status='ADMITTED', admitted=True,
        counts=counts, events=events, records=records, gaps=[],
        repair_counts={kind: sum(r['mapping_repair'] == kind for r in records) for kind in
            ('unchanged', 'full_period_communication_proxy', 'dated_business_proxy', 'unresolved')},
        source_only=True, momentum_computed=False, eligibility_changed=False, rankings_computed=False,
        orders_computed=False, returns_computed=False, model_fits=0, strict_pit=False)


def endpoints(meta, calendar, i):
    pos = calendar.index(meta['dates'][i])
    return pd.Timestamp(calendar[pos-252]), pd.Timestamp(calendar[pos-42])


def fixture():
    days = pd.bdate_range('2015-01-01', '2016-11-10')
    names = ['AAA', 'BBB', 'COMM', 'FUND', 'OFF']
    panel = {'adj': pd.DataFrame(np.nan, index=days, columns=names+['SPY', 'XLP', 'XLK', 'XLC']),
             'vox': pd.DataFrame({'VOX': np.nan}, index=days)}
    meta = dict(dates=['2016-08-17', '2016-08-31', '2016-09-30', '2016-10-31'],
                symbols=[names.copy() for _ in range(4)], etfs=['FUND'])
    arrays = dict(eligible=np.array([[True, True, True, True, False]]*4),
                  state_weak_narrow=np.array([True, True, False, True]))
    calendar = [str(d.date()) for d in days]
    for i, ends, vox in ((1, [125.,150.,150.,137.5,137.5,112.5,175.],125.),
                         (2, [125.,150.,150.,137.5,90.,90.,100.],90.),
                         (3, [125.,150.,112.5,125.,112.5,125.,125.],125.)):
        start, end = endpoints(meta, calendar, i)
        panel['adj'].loc[start, :] = 100.
        panel['adj'].loc[end, ['AAA','BBB','COMM','FUND','SPY','XLP','XLK']] = ends
        panel['vox'].at[start, 'VOX'] = 100.
        panel['vox'].at[end, 'VOX'] = vox
    report = make_report(panel, meta, calendar, arrays, {'AAA':'XLP','BBB':'XLK','COMM':'VOX'})
    return panel, meta, calendar, arrays, report


def run(data, limit=None):
    return k.build_features(*data, limit=limit)


def physical(data, n, crop_report=True):
    panel, meta, calendar, arrays, report = data
    mm = deepcopy(meta); mm['dates'], mm['symbols'] = mm['dates'][:n], mm['symbols'][:n]
    aa = {key: value[:n].copy() for key, value in arrays.items()}
    cutoff = mm['dates'][-1] if n else None
    pp = {name: frame.loc[:cutoff].copy(deep=True) if n else frame.iloc[:0].copy(deep=True)
          for name, frame in panel.items()}
    cc = calendar[:calendar.index(cutoff)+2] if n else []
    rr = deepcopy(report)
    if crop_report:
        rr['events'] = rr['events'][:n]
        rr['records'] = [r for r in rr['records'] if r['query_index'] < n]
    return pp, mm, cc, aa, rr


class GateTests(unittest.TestCase):
    def assert_values(self, left, right):
        self.assertEqual(set(left), set(right))
        for name in left:
            np.testing.assert_array_equal(left[name], right[name], err_msg=name)

    def test_known_values_counts_and_exact_schema(self):
        v, audit = run(fixture())
        self.assertEqual(tuple(v), k.VALUE_KEYS)
        np.testing.assert_array_equal(v['candidate_eligible'], [[1,1,1,1,0],[1,0,1,0,0],[1,1,1,1,0],[0,1,0,1,0]])
        np.testing.assert_array_equal(v['benchmark_momentum'][1,:4], [.125,.75,.25,.375])
        np.testing.assert_array_equal(v['momentum'][1,:4], [.25,.5,.5,.375])
        expected = [4,5,20,16,3,3,12,9,3,24,24,6,4,12,0,0]
        self.assertEqual([audit[key] for key in k.COUNT_FIELDS], expected)
        self.assertEqual(set(audit), set(k.COUNT_FIELDS) | {'schema','start_offset','end_offset','return_count',
            'rankings_recomputed','source_missing_filled','portfolio_returns_calculated','event_history','cell_history'})
        self.assertEqual(len(audit['event_history'][0]), 17)
        self.assertEqual(len(audit['cell_history'][0]), 17)
        self.assertEqual([x['branch'] for x in audit['event_history']], ['initial_partial']+['ownsector_relative_gate']*3)
        json.dumps(audit, allow_nan=False)

    def test_single_sector_gate_in_both_directions_not_spy_intersection(self):
        data = fixture(); panel, meta, calendar, _, _ = data
        v, _ = run(data)
        self.assertTrue(v['candidate_eligible'][1,0]) # .25 beats sector .125, trails SPY .375
        self.assertFalse(v['candidate_eligible'][1,1]) # .5 beats SPY .375, trails sector .75
        _, end = endpoints(meta, calendar, 1)
        panel['adj'].at[end,'SPY'] = 10000.
        changed, _ = run(data)
        for key in k.VALUE_KEYS:
            np.testing.assert_array_equal(changed[key][:,:3], v[key][:,:3])

    def test_exact_ties_adjacent_float_and_no_epsilon(self):
        data=fixture();panel,meta,calendar,_,_=data;_,end=endpoints(meta,calendar,1)
        panel['adj'].at[end,'XLP']=125.
        for value, keep in ((125.,False),(np.nextafter(125.,np.inf),True),(np.nextafter(125.,0.),False)):
            panel['adj'].at[end,'AAA']=value
            v,_=run(data);self.assertEqual(v['candidate_eligible'][1,0],keep)

    def test_nonpositive_assigned_benchmark_preserves_positive_original_E(self):
        data=fixture();v,_=run(data)
        np.testing.assert_array_equal(v['candidate_eligible'][2],data[3]['eligible'][2])
        np.testing.assert_array_equal(v['threshold'][2,:4],np.zeros(4))

    def test_required_set_subset_and_numeric_nan_support(self):
        v,_=run(fixture())
        expected=v['original_eligible'].copy();expected[0]=False
        np.testing.assert_array_equal(v['required'],expected)
        self.assertFalse((v['candidate_eligible'] & ~v['original_eligible']).any())
        for key in ('momentum','benchmark_momentum','threshold'):
            self.assertEqual(v[key].dtype,np.float64);self.assertEqual(v[key].shape,(4,5))
            np.testing.assert_array_equal(np.isfinite(v[key]),expected)

    def test_etf_identity_is_explicit_and_mismatch_is_rejected(self):
        data=fixture();data[1]['etfs']=['AAA']
        with self.assertRaises(ValueError):run(data)
        data=fixture();data[1]['etfs']=['FUND','FUND']
        with self.assertRaises(ValueError):run(data)
        data=fixture();del data[1]['etfs']
        with self.assertRaises(ValueError):run(data)

    def test_etf_spy_equality_rejects_even_etf_named_spy(self):
        data=fixture();panel,meta,calendar,arrays,_=data
        for row in meta['symbols']:row[3]='SPY'
        meta['etfs']=['SPY'];report=make_report(panel,meta,calendar,arrays)
        # Later fixture SPY is negative and cannot be original-E eligible.
        v,_=run((panel,meta,calendar,arrays,report),2)
        self.assertFalse(v['candidate_eligible'][1,3]);self.assertTrue(v['candidate_eligible'][0,3])

    def test_dedicated_vox_never_uses_xlc_or_adj_vox(self):
        data=fixture();panel,meta,calendar,_,_=data;v,_=run(data)
        panel['adj']['XLC']=np.inf;panel['adj']['VOX']=999999.
        unchanged,_=run(data);self.assert_values(v,unchanged)
        _,end=endpoints(meta,calendar,1);panel['vox'].at[end,'VOX']=175.
        changed,a=run(data);self.assertFalse(changed['candidate_eligible'][1,2])
        cell=[c for c in a['cell_history'] if c['query_index']==1 and c['symbol']=='COMM'][0]
        self.assertEqual(cell['benchmark_source'],'vox');self.assertEqual(cell['original_benchmark'],'XLC')

    def test_empty_E_and_stock_only_rows_do_not_consume_spy_prices(self):
        data=fixture();panel,meta,calendar,arrays,_=data
        arrays['eligible'][1:,3]=False
        report=make_report(panel,meta,calendar,arrays)
        panel['adj']=panel['adj'].drop(columns=['SPY'])
        v,a=run((panel,meta,calendar,arrays,report))
        self.assertEqual(a['spy_price_observations'],0);self.assertEqual(a['etf_required_cells'],0)
        arrays['eligible'][1:]=False;report=make_report(panel,meta,calendar,arrays)
        for frame in panel.values():frame.iloc[:,:]=np.nan
        v,a=run((panel,meta,calendar,arrays,report))
        self.assertEqual(a['active_events'],3);self.assertEqual(a['required_cells'],0)
        for name in k.VALUE_KEYS[3:]:self.assertTrue(np.isnan(v[name]).all())

    def test_named_partial_no_prices_no_formation_history(self):
        data=physical(fixture(),1);panel,meta,calendar,arrays,_=data
        calendar=['2016-08-17','2016-08-18']
        panel={key:frame.iloc[:0].copy() for key,frame in panel.items()}
        report=make_report(panel,meta,calendar,arrays)
        with patch.object(k,'_momentum',side_effect=AssertionError('numeric read')):
            v,a=run((panel,meta,calendar,arrays,report))
        np.testing.assert_array_equal(v['candidate_eligible'],arrays['eligible'])
        self.assertEqual(a['cell_history'],[])
        self.assertIsNone(a['event_history'][0]['source_start'])

    def test_regular_first_row_is_not_bypassed(self):
        panel,meta,calendar,arrays,_=fixture()
        meta=dict(meta,dates=meta['dates'][1:2],symbols=meta['symbols'][1:2])
        arrays={key:value[1:2].copy() for key,value in arrays.items()}
        report=make_report(panel,meta,calendar,arrays,{'AAA':'XLP','BBB':'XLK','COMM':'VOX'})
        v,a=run((panel,meta,calendar,arrays,report))
        self.assertEqual(a['active_events'],1);self.assertEqual(a['required_cells'],4)
        np.testing.assert_array_equal(v['candidate_eligible'][0],[1,0,1,0,0])

    def test_weak_state_toggle_only_changes_descriptive_audit(self):
        data=fixture();v,a=run(data);data[3]['state_weak_narrow']=~data[3]['state_weak_narrow']
        vv,aa=run(data);self.assert_values(v,vv)
        a.pop('weak_events');aa.pop('weak_events')
        for x,y in zip(a['event_history'],aa['event_history']):x.pop('weak_narrow');y.pop('weak_narrow')
        self.assertEqual(a,aa)

    def test_positive_power_two_scaling_each_source_independently(self):
        data=fixture();v,_=run(data)
        for j,col in enumerate(data[0]['adj']):data[0]['adj'][col]*=2.**j
        data[0]['vox']['VOX']*=8.
        vv,_=run(data);self.assert_values(v,vv)


class ReportTests(unittest.TestCase):
    def test_unadmitted_scope_fails_before_numeric_access(self):
        cases=[('admitted',False),('status','NOT_ADMITTED'),('gaps',[{}]),('source_only',False),
               ('momentum_computed',True),('model_fits',1),('model_fits',False),('strict_pit',True)]
        for key,value in cases:
            data=fixture();data[4][key]=value
            with self.subTest(key=key),patch.object(k,'_momentum',side_effect=AssertionError('numeric read')):
                with self.assertRaises(ValueError):run(data)

    def test_missing_extra_duplicate_and_reordered_consumed_records_fail(self):
        for mode in ('missing','extra','duplicate','reordered','queryorder'):
            data=fixture();rows=data[4]['records']
            if mode=='missing':rows.pop(0)
            elif mode=='extra':
                x=deepcopy(rows[0]);x['column_index']=4;x['symbol']='OFF';rows.insert(4,x)
            elif mode=='duplicate':rows.insert(1,deepcopy(rows[0]))
            elif mode=='reordered':rows[0],rows[1]=rows[1],rows[0]
            else:rows[0],rows[4]=rows[4],rows[0]
            with self.subTest(mode=mode),self.assertRaises(ValueError):run(data)

    def test_record_identity_dtype_and_endpoint_provenance_fail(self):
        cases=[('query_index',True),('query_index',-1),('column_index',True),('symbol','OTHER'),
               ('date','2016-07-29'),('is_etf',1),('is_etf',True),('mapping_status','missing'),
               ('admitted',False),('gap_reasons',['bad']),('original_mapping_source','invented')]
        for key,value in cases:
            data=fixture();data[4]['records'][0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):run(data)
        for key,value in [('symbol','SPY'),('date','2015-01-02'),('role','market_start'),('status','missing'),('reason','bad')]:
            data=fixture();data[4]['records'][0]['endpoints'][0][key]=value
            with self.subTest(endpointkey=key),self.assertRaises(ValueError):run(data)

    def test_source_event_official_clock_and_required_axes_fail(self):
        changes=[('source_positions',[1,211]),('source_dates',['2015-01-01','2015-10-22']),
                 ('required_indices',[0,1]),('original_eligible_indices',[0,1]),('active',False),
                 ('initial_partial',True),('admitted_count',0),('required_count',True),('gap_count',1),
                 ('date','2016-08-30'),('query_index',True),('market_endpoints',[])]
        for key,value in changes:
            data=fixture();data[4]['events'][1][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):run(data)

    def test_full_xlc_repair_and_nonmissing_mapping_cannot_be_changed(self):
        for key,value in [('benchmark','XLC'),('mapping_repair','unchanged'),('mapping_evidence',[]),
                          ('mapping_source','query_date_sector_panel')]:
            data=fixture();data[4]['records'][2][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):run(data)
        data=fixture();data[4]['records'][0]['benchmark']='VOX'
        with self.assertRaises(ValueError):run(data)
        data=fixture();data[4]['records'][0]['original_benchmark']='SPY'
        with self.assertRaises(ValueError):run(data)

    def test_all_six_dated_business_rule_boundaries_and_evidence(self):
        for symbol,rule in k.DATED_RULES.items():
            benchmark,start,end,evidence=rule
            record=dict(symbol=symbol,is_etf=False,original_benchmark=None,benchmark=benchmark,
                original_mapping_source='query_date_sector_panel',mapping_source='dated_business_proxy',
                mapping_repair='dated_business_proxy',mapping_evidence=evidence.copy(),
                mapping_status='available',admitted=True,gap_reasons=[])
            with self.subTest(symbol=symbol):
                self.assertEqual(k._mapping(record,False,start),benchmark)
                before=str((pd.Timestamp(start)-pd.Timedelta(days=1)).date())
                with self.assertRaises(ValueError):k._mapping(record,False,before)
                if end:
                    with self.assertRaises(ValueError):k._mapping(record,False,end)
                bad=deepcopy(record);bad['mapping_evidence']=['wrong']
                with self.assertRaises(ValueError):k._mapping(bad,False,start)
        record['symbol']='UNKNOWN'
        with self.assertRaises(ValueError):k._mapping(record,False,'2026-01-30')

    def test_dated_business_mapping_is_used_in_full_build(self):
        data=fixture();panel,meta,calendar,arrays,report=data
        for row in meta['symbols']:row[0]='BUD'
        panel['adj']=panel['adj'].rename(columns={'AAA':'BUD'})
        report=make_report(panel,meta,calendar,arrays,{'BUD':'XLP','COMM':'VOX'})
        for row in report['records']:
            if row['symbol']=='BUD':
                row.update(original_benchmark=None,mapping_repair='dated_business_proxy',
                    mapping_source='dated_business_proxy',mapping_evidence=k.DATED_RULES['BUD'][3].copy())
        v,a=run((panel,meta,calendar,arrays,report))
        self.assertTrue(v['candidate_eligible'][1,0])
        self.assertEqual(a['cell_history'][0]['mapping_evidence'],k.DATED_RULES['BUD'][3])

    def test_etf_spy_provenance_has_priority(self):
        for key,value in [('benchmark','XLK'),('original_benchmark','XLC'),('mapping_repair','dated_business_proxy'),
                          ('mapping_source','query_date_sector_panel'),('mapping_evidence',[])]:
            data=fixture();data[4]['records'][3][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):run(data)


class SourceTests(unittest.TestCase):
    def test_all_required_endpoint_roles_invalid_price_hard_fail(self):
        for label,symbol,offset in [('adj','AAA',0),('adj','AAA',1),('adj','XLP',0),('adj','XLP',1),
                                    ('vox','VOX',0),('vox','VOX',1)]:
            for value in [None,pd.NA,True,np.bool_(False),'100',0.,-1.,np.nan,np.inf,-np.inf]:
                data=fixture();start,end=endpoints(data[1],data[2],1)
                frame=data[0][label];frame[symbol]=frame[symbol].astype(object)
                frame.at[(start,end)[offset],symbol]=value
                with self.subTest(label=label,symbol=symbol,offset=offset,value=value),self.assertRaises(ValueError):run(data)

    def test_missing_required_row_column_and_vox_no_fallback(self):
        for label,symbol,mode in [('adj','AAA','column'),('adj','XLP','row'),('vox','VOX','row')]:
            data=fixture();start,_=endpoints(data[1],data[2],1)
            data[0][label]=data[0][label].drop(columns=symbol) if mode=='column' else data[0][label].drop(index=start)
            with self.subTest(label=label,mode=mode),self.assertRaises(ValueError):run(data)

    def test_original_positive_E_parity_is_integrity_gate(self):
        for endpoint in (100.,90.):
            data=fixture();_,end=endpoints(data[1],data[2],1);data[0]['adj'].at[end,'AAA']=endpoint
            with self.assertRaisesRegex(ValueError,'positive momentum'):run(data)

    def test_derived_overflow_aborts_for_security_and_benchmark(self):
        for label,symbol in [('adj','AAA'),('adj','XLP'),('vox','VOX')]:
            data=fixture();start,end=endpoints(data[1],data[2],1)
            data[0][label].at[start,symbol]=np.nextafter(0.,1.)
            data[0][label].at[end,symbol]=np.finfo(float).max
            with self.subTest(symbol=symbol),self.assertRaises(FloatingPointError):run(data)

    def test_internal_dates_recent42_and_future_values_not_consumed(self):
        data=fixture();v,a=run(data);needed=set()
        for i in (1,2,3):needed.update(endpoints(data[1],data[2],i))
        for frame in data[0].values():
            for day in frame.index:
                if day not in needed:frame.loc[day,:]=np.inf
        data[0]['adj']['OFF']=np.nan
        vv,aa=run(data)
        for key in v:np.testing.assert_array_equal(v[key],vv[key])
        self.assertEqual(a,aa)

    def test_price_frame_schema_and_dedicated_vox(self):
        for mode in ('key','column','duplicate','unsorted','timezone','nonmidnight','duplicatecolumn'):
            data=fixture()
            if mode=='key':data[0]['extra']=data[0]['adj']
            elif mode=='column':data[0]['vox']=data[0]['vox'].rename(columns={'VOX':'XLC'})
            elif mode=='duplicate':data[0]['adj']=pd.concat([data[0]['adj'],data[0]['adj'].iloc[-1:]])
            elif mode=='unsorted':data[0]['adj']=data[0]['adj'].iloc[::-1]
            elif mode=='timezone':data[0]['adj'].index=data[0]['adj'].index.tz_localize('UTC')
            elif mode=='nonmidnight':data[0]['adj'].index=data[0]['adj'].index+pd.Timedelta(hours=1)
            else:data[0]['adj']=pd.concat([data[0]['adj'],data[0]['adj'][['AAA']]],axis=1)
            with self.subTest(mode=mode),self.assertRaises(ValueError):run(data)


class PrefixAndSchemaTests(unittest.TestCase):
    def test_every_logical_physical_prefix_full_and_cropped_report(self):
        data=fixture();full,audit=run(data)
        for n in range(5):
            expected,ea=run(data,n)
            for crop in (False,True):
                actual,aa=run(physical(data,n,crop))
                self.assertEqual(ea,aa)
                for key in full:np.testing.assert_array_equal(actual[key],full[key][:n])
            self.assertEqual(ea['event_history'],audit['event_history'][:n])
            self.assertEqual(ea['cell_history'],[r for r in audit['cell_history'] if r['query_index']<n])

    def test_future_report_meta_and_boolean_poison_ignored_by_prefix(self):
        data=fixture();v,a=run(data,2)
        data[1]['dates'][2:]=['bad','bad'];data[1]['symbols'][2:]=[None,None]
        data[3]['eligible']=data[3]['eligible'].astype(object);data[3]['eligible'][2:]=object()
        data[3]['state_weak_narrow']=data[3]['state_weak_narrow'].astype(object);data[3]['state_weak_narrow'][2:]=None
        data[4]['events'][2:]=[None,object()]
        data[4]['records']=[r if r['query_index']<2 else {'query_index':r['query_index'],'poison':object()} for r in data[4]['records']]
        vv,aa=run(data,2);self.assertEqual(a,aa)
        for key in v:np.testing.assert_array_equal(v[key],vv[key])

    def test_nonmutation_and_detached_audit_and_arrays(self):
        data=fixture();panel,meta,calendar,arrays,report=data;before=deepcopy(data)
        v,a=run(data)
        for key in panel:pd.testing.assert_frame_equal(panel[key],before[0][key])
        self.assertEqual(meta,before[1]);self.assertEqual(calendar,before[2]);self.assertEqual(report,before[4])
        for key in arrays:np.testing.assert_array_equal(arrays[key],before[3][key])
        v['original_eligible'][0,0]=False;a['event_history'][0]['symbols'][0]='CHANGED'
        self.assertTrue(arrays['eligible'][0,0]);self.assertEqual(meta['symbols'][0][0],'AAA')

    def test_boolean_shapes_limits_and_etf_schema_rejected(self):
        for mode in ('eligible_int','weak_int','weak_column','metalen','symbolwidth','duplicatesymbol','badetf'):
            data=fixture()
            if mode=='eligible_int':data[3]['eligible']=data[3]['eligible'].astype(int)
            elif mode=='weak_int':data[3]['state_weak_narrow']=data[3]['state_weak_narrow'].astype(int)
            elif mode=='weak_column':data[3]['state_weak_narrow']=data[3]['state_weak_narrow'][:,None]
            elif mode=='metalen':data[1]['dates'].pop()
            elif mode=='symbolwidth':data[1]['symbols'][1].pop()
            elif mode=='duplicatesymbol':data[1]['symbols'][1][1]='AAA'
            else:data[1]['etfs']=[' FUND ']
            with self.subTest(mode=mode),self.assertRaises(ValueError):run(data)
        for n in (True,-1,5,1.5):
            with self.subTest(limit=n),self.assertRaises(ValueError):run(fixture(),n)

    def test_calendar_canonical_unique_order_monthend_and_history(self):
        for mode in ('duplicate','reverse','weekend','noncanonical','witness','history'):
            data=list(fixture())
            if mode=='duplicate':data[2].append(data[2][-1])
            elif mode=='reverse':data[2].reverse()
            elif mode=='weekend':data[2].append('2016-11-12')
            elif mode=='noncanonical':data[2][0]='2015-1-1'
            elif mode=='witness':data[2]=data[2][:data[2].index('2016-10-31')+1]
            else:data[2]=data[2][data[2].index('2016-08-17'):]
            with self.subTest(mode=mode),self.assertRaises(ValueError):run(data)


if __name__ == '__main__':
    unittest.main(verbosity=2)
