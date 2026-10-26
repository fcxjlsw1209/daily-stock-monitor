"""Synthetic-only source admission tests: no historical arrays or replay."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import ownsector238_source_audit as audit


def frozen_mapper(day, symbols, sectors, etfs):
    result = {}
    for symbol in symbols:
        if symbol in etfs: result[symbol] = None
        elif symbol in audit.FOREIGN: result[symbol] = audit.FOREIGN[symbol]
        elif day in sectors.index and symbol in sectors.columns:
            result[symbol] = audit.SECTOR.get(sectors.at[day,symbol])
        elif pd.Timestamp(day) in sectors.index and symbol in sectors.columns:
            result[symbol] = audit.SECTOR.get(sectors.at[pd.Timestamp(day),symbol])
        else: result[symbol] = None
    return result


def fixture():
    sessions = pd.bdate_range('2015-01-01','2017-03-03')
    days = ['2016-08-17','2016-08-31','2016-09-30','2016-10-31']
    symbols = ['AAA','ASML','QQQ','BBB']
    meta = dict(dates=days,symbols=[symbols.copy() for _ in days],etfs=['QQQ'])
    prices = pd.DataFrame(100.,index=sessions,columns=symbols+['SPY','XLK','XLV'])
    sectors = pd.DataFrame('Technology',index=sessions,columns=symbols)
    sectors['BBB'] = 'Healthcare'; sectors['ASML'] = 'Healthcare'
    eligible = np.ones((len(days),len(symbols)),dtype=bool)
    eligible[2,3] = False
    return {'adj':prices,'sectors':sectors},meta,[str(x.date()) for x in sessions],eligible


def run(args, limit=None, mapper=frozen_mapper):
    return audit.inspect_sources(*args,mapper,limit=limit)


class SourceInspectorTests(unittest.TestCase):
    def test_valid_dispatch_counts_and_no_strategy_math(self):
        report = run(fixture()); counts = report['counts']
        self.assertTrue(report['admitted']); self.assertEqual(report['status'],'ADMITTED')
        self.assertEqual(counts['initial_partial_events'],1)
        self.assertEqual(counts['regular_events'],3)
        self.assertEqual(counts['required_cells'],11)
        self.assertEqual(counts['market_endpoint_checks'],6)
        self.assertEqual(counts['security_benchmark_endpoint_checks'],44)
        self.assertEqual(counts['valid_endpoints'],50)
        self.assertEqual(report['records'][1]['benchmark'],'XLK')
        self.assertEqual(report['records'][1]['mapping_source'],'frozen_FOREIGN_override')
        self.assertEqual(report['records'][2]['benchmark'],'SPY')
        self.assertEqual(report['records'][2]['mapping_source'],'original_META.etfs_SPY')
        self.assertEqual(set(counts),set(audit.COUNT_FIELDS))
        for key in ('momentum_computed','eligibility_changed','rankings_computed','orders_computed','returns_computed','strict_pit'):
            self.assertIs(report[key],False)
        self.assertEqual(report['model_fits'],0)
        json.dumps(report,allow_nan=False)

    def test_unknown_stock_mapping_collects_all_cells(self):
        args=fixture();args[0]['sectors']['AAA']='Unknown'
        report=run(args)
        self.assertFalse(report['admitted']);self.assertEqual(report['counts']['mapping_missing_cells'],3)
        cells=[r for r in report['records'] if r['symbol']=='AAA']
        self.assertTrue(all(r['benchmark'] is None and r['gap_reasons']==['missing_benchmark_mapping'] for r in cells))
        self.assertEqual(report['counts']['missing_endpoints'],6)
        self.assertTrue(all(r['mapping_source']=='query_date_sector_panel' for r in cells))

    def test_no_sector_forward_fill(self):
        args=fixture();args[0]['sectors']=args[0]['sectors'].drop(pd.Timestamp(args[1]['dates'][1]))
        report=run(args);cells=[x for x in report['records'] if x['query_index']==1]
        self.assertEqual([x['benchmark'] for x in cells],[None,'XLK','SPY',None])

    def test_partial_needs_no_sources_or_mapping(self):
        args=fixture();pp,mm,cc,ee=audit.physical_prefix(*args,1)
        pp={k:v.iloc[:0] for k,v in pp.items()}
        rr=audit.inspect_sources(pp,mm,cc,ee,lambda *a: (_ for _ in ()).throw(AssertionError('mapper invoked')))
        self.assertTrue(rr['admitted']);self.assertEqual(rr['counts']['required_cells'],0)
        self.assertEqual(rr['records'],[]);self.assertEqual(rr['events'][0]['market_endpoints'],[])

    def test_first_regular_is_not_partial(self):
        pp,mm,cc,ee=fixture();mm=dict(mm,dates=mm['dates'][1:],symbols=mm['symbols'][1:]);ee=ee[1:]
        rr=run((pp,mm,cc,ee));self.assertEqual(rr['counts']['initial_partial_events'],0)
        self.assertTrue(rr['events'][0]['active'])

    def test_empty_regular_eligibility_still_checks_market(self):
        args=fixture();args[3][:]=False;args[0]['adj']=args[0]['adj'].drop(columns='SPY')
        rr=run(args);self.assertFalse(rr['admitted']);self.assertEqual(rr['counts']['required_cells'],0)
        self.assertEqual(rr['counts']['market_gap_events'],3);self.assertEqual(rr['counts']['missing_endpoints'],6)
        self.assertTrue(all(g['scope']=='market' and g['symbol']==g['benchmark']=='SPY' for g in rr['gaps']))

    def test_endpoint_status_taxonomy_and_precedence(self):
        day=pd.Timestamp('2020-01-31');frame=pd.DataFrame({'A':[1.]},index=[day]).astype(object)
        cases=[(None,'missing','missing_value'),(pd.NA,'missing','missing_value'),(float('nan'),'missing','missing_value'),
            (True,'invalid','invalid_boolean'),(np.bool_(False),'invalid','invalid_boolean'),
            ('100','invalid','invalid_nonreal'),(1+0j,'invalid','invalid_nonreal'),(float('inf'),'invalid','invalid_nonfinite'),
            (-float('inf'),'invalid','invalid_nonfinite'),(0,'invalid','invalid_nonpositive'),(-1,'invalid','invalid_nonpositive'),
            (10**400,'invalid','invalid_numeric_conversion'),(np.float64(2),'valid',None)]
        for value,status,reason in cases:
            with self.subTest(value=repr(value)):
                frame.at[day,'A']=value;rr=audit.endpoint(frame,'security_start','A',day)
                self.assertEqual((rr['status'],rr['reason']),(status,reason));self.assertNotIn('value',rr)
        self.assertEqual(audit.endpoint(frame,'x',None,day+pd.Timedelta(days=1))['reason'],'missing_benchmark_mapping')
        self.assertEqual(audit.endpoint(frame,'x','ZZ',day+pd.Timedelta(days=1))['reason'],'missing_row')
        self.assertEqual(audit.endpoint(frame,'x','ZZ',day)['reason'],'missing_column')

    def test_multiple_gaps_retained_market_first(self):
        args=fixture();pp,mm,cc,ee=audit.physical_prefix(*args,2);day=pd.Timestamp(cc[cc.index(mm['dates'][1])-252])
        pp['adj']=pp['adj'].astype(object);pp['adj'].at[day,'SPY']=None;pp['adj'].at[day,'AAA']=False
        pp['sectors']['BBB']='Unknown'
        rr=run((pp,mm,cc,ee));self.assertEqual(rr['gaps'][0]['scope'],'market')
        self.assertEqual([g['symbol'] for g in rr['gaps']],['SPY','AAA','QQQ','BBB'])
        self.assertEqual(rr['counts']['invalid_endpoints'],1)
        self.assertEqual(rr['counts']['gap_cells'],3)

    def test_noneligible_bad_cells_do_not_create_gap(self):
        args=fixture();args[3][:,0]=False;args[0]['adj']['AAA']=np.nan;args[0]['sectors']['AAA']='Unknown'
        self.assertTrue(run(args)['admitted'])

    def test_mapper_schema_and_precedence_rejected(self):
        args=fixture()
        for fn in (lambda *a:{},lambda *a:{s:' ' for s in a[1]},lambda *a:{s:'SPY' for s in a[1]}):
            with self.assertRaises(ValueError):run(args,mapper=fn)
        def wrong(day,symbols,sectors,etfs):
            m=frozen_mapper(day,symbols,sectors,etfs);m['ASML']='XLV';return m
        with self.assertRaises(ValueError):run(args,mapper=wrong)

    def test_sector_callback_asof_and_detached(self):
        args=fixture();before=args[0]['sectors'].copy(deep=True)
        def checked(day,names,sectors,etfs):
            self.assertLessEqual(sectors.index[-1],pd.Timestamp(day))
            answer=frozen_mapper(day,names,sectors,etfs);sectors.iloc[:,:]='Unknown';names[:]=[];etfs[:]=[]
            return answer
        self.assertTrue(run(args,mapper=checked)['admitted'])
        pd.testing.assert_frame_equal(before,args[0]['sectors'])

    def test_all_logical_physical_prefixes_and_no_mutation(self):
        args=fixture();copyargs=copy.deepcopy(args);full=run(args)
        for n in range(5):
            a=run(args,limit=n);b=run(audit.physical_prefix(*args,n));self.assertEqual(a,b)
            self.assertEqual(a['events'],full['events'][:n])
            self.assertEqual(a['records'],[r for r in full['records'] if r['query_index']<n])
        for k in args[0]:pd.testing.assert_frame_equal(args[0][k],copyargs[0][k])
        self.assertEqual(args[1:3],copyargs[1:3]);np.testing.assert_array_equal(args[3],copyargs[3])

    def test_future_values_and_future_eligibility_not_consumed(self):
        args=fixture();expected=run(args,limit=2);day=pd.Timestamp(args[1]['dates'][1])
        args[0]['adj']=args[0]['adj'].astype(object);args[0]['adj'].loc[args[0]['adj'].index>day,:]='poison'
        args[0]['sectors'].loc[args[0]['sectors'].index>day,:]='Unknown'
        e=args[3].astype(object);e[2:,:]='bad';args=args[:3]+(e,)
        args[1]['dates'][2]='invalid-future';args[1]['symbols'][2]=['bad']
        self.assertEqual(run(args,limit=2),expected)
        with self.assertRaises(ValueError):run(args)

    def test_schema_dates_id_and_boolean_errors(self):
        mutations=[lambda a:a[1]['dates'].__setitem__(1,'2016-8-31'),
            lambda a:a[1]['dates'].__setitem__(1,'2016-08-30'),
            lambda a:a[1]['symbols'][1].__setitem__(1,'AAA'),
            lambda a:a[1]['etfs'].append('QQQ'),
            lambda a:a[1]['symbols'][1].__setitem__(0,' AAA'),
            lambda a:a[1]['dates'].__setitem__(2,'2016-08-17')]
        for change in mutations:
            args=fixture();change(args)
            with self.assertRaises((ValueError,TypeError)):run(args)
        args=fixture();args=args[:3]+(args[3].astype(int),)
        with self.assertRaises(ValueError):run(args)
        for limit in (-1,True,5,1.0):
            with self.assertRaises(ValueError):run(fixture(),limit=limit)

    def test_malformed_frame_and_calendar_axes(self):
        args=fixture();args[0]['adj']=pd.concat([args[0]['adj'].iloc[:1],args[0]['adj']])
        with self.assertRaises(ValueError):run(args)
        args=fixture();args[0]['sectors'].columns=['AAA','AAA','QQQ','BBB']
        with self.assertRaises(ValueError):run(args)
        args=fixture();args[2].insert(1,args[2][0])
        with self.assertRaises(ValueError):run(args)
        args=fixture();args[2][:]=args[2][300:]
        with self.assertRaises(ValueError):run(args)
        args=fixture();end=args[2].index(args[1]['dates'][-1]);args[2][:]=args[2][:end+1]
        with self.assertRaises(ValueError):run(args)

    def test_mapping_constants_bound_to_frozen_source(self):
        audit.mapping_constants()

    def test_valid_numeric_magnitudes_never_affect_report(self):
        args=fixture();expected=run(args)
        args[0]['adj']*=1e200
        self.assertEqual(run(args),expected)


class RunnerBoundaryTests(unittest.TestCase):
    def test_source_registration_and_audit_are_write_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with mock.patch.object(audit,'OUT',root),mock.patch.object(audit,'source_binding',side_effect=AssertionError('loaded')):
                with self.assertRaisesRegex(ValueError,'New source-only'):audit.register()
            (root/'REPORT.json').write_text('{}')
            with mock.patch.object(audit,'OUT',root),mock.patch.object(audit,'gate',return_value={}),mock.patch.object(audit,'previous',side_effect=AssertionError('loaded')):
                with self.assertRaisesRegex(ValueError,'write-once'):audit.audit('a'*64)

    def test_formal_tests_write_once_precedes_suite_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'TESTS.json';p.write_text('{}')
            with mock.patch.object(audit,'TESTS',p),mock.patch.object(audit,'previous',side_effect=AssertionError('loaded')):
                with self.assertRaisesRegex(ValueError,'write-once'):audit.test()

    def test_audit_plan_hash_gate_precedes_inputs(self):
        with mock.patch.object(audit,'previous',side_effect=AssertionError('loaded')):
            for value in (None,'wrong',True):
                with self.assertRaises(ValueError):audit.gate(value)

    def test_no_strategy_load_or_math_calls_in_actual_runner(self):
        import inspect
        src=inspect.getsource(audit.audit)
        for forbidden in ('study.load(', 'scope.load(', 'modified_pre', 'ranked_forecasts', 'verify_panel_parity', 'select_boundary'):
            self.assertNotIn(forbidden,src)
        self.assertIn("['eligible']",src)


if __name__=='__main__':unittest.main(verbosity=2)
