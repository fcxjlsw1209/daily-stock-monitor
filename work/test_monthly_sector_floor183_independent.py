"""Synthetic-only independent checks of source clocks, conditional means and ranks."""
import ast
import copy
import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import monthly_sector_floor183_independent as audit


def price_path(returns):
    return np.exp(np.r_[0., np.cumsum(returns)])


def mapper(day, names, sectors, etfs):
    # Explicit synthetic callback for the original frozen classification contract.
    return {s: audit.benchmark_for(pd.Timestamp(day), s, sectors, etfs) for s in names}


def fixture(width=5, count=3):
    index = pd.bdate_range('2014-01-02', periods=400)
    calendar = index.strftime('%Y-%m-%d').tolist()
    names = ['S' + str(i) for i in range(width-1)] + ['QQQ']
    t = np.arange(len(index)-1)
    sector_returns = np.where(t % 3 == 0, .005, -.003)
    frame = pd.DataFrame({s: price_path(sector_returns + .0001 * (j+1) + .002*np.sin(t * (.3+j*.01)))
                          for j, s in enumerate(names)}, index=index)
    for s in ('XLK', 'XLC', 'XLRE', 'XLV'):
        frame[s] = price_path(sector_returns)
    sectors = pd.DataFrame('Technology', index=index, columns=names)
    dates = [calendar[i] for i in (252, 273, 294)[:count]]
    return {'adj': frame, 'sectors': sectors}, dict(dates=dates, symbols=[names[:] for _ in dates], etfs=['QQQ']), calendar


def example_pair(up=100, down=60):
    market = np.r_[np.repeat(.005, up), np.repeat(-.004, down), np.zeros(210-up-down)]
    excess = np.r_[np.repeat(.002, up), np.repeat(-.001, down), np.repeat(.02, 210-up-down)]
    return price_path(market + excess), price_path(market)


class ScalarTests(unittest.TestCase):
    def test_conditional_floor_and_count_weighted_control_exclude_flats(self):
        stock, sector = example_pair()
        result = audit.measure_oracle(stock, sector)
        self.assertTrue(result['available'])
        self.assertEqual((result['up_count'], result['down_count'], result['flat_count']), (100, 60, 50))
        self.assertAlmostEqual(result['candidate'], -.001)
        self.assertAlmostEqual(result['control'], (100*.002-60*.001)/160)
        self.assertNotAlmostEqual(result['control'], (.002-.001)/2)
        self.assertNotAlmostEqual(result['control'], np.mean(result['excess_log_returns']))
        self.assertEqual((result['conditional_mean_estimates'], result['pooled_mean_estimates']), (2, 1))

    def test_40_boundary_and_one_sided_neutral(self):
        for up, down, valid in ((40, 40, True), (39, 100, False), (100, 39, False), (0, 210, False), (210, 0, False), (0, 0, False)):
            result = audit.measure_oracle(*example_pair(up, down))
            self.assertEqual(result['available'], valid)
            self.assertEqual(len(result['excess_log_returns']), 210)
            self.assertEqual(result['conditional_mean_estimates'], 2 if valid else 0)
            self.assertEqual(result['pooled_mean_estimates'], 1 if valid else 0)
            if not valid:
                self.assertEqual(result['reason'], 'insufficient_groups')
                self.assertIsNone(result['mean_up']); self.assertIsNone(result['candidate'])

    def test_original_sector_prices_not_rounded_log_sign(self):
        lo = 1e300; hi = np.nextafter(lo, math.inf)
        self.assertEqual(math.log(lo), math.log(hi))
        sector = np.array([lo if i%2 == 0 else hi for i in range(211)])
        result = audit.measure_oracle(sector, sector)
        self.assertTrue(result['available'])
        self.assertEqual((result['up_count'], result['down_count'], result['flat_count']), (105, 105, 0))
        self.assertEqual(result['candidate'], 0.); self.assertEqual(result['control'], 0.)

    def test_all211_required_and_failure_priority(self):
        stock, sector = example_pair()
        for source in (0, 1):
            for index in (0, 1, 41, 100, 209, 210):
                for bad in (0., -1., np.nan, np.inf):
                    pair = [stock.copy(), sector.copy()]; pair[source][index] = bad
                    result = audit.measure_oracle(*pair)
                    self.assertFalse(result['available']); self.assertEqual(result['conditional_mean_estimates'], 0)
                    self.assertEqual(result['sector_states'], [])
        a, b = stock.astype(object), sector.astype(object); a[0] = 0.; b[0] = 'poison'
        self.assertEqual(audit.measure_oracle(a,b)['reason'], 'invalid_stock_price')
        a[0] = 'bad'
        self.assertEqual(audit.measure_oracle(a,b)['reason'], 'nonnumeric_stock_price')
        a[0] = 1.
        self.assertEqual(audit.measure_oracle(a,b)['reason'], 'nonnumeric_benchmark_price')
        for size in (210, 212):
            with self.assertRaises(ValueError): audit.measure_oracle(np.ones(size), sector)

    def test_stable_pointwise_logs_and_arithmetic_failure(self):
        p = np.array([5e-324 if i%2 == 0 else 1e308 for i in range(211)])
        result = audit.measure_oracle(p, p)
        self.assertTrue(result['available']); self.assertEqual(result['candidate'], 0.)
        with patch.object(audit.math, 'log', return_value=math.inf), self.assertRaises(FloatingPointError):
            audit.measure_oracle(*example_pair())
        with patch.object(audit.math, 'fsum', return_value=math.inf), self.assertRaises(FloatingPointError):
            audit.measure_oracle(*example_pair())

    def test_scale_and_no_source_mutation(self):
        a,b = example_pair(); old = (a.copy(),b.copy()); expected=audit.measure_oracle(a,b)
        for scale in (.01, 1e-140, 1e140):
            audit.record_check(audit.measure_oracle(a*scale,b*scale), expected)
        np.testing.assert_array_equal(a,old[0]); np.testing.assert_array_equal(b,old[1])


class FeatureTests(unittest.TestCase):
    def test_all49_window_counts_and_mapping_receipt(self):
        panel,meta,cal=fixture(49); before=copy.deepcopy(panel); old_meta=copy.deepcopy(meta)
        values,record=audit.independent_features(panel,meta,cal,mapper)
        receipt=audit.verify_features(panel,meta,cal,mapper,values,record)
        self.assertEqual(receipt['available_cells'], 3*48)
        self.assertEqual(receipt['conditional_mean_estimates'], 2*3*48)
        self.assertEqual(receipt['pooled_mean_estimates'], 3*48)
        self.assertEqual(receipt['mapping_source_sha256'], audit.MAPPING_SHA)
        for t,row in zip((252,273,294),record['event_history']):
            self.assertEqual((row['price_start'],row['price_end']), (cal[t-252],cal[t-42]))
            self.assertEqual(row['windows']['first_return_date'],cal[t-251])
            self.assertEqual(row['cells'][-1]['reason'],'etf_neutral')
        for k in panel: pd.testing.assert_frame_equal(panel[k],before[k])
        self.assertEqual(meta,old_meta); json.dumps(record,allow_nan=False); json.dumps(receipt,allow_nan=False)

    def test_prefix_future_price_and_future_classification_poison(self):
        panel,meta,cal=fixture(); full,rec=audit.independent_features(panel,meta,cal,mapper)
        for n in range(4):
            small,srec=audit.independent_features(panel,meta,cal,mapper,limit=n)
            for key in full: np.testing.assert_array_equal(small[key],full[key][:n])
            self.assertEqual(srec['event_history'],rec['event_history'][:n])
        class Poison:
            def __float__(self): raise AssertionError('Future price used')
        panel['adj']=panel['adj'].astype(object)
        panel['adj'].loc[panel['adj'].index>pd.Timestamp(cal[210]),:]=Poison()
        panel['sectors'].loc[panel['sectors'].index>pd.Timestamp(cal[252]),:]='future'
        def checked_mapper(day,names,sectors,etfs):
            self.assertLessEqual(sectors.index.max(),pd.Timestamp(day))
            return mapper(day,names,sectors,etfs)
        values,_=audit.independent_features(panel,meta,cal,checked_mapper,limit=1)
        for key in full: np.testing.assert_array_equal(values[key],full[key][:1])
        panel['adj']=panel['adj'].loc[:cal[210]]
        values,_=audit.independent_features(panel,meta,cal,checked_mapper,limit=1)
        for key in full: np.testing.assert_array_equal(values[key],full[key][:1])

    def test_foreign_unknown_etf_and_xlc_xlre_without_substitution(self):
        panel,meta,cal=fixture(); names=['ASML','S0','S1','S2','QQQ']; meta['symbols']=[names]*3
        panel['adj']['ASML']=panel['adj']['S0']; panel['sectors']['ASML']='unknown'
        panel['sectors']['S0']='Communication Services'; panel['sectors']['S1']='Real Estate'
        panel['sectors']['S2']='unknown'
        values,rec=audit.independent_features(panel,meta,cal,mapper)
        self.assertEqual(rec['event_history'][0]['benchmarks'],dict(ASML='XLK',S0='XLC',S1='XLRE',S2=None,QQQ=None))
        self.assertTrue(values['available'][:,:3].all()); self.assertFalse(values['available'][:,3:].any())
        panel['adj']=panel['adj'].drop(columns='XLC')
        _,rec=audit.independent_features(panel,meta,cal,mapper)
        self.assertEqual(rec['event_history'][0]['cells'][1]['reason'],'missing_benchmark_symbol')
        panel['adj'].loc[cal[0],'XLRE']=np.nan
        _,rec=audit.independent_features(panel,meta,cal,mapper)
        self.assertEqual(rec['event_history'][0]['cells'][2]['reason'],'invalid_benchmark_price')

    def test_mapping_callback_tamper_and_pins(self):
        panel,meta,cal=fixture()
        def wrong(day,names,sectors,etfs):
            result=mapper(day,names,sectors,etfs); result[names[0]]='XLV'; return result
        with self.assertRaises(ValueError): audit.independent_features(panel,meta,cal,wrong)
        with patch.object(audit,'MAPPING_SHA','0'*64),self.assertRaises(ValueError):
            audit.independent_features(panel,meta,cal,mapper)
        with patch.dict(audit.SECTOR,{'Technology':'XLV'}),self.assertRaises(ValueError):
            audit.mapping_constants()

    def test_missing_precedence_and_true_session_axis(self):
        panel,meta,cal=fixture(); panel['adj']=panel['adj'].drop(index=cal[10],columns='S0')
        _,rec=audit.independent_features(panel,meta,cal,mapper)
        self.assertEqual([c['reason'] for c in rec['event_history'][0]['cells']],
                         ['missing_symbol']+['missing_official_session']*3+['etf_neutral'])
        meta['dates'][0]=cal[251]
        _,rec=audit.independent_features(panel,meta,cal,mapper)
        self.assertIsNone(rec['event_history'][0]['windows'])
        self.assertEqual(rec['event_history'][0]['cells'][0]['reason'],'insufficient_history')
        self.assertEqual(rec['event_history'][0]['cells'][-1]['reason'],'etf_neutral')
        panel,meta,cal=fixture(); holiday=cal.pop(100); panel['adj']=panel['adj'].drop(index=holiday)
        meta['dates'][0]=cal[252]
        _,rec=audit.independent_features(panel,meta,cal,mapper)
        self.assertEqual(rec['event_history'][0]['price_end'],cal[210])
        self.assertEqual(rec['event_history'][0]['missing_official_sessions'],[])

    def test_full_audit_and_values_tamper_rejected(self):
        panel,meta,cal=fixture(); values,rec=audit.independent_features(panel,meta,cal,mapper)
        for field in ('mean_up','pooled_mean'):
            bad=copy.deepcopy(rec); bad['event_history'][0]['cells'][0][field]+=.1
            with self.assertRaises(ValueError): audit.verify_features(panel,meta,cal,mapper,values,bad)
        bad=copy.deepcopy(rec); bad['event_history'][0]['cells'][0]['sector_states'][0]*=-1
        with self.assertRaises(ValueError): audit.verify_features(panel,meta,cal,mapper,values,bad)
        for field in ('conditional_mean_estimates','pooled_mean_estimates','minimum_group_count'):
            bad=copy.deepcopy(rec); bad[field]+=1
            with self.assertRaises(ValueError): audit.verify_features(panel,meta,cal,mapper,values,bad)
        bad=copy.deepcopy(values); bad['control'][1,0]+=.1
        with self.assertRaises(ValueError): audit.verify_features(panel,meta,cal,mapper,bad,rec)

    def test_schema_and_masks_fail(self):
        panel,meta,cal=fixture(); values,_=audit.independent_features(panel,meta,cal,mapper)
        for field in ('available','candidate'):
            bad=copy.deepcopy(values)
            if field=='available': bad[field]=bad[field].astype(int)
            else: bad[field][0,0]=np.nan
            with self.assertRaises(ValueError): audit.arrays(bad)
        for n in (-1,4,True,.5):
            with self.assertRaises(ValueError): audit.independent_features(panel,meta,cal,mapper,limit=n)
        del meta['etfs']
        with self.assertRaises(ValueError): audit.independent_features(panel,meta,cal,mapper)

    def test_subtolerance_tie_break_is_not_silently_accepted(self):
        panel,meta,cal=fixture(); panel['adj']['S1']=panel['adj']['S0']
        values,record=audit.independent_features(panel,meta,cal,mapper)
        changed=copy.deepcopy(values)
        changed['candidate'][1,0]=np.nextafter(changed['candidate'][1,0],math.inf)
        with self.assertRaisesRegex(ValueError,'Independent raw rank differs'):
            audit.verify_features(panel,meta,cal,mapper,changed,record)

    def test_production_both_fixtures_and_failure_cross(self):
        import monthly_sector_floor183_kernel as kernel
        import test_monthly_sector_floor183_kernel as producer_tests
        # Production is imported only in this synthetic cross-test, never by the oracle.
        for make, callback in ((fixture,mapper),(producer_tests.fixture,producer_tests.mapper)):
            panel,meta,cal=make()
            for variant in ('normal','missing_session','floor','invalid_source','unknown'):
                p,m=copy.deepcopy(panel),copy.deepcopy(meta)
                if variant=='missing_session': p['adj']=p['adj'].drop(index=p['adj'].index[10])
                elif variant=='floor':
                    for s in ('XLK','XLC','XLY'):
                        if s in p['adj']: p['adj'][s]=1.
                elif variant=='invalid_source': p['adj'].iloc[0,0]=np.nan
                elif variant=='unknown': p['sectors'].iloc[:,:]='unknown'
                values,record=kernel.build_features(p,m,cal,callback)
                self.assertTrue(audit.verify_features(p,m,cal,callback,values,record)['passed'])
            for n in range(len(meta['dates'])+1):
                values,record=kernel.build_features(panel,meta,cal,callback,limit=n)
                small_meta=dict(meta,dates=meta['dates'][:n],symbols=meta['symbols'][:n])
                # A zero-event metadata set cannot carry a width. Compare the oracle prefix instead.
                wanted,expected=audit.independent_features(panel,meta,cal,callback,limit=n)
                for key in wanted: np.testing.assert_array_equal(values[key],wanted[key])
                audit.record_check(record,expected)


class RankTests(unittest.TestCase):
    def fixture(self):
        candidate=np.tile([3.,1.,1.,-1.,np.nan],(4,1)); control=np.tile([1.,1.,1.,1.,np.nan],(4,1))
        candidate[2,1:]=np.nan; control[2,1:]=np.nan
        candidate[3,:]=np.nan; control[3,:]=np.nan
        return dict(candidate=candidate,control=control,available=np.isfinite(candidate))

    def test_manual_average_percentile_ties_offset_and_max(self):
        values=self.fixture(); ranks=audit.independent_ranks(values)
        np.testing.assert_array_equal(ranks[audit.ARMS[1]][1],[.5,.125,.125,-.25,0.])
        np.testing.assert_array_equal(ranks[audit.ARMS[2]][1],[.125,.125,.125,.125,0.])
        deltas=audit.independent_deltas(values); receipt=audit.verify_deltas(values,deltas)
        self.assertTrue(receipt['average_percentile_ranking_checked'])
        self.assertEqual(receipt['max_absolute_delta'],.025)
        for arm in deltas:
            self.assertTrue((deltas[arm][0]==0.).all()); self.assertTrue((deltas[arm][2:]==0.).all())
        self.assertGreater(deltas[audit.ARMS[2]][1,0],0.) # all-tied common offset intentionally kept

    def test_exact_near_ties_no_snap_and_pandas_equivalence(self):
        values=self.fixture(); values['candidate'][1,:4]=[1.,np.nextafter(1.,2.),-1.,1.]
        ranks=audit.independent_ranks(values)
        for arm,key in audit.RAW.items():
            expected=np.zeros(values['available'].shape)
            for i,mask in enumerate(values['available']):
                if mask.sum()>=2: expected[i,mask]=pd.Series(values[key][i,mask]).rank(method='average',pct=True).to_numpy()-.5
            np.testing.assert_array_equal(ranks[arm],expected)

    def test_prefix_neutral_exact_no_filter_or_tanh(self):
        values=self.fixture(); expected=audit.independent_deltas(values)
        for n in range(5):
            small={k:v[:n] for k,v in values.items()}; got=audit.independent_deltas(small)
            for arm in got: np.testing.assert_array_equal(got[arm],expected[arm][:n])
        for arm in expected:
            bad={k:v.copy() for k,v in expected.items()}; bad[arm][1,0]=0.
            with self.assertRaises(ValueError): audit.verify_deltas(values,bad)
            bad={k:v.copy() for k,v in expected.items()}; bad[arm][0,0]=1e-30
            with self.assertRaises(ValueError): audit.verify_deltas(values,bad)
        with self.assertRaises(ValueError): audit.independent_deltas(values,.02)

    def test_no_production_imports(self):
        text=Path(audit.__file__).read_text(); tree=ast.parse(text)
        names=[n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        names += [a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names]
        self.assertFalse(any('kernel' in name or 'host' in name for name in names))
        self.assertNotIn('__main__',text)


class AccountTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import test_monthly_upside_retention181_independent as previous_tests
        with patch.object(previous_tests,'audit',audit): cls.args=previous_tests.account_fixture()

    def test_nine_paths45_slices(self):
        receipt=audit.audit_outcomes(*self.args)
        self.assertEqual((receipt['checked_paths'],receipt['checked_slices']),(9,45))
        self.assertEqual(receipt['outcome_helper_sha256'],audit.OUTCOME_SHA)

    def test_fills_metrics_and_source_hash_failures(self):
        with patch.object(audit,'OUTCOME_SHA','0'*64),self.assertRaises(ValueError): audit.audit_outcomes(*self.args)
        args=list(self.args); args[3]=copy.deepcopy(args[3])
        args[3][audit.ARMS[1]]['next_open_cost']['TRADES'][0]['cost']+=1.
        with self.assertRaises(ValueError): audit.audit_outcomes(*args)
        args=list(self.args); args[4]=copy.deepcopy(args[4])
        args[4][audit.ARMS[2]]['same_close_zero']['slices']['first88']['cagr']+=.01
        with self.assertRaises(ValueError): audit.audit_outcomes(*args)


if __name__=='__main__':
    unittest.main()
