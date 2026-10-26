"""Synthetic-only own-volume pairing: source clocks, strict medians and ranks."""
import copy
import math
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import monthly_volume_pair203_kernel as k


def mapper(day, symbols, sectors, etfs):
    mapping = {'Technology': 'XLK', 'Communication Services': 'XLC', 'Energy': 'XLE'}
    return {s: None if s in etfs else 'XLK' if s == 'ASML' else
            mapping.get(sectors.at[pd.Timestamp(day), s]) if s in sectors.columns and
            pd.Timestamp(day) in sectors.index else None for s in symbols}


def fixture():
    days = pd.bdate_range('2014-01-01', '2017-01-04'); x = np.arange(len(days), dtype=float)
    names = ['A', 'B', 'ASML', 'FUND', 'UNKNOWN']
    adj = pd.DataFrame({s: 100. + j + .02*x + 3*np.sin(x/(9+j))
                       for j, s in enumerate(names+['XLK', 'XLC', 'XLE', 'XLRE'])}, index=days)
    volume = pd.DataFrame({s: 100. + 2*(x % (53+11*j)) + 20*np.sin(x/(8+j))
                          for j, s in enumerate(names)}, index=days)
    sectors = pd.DataFrame(dict(A='Technology', B='Communication Services', ASML='Energy',
                                FUND='Technology', UNKNOWN='Unknown'), index=days)
    meta = dict(dates=['2016-08-17', '2016-08-31', '2016-09-30'],
                symbols=[names[:] for _ in range(3)], etfs=['FUND'])
    return {'adj': adj, 'volume': volume, 'sectors': sectors}, meta, days.strftime('%Y-%m-%d').tolist()


def toy(reverse=False):
    stock = np.full(253, 100.); benchmark = np.full(253, 100.); volume = np.full(315, 100.)
    for j in range(63,315):
        high = ((j-63)//21 % 2 == 0) != reverse
        prior = volume[j-63:j]
        volume[j] = 2*max(prior) if high else .5*min(prior)
    for block in range(1,13): stock[21*block] = 200. if block % 2 else 100.
    return stock, benchmark, volume


class MeasureTests(unittest.TestCase):
    def test_canonical_pairing_sign_and_matched_control(self):
        a,b=k.measure(*toy()),k.measure(*toy(True))
        self.assertEqual(a['control'],b['control']);self.assertGreater(a['candidate'],a['control'])
        self.assertLess(b['candidate'],b['control']);self.assertEqual(a['high_counts'],[21,0]*6)
        self.assertEqual(a['activity'],[1.,0.]*6)
        c=math.fsum(a['excess_log_returns'])/12;h=math.fsum(a['activity'])/12
        wanted=c+math.fsum((x-h)*(y-c) for x,y in zip(a['activity'],a['excess_log_returns']))/12
        self.assertEqual(a['candidate'],wanted)
        self.assertEqual(a['stock_log_gains'][0],math.log(200.)-math.log(100.))

    def test_previous63_median_excludes_current_and_strict_tie(self):
        p,b,v=toy();v[:63]=np.arange(1.,64.);v[63:]=33.
        a=k.measure(p,b,v)
        self.assertEqual(a['preceding_medians'][0],32.)
        self.assertEqual(a['high_volume_flags'][0],1)
        self.assertEqual(a['preceding_medians'][1],33.)
        self.assertEqual(a['high_volume_flags'][1],0)
        changed=v.copy();changed[63]=1e90
        z=k.measure(p,b,changed)
        self.assertEqual(z['preceding_medians'][0],a['preceding_medians'][0])
        self.assertEqual(z['high_volume_flags'][0],1)
        self.assertEqual(a['preceding_medians'][-1],float(sorted(v[251:314])[31]))

    def test_all_nine_counts_and_last_block_alignment(self):
        a=k.measure(*toy())
        expected=(506,315,252,252,12,12,1,2,2)
        self.assertEqual(tuple(a[x] for x in k.COUNT_FIELDS),expected)
        self.assertEqual(a['endpoint_indices'],list(range(0,253,21)))
        self.assertEqual(a['volume_block_start_indices'],list(range(63,315,21)))
        self.assertEqual(a['volume_block_end_indices'],list(range(83,315,21)))
        self.assertEqual(len(a['preceding_medians']),252);self.assertEqual(len(a['high_volume_flags']),252)
        self.assertTrue(all(type(x) is int for x in a['high_volume_flags']+a['high_counts']))
        self.assertEqual(a['high_counts'][-1],sum(a['high_volume_flags'][231:252]))

    def test_exact_constant_branches_and_no_nearconstant_snap(self):
        p,b,v=toy();v[:]=100.;a=k.measure(p,b,v)
        self.assertEqual(a['constant_branch'],'all_activity_equal')
        self.assertEqual(a['candidate'].hex(),a['control'].hex());self.assertEqual(a['covariance'],0.)
        p,b,v=toy();p=b.copy();a=k.measure(p,b,v)
        self.assertEqual(a['constant_branch'],'all_excess_equal');self.assertEqual(a['candidate'].hex(),a['control'].hex())
        p[21]=100.*(1+2**-42);a=k.measure(p,b,v)
        self.assertIsNone(a['constant_branch']);self.assertGreater(len(set(a['excess_log_returns'])),1)

    def test_complete_source_admission_precedes_any_signal(self):
        for source,label in ((0,'stock_price'),(1,'benchmark_price'),(2,'volume')):
            for value,prefix in ((np.nan,'invalid'),(0.,'invalid'),(-1.,'invalid'),(np.inf,'invalid'),
                                 (True,'nonnumeric'),('2','nonnumeric'),(None,'nonnumeric'),(1+0j,'nonnumeric')):
                args=[x.astype(object) for x in toy()];args[source][11]=value
                with self.subTest(source=label,value=value):
                    with patch.object(k.math,'log',side_effect=AssertionError('Rejected sources must not calculate logs')):
                        a=k.measure(*args)
                    self.assertFalse(a['available']);self.assertEqual(a['reason'],prefix+'_'+label)
                    for key in ('preceding_medians','high_volume_flags','high_counts','activity','excess_log_returns'):
                        self.assertEqual(a[key],[])
                    self.assertTrue(all(a[key]==0 for key in k.COUNT_FIELDS))
        p,b,v=toy();p[3]=0.;b[4]=0.;v[5]=0.
        self.assertEqual(k.measure(p,b,v)['reason'],'invalid_stock_price')

    def test_volume_scale_invariance_exact_and_input_immutability(self):
        args=toy();before=[x.copy() for x in args];a=k.measure(*args)
        for scale in (2.**-200,2.**200):
            b=k.measure(args[0],args[1],args[2]*scale)
            for key in a:
                if key=='preceding_medians':
                    self.assertEqual(b[key],[v*scale for v in a[key]])
                else:self.assertEqual(a[key],b[key])
        for x,y in zip(args,before):np.testing.assert_array_equal(x,y)

    def test_extreme_finite_prices_log_difference_not_ratio(self):
        p,b,v=toy();p[::21]=[5e-324,1e308]*6+[5e-324]
        a=k.measure(p,b,v);self.assertTrue(a['available']);self.assertTrue(math.isfinite(a['candidate']))
        with patch.object(k.math,'log',return_value=float('inf')):
            with self.assertRaises(FloatingPointError):k.measure(*toy())

    def test_exact_source_shapes(self):
        p,b,v=toy()
        for args in ((p[:-1],b,v),(p,b[:-1],v),(p,b,v[:-1]),(p,b,v.reshape(315,1))):
            with self.assertRaises(ValueError):k.measure(*args)


class BuilderTests(unittest.TestCase):
    def test_full_windows_medians_activity_alignment_mapping_and_counts(self):
        p,m,c=fixture();v,a=k.build_features(p,m,c,mapper)
        self.assertEqual(set(v),{'candidate','control','available','candidate_rank','control_rank'})
        self.assertEqual(a['available_by_event'],[3]*3)
        for field,percell in zip(k.COUNT_FIELDS,(506,315,252,252,12,12,1,2,2)):
            self.assertEqual(a[field],9*percell)
        self.assertEqual(a['reason_counts'],{'available':9,'etf_neutral':3,'missing_benchmark':3})
        for row in a['event_history']:
            t=c.index(row['date']);self.assertEqual(row['price_dates'],c[t-294:t-41])
            self.assertEqual(row['volume_dates'],c[t-356:t-41]);self.assertEqual(row['activity_dates'],row['price_dates'][1:])
            self.assertEqual(row['median_start_dates'],c[t-356:t-104])
            self.assertEqual(row['median_end_dates'],c[t-294:t-42])
            self.assertEqual(row['volume_block_end_dates'],row['endpoint_dates'][1:])
            self.assertEqual(row['cells'][2]['benchmark'],'XLK')
            cell=row['cells'][0]
            self.assertEqual(cell['volume_block_start_dates'],row['volume_block_start_dates'])
            self.assertEqual(cell['volume_block_end_dates'],row['volume_block_end_dates'])
        self.assertTrue(np.isfinite(v['candidate'][0,:3]).all());self.assertTrue((v['candidate_rank'][:,3:]==0.).all())

    def test_missing_volume_or_price_rows_never_compress_calendar(self):
        p,m,c=fixture();t=c.index(m['dates'][0]);missing=c[t-340]
        p['volume']=p['volume'].drop(index=missing)
        v,a=k.build_features(p,m,c,mapper,1)
        self.assertEqual(a['event_history'][0]['missing_official_volume_sessions'],[missing])
        self.assertEqual(a['available_cells'],0)
        self.assertEqual(a['event_history'][0]['cells'][0]['reason'],'missing_official_volume_session')
        p,m,c=fixture();missing=c[t-100];p['adj']=p['adj'].drop(index=missing)
        _,a=k.build_features(p,m,c,mapper,1)
        self.assertEqual(a['event_history'][0]['missing_official_price_sessions'],[missing])
        self.assertEqual(a['event_history'][0]['cells'][0]['reason'],'missing_official_price_session')

    def test_volume_baseline_prefix_and_nonendpoint_prices_required(self):
        for frame,symbol,offset,reason in (('volume','A',-356,'invalid_volume'),
            ('volume','A',-300,'invalid_volume'),('volume','A',-42,'invalid_volume'),
            ('adj','A',-283,'invalid_stock_price'),('adj','XLK',-283,'invalid_benchmark_price')):
            p,m,c=fixture();t=c.index(m['dates'][0]);p[frame].at[pd.Timestamp(c[t+offset]),symbol]=0.
            v,a=k.build_features(p,m,c,mapper,1)
            self.assertFalse(v['available'][0,0]);self.assertEqual(a['event_history'][0]['cells'][0]['reason'],reason)

    def test_unconsumed_sources_and_neutral_candidates_not_read(self):
        p,m,c=fixture();want,a=k.build_features(p,m,c,mapper)
        p['adj']['XLRE']='unused';p['adj']['FUND']='unused';p['adj']['UNKNOWN']='unused'
        p['volume']['XLK']='unconsumed benchmark volume';p['volume']['FUND']='unused';p['volume']['UNKNOWN']='unused'
        got,b=k.build_features(p,m,c,mapper)
        for key in want:np.testing.assert_array_equal(got[key],want[key])
        self.assertEqual(a,b)

    def test_missing_column_priority_and_distinct_identity(self):
        for frame,column,cell,reason in (('adj','A',0,'missing_symbol'),('adj','XLC',1,'missing_benchmark_symbol'),
                                        ('volume','A',0,'missing_volume_symbol')):
            p,m,c=fixture();p[frame]=p[frame].drop(columns=column)
            _,a=k.build_features(p,m,c,mapper,1);self.assertEqual(a['event_history'][0]['cells'][cell]['reason'],reason)
        p,m,c=fixture()
        def duplicate(d,ss,sectors,etfs):
            result=mapper(d,ss,sectors,etfs);result['A']='A';return result
        with self.assertRaises(ValueError):k.build_features(p,m,c,duplicate)

    def test_separate_calendar_window_audit_when_volume_history_insufficient(self):
        p,m,c=fixture();t=c.index(m['dates'][0]);short=c[t-320:]
        v,a=k.build_features(p,m,short,mapper,1);row=a['event_history'][0]
        self.assertEqual(len(row['price_dates']),253);self.assertEqual(row['volume_dates'],[])
        self.assertIsNone(row['volume_start']);self.assertFalse(v['available'].any())
        self.assertEqual(row['cells'][0]['reason'],'insufficient_history')
        self.assertEqual(row['cells'][0]['endpoint_dates'],[])

    def test_symmetric_ranks_ties_singleton_missing_and_permutation(self):
        np.testing.assert_array_equal(k.centered_rank(np.array([2.,2.,6.,np.nan]),np.array([True,True,True,False])),[-.25,-.25,.5,0.])
        np.testing.assert_array_equal(k.centered_rank(np.ones(3),np.ones(3,bool)),np.zeros(3))
        np.testing.assert_array_equal(k.centered_rank(np.array([1.,np.nan]),np.array([True,False])),np.zeros(2))
        p,m,c=fixture();v,_=k.build_features(p,m,c,mapper)
        order=[2,0,4,1,3];mm=dict(m,symbols=[[ss[j] for j in order] for ss in m['symbols']])
        got,_=k.build_features(p,mm,c,mapper)
        for key in v:np.testing.assert_array_equal(got[key],v[key][:,order])

    def test_all_prefixes_recent42_and_future_values_isolated(self):
        p,m,c=fixture();v,a=k.build_features(p,m,c,mapper)
        for n in range(4):
            small={key:frame.loc[:m['dates'][n-1]].copy() if n else frame.iloc[:0].copy() for key,frame in p.items()}
            got,b=k.build_features(small,m,c,mapper,n)
            for key in v:np.testing.assert_array_equal(got[key],v[key][:n])
            self.assertEqual(b['event_history'],a['event_history'][:n])
        poisoned={key:frame.astype(object) for key,frame in p.items()};t=c.index(m['dates'][0])
        for key in ('adj','volume'):poisoned[key].loc[c[t-41]:]='unused recent and future'
        poisoned['sectors'].loc[poisoned['sectors'].index>m['dates'][0]]='future classification'
        def checked(day,ss,sectors,etfs):
            self.assertLessEqual(sectors.index[-1],pd.Timestamp(day));return mapper(day,ss,sectors,etfs)
        got,b=k.build_features(poisoned,m,c,checked,1)
        for key in v:np.testing.assert_array_equal(got[key],v[key][:1])
        self.assertEqual(b['event_history'],a['event_history'][:1])

    def test_query_date_mapping_held_over_entire_formation(self):
        p,m,c=fixture();p['sectors'].loc[m['dates'][1]:,'A']='Communication Services'
        _,a=k.build_features(p,m,c,mapper)
        self.assertEqual([r['benchmarks']['A'] for r in a['event_history']],['XLK','XLC','XLC'])
        self.assertEqual(a['event_history'][1]['cells'][0]['benchmark'],'XLC')

    def test_schema_prefix_guard_and_source_immutability(self):
        p,m,c=fixture();before=copy.deepcopy((p,m,c));k.build_features(p,m,c,mapper)
        for key in p:pd.testing.assert_frame_equal(p[key],before[0][key])
        self.assertEqual(m,before[1]);self.assertEqual(c,before[2])
        for limit in (-1,4,True,1.5):
            with self.assertRaises(ValueError):k.build_features(p,m,c,mapper,limit)
        with self.assertRaises(ValueError):k.build_features({'adj':p['adj'],'sectors':p['sectors']},m,c,mapper)
        with self.assertRaises(ValueError):k.build_features(p,m,c,lambda *args:{})
        with self.assertRaises(ValueError):k.build_features(p,m,c,lambda d,ss,sectors,etfs:{s:'XLK' for s in ss})
        wrong=copy.deepcopy(m);wrong['dates'][1]='2016-08-30'
        with self.assertRaises(ValueError):k.build_features(p,wrong,c,mapper)
        with self.assertRaises(ValueError):k.build_features(p,m,c[:c.index(m['dates'][-1])+1],mapper)


if __name__=='__main__':unittest.main(verbosity=2)
