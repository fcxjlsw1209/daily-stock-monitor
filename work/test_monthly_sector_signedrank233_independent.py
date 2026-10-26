"""Synthetic-only independent233 rational ranks, clocks and unchanged selection."""
import copy
from fractions import Fraction
import inspect
import math
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import monthly_sector_signedrank233_independent as audit

def mapper(day, names, sectors, etfs):
    helper = audit.frozen_helper()
    helper.mapping_constants()
    return {s: helper.benchmark_for(pd.Timestamp(day), s, sectors, etfs) for s in names}


def fixture(count=3, width=4):
    index = pd.bdate_range('2014-01-01', periods=1000)
    x = np.arange(len(index), dtype=float)
    bench = np.exp(.00015 * x + .01 * np.sin(x * .05))
    prices = pd.DataFrame({'A': bench * np.exp(.0003 * x),
        'C': bench * np.exp(.08 * np.sin(x / 130.)), 'BABA': bench * np.exp(-.0002 * x),
        'FUND': bench, 'XLK': bench, 'XLC': bench, 'XLY': bench}, index=index)
    names = ['A', 'C', 'BABA', 'FUND'][:width]
    sectors = pd.DataFrame('Technology', index=index, columns=names)
    if 'C' in sectors:
        sectors.C = 'Communication Services'
    if 'BABA' in sectors:
        sectors.BABA = 'Energy'  # Explicit foreign mapping has priority.
    dates = ['2016-08-17', '2016-08-31', '2016-09-30'][:count]
    return {'adj': prices, 'sectors': sectors}, dict(dates=dates, symbols=[names[:] for _ in dates],
        etfs=['FUND']), index.strftime('%Y-%m-%d').tolist()


def prices_from_levels(levels):
    return np.interp(np.arange(253), np.arange(0, 253, 21), np.asarray(levels, float))



def with_ends(args):
    p,m,c=copy.deepcopy(args)
    m['ends']=['2016-08-31','2016-09-30','2016-10-31'][:len(m['dates'])]
    return p,m,c


def physical_prefix(args,n):
    p,m,c=copy.deepcopy(args);cut=m['dates'][n-1] if n else None
    p={k:f.loc[:cut].copy() if n else f.iloc[:0].copy() for k,f in p.items()}
    for k in ('dates','symbols','ends'):
        if k in m:m[k]=m[k][:n]
    return p,m,c


def original(meta):
    n,w=len(meta['dates']),len(meta['symbols'][0]);shape=(n,w)
    pre=np.tile(np.array([.25,.20,.10,.095])[:w],(n,1))
    z=np.zeros(shape);f=np.zeros(shape,bool)
    return dict(baseline=pre.copy(),core=np.zeros(shape),core_weight=np.full((n,1),.5),
        eligible=np.ones(shape,bool),effects=np.tile([0.,0.,-.4,.4],(n,1))[:,:w],
        state_calm10=np.ones(n,bool),state_weak_narrow=np.zeros(n,bool),
        pre_score=pre.copy(),rank_resid=z.copy(),old_ml4=z.copy(),rank_log_excess=z.copy(),
        extension_available=f.copy(),forecast_available=f.copy(),local_available=f.copy(),
        model_available=f.copy(),raw_local_constant=z.copy(),raw_local_linear=z.copy(),raw_source_log=z.copy())


def order_rows(scores,choices,meta):
    return {arm:[dict(date=meta['dates'][i],end=meta['ends'][i],signal_date=meta['dates'][i],
        symbols=[meta['symbols'][i][j] for j in ids],scores=[float(scores[arm][i,j]) for j in ids])
        for i,ids in enumerate(choices[arm])] for arm in audit.ARMS}


class ScalarTests(unittest.TestCase):
    def test_all_positive_negative_and_zero_bounds(self):
        for growth,target in ((2.,.75),(.5,-.75),(1.,0.)):
            a=prices_from_levels([growth**i for i in range(13)])
            got=audit.measure_oracle(a,np.ones(253))
            self.assertEqual(got['candidate'],target);self.assertEqual(got['control'],target)
            self.assertTrue(got['constant_magnitude_copy']);self.assertEqual(got['twice_rank_sum'],156)
            self.assertEqual(got['twice_magnitude_ranks'],[13]*12)

    def test_larger_losses_reverse_sign_balance_without_new_amplitude(self):
        levels=[1.]
        for growth in [2.]*7+[.25]*5:levels.append(levels[-1]*growth)
        got=audit.measure_oracle(prices_from_levels(levels),np.ones(253))
        self.assertEqual(got['relative_signs'],[1]*7+[-1]*5)
        self.assertEqual(got['twice_magnitude_ranks'],[8]*7+[20]*5)
        self.assertEqual(got['candidate_numerator'],-44)
        self.assertEqual(got['candidate'],-44/208);self.assertEqual(got['control'],2/16)

    def test_all_equal_magnitudes_copy_with_mixed_signs(self):
        levels=[1.]
        for growth in [2.,.5]*5+[2.,2.]:levels.append(levels[-1]*growth)
        got=audit.measure_oracle(prices_from_levels(levels),np.ones(253))
        self.assertTrue(got['constant_magnitude_copy']);self.assertEqual(got['candidate'],got['control'])
        self.assertEqual(got['candidate_numerator'],13*sum(got['relative_signs']))
        self.assertEqual(np.float64(got['candidate']).tobytes(),np.float64(got['control']).tobytes())

    def test_zero_blocks_participate_in_all_twelve_ranks(self):
        levels=[1.]*7+[2.**i for i in range(1,7)]
        got=audit.measure_oracle(prices_from_levels(levels),np.ones(253))
        self.assertEqual(got['relative_signs'],[0]*6+[1]*6)
        self.assertEqual(got['twice_magnitude_ranks'],[7]*6+[19]*6)
        self.assertEqual(got['candidate'],114/208);self.assertEqual(got['control'],6/16)
        self.assertEqual(got['magnitude_pair_comparisons'],66)

    def test_exact_fraction_near_ties_and_extreme_finite_endpoints(self):
        a=1e300;b=float(np.nextafter(a,math.inf));self.assertEqual(math.log(a),math.log(b))
        self.assertGreater(audit.relative_z(b,a,a,a),0)
        self.assertEqual(audit.relative_z(b,a,b,a),0)
        self.assertEqual(audit.relative_z(5e-324,1e308,5e-324,1e308),0)
        levels=[1e308 if i%2==0 else 5e-324 for i in range(13)]
        own=np.ones(253);own[::21]=levels
        got=audit.measure_oracle(own,np.ones(253))
        self.assertTrue(got['available']);self.assertEqual(got['candidate'],0.)
        self.assertTrue(all(isinstance(Fraction(z),Fraction) for z in got['exact_relative_z']))

    def test_tie_rank_pair_formula_matches_independent_sorting(self):
        levels=[1,2,2,8,4,4,8,32,16,8,8,16,8]
        got=audit.measure_oracle(prices_from_levels(levels),np.ones(253))
        vals=[abs(Fraction(x)) for x in got['exact_relative_z']]
        expected=[2*sum(v<x for v in vals)+sum(v==x for v in vals)+1 for x in vals]
        self.assertEqual(got['twice_magnitude_ranks'],expected)
        self.assertEqual(got['candidate'],sum(s*r for s,r in zip(got['relative_signs'],expected))/208)

    def test_whole_path_validation_rejects_nonendpoint_bad_values(self):
        for index in (0,1,20,21,251,252):
            for bad,reason in ((True,'nonnumeric_'),('2','nonnumeric_'),(None,'nonnumeric_'),
                    (0.,'invalid_'),(-1.,'invalid_'),(math.nan,'invalid_'),(math.inf,'invalid_'),(10**400,'invalid_')):
                stock=np.ones(253,dtype=object);stock[index]=bad
                got=audit.measure_oracle(stock,np.ones(253))
                self.assertEqual(got['reason'],reason+'stock_price');self.assertEqual(got['twice_rank_sum'],0)
        stock=np.ones(253);bench=np.ones(253,dtype=object);bench[3]=True
        self.assertEqual(audit.measure_oracle(stock,bench)['reason'],'nonnumeric_benchmark_price')
        stock[3]=0
        self.assertEqual(audit.measure_oracle(stock,bench)['reason'],'invalid_stock_price')
        with self.assertRaises(ValueError):audit.measure_oracle(stock[:-1],bench)

    def test_positive_binary_rescale_invariance_and_immutability(self):
        stock=prices_from_levels([2.**(i%4) for i in range(13)]);bench=np.ones(253)
        before=(stock.copy(),bench.copy());expected=audit.measure_oracle(stock,bench)
        for scale in (2.**-500,2.**500):
            audit.record_check(audit.measure_oracle(stock*scale,bench/scale),expected)
        np.testing.assert_array_equal(stock,before[0]);np.testing.assert_array_equal(bench,before[1])


class FeatureTests(unittest.TestCase):
    def test_complete_audit_counts_flags_and_control192_parity(self):
        import monthly_sector_recurrence192_independent as prior
        args=fixture();v,a=audit.independent_features(*args,mapper)
        old,_=prior.independent_features(*args,mapper)
        audit.exact_array(v['control'],old['candidate'],'old192 relative control')
        audit.exact_array(v['available'],old['available'],'old192 support')
        check=audit.verify_features(*args,mapper,v,a)
        self.assertTrue(all(check[k] is True for k in audit.FEATURE_FLAGS))
        N=a['available_cells']
        for k,multiple in [('source_price_observations',506),('relative_sign_observations',12),
                ('magnitude_rank_observations',12),('magnitude_pair_comparisons',66)]:self.assertEqual(a[k],multiple*N)

    def test_original_mapping_etf_foreign_and_missing_are_neutral(self):
        args=fixture();v,a=audit.independent_features(*args,mapper)
        e=a['event_history'][0]
        self.assertEqual(e['benchmarks']['BABA'],'XLY');self.assertIsNone(e['benchmarks']['FUND'])
        self.assertEqual(e['cells'][3]['reason'],'etf_neutral')
        self.assertFalse(v['available'][0,3])
        def bad(*unused):return {s:'XLK' for s in args[1]['symbols'][0]}
        with self.assertRaises(ValueError):audit.independent_features(*args,bad)

    def test_full_source_window_and_no_price_after_formation_end(self):
        args=fixture(count=1);before=audit.independent_features(*args,mapper)
        end=before[1]['event_history'][0]['price_end']
        args[0]['adj'].loc[args[0]['adj'].index>pd.Timestamp(end),:]=np.nan
        after=audit.independent_features(*args,mapper)
        audit.record_check(after[1],before[1])
        for k in before[0]:audit.exact_array(after[0][k],before[0][k],k)

    def test_query_known_mapping_ignores_future_classifications(self):
        args=fixture(count=1);v,a=audit.independent_features(*args,mapper)
        args[0]['sectors'].loc[args[0]['sectors'].index>args[1]['dates'][0],:]='bad-future'
        audit.record_check(audit.independent_features(*args,mapper)[1],a)

    def test_missing_official_day_not_compressed(self):
        args=fixture(count=1);t=args[2].index(args[1]['dates'][0]);day=args[2][t-100]
        args[0]['adj']=args[0]['adj'].drop(index=day)
        v,a=audit.independent_features(*args,mapper)
        self.assertEqual(a['event_history'][0]['cells'][0]['reason'],'missing_official_session')
        self.assertFalse(v['available'].any())

    def test_metadata_calendar_and_empty_prefix(self):
        args=fixture();v,a=audit.independent_features(*args,mapper,limit=0)
        self.assertEqual(v['available'].shape,(0,4))
        p,m,c=physical_prefix(args,0);vv,aa=audit.independent_features(p,m,c,mapper)
        self.assertEqual(vv['available'].shape,(0,0))
        for kind in ('notmonth','unsorted','duplicate','nomonthwitness','badprefix'):
            p,m,c=copy.deepcopy(args)
            if kind=='notmonth':m['dates'][1]='2016-08-30'
            elif kind=='unsorted':c=c[::-1]
            elif kind=='duplicate':m['symbols'][0][1]='A'
            elif kind=='nomonthwitness':c=c[:c.index(m['dates'][-1])+1]
            with self.assertRaises(ValueError):audit.independent_features(p,m,c,mapper,limit=True if kind=='badprefix' else None)

    def test_value_audit_mutations_rejected_exactly(self):
        args=fixture(count=1);v,a=audit.independent_features(*args,mapper)
        for kind in ('raw','rank','ratio','denominator','copy','count','clock'):
            vv,aa=copy.deepcopy((v,a));cell=aa['event_history'][0]['cells'][0]
            if kind=='raw':vv['candidate'][0,0]=np.nextafter(vv['candidate'][0,0],math.inf)
            elif kind=='rank':cell['twice_magnitude_ranks'][0]+=1
            elif kind=='ratio':cell['exact_relative_z'][0]='0'
            elif kind=='denominator':aa['candidate_denominator']=156
            elif kind=='copy':cell['constant_magnitude_copy']=not cell['constant_magnitude_copy']
            elif kind=='count':aa['magnitude_pair_comparisons']+=1
            else:cell['windows']['formation']['price_end']=args[1]['dates'][0]
            with self.assertRaises(ValueError):audit.verify_features(*args,mapper,vv,aa)

    def test_all_physical_and_logical_feature_prefixes(self):
        args=fixture();full=audit.independent_features(*args,mapper)
        for n in range(4):
            v,a=audit.independent_features(*args,mapper,limit=n)
            audit.record_check(a['event_history'],full[1]['event_history'][:n])
            for k in v:audit.exact_array(v[k],full[0][k][:n],k)
            p,m,c=physical_prefix(args,n);vv,aa=audit.independent_features(p,m,c,mapper)
            audit.record_check(aa['event_history'],a['event_history'])
            if n:
                for k in v:audit.exact_array(v[k],vv[k],k)
            else:self.assertEqual(vv['available'].shape,(0,0))


class DeltaSelectionTests(unittest.TestCase):
    def test_linear_initial_missing_and_no_cross_sectional_ranking(self):
        values={'available':np.array([[1,1],[1,0],[1,1]],bool),
            'candidate':np.array([[.75,-.75],[44/208,np.nan],[-.75,0.]]),
            'control':np.array([[.75,-.75],[.125,np.nan],[.75,0.]])}
        d=audit.independent_deltas(values);check=audit.verify_deltas(values,d)
        self.assertTrue(all(check[k] is True for k in audit.DELTA_FLAGS))
        self.assertEqual(d[audit.ARMS[1]][1,0],.025*(44/208))
        self.assertEqual(d[audit.ARMS[1]][0,0],0.);self.assertEqual(d[audit.ARMS[1]][1,1],0.)
        for key in d:
            wrong=copy.deepcopy(d);wrong[key][1,1]=1e-16
            with self.assertRaises(ValueError):audit.verify_deltas(values,wrong)
        with self.assertRaises(ValueError):audit.independent_deltas(values,dose=.05)

    def test_selection_actual_original_helper_and_exact_orders(self):
        args=with_ends(fixture());v,a=audit.independent_features(*args,mapper);o=original(args[1]);before=copy.deepcopy(o)
        scores,choices=audit.selection_oracle(o,args[1],v);orders=order_rows(scores,choices,args[1])
        receipt=audit.verify_selection(o,args[1],v,orders,scores=scores)
        self.assertTrue(all(receipt[k] is True for k in audit.SELECTION_FLAGS))
        for k in o:audit.exact_array(o[k],before[k],k)
        for arm in audit.ARMS[1:]:audit.exact_array(scores[arm][0],scores[audit.ARMS[0]][0],'initial')
        wrong=copy.deepcopy(orders);wrong[audit.ARMS[1]][1]['scores'][0]+=1e-12
        with self.assertRaises(ValueError):audit.verify_selection(o,args[1],v,wrong)
        wrong=copy.deepcopy(scores);wrong[audit.ARMS[1]][1,0]+=1e-12
        with self.assertRaises(ValueError):audit.verify_selection(o,args[1],v,orders,scores=wrong)

    def test_missing_and_allzero_raw_exact_original(self):
        args=with_ends(fixture());o=original(args[1]);n,w=o['baseline'].shape
        for available in (False,True):
            mask=np.full((n,w),available,bool);v={'available':mask,'candidate':np.where(mask,0.,np.nan),'control':np.where(mask,0.,np.nan)}
            scores,choices=audit.selection_oracle(o,args[1],v)
            for arm in audit.ARMS[1:]:audit.exact_array(scores[arm],scores[audit.ARMS[0]],arm);self.assertEqual(choices[arm],choices[audit.ARMS[0]])

    def test_selection_prefixes_and_cash_two_halves(self):
        args=with_ends(fixture());v,a=audit.independent_features(*args,mapper);o=original(args[1]);s,c=audit.selection_oracle(o,args[1],v)
        for n in range(4):
            ss,cc=audit.selection_oracle(o,args[1],v,n=n)
            for arm in audit.ARMS:audit.exact_array(ss[arm],s[arm][:n],arm);self.assertEqual(cc[arm],c[arm][:n])
        for count in (1,2):
            oo=copy.deepcopy(o);oo['eligible'][:]=False;oo['eligible'][:,:count]=True
            _,cc=audit.selection_oracle(oo,args[1],v)
            self.assertTrue(all(len(row)==(0 if count==1 else 2) for arm in audit.ARMS for row in cc[arm]))

    def test_budget_grouping_and_no_double_episode(self):
        args=with_ends(fixture());o=original(args[1]);o['state_weak_narrow'][:]=True
        mask=np.ones(o['baseline'].shape,bool);v={'available':mask,'candidate':np.zeros(mask.shape),'control':np.zeros(mask.shape)}
        s,c=audit.selection_oracle(o,args[1],v)
        x=audit.frozen_source_oracle().old.original_helper().original_prefix(o,args[1],None)
        budget=np.minimum(.05,np.maximum(0.,.10-(.025*x['calm']+.05*x['weak'])))
        expected=x['local'].copy()
        for i in range(3):
            for j in (2,3):expected[i,j]+=budget[i]*(min(.5,max(-.5,x['support'][i,j]+x['effects'][i,j]))-x['support'][i,j])
        for arm in audit.ARMS:audit.exact_array(s[arm],expected,arm)

    def test_no_new_producer_import_and_frozen_helper_pin(self):
        self.assertNotIn('import monthly_sector_signedrank233_kernel',inspect.getsource(audit))
        with patch.object(audit,'FROZEN_HELPER_SHA','bad'),self.assertRaises(ValueError):audit.accounting()


class OutcomeTests(unittest.TestCase):
    def test_independent_nine_paths_and_45_slices(self):
        import test_monthly_upside_retention181_independent as helper
        with patch.object(helper,'audit',audit):args=helper.account_fixture()
        receipt=audit.audit_outcomes(*args)
        self.assertEqual((receipt['checked_paths'],receipt['checked_slices']),(9,45))
        bad=copy.deepcopy(args);bad[4][audit.ARMS[1]]['same_close_zero']['metrics']['cagr']+=.01
        with self.assertRaises((ValueError,AssertionError)):audit.audit_outcomes(*bad)


class BilateralTests(unittest.TestCase):
    def test_both_independent_and_producer_fixtures_all_prefixes(self):
        import monthly_sector_signedrank233_kernel as producer
        from test_monthly_sector_signedrank233_kernel import fixture as producer_fixture
        for args in (fixture(),producer_fixture()):
            full=producer.build_features(*args,mapper)
            audit.verify_features(*args,mapper,*full)
            for n in range(len(args[1]['dates'])+1):
                logical=producer.build_features(*args,mapper,limit=n)
                audit.verify_features(*args,mapper,*logical,limit=n)
                pp=physical_prefix(args,n);out=producer.build_features(*pp,mapper)
                audit.verify_features(*pp,mapper,*out)
                audit.record_check(logical[1]['event_history'],out[1]['event_history'])
                if n:
                    for k in audit.VALUE_KEYS:audit.exact_array(out[0][k],full[0][k][:n],k)


if __name__=='__main__':unittest.main(verbosity=2)
