"""258 invented-source proofs; producer imported only in the final bilateral test."""
import copy
import hashlib
import math
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import monthly_downside_basket258_independent as audit

fixture_module=audit.load_pinned(audit.PRIOR_TEST,audit.PRIOR_TEST_SHA,'downside258_independent_fixture')
fixture=fixture_module.fixture
physical=fixture_module.physical

def run(args,n=None):
    p,m,c,a,r=args;f=audit.independent_features(p,m,c,a,r,n)
    s,ch,pre,d=audit.selection_oracle(a,m,f,n)
    return f,audit.orders_from_choices(s,ch,m),s,pre,d

def verify(args,out,n=None):
    p,m,c,a,r=args;f,o,s,pre,d=out
    return audit.verify_prepared(p,m,c,a,r,f,o,scores=s,pre_scores=pre,audit=d,n=n)

def extend(rows):return [list(map(float,row))+[float(row[-1])]*(253-len(row)) for row in rows]

class Mathematics(unittest.TestCase):
    def test_flat_is_valid_zero(self):
        s=audit.measure_oracle([[3.]*253,[7.]*253,[11.]*253])
        self.assertTrue(s['zero_downside']);self.assertTrue(s['zero_symmetric'])
        self.assertEqual(s['zero_returns'],252);self.assertEqual(s['symmetric_moment'],0.)
        audit.statistics_check(s)

    def test_increasing_path_zero_downside_positive_symmetric(self):
        s=audit.measure_oracle([[float(1+k) for k in range(253)]]*3)
        self.assertTrue(s['zero_downside']);self.assertFalse(s['zero_symmetric'])
        self.assertEqual(s['positive_returns'],252)
        self.assertEqual(s['symmetric_square_sum'],math.fsum(r*r for r in s['portfolio_returns']))

    def test_negative_and_mixed_decomposition(self):
        s=audit.measure_oracle(extend([[1,1.5,1.1,1.8],[1,.8,.5,.7],[1,1,1,1]]))
        self.assertGreater(s['negative_returns'],0);self.assertGreater(s['positive_returns'],0)
        self.assertAlmostEqual(s['symmetric_moment'],s['downside_moment']+s['upside_moment'])
        self.assertEqual(s['downside_square_sum'],math.fsum(min(r,0.)*min(r,0.) for r in s['portfolio_returns']))

    def test_fixed_units_not_daily_equal_weight(self):
        rows=extend([[1,2,3],[1,1,1],[1,1,1]])
        s=audit.measure_oracle(rows)
        self.assertEqual(s['portfolio_returns'][1],.25)
        daily_equal=math.fsum(rows[i][2]/rows[i][1]-1 for i in range(3))/3
        self.assertNotEqual(s['portfolio_returns'][1],daily_equal)
        self.assertAlmostEqual(s['direct_returns'][1],.25)

    def test_exact_representable_scaling_and_permutation(self):
        rows=extend([[1,2,1.5],[2,3,2],[4,4.5,5]])
        s=audit.measure_oracle(rows)
        z=audit.measure_oracle([[v*8 for v in rows[2]],rows[0],rows[1]])
        for key in ('portfolio_returns','downside_moment','symmetric_moment','upside_moment'):
            audit.record_check(s[key],z[key],key)

    def test_underflow_is_not_legal_zero(self):
        for fn,args in ((audit.multiply,(1e-200,1e-200)),(audit.quotient,(5e-324,252.))):
            with self.assertRaises(ArithmeticError):fn(*args)
        rows=[[2.]*253 for _ in range(3)];rows[0][1]=5e-324
        with self.assertRaises(ArithmeticError):audit.measure_oracle(rows)

    def test_shape_and_source_types(self):
        for invalid in (True,None,'1',0.,-1.,math.inf,math.nan):
            rows=[[1.]*253 for _ in range(3)];rows[0][5]=invalid
            with self.assertRaises((ValueError,ArithmeticError)):audit.measure_oracle(rows)
        with self.assertRaises(ValueError):audit.measure_oracle([[1.]*252]*3)

    def test_exact_objective_and_sign_mutations(self):
        good=audit.measure_oracle(extend([[1,1.25,.9]]*3))
        for key in ('downside_moment','symmetric_square_sum'):
            bad=copy.deepcopy(good);bad[key]=float(np.nextafter(bad[key],math.inf))
            with self.assertRaises(ValueError):audit.statistics_check(bad)
        bad=copy.deepcopy(good);bad['negative_returns']+=1
        with self.assertRaises(ValueError):audit.statistics_check(bad)

    def test_nav_is_only_continuous_diagnostic(self):
        s=audit.measure_oracle(extend([[1,2,3]]*3));s['direct_returns'][0]+=1e-13
        audit.statistics_check(s)
        s['direct_returns'][0]+=1e-6
        with self.assertRaises(ValueError):audit.statistics_check(s)


class FullSourceAndSelection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.args=fixture();cls.out=run(cls.args)

    def test_full_receipts_and_preservation(self):
        r=verify(self.args,self.out);self.assertTrue(r['passed'])
        n=len(self.args[1]['dates']);self.assertEqual(r['ordered_choices_checked'],3*n)
        self.assertEqual(r['original_episode_calls'],2)
        self.assertEqual(r['descriptive_path_evaluations'],r['counts']['trio_statistics'])
        self.assertEqual(r['counts']['formation_return_observations'],252*r['counts']['trio_statistics'])
        for a in audit.ARMS[1:]:
            np.testing.assert_array_equal(self.out[2][a],self.out[2][audit.ARMS[0]])
            np.testing.assert_array_equal(self.out[3][a],self.out[3][audit.ARMS[0]])

    def test_source_before_selection_receipt(self):
        p,m,c,a,r=self.args;f=self.out[0]
        with patch.object(audit,'selection_oracle',side_effect=AssertionError('Must not select')):
            proof=audit.verify_features(p,m,c,a,r,f)
            cached=audit.verify_features_cached(a,m,f)
        self.assertFalse(proof['candidate_selection_computed']);self.assertTrue(proof['source_rebuilt'])
        self.assertEqual(cached['descriptive_path_evaluations'],0)

    def test_gate_receipts_full_and_cached(self):
        p,m,c,a,r=self.args;f,o,s,pre,d=copy.deepcopy(self.out)
        inc=audit.verify_incumbent(a,m,o[audit.ARMS[0]],s[audit.ARMS[0]],pre[audit.ARMS[0]])
        d['incumbent_gate']=inc;d['source_gate']=audit.verify_features(p,m,c,a,r,f)
        verify(self.args,(f,o,s,pre,d))
        d['source_gate']=audit.verify_features_cached(a,m,f)
        audit.verify_selection(a,m,f,o,scores=s,pre_scores=pre,audit=d)
        d['source_gate']['candidate_selection_computed']=True
        with self.assertRaises(ValueError):audit.verify_selection(a,m,f,o,scores=s,pre_scores=pre,audit=d)

    def test_all_logical_and_physical_prefixes(self):
        p,m,c,a,r=self.args
        for n in range(len(m['dates'])+1):
            out=run(self.args,n);verify(self.args,out,n)
            part=physical(self.args,n);other=run(part,n);verify(part,other,n)
            audit.record_check(out[0],other[0],'Prefix sources');audit.record_check(out[1],other[1],'Prefix orders')
            self.assertEqual(other[4]['counts']['original_episode_calls'],1)

    def test_future_metadata_and_feature_bodies_unconsumed(self):
        class Poison(dict):
            def __deepcopy__(self,memo):raise AssertionError('Future deepcopy')
            def __eq__(self,o):raise AssertionError('Future compare')
        p,m,c,a,r=copy.deepcopy(self.args);f=copy.deepcopy(self.out[0]);n=1
        for i in range(n,len(m['dates'])):
            f[i]=Poison();r['calendar']['windows'][i]=Poison(query_index=i)
        out=run((p,m,c,a,r),n);verify((p,m,c,a,r),out,n)
        audit.verify_features_cached(a,m,f,n)

    def test_partial_and_two_names_do_not_read_prices(self):
        p,m,c,a,r=copy.deepcopy(self.args)
        p['adj']=p['adj'].astype(object);p['adj'].iloc[:,:]='poison'
        out=run((p,m,c,a,r),1);self.assertEqual(out[0][0]['branch'],'initial_partial')
        a['eligible'][1,:]=False;a['eligible'][1,:2]=True
        out=run((p,m,c,a,r),2);self.assertEqual(out[0][1]['branch'],'fewer_than_three')
        for arm in audit.ARMS:self.assertEqual(out[1][arm],out[1][audit.ARMS[0]])

    def test_missing_source_common_incumbent_fallback(self):
        p,m,c,a,r=copy.deepcopy(self.args);e=self.out[0][1];symbol=e['symbols'][e['shortlist_indices'][0]]
        p['adj'].at[pd.Timestamp(e['source_dates'][100]),symbol]=math.nan
        out=run((p,m,c,a,r),2);self.assertEqual(out[0][1]['branch'],'missing_source')
        self.assertTrue(all(t['statistics'] is None for t in out[0][1]['trios']))
        for arm in audit.ARMS:self.assertEqual(out[1][arm][1],out[1][audit.ARMS[0]][1])

    def test_missing_cannot_hide_another_invalid_source(self):
        p,m,c,a,r=copy.deepcopy(self.args);e=self.out[0][1];j,k=e['shortlist_indices'][:2]
        p['adj'].at[pd.Timestamp(e['source_dates'][100]),e['symbols'][j]]=math.nan
        p['adj'].at[pd.Timestamp(e['source_dates'][200]),e['symbols'][k]]=0.
        with self.assertRaises(ValueError):run((p,m,c,a,r),2)

    def test_missing_row_and_column_never_substitute_rank6(self):
        for mode in ('row','column'):
            p,m,c,a,r=copy.deepcopy(self.args);e=self.out[0][1]
            if mode=='row':p['adj']=p['adj'].drop(pd.Timestamp(e['source_dates'][100]))
            else:p['adj']=p['adj'].drop(columns=e['symbols'][e['shortlist_indices'][0]])
            out=run((p,m,c,a,r),2)
            self.assertEqual(out[0][1]['shortlist_indices'],e['shortlist_indices'])
            self.assertEqual(out[0][1]['branch'],'missing_source')

    def test_visible_listing_and_future_listing(self):
        for visible in (True,False):
            args=copy.deepcopy(self.args);fixture_module.listing(args,visible=visible)
            if visible:
                with self.assertRaises(ValueError):run(args,2)
            else:verify(args,run(args,2),2)

    def test_zero_objectives_incumbent_tie(self):
        p,m,c,a,r=copy.deepcopy(self.args);p['adj'].iloc[:,:]=10.
        out=run((p,m,c,a,r));e=out[0][1];d=out[4]['events'][1]
        self.assertEqual(e['branch'],'available')
        for arm in audit.ARMS[1:]:
            self.assertEqual(d['tied_trio_indices'][arm],e['feasible_trio_indices'])
            self.assertEqual(d['choices'][arm],e['incumbent_choices'])
            self.assertEqual(e['trios'][d['winner_trio_indices'][arm]]['indices'],e['incumbent_trio'])
        verify((p,m,c,a,r),out)

    def test_cached_statistics_and_source_endpoint_tampering(self):
        p,m,c,a,r=self.args
        for kind in ('moment','endpoint','cap','choices','count'):
            f,o,s,pre,d=copy.deepcopy(self.out);e=f[1];k=e['feasible_trio_indices'][0]
            if kind=='moment':e['trios'][k]['statistics']['downside_moment']=float(np.nextafter(e['trios'][k]['statistics']['downside_moment'],math.inf))
            elif kind=='endpoint':e['source_cells'][0]['end_price']*=1.01
            elif kind=='cap':e['trios'][k]['feasible']=not e['trios'][k]['feasible']
            elif kind=='choices':o[audit.ARMS[1]][1]['symbols']=[]
            else:d['counts']['formation_return_observations']+=1
            with self.assertRaises(ValueError):audit.verify_selection(a,m,f,o,scores=s,pre_scores=pre,audit=d)

    def test_exact_incumbent_gate_rejects_score_or_order_mutation(self):
        p,m,c,a,r=self.args;f,o,s,pre,d=self.out;arm=audit.ARMS[0]
        bad=s[arm].copy();bad[1,0]=np.nextafter(bad[1,0],math.inf)
        with self.assertRaises(ValueError):audit.verify_incumbent(a,m,o[arm],bad,pre[arm])
        orders=copy.deepcopy(o[arm]);orders[1]['symbols']=[]
        with self.assertRaises(ValueError):audit.verify_incumbent(a,m,orders,s[arm],pre[arm])

    def test_source_window_is_253_ending_skip42(self):
        p,m,c,a,r=self.args
        for e in self.out[0]:
            t=c.index(e['date']);self.assertEqual(e['source_dates'],c[t-294:t-41])
            self.assertEqual(len(e['source_dates']),253)

    def test_original_arrays_and_input_metadata_are_unchanged(self):
        p,m,c,a,r=copy.deepcopy(self.args);before={k:v.tobytes() for k,v in a.items()}
        mm=copy.deepcopy(m);rr=copy.deepcopy(r);pp=p['adj'].copy()
        verify((p,m,c,a,r),run((p,m,c,a,r)))
        self.assertEqual(before,{k:v.tobytes() for k,v in a.items()})
        self.assertEqual(m,mm);self.assertEqual(r,rr);pd.testing.assert_frame_equal(p['adj'],pp)


class Accounting(unittest.TestCase):
    @staticmethod
    def args():
        p,o,orders,exports,results,m=fixture_module.AccountingTests.args();first=next(iter(orders))
        # Older181/247 fixture predates the saved full122 aggregate field.
        # Supply its already independently generated total in the modern schema.
        for result in results[first].values():
            result['slices']['full122']['fees_paid']=result['metrics']['fees_paid']
        return p,o,{a:copy.deepcopy(orders[first]) for a in audit.ARMS},\
          {a:copy.deepcopy(exports[first]) for a in audit.ARMS},{a:copy.deepcopy(results[first]) for a in audit.ARMS},m

    def test_nine_paths_fortyfive_slices_eighteen_training_groups(self):
        proof=audit.audit_outcomes(*self.args())
        self.assertEqual((proof['checked_paths'],proof['checked_slices'],proof['checked_training_groups']),(9,45,18))
        self.assertEqual(set(proof['paths']),set(audit.ARMS))

    def test_zero_period_trade_and_aggregate_tiny_fees_rejected(self):
        for kind in ('period','trade','metric','slice'):
            args=list(self.args());ex=args[3][audit.ARMS[1]]['same_close_zero'];r=args[4][audit.ARMS[1]]['same_close_zero']
            if kind=='period':ex['HOLDINGS'][0]['fee']=1e-20
            elif kind=='trade':ex['TRADES'][0]['cost']=1e-20
            elif kind=='metric':r['metrics']['fees_paid']=1e-20
            else:r['slices']['full122']['fees_paid']=1e-20
            with self.assertRaises(ValueError):audit.audit_outcomes(*args)

    def test_cost_slice_mutation_rejected(self):
        args=list(self.args());args[4][audit.ARMS[2]]['next_open_cost']['slices']['later34']['cagr']+=.1
        with self.assertRaises((ValueError,AssertionError)):audit.audit_outcomes(*args)


class BilateralCross(unittest.TestCase):
    def test_full_source_decisions_host_gates_and_every_synthetic_prefix(self):
        # The sole permitted producer import is this bilateral test; expected
        # source, canonical statistics, original scores and choices remain ours.
        import monthly_downside_basket258_kernel as producer
        import test_monthly_downside_basket258_kernel as fixtures
        import monthly_downside_basket258_host as host
        for path,digest in (
          ('monthly_downside_basket258_kernel.py','a4dc0dc30efa12d6ddcf9ea486d89ec28dd259718c4bd2e88a05b00143e33ace'),
          ('test_monthly_downside_basket258_kernel.py','52b51f62bb4fef77bff2e8afac696e6302cebc0cabdbe50a7f2259eb5a587863')):
            self.assertEqual(hashlib.sha256((audit.ROOT/'work'/path).read_bytes()).hexdigest(),digest)
        own=fixture();study=fixture_module.load_fixture().real_study()
        own_args=(study,own[3],own[1],own[0],own[2],own[4])
        for args in (fixtures.fixture(),own_args):
            study,arrays,meta,panel,calendar,review=args;n=len(meta['dates']);width=arrays['eligible'].shape[1]
            base=audit.original_base(arrays,meta,n)
            expected=audit.orders_from_choices({a:base['F'] for a in audit.ARMS},{a:base['choices'] for a in audit.ARMS},meta)[audit.ARMS[0]]
            full,item=host.prepare_one(*args,n,expected,width=width)
            self.assertEqual(full[1][audit.ARMS[0]],expected)
            for count in range(n+1):
                cached,proof=host.decisions(study,arrays,meta,full[0],count,expected[:count],item['gates']['source'],width=width)
                part=fixtures.physical(args,count)
                rebuilt,physical_proof=host.prepare_one(*part,count,expected[:count],width=width)
                audit.record_check(full[0][:count],rebuilt[0],'Physical source prefix')
                audit.record_check(cached[0],rebuilt[1],'Cached versus physical orders')
                audit.record_check(cached[3]['events'],rebuilt[4]['events'],'Cached versus physical audit events')
                for arm in audit.ARMS:
                    audit.exact_array(cached[1][arm],rebuilt[2][arm],'Final scores')
                    audit.exact_array(cached[2][arm],rebuilt[3][arm],'Pre scores')
                self.assertEqual(proof['selection']['cached_path_validations'],cached[3]['counts']['trio_statistics'])
                self.assertEqual(physical_proof['full']['original_episode_calls'],2)
            # Future containers must not be inspected by either implementation.
            class Poison(dict):
                def __deepcopy__(self,memo):raise AssertionError('Future copied')
                def __eq__(self,other):raise AssertionError('Future compared')
            for count in (0,1):
                poisoned=copy.deepcopy(full[0]);pm=copy.deepcopy(meta)
                for j in range(count,n):
                    poisoned[j]=Poison()
                    for key in ('dates','ends','symbols'):pm[key][j]=Poison()
                o,s,p,d=producer.decisions(study,arrays,pm,poisoned,count)
                audit.verify_selection(arrays,pm,poisoned,o,scores=s,pre_scores=p,audit=d,n=count)
            # A rejected saved incumbent must terminate before new moments.
            bad=copy.deepcopy(expected);bad[0]['symbols']=['wrong']
            with patch.object(producer,'measure',side_effect=AssertionError('New moments before incumbent gate')):
                with self.assertRaises(ValueError):host.prepare_one(*args,n,bad,width=width)

if __name__=='__main__':unittest.main(verbosity=2)
