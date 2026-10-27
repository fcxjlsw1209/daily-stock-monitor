"""Synthetic independent273 tests; imported producer fixtures contain no I/O."""
from copy import deepcopy
import ast
import hashlib
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import monthly_matured_clock273_independent as o
import monthly_matured_clock273_kernel as k
from test_monthly_matured_clock273_kernel import fixture, refresh


def prepare(t,m,a,n=None):
    return k.fit_models(t,m,a,n=n,control_validator=lambda v:o.verify_control_models(t,m,a,v,n=n))


class Poison:
    def __iter__(self):raise AssertionError('Future body')
    def __getitem__(self,key):raise AssertionError('Future body')
    def __deepcopy__(self,memo):raise AssertionError('Future body')


class IndependentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t,cls.m,cls.a=fixture();cls.b=prepare(cls.t,cls.m,cls.a)

    def inputs(self):return deepcopy((self.t,self.m,self.a,self.b))

    def rejected(self,change):
        t,m,a,b=self.inputs();change(t,m,a,b)
        with self.assertRaises((ValueError,AssertionError,KeyError,TypeError)):
            o.verify_models(t,m,a,b)

    def test_independent_model_and_real_work(self):
        p=o.verify_models(self.t,self.m,self.a,self.b)
        self.assertTrue(p['passed']);self.assertFalse(p['producer_modules_imported'])
        self.assertFalse(p['source_files_authenticated']);self.assertFalse(p['labels_reconstructed'])
        self.assertEqual(p['work']['canonical_solves'],5);self.assertEqual(p['work']['svd_solves'],5)
        self.assertEqual(p['work']['original_local_fits'],0);self.assertEqual(p['work']['original_other_target_fits'],0)
        self.assertEqual(p['work']['training_rows_scanned'],2*self.b['audit']['counts']['training_rows'])

    def test_callback_is_before_every_monthly_fit(self):
        seen=[];real=k._record
        def record(s,kind,*args):seen.append(kind);return real(s,kind,*args)
        def gate(v):
            self.assertEqual(seen,['annual','annual']);self.assertIsNone(v['candidate'])
            return o.verify_control_models(self.t,self.m,self.a,v)
        with patch.object(k,'_record',side_effect=record):b=k.fit_models(self.t,self.m,self.a,control_validator=gate)
        self.assertEqual(seen,['annual','annual','monthly','monthly','monthly'])
        o.verify_models(self.t,self.m,self.a,b)

    def test_callback_reject_stops_monthly(self):
        kinds=[];real=k._record
        def record(s,kind,*args):kinds.append(kind);return real(s,kind,*args)
        def gate(v):
            v['control']['audit']['fits'][0]['weights'][0]*=2
            return o.verify_control_models(self.t,self.m,self.a,v)
        with patch.object(k,'_record',side_effect=record):
            with self.assertRaises(ValueError):k.fit_models(self.t,self.m,self.a,control_validator=gate)
        self.assertEqual(kinds,['annual','annual'])

    def test_independence_has_no_producer_or_source_import(self):
        tree=ast.parse(Path(o.__file__).read_text())
        modules=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        modules += [a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names]
        self.assertEqual(set(modules),{'datetime','hashlib','math','numpy','threadpoolctl'})
        self.assertEqual(o.dependency_files(),())

    def test_exact_annual_control_tamper(self):
        self.rejected(lambda t,m,a,b:b['control']['raw'].__setitem__((1,0),b['control']['raw'][1,0]+1e-12))

    def test_source_raw_not_alternate_cache(self):
        self.rejected(lambda t,m,a,b:a['raw_source_log'].__setitem__((1,0),a['raw_source_log'][1,0]+1e-13))

    def test_future_label_membership_tamper(self):
        self.rejected(lambda t,m,a,b:b['candidate']['audit']['fits'][0]['training_row_ids'].append(len(t['dates'])-1))

    def test_maturity_and_lower_bounds_tamper(self):
        for key,value in [('cutoff','2016-09-30'),('lower_bound','2011-08-30'),('training_months',999)]:
            with self.subTest(key=key):self.rejected(lambda t,m,a,b:b['candidate']['audit']['fits'][0].__setitem__(key,value))

    def test_row_order_symbol_identity_tamper(self):
        for key in ('training_row_ids','symbols','signal_dates','label_end_dates'):
            with self.subTest(key=key):
                def change(t,m,a,b):b['control']['audit']['fits'][0][key].reverse()
                self.rejected(change)

    def test_weights_and_recency_clock_tamper(self):
        self.rejected(lambda t,m,a,b:b['candidate']['audit']['fits'][0]['weights'].__setitem__(0,999.))
        self.rejected(lambda t,m,a,b:b['candidate']['audit']['fits'][0].__setitem__('weights_sha256','0'*64))

    def test_coefficients_rhs_scaler_and_prediction_tamper(self):
        for key in ('mean','scale','beta','rhs','raw_predictions'):
            with self.subTest(key=key):
                self.rejected(lambda t,m,a,b:b['candidate']['audit']['fits'][0][key].__setitem__(0,b['candidate']['audit']['fits'][0][key][0]+1e-14))

    def test_no_tolerance_for_rank_or_delta(self):
        rr,dd,aa=k.rank_and_delta(self.a,self.m['monthly'],self.b)
        for field in ('rank','delta'):
            r,d=rr.copy(),dd.copy()
            (r if field=='rank' else d)[1,0]+=1e-14
            with self.assertRaises(ValueError):o.verify_delta(self.a,self.m['monthly'],self.b,r,d,aa)

    def test_counts_label_hash_and_gate_schema_tamper(self):
        self.rejected(lambda t,m,a,b:b['audit']['counts'].__setitem__('monthly_ridge_fits',4))
        self.rejected(lambda t,m,a,b:b['audit']['label_preservation']['target_sha256'].__setitem__('simple_excess','0'*64))
        self.rejected(lambda t,m,a,b:b['audit'].__setitem__('control_gate',dict(passed=True)))

    def test_all_four_targets_and_common_month_two_rows(self):
        t,m,a,_=self.inputs();ids=np.flatnonzero(t['dates']==np.datetime64('2015-06-30'))
        t['target_loss_utility'][ids[:3]]=np.nan;refresh(t,m,a)
        b=prepare(t,m,a);o.verify_models(t,m,a,b)
        for r in b['control']['audit']['fits']+b['candidate']['audit']['fits']:
            self.assertFalse(set(ids)&set(r['training_row_ids']))

    def test_no_candidate_fit_from_unready_or_nonweak(self):
        self.assertEqual([r['query_indices'] for r in self.b['candidate']['audit']['fits']],[[1],[3],[5]])
        r,d,a=k.rank_and_delta(self.a,self.m['monthly'],self.b)
        p=o.verify_delta(self.a,self.m['monthly'],self.b,r,d,a)
        self.assertEqual(p['work']['delta_evaluations'],11)
        for i in (0,2,4):self.assertEqual(d[i].tobytes(),np.zeros(4).tobytes())

    def test_readiness_cannot_shrink_to_local_available_cells(self):
        self.rejected(lambda t,m,a,b:a['paired_month_ready'].__setitem__(4,True))

    def test_inactive_or_outside_U_mutation_rejected(self):
        for i,j in ((0,0),(2,0),(3,1),(4,0)):
            t,m,a,b=self.inputs();b['candidate']['raw'][i,j]=.123
            with self.assertRaises(ValueError):o.verify_models(t,m,a,b)

    def test_exact_ties_and_small_rank_domains(self):
        for values,expected in [([],[]),([0.],[.5]),([1.,1.],[.25,.25]),([3.,1.,2.],[.5,-1/6,1/6])]:
            actual=o.rank(values)
            np.testing.assert_allclose(actual,expected,atol=1e-16,rtol=0)
        t,m,a,_=self.inputs();t['event_plain'][:]=.05;t['train_plain'][:,3]=.1;refresh(t,m,a)
        b=prepare(t,m,a);o.verify_models(t,m,a,b)
        self.assertEqual(b['candidate']['rank'][1].tolist(),[.125]*4)

    def test_no_fit_or_training_access_for_empty_and_initial(self):
        t,m,a,_=self.inputs()
        for n in (0,1):
            b=prepare(t,m,a,n)
            poison={key:Poison() for key in t}
            with patch.object(o,'ridge_fit',side_effect=AssertionError('No fit')):
                p=o.verify_models(poison,m,a,b,n=n)
            self.assertEqual(p['work']['canonical_solves'],0);self.assertEqual(p['work']['training_rows_scanned'],0)

    def test_future_numeric_values_ignored_before_copy(self):
        t,m,a,_=self.inputs();b=prepare(t,m,a,2);future=t['ends']>np.datetime64('2016-08-31')
        for target in o.TARGETS:t['target_'+target][future]=np.inf
        t['train_plain'][future]=np.inf
        m['training']['symbols'][-1]=Poison()
        o.verify_models(t,m,a,b,n=2)

    def test_all_prefixes_physical_and_cached(self):
        for n in range(len(self.m['monthly']['dates'])+1):
            small=prepare(self.t,self.m,self.a,n);o.verify_models(self.t,self.m,self.a,small,n=n)
            pt,pm,pa,_=k.physical_inputs(self.t,self.m,self.a,n)
            physical=prepare(pt,pm,pa);o.verify_models(pt,pm,pa,physical)
            for role in ('control','candidate'):
                for key in ('raw','rank','available'):
                    self.assertEqual(small[role][key].tobytes(),physical[role][key].tobytes())
            rr,dd,aa=k.rank_and_delta(self.a,self.m['monthly'],self.b,n=n)
            p=o.verify_delta(self.a,self.m['monthly'],self.b,rr,dd,aa,n=n)
            self.assertEqual(p['work']['canonical_solves'],0)

    def test_cached_future_records_and_prediction_poison(self):
        for n in (0,1,2):
            t,m,a,b=self.inputs();rr,dd,aa=k.rank_and_delta(a,m['monthly'],b,n=n)
            b['audit']=Poison()
            for role in ('control','candidate'):
                b[role]['audit']=Poison();b[role]['raw'][n:]=np.inf;b[role]['rank'][n:]=np.inf
            o.verify_delta(a,m['monthly'],b,rr,dd,aa,n=n)

    def test_consumed_cached_availability_or_raw_tamper(self):
        for field in ('available','raw'):
            t,m,a,b=self.inputs();r,d,au=k.rank_and_delta(a,m['monthly'],b,n=2)
            if field=='available':b['candidate'][field][1,0]=False
            else:b['control'][field][1,0]+=.001
            with self.assertRaises(ValueError):o.verify_delta(a,m['monthly'],b,r,d,au,n=2)

    def test_no_prediction_prefix_does_not_borrow_full_year_model(self):
        t,m,a,_=self.inputs();a['eligible'][1]=False;refresh(t,m,a)
        full=prepare(t,m,a);small=prepare(t,m,a,2)
        self.assertEqual(small['control']['audit']['fits'][0]['status'],'no_prediction_rows')
        self.assertIsNone(small['control']['audit']['fits'][0]['beta'])
        o.verify_models(t,m,a,small,n=2)
        small['control']['audit']['fits'][0]=deepcopy(full['control']['audit']['fits'][0])
        with self.assertRaises(ValueError):o.verify_models(t,m,a,small,n=2)

    def test_cold_annual_never_expands_original_U(self):
        t,m,a,_=self.inputs();t['common_good'][t['dates']<np.datetime64('2014-08-01')]=False;refresh(t,m,a)
        b=prepare(t,m,a);o.verify_models(t,m,a,b)
        self.assertEqual(b['control']['audit']['fits'][0]['status'],'cold_start')
        self.assertNotIn([1],[r['query_indices'] for r in b['candidate']['audit']['fits']])

    def test_monthly_insufficient_history_cannot_fallback(self):
        t,m,a,_=self.inputs();t['common_good'] &= (t['dates']>=np.datetime64('2011-01-01'))&(t['dates']<=np.datetime64('2012-06-30'))
        refresh(t,m,a);inp=o._header(t,m,a,2)
        with self.assertRaisesRegex(ValueError,'at least18'):o._expected(inp,True,o._work())

    def test_bad_metadata_and_native_prefix(self):
        for n in (-1,True,99):
            with self.assertRaises(ValueError):o.verify_models(self.t,self.m,self.a,self.b,n=n)
        for field in ('offset','duplicate','querydate'):
            t,m,a,b=self.inputs()
            if field=='offset':t['offsets'][20]+=1
            elif field=='duplicate':m['training']['symbols'][20][0]='B'
            else:t['event_dates'][4]+=np.timedelta64(1,'D')
            with self.assertRaises(ValueError):o.verify_models(t,m,a,b)

    def test_svd_is_independent_and_failure_is_not_rank_tolerance(self):
        with patch.object(np.linalg,'svd',side_effect=np.linalg.LinAlgError('independent failure')):
            with self.assertRaises(np.linalg.LinAlgError):o.verify_models(self.t,self.m,self.a,self.b)

    def test_source_immutable_and_preservation_guard(self):
        t,m,a,b=self.inputs();before={k:v.tobytes() for k,v in t.items()}
        o.verify_models(t,m,a,b)
        self.assertEqual(before,{k:v.tobytes() for k,v in t.items()})
        real=o.ridge_fit;done=[]
        def mutate(*args):
            result=real(*args)
            if not done:t['target_log_excess'][100]+=.01;done.append(True)
            return result
        with patch.object(o,'ridge_fit',side_effect=mutate):
            with self.assertRaisesRegex(ValueError,'preservation'):o.verify_models(t,m,a,b)

    def test_signed_zero_in_inactive_raw_is_not_snapped(self):
        t,m,a,_=self.inputs();a['raw_source_log'][3,1]=-0.
        b=prepare(t,m,a);o.verify_models(t,m,a,b)
        b['candidate']['raw'][3,1]=0.
        with self.assertRaises(ValueError):o.verify_models(t,m,a,b)

    def test_five_year_leap_and_month_weight_mass(self):
        self.assertEqual(str(o.lower_bound(np.datetime64('2020-02-29'))),'2015-02-28')
        ds=np.asarray(['2015-01-31','2015-01-31','2015-02-28'],dtype='datetime64[D]')
        weights=o.month_weights(ds,np.datetime64('2015-03-31'))
        self.assertAlmostEqual(float(weights.sum()),3.);self.assertEqual(weights[0],weights[1])
        self.assertAlmostEqual((weights[0]+weights[1])/weights[2],2.**(-1/12))


if __name__=='__main__':unittest.main()
