"""Invented separate history axes; never open source caches or compute returns."""
import ast
import calendar
import copy
import math
import unittest
from pathlib import Path

import numpy as np
import monthly_joint_error226_model as frozen
from test_monthly_joint_error226_model import fixture as old_fixture, month_pair
import monthly_separate_history230_model as model


def fixture(n_history=18, symbols=('A','B','C','D','E','F'), start=(2013,6),
            current=((2015,1),(2015,6))):
    old = old_fixture(n_history=n_history, symbols=symbols, start=start, current=current)
    train, tm, hm, source, use, pre, eligible = old
    first = n_history
    qm = {k: copy.deepcopy(v[first:]) for k,v in hm.items()}
    args = [train, tm, qm, copy.deepcopy(hm), source, use, pre[first:].copy(), eligible[first:].copy()]
    current_available = source['available'].reshape(use.shape)[first:].copy()
    return args, current_available


def physical_prefix(value, n):
    args, current_available = copy.deepcopy(value)
    train, tm, qm, hm, source, use, pre, eligible = args
    if type(n) is not int or not 0 <= n <= len(qm['dates']):
        raise ValueError('Invalid current synthetic prefix')
    width = use.shape[1]
    smallqm = {k: copy.deepcopy(v[:n]) for k,v in qm.items()}
    last = qm['dates'][n-1] if n else None
    cutoff = str(int(last[:4])-1)+'-12-31' if n else None
    history_n = sum(d <= last for d in hm['dates']) if n else 0
    smallhm = {k: copy.deepcopy(v[:history_n]) for k,v in hm.items()}
    keep = [i for i,e in enumerate(tm['ends']) if cutoff is not None and e <= cutoff]
    ids = np.array([j for i in keep for j in range(train['offsets'][i],train['offsets'][i+1])], dtype=int)
    smalltm = {k:[copy.deepcopy(v[i]) for i in keep] for k,v in tm.items()}
    smalltrain = {k:np.asarray(v)[ids].copy() for k,v in train.items() if k!='offsets'}
    smalltrain['offsets'] = np.r_[0,np.cumsum(np.array([len(tm['symbols'][i]) for i in keep],dtype=int))]
    smallsource = {k:np.asarray(v)[:history_n*width].copy() for k,v in source.items()}
    return ([smalltrain,smalltm,smallqm,smallhm,smallsource,use[:history_n].copy(),
             pre[:n].copy(),eligible[:n].copy()], current_available[:n].copy())


def run(value):
    args, current_available = value
    return model.build(*args, current_available=current_available)


def event(output, i=-1):
    return output['audit']['event_history'][i]


def snapshot(value):
    if isinstance(value, dict): return {k:snapshot(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)): return [snapshot(v) for v in value]
    a=np.asarray(value)
    return (a.dtype.str,a.shape,a.tobytes())


def equivalence_fixture():
    train,tm,qm,source,use,pre,eligible = old_fixture()
    return ([train,tm,qm,copy.deepcopy(qm),source,use,pre,eligible],
            source['available'].reshape(use.shape).copy())


def partial_fixture():
    # Exactly 67 early natural months precede the original partial event.
    base = old_fixture(n_history=67,start=(2011,1),current=((2016,8),(2018,1)))
    train,tm,hm,source,use,pre,eligible=base
    insert=67; width=use.shape[1]
    hm=copy.deepcopy(hm)
    hm['dates'].insert(insert,model.INITIAL);hm['ends'].insert(insert,'2016-08-31')
    hm['symbols'].insert(insert,list(hm['symbols'][0]))
    source={k:np.insert(v,insert*width,np.zeros(width,dtype=v.dtype)) for k,v in source.items()}
    source['available'][insert*width:(insert+1)*width]=True
    use=np.insert(use,insert,True,axis=0)
    qm={k:copy.deepcopy(v[insert:]) for k,v in hm.items()}
    currentpre=np.vstack([pre[insert],pre[insert:]])
    currenteligible=np.ones(currentpre.shape,bool)
    return ([train,tm,qm,hm,source,use,currentpre,currenteligible],np.ones(currentpre.shape,bool))


class FrozenMathTests(unittest.TestCase):
    def test_measure_and_rank_asts_exactly_frozen(self):
        old=ast.parse(Path(frozen.__file__).read_text());new=ast.parse(Path(model.__file__).read_text())
        names=('measure','effect_ranks','_sum','_square','_divide','_finite','_number','_domain')
        for name in names:
            a=next(x for x in old.body if isinstance(x,ast.FunctionDef) and x.name==name)
            b=next(x for x in new.body if isinstance(x,ast.FunctionDef) and x.name==name)
            self.assertEqual(ast.dump(a),ast.dump(b),name)

    def test_analytic_joint_vs_diagonal_values_unchanged(self):
        matrix=[[1.,1.,1.,-1.,0.],[-1.,-1.,-1.,1.,0.]]
        z=model.measure(matrix)
        self.assertEqual(z['candidate'],[2.,2./9.,8./9.])
        self.assertEqual(z['control'],[2./3.,2./3.,4./9.])
        self.assertEqual(model.effect_ranks(z['candidate']),[-.5,.5,0.])

    def test_exact_226_equivalence_when_history_equals_current(self):
        f=equivalence_fixture();a=f[0]
        old=frozen.build(a[0],a[1],a[2],a[4],a[5],a[6],a[7])
        new=run(f)
        for key in model.ARRAY_KEYS:np.testing.assert_array_equal(new['arrays'][key],old['arrays'][key])
        normalized=copy.deepcopy(new['audit'])
        self.assertEqual(normalized.pop('history_months'),len(a[2]['dates']))
        self.assertTrue(normalized.pop('separate_history_axis'))
        normalized['arms']=list(frozen.ARMS)
        self.assertEqual(normalized,old['audit'])

    def test_zero_and_underflow_semantics_unchanged(self):
        self.assertEqual(model.measure([[1.,2.,3.]]*18)['candidate'],[0.])
        self.assertEqual(model.effect_ranks([0.,0.,0.]),[0.,0.,0.])
        for matrix in ([[1e-200,0.,0.],[-1e-200,0.,0.]],[[1e308,0.,0.],[-1e308,0.,0.]]):
            with self.assertRaises(FloatingPointError):model.measure(matrix)


class SeparateAxisTests(unittest.TestCase):
    def test_early_history_makes_shared_H_without_new_current_queries(self):
        f=fixture();z=run(f)
        self.assertEqual(z['arrays']['ready'].tolist(),[True,True])
        self.assertEqual(z['audit']['query_months'],2);self.assertEqual(z['audit']['history_months'],20)
        self.assertEqual(len(event(z)['H']),18)
        self.assertEqual(event(z)['ordered_top2'],[0,1]);self.assertEqual(event(z)['J'],[2,3,4])
        self.assertEqual(event(z)['counts']['error_observations'],90)
        self.assertEqual(z['arrays']['candidate_effect'][-1].tolist(),[0.,0.,-.5,.5,0.,0.])
        a=f[0]
        a[3]={k:v[-2:] for k,v in a[3].items()};a[4]={k:v[-12:] for k,v in a[4].items()};a[5]=a[5][-2:]
        self.assertFalse(run(f)['arrays']['ready'].any())

    def test_seventeen_complete_months_remains_common_fallback(self):
        z=run(fixture(n_history=17))
        self.assertFalse(z['arrays']['ready'].any())
        self.assertEqual(event(z)['status'],'insufficient_history')
        self.assertEqual(z['audit']['parameter_estimates'],0)

    def test_current_cached_count_uses_explicit_current_axis_only(self):
        f=fixture();f[1][:]=False;f[1][0,0]=True;f[1][1,:2]=True
        z=run(f)
        self.assertEqual([e['counts']['cached_forecast_rows'] for e in z['audit']['event_history']],[1,2])
        self.assertEqual(z['audit']['counts']['cached_forecast_rows'],3)
        self.assertTrue(z['arrays']['ready'].all())
        self.assertEqual(event(z)['J'],[2,3,4])

    def test_current_availability_does_not_change_risk_or_effects(self):
        f=fixture();a=run(f);f[1][:]=False;b=run(f)
        for key in model.ARRAY_KEYS:np.testing.assert_array_equal(a['arrays'][key],b['arrays'][key])
        self.assertEqual(b['audit']['counts']['cached_forecast_rows'],0)

    def test_all_counts_are_sums_of_current_events_and_truthful_parameters(self):
        z=run(fixture());c=z['audit']['counts']
        self.assertEqual(tuple(c),model.COUNT_FIELDS)
        self.assertEqual(c['parameter_estimates'],32)
        for key in model.COUNT_FIELDS:
            self.assertEqual(c[key],sum(e['counts'][key] for e in z['audit']['event_history']))
        self.assertEqual(z['audit']['new_statistical_estimation_queries'],2)
        for key in ('source_model_refits','new_ridge_fits','new_tree_fits'):self.assertEqual(z['audit'][key],0)
        self.assertNotIn('new_model_fits',z['audit'])

    def test_partial_at_history_index67_skipped_but_current_initial_fallback(self):
        f=partial_fixture();self.assertEqual(f[0][3]['dates'][67],model.INITIAL)
        z=run(f)
        self.assertEqual(event(z,0)['status'],'initial_partial')
        self.assertEqual(event(z)['counts']['skipped_initial_months'],1)
        self.assertNotIn(model.INITIAL,event(z)['H'])
        self.assertTrue(z['arrays']['ready'][-1])

    def test_partial_terminal_history_prefix_needs_no_future_august31(self):
        f=physical_prefix(partial_fixture(),1)
        self.assertEqual(f[0][3]['dates'][-1],model.INITIAL)
        z=run(f)
        self.assertEqual(z['audit']['history_months'],68)
        self.assertEqual(event(z)['status'],'initial_partial')

    def test_partial_pair_bad_end_or_supplied_nonadjacent_formal_month_rejected(self):
        for bad in ('end','next','duplicate'):
            f=partial_fixture();hm=f[0][3]
            if bad=='end':hm['ends'][67]='2016-09-30'
            elif bad=='next':hm['dates'][68]='2016-09-30';hm['ends'][68]='2016-10-31'
            else:hm['dates'][68]='2016-08-18'
            with self.subTest(bad=bad),self.assertRaises(ValueError):run(f)

    def test_no_arbitrary_extra_partial_month_exception(self):
        f=fixture();f[0][3]['ends'][0]=f[0][3]['dates'][0]
        with self.assertRaises(ValueError):run(f)
        f=fixture();f[0][2]['dates'][1]=model.INITIAL;f[0][2]['ends'][1]='2016-08-31'
        with self.assertRaises(ValueError):run(f)

    def test_physical_current_prefix_cuts_history_by_date_not_row_count(self):
        f=fixture();full=run(f)
        for n in range(len(f[0][2]['dates'])+1):
            small=physical_prefix(f,n);z=run(small)
            self.assertEqual(z['audit']['event_history'],full['audit']['event_history'][:n])
            for key in model.ARRAY_KEYS:np.testing.assert_array_equal(z['arrays'][key],full['arrays'][key][:n])
            if n:self.assertGreater(len(small[0][3]['dates']),n)
            else:
                self.assertEqual(small[0][3]['dates'],[])
                self.assertTrue(all(v==0 for v in z['audit']['counts'].values()))

    def test_every_old_axis_physical_prefix_retains_226_equivalence(self):
        f=equivalence_fixture();full=run(f)
        for n in range(len(f[0][2]['dates'])+1):
            z=run(physical_prefix(f,n))
            self.assertEqual(z['audit']['event_history'],full['audit']['event_history'][:n])
            for key in model.ARRAY_KEYS:np.testing.assert_array_equal(z['arrays'][key],full['arrays'][key][:n])

    def test_future_history_and_current_poison_removed_before_model(self):
        f=fixture();expected=run(physical_prefix(f,1));a=f[0]
        a[4]['raw_local_linear']=a[4]['raw_local_linear'].astype(object)
        a[4]['raw_local_linear'][-6:]='future_forecast'
        a[6]=a[6].astype(object);a[6][-1]='future_score'
        f=(a,f[1].astype(object));f[1][-1]='future_available'
        # The real prefix exposes only a bool current mask, not future poison.
        small=physical_prefix(f,1);small=(small[0],small[1].astype(bool))
        self.assertEqual(run(small)['audit'],expected['audit'])

    def test_future_training_targets_and_features_not_consumed(self):
        f=fixture();before=run(f);train=f[0][0]
        train['train_plain']=train['train_plain'].astype(object);train['train_plain'][-12:]='future_plain'
        for key in model.TARGETS:
            name='target_'+key;train[name]=train[name].astype(object);train[name][-12:]='future_label'
        self.assertEqual(run(f)['audit'],before['audit'])

    def test_history_cutoff_and_exclusive_five_year_lower_bound(self):
        f=fixture(n_history=62,start=(2010,12),current=((2017,1),))
        row=event(run(f));self.assertEqual(row['cutoff'],'2016-12-31')
        self.assertEqual(row['lookback_exclusive_start'],'2011-12-31')
        self.assertTrue(all(d>'2011-12-31' for d in row['H']))
        self.assertNotIn('2011-12-31',row['H'])

    def test_missing_security_drops_one_common_month_not_one_column(self):
        f=fixture(n_history=19,start=(2013,5));f[0][3]['symbols'][0][4]='OTHER'
        z=run(f);row=event(z)
        self.assertEqual(len(row['H']),18);self.assertTrue(z['arrays']['ready'][-1])
        self.assertEqual(row['counts']['missing_security_rows'],1)
        self.assertEqual(row['counts']['incomplete_history_months'],1)

    def test_historical_use_and_source_mask_not_current_available_gate_H(self):
        for which in ('use','source'):
            f=fixture(n_history=19,start=(2013,5));a=f[0]
            if which=='use':a[5][0,3]=False
            else:a[4]['available'][3]=False;a[4]['raw_local_linear'][3]=np.nan
            row=event(run(f));self.assertEqual(len(row['H']),18)
            self.assertEqual(row['counts']['unsupported_source_rows'],1)

    def test_exact_training_join_hard_error_only_for_mature_consumed_row(self):
        f=fixture();f[0][1]['symbols'][0][3]='OTHER'
        with self.assertRaisesRegex(ValueError,'Missing mature exact'):run(f)
        f[0][5][0,3]=False
        self.assertEqual(event(run(f))['status'],'insufficient_history')

    def test_zero_complete_risk_is_ready_noop_and_not_fallback(self):
        f=fixture();f[0][4]['raw_local_linear'][:108]=np.tile([1.,2.,3.,4.,5.,6.],18)
        z=run(f);self.assertTrue(z['arrays']['ready'].all())
        self.assertTrue((z['arrays']['candidate_effect']==0.).all())
        self.assertEqual(event(z)['counts']['parameter_estimates'],16)

    def test_top2_boundary_gap_and_fifth_position_unchanged(self):
        f=fixture();f[0][6][-1]=[.4,.3,0.,-.02,-float(np.nextafter(.02,math.inf)),-.3]
        self.assertEqual(event(run(f))['J'],[2,3])
        f[0][6][-1]=[.4,.3,.2,.2,.2,.2]
        self.assertEqual(event(run(f))['J'],[2,3,4])
        self.assertNotIn(5,event(run(f))['B'])

    def test_current_cash_two_or_single_boundary_no_history_statistics(self):
        for count in range(4):
            f=fixture();f[0][7][-1]=False;f[0][7][-1,:count]=True
            e=event(run(f));self.assertEqual(e['status'],'insufficient_candidates')
            self.assertEqual(e['history'],[])

    def test_empty_history_is_valid_and_all_current_rows_fallback(self):
        f=fixture();a=f[0];a[3]={k:[] for k in a[3]};a[4]={k:v[:0] for k,v in a[4].items()};a[5]=a[5][:0]
        z=run(f);self.assertFalse(z['arrays']['ready'].any());self.assertEqual(z['audit']['history_months'],0)
        self.assertEqual(z['audit']['counts']['cached_forecast_rows'],12)

    def test_masks_widths_shapes_and_dates_fail_closed(self):
        for which in ('current_flat','current_int','historywidth','prewidth','duplicate','offset','rawmask'):
            f=fixture();a=f[0]
            if which=='current_flat':f=(a,f[1].reshape(-1))
            elif which=='current_int':f=(a,f[1].astype(int))
            elif which=='historywidth':a[5]=a[5][:,:5]
            elif which=='prewidth':a[6]=a[6][:,:5]
            elif which=='duplicate':a[3]['symbols'][0][1]='A'
            elif which=='offset':a[0]['offsets']=a[0]['offsets'].astype(float)
            else:a[4]['raw_local_linear'][0]=np.nan
            with self.subTest(which=which),self.assertRaises(ValueError):run(f)

    def test_unused_source_fields_not_consumed_and_no_mutation(self):
        f=fixture();before=snapshot(f);expected=run(f);self.assertEqual(snapshot(f),before)
        for key in set(f[0][4])-{'available','raw_local_linear'}:f[0][4][key]=np.array(['unused'],object)
        self.assertEqual(run(f)['audit'],expected['audit'])

    def test_current_availability_mandatory_keyword_and_frozen_dependency_pinned(self):
        args,current=fixture()
        with self.assertRaises(TypeError):model.build(*args)
        self.assertIn(Path(frozen.__file__).resolve(),model.dependency_files())
        self.assertTrue(all(p.is_file() for p in model.dependency_files()))


if __name__=='__main__':
    unittest.main(verbosity=2)
