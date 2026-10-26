"""Invented prices, original-score fixtures, source metadata and explicit ledgers."""
import copy
import hashlib
import importlib.util
import itertools
import json
import math
import pickle
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import monthly_portfolio_path247_independent as audit


def load_fixture():
    return audit.load_pinned(audit.PRIOR_TEST,audit.PRIOR_TEST_SHA,'portfolio247_test_original_fixture')


def review_for(meta,calendar):
    windows=[]
    for i,day in enumerate(meta['dates']):
        t=calendar.index(day);dates=calendar[t-294:t-41]
        windows.append(dict(query_index=i,date=day,initial_partial=day=='2016-08-17',query_session_position=t,
            source_start_position=t-294,source_end_position=t-42,price_start=dates[0],price_end=dates[-1],
            first_return_date=dates[1],next_official_session=calendar[t+1],prices=253,return_intervals=252,
            source_dates_sha256=hashlib.sha256(json.dumps(dates,separators=(',',':')).encode()).hexdigest()))
    return dict(calendar=dict(windows=windows),listing_bounds=[],listing_identity_cells=[])


def fixture():
    a,m=load_fixture().fixture()
    idx=pd.bdate_range('2013-01-02',max(m['ends'])+' 00:00:00')
    calendar=idx.strftime('%Y-%m-%d').tolist();t=np.arange(len(idx),dtype=float)
    names=sorted(set(itertools.chain.from_iterable(m['symbols'])))
    frame=pd.DataFrame({s:50.*np.exp((.0002+j*.00003)*t+.07*np.sin(t*(.071+j*.017))) for j,s in enumerate(names)},index=idx)
    return {'adj':frame},m,calendar,a,review_for(m,calendar)


def physical(args,n):
    p,m,c,a,r=copy.deepcopy(args)
    a={k:v[:n].copy() for k,v in a.items()};m={k:(v[:n] if k in ('dates','ends','symbols') else v) for k,v in m.items()}
    if n:
        end=c.index(m['dates'][-1]);c=c[:end+2];p['adj']=p['adj'].loc[:m['dates'][-1]].copy()
    else:c=[];p['adj']=p['adj'].iloc[:0].copy()
    r['calendar']['windows']=r['calendar']['windows'][:n]
    r['listing_identity_cells']=[x for x in r['listing_identity_cells'] if x['query_index']<n]
    return p,m,c,a,r


def run(args,n=None):
    p,m,c,a,r=args;features=audit.independent_features(p,m,c,a,r,n)
    scores,choices,pres,decision=audit.selection_oracle(a,m,features,n)
    orders=audit.orders_from_choices(scores,choices,m)
    return features,orders,scores,pres,decision


def verify(args,out,n=None):
    p,m,c,a,r=args;f,o,s,pres,d=out
    return audit.verify_prepared(p,m,c,a,r,f,o,scores=s,pre_scores=pres,audit=d,n=n)


def listing(args,index=1,j=None,visible=True):
    p,m,c,a,r=args;base=audit.original_base(a,m,len(m['dates']))
    if j is None:j=base['pre_order'][index][0]
    symbol=m['symbols'][index][j];bound=dict(symbol=symbol,issuer_key='synthetic-issuer',not_before='2016-06-01',
        available_from='2016-06-02' if visible else '2099-01-02',kind='synthetic',evidence_id='synthetic-evidence')
    r['listing_bounds']=[bound]
    r['listing_identity_cells']=[]
    for i,names in enumerate(m['symbols']):
        if symbol in names:
            w=r['calendar']['windows'][i]
            r['listing_identity_cells'].append(dict(query_index=i,column_index=names.index(symbol),date=m['dates'][i],
                symbol=symbol,issuer_key=bound['issuer_key'],new_price_start=w['price_start'],new_price_end=w['price_end'],
                bound_visible=bound['available_from']<=m['dates'][i],start_precedes_bound=w['price_start']<bound['not_before']))
    return j


class MathTests(unittest.TestCase):
    def test_fixed_units_and_covariance_decomposition(self):
        p=[[1.,2.,2.4,2.1],[1.,.8,1.1,1.2],[1.,1.2,1.5,1.6]]
        r=audit.measure_oracle(p)
        self.assertEqual(r['normalized_prices'],p)
        self.assertAlmostEqual(r['portfolio_returns'][0],1./3.)
        self.assertAlmostEqual(r['portfolio_returns'][1],1.25-1.)
        self.assertNotAlmostEqual(r['portfolio_returns'][1],sum((p[i][2]/p[i][1]-1.)/3 for i in range(3)))
        self.assertAlmostEqual(r['joint_variance'],r['diagonal_variance']+r['cross_covariance'])
        self.assertLessEqual(r['joint_variance'],3*r['diagonal_variance']+1e-15)
        self.assertTrue(r['diagnostics']['nav_identity_passed'])

    def test_flat_zero_and_zero_joint_with_positive_diagonal_are_common_unavailable(self):
        flat=audit.measure_oracle([[1.,1.,1.]]*3)
        self.assertFalse(flat['available']);self.assertIsNone(flat['joint_ratio']);self.assertEqual(flat['diagonal_variance'],0.)
        hedge=audit.measure_oracle([[1.,1.25,1.],[1.,.75,1.],[1.,1.,1.]])
        self.assertEqual(hedge['joint_variance'],0.);self.assertGreater(hedge['diagonal_variance'],0.)
        self.assertFalse(hedge['available']);self.assertIsNone(hedge['diagonal_ratio'])

    def test_negative_mean_is_valid_and_identical_paths_ratio_relation(self):
        p=[1.,.9,.92,.82,.85];r=audit.measure_oracle([p,p,p])
        self.assertTrue(r['available']);self.assertLess(r['mean_return'],0.)
        self.assertAlmostEqual(r['joint_variance'],3*r['diagonal_variance'])
        self.assertAlmostEqual(r['diagonal_ratio'],math.sqrt(3)*r['joint_ratio'])

    def test_scaling_and_column_permutation_preserve_portfolio_mean_and_risk(self):
        paths=[[1.,1.1,.95,1.2],[2.,1.8,2.2,2.1],[4.,4.2,3.9,4.5]]
        left=audit.measure_oracle(paths);right=audit.measure_oracle([[v*8 for v in paths[2]],paths[0],paths[1]])
        for key in ('mean_return','joint_variance','diagonal_variance','joint_ratio','diagonal_ratio'):
            self.assertAlmostEqual(left[key],right[key],places=13)

    def test_invalid_shape_source_overflow_and_false_zero_underflow_hard_fail(self):
        for bad in (True,'1',0.,-1.,float('inf'),float('nan')):
            with self.subTest(bad=bad),self.assertRaises(ValueError):audit.measure_oracle([[1.,bad,1.]]*3)
        with self.assertRaises(ValueError):audit.measure_oracle([[1.,1.]]*3)
        with self.assertRaises(ArithmeticError):audit.measure_oracle([[1e308,5e-324,1e308]]*3)
        with self.assertRaises(ArithmeticError):audit.measure_oracle([[5e-324,1e308,1.]]*3)
        with self.assertRaises(ArithmeticError):audit.product(1e-250,1e-250)

    def test_rank_ties_zero_and_singleton(self):
        self.assertEqual(audit.rank_values([0.,0.,0.]),[2./3.-.5]*3)
        self.assertEqual(audit.rank_values([0.]),[.5])
        self.assertEqual(audit.rank_values([-1.,0.,1.]),[1/3-.5,2/3-.5,.5])


class SourceSelectionTests(unittest.TestCase):
    def test_full_independent_receipt_counts_budget_and_no_mutation(self):
        args=fixture();before=pickle.dumps(args);out=run(args);check=verify(args,out)
        self.assertTrue(all(check[f] for f in audit.PREPARE_FLAGS));self.assertEqual(check['ordered_choices_checked'],12)
        self.assertEqual(check['counts'],out[-1]['counts']);self.assertEqual(check['counts']['original_episode_calls'],1)
        self.assertEqual(check['counts']['new_model_fits'],0);self.assertEqual(check['continuous_atol'],2e-10)
        self.assertEqual(pickle.dumps(args),before)
        for arm in audit.ARMS:audit.exact_array(out[2][arm],out[2][audit.ARMS[0]],'Common original F')

    def test_original_frozen_episode_exact_and_calm_extension_budget(self):
        args=fixture();p,m,c,a,r=args
        a['extension_available'][:]=True;a['forecast_available'][:]=True;a['rank_log_excess'][:]=.2
        base=audit.original_base(a,m,3);s,choices,pres,_,_=audit.old._selection(a,m,3)
        audit.exact_array(base['F'],s[audit.old.ARMS[0]],'Independent original F')
        self.assertEqual(base['choices'],choices[audit.old.ARMS[0]])
        self.assertGreater(base['budget_evidence']['calm_nonweak_extension_count'],0)
        verify(args,run(args))

    def test_initial_and_two_halves_cash_do_not_read_prices(self):
        args=physical(fixture(),1);args[0]['adj']=args[0]['adj'].astype(object);args[0]['adj'][:]='not-a-price'
        out=run(args);self.assertEqual(out[0][0]['branch'],'initial_partial');self.assertEqual(out[0][0]['source_cells'],[])
        for arm in audit.ARMS:self.assertEqual(out[1][arm],out[1][audit.ARMS[0]])
        for count in (0,1,2):
            args=fixture();args[3]['eligible'][:,count:]=False;args[0]['adj']=args[0]['adj'].astype(object);args[0]['adj'][:]='not-a-price'
            out=run(args);self.assertEqual(out[0][1]['branch'],'fewer_than_three')
            for arm in audit.ARMS:self.assertEqual(len(out[1][arm][1]['symbols']),2 if count==2 else 0)

    def test_top5_and_regret_caps_do_not_replace_missing_with_rank6(self):
        args=fixture();out=run(args);ev=out[0][1]
        self.assertEqual(ev['shortlist_indices'],[0,1,2,3,4]);self.assertEqual(len(ev['trios']),10)
        H=[ev['trios'][j] for j in ev['feasible_trio_indices']]
        self.assertTrue(any(t['indices']==ev['incumbent_trio'] for t in H))
        self.assertTrue(all(t['raw_core_mean']>=ev['reference_raw_core_mean']-.02 and t['final_mean']>=ev['reference_final_mean']-.02 for t in H))
        args[0]['adj']=args[0]['adj'].drop(columns=[ev['symbols'][0]])
        out=run(args);self.assertEqual(out[0][1]['branch'],'missing_source')
        self.assertEqual(out[0][1]['shortlist_indices'],ev['shortlist_indices'])
        for arm in audit.ARMS:self.assertEqual(out[1][arm][1],out[1][audit.ARMS[0]][1])

    def test_missing_row_nan_observation_counts_and_bad_remaining_value_hardfail(self):
        args=fixture();out=run(args);ev=out[0][1];j=ev['shortlist_indices'][0];s=ev['symbols'][j]
        args[0]['adj']=args[0]['adj'].drop(index=ev['source_dates'][100])
        out=run(args);self.assertEqual(out[0][1]['source_cells'][0]['observations_checked'],252)
        self.assertEqual(out[0][1]['source_cells'][0]['reason'],'missing_official_session')
        args[0]['adj'].loc[ev['source_dates'][101],s]=float('inf')
        with self.assertRaises(ValueError):run(args)
        args=fixture();args[0]['adj'].loc[ev['source_dates'][100],s]=np.nan
        out=run(args);self.assertEqual(out[0][1]['source_cells'][0]['reason'],'missing_nan')
        self.assertEqual(out[0][1]['source_cells'][0]['observations_checked'],253)

    def test_missing_early_shortlist_cannot_hide_later_malformed_source(self):
        args=fixture();ev=run(args)[0][1]
        args[0]['adj']=args[0]['adj'].drop(columns=ev['symbols'][ev['shortlist_indices'][0]])
        args[0]['adj'].loc[ev['source_dates'][5],ev['symbols'][ev['shortlist_indices'][-1]]]=0.
        with self.assertRaises(ValueError):run(args)

    def test_visible_listing_contradiction_not_hidden_by_other_missing_source(self):
        args=fixture();j=listing(args);ev=run(fixture())[0][1]
        other=next(k for k in ev['shortlist_indices'] if k!=j)
        args[0]['adj']=args[0]['adj'].drop(columns=ev['symbols'][other])
        with self.assertRaises(ValueError):run(args)
        args=fixture();listing(args,visible=False);verify(args,run(args))
        args=physical(fixture(),2);j=listing(args);args[0]['adj'].loc[ev['source_dates'][0],ev['symbols'][j]]=np.nan
        out=run(args);self.assertEqual(out[0][1]['branch'],'missing_source')

    def test_review_clock_and_identity_tampering_rejected(self):
        for mutate in (lambda r:r['calendar']['windows'][1].update(price_start='2000-01-01'),
                       lambda r:r['calendar']['windows'][1].update(source_dates_sha256='0'*64),
                       lambda r:r['calendar']['windows'][1].update(prices=True)):
            args=fixture();mutate(args[-1])
            with self.assertRaises(ValueError):run(args)
        args=fixture();listing(args,visible=False);args[-1]['listing_identity_cells'][1]['issuer_key']='changed'
        with self.assertRaises(ValueError):run(args)

    def test_incomplete_or_noncanonical_calendar_is_hardfail_not_fallback(self):
        args=fixture();args[2][:]=args[2][-50:]
        with self.assertRaises(ValueError):run(args)
        args=fixture();args[2][1]=args[2][0]
        with self.assertRaises(ValueError):run(args)

    def test_zero_variance_in_any_feasible_trio_common_fallback_and_all_statistics_retained(self):
        args=fixture();args[0]['adj'][:]=1.
        out=run(args);ev=out[0][1];self.assertEqual(ev['branch'],'zero_variance')
        self.assertEqual(sum(t['statistics'] is not None for t in ev['trios']),len(ev['feasible_trio_indices']))
        for arm in audit.ARMS:self.assertEqual(out[1][arm],out[1][audit.ARMS[0]])
        verify(args,out)

    def test_constant_ratio_ranks_not_forced_zero_and_structure_ties_prefer_incumbent(self):
        args=fixture();a=args[3];a['core'][:]=0.;a['baseline'][:]=0.;a['pre_score'][:]=0.;a['effects'][:]=0.
        frame=args[0]['adj'];path=frame.iloc[:,0].copy()
        for col in frame.columns:frame[col]=path
        out=run(args);ev=out[-1]['events'][1]
        for k in ev['feasible_trio_indices']:
            self.assertAlmostEqual(ev['trios'][k]['joint_q'],.5/len(ev['feasible_trio_indices']))
        for arm in audit.ARMS:self.assertEqual(out[1][arm][1],out[1][audit.ARMS[0]][1])

    def test_exact_utility_funding_and_same_means(self):
        args=fixture();out=run(args);ev=out[-1]['events'][1];scale=1.-.025/(1.-ev['core_weight'])
        for k in ev['feasible_trio_indices']:
            t=ev['trios'][k]
            self.assertEqual(t['joint_utility'],t['weighted_core_mean']+scale*t['auxiliary_mean']+.025*t['joint_q'])
            self.assertEqual(t['structural_utility'],t['final_mean'])
            stats=t['statistics'];self.assertEqual(stats['joint_ratio']*stats['joint_sd'],stats['mean_return'])
        self.assertEqual(ev['noncore_scale'],scale)

    def test_full_audit_scores_orders_and_rank_tampering_detected(self):
        args=fixture();out=run(args)
        for part,mutate in ((2,lambda x:x[audit.ARMS[1]].__setitem__((1,0),99.)),
            (3,lambda x:x[audit.ARMS[2]].__setitem__((1,0),99.)),
            (4,lambda x:x['counts'].__setitem__('new_model_fits',1)),
            (4,lambda x:x['events'][1]['choices'][audit.ARMS[1]].reverse()),
            (0,lambda x:x[1]['trios'][0].__setitem__('raw_core_mean',99.))):
            bad=copy.deepcopy(out);mutate(bad[part])
            with self.assertRaises(ValueError):verify(args,bad)

    def test_continuous_tolerance_never_accepts_changed_zero_gate(self):
        ref=audit.measure_oracle([[1.,1.,1.]]*3);got=copy.deepcopy(ref);got['joint_variance']=1e-30
        with self.assertRaises(ValueError):audit.continuous_check(got,ref)
        got=copy.deepcopy(ref);got['mean_return']=1e-12;audit.continuous_check(got,ref)

    def test_continuous_nearness_cannot_change_exact_ratio_ranks(self):
        args=fixture();frame=args[0]['adj'];path=frame.iloc[:,0].copy()
        for col in frame.columns:frame[col]=path
        features,_,_,_,_=run(args);ev=features[1]
        self.assertGreater(len(ev['feasible_trio_indices']),1)
        t=ev['trios'][ev['feasible_trio_indices'][0]]['statistics']
        t['mean_return']+=1e-15
        t['joint_ratio']=t['mean_return']/t['joint_sd'];t['diagonal_ratio']=t['mean_return']/t['diagonal_sd']
        a,m=args[3],args[1];scores,choices,pres,decision=audit.selection_oracle(a,m,features)
        out=(features,audit.orders_from_choices(scores,choices,m),scores,pres,decision)
        # Cached arithmetic is self-consistent; full independent source ranks reject it.
        audit.verify_selection(a,m,features,out[1],scores=scores,pre_scores=pres,audit=decision)
        with self.assertRaises(ValueError):verify(args,out)

    def test_all_logical_and_physical_prefixes_and_future_poison(self):
        args=fixture();full=run(args)
        for n in range(len(args[1]['dates'])+1):
            logical=run(args,n);physical_args=physical(args,n);actual=run(physical_args)
            audit.record_check(logical[0],full[0][:n],'Feature prefix');audit.record_check(logical[0],actual[0],'Physical features')
            audit.record_check(logical[1],actual[1],'Physical choices');verify(physical_args,actual)
            self.assertEqual(logical[-1]['events'],full[-1]['events'][:n]);verify(args,logical,n)
        poisoned=copy.deepcopy(args);poisoned[1]['symbols'][-1]=['bad']
        poisoned[0]['adj']=poisoned[0]['adj'].astype(object)
        poisoned[0]['adj'].loc[poisoned[0]['adj'].index>args[1]['dates'][1]]='future-invalid'
        run(poisoned,2)

    def test_cached_selection_does_not_read_source_and_binds_original_input(self):
        args=fixture();out=run(args);p,m,c,a,r=args;features,orders,scores,pres,d=out
        with patch.object(audit,'measure_oracle',wraps=audit.measure_oracle) as called:
            audit.verify_selection(a,m,features,orders,scores=scores,pre_scores=pres,audit=d)
            called.assert_not_called()
        a['pre_score'][1,0]+=.01
        with self.assertRaises(ValueError):audit.verify_selection(a,m,features,orders,scores=scores,pre_scores=pres,audit=d)


class AccountingTests(unittest.TestCase):
    @staticmethod
    def args():
        h=audit.load_pinned(audit.ACCOUNT_TEST,audit.ACCOUNT_TEST_SHA,'portfolio247_account_fixture')
        p,o,orders,exports,results,m=h.account_fixture();first=next(iter(orders))
        return p,o,{k:copy.deepcopy(orders[first]) for k in audit.ARMS},\
            {k:copy.deepcopy(exports[first]) for k in audit.ARMS},{k:copy.deepcopy(results[first]) for k in audit.ARMS},m

    def test_twelve_paths_sixty_slices_and_twentyfour_training_groups(self):
        report=audit.audit_outcomes(*self.args())
        self.assertEqual((report['checked_paths'],report['checked_slices'],report['checked_training_groups']),(12,60,24))
        self.assertEqual(set(report['paths']),set(audit.ARMS))

    def test_cost_only_diagnostic_and_ledger_mutation_rejected(self):
        args=list(self.args());args[3][audit.ARMS[1]]['same_close_zero']['HOLDINGS'][0]['fee']=1e-15
        with self.assertRaises(ValueError):audit.audit_outcomes(*args)
        args=list(self.args());args[4][audit.ARMS[3]]['next_open_cost']['slices']['later34']['cagr']+=.1
        with self.assertRaises((ValueError,AssertionError)):audit.audit_outcomes(*args)


class BilateralCross(unittest.TestCase):
    def test_producer_fixture_full_artifacts_and_all_logical_physical_prefixes(self):
        import monthly_portfolio_path247_kernel as producer
        import test_monthly_portfolio_path247_kernel as fixtures
        # The one permitted cross: the new producer is never an oracle import.
        own=fixture()
        own_args=(load_fixture().real_study(),own[3],own[1],own[0],own[2],own[4])
        for args in (fixtures.fixture(),own_args):
            study,arrays,meta,panel,calendar,review=args
            full=producer.prepare(*args,len(meta['dates']))
            verify((panel,meta,calendar,arrays,review),full)
            for n in range(len(meta['dates'])+1):
                f,o,s,p,a=producer.prepare(*args,n)
                audit.verify_prepared(panel,meta,calendar,arrays,review,f,o,scores=s,pre_scores=p,audit=a,n=n)
                oo,ss,pp,aa=producer.decisions(study,arrays,meta,full[0],n)
                audit.verify_selection(arrays,meta,full[0],oo,scores=ss,pre_scores=pp,audit=aa,n=n)
                physical_args=fixtures.physical(args,n)
                ps,pa,pm,ppanel,pc,pr=physical_args
                f,o,s,p,a=producer.prepare(*physical_args,n)
                audit.verify_prepared(ppanel,pm,pc,pa,pr,f,o,scores=s,pre_scores=p,audit=a,n=n)



if __name__=='__main__':
    if '--self-only' in sys.argv:
        sys.argv.remove('--self-only')
        suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(c) for c in (MathTests,SourceSelectionTests,AccountingTests))
        result=unittest.TextTestRunner(verbosity=2).run(suite);sys.exit(not result.wasSuccessful())
    unittest.main(verbosity=2)
