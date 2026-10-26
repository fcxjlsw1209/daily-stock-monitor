"""Round118 synthetic cross-implementation tests; no actual market arrays."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

spec=importlib.util.spec_from_file_location('listwise118_independent_tested',Path(__file__).with_name('monthly_listwise118_independent.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
producer=m.module('listwise118_synthetic_producer',m.P/'model.py')


def fixture():
    rng=np.random.default_rng(118)
    months=pd.date_range('2011-01-31',periods=72,freq='ME')
    dates=np.repeat(months.values.astype('datetime64[D]'),5)
    ends=np.repeat((months+pd.offsets.MonthEnd(1)).values.astype('datetime64[D]'),5)
    symbols=np.tile(np.array(list('ABCDE')),len(months));X=rng.uniform(-.49,.49,(len(dates),10))
    y=.2*X[:,0]-.1*X[:,1]+rng.normal(0,.02,len(dates))
    ordinal=np.concatenate([m.native_ordinal(row) for row in y.reshape(-1,5)])
    targets={key:y.copy() for key in m.TARGETS};good=np.ones(len(dates),bool)
    edates=np.array(['2015-03-31','2015-09-30','2016-06-30','2017-08-31'],dtype='datetime64[D]')
    EX=rng.uniform(-.45,.45,(len(edates),10))
    return [dates,ends,symbols,X,ordinal,targets,good.copy(),good.copy(),edates,np.array(['A','B','C','D']),EX,np.ones(len(edates),bool)]


def order_fixture():
    n,k=3,5;shape=(n,k);raw=np.tile(np.arange(k,dtype=float),(n,1));rank=np.tile(m.h.ordinal_rank(raw[0]),(n,1))
    baseline=np.tile([.3,.28,.22,.21,.20],(n,1));calm=np.array([False,True,False])
    a=dict(baseline=baseline,core=baseline*.5,core_weight=np.where(calm,.5,.52),eligible=np.ones(shape,bool),pre_score=baseline.copy(),effects=np.zeros(shape),state_weak_narrow=np.array([True,False,False]),state_calm10=calm,forecast_available=np.ones(shape,bool),rank_log_excess=rank,old_ml4=rank.copy(),extension_available=np.ones(shape,bool),model_available=np.ones(shape,bool),raw_local_constant=raw.copy(),raw_local_linear=raw.copy(),raw_source_log=raw.copy(),rank_local_linear=rank.copy(),listwise_model_available=np.ones(shape,bool),raw_matched_ordinal=raw.copy(),raw_listwise_top3=raw[:,::-1].copy())
    ds=pd.date_range('2020-01-31',periods=n+1,freq='ME').strftime('%Y-%m-%d').tolist();meta=dict(dates=ds[:-1],ends=ds[1:],symbols=[list('ABCDE') for _ in range(n)])
    return a,meta


def stats(cagr=.5,sharpe=1.5):
    return dict(training=dict(cagr=cagr,sharpe=sharpe,max_drawdown=-.3),early=dict(cagr=.4),recent=dict(cagr=.6))


class ListwiseMathTests(unittest.TestCase):
    def test_fractional_boundary_ties_and_permutations(self):
        values=np.array([5.,4.,3.,3.,1.]);expected=np.array([1/3,1/3,1/6,1/6,0.])
        np.testing.assert_array_equal(m.top3_distribution(values),expected)
        order=np.array([3,0,4,1,2]);np.testing.assert_array_equal(m.top3_distribution(values[order]),expected[order])
    def test_small_and_all_tied_uniform(self):
        for n in (1,2,3,7):np.testing.assert_allclose(m.top3_distribution(np.ones(n)),np.full(n,1/n),atol=1e-16)
        np.testing.assert_array_equal(m.top3_distribution([3.,1.,2.]),np.full(3,1/3))
    def test_ordinal_is_fixed_point_including_ties(self):
        for values in ([1.,2.,2.,4.],[1.],[5.]*8):
            ordinal=m.native_ordinal(values);np.testing.assert_array_equal(m.native_ordinal(ordinal),ordinal)
    def test_gradient_and_hessian_finite_differences(self):
        rng=np.random.default_rng(2);Z=rng.normal(size=(11,10));q=np.r_[m.top3_distribution(np.arange(5.)),m.top3_distribution(np.arange(6.))]
        groups=[np.arange(5),np.arange(5,11)];weights=[3.,5.];b=rng.normal(0,.1,10);f,g,H=m.grouped_objective(b,Z,q,groups,weights)
        delta=1e-5
        for j in range(10):
            step=np.eye(10)[j]*delta;fp,gp,_=m.grouped_objective(b+step,Z,q,groups,weights);fm,gm,_=m.grouped_objective(b-step,Z,q,groups,weights)
            self.assertAlmostEqual((fp-fm)/(2*delta),g[j],places=7)
            np.testing.assert_allclose((gp-gm)/(2*delta),H[:,j],atol=1e-8,rtol=0.)
        self.assertGreaterEqual(np.linalg.eigvalsh(H).min(),10.-1e-12)
    def test_group_shift_invariance_and_extreme_stable_scores(self):
        rng=np.random.default_rng(3);Z=rng.normal(size=(8,10));b=rng.normal(size=10);q=np.r_[m.top3_distribution(range(4)),m.top3_distribution(range(4))];groups=[np.arange(4),np.arange(4,8)]
        other=Z.copy();other[:4]+=2.;other[4:]-=3.
        for a,v in zip(m.grouped_objective(b,Z,q,groups,[2.,4.]),m.grouped_objective(b,other,q,groups,[2.,4.])):np.testing.assert_allclose(a,v,atol=1e-12,rtol=0.)
        self.assertTrue(np.isfinite(m.grouped_objective(b*1e4,Z,q,groups,[2.,4.])[0]))
    def test_hybrid_solver_certificate_and_uniform_zero(self):
        Z=np.zeros((5,10));q=m.top3_distribution(range(5));fit=m.fit_listwise(Z,q,[np.arange(5)],[4.]);np.testing.assert_array_equal(fit['coefficient'],np.zeros(10))
        self.assertEqual(fit['gradient_inf'],0.)
        with self.assertRaises(ValueError):m.scalar_certificate(np.ones(10),Z,q,[np.arange(5)],[4.])
    def test_group_membership_and_probability_errors_fatal(self):
        with self.assertRaises(ValueError):m.fit_listwise(np.zeros((4,10)),np.full(4,.25),[np.array([0,1,2])],[1.])
        with self.assertRaises(ValueError):m.fit_listwise(np.zeros((4,10)),np.ones(4),[np.arange(4)],[1.])


class AnnualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.args=fixture();cls.oracle=m.annual_oracle(*cls.args);cls.actual=producer.annual_forecasts(*cls.args)
    def test_all_raws_audits_and_certificates(self):
        proof=m.verify_model(self.actual,self.oracle);self.assertEqual(proof['annual_fits_per_arm'],3)
        self.assertTrue(self.oracle['available'].all())
    def test_whole_list_restriction_not_valid_sublist(self):
        args=copy.deepcopy(self.args);args[7][2]=False
        usable,q,records=m.complete_cohort(*args[:8]);self.assertFalse(usable[:5].any());self.assertTrue(np.isnan(q[:5]).all());self.assertFalse(records[0]['complete'])
    def test_all_four_targets_and_features_restrict_entire_month(self):
        for field in ('target','feature'):
            args=copy.deepcopy(self.args)
            if field=='target':args[5]['loss_utility'][0]=np.nan
            else:args[3][0,1]=np.nan
            usable,_,_=m.complete_cohort(*args[:8]);self.assertFalse(usable[:5].any());self.assertTrue(usable[5:].all())
    def test_noneligible_rows_do_not_enter_q(self):
        args=copy.deepcopy(self.args);args[6][4]=False;args[7][4]=False;args[4][:4]=m.native_ordinal(args[4][:4]);args[4][4]=np.nan
        usable,q,_=m.complete_cohort(*args[:8]);self.assertFalse(usable[4]);self.assertTrue(np.isnan(q[4]));self.assertAlmostEqual(q[:4].sum(),1.)
    def test_bad_ordinal_or_duplicate_identity_rejected(self):
        args=copy.deepcopy(self.args);args[4][0]+=.01
        with self.assertRaises(AssertionError):m.complete_cohort(*args[:8])
        args=copy.deepcopy(self.args);args[2][0]=args[2][1]
        with self.assertRaises(ValueError):m.complete_cohort(*args[:8])
    def test_weights_scaler_groups_and_annual_boundaries(self):
        for row in self.oracle['audit']['annual']:
            ids=np.array(row['training_indices']);self.assertTrue((self.args[0][ids]>np.datetime64(row['lookback_exclusive_start'])).all());self.assertTrue((self.args[1][ids]<=np.datetime64(row['cutoff'])).all())
            self.assertAlmostEqual(sum(row['weights']),len(ids));self.assertAlmostEqual(sum(row['group_weights']),len(ids));self.assertEqual(len(row['group_rows']),row['training_months'])
            for group in row['group_rows']:self.assertAlmostEqual(sum(row['q'][group]),1.)
    def test_five_year_exclusive_and_prior_dec31_maturity(self):
        ds=np.array(['2009-12-31','2010-01-31','2014-12-31','2014-11-28'],dtype='datetime64[D]');ends=np.array(['2010-01-29','2010-02-26','2015-01-30','2014-12-31'],dtype='datetime64[D]')
        ids,_,_=m.training(np.ones(4,bool),ds,ends,2015);np.testing.assert_array_equal(ids,[1,3])
    def test_17_vs18_and_query_own_future_label_independence(self):
        args=copy.deepcopy(self.args);args[8]=np.array(['2013-07-31'],dtype='datetime64[D]');args[9]=np.array(['NEW']);args[10]=args[10][:1];args[11]=args[11][:1]
        args[7][:]=False;args[7][:17*5]=True
        self.assertFalse(m.annual_oracle(*args)['available'][0])
        args[7][:18*5]=True;self.assertTrue(m.annual_oracle(*args)['available'][0])
        args[7][:]=True
        args[8]=np.array(['2015-03-31'],dtype='datetime64[D]');before=m.annual_oracle(*args);future=args[0]>=np.datetime64('2015-01-01');args[4][future]=np.nan;args[7][future]=False
        after=m.annual_oracle(*args)
        for key in before['raw']:np.testing.assert_array_equal(before['raw'][key],after['raw'][key])
    def test_future_poison_and_event_prefix(self):
        args=copy.deepcopy(self.args);future=args[0]>=np.datetime64('2015-01-01');args[3][future]=.49;args[4][future]=0.
        for target in args[5].values():target[future]=9.
        got=m.annual_oracle(*args)
        for key in got['raw']:np.testing.assert_array_equal(got['raw'][key][:2],self.oracle['raw'][key][:2])
        args=copy.deepcopy(self.args)
        for i in (8,9,10,11):args[i]=args[i][:2]
        got=m.annual_oracle(*args)
        for key in got['raw']:np.testing.assert_array_equal(got['raw'][key],self.oracle['raw'][key][:2])
    def test_no_queries_no_fit(self):
        args=copy.deepcopy(self.args);args[11][:]=False;got=m.annual_oracle(*args);self.assertFalse(got['available'].any());self.assertTrue(all(x['status']=='no_queries' for x in got['audit']['annual']))
    def test_mutated_published_forecast_and_q_fail(self):
        actual=copy.deepcopy(self.actual);actual['raw']['listwise_top3'][0]+=.001
        with self.assertRaises(AssertionError):m.verify_model(actual,self.oracle)
        actual=copy.deepcopy(self.actual);actual['audit']['annual'][0]['q'][0]+=.001
        with self.assertRaises(AssertionError):m.verify_model(actual,self.oracle)


class CrossImplementationTests(unittest.TestCase):
    def compare(self,args):
        expected=m.annual_oracle(*args);actual=producer.annual_forecasts(*args)
        m.verify_model(actual,expected);return actual,expected
    def test_exact_q_boundary_ties_all_ties_and_small_lists(self):
        for values in ([1.,2.,3.,3.,3.,4.],[3.]*7,[1.],[1.,2.],[1.,2.,3.],list(range(49))):
            ordinal=m.native_ordinal(values)
            np.testing.assert_array_equal(producer.top3_distribution(ordinal),m.top3_distribution(ordinal))
    def test_all_annual_fits_match_and_source_ridge_exact(self):
        args=fixture();actual,expected=self.compare(args)
        for row in actual['audit']['annual']:
            ids=np.array(row['training_indices']);queries=np.array(row['query_indices']);weights=np.array(row['weights'])
            for key,y in [('source_log',args[5]['log_excess']),('matched_ordinal',args[4])]:
                original=producer.numeric._fit_ridge(args[3][ids],y[ids],weights,10.)
                np.testing.assert_array_equal(actual['raw'][key][queries],producer.numeric._predict(original,args[10][queries]))
    def test_different_solver_same_convex_optimum_and_derivatives(self):
        rng=np.random.default_rng(83);Z=rng.normal(size=(23,10));groups=[np.arange(7),np.arange(7,15),np.arange(15,23)];weights=np.array([10.,23.,17.]);q=np.concatenate([m.top3_distribution(m.native_ordinal(Z[g,0])) for g in groups])
        beta=rng.normal(0,.1,10)
        for a,b in zip(m.grouped_objective(beta,Z,q,groups,weights),producer.listwise_value_gradient_hessian(Z,q,groups,weights,beta)):np.testing.assert_allclose(a,b,atol=1e-12,rtol=0.)
        independent=m.fit_listwise(Z,q,groups,weights);production,cert=producer.fit_listwise(Z,q,groups,weights)
        np.testing.assert_allclose(production,independent['coefficient'],atol=1e-9,rtol=0.)
        m.scalar_certificate(production,Z,q,groups,weights)
    def test_whole_month_missing_and_cold_queries(self):
        args=fixture();args[7][1]=False;args[3][7,0]=np.nan;args[5]['loss_utility'][13]=np.nan;args[4][19]=np.nan
        args[8][0]=np.datetime64('2011-01-31');args[10][1,0]=np.nan;args[11][2]=False
        actual,_=self.compare(args);np.testing.assert_array_equal(actual['available'],[False,False,False,True])
    def test_varying_original_lists_including_empty_and_small(self):
        args=fixture()
        for month in range(12):
            lo=month*5;n=month%6;args[6][lo:lo+5]=False;args[7][lo:lo+5]=False;args[4][lo:lo+5]=np.nan
            args[6][lo:lo+n]=True;args[7][lo:lo+n]=True
            if n:args[4][lo:lo+n]=m.native_ordinal(args[5]['log_excess'][lo:lo+n])
        self.compare(args)
    def test_future_label_change_and_prefix_both_implementations(self):
        args=fixture();full,_=self.compare(args)
        for i in (8,9,10,11):args[i]=args[i][:2]
        before,_=self.compare(args);future=args[1]>np.datetime64('2014-12-31');args[3][future]*=-1
        for y in args[5].values():y[future]+=100.
        after,_=self.compare(args)
        for key in full['raw']:
            np.testing.assert_array_equal(before['raw'][key],full['raw'][key][:2]);np.testing.assert_array_equal(before['raw'][key],after['raw'][key])
    def test_current_query_ranks_exact_across_solvers(self):
        args=fixture();args[8]=np.repeat(np.array(['2015-04-30','2016-04-29'],dtype='datetime64[D]'),6);args[9]=np.tile(list('ABCDEF'),2);args[10]=np.random.default_rng(46).uniform(-.49,.49,(12,10));args[11]=np.ones(12,bool)
        actual,expected=self.compare(args)
        for key in m.MODES:
            for lo in (0,6):np.testing.assert_array_equal(m.h.ordinal_rank(actual['raw'][key][lo:lo+6]),m.h.ordinal_rank(expected['raw'][key][lo:lo+6]))
    def test_audit_tamper_parameters_indices_cohorts_and_certificate(self):
        args=fixture();actual,expected=self.compare(args)
        changes=[lambda x:x['audit'].__setitem__('alpha',9.),lambda x:x['audit']['cohorts'][0].__setitem__('eligible_count',4),lambda x:x['audit']['annual'][0].__setitem__('training_indices_sha256','x'),lambda x:x['audit']['annual'][0]['listwise_top3'].__setitem__('max_iterations',201)]
        for change in changes:
            bad=copy.deepcopy(actual);change(bad)
            with self.assertRaises((ValueError,AssertionError)):m.verify_model(bad,expected)


class PlacementTests(unittest.TestCase):
    def test_identical_ordinal_and_quarter_slot_budget(self):
        a,meta=order_fixture();old,pre,_=m.score_orders(a,meta,m.NAMES[0]);new,npres,_=m.score_orders(a,meta,m.NAMES[1]);m.h.assert_exact_tree(new,old);m.h.assert_array_bytes(npres,pre)
        _,changed,_=m.score_orders(a,meta,m.NAMES[2]);self.assertLessEqual(np.max(abs(changed-pre)),.0125+1e-15);m.h.assert_array_bytes(changed[2],pre[2])
    def test_whole_event_month_paired_fallback(self):
        a,meta=order_fixture();a['raw_listwise_top3'][0,2]=np.nan;_,ready,_,_=m.ranked_pair(a,3);self.assertFalse(ready[0]);original=m.score_orders(a,meta,m.NAMES[0])[0]
        for name in m.MODES:m.h.assert_exact_tree(m.score_orders(a,meta,name)[0][0],original[0])
    def test_prefix_orders_exact(self):
        a,meta=order_fixture()
        for name in m.NAMES:
            orders,pre,scores=m.score_orders(a,meta,name)
            for n in (0,1,2):
                oo,pp,ss=m.score_orders(a,meta,name,n);m.h.assert_exact_tree(oo,orders[:n]);m.h.assert_array_bytes(pp,pre[:n]);m.h.assert_array_bytes(ss,scores[:n])
    def test_only_candidate_two_passes_both_and_strictly_beats_incumbent(self):
        rows=[stats(),stats(.7),stats(.6)];self.assertEqual(m.select(rows)['primary'],m.NAMES[0]);rows[1]=stats(.6);self.assertEqual(m.select(rows)['primary'],m.NAMES[2]);rows[1]=stats(.55);self.assertEqual(m.select(rows)['primary'],m.NAMES[2])
    def test_noneligible_outlier_does_not_change_ranking_scope(self):
        a,meta=order_fixture();a['eligible'][0,-1]=False;before=m.ranked_pair(a,3)[0];a['raw_listwise_top3'][0,-1]=1e8;after=m.ranked_pair(a,3)[0]
        for key in m.MODES:m.h.assert_array_bytes(before[key],after[key])
    def test_monthly_comparison_and_attribution(self):
        px=pd.DataFrame([[100.,100.,100.],[100.,110.,120.]],index=pd.to_datetime(['2020-01-01','2020-02-01']),columns=list('ABC'));meta=dict(dates=['2020-01-01'],ends=['2020-02-01']);orders={name:[dict(symbols=[symbol])] for name,symbol in zip(m.NAMES,'ABC')}
        rows=m.month_comparison(orders,px,meta);self.assertAlmostEqual(rows[0]['listwise_minus_ordinal'],.1)
        metrics={name:dict(total_return=rows[0]['returns'][name],years=1.) for name in m.NAMES};attribution=m.attribution(rows,metrics)
        self.assertAlmostEqual(attribution[m.NAMES[2]]['undo_best1_cagr'],0.);self.assertEqual(attribution[m.NAMES[2]]['changed_months'],1)
    def test_all_nine_synthetic_execution_paths(self):
        dates=pd.to_datetime(['2016-12-29','2016-12-30','2026-01-02','2026-01-05']);px=pd.DataFrame({'A':[100.,110.,120.,130.],'B':[100.,99.,101.,103.],'C':[100.,98.,101.,102.]},index=dates)
        checked=0
        for name,symbol in zip(m.NAMES,'ABC'):
            orders=[dict(date='2016-12-29',signal_date='2016-12-29',end='2026-01-05',symbols=[symbol],scores=[.1])]
            for mode,cost in (('close',0.),('open',0.),('open',.0005)):
                met,nav,trades,hold=m.h.replay(px,px,orders,dates,execution=mode,cost=cost);checked+=1
                self.assertEqual(len(nav),4);self.assertEqual(len(hold),1);self.assertTrue(all(row['partial'] for row in met['annual_returns']));self.assertEqual(len(trades),1)
                if cost:self.assertGreater(met['fees_paid'],0.)
        self.assertEqual(checked,9)
    def test_no_implicit_actual_reads_and_registration_gate(self):
        with patch.object(m.Reader,'sha',side_effect=AssertionError('actual reads forbidden')),contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):m.main([])
            with self.assertRaises(ValueError):m.main(['--phase','prepare','--plan-sha','bad'])
            with patch.object(m,'registration',side_effect=ValueError('registration gate')):
                with self.assertRaisesRegex(ValueError,'registration gate'):m.main(['--phase','prepare','--plan-sha','0'*64])


if __name__=='__main__':unittest.main(verbosity=2)
