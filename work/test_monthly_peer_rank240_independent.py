"""Synthetic source, mid-CDF, original-adapter and independent ledger checks."""
import ast
import copy
import hashlib
import importlib.util
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import monthly_peer_rank240_independent as audit


def prior_fixture_module():
    if hashlib.sha256(audit.PRIOR_TEST.read_bytes()).hexdigest()!=audit.PRIOR_TEST_SHA:
        raise ValueError('Frozen prior synthetic fixture changed')
    spec=importlib.util.spec_from_file_location('peer_rank240_prior_fixture',audit.PRIOR_TEST)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


helper=prior_fixture_module()
fixture,physical,warm_fixture=helper.fixture,helper.physical,helper.warm_fixture
selection_fixture,make_report=helper.selection_fixture,helper.make_report


def orders(scores,choices,meta):
    return {arm:[dict(date=meta['dates'][i],end=meta['ends'][i],signal_date=meta['dates'][i],
        symbols=[meta['symbols'][i][j] for j in ids],scores=[float(scores[arm][i,j]) for j in ids])
        for i,ids in enumerate(choices[arm])] for arm in audit.ARMS}


def endpoints(args,names,prices):
    for d in args[1]['dates']:
        if d=='2016-08-17':continue
        end=args[2][args[2].index(d)-42]
        args[0]['adj'].loc[end,names]=prices


class FeatureTests(unittest.TestCase):
    def test_signed_pair_count_and_final_division(self):
        self.assertEqual(audit.midcdf(2.,[1.,2.,2.,4.]),(0,8,0.))
        self.assertEqual(audit.midcdf(1.,[1.,2.,3.]),(-2,6,-1/3))
        self.assertEqual(audit.midcdf(3.,[1.,2.,3.]),(2,6,1/3))
        self.assertEqual(audit.midcdf(-0.,[-0.,0.,0.]),(0,6,0.))
        with self.assertRaises(ValueError):audit.midcdf(0.,[])
        with self.assertRaises(ValueError):audit.midcdf(0.,[math.inf])

    def test_exact_multi_sector_numbers_and_integer_audit(self):
        args=fixture();v,r=audit.independent_features(*args)
        audit.exact_array(v['peer_target'][1,:7],np.array([-3/8,-1/8,1/8,3/8,-1/3,0.,1/3]),'Hand CDF')
        audit.exact_array(v['absolute_target'][1,:7],np.array([-1/14,1/7,2/7,3/7,-3/7,-2/7,-1/14]),'Global midCDF')
        self.assertEqual(v['momentum'][1,:8].tolist(),[.25,.5,.75,1.,-.5,.125,.25,-.25])
        self.assertEqual(v['peer_median'][1,:7].tolist(),[.75,.75,.5,.5,.1875,-.125,-.1875])
        cells=r['events'][1]['cells']
        self.assertEqual((cells[0]['peer_numerator'],cells[0]['peer_denominator']),(-3,8))
        self.assertEqual((cells[0]['absolute_numerator'],cells[0]['absolute_denominator']),(-1,14))
        self.assertEqual(r['counts'],dict(events=3,stock_cells=24,mature_cells=16,supported_cells=14,endpoint_observations=32,new_model_fits=0))
        self.assertTrue(all(audit.verify_features(*args,v,r)[k] for k in audit.FEATURE_FLAGS))

    def test_tied_magnitudes_keep_exact_midpoint_and_group_size(self):
        args=fixture();endpoints(args,list('ABCD'),[128.,128.,256.,256.])
        v,r=audit.independent_features(*args)
        audit.exact_array(v['peer_target'][1,:4],np.array([-.25,-.25,.25,.25]),'Tie midpoints')
        self.assertEqual(r['events'][1]['cells'][0]['peer_numerator'],-2)
        self.assertEqual(v['peer_target'][1,6],1/3)
        self.assertLess(abs(v['peer_target'][1,6]),.5)

    def test_all_ties_zero_is_available(self):
        args=fixture();endpoints(args,list('ABCDEFGH'),[128.]*8)
        v,r=audit.independent_features(*args)
        for k in ('peer_target','absolute_target'):
            self.assertTrue((v[k][v['supported']]==0.).all())
            self.assertEqual(v[k][v['supported']].tobytes(),np.zeros(14).tobytes())
        self.assertTrue(v['supported'][1,:7].all());self.assertTrue(v['mature'][1,7])
        self.assertFalse(v['supported'][1,7]);self.assertIsNone(r['events'][1]['cells'][7]['peer_denominator'])

    def test_single_sector_two_arms_exact(self):
        args=list(fixture());args[4]=make_report(args[1],args[2])
        v,r=audit.independent_features(*args)
        audit.exact_array(v['peer_target'],v['absolute_target'],'Single sector identity')
        for row in r['events'][1]['cells']:
            self.assertEqual(row['peer_numerator'],row['absolute_numerator'])
            self.assertEqual(row['peer_denominator'],row['absolute_denominator'])

    def test_independent_sector_monotone_change_invariant_global_changes(self):
        args=fixture();v,_=audit.independent_features(*args)
        changed=copy.deepcopy(args);endpoints(changed,list('ABCD'),[16.,32.,48.,64.])
        new,_=audit.independent_features(*changed)
        audit.exact_array(new['peer_target'],v['peer_target'],'Sector order invariance')
        self.assertFalse(np.array_equal(new['absolute_target'][1,:7],v['absolute_target'][1,:7]))
        self.assertFalse(np.array_equal(new['peer_median'][1,:4],v['peer_median'][1,:4]))

    def test_existing_source_momentum_median_are_exact_not_redefined(self):
        args=fixture();old,old_a=audit.old.independent_features(*args);new,new_a=audit.independent_features(*args)
        for k in ('mature','supported','momentum','peer_median'):audit.exact_array(new[k],old[k],k)
        self.assertEqual(new_a['counts'],old_a['counts'])
        self.assertFalse(np.array_equal(new['peer_target'][1,:7],old['peer_target'][1,:7]))
        for x,y in zip(new_a['events'][1]['cells'],old_a['events'][1]['cells']):
            self.assertEqual(x['peer_difference'],y['peer_difference'])

    def test_no_eligibility_filter_no_input_mutation(self):
        args=fixture();before=copy.deepcopy(args);v,r=audit.independent_features(*args)
        self.assertTrue(v['supported'][1,1]);self.assertTrue(v['supported'][1,5])
        pd.testing.assert_frame_equal(args[0]['adj'],before[0]['adj']);self.assertEqual(args[4],before[4])
        args[3]['eligible'][:]=False;new,ra=audit.independent_features(*args)
        for k in audit.VALUE_KEYS:audit.exact_array(new[k],v[k],k)
        audit.record_check(ra,r)

    def test_source_malformed_mature_endpoint_and_peer_inventory_rejected(self):
        for mutation in ('duplicate_issuer','missing_peer','bool_peer','invalid_endpoint','false_admission'):
            args=fixture()
            if mutation=='duplicate_issuer':args[4]['records'][9]['issuer_key']=args[4]['records'][8]['issuer_key']
            elif mutation=='missing_peer':args[4]['records'][8]['peer_indices'].pop()
            elif mutation=='bool_peer':args[4]['events'][1]['stock_indices'][0]=False
            elif mutation=='invalid_endpoint':
                d=args[4]['events'][1]['source_dates'][0];args[0]['adj'].at[pd.Timestamp(d),'A']=True
            else:args[4]['admitted']=False
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):audit.independent_features(*args)

    def test_warm_and_partial_sources_unread(self):
        args=warm_fixture();v,r=audit.independent_features(*args)
        self.assertFalse(v['mature'][:,3].any());self.assertTrue(np.isnan(v['momentum'][:,3]).all())
        self.assertEqual(r['counts']['stock_cells'],15)
        small=physical(fixture(),1);small[0]['adj']=small[0]['adj'].iloc[:0]
        v,r=audit.independent_features(*small);self.assertEqual(r['counts']['endpoint_observations'],0)
        self.assertFalse(v['supported'].any())

    def test_future_poison_and_every_physical_prefix(self):
        args=fixture();full=audit.independent_features(*args)
        for n in range(4):
            got=audit.independent_features(*physical(args,n));logical=audit.independent_features(*args,limit=n)
            audit.record_check(got[1],logical[1])
            for k in audit.VALUE_KEYS:audit.exact_array(got[0][k],full[0][k][:n],k)
        poison=copy.deepcopy(args);poison[1]['dates'][2]='future';poison[1]['symbols'][2]=None
        poison[3]['eligible']=poison[3]['eligible'].astype(object);poison[3]['eligible'][2]='future'
        poison[4]['events'][2]=None
        poison[4]['records']=[r if r['query_index']<2 else dict(query_index=2,garbage=object()) for r in poison[4]['records']]
        poison[0]['adj'].loc[pd.Timestamp(args[1]['dates'][1])+pd.Timedelta(days=1):]='future'
        v,r=audit.independent_features(*poison,limit=2)
        for k in audit.VALUE_KEYS:audit.exact_array(v[k],full[0][k][:2],k)

    def test_feature_integer_audit_and_range_tamper_rejected(self):
        args=fixture();v,r=audit.independent_features(*args)
        for field in ('peer_target','absolute_target'):
            bad=copy.deepcopy(v);bad[field][1,0]=0.
            with self.assertRaises(ValueError):audit.verify_features(*args,bad,r)
        for field in ('peer_numerator','peer_denominator','absolute_numerator','absolute_denominator'):
            bad=copy.deepcopy(r);bad['events'][1]['cells'][0][field]=True
            with self.assertRaises(ValueError):audit.verify_features(*args,v,bad)
        for edge in (-.5,.5,math.inf):
            bad=copy.deepcopy(v);bad['peer_target'][1,0]=edge
            with self.assertRaises(ValueError):audit.selection_evidence(args[3],args[1],bad,3)


class SelectionTests(unittest.TestCase):
    def test_real_original_adapter_exact_full_pres_scores_choices_and_one_episode(self):
        args=selection_fixture();_,m,_,a,_=args;v,_=audit.independent_features(*args)
        expected,chosen,pres=audit.selection_oracle(a,m,v)
        study=helper.load('outputs/monthly_local_linear70/study.py','peer_rank240_test_original')
        adapter=helper.load('outputs/monthly_path70_research/incumbent_adapter.py','peer_rank240_test_adapter')
        x=audit.frozen_source_oracle().old.original_helper().original_prefix(a,m,3)
        calls=[];actual={};oo={};policy=dict(budget=.05,ceiling=5,gap=.02)
        for arm,key in zip(audit.ARMS,(None,'peer_target','absolute_target')):
            factor=np.full(a['eligible'].shape,np.nan) if key is None else v[key]
            def boundary(pre,eligible,names,effects,original,rule):
                calls.append(arm);audit.exact_array(pre,pres[arm],'Independent funded pre')
                audit.exact_array(eligible,a['eligible'],'Original E')
                return study.ep.select_boundary(pre,eligible,names,effects,original,rule)
            cc,ss=adapter.score_and_choose('incumbent66',.05,factor,a,m,3,pre_score=x['local'],effects=a['effects'],
                episode_policy=policy,boundary_selector=boundary)
            self.assertEqual(cc,chosen[arm]);audit.exact_array(ss,expected[arm],arm)
            actual[arm]=ss;oo[arm]=study.rp.make_orders(cc,ss,m)
        self.assertEqual(calls,list(audit.ARMS))
        out=audit.verify_selection(a,m,v,oo,scores=actual,pre_scores=pres)
        self.assertTrue(all(out[k] for k in audit.SELECTION_FLAGS))

    def test_constant_zero_replaces_support_and_missing_does_not(self):
        args=selection_fixture();_,m,_,a,_=args;endpoints(args,list('ABCDEFGH'),[128.]*8)
        v,_=audit.independent_features(*args);ss,cc,pp=audit.selection_oracle(a,m,v)
        x=audit.frozen_source_oracle().old.original_helper().original_prefix(a,m,3)
        self.assertEqual(v['peer_target'][1,0],0.)
        self.assertNotEqual(pp[audit.ARMS[1]][1,0],x['local'][1,0])
        self.assertEqual(pp[audit.ARMS[1]][1,0],x['local'][1,0]+.05*(0.-x['support'][1,0]))
        for arm in audit.ARMS:audit.exact_array(pp[arm][0],x['local'][0],'Initial unchanged')
        self.assertEqual(pp[audit.ARMS[1]][1,7],x['local'][1,7])
        self.assertEqual(pp[audit.ARMS[1]][1,8],x['local'][1,8])

    def test_budget_includes_extra_calm_nonweak_slot(self):
        args=selection_fixture();_,m,_,a,_=args
        a['extension_available'][:]=True;a['forecast_available'][:]=True;a['rank_log_excess'][:]=.2
        a['eligible'][2,1]=False;v,_=audit.independent_features(*args);ss,cc,pp=audit.selection_oracle(a,m,v)
        out=audit.verify_selection(a,m,v,orders(ss,cc,m),scores=ss,pre_scores=pp)
        self.assertEqual(out['calm_nonweak_extension_count'],8)
        self.assertGreaterEqual(out['minimum_remaining_support'],.33-1e-12)
        self.assertLessEqual(out['maximum_budget_sum_error'],1e-12)

    def test_all_selection_prefixes_original_model_domain_cash_two_three(self):
        for count in (0,1,2,9):
            args=selection_fixture();_,m,_,a,_=args;a['eligible'][:,count:]=False
            a['forecast_available'][:]=True;a['model_available'][:]=True
            a['raw_local_constant'][:]=np.arange(9);a['raw_local_linear'][:]=np.arange(9)[::-1]
            v,_=audit.independent_features(*args);full,cc,pp=audit.selection_oracle(a,m,v)
            for arm in audit.ARMS:self.assertEqual(len(cc[arm][1]),3 if count==9 else 2 if count==2 else 0)
            for n in range(4):
                ss,ch,pres=audit.selection_oracle(a,m,v,n)
                audit.verify_selection(a,m,v,orders(ss,ch,m),scores=ss,pre_scores=pres,n=n)
                for arm in audit.ARMS:audit.exact_array(ss[arm],full[arm][:n],arm)

    def test_full_pre_score_order_tamper_and_source_validation_split(self):
        args=selection_fixture();_,m,_,a,_=args;v,r=audit.independent_features(*args)
        ss,ch,pres=audit.selection_oracle(a,m,v);oo=orders(ss,ch,m)
        for field in ('scores','pre_scores'):
            data=dict(scores=copy.deepcopy(ss),pre_scores=copy.deepcopy(pres));data[field][audit.ARMS[1]][1,0]+=.001
            with self.assertRaises(ValueError):audit.verify_selection(a,m,v,oo,**data)
        bad=copy.deepcopy(oo);bad[audit.ARMS[1]][1]['symbols'].reverse()
        with self.assertRaises(ValueError):audit.verify_selection(a,m,v,bad,scores=ss,pre_scores=pres)
        # Exact sector CDF needs the supplied source report, not selection meta.
        bad=copy.deepcopy(v);bad['peer_target'][1,0]=0.
        audit.selection_evidence(a,m,bad,3)
        with self.assertRaises(ValueError):audit.verify_features(*args,bad,r)

    def test_dependency_identity_and_no_new_producer_imports(self):
        tree=ast.parse(Path(audit.__file__).read_text())
        imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        imports += [a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names]
        self.assertNotIn('monthly_peer_rank240_kernel',imports);self.assertNotIn('monthly_peer_rank240_host',imports)
        deps=audit.dependency_files();self.assertIn(audit.PRIOR,deps);self.assertIn(audit.PRIOR_TEST,deps)
        self.assertIn(audit.ROOT/'work/test_monthly_upside_retention181_independent.py',deps)
        with patch.object(audit,'PRIOR_SHA','bad'),self.assertRaises(ValueError):audit.prior()


class OutcomeTests(unittest.TestCase):
    def test_nine_paths_forty_five_slices_and_fee_tamper(self):
        import test_monthly_upside_retention181_independent as fixture_helper
        with patch.object(fixture_helper,'audit',audit):args=fixture_helper.account_fixture()
        result=audit.audit_outcomes(*args)
        self.assertEqual((result['checked_paths'],result['checked_slices']),(9,45))
        self.assertEqual(set(result['paths']),set(audit.ARMS))
        for path,key in [('same_close_zero','fees_paid'),('next_open_cost','cagr')]:
            wrong=copy.deepcopy(args);wrong[4][audit.ARMS[1]][path]['metrics'][key]+=1.
            with self.assertRaises((ValueError,AssertionError)):audit.audit_outcomes(*wrong)


class BilateralTests(unittest.TestCase):
    def test_unique_both_fixtures_all_prefix_values_full_integer_audit(self):
        import monthly_peer_rank240_kernel as producer
        from test_monthly_peer_rank240_kernel import fixture as other_fixture
        for args in (fixture(),warm_fixture(),other_fixture()):
            full=producer.build_features(*args);audit.verify_features(*args,*full)
            for n in range(len(args[1]['dates'])+1):
                logical=producer.build_features(*args,limit=n);audit.verify_features(*args,*logical,limit=n)
                small=physical(args,n);got=producer.build_features(*small);audit.verify_features(*small,*got)
                audit.record_check(logical[1],got[1])
                for k in audit.VALUE_KEYS:audit.exact_array(full[0][k][:n],got[0][k],k)


if __name__=='__main__':unittest.main(verbosity=2)
