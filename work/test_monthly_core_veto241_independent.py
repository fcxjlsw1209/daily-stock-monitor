"""Synthetic third-seat constraints and independent accounting only."""
import ast
import copy
import hashlib
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

import monthly_core_veto241_independent as audit


def helper():
    if hashlib.sha256(audit.PRIOR_TEST.read_bytes()).hexdigest()!=audit.PRIOR_TEST_SHA:
        raise ValueError('Frozen synthetic fixture changed')
    spec=importlib.util.spec_from_file_location('core_veto241_test_prior',audit.PRIOR_TEST)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


fixture_helper=helper()


def fixture():
    _,meta,_,a,_=fixture_helper.selection_fixture()
    a['core'][:]=np.array([.25,.1875,.125,.0625,.15625,0.,-.0625,-.125,-.1875])
    a['baseline']=a['core'].copy()  # Original aggregate support exactly zero.
    return a,meta


def orders(scores,choices,meta):
    return {arm:[dict(date=meta['dates'][i],end=meta['ends'][i],signal_date=meta['dates'][i],
        symbols=[meta['symbols'][i][j] for j in ids],scores=[float(scores[arm][i,j]) for j in ids])
        for i,ids in enumerate(choices[arm])] for arm in audit.ARMS}


def run(args,n=None):
    a,m=args;s,c,p,r=audit.selection_oracle(a,m,n)
    return orders(s,c,m),s,p,r


def physical(args,n):
    a,m=copy.deepcopy(args)
    return {k:v[:n].copy() for k,v in a.items()}, {k:(v[:n] if k in ('dates','ends','symbols') else v) for k,v in m.items()}


def real_study():return fixture_helper.helper.load('outputs/monthly_local_linear70/study.py','core_veto241_test_original')


class SelectionTests(unittest.TestCase):
    def test_known_core_veto_selects_next_admissible_not_always_prethird(self):
        a,m=fixture();s,c,p,r=audit.selection_oracle(a,m)
        ev=r['events'][1]
        self.assertEqual(ev['pre_order'],list(range(9)));self.assertEqual(ev['boundary_indices'],[2,3,4])
        self.assertEqual(ev['admitted_indices'],[2,4]);self.assertEqual(ev['vetoed_indices'],[3])
        self.assertEqual((ev['original_winner'],ev['candidate_winner'],ev['control_winner']),(3,4,2))
        self.assertEqual(set(c[audit.ARMS[0]][1]),{0,1,3})
        self.assertEqual(set(c[audit.ARMS[1]][1]),{0,1,4})
        self.assertEqual(set(c[audit.ARMS[2]][1]),{0,1,2})
        self.assertEqual(r['counts']['vetoed_winner_events'],2)
        self.assertEqual(r['counts']['candidate_changed_holdings'],2)
        out=audit.verify_selection(a,m,orders(s,c,m),scores=s,pre_scores=p,audit=r)
        self.assertTrue(all(out[k] for k in audit.SELECTION_FLAGS))

    def test_equal_core_is_admitted_exactly_and_one_ulp_lower_rejected(self):
        for core,admit in ((.125,True),(np.nextafter(.125,-np.inf),False)):
            a,m=fixture();a['core'][1,3]=core;a['baseline'][1,3]=core
            _,_,_,r=audit.selection_oracle(a,m)
            self.assertEqual(3 in r['events'][1]['admitted_indices'],admit)

    def test_original_full_scores_and_pre_are_not_recalculated_after_veto(self):
        a,m=fixture();s,c,p,r=audit.selection_oracle(a,m)
        audit.exact_array(s[audit.ARMS[0]],s[audit.ARMS[1]],'All original F including vetoed')
        self.assertGreater(s[audit.ARMS[1]][1,3],p[audit.ARMS[1]][1,3])
        for arm in audit.ARMS:audit.exact_array(p[arm],p[audit.ARMS[0]],'All original P')
        audit.exact_array(s[audit.ARMS[2]][1:],p[audit.ARMS[2]][1:],'Ordinary control P')

    def test_initial_partial_is_exact_incumbent_even_with_actual_episode_change(self):
        a,m=fixture();s,c,p,r=audit.selection_oracle(a,m)
        self.assertNotEqual(s[audit.ARMS[0]][0,3],p[audit.ARMS[0]][0,3])
        for arm in audit.ARMS:
            audit.exact_array(s[arm][0],s[audit.ARMS[0]][0],arm)
            self.assertEqual(c[arm][0],c[audit.ARMS[0]][0])
        ev=r['events'][0];self.assertTrue(ev['initial_partial'])
        self.assertEqual(ev['admitted_indices'],ev['boundary_indices']);self.assertEqual(ev['vetoed_indices'],[])
        self.assertEqual(r['counts']['core_compared_cells'],6)

    def test_first_regular_row_is_not_partial(self):
        a,m=fixture();a={k:v[1:2].copy() for k,v in a.items()}
        m={k:(v[1:2] if k in ('dates','ends','symbols') else v) for k,v in m.items()}
        _,_,_,r=audit.selection_oracle(a,m)
        self.assertFalse(r['events'][0]['initial_partial']);self.assertEqual(r['events'][0]['vetoed_indices'],[3])

    def test_gap_exact_boundary_and_rank_six_never_promoted(self):
        a,m=fixture();a['pre_score'][:,2:]=[0.,-.02,np.nextafter(-.02,-np.inf),-.03,-.04,-.05,-.06]
        a['effects'][:,5]=100.;a['core'][:,5]=.25;a['baseline'][:,5]=.25
        _,_,_,r=audit.selection_oracle(a,m)
        self.assertEqual(r['events'][1]['boundary_indices'],[2,3])
        self.assertNotIn(5,r['events'][1]['admitted_indices'])

    def test_symbol_ties_and_locked_top_two_even_when_third_has_higher_final_score(self):
        a,m=fixture();a['pre_score'][:]=[.14,.13,.125,.125,.125,.02,.01,0.,-.01]
        a['effects'][:,2:5]=.5;a['core'][:,2:5]=.125;a['baseline'][:,2:5]=.125
        for names in m['symbols']:names[2:5]=['Z','Y','X']
        s,c,_,r=audit.selection_oracle(a,m);ev=r['events'][1]
        self.assertEqual(ev['pre_third'],4);self.assertEqual(ev['original_winner'],4)
        self.assertEqual(ev['locked_top2'],[0,1]);self.assertEqual(c[audit.ARMS[1]][1][0],4)
        self.assertEqual(set(c[audit.ARMS[1]][1]),{0,1,4})

    def test_cash_two_and_three_choices_with_audit_nulls(self):
        for count in (0,1,2,3):
            a,m=fixture();a['eligible'][:,count:]=False;s,c,p,r=audit.selection_oracle(a,m)
            for arm in audit.ARMS:self.assertEqual(len(c[arm][1]),0 if count<2 else count)
            if count<3:
                self.assertEqual(r['events'][1]['boundary_indices'],[])
                self.assertIsNone(r['events'][1]['pre_third']);self.assertIsNone(r['events'][1]['core_anchor'])
                self.assertIsNone(r['events'][1]['original_winner'])
            audit.verify_selection(a,m,orders(s,c,m),scores=s,pre_scores=p,audit=r)

    def test_all_core_ties_recover_original_episode(self):
        a,m=fixture();a['core'][:]=0.;a['baseline'][:]=0.
        s,c,p,r=audit.selection_oracle(a,m)
        self.assertEqual(c[audit.ARMS[1]],c[audit.ARMS[0]])
        self.assertEqual(r['counts']['vetoed_cells'],0)

    def test_original_adapter_exact_scores_and_one_episode(self):
        a,m=fixture();s,c,p,r=audit.selection_oracle(a,m)
        study=real_study();calls=[]
        expected_pre=study.modified_pre_score(study.configs()[4],a,3)
        audit.exact_array(expected_pre,p[audit.ARMS[0]],'Independent pre rebuild')
        original_selector=study.ep.select_boundary
        def boundary(*args,**kwargs):calls.append(1);return original_selector(*args,**kwargs)
        cc,ss=study.choices(study.configs()[4],a,m,3,policy=dict(study.POLICY_FIELDS),boundary_selector=boundary)
        self.assertEqual(calls,[1]);self.assertEqual(cc,c[audit.ARMS[0]])
        audit.exact_array(ss,s[audit.ARMS[0]],'Independent scalar original episode')

    def test_budget_calm_nonweak_and_existing_model_domain(self):
        a,m=fixture();a['extension_available'][:]=True;a['forecast_available'][:]=True
        a['rank_log_excess'][:]=.2;a['eligible'][2,1]=False
        a['model_available'][:]=True;a['raw_local_linear'][:]=np.arange(9)
        s,c,p,r=audit.selection_oracle(a,m)
        out=audit.verify_selection(a,m,orders(s,c,m),scores=s,pre_scores=p,audit=r)
        self.assertEqual(out['calm_nonweak_extension_count'],8)
        self.assertGreaterEqual(out['minimum_remaining_support'],.38-1e-12)
        self.assertLessEqual(out['maximum_budget_sum_error'],1e-12)

    def test_no_input_mutation_and_every_logical_physical_prefix(self):
        args=fixture();before=copy.deepcopy(args);s,c,p,r=audit.selection_oracle(*args)
        for n in range(4):
            small=physical(args,n);ss,cc,pp,rr=audit.selection_oracle(*small)
            logical=audit.selection_oracle(*args,n=n)
            audit.record_check(rr,logical[3]);self.assertEqual(rr['events'],r['events'][:n])
            for arm in audit.ARMS:
                audit.exact_array(ss[arm],s[arm][:n],arm);audit.exact_array(pp[arm],p[arm][:n],arm)
                self.assertEqual(cc[arm],c[arm][:n])
            audit.verify_selection(*small,orders(ss,cc,small[1]),scores=ss,pre_scores=pp,audit=rr)
            self.assertEqual(rr['counts']['original_episode_calls'],1)
        for k,v in args[0].items():audit.exact_array(v,before[0][k],k)
        self.assertEqual(args[1],before[1])

    def test_future_numeric_and_metadata_poison_not_consumed(self):
        a,m=fixture();base=audit.selection_oracle(a,m,n=2)
        a['core'][2]=np.nan;a['pre_score'][2]=np.nan;a['effects'][2]=np.inf
        m['dates'][2]='bad';m['ends'][2]='bad';m['symbols'][2]=None
        got=audit.selection_oracle(a,m,n=2)
        for arm in audit.ARMS:audit.exact_array(got[0][arm],base[0][arm],arm)
        audit.record_check(got[3],base[3])

    def test_bad_original_sources_states_shapes_and_initial_position_fail(self):
        for mutation in ('core','weight','support','nonbool','duplicate','end','latepartial'):
            a,m=fixture()
            if mutation=='core':a['core'][1,2]=np.nan
            elif mutation=='weight':a['core_weight'][1,0]=.5
            elif mutation=='support':a['baseline'][1,2]=10.
            elif mutation=='nonbool':a['eligible']=a['eligible'].astype(int)
            elif mutation=='duplicate':m['symbols'][1][0]=m['symbols'][1][1]
            elif mutation=='end':m['ends'][1]=m['dates'][1]
            else:m['dates'][1]='2016-08-17'
            with self.subTest(mutation=mutation),self.assertRaises((ValueError,AssertionError)):audit.selection_oracle(a,m)

    def test_tampered_full_scores_pre_orders_and_complete_audit_rejected(self):
        a,m=fixture();o,s,p,r=run((a,m))
        for field in ('scores','pre_scores'):
            data=dict(scores=copy.deepcopy(s),pre_scores=copy.deepcopy(p),audit=r)
            data[field][audit.ARMS[1]][1,3]+=.001
            with self.assertRaises(ValueError):audit.verify_selection(a,m,o,**data)
        for field in ('admitted_indices','candidate_winner','core_anchor'):
            bad=copy.deepcopy(r)
            if field=='admitted_indices':bad['events'][1][field].append(3)
            elif field=='candidate_winner':bad['events'][1][field]=True
            else:bad['events'][1][field]+=.001
            with self.assertRaises(ValueError):audit.verify_selection(a,m,o,scores=s,pre_scores=p,audit=bad)
        bad=copy.deepcopy(r);bad['counts']['original_episode_calls']=2
        with self.assertRaises(ValueError):audit.verify_selection(a,m,o,scores=s,pre_scores=p,audit=bad)
        bad=copy.deepcopy(o);bad[audit.ARMS[1]][1]['symbols'].reverse()
        with self.assertRaises(ValueError):audit.verify_selection(a,m,bad,scores=s,pre_scores=p,audit=r)

    def test_dependency_pin_no_production_oracle_import(self):
        tree=ast.parse(Path(audit.__file__).read_text())
        imports=[x.module for x in ast.walk(tree) if isinstance(x,ast.ImportFrom)]
        imports += [a.name for x in ast.walk(tree) if isinstance(x,ast.Import) for a in x.names]
        self.assertNotIn('monthly_core_veto241_kernel',imports);self.assertNotIn('monthly_core_veto241_host',imports)
        deps=audit.dependency_files();self.assertIn(audit.PRIOR,deps);self.assertIn(audit.PRIOR_TEST,deps)
        with patch.object(audit,'PRIOR_SHA','bad'),self.assertRaises(ValueError):audit.prior()


class OutcomeTests(unittest.TestCase):
    def test_nine_paths_forty_five_slices_and_fee_tamper(self):
        import test_monthly_upside_retention181_independent as fixture_module
        with patch.object(fixture_module,'audit',audit):args=fixture_module.account_fixture()
        result=audit.audit_outcomes(*args)
        self.assertEqual((result['checked_paths'],result['checked_slices']),(9,45))
        self.assertEqual(set(result['paths']),set(audit.ARMS))
        for path,key in [('same_close_zero','fees_paid'),('next_open_cost','cagr')]:
            wrong=copy.deepcopy(args);wrong[4][audit.ARMS[1]][path]['metrics'][key]+=1.
            with self.assertRaises((ValueError,AssertionError)):audit.audit_outcomes(*wrong)


class BilateralTests(unittest.TestCase):
    def test_unique_producer_both_fixtures_all_prefixes_full_scores_orders_audits(self):
        import monthly_core_veto241_kernel as producer
        from test_monthly_core_veto241_kernel import fixture as other_fixture
        study,a,m=other_fixture()
        for study,a,m in ((study,a,m),(real_study(),*fixture())):
            o,s,p,r=producer.decisions(study,a,m,len(m['dates']))
            audit.verify_selection(a,m,o,scores=s,pre_scores=p,audit=r)
            for n in range(len(m['dates'])+1):
                aa,mm=physical((a,m),n);oo,ss,pp,rr=producer.decisions(study,aa,mm,n)
                audit.verify_selection(aa,mm,oo,scores=ss,pre_scores=pp,audit=rr)
                self.assertEqual(rr['events'],r['events'][:n])
                for arm in audit.ARMS:
                    self.assertEqual(oo[arm],o[arm][:n]);audit.exact_array(ss[arm],s[arm][:n],arm)
                    audit.exact_array(pp[arm],p[arm][:n],arm)


if __name__=='__main__':unittest.main(verbosity=2)
