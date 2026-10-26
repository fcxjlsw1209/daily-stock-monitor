"""Generated-only 149 independent tests; no current producer is imported."""
from copy import deepcopy
from datetime import date,timedelta
from decimal import localcontext
from fractions import Fraction
from pathlib import Path
import argparse
import ast
import json
import sys
import unittest
from unittest.mock import patch
import numpy as np
import monthly_stock_comp149_prepare_review as r
import monthly_stock_comp149_review_contract as c
import test_monthly_cash_quality146_prepare_review as oldtests
import test_stock_comp_source_admission_v1_independent as sourcetests


def component(original,amount='5'):
    result={key:deepcopy(original[key]) for key in r.META}
    result.update(fact_admitted=False,model_admitted=False,strategy_admitted=False)
    value=deepcopy(original['fields']['current']['operating_cash_flow'])
    value.update(value_decimal=amount,source_sha256=original['source']['sha256'])
    result['stock_compensation']=dict(field='stock_compensation_addback',value=value,gaps=[],candidate_only=True,
        accounting_review_flags=[],fact_admitted=False,model_admitted=False,strategy_admitted=False)
    result['stock_compensation_admission']=dict(admitted=True,reasons=[],definition=r.SBC_DEFINITION)
    return result


def fixture(dates=None):
    meta,data=oldtests.fixture(dates)
    for event in data['events']:event['fiscal_year']=2025
    sbc=dict(schema_version=1,method=r.MEASUREMENT,events=[component(e) for e in data['events']])
    return meta,data,sbc


def score_fixture():return oldtests.score_fixture()


def audit_fixture():
    meta,data,sbc=fixture();expected=r.features_oracle(meta,data,sbc)
    saved,_=oldtests.audit_fixture();saved.update(cells=deepcopy(expected['cells']),
        measurement=r.MEASUREMENT,stock_compensation_definition=r.SBC_DEFINITION,
        arithmetic_method='exact_fraction_scaled_cashflow_minus_signed_sbc_over_average_assets_then_bounded_float',
        source_verification='caller_verified_original146_and_new_stock_compensation_admission_closures_required',
        common_available_cells=int(expected['arrays']['common_available'].sum()),
        original_cash_quality_available_cells=expected['original_cash_quality_available_cells'],
        reasons=dict(r.Counter(cell['reason'] for cell in expected['cells'])))
    for key in ('earnings_and_cashflow_controls_identical_to146_on_common_mask','common_mask_subset_of146',
                'same_selected_current_event_for_all_fields','signed_stock_compensation_preserved',
                'stock_compensation_raw_flags_do_not_substitute_new_admission',
                'adjusted_cashflow_is_research_measure_not_reported_GAAP_cashflow'):
        saved[key]=True
    refs={key:dict(path=path,sha256=str(i)*64) for i,(key,path) in enumerate(c.SBC_REFS.items())}
    saved.update({key+'_sha256':ref['sha256'] for key,ref in refs.items()})
    return saved,expected,refs


def source_fixture(change=None):
    """44 synthetic documents, three patches and both complete proof closures."""
    reader=oldtests.MemoryReader()
    original,raw,patches,merged,output=sourcetests.population()
    html=[];visual=[]
    for event in original:
        eid=event['event_id']
        if event['symbol']=='AAPL':
            value=merged[eid]['stock_compensation']['value']
            visual.append(dict(event_id=eid,value_decimal=value['value_decimal'],source=deepcopy(value['source']),
                               scale=value['scale'],current_period_end=value['period']['period_end']))
            html.append(dict(event_id=eid,status='PDF_requires_visual_review'))
        else:html.append(dict(event_id=eid,status='HTML_physical_cell_verified',cell=dict(verified=True,source_sha256=event['source']['sha256'])))
    documents=dict(raw=dict(events=raw),source_plan=dict(synthetic=True),source_manifest=dict(synthetic=True),
        html_review=dict(passed=True,event_reviews=html),pdf_visual=dict(passed_visual_identification=True,pages_reviewed=10,events=visual),
        patch_plan=dict(synthetic=True),patch=dict(events=patches),patch_audit=dict(passed=True),patch_manifest=dict(synthetic=True),
        old_dataset=dict(schema_version=1,method=r.METHOD,events=original),old_admission=dict(passed=True),
        old_review=dict(passed=True),sessions=dict(sessions=['2025-01-07']),tests=dict(passed=True))
    if change:change(documents,output)
    refs={key:reader.add('synthetic/'+key+'.json',obj) for key,obj in documents.items()}
    code=reader.add('synthetic/independent_source.py','generated source code')
    policy=dict(schema_version=1,method=r.MEASUREMENT,purpose=r.MEASUREMENT,definition=r.SBC_DEFINITION,
        expected_events=44,same_current_report=True,latest_event_before_validity=True,invalid_latest_blocks_older_fallback=True,
        preserve_all_13_original_holds=True,annual_days=[330,400],max_age_days=550,same_report_literal_currency_assumption=True,
        financial_ratios_permitted=False,strategy_registration_permitted=False,conditional_research_only=True,
        first_public_version_verified=False,complete_correction_history_verified=False,certified_daily_security_master=False,
        iso_currency_certified=False,accounting='synthetic signed reported value',accounting_changes='preserve synthetic flags',
        runtime=sys.version,inputs=refs,source_code_hashes={code['path']:code['sha256']})
    policyref=reader.add(c.SBC_REFS['sbc_policy'],policy)
    dataset=reader.add(c.SBC_REFS['sbc_dataset'],dict(schema_version=1,method=r.MEASUREMENT,
        policy_sha256=policyref['sha256'],conditional_research_only=True,events=output))
    admitted=[e['event_id'] for e in output if e['stock_compensation_admission']['admitted']]
    holds=[dict(event_id=e['event_id'],**e['stock_compensation_admission']) for e in output if not e['stock_compensation_admission']['admitted']]
    shared=dict(passed=True,events=44,admitted_event_ids=admitted,holds=holds,admitted_events=len(admitted),held_events=len(holds),
                policy_sha256=policyref['sha256'],financial_ratios=0,factor_calculations=0,return_calculations=0,
                first_public_version_verified=False,complete_correction_history_verified=False,certified_daily_security_master=False)
    closure={v['path']:v['sha256'] for v in refs.values()};closure[code['path']]=code['sha256']
    admission=reader.add(c.SBC_REFS['sbc_admission'],dict(shared,stage='conditional_stock_compensation_source_admission',
        data_sha256=dataset['sha256'],input_hashes=closure,model_fits=0,observer_changes=0,
        source_values_independently_reviewed=True,accounting_flags_preserved=True,accounting_flag_disposition=policy['accounting_changes']))
    reviewclosure=dict(closure,**{v['path']:v['sha256'] for v in (dataset,admission,policyref)})
    review=reader.add(c.SBC_REFS['sbc_review'],dict(shared,stage='independent_conditional_stock_compensation_source_review',
        source_only=True,input_hashes=reviewclosure,inherited_holds=13,all_source_candidates_preserved=True,
        original_rejections_preserved=True,independent_decisions_match=True,three_patch_cells_bound_to_original_sources=True,
        all_present_values_match_independent_physical_or_visual_proof=True,producer_admission_code_imported=False))
    plan=dict(inputs=dict(admitted_dataset=refs['old_dataset'],source_admission=refs['old_admission'],sessions=refs['sessions']),
              sbc_dataset=dataset,sbc_admission=admission,sbc_policy=policyref,sbc_review=review)
    pins={key:reader.sha(key) for key in reader.documents}
    return reader,plan,pins,documents['old_dataset']


def source_check(reader,plan,pins,data):
    # This one immutable production dataset pin is replaced only by the generated fixture digest.
    with patch.object(c,'DATA_SHA',plan['inputs']['admitted_dataset']['sha256']):
        return c.verify_stock_comp_source(reader,plan,pins,data)


class ArithmeticTests(unittest.TestCase):
    def test_three_independent_integer_values(self):
        meta,data,sbc=fixture();out=r.features_oracle(meta,data,sbc)
        self.assertEqual([out['arrays'][key][0,0] for key in r.EXTRA[:3]],[1/11,3/13,1/5])
    def test_zero_sbc_is_exact_cashflow_control(self):
        meta,data,sbc=fixture();sbc['events'][0]['stock_compensation']['value']['value_decimal']='0'
        out=r.features_oracle(meta,data,sbc)
        np.testing.assert_array_equal(out['arrays'][r.EXTRA[1]],out['arrays'][r.EXTRA[2]])
    def test_signed_negative_sbc_is_not_absolute_or_clipped(self):
        meta,data,sbc=fixture();sbc['events'][0]['stock_compensation']['value']['value_decimal']='-20'
        self.assertEqual(r.features_oracle(meta,data,sbc)['arrays'][r.EXTRA[2]][0,0],1/3)
    def test_exact_cancellation_under_small_decimal_context(self):
        meta,data,sbc=fixture();data['events'][0]['fields']['current']['operating_cash_flow']['value_decimal']=str(10**150+2)
        sbc['events'][0]['stock_compensation']['value']['value_decimal']=str(10**150+1)
        with localcontext() as context:
            context.prec=3
            out=r.features_oracle(meta,data,sbc)
        self.assertEqual(out['arrays'][r.EXTRA[2]][0,0],1/101)
    def test_extreme_signed_values_have_finite_unit_endpoints(self):
        for amount,expected in [('1e1000',-1.),('-1e1000',1.)]:
            meta,data,sbc=fixture();sbc['events'][0]['stock_compensation']['value']['value_decimal']=amount
            self.assertEqual(r.features_oracle(meta,data,sbc)['arrays'][r.EXTRA[2]][0,0],expected)
    def test_integer_closed_form_matches_fraction_for_signed_scaled_cases(self):
        for cash,ni,current,prior,sbc in [(30,10,150,50,5),(1,-2,3,4,-5),(-20,0,10,8,21),(0,0,1,1,0)]:
            pairs=[(cash,2),(ni,-2),(current,3),(prior,1),(sbc,2)]
            actual=r.bounded_from_cells(pairs)
            vals=[Fraction(v)*Fraction(10)**exp for v,exp in pairs];cf,n,a,b,s=vals
            ratios=(n/((a+b)/2),cf/((a+b)/2),(cf-s)/((a+b)/2))
            self.assertEqual(actual,tuple(float(x/(1+abs(x))) for x in ratios))
    def test_cfo_sbc_scales_must_match_numerically(self):
        meta,data,sbc=fixture();cell=sbc['events'][0]['stock_compensation']['value']
        cell['scale']='1.00';self.assertTrue(r.features_oracle(meta,data,sbc)['arrays']['common_available'][0,0])
        cell['scale']='10';out=r.features_oracle(meta,data,sbc)
        self.assertEqual(out['cells'][0]['reason'],'stock_compensation_cashflow_scale_mismatch')
    def test_bad_monetary_literals_never_become_zero(self):
        for amount in ('NaN','Infinity','1,234',' 5',True):
            meta,data,sbc=fixture();sbc['events'][0]['stock_compensation']['value']['value_decimal']=amount
            out=r.features_oracle(meta,data,sbc)
            self.assertFalse(out['arrays']['common_available'][0,0]);self.assertTrue(all(np.isnan(out['arrays'][k][0,0]) for k in r.EXTRA[:3]))
    def test_nonpositive_asset_endpoints_block_all_arms(self):
        for side in r.SIDES:
            for amount in ('0','-1'):
                meta,data,sbc=fixture();data['events'][0]['fields'][side]['total_assets']['value_decimal']=amount
                self.assertFalse(r.features_oracle(meta,data,sbc)['arrays']['common_available'][0,0])


class CausalTests(unittest.TestCase):
    def test_missing_sbc_narrows_shared_mask(self):
        meta,data,sbc=fixture();sbc['events']=[];out=r.features_oracle(meta,data,sbc)
        self.assertEqual(out['original_cash_quality_available_cells'],2)
        self.assertFalse(out['arrays']['common_available'].any())
        self.assertTrue(all(np.isnan(out['arrays'][k]).all() for k in r.EXTRA[:3]))
    def test_old_gate_cannot_be_overridden_by_new_gate(self):
        meta,data,sbc=fixture();data['events'][0]['cash_quality_admission']['admitted']=False
        out=r.features_oracle(meta,data,sbc);self.assertFalse(out['arrays']['common_available'].any())
    def test_latest_missing_sbc_blocks_older_valid_report(self):
        meta,data,sbc=fixture();later=deepcopy(data['events'][0]);later['event_id']='later';oldtests.reschedule(later,'2026-02-20')
        data['events'].append(later);out=r.features_oracle(meta,data,sbc)
        self.assertEqual(out['cells'][0]['current_event_id'],'later');self.assertFalse(out['arrays']['common_available'].any())
    def test_latest_new_hold_blocks_older_valid_report(self):
        meta,data,sbc=fixture();later=deepcopy(data['events'][0]);later['event_id']='later';oldtests.reschedule(later,'2026-02-20')
        data['events'].append(later);new=component(later);new['stock_compensation_admission'].update(admitted=False,reasons=['held']);sbc['events'].append(new)
        out=r.features_oracle(meta,data,sbc)
        self.assertEqual(out['cells'][0]['reason'],'current_not_stock_compensation_admitted')
    def test_identical_duplicates_ok_conflicting_duplicates_fail(self):
        meta,data,sbc=fixture();sbc['events'].append(deepcopy(sbc['events'][0]));self.assertTrue(r.features_oracle(meta,data,sbc)['arrays']['common_available'][0,0])
        sbc['events'][1]['stock_compensation']['value']['value_decimal']='99'
        self.assertEqual(r.features_oracle(meta,data,sbc)['cells'][0]['reason'],'conflicting_stock_compensation_event')
    def test_future_valid_or_invalid_append_cannot_change_history(self):
        meta,data,sbc=fixture();before=r.features_oracle(meta,data,sbc)
        later=deepcopy(data['events'][0]);later['event_id']='future';oldtests.reschedule(later,'2026-12-01');later['cash_quality_admission']['admitted']=False
        data['events'].append(later);sbc['events'].append(component(later))
        self.assertEqual(r.features_oracle(meta,data,sbc)['cells'],before['cells'])
    def test_provenance_period_currency_and_availability_required(self):
        for key,value in [('source',{}),('source_sha256','z'*64),('period',{}),('available_from','1900-01-01'),('currency_literal','EUR')]:
            meta,data,sbc=fixture();sbc['events'][0]['stock_compensation']['value'][key]=value
            self.assertFalse(r.features_oracle(meta,data,sbc)['arrays']['common_available'].any())
    def test_all_ten_metadata_fields_are_locked(self):
        for key in r.META:
            meta,data,sbc=fixture();sbc['events'][0][key]='changed'
            self.assertFalse(r.features_oracle(meta,data,sbc)['arrays']['common_available'].any())
    def test_new_admission_strict_structure_definition_and_reasons(self):
        for key,value in [('admitted',1),('definition','other'),('reasons',['held'])]:
            meta,data,sbc=fixture();sbc['events'][0]['stock_compensation_admission'][key]=value
            self.assertFalse(r.features_oracle(meta,data,sbc)['arrays']['common_available'].any())
        meta,data,sbc=fixture();sbc['events'][0]['stock_compensation_admission']['extra']=True
        self.assertFalse(r.features_oracle(meta,data,sbc)['arrays']['common_available'].any())
    def test_raw_flags_do_not_substitute_new_admission(self):
        meta,data,sbc=fixture();sbc['events'][0].pop('stock_compensation_admission')
        self.assertFalse(r.features_oracle(meta,data,sbc)['arrays']['common_available'].any())
    def test_old_550_day_expiry_still_applies(self):
        end=date(2025,12,31);dates=[str(end+timedelta(days=d)) for d in (550,551)]
        meta,data,sbc=fixture(dates);out=r.features_oracle(meta,data,sbc)
        self.assertTrue(out['arrays']['common_available'][0,0]);self.assertFalse(out['arrays']['common_available'][1,0])
    def test_annual_duration_boundaries_preserved(self):
        for current,previous,valid in ((330,400,True),(400,330,True),(329,365,False),(365,401,False)):
            meta,data,sbc=fixture();event=data['events'][0];end=date(2025,12,31);start=end-timedelta(days=current-1);prior=start-timedelta(days=1)
            event['periods']=dict(current=oldtests.period(str(start),str(end)),comparison=oldtests.period(str(prior-timedelta(days=previous-1)),str(prior)))
            for side in r.SIDES:
                for cell in event['fields'][side].values():cell['period']=deepcopy(event['periods'][side])
            sbc['events']=[component(event)]
            self.assertEqual(r.features_oracle(meta,data,sbc)['arrays']['common_available'][0,0],valid)
    def test_etf_rows_remain_missing(self):
        meta,data,sbc=fixture();out=r.features_oracle(meta,data,sbc)
        self.assertFalse(out['arrays']['common_available'][:,32:].any())
    def test_no_source_or_input_mutation(self):
        meta,data,sbc=fixture();before=deepcopy((meta,data,sbc));r.features_oracle(meta,data,sbc)
        self.assertEqual((meta,data,sbc),before)
    def test_comparison_flows_are_never_used(self):
        meta,data,sbc=fixture();before=r.features_oracle(meta,data,sbc)
        data['events'][0]['fields']['comparison']['operating_cash_flow']=None
        data['events'][0]['fields']['comparison']['net_income']={'poison':True}
        self.assertEqual(r.features_oracle(meta,data,sbc)['cells'],before['cells'])

class ScoreTests(unittest.TestCase):
 def test_direct_fixed_dose_not_support_clip(self):
  a,meta,hand=score_fixture();_,_,delta=r.score_oracle(a,meta,hand,-hand,hand/2)
  np.testing.assert_array_equal(delta[r.NAMES[1]],np.nan_to_num(.01*hand))
  np.testing.assert_array_equal(delta[r.NAMES[2]],np.nan_to_num(-.01*hand))
 def test_missing_and_zero_exact_baseline(self):
  a,meta,hand=score_fixture()
  for fill in (0.,np.nan):
   hand[:,:]=fill;hand[:,-1]=np.nan;scores,choices,_=r.score_oracle(a,meta,hand,hand,hand)
   for name in r.NAMES[1:]:
    self.assertEqual(scores[name].tobytes(),scores[r.NAMES[0]].tobytes());self.assertEqual(choices[name],choices[r.NAMES[0]])
 def test_ineligible_gets_no_delta(self):
  a,meta,hand=score_fixture();a['eligible'][0,0]=False
  _,_,d=r.score_oracle(a,meta,hand,hand,hand)
  self.assertEqual(d[r.NAMES[1]][0,0],0.)
 def test_exact_one_episode_per_arm_and_pre_score_dose(self):
  a,meta,hand=score_fixture();h=r.old.original_helper();prefix=h.original_prefix(a,meta,None)
  with patch.object(r.old,'original_helper',return_value=h),patch.object(h,'episode',wraps=h.episode) as spy:
   r.score_oracle(a,meta,hand,hand,hand);self.assertEqual(spy.call_count,4)
   expected=prefix['local'].copy();delta=np.nan_to_num(.01*hand);mask=np.isfinite(hand)&(hand!=0)
   expected[mask]+=delta[mask];np.testing.assert_array_equal(spy.call_args_list[1].args[0],expected)
 def test_masks_must_match_and_etf_stays_missing(self):
  a,meta,hand=score_fixture();other=hand.copy();other[0,0]=np.nan
  with self.assertRaises(ValueError):r.score_oracle(a,meta,hand,other,hand)
  hand[:,-1]=0.
  with self.assertRaises(ValueError):r.score_oracle(a,meta,hand,hand,hand)
 def test_nonfinite_or_unbounded_hand_rejected(self):
  for value in (np.inf,1.1,-1.1):
   a,meta,hand=score_fixture();hand[0,0]=value
   with self.assertRaises(ValueError):r.score_oracle(a,meta,hand,hand,hand)
 def test_rounded_unit_endpoints_keep_declared_dose(self):
  a,meta,hand=score_fixture();hand[0,:2]=[-1.,1.]
  _,_,delta=r.score_oracle(a,meta,hand,hand,hand)
  self.assertEqual(list(delta[r.NAMES[1]][0,:2]),[-.01,.01])
 def test_prefix_does_not_read_future_hands(self):
  a,meta,hand=score_fixture();wanted=r.score_oracle(a,meta,hand,hand,hand,1);hand[1]=np.inf
  actual=r.score_oracle(a,meta,hand,hand,hand,1)
  for name in r.NAMES:self.assertEqual(actual[0][name].tobytes(),wanted[0][name].tobytes());self.assertEqual(actual[1][name],wanted[1][name])
 def test_original_arrays_remain_unmodified(self):
  a,meta,hand=score_fixture();before={k:v.tobytes() for k,v in a.items()};r.score_oracle(a,meta,hand,hand,hand)
  self.assertEqual(before,{k:v.tobytes() for k,v in a.items()})
 def test_saved_complete_scores_and_ordered_holdings(self):
  a,meta,hand=score_fixture();scores,choices,_=r.score_oracle(a,meta,hand,hand,hand);orders=r.ordered_orders(meta,scores,choices)
  r.verify_scores_and_orders(scores,orders,meta,scores,choices)
  bad=deepcopy(orders);bad[r.NAMES[3]][0]['symbols'].reverse()
  with self.assertRaises(ValueError):r.verify_scores_and_orders(scores,bad,meta,scores,choices)
 def test_broadcast_score_shape_is_not_accepted(self):
  a,meta,hand=score_fixture();scores,choices,_=r.score_oracle(a,meta,hand,hand,hand)
  orders=r.ordered_orders(meta,scores,choices);bad=dict(scores);bad[r.NAMES[0]]=scores[r.NAMES[0]][:,:1]
  with self.assertRaisesRegex(ValueError,'Full score matrix'):r.verify_scores_and_orders(bad,orders,meta,scores,choices)
 def test_27_original_array_bytes_required(self):
  original={str(i):np.array([i],dtype=np.int64) for i in range(27)}
  prepared=dict(original,**{k:np.zeros((1,49)) for k in r.EXTRA});r.verify_original_arrays(original,prepared)
  prepared['0']=np.array([0],dtype=np.float64)
  with self.assertRaises(ValueError):r.verify_original_arrays(original,prepared)



def proposal_fixture(change=None):
    reader=oldtests.MemoryReader()
    concept=reader.add(c.CONCEPT,dict(synthetic_hypothesis=True))
    inputs={key:reader.add('synthetic/'+key+'.json',obj) for key,obj in (
        ('admitted_dataset',dict(synthetic=True)),('source_admission',dict(events=44,admitted_events=31,held_events=13)))}
    policy=reader.add('synthetic/old_policy.json',dict(synthetic=True))
    refs={key:reader.add(path,dict(synthetic=True)) for key,path in c.SBC_REFS.items()}
    sources=dict(policy=policy,dataset=inputs['admitted_dataset'],admission=inputs['source_admission'],
                 full_inventory=44,admitted=31,held=13,**refs)
    proposal=dict(round=149,arms=list(r.NAMES),primary_challenger=r.NAMES[3],dose=.01,
        selection_contract=c.proposal_contract(),measurement=c.measurement_contract(),concept=concept,novelty=concept,
        status='FINAL_PROSPECTIVE_SPECIFICATION_BEFORE_REGISTRATION',actual_new_factors=0,actual_new_orders=0,
        actual_new_returns=0,sources=sources)
    if change:change(proposal)
    inputs['proposal']=reader.add('synthetic/proposal.json',proposal)
    return reader,dict(inputs=inputs,source_policy=policy,**refs),{k:reader.sha(k) for k in reader.documents},concept['sha256']


class ProposalTests(unittest.TestCase):
    def check(self,change=None):
        reader,plan,pins,concept_sha=proposal_fixture(change)
        with patch.object(c,'CONCEPT_SHA',concept_sha):return c.verify_proposal(reader,plan,pins)
    def test_complete_generated_prospective_contract(self):
        self.assertEqual(self.check()['arms'],list(r.NAMES))
    def test_changed_selection_or_measurement_rejected(self):
        for field,key,value in [('selection_contract','training_months',87),('selection_contract','next_open_cost_cagr','other'),
                                ('measurement','common_mask','separate'),('measurement','intramonth_trades',True)]:
            with self.assertRaises(ValueError):self.check(lambda p:p[field].update({key:value}))
    def test_changed_or_missing_source_ref_is_rejected(self):
        for key in c.SBC_REFS:
            with self.assertRaises(ValueError):self.check(lambda p:p['sources'].pop(key))
        with self.assertRaises(ValueError):self.check(lambda p:p['sources'].update(admitted=32))
    def test_prior_actual_calculation_or_wrong_dose_is_rejected(self):
        for key,value in [('actual_new_factors',1),('actual_new_orders',True),('dose',.02),('round',148)]:
            with self.assertRaises(ValueError):self.check(lambda p:p.update({key:value}))


class SourceContractTests(unittest.TestCase):
    def test_complete_generated_source_closure(self):
        reader,plan,pins,data=source_fixture();out=source_check(reader,plan,pins,data)
        self.assertEqual(len(out['events']),44)
        self.assertEqual(sum(e['stock_compensation_admission']['admitted'] for e in out['events']),31)
    def test_all_thirteen_holds_preserved_without_ratios(self):
        reader,plan,pins,data=source_fixture()
        with patch.object(r,'bounded_from_cells',side_effect=AssertionError('No source-gate ratios')):
            out=source_check(reader,plan,pins,data)
        self.assertEqual(sum(not e['stock_compensation_admission']['admitted'] for e in out['events']),13)
    def test_changed_source_output_is_rejected_even_with_self_consistent_hash(self):
        reader,plan,pins,data=source_fixture(lambda docs,out:out[0]['stock_compensation']['value'].update(value_decimal='99'))
        with self.assertRaisesRegex(ValueError,'candidate or accounting'):source_check(reader,plan,pins,data)
    def test_old_hold_cannot_be_promoted(self):
        reader,plan,pins,data=source_fixture(lambda docs,out:out[-1]['stock_compensation_admission'].update(admitted=True,reasons=[]))
        with self.assertRaisesRegex(ValueError,'eligibility decision'):source_check(reader,plan,pins,data)
    def test_missing_patched_source_proof_fails(self):
        reader,plan,pins,data=source_fixture(lambda docs,out:docs['pdf_visual']['events'][0].update(value_decimal='99'))
        with self.assertRaisesRegex(ValueError,'PDF visual'):source_check(reader,plan,pins,data)
    def test_omitted_html_census_row_fails(self):
        reader,plan,pins,data=source_fixture(lambda docs,out:docs['html_review']['event_reviews'].pop())
        with self.assertRaisesRegex(ValueError,'full census'):source_check(reader,plan,pins,data)
    def test_raw_admission_flag_cannot_be_promoted(self):
        def mutate(docs,out):
            docs['raw']['events'][-1]['fact_admitted']=True
            out[-1]['fact_admitted']=True
        reader,plan,pins,data=source_fixture(mutate)
        with self.assertRaisesRegex(ValueError,'raw flags'):source_check(reader,plan,pins,data)
    def test_unbound_source_code_fails(self):
        reader,plan,pins,data=source_fixture();del pins[str(reader.path('synthetic/independent_source.py'))]
        with self.assertRaisesRegex(ValueError,'Full registered'):source_check(reader,plan,pins,data)
    def test_dataset_pin_and_version_must_be_fixed(self):
        reader,plan,pins,data=source_fixture()
        with self.assertRaisesRegex(ValueError,'Unchanged146'):c.verify_stock_comp_source(reader,plan,pins,data)
        plan['sbc_dataset']['path']=c.SBC_REFS['sbc_dataset'].replace('_v2','_v1')
        with self.assertRaisesRegex(ValueError,'Exact SBC source path'):source_check(reader,plan,pins,data)
    def test_all_four_top_level_refs_are_required(self):
        for key in c.SBC_REFS:
            reader,plan,pins,data=source_fixture();del plan[key]
            with self.assertRaises(KeyError):source_check(reader,plan,pins,data)
    def test_changed_document_under_old_digest_fails(self):
        reader,plan,pins,data=source_fixture()
        reader.documents[str(reader.path(plan['sbc_admission']['path']))]['held_events']=0
        with self.assertRaisesRegex(ValueError,'digest'):source_check(reader,plan,pins,data)
    def test_proposal_contract_fixes_arithmetic_dose_and_dual_gates(self):
        measurement=c.measurement_contract();selection=c.proposal_contract()
        self.assertEqual(measurement['comp_adjusted_cashflow'],'(CFO_current-SBC_current)/average_assets')
        self.assertEqual(measurement['unchanged_original_arrays'],27)
        self.assertFalse(selection['later34_for_selection']);self.assertEqual(selection['training_months'],88)
        for field,value in [('new_model_fits',False),('training_months',87),('next_open_cost_cagr','greater_incumbent_only')]:
            altered=deepcopy(selection);altered[field]=value
            with self.assertRaises(ValueError):c.exact(altered,selection)
    def test_no_producer_imports_in_new_independent_modules(self):
        for module in (r,c):
            tree=ast.parse(Path(module.__file__).read_text())
            for node in ast.walk(tree):
                if isinstance(node,ast.Import):
                    self.assertFalse(any(a.name.endswith(('_factor','_runner')) for a in node.names))
                if isinstance(node,ast.ImportFrom):
                    self.assertFalse((node.module or '').endswith(('_factor','_runner')))
    def test_source_files_contains_independent_transitive_dependencies(self):
        names={p.name for p in r.source_files()}
        self.assertTrue({'monthly_cash_quality146_prepare_review.py','test_monthly_cash_quality146_prepare_review.py',
                         'stock_comp_source_admission_v1_independent.py','test_stock_comp_source_admission_v1_independent.py',
                         'monthly_stock_comp149_review_contract.py'}.issubset(names))


class AuditTests(unittest.TestCase):
    def test_valid_audit_and_all_six_source_bindings(self):
        saved,expected,refs=audit_fixture()
        r.verify_audit(saved,expected,dataset_sha='d'*64,admission_sha='a'*64,sbc_refs=refs)
        for key in refs:
            broken=deepcopy(saved);broken[key+'_sha256']='x'*64
            with self.assertRaisesRegex(ValueError,'source audit binding'):
                r.verify_audit(broken,expected,dataset_sha='d'*64,admission_sha='a'*64,sbc_refs=refs)
    def test_changed_cell_reason_value_type_or_extra_key_fails(self):
        for key,value in [('reason','wrong'),('available',1),('comp_adjusted_cash_flow_bounded',.5),('unexpected',True)]:
            saved,expected,_=audit_fixture();saved['cells'][0][key]=value
            with self.assertRaises(ValueError):r.verify_audit(saved,expected)
    def test_wrong_mask_original_count_or_method_fails(self):
        for key,value in [('common_available_cells',0),('original_cash_quality_available_cells',0),('arithmetic_method','float_first')]:
            saved,expected,_=audit_fixture();saved[key]=value
            with self.assertRaises(ValueError):r.verify_audit(saved,expected)
    def test_each_required_new_semantic_claim_is_checked(self):
        for key in ('earnings_and_cashflow_controls_identical_to146_on_common_mask','common_mask_subset_of146',
                    'same_selected_current_event_for_all_fields','signed_stock_compensation_preserved',
                    'stock_compensation_raw_flags_do_not_substitute_new_admission',
                    'adjusted_cashflow_is_research_measure_not_reported_GAAP_cashflow'):
            saved,expected,_=audit_fixture();saved[key]=False
            with self.assertRaises(ValueError):r.verify_audit(saved,expected)
    def test_no_actual_read_without_explicit_plan_digest(self):
        with patch.object(r.old,'Reader') as reader:
            for digest in (None,'bad','A'*64):
                with self.assertRaisesRegex(ValueError,'Explicit PLAN SHA'):r.audit('/synthetic','PLAN.json',digest)
            reader.assert_not_called()
    def test_all123_prefixes_and_four_arm_fallback_generated_axes(self):
        dates=[str(d.date()) for d in r.pd.date_range('2026-03-01',periods=122,freq='MS')]
        meta,data,sbc=fixture(dates);meta['ends']=dates[:]
        template,_,_=score_fixture();arrays={}
        for key,value in template.items():arrays[key]=np.resize(value,(122,49) if value.ndim==2 else (122,))
        features=r.features_oracle(meta,data,sbc)
        scores,choices,_=r.score_oracle(arrays,meta,*(features['arrays'][k] for k in r.EXTRA[:3]))
        r.verify_prefixes(arrays,meta,data,sbc,features,scores,choices)
    def test_actual_receipt_has_no_duplicate_keyword_expansions(self):
        # CHECKS is expanded into the returned actual receipt; duplicate keywords fail only at runtime.
        tree=ast.parse(Path(r.__file__).read_text())
        audit=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='audit')
        returned=next(n.value for n in ast.walk(audit) if isinstance(n,ast.Return))
        explicit={k.arg for k in returned.keywords if k.arg is not None}
        self.assertFalse(explicit.intersection(r.CHECKS))


if __name__=='__main__':
    if '--receipt' not in sys.argv:unittest.main(verbosity=2)
    else:
        parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--receipt',type=Path,required=True)
        args=parser.parse_args();target=r.ROOT/'work/monthly_stock_comp149_prepare_review_SYNTHETIC.json'
        r.require(args.receipt.resolve()==target,'Dedicated synthetic receipt path')
        r.require(not target.exists(),'Preserve existing synthetic receipt')
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
        receipt=r.synthetic_receipt(result)
        with target.open('x') as stream:json.dump(receipt,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
        raise SystemExit(0 if result.wasSuccessful() else 1)
