"""Generated-only independent oracle tests. Never import the new producer."""
from copy import deepcopy
from datetime import date,timedelta
from decimal import localcontext
from fractions import Fraction
import ast
import argparse
import hashlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import monthly_asset_growth147_prepare_review as r


def period(start,end):
 return dict(period_start=start,period_end=end,days_inclusive=(date.fromisoformat(end)-date.fromisoformat(start)).days+1)


def event(key='fixture',symbol='AAPL'):
 p=dict(current=period('2025-01-01','2025-12-31'),comparison=period('2024-01-01','2024-12-31'))
 source=dict(path='synthetic/fixture.html',sha256='a'*64)
 out=dict(event_id=key,symbol=symbol,release_date='2026-01-30',available_from='2026-02-02',periods=p,source=source,
  identity=dict(issuer_id='fixture_'+symbol,security_id=symbol,identity_valid_from='2026-02-02',identity_valid_through=str(date(2025,12,31)+timedelta(days=550))),
  inherited_rejection_codes=[],metadata_gaps=[],prior_field_admitted=True,fact_admitted=False,model_admitted=False,strategy_admitted=False,
  cash_quality_admission=dict(admitted=True,reasons=[],definition=r.DEFINITION))
 out['fields']={side:{field:dict(value_decimal=value,scale='1',currency_literal='$',currency_iso=None,
  fact_admitted=False,locator=dict(period_end=p[side]['period_end'],ni_scope='indirect_method_consolidated_start'),
  source=deepcopy(source),period=deepcopy(p[side]),available_from=out['available_from'])
  for field,value in [('operating_cash_flow','30'),('net_income','10'),('total_assets','150' if side=='current' else '50')]} for side in r.SIDES}
 return out


def fixture(dates=None):
 names=['AAPL','AMD','MSFT','NVDA']+['S'+str(j) for j in range(28)]+['ETF'+str(j) for j in range(17)]
 dates=['2026-03-02','2026-04-01'] if dates is None else dates
 return dict(dates=dates,symbols=[names[:] for _ in dates],etfs=names[32:]),dict(schema_version=1,method=r.METHOD,events=[event()])


def reschedule(e,available):
 e['available_from']=e['identity']['identity_valid_from']=available
 for side in r.SIDES:
  for cell in e['fields'][side].values():cell['available_from']=available


def score_fixture():
 shape=(2,5);baseline=np.array([[.16,.12,.08,.07,-.10],[.15,.13,.10,.09,-.10]])
 core=np.array([[.10,.08,.05,.04,-.10],[.10,.08,.06,.05,-.10]])
 a=dict(baseline=baseline,core=core,eligible=np.ones(shape,bool),core_weight=np.array([.5,.52]),
  state_weak_narrow=np.array([False,True]),state_calm10=np.array([True,False]),pre_score=baseline.copy(),effects=np.zeros(shape),
  forecast_available=np.zeros(shape,bool),rank_log_excess=np.zeros(shape),old_ml4=np.zeros(shape),extension_available=np.zeros(shape,bool),
  model_available=np.zeros(shape,bool),raw_local_constant=np.zeros(shape),raw_local_linear=np.zeros(shape),raw_source_log=np.zeros(shape))
 m=dict(symbols=[list('ABCDE'),list('ABCDE')],etfs=['E'],dates=['2026-03-02','2026-04-01'],ends=['2026-03-31','2026-04-30'])
 hand=np.array([[.8,.2,-.4,0,np.nan],[.3,-.1,0,-.9,np.nan]])
 return a,m,hand


def audit_fixture():
 meta,data=fixture();expected=r.features_oracle(meta,data)
 saved=dict(cells=deepcopy(expected['cells']),new_model_fits=0,returns_or_nav_computed=False,
  original_arrays_unchanged=True,original_auxiliary_states_and_core_coefficients_preserved=True,
  latest_event_before_validity_filter=True,same_current_release_for_all_cells=True,
  annual_duration_normalization=False,comparison_flow_values_used=False,
  measurement=r.MEASUREMENT,earnings_and_low_accrual_controls_identical_to146=True,
  common_mask_identical_to146=True,growth_denominator='comparison_total_assets',
  exact_feature_range='(-1,0.5); float rounding may reach an endpoint',
  dose_0_01_exact_score_range='(-0.01,0.005)',
  expansion_negative_contraction_positive_zero_change_zero=True,
  asset_growth_not_organic_investment_or_CMA_replication=True,
  arithmetic_method='exact_fraction_scaled_endpoints_negative_asset_change_over_prior_assets_then_bounded_float',
  source_verification='caller_verified_147_source_use_review_and_frozen146_source_closure_required',
  currency_scope='same_report_dollar_literal_assumption_not_ISO_certification',
  original_months=2,original_width=49,common_available_cells=2,
  admitted_dataset_sha256='d'*64,source_admission_sha256='a'*64,
  asset_use_policy_sha256='p'*64,asset_use_review_sha256='r'*64,
  reasons=dict(r.Counter(c['reason'] for c in expected['cells'])))
 return saved,expected


class MemoryReader:
 """Synthetic JSON documents only; never read repository data."""
 def __init__(self):self.documents={};self.seen={}
 def path(self,path):
  path=Path(path)
  return path if path.is_absolute() else Path('/synthetic')/path
 def sha(self,path):
  key=str(self.path(path));value=hashlib.sha256(json.dumps(self.documents[key],sort_keys=True).encode()).hexdigest()
  self.seen[key]=value;return value
 def json(self,path):self.sha(path);return deepcopy(self.documents[str(self.path(path))])
 def add(self,path,value):
  self.documents[str(self.path(path))]=value
  return dict(path=path,sha256=self.sha(path))
 def verify(self,refs):
  result={}
  for path,sha in refs.items():
   r.require(self.sha(path)==sha,'Synthetic hash binding');result[str(self.path(path))]=sha
  return result


def admission_fixture():
 reader=MemoryReader();sessions=['2026-01-30','2026-02-02','2026-02-03']
 implementation=reader.add('work/cash_quality_source_admission_v1.py','synthetic source code')
 accounting_rows=[];events=[]
 for i in range(44):
  e=event('synthetic_'+str(i));e.pop('cash_quality_admission')
  if i==43:e['inherited_rejection_codes']=['legacy_hold'];e['prior_field_admitted']=False
  accounting_rows.append(dict(event_id=e['event_id'],inherited_rejection_codes=e['inherited_rejection_codes'],
   inherited_rejections_unchanged=True))
  events.append(e)
 accounting=reader.add('accounting.json',dict(stage='source_only_field_scoped_accounting_review',source_only=True,events=accounting_rows))
 for e,a in zip(events,accounting_rows):
  e['union_provenance']=dict(accounting_review_reference=accounting,accounting_event_review=a)
 evidence=reader.add('evidence.json',dict(synthetic=True))
 union_plan=reader.add('union/PLAN.json',dict(source_hashes={evidence['path']:evidence['sha256']},
  source_code_hashes={implementation['path']:implementation['sha256']}))
 raw=reader.add('union/RAW_FACTS.json',dict(plan_sha256=union_plan['sha256'],events=events))
 manifest=reader.add('union/MANIFEST.json',dict(plan_sha256=union_plan['sha256'],output_hashes={'RAW_FACTS.json':raw['sha256']}))
 review=reader.add('union/REVIEW.json',dict(passed=True,input_hashes={v['path']:v['sha256'] for v in (union_plan,raw,manifest)}))
 inputs=dict(union_plan=union_plan,union_manifest=manifest,union_data=raw,union_independent=review,accounting=accounting,
  sessions=reader.add('sessions.json',dict(sessions=sessions)),concept=reader.add('concept.json',dict(synthetic=True)),
  tests=reader.add('tests.json',dict(passed=True,source_sha256=implementation['sha256'])))
 policy=reader.add('POLICY.json',dict(schema_version=1,method=r.METHOD,definition=r.DEFINITION,expected_events=44,
  runtime=r.sys.version,implementation_sha256=implementation['sha256'],inputs=inputs,
  preserve_all_inherited_holds=True,same_report_literal_currency_assumption=True,
  use_comparison_flows=False,financial_ratios_permitted=False,strategy_registration_permitted=False,
  annual_days=[330,400],max_age_days=550))
 rows=deepcopy(events)
 for i,e in enumerate(rows):
  e['cash_quality_admission']=dict(admitted=i!=43,reasons=[] if i!=43 else ['preserved_inherited_source_hold'],definition=r.DEFINITION)
 dataset=reader.add('DATASET.json',dict(schema_version=1,method=r.METHOD,events=rows,policy_sha256=policy['sha256'],conditional_research_only=True))
 receipt=reader.add('ADMISSION.json',dict(schema_version=1,stage='conditional_cash_quality_source_admission',passed=True,method=r.METHOD,
  data_sha256=dataset['sha256'],policy_sha256=policy['sha256'],events=44,admitted_events=43,held_events=1,
  admitted_event_ids=[e['event_id'] for e in rows[:-1]],holds=[dict(event_id=rows[-1]['event_id'],**rows[-1]['cash_quality_admission'])],
  source_hashes={v['path']:v['sha256'] for v in inputs.values()},
  original_raw_events_preserved=True,original_rejections_preserved=True,
  first_public_version_verified=False,complete_correction_history_verified=False,certified_daily_security_master=False,
  currency_iso_not_newly_certified=True,financial_ratios=0,factor_calculations=0,portfolio_calculations=0,
  return_calculations=0,network_requests=0,observer_changes=0))
 plan=dict(source_policy=policy,inputs=dict(admitted_dataset=dataset,source_admission=receipt))
 pins={path:reader.sha(path) for path in reader.documents}
 return reader,plan,pins,sessions


class ArithmeticTests(unittest.TestCase):
 def test_three_hand_calculations(self):
  result=r.report_oracle(event(),'2026-03-02')
  self.assertTrue(result['available'])
  self.assertEqual([result[k] for k in r.EXTRA[:3]],[1/11,1/6,-2/3])
 def test_independent_scale_alignment(self):
  e=event();e['fields']['current']['net_income'].update(value_decimal='0.01',scale='1e3')
  self.assertEqual(r.report_oracle(e,'2026-03-02')['earnings_bounded'],1/11)
 def test_cancellation_of_very_large_amounts_is_exact(self):
  e=event();e['fields']['current']['operating_cash_flow']['value_decimal']=str(10**120+2)
  e['fields']['current']['net_income']['value_decimal']=str(10**120+1)
  with localcontext() as context:
   context.prec=3
   actual=r.report_oracle(e,'2026-03-02')
  self.assertEqual(actual['low_accrual_bounded'],1/101)
 def test_zero_and_negative_ratios(self):
  e=event();e['fields']['current']['net_income']['value_decimal']='0'
  e['fields']['current']['operating_cash_flow']['value_decimal']='-25'
  actual=r.report_oracle(e,'2026-03-02')
  self.assertEqual([actual[k] for k in r.EXTRA[:3]],[0,-.2,-2/3])
 def test_huge_ratio_has_finite_float_limit(self):
  e=event();e['fields']['current']['net_income']['value_decimal']='1e500'
  actual=r.report_oracle(e,'2026-03-02')
  self.assertEqual(actual['earnings_bounded'],1.)
  self.assertEqual(actual['low_accrual_bounded'],-1.)
 def test_closed_form_matches_generated_rational_examples(self):
  for c,n,a,b in [(13,-7,12,25),(-40,-40,20,80),(0,0,1,1),(1,2,100,100)]:
   actual=r.bounded_from_cells([(c,0),(n,0),(a,0),(b,0)])
   average=Fraction(a+b,2);values=(Fraction(n)/average,Fraction(c-n)/average,-Fraction(a-b,b))
   self.assertEqual(actual,tuple(float(x/(1+abs(x))) for x in values))
 def test_asset_expansion_contraction_and_no_change_direction(self):
  for current,prior,wanted in ((50,100,1/3),(100,100,0.),(200,100,-.5)):
   result=r.bounded_from_cells([(30,0),(10,0),(current,0),(prior,0)])
   self.assertEqual(result[2],wanted)
 def test_nearly_total_contraction_and_huge_expansion_bounds(self):
  contraction=r.bounded_from_cells([(1,0),(1,0),(1,-500),(1,0)])[2]
  expansion=r.bounded_from_cells([(1,0),(1,0),(1,500),(1,0)])[2]
  self.assertEqual(contraction,.5);self.assertEqual(expansion,-1.)
 def test_equal_assets_in_different_scales_are_zero_growth(self):
  self.assertEqual(r.bounded_from_cells([(3,1),(1,1),(15,1),(150000,-3)])[2],0.)
 def test_assets_near_tie_preserves_sign_before_float(self):
  a=10**200
  self.assertLess(r.bounded_from_cells([(30,0),(10,0),(a+1,0),(a,0)])[2],0.)
  self.assertGreater(r.bounded_from_cells([(30,0),(10,0),(a,0),(a+1,0)])[2],0.)
 def test_assets_cannot_widen_mask_when_cash_flow_is_missing(self):
  e=event();e['fields']['current']['operating_cash_flow']=None
  result=r.report_oracle(e,'2026-03-02')
  self.assertFalse(result['available']);self.assertIsNone(result['low_asset_growth_bounded'])
 def test_nonpositive_assets_fail_common_mask(self):
  for side in r.SIDES:
   for amount in ('0','-1'):
    e=event();e['fields'][side]['total_assets']['value_decimal']=amount
    actual=r.report_oracle(e,'2026-03-02');self.assertFalse(actual['available'])
    self.assertEqual(actual['reason'],'nonpositive_asset_endpoint')
 def test_invalid_monetary_representations_fail(self):
  for amount in ('NaN','Infinity','1,000',' 10','True'):
   e=event();e['fields']['current']['net_income']['value_decimal']=amount
   self.assertEqual(r.report_oracle(e,'2026-03-02')['reason'],'invalid_current_net_income')
 def test_nonpositive_scales_fail(self):
  for scale in ('0','-1'):
   e=event();e['fields']['current']['net_income']['scale']=scale
   self.assertFalse(r.report_oracle(e,'2026-03-02')['available'])
 def test_comparison_flow_values_are_unused(self):
  e=event();e['fields']['comparison']['operating_cash_flow']=None;e['fields']['comparison']['net_income']={'poison':True}
  self.assertEqual(r.report_oracle(e,'2026-03-02'),r.report_oracle(event(),'2026-03-02'))
 def test_all_four_provenance_cells_required(self):
  for side,field in r.REQUIRED:
   e=event();e['fields'][side][field]['available_from']='2026-02-03'
   self.assertEqual(r.report_oracle(e,'2026-03-02')['reason'],'invalid_'+side+'_'+field)


class CausalTests(unittest.TestCase):
 def test_latest_invalid_blocks_older_valid(self):
  e=event('later');reschedule(e,'2026-02-20');e['cash_quality_admission']['admitted']=False
  actual=r.cell_oracle([event(),e],'2026-03-02')
  self.assertFalse(actual['available']);self.assertEqual(actual['current_event_id'],'later')
 def test_latest_missing_blocks_older_valid(self):
  e=event('later');reschedule(e,'2026-02-20');e['fields']['current']['net_income']=None
  actual=r.cell_oracle([event(),e],'2026-03-02')
  self.assertEqual(actual['reason'],'missing_current_net_income')
 def test_future_poison_is_not_examined(self):
  future=dict(event_id='future',available_from='2026-12-01',poison=True)
  self.assertEqual(r.cell_oracle([event(),future],'2026-03-02'),r.cell_oracle([event()],'2026-03-02'))
 def test_duplicate_and_ambiguous_latest(self):
  e=event();self.assertTrue(r.cell_oracle([e,deepcopy(e)],'2026-03-02')['available'])
  other=event();other['fields']['current']['net_income']['value_decimal']='99'
  self.assertEqual(r.cell_oracle([e,other],'2026-03-02')['reason'],'conflicting_event_id')
  self.assertEqual(r.cell_oracle([e,event('other')],'2026-03-02')['reason'],'current_ambiguous_latest_event')
 def test_order_invariance(self):
  e=event('earlier');reschedule(e,'2026-02-01')
  self.assertEqual(r.cell_oracle([event(),e],'2026-03-02'),r.cell_oracle([e,event()],'2026-03-02'))
 def test_bad_known_time_never_silently_dropped(self):
  e=event();e['available_from']='bad'
  with self.assertRaises(ValueError):r.cell_oracle([e],'2026-03-02')
 def test_exact_550_day_boundary(self):
  e=event();end=date(2025,12,31)
  self.assertTrue(r.report_oracle(e,str(end+timedelta(days=550)))['available'])
  self.assertEqual(r.report_oracle(e,str(end+timedelta(days=551)))['reason'],'stale_current_report')
 def test_both_annual_boundaries_and_no_duration_normalization(self):
  e=event();expected=r.report_oracle(e,'2026-03-02')
  for current,previous in ((330,400),(400,330)):
   end=date(2025,12,31);start=end-timedelta(days=current-1);prior=end-timedelta(days=current)
   e['periods']=dict(current=period(str(start),str(end)),comparison=period(str(prior-timedelta(days=previous-1)),str(prior)))
   for side in r.SIDES:
    for cell in e['fields'][side].values():cell['period']=deepcopy(e['periods'][side])
   actual=r.report_oracle(e,'2026-03-02')
   self.assertTrue(actual['available']);self.assertEqual([actual[k] for k in r.EXTRA[:3]],[expected[k] for k in r.EXTRA[:3]])
 def test_unknown_comparison_duration_invalid(self):
  e=event();e['periods']['comparison']['period_start']=None
  self.assertEqual(r.report_oracle(e,'2026-03-02')['reason'],'comparison_invalid_period_dates')
 def test_inherited_hold_is_not_overridden(self):
  e=event();e['inherited_rejection_codes']=['hold']
  self.assertEqual(r.report_oracle(e,'2026-03-02')['reason'],'inherited_rejection_not_cleared')
 def test_admission_definition_and_reasons_required(self):
  for change in ({'definition':'old'},{'reasons':['hold']},{'admitted':1}):
   e=event();e['cash_quality_admission'].update(change)
   self.assertFalse(r.report_oracle(e,'2026-03-02')['available'])


class FeatureTests(unittest.TestCase):
 def test_mask_and_all98_audit_cells(self):
  meta,data=fixture();actual=r.features_oracle(meta,data)
  self.assertEqual(len(actual['cells']),98);self.assertEqual(actual['arrays']['common_available'].sum(),2)
  for k in r.EXTRA[:3]:np.testing.assert_array_equal(np.isfinite(actual['arrays'][k]),actual['arrays']['common_available'])
  self.assertFalse(actual['arrays']['common_available'][:,32:].any())
 def test_etf_event_never_becomes_feature(self):
  meta,data=fixture();data['events'].append(event('etf','ETF0'))
  self.assertEqual(r.features_oracle(meta,data)['cells'][32]['reason'],'etf_not_applicable')
 def test_prefix_invariance(self):
  meta,data=fixture();full=r.features_oracle(meta,data)
  for n in range(3):
   sub=dict(meta,dates=meta['dates'][:n],symbols=meta['symbols'][:n]);partial=r.features_oracle(sub,data)
   self.assertEqual(partial['cells'],full['cells'][:n*49])
   for k in r.EXTRA:np.testing.assert_array_equal(partial['arrays'][k],full['arrays'][k][:n])
 def test_raw_inputs_not_mutated(self):
  meta,data=fixture();before=deepcopy((meta,data));r.features_oracle(meta,data);self.assertEqual((meta,data),before)
 def test_empty_dataset_is_missing_not_zero(self):
  meta,data=fixture();data['events']=[];actual=r.features_oracle(meta,data)
  self.assertTrue(np.isnan(actual['arrays'][r.EXTRA[0]]).all());self.assertFalse(actual['arrays']['common_available'].any())
 def test_schema_or_axes_fail(self):
  meta,data=fixture();data['method']='old'
  with self.assertRaises(ValueError):r.features_oracle(meta,data)
  meta,data=fixture();meta['dates'].reverse()
  with self.assertRaises(ValueError):r.features_oracle(meta,data)
 def test_no_new_producer_imports(self):
  tree=ast.parse(Path(r.__file__).read_text())
  for node in ast.walk(tree):
   if isinstance(node,ast.Import):self.assertFalse(any(name.name.endswith(('_factor','_runner')) for name in node.names))
   if isinstance(node,ast.ImportFrom):self.assertFalse((node.module or '').endswith(('_factor','_runner')))


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


class AdmissionAndAuditTests(unittest.TestCase):
 def test_valid_gate_and_required_legacy_hold(self):
  e=event();sessions=['2026-01-30','2026-02-02']
  self.assertTrue(r.source_gate_oracle(e,sessions))
  e['prior_field_admitted']=False;self.assertFalse(r.source_gate_oracle(e,sessions))
 def test_first_session_is_strictly_after_release(self):
  e=event();self.assertFalse(r.source_gate_oracle(e,['2026-01-30','2026-02-01','2026-02-02']))
 def test_source_gate_does_not_claim_signal_day_validity(self):
  e=event();e['release_date']='2028-01-03';reschedule(e,'2028-01-04')
  self.assertTrue(r.source_gate_oracle(e,['2028-01-03','2028-01-04']))
  self.assertEqual(r.report_oracle(e,'2028-02-01')['reason'],'stale_current_report')
 def test_source_admission_does_not_compute_ratios(self):
  with patch.object(r,'report_oracle',side_effect=AssertionError('No factor computation in source gate')),patch.object(r,'bounded_from_cells',side_effect=AssertionError('No ratio')):
   self.assertTrue(r.source_gate_oracle(event(),['2026-02-02']))
 def test_all_four_raw_cells_have_preserved_flags_scope_and_currency(self):
  for side,field in r.REQUIRED:
   for key,value in [('fact_admitted',True),('currency_iso','EUR')]:
    e=event();e['fields'][side][field][key]=value
    self.assertFalse(r.source_gate_oracle(e,['2026-02-02']))
   e=event();e['fields'][side][field]['locator']['period_end']='2025-01-01'
   self.assertFalse(r.source_gate_oracle(e,['2026-02-02']))
  e=event();e['fields']['current']['net_income']['locator']['ni_scope']='other'
  self.assertFalse(r.source_gate_oracle(e,['2026-02-02']))
 def test_valid_synthetic44_event_admission_closure(self):
  reader,plan,pins,sessions=admission_fixture()
  result=r.verify_admission(reader,plan,pins,sessions)
  self.assertEqual(len(result['events']),44)
  self.assertEqual(sum(e['cash_quality_admission']['admitted'] for e in result['events']),43)
 def test_dataset_monetary_or_metadata_changes_fail(self):
  for key,value in [('release_date','2026-01-29'),('symbol','AMD')]:
   reader,plan,pins,sessions=admission_fixture()
   reader.documents[str(reader.path('DATASET.json'))]['events'][0][key]=value
   with self.assertRaisesRegex(ValueError,'Raw event preservation'):r.verify_admission(reader,plan,pins,sessions)
 def test_invalid_reclassified_hold_fails(self):
  reader,plan,pins,sessions=admission_fixture()
  reader.documents[str(reader.path('DATASET.json'))]['events'][-1]['cash_quality_admission'].update(admitted=True,reasons=[])
  with self.assertRaisesRegex(ValueError,'Independent admission decision'):r.verify_admission(reader,plan,pins,sessions)
 def test_receipt_cannot_omit_hold(self):
  reader,plan,pins,sessions=admission_fixture()
  reader.documents[str(reader.path('ADMISSION.json'))]['holds']=[]
  with self.assertRaisesRegex(ValueError,'Every admission hold'):r.verify_admission(reader,plan,pins,sessions)
 def test_conditional_scope_cannot_be_promoted(self):
  reader,plan,pins,sessions=admission_fixture()
  reader.documents[str(reader.path('DATASET.json'))]['conditional_research_only']=False
  with self.assertRaisesRegex(ValueError,'Conditional dataset'):r.verify_admission(reader,plan,pins,sessions)
 def test_source_policy_must_be_in_registered_closure(self):
  reader,plan,pins,sessions=admission_fixture();del pins[str(reader.path('POLICY.json'))]
  with self.assertRaisesRegex(ValueError,'Source policy binding'):r.verify_admission(reader,plan,pins,sessions)
 def test_valid_feature_audit_and_bound_inputs(self):
  saved,expected=audit_fixture();r.verify_audit(saved,expected,dataset_sha='d'*64,admission_sha='a'*64)
  with self.assertRaisesRegex(ValueError,'Source audit bindings'):
   r.verify_audit(saved,expected,dataset_sha='x'*64,admission_sha='a'*64)
 def test_asset_use_binding_required_when_supplied(self):
  saved,expected=audit_fixture()
  r.verify_audit(saved,expected,asset_policy_sha='p'*64,asset_review_sha='r'*64)
  for changes in ({'asset_use_policy_sha256':'wrong'},{'asset_use_review_sha256':'wrong'}):
   with self.assertRaisesRegex(ValueError,'Asset-use audit bindings'):
    r.verify_audit(dict(saved,**changes),expected,asset_policy_sha='p'*64,asset_review_sha='r'*64)
 def test_feature_audit_every_cell_reason_and_type_checked(self):
  for key,value in [('reason','wrong'),('available',1),('earnings_bounded',.5)]:
   saved,expected=audit_fixture();saved['cells'][0][key]=value
   with self.assertRaises(ValueError):r.verify_audit(saved,expected)
 def test_feature_audit_cannot_claim_different_computation(self):
  for key,value in [('comparison_flow_values_used',True),('annual_duration_normalization',True),('new_model_fits',1),('currency_scope','ISO')]:
   saved,expected=audit_fixture();saved[key]=value
   with self.assertRaises(ValueError):r.verify_audit(saved,expected)
 def test_unknown_feature_cell_metadata_is_not_silently_ignored(self):
  saved,expected=audit_fixture();saved['cells'][0]['source_assumption_cleared']=True
  with self.assertRaisesRegex(ValueError,'Exact feature audit cell schema'):r.verify_audit(saved,expected)
 def test_explicit_plan_sha_before_reader_construction(self):
  with patch.object(r.old,'Reader') as reader:
   for digest in (None,'bad','A'*64):
    with self.assertRaisesRegex(ValueError,'Explicit PLAN SHA'):r.audit('/synthetic','PLAN.json',digest)
   reader.assert_not_called()
 def test_all123_prefixes_and_four_arm_fallback_on_synthetic_axes(self):
  dates=[str(d.date()) for d in r.pd.date_range('2026-03-01',periods=122,freq='MS')]
  meta,data=fixture(dates);meta['ends']=dates[:]
  template,_,_=score_fixture();arrays={}
  for key,value in template.items():
   arrays[key]=np.resize(value,(122,49) if value.ndim==2 else (122,))
  features=r.features_oracle(meta,data)
  scores,choices,_=r.score_oracle(arrays,meta,*(features['arrays'][k] for k in r.EXTRA[:3]))
  r.verify_prefixes(arrays,meta,data,features,scores,choices)


if __name__=='__main__':
 if '--receipt' not in r.sys.argv:
  unittest.main()
 else:
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--receipt',type=Path,required=True)
  args=parser.parse_args()
  expected=r.ROOT/'work/monthly_asset_growth147_prepare_review_SYNTHETIC.json'
  r.require(args.receipt.resolve()==expected,'Dedicated synthetic receipt path')
  r.require(not expected.exists(),'Preserve existing synthetic receipt')
  logpath=r.ROOT/r.STUDY/'PREPARE_REVIEW_TESTS.log'
  r.require(not logpath.exists(),'Preserve existing synthetic log')
  logpath.parent.mkdir(parents=True,exist_ok=True)
  stream=io.StringIO()
  result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(r.sys.modules[__name__]))
  log=stream.getvalue();print(log)
  with logpath.open('x') as output:output.write(log)
  receipt=r.synthetic_receipt(result)
  receipt['test_log']=dict(path=str(logpath),sha256=hashlib.sha256(logpath.read_bytes()).hexdigest())
  with expected.open('x') as stream:
   json.dump(receipt,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
  print(json.dumps(dict(passed=receipt['passed'],tests=receipt['tests_run'],source_files=len(receipt['source_hashes']),
   receipt=str(expected),sha256=hashlib.sha256(expected.read_bytes()).hexdigest(),
   checker_sha256=receipt['source_hashes'][str(Path(r.__file__).resolve())],
   tests_sha256=receipt['source_hashes'][str(Path(__file__).resolve())])))
  raise SystemExit(0 if result.wasSuccessful() else 1)
