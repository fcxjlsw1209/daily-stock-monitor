"""Publish the independent read-only review of the fixed round193 code bytes.

This entry point hashes and parses source only. It imports no strategy modules,
loads no financial/market arrays, and runs no factor or portfolio calculations.
"""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PINS={
'work/monthly_cash_ppe193_host.py':'ce10ffb27160d2925226e0a5ff83f4e788c6e1ef41e35c6e494f1c4094cfcdec',
'work/test_monthly_cash_ppe193_host.py':'e492bf63c6e807892a6cf76b6f0365793a0cd4db2f58e39270368cf4b5b1560a',
'work/capex193_factor.py':'410cbb050e99c68c402095e20c2cd472f374f48ab032f64fc8d26365ad9a8c7f',
'work/test_capex193_factor.py':'b773f49d3500a639bbd34a6c11701aa1726900ad3190438bd2a7bc637e682d5b',
'work/monthly_cash_ppe193_independent.py':'a57843087de87a2ebcbe01485342f4eee2c6f39694ff14109b6442f78cba35d4',
'work/test_monthly_cash_ppe193_independent.py':'8bd0d2e9ee9d11ff49abda9ce7486a07bba84dde6195ab4962a216d9d3bfc145'}
BINDINGS={
'work/capex193_experiment_concept.json':'14e69a47bbf3ea9bbc8d60026e1f5092504d28778f52d622b7560b28bc5c1689',
'outputs/monthly_capex193_source/independent/ADMISSION_REVIEW_V2.json':'61a176a244e3e79a80d9d8c909e8193a15adb2b932d65813293d88b013207d08',
'work/capex193_independent_admission_review_v2.py':'7f759db4fab95c0c36c6c7e165d8680ab1a64b656c01798529bdc29c47f0a9c7',
'work/monthly_cash_quality146_factor.py':'6f5d819488c7952cb32894cf4b5a4f549c73adb570e949fd47c3c6ba2e561eea',
'work/monthly_same_report145_runner.py':'f400350c20a6d95f1d7ed60753ff529b5544b2b8901abbd0e801999843500939',
'work/monthly_sector_recurrence192_independent.py':'f78cc005827fd977ec431bdec4af0f4b7160694cb3d382df0115e8e00e0bedf8'}
def sha(p):return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
def require(ok, message):
 if not ok:raise ValueError(message)
def main():
 for p,h in {**PINS,**BINDINGS}.items():require(sha(p)==h,'Reviewed source changed: '+p)
 trees={p:ast.parse((ROOT/p).read_text()) for p in PINS}
 oracle=trees['work/monthly_cash_ppe193_independent.py']
 imports=[]
 for node in ast.walk(oracle):
  if isinstance(node,ast.Import):imports.extend(a.name for a in node.names)
  elif isinstance(node,ast.ImportFrom):imports.append(node.module)
 require(not any(x and ('capex193_factor' in x or 'monthly_cash_quality146_factor' in x) for x in imports),'Oracle imports producer numeric implementation')
 tests={p:sum(isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name.startswith('test_') for n in ast.walk(t)) for p,t in trees.items() if Path(p).name.startswith('test_')}
 require(sum(tests.values())==72,'Reviewed synthetic test census changed')
 source=json.loads((ROOT/'outputs/monthly_capex193_source/independent/ADMISSION_REVIEW_V2.json').read_text())
 require(source['passed'] is True and (source['events'],source['admitted_events'],source['held_events'],source['closure_pins_verified'])==(44,24,20,249),'Source review not complete')
 result=dict(schema='capex193_independent_code_review_v1',created_utc=datetime.now(timezone.utc).isoformat(),passed=True,
  review_scope='Read-only source and synthetic-test review, after independent original-document and source-admission review; no actual numerical execution.',
  code_sha256=PINS,input_sha256=BINDINGS,review_code_sha256=sha('work/capex193_independent_code_review.py'),
  source_admission=dict(events=44,admitted=24,held=20,closure_pins_verified=249,all_original_fields_preserved=True),
  resolved_findings=[
   'Assembler now consumes the actual independent REVIEW and producer VISUAL schemas.',
   'Absent PP&E row dollar literal remains None; statement evidence is tied to exact original CFO locator/table/segment or PDF page, same source and units.',
   'Acquisition of PP&E itself is accepted; mixed intangible/business acquisitions and noncash/principal substitutes are rejected.',
   'Host/oracle receipt names are compatible and exercised through actual pure synthetic 122x49 API integration.',
   'V2 source review excludes obsolete numerical implementation hashes from its active source closure; old V1 remains historical only.'],
  checks=dict(
   latest_report_selected_before_validity=True,held_latest_has_no_fallback=True,
   future_amounts_skipped_after_available_clock=True,first_session_after_release_from_pinned_source_gate=True,
   fiscal_age_550_days_inclusive=True,original_annual_330_to_400_and_contiguous_periods=True,
   original_net_income_gate_retained_but_not_new_ratio_input=True,comparison_flow_amounts_unused=True,
   normalized_ppe_matches_raw_evidence=True,cfo_current_comparison_assets_same_report=True,
   signed_ppe_nonnegative_outflow_and_equal_statement_scale=True,
   arbitrary_signed_cfo_and_ppe_exceeding_cfo_supported=True,
   exact_fraction_ratio_then_one_bounded_float=True,
   oracle_uses_independent_closed_form=True,oracle_imports_no_producer_numeric_module=True,
   oracle_checks_complete_arrays_and_per_cell_audit_exactly=True,
   original_eligible_only_direct_dose_point01=True,no_cross_sectional_rank_or_second_nonlinearity=True,
   no_added_first_partial_suppression_consistent_with_financial146=True,
   original_27_arrays_and_support_preserved=True,original_episode_once_per_arm=True,
   missing_and_zero_inputs_exact_incumbent_fallback=True,
   five_physically_truncated_feature_prefixes_and_123_decision_prefixes=True,
   source_admission_gate_before_real_loader=True,explicit_plan_and_runtime_hash_gate=True,
   first88_only_primary_selection=True,control_not_promotable=True,
   exact_pinned_financial145_risk_guards=True,
   close_strict_vs_both_open_noninferior_vs_incumbent_strict_vs_control=True,
   early53_recent35_for_full_training_and_half_split_for_walk_prefixes=True,
   walkforward_36_60_76_decisions_precede_60_76_88_evaluation=True,
   later34_diagnostic_only=True,independent_9_paths_45_slices_and_18_training_metric_checks=True,
   inherited_sources_and_21444_protection_gate_retained=True,write_once_and_central_compare_swap_retained=True),
  synthetic_tests=dict(source_test_functions=tests,total=72,producer_reports_all_passed=True,counts_reported_by_owners=dict(host=30,factor=23,oracle=19),reviewer_reran_suites=False,formal_combined_receipt='Root to run after this review; this is not a substitute for formal TESTS.'),
  limitations=[
   'Factor and numerical oracle share an implementation author; this separate reviewer examined both source paths and the exact-rational algebra. The oracle independently reconstructs values without importing producer numerical functions.',
   'Source support is limited to the reviewed issuer reports and retains all historical currency, as-filed, security-master and accounting-comparability limitations.',
   'Historical shared PDFium current verification remains failed; only new193 source runtime is admitted. No historical-runtime success is claimed.',
   'Historical periods were reused; neither later diagnostics nor walk-forward summaries are pristine independent OOS.',
   'Static and synthetic review cannot assert actual prepare/selection/accounting success; those require future registered phases.'],
  actual_financial_ratios_computed=False,actual_features_computed=False,actual_portfolio_returns_computed=False,
  strategy_registered_by_reviewer=False,production_files_modified=False,unresolved_blockers=[])
 out=ROOT/'outputs/monthly_capex193_source/independent/CODE_REVIEW.json'
 with out.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
 print(json.dumps(dict(passed=True,path=str(out.relative_to(ROOT)),sha256=hashlib.sha256(out.read_bytes()).hexdigest(),reviewed_test_functions=72)))
if __name__=='__main__':main()
