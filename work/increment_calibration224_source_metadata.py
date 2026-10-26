"""224 source/coverage review: hashes, saved receipts, JSON clocks only.
No numerical arrays, price rows, actual forecasts, products, fits or outcomes.
"""
from pathlib import Path
from datetime import date, datetime, timezone
import argparse, hashlib, json
R = Path(__file__).resolve().parents[1]
CONTEXT = 'work/monthly_increment_calibration224_context.json'
CONTEXT_SHA = '0381d106c1f355f752987e7cb1c4a1a07a8ee1b05e1091dff66080a6a0f17244'
COVERAGE = 'work/after223_increment_calibration_coverage.json'
OUTPUT = 'work/increment_calibration224_SOURCE_REVIEW.json'
PREVIOUS = 'work/disagreement216_SOURCE_REVIEW.json'
PREVIOUS_SHA = '3cc61c551eccf4ad0455d6f708cd67d2a5d800769b3a3b757a727af7f30494e5'
LEAD = 'work/after223_incremental_calibration_lead.json'
LEAD_SHA = '73caac27461f078cdde9b376242e094c3e9a713764afb516c6f8b6df1de38d22'
SOURCE = 'outputs/monthly_identity_transfer206/'
REBUILD = 'outputs/monthly_ptr_pool_source169/fixed_incumbent_rebuild/control/'
LOCAL = 'outputs/monthly_local_linear70/'
EXTERNAL = {'/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/bin/python3.9': '6b24be8ba00c4abc8b3e7e2620bd710e2cf7a4312d00c147f8f05fbb6f3d297a'}
PINS = {}
def require(x, message):
    if not x: raise ValueError(message)
def path(p):
    p = Path(p); p = p if p.is_absolute() else R / p
    require('CloudStorage' not in str(p), 'Cloud path forbidden')
    require((p.is_relative_to(R) and p.resolve().is_relative_to(R)) or str(p) in EXTERNAL, 'Nonlocal source')
    return p
def sha(p):
    h = hashlib.sha256()
    with path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''): h.update(chunk)
    return h.hexdigest()
def bind(p, expected=None):
    p = path(p); got = sha(p)
    require(expected is None or expected == got, 'Digest mismatch ' + str(p))
    require(str(p) not in PINS or PINS[str(p)] == got, 'Source changed during audit')
    PINS[str(p)] = got
    return got
def read(p, expected=None):
    bind(p, expected)
    return json.loads(path(p).read_text())
def canonical(d):
    require(type(d) is str and date.fromisoformat(d).isoformat() == d, 'Noncanonical date')
    return d
def month_meta(m, initial=False):
    ds, es, ids = m['dates'], m['ends'], m['symbols']
    require(len(ds) == len(es) == len(ids) and ds == sorted(set(ds)), 'Metadata axes')
    for d, e, symbols in zip(ds, es, ids):
        canonical(d); canonical(e)
        require(e > d and len(symbols) == 49 and len(set(symbols)) == 49, 'Metadata identity width/end')
        require(all(type(s) is str and s.strip() == s and s for s in symbols), 'Metadata identity strings')
        dy, dm = map(int, d[:7].split('-')); ny, nm = (dy+1,1) if dm == 12 else (dy,dm+1)
        require((initial and d == '2016-08-17' and e == '2016-08-31') or e[:7] == f'{ny:04d}-{nm:02d}', 'Target next natural month')
    formal = [d[:7] for d in ds if not (initial and d == '2016-08-17')]
    require(len(formal) == len(set(formal)), 'Duplicate formal signal month')
    return ds, es, ids

def metadata_review(previous):
    meta = read(SOURCE+'prepared/META.json', previous['evidence_sha256'][SOURCE+'prepared/META.json'])
    training = read(REBUILD+'MONTHLY_META.json', previous['evidence_sha256'][REBUILD+'MONTHLY_META.json'])
    require(meta == training['monthly'] == read(LOCAL+'META.json', previous['evidence_sha256'][LOCAL+'META.json']), 'Query metadata not exact')
    ds, es, ids = month_meta(meta, True)
    td, te, ti = month_meta(training['training'])
    require(len(ds) == 122 and len(td) == 180, 'Frozen metadata dimensions')
    keys = [(d,s,e) for d,e,row in zip(td,te,ti) for s in row]
    require(len(keys) == len(set(keys)) == 8820, 'Duplicate training triple')
    keys = set(keys); years = sorted({int(d[:4]) for d in ds}); latest = f'{years[-1]-1}-12-31'
    unmatched = []; matched_months = 0; matched_keys = 0
    for d,e,row in zip(ds,es,ids):
        if d == '2016-08-17': continue
        missing = [(d,s,e) for s in row if (d,s,e) not in keys]
        if missing:
            require(e > latest, 'Mature expected label keys absent')
            unmatched.append(dict(date=d,end=e,missing_keys=len(missing),reason='after_latest_annual_cutoff_not_a_consumed_calibration_label'))
        else:
            matched_months += 1; matched_keys += len(row)
    require(matched_months == 113 and matched_keys == 5537 and len(unmatched) == 8, 'Frozen exact join coverage')
    windows = []
    for y in years:
        start, cutoff = f'{y-6}-12-31', f'{y-1}-12-31'
        use = [d for d,e in zip(ds,es) if d != '2016-08-17' and d > start and e <= cutoff]
        windows.append(dict(year=y,maximum_months=len(use),first=min(use) if use else None,last=max(use) if use else None))
    require(windows == previous['calibration_window_metadata'], 'Historical metadata window changed')
    audit = read(SOURCE+'prepared/MODEL_AUDIT.json', previous['evidence_sha256'][SOURCE+'prepared/MODEL_AUDIT.json'])
    annual = []
    for a in audit['source']['annual']:
        y = a['year']; require(a['cutoff'] == f'{y-1}-12-31' and a['lookback_exclusive_start'] == f'{y-6}-12-31', 'Forecast annual vintage')
        require(a['latest_label_end'] is None or a['latest_label_end'] <= a['cutoff'], 'Forecast used future label')
        annual.append({k:a[k] for k in ('year','cutoff','lookback_exclusive_start','prediction_rows','latest_label_end')})
    require(annual == previous['source_annual'] and len(annual) == 11, 'Source annual metadata changed')
    identity = dict(query_events=122,query_width=49,query_rows=5978,training_months=180,training_identity_keys=len(keys),unique_training_identity_keys=True,matched_formal_months=matched_months,matched_formal_security_keys=matched_keys,maximum_mature_formal_months=sum(d!='2016-08-17' and e<=latest for d,e in zip(ds,es)),initial_partial_excluded='2016-08-17',future_unmatched_queries=unmatched,numeric_offset_values_not_decoded=True,saved169_offsets_parity_reused=True,query_metadata_equals_local67_and169=True)
    require(identity == previous['metadata_identity_review'], 'Strict metadata review differs')
    return identity, windows, annual

def coverage():
    require(not path(COVERAGE).exists(), 'Coverage is write-once')
    lead = read(LEAD, LEAD_SHA); previous = read(PREVIOUS, PREVIOUS_SHA)
    for p,h in lead['evidence_sha256'].items(): bind(p,h)
    for p in ('work/after205_model_coverage.json','outputs/monthly_honest_local70/test_study.py', 'work/monthly_disagreement216_model.py','work/monthly_disagreement216_host.py','work/monthly_forecast_blend211_host.py','outputs/monthly_local_linear70/model.py','outputs/monthly_target_definition70/labels.py'): bind(p)
    identity, windows, annual = metadata_review(previous)
    for p in (SOURCE+'prepared/FORECASTS.npz', REBUILD+'TRAIN_INPUTS.npz'):
        bind(p,previous['evidence_sha256'][p])
    missing = [f'outputs/{family}/{name}' for family in ('monthly_honest_local70','monthly_kernel_residual70') for name in ('model.py','study.py','features.py','PLAN.json','DESIGN.json')]
    require(all(not path(p).exists() for p in missing), 'Previously missing implementation now exists; inspect before claiming UNKNOWN')
    implementation = bind(Path(__file__))
    report = dict(schema='increment_calibration224_bounded_coverage_v1',created_utc=datetime.now(timezone.utc).isoformat(),status='BOUNDED_EXECUTABLE_DEFINITION_AND_SOURCE_METADATA_REVIEW',family_novelty=False,exact_historical_overlap_status='UNKNOWN for missing honest_local70/kernel_residual70 numeric implementations; no exhaustive novelty claim',input_sha256=dict(sorted(PINS.items())),prior_coverage=lead['prior_coverage'],unknown_coverage=[dict(family='monthly_honest_local70',status='UNKNOWN',retained_test='outputs/monthly_honest_local70/test_study.py',finding='Selector tests establish paired raw forecasts, quarter-slot, original U, one episode and fallback; they do not define residual training, coefficient formula, folds or identity conditioning.'),dict(family='monthly_kernel_residual70',status='UNKNOWN',finding='No directory or recoverable numeric definition found in the bounded filename/reference search.')],missing_checked_paths=missing,bounded_search=dict(scopes=['work/*.py','work/*coverage*.json','work/*context*.json','outputs/**/*.py excluding raw/sources/prices and venv'],reference_results=['outputs/monthly_honest_local70/test_study.py','work/after205_model_coverage.json','work/after210_forecast_blend_coverage.json','work/after214_own_error_coverage.json','work/monthly_relative_tail190_central_before_coverage.json','work/monthly_relative_severity191_central_before_coverage.json'],central_before_outcomes_not_read=True),contrast=dict(candidate='Own-security convex calibration of cached local-linear minus pooled forecast increment against unchanged mature target minus pooled forecast.',control='Same two endpoints and admitted monthly cohort; shared annual month-equal pooled convex coefficient.',not_new='Calibration, error-reliability and local-vs-global forecast families already tested.',remaining_risk='Own-specific sample sparsity and repeated historical research; no gain is assumed.'),reuse=dict(source206_keys=['raw_local_linear','raw_source_log','available'],do_not_reuse211_endpoint_names=True,old211_endpoints='raw_candidate peer versus raw_control own-constant, not local-linear versus pooled',strict_216_join_reusable=True,original_U_explicit=True,no_local_available_filter=True,no_target_reconstruction_or_recentering=True,source_predictions_annual_asof_replay_not_archived_live_vintages=True),metadata_identity_review=identity,calibration_window_metadata=windows,source_annual=annual,actual_numeric_support='UNKNOWN; no arrays decoded, >=2 rows/month and >=18 own months not evaluated',source_npz_decoded=False,training_npz_decoded=False,actual_new_features=0,actual_new_losses=0,actual_new_parameters=0,actual_new_model_fits=0,actual_new_orders=0,actual_new_returns=0,strategy_registered=False,observers_modified=False,strict_pit=False,implementation=dict(path=str(Path(__file__).relative_to(R)),sha256=implementation))
    with path(COVERAGE).open('x') as f: json.dump(report,f,indent=2,sort_keys=True); f.write('\n')
    print(json.dumps(dict(path=COVERAGE,sha256=sha(COVERAGE),implementation_sha256=implementation,pins=len(PINS),source_only=True)))

def source(context_sha):
    require(context_sha == CONTEXT_SHA and not path(OUTPUT).exists(), 'Context/write-once source gate')
    context = read(CONTEXT, CONTEXT_SHA); coverage_data = read(COVERAGE); previous = read(PREVIOUS, PREVIOUS_SHA)
    require(context['round'] == 224 and context['name'] == 'monthly_increment_calibration224', 'Experiment identity')
    require(context['formation']['parameters'] == dict(min_months=18,min_rows_per_month=2,lookback_calendar_years=5), 'Fixed support')
    evidence = dict(context['evidence_sha256'])
    for p,h in evidence.items(): bind(p,h)
    for p,h in coverage_data['input_sha256'].items(): bind(p,h)
    plan = read(SOURCE+'PLAN.json',evidence[SOURCE+'PLAN.json'])
    raw = plan['raw_source_hashes']; local_plan = read(LOCAL+'PLAN.json',evidence[LOCAL+'PLAN.json'])
    require(len(raw) == 64 and raw == local_plan['raw_source_hashes'], 'Original raw source map')
    for p,h in raw.items(): bind(p,h)
    for p,h in plan['input_sha256'].items(): bind(p,h)
    closed = read(SOURCE+'COMPLETION.json',evidence[SOURCE+'COMPLETION.json'])
    prepared = read(SOURCE+'prepared/COMPLETION.json',evidence[SOURCE+'prepared/COMPLETION.json'])
    require(closed['status'] == 'COMPLETED_INDEPENDENTLY_VERIFIED' and prepared['status'] == 'COMPLETE' and closed['plan_sha256'] == prepared['plan_sha256'] == sha(SOURCE+'PLAN.json'), 'Frozen206 closure')
    require(len(prepared['artifacts_sha256']) == 7 and prepared['original_array_count'] == 27, 'Exact source manifest')
    for p,h in prepared['artifacts_sha256'].items(): bind(SOURCE+'prepared/'+p,h)
    independent = read(SOURCE+'prepared/INDEPENDENT_PREPARE.json',evidence[SOURCE+'prepared/INDEPENDENT_PREPARE.json'])
    for k in ('passed','annual_prior_december_cutoffs_checked','five_year_mature_cohort_checked','four_target_common_cohort_checked','original_source_and_local_parity','availability_unchanged','no_future_or_outcome_inputs','audit_checked'): require(independent['model'].get(k) is True,'Saved206 numerical proof missing '+k)
    require(independent['model_prefixes'] == [0,1,53,88,122] and independent['model_prefix_sources_truncated'] is True and independent['original_arrays_unchanged'] is True,'Saved physical-prefix proof')
    parity = read(SOURCE+'prepared/SOURCE_PARITY.json',evidence[SOURCE+'prepared/SOURCE_PARITY.json'])
    require(parity['raw_source_hashes'] == raw and all(parity.get(k) is True for k in ('original_training_cache_exact','original_identity_exact','original_control_constant_exact')), 'Original cache parity')
    saved = read(REBUILD+'PARITY.json',evidence[REBUILD+'PARITY.json'])
    require(saved['passed'] is True,'169 cache not independently accepted')
    for k in ('dates','ends','train_plain','common_good','target_simple_excess','target_log_excess','target_vol_scaled','target_loss_utility','offsets'):
        item = saved['checks']['monthly'][k]; require(item.get('exact') is True or item.get('max_abs_error') == 0.,'Original169 array parity '+k)
    identity, windows, annual = metadata_review(previous)
    for p in (SOURCE+'prepared/FORECASTS.npz', REBUILD+'TRAIN_INPUTS.npz'): bind(p,evidence[p])
    static = {'labels':path('outputs/monthly_target_definition70/labels.py').read_text(),'local':path(LOCAL+'model.py').read_text(),'source':path('work/monthly_identity_transfer206_model.py').read_text(),'join':path('work/monthly_disagreement216_model.py').read_text(),'slot':path('work/monthly_disagreement216_host.py').read_text()}
    for token in ('clipped = np.clip(values, -limit, limit)','targets[arm][selected] = clipped - clipped.mean()'): require(token in static['labels'],'Fixed unchanged label transformation')
    for token in ("raw = {key: ref['log_excess'].copy()", "train = usable & (dates > start) & (ends <= cutoff)","for key, values in raw.items()"):
        require(token in static['local'],'Frozen local forecast lineage')
    implementation = bind(Path(__file__))
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(),passed=True,context_sha256=CONTEXT_SHA,coverage_sha256=sha(COVERAGE),evidence_sha256=evidence,cached_forecasts_are_prior_annual_asof_replays=True,not_realtime_archival_vintages=True,uses_fitted_residuals_as_historical_errors=False,matched_formal_months=113,source_annual=annual,calibration_window_metadata=windows,numeric_calibration_coverage_not_yet_computed=True,numeric_own_security_support_not_yet_computed=True,numeric_pooled_calibration_support_not_yet_computed=True,source_npz_decoded=False,training_npz_decoded=False,source_npz_directory_metadata_checked=False,source_bytes_reverified=True,own_history_mask='originalU AND206available',current_and_historical_prediction_key='raw_local_linear',disagreement_prediction_key='raw_source_log',common_cohort_policy='Mature exact originalU &206available rows, exact date-ID-end join, original common_good and finite plain10/four targets; keep only months with >=2 valid rows for BOTH own and pooled calibration.',pooled_history_current_identity_filter=False,local_available_filter=False,current_prediction_availability_unchanged=True,metadata_identity_review=identity,source_parallel_prediction_evidence=previous['source_parallel_prediction_evidence'],target_semantics_review=dict(target='target_log_excess',source_formula='log(asset_gross)-log(SPY_gross), clipped [-1,1] before centering on original monthly common cohort',centering_can_exceed_precenter_clip_bound=True,target_not_recentered_for_own_or_pooled_calibration=True,both_predictions_same_target_units=True,label_reconstruction_performed=False,calibration_increment='d=p-b and unchanged y-b; not current-model fitted residuals'),pre_registration_boundary=dict(admitted_scope='Pinned cached annual predictions and original cached labels/identity may be used only by a prospectively registered224; metadata review is not actual numerical support or strict PIT certification.',strict_pit=False,source_vintages='Retrospective inputs and annual causal replay, not archived real-time forecasts',cold_start_implication='2016/17/18 maximum0/4/16mature months; own support later UNKNOWN until registration',numeric_checks_required_after_PLAN=['Source p/b/availability exact cached parity','Original historical U and common >=2 valid rows/month','Exact training offsets and unchanged four targets/ten-feature masks','No current-ID or local_available pooled-history filter','Independent finite products, monthly means, coefficients, endpoint copies/fallbacks and fit accounting','All123 physically truncated source/query/label prefixes'],fit_accounting=context['formation']['fit_accounting']),source_validation=dict(context_evidence_pins=len(evidence),source_plan_input_pins=len(plan['input_sha256']),raw_source_pins=len(raw),source_prepared_artifacts=len(prepared['artifacts_sha256']),original_arrays_saved_exact=27,all_hashes_passed=True,unique_verified_pins=len(PINS),runtime_external_allowlist=EXTERNAL,mutable_central_persistent_pins=0,source_code_imported=False,actual_numeric_arrays_decoded=False),verified_input_sha256=dict(sorted(PINS.items())),source_review_reused=PREVIOUS,source_review_reused_sha256=PREVIOUS_SHA,raw_source_hashes=raw,source_admission_metadata_only=True,strict_pit=False,observers_modified=False,central_modified=False,actual_new_losses=0,actual_new_coefficients=0,actual_new_parameters=0,actual_new_calibration_coefficients=0,actual_new_model_fits=0,actual_new_features=0,actual_new_forecasts=0,actual_new_orders=0,actual_new_returns=0,limitations=context['limitations'],implementation=dict(path=str(Path(__file__).relative_to(R)),sha256=implementation),host_not_pinned_to_avoid_cycle=True)
    report['source_parallel_prediction_evidence'] = dict(report['source_parallel_prediction_evidence'])
    report['source_parallel_prediction_evidence']['npz_member_names_source'] = 'Prior frozen216 receipt only; not reopened or decoded in224 review'
    with path(OUTPUT).open('x') as f: json.dump(report,f,indent=2,sort_keys=True); f.write('\n')
    print(json.dumps(dict(path=OUTPUT,sha256=sha(OUTPUT),implementation_sha256=implementation,files=len(PINS),annual_years=len(annual),source_only=True)))
if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('coverage','source'),required=True);p.add_argument('--context-sha256');a=p.parse_args()
    if a.mode=='coverage': coverage()
    else: source(a.context_sha256)
