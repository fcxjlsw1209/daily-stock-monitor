"""226 source review: existing byte hashes, saved certificates, JSON axes/clocks.
No NPZ decode, strategy imports, new errors, common support, risks or returns.
"""
from pathlib import Path
from datetime import date, datetime, timezone
import argparse, hashlib, json
R = Path(__file__).resolve().parents[1]
CONTEXT = 'work/monthly_joint_error226_context.json'
CONTEXT_SHA = 'e3e73b2c5294d67377f28b0c778605dff4d38bb9833ac8058e03ac3b14f13224'
COVERAGE = 'work/after225_prediction_error_covariance_coverage.json'
COVERAGE_SHA = 'bfc526e34c4a1473340aa14792c555ded106c3d323a74f37e161fc468dbfa26a'
OUTPUT = 'work/joint_error226_SOURCE_REVIEW.json'
PREVIOUS = 'work/disagreement216_SOURCE_REVIEW.json'
PREVIOUS_SHA = '3cc61c551eccf4ad0455d6f708cd67d2a5d800769b3a3b757a727af7f30494e5'
CALENDAR = 'outputs/monthly_paired_target143/CALENDAR.json'
CALENDAR_SHA = '8ee4da69fba84db1edee37a7e0f5af5d3ab16c6257e5d37cddbd531f01404622'
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


def source(context_sha):
    require(context_sha == CONTEXT_SHA and not path(OUTPUT).exists(), 'Context/write-once gate')
    context = read(CONTEXT, CONTEXT_SHA)
    require(context['round'] == 226 and context['name'] == 'monthly_joint_error226', 'Round identity')
    coverage = read(COVERAGE, COVERAGE_SHA)
    require(coverage['coverage_scope']['strategy_parameters_locked'] is False, 'Coverage is not the strategy registration')
    previous = read(PREVIOUS, PREVIOUS_SHA)
    evidence = context['evidence_sha256']
    for p,h in evidence.items(): bind(p,h)
    for p,h in coverage['source_sha256'].items(): bind(p,h)
    # Original immutable 206 plan is the dependency map, not a new model import.
    inherited = previous['evidence_sha256']
    plan = read(SOURCE+'PLAN.json', inherited[SOURCE+'PLAN.json'])
    local_plan = read(LOCAL+'PLAN.json', inherited[LOCAL+'PLAN.json'])
    raw = plan['raw_source_hashes']
    require(len(raw) == 64 and raw == local_plan['raw_source_hashes'], 'Original64 raw source map')
    for p,h in raw.items(): bind(p,h)
    for p,h in plan['input_sha256'].items(): bind(p,h)
    closed = read(SOURCE+'COMPLETION.json', inherited[SOURCE+'COMPLETION.json'])
    prepared = read(SOURCE+'prepared/COMPLETION.json', inherited[SOURCE+'prepared/COMPLETION.json'])
    require(closed['status'] == 'COMPLETED_INDEPENDENTLY_VERIFIED' and prepared['status'] == 'COMPLETE', 'Incomplete206')
    require(closed['plan_sha256'] == prepared['plan_sha256'] == sha(SOURCE+'PLAN.json'), '206 PLAN binding')
    exact_members = {'RANKS_DELTAS.npz','ORDERS.json','INDEPENDENT_PREPARE.json','MODEL_AUDIT.json','FORECASTS.npz','SOURCE_PARITY.json','META.json'}
    require(set(prepared['artifacts_sha256']) == exact_members and prepared['original_array_count'] == 27, 'Exact206 prepared manifest')
    for p,h in prepared['artifacts_sha256'].items(): bind(SOURCE+'prepared/'+p,h)
    independent = read(SOURCE+'prepared/INDEPENDENT_PREPARE.json', inherited[SOURCE+'prepared/INDEPENDENT_PREPARE.json'])
    proof_fields = ('passed','annual_prior_december_cutoffs_checked','five_year_mature_cohort_checked',
        'four_target_common_cohort_checked','original_source_and_local_parity','availability_unchanged',
        'no_future_or_outcome_inputs','audit_checked')
    for k in proof_fields: require(independent['model'].get(k) is True, 'Saved206 proof missing '+k)
    require(independent['model_prefixes'] == [0,1,53,88,122] and independent['model_prefix_sources_truncated'] is True and independent['original_arrays_unchanged'] is True, 'Saved206 prefix/27-array proof')
    parity = read(SOURCE+'prepared/SOURCE_PARITY.json', inherited[SOURCE+'prepared/SOURCE_PARITY.json'])
    require(parity['raw_source_hashes'] == raw and all(parity.get(k) is True for k in ('original_training_cache_exact','original_identity_exact','original_control_constant_exact')), 'Saved206 source parity')
    saved = read(REBUILD+'PARITY.json', inherited[REBUILD+'PARITY.json'])
    require(saved['passed'] is True, '169 original accepted cache')
    target_fields = ('dates','ends','train_plain','common_good','target_simple_excess','target_log_excess','target_vol_scaled','target_loss_utility','offsets')
    for k in target_fields:
        item = saved['checks']['monthly'][k]
        require(item.get('exact') is True or item.get('max_abs_error') == 0., 'Saved169 exact parity '+k)
    for p in (SOURCE+'prepared/FORECASTS.npz', REBUILD+'TRAIN_INPUTS.npz'): bind(p,inherited[p])
    identity, windows, annual = metadata_review(previous)
    meta = read(SOURCE+'prepared/META.json', inherited[SOURCE+'prepared/META.json'])
    training = read(REBUILD+'MONTHLY_META.json', inherited[REBUILD+'MONTHLY_META.json'])
    calendar = read(CALENDAR,CALENDAR_SHA)
    sessions = calendar['sessions']
    require(sessions == sorted(set(sessions)) and all(canonical(d) == d for d in sessions), 'Calendar canonical unique order')
    last_by_month = {d[:7]:d for d in sessions}
    session_set = set(sessions)
    td,te,_ = month_meta(training['training'])
    for d,e in zip(td,te):
        require(d in session_set and e in session_set, 'Training metadata outside calendar')
        require(d == last_by_month[d[:7]] and e == last_by_month[e[:7]], 'Training not exact natural-month sessions')
    latest_cutoff = max(a['cutoff'] for a in annual)
    for d,e in zip(meta['dates'],meta['ends']):
        require(d in session_set and e in session_set, 'Query metadata outside calendar')
        if d != '2016-08-17': require(d == last_by_month[d[:7]], 'Formal query not exact monthend')
        if e <= latest_cutoff: require(e == last_by_month[e[:7]], 'Consumed historical label not exact monthend')
    clock = dict(calendar_path=CALENDAR,calendar_sha256=CALENDAR_SHA,
        start=calendar['start'],end=calendar['end'],sessions=len(sessions),
        all_training_dates_and_ends_exact_monthend=True,all_query_dates_and_ends_in_calendar=True,
        initial_query_partial='2016-08-17',last_partial_label='2026-09-24',
        only_labels_mature_by_latest_cutoff_require_full_end=True,latest_annual_cutoff=latest_cutoff,
        no_price_arrays_read=True)
    # Read frozen code as text only; never import model/strategy modules.
    paths = dict(labels='outputs/monthly_target_definition70/labels.py', local=LOCAL+'model.py',
                 source='work/monthly_identity_transfer206_model.py', source_host='work/monthly_identity_transfer206_host.py')
    static = {}
    for k,p in paths.items():
        bind(p,inherited.get(p))
        static[k] = path(p).read_text()
    for token in ('clipped = np.clip(values, -limit, limit)', 'targets[arm][selected] = clipped - clipped.mean()'):
        require(token in static['labels'], 'Original target transformation')
    for token in ("raw = {key: ref['log_excess'].copy()", "train = usable & (dates > start) & (ends <= cutoff)"):
        require(token in static['local'], 'Original local annual lineage')
    require("RAW_KEYS = ('source_log', 'local_linear', 'local_constant', 'candidate', 'control')" in static['source'], '206 raw source keys')
    require("'raw_'+key:value for key,value in forecast['raw'].items()" in static['source_host'], '206 saved raw key mapping')
    derivation = 'work/increment_calibration224_source_metadata.py'
    bind(derivation,'030ef9b730b0a84e66c881495abc47722f85b908accc6492beabed5663f7c477')
    implementation = bind(Path(__file__))
    report = dict(
        schema='joint_error226_source_metadata_v1',created_utc=datetime.now(timezone.utc).isoformat(),
        passed=True,source_only=True,context_sha256=CONTEXT_SHA,coverage_sha256=COVERAGE_SHA,evidence_sha256=evidence,
        source_admission_metadata_only=True,source_bytes_reverified=True,metadata_axes_verified=True,
        calendar_clock_verified=True,annual_cutoff_calendar_verified=True,
        raw_source_hashes_verified=True,raw_source_map_matches_original=True,
        source_annual=annual,calibration_window_metadata=windows,calendar_clock_review=clock,
        metadata_identity_review=identity,matched_formal_months=113,
        source_axes=dict(query_months=122,query_width=49,query_rows=5978,training_months=180,
            training_rows=8820,matched_formal_months=113,matched_formal_identity_keys=5537,
            maximum_mature_formal_months=112,source_years=11),
        current_and_historical_prediction_key='raw_local_linear',target_key='target_log_excess',
        historical_use='originalU AND206available',local_available_filter=False,
        uses_fitted_residuals_as_historical_errors=False,
        cached_forecasts_are_prior_annual_asof_replays=True,not_realtime_archival_vintages=True,
        source_npz_decoded=False,training_npz_decoded=False,npz_decoded=False,
        source_code_imported=False,source_npz_directory_metadata_checked=False,
        actual_joint_history_support_computed=False,actual_source_coverage_recomputed=False,
        actual_new_estimates=0,actual_new_statistics=0,actual_new_features=False,
        actual_new_errors=0,actual_new_parameters=0,actual_new_models=0,
        actual_new_orders=False,actual_new_returns=False,
        raw_source_hashes=raw,
        source_prepared_manifest=dict(path=SOURCE+'prepared/COMPLETION.json',sha256=sha(SOURCE+'prepared/COMPLETION.json'),
            artifacts_sha256=prepared['artifacts_sha256'],original_array_count=27),
        saved_numerical_certificate=dict(source206_independent_path=SOURCE+'prepared/INDEPENDENT_PREPARE.json',
            source206_independent_sha256=sha(SOURCE+'prepared/INDEPENDENT_PREPARE.json'),
            reused_true_model_fields=list(proof_fields),physical_model_prefixes=[0,1,53,88,122],
            original_array_parity=True,source169_parity_path=REBUILD+'PARITY.json',
            source169_parity_sha256=sha(REBUILD+'PARITY.json'),original169_exact_fields=list(target_fields),
            source_numerical_checks_reexecuted=False),
        target_semantics_review=dict(target='target_log_excess',
            formula='log(asset_gross)-log(SPY_gross), clipped[-1,1] then centered on the original common monthly cohort',
            centering_can_exceed_precenter_clip_bound=True,unchanged_original_target_required=True,
            target_not_recentered_on_B_or_H=True,label_reconstruction_performed=False,
            error='Original dated annual-asof p minus original cached y; no current fitted residual backfill',
            risk_scope='Relative transformed-target prediction-error risk, not actual EW3 cash-return volatility'),
        common_history_policy=dict(original_history_mask='originalU AND206available',
            training_mask='169common_good AND finite original10 and4 targets AND finite p/y',
            exact_join=['signal_date','security_id','label_end'],
            joint_set='One shared H across every member of the signal-time candidate union B; never pairwise or per-trio deletion',
            current_B_or_H_computed=False,joint_support='UNKNOWN_UNTIL_REGISTERED_PREPARATION',
            missing_or_insufficient_history='Both candidate/control exact incumbent fallback as declared context; eligibility and domain unchanged',
            local_available_filter=False,initial_partial_excluded=True),
        pre_registration_boundary=dict(
            admitted_scope='Byte-verified original inputs and saved certificates only; immutable PLAN must precede new numerical support/errors/covariance/orders/returns.',
            cold_start_upper_bounds='2016/2017/2018 have at most0/4/16 mature formal source months, so cannot meet18; later joint support UNKNOWN.',
            source_vintages='Retrospective provider/identity vintages and causal annual replay, not real-time archived forecasts',
            strict_pit=False,numeric_checks_required_after_PLAN=[
                'Original p/available/27 arrays and exact identity/label alignment',
                'Original historical U, unchanged169good/four-target/plain10 masks',
                'One identical complete mature H for all candidate-union members',
                'Risk estimates and centered-error means use only mature dates',
                'Independent complete-matrix identity/PSD and support/fallback semantics',
                'Full decision prefixes and physically truncated histories']),
        source_validation=dict(context_evidence_pins=len(evidence),source_plan_input_pins=len(plan['input_sha256']),
            raw_source_pins=64,source_prepared_artifacts=7,original_arrays_saved_exact=27,
            all_hashes_passed=True,unique_verified_pins=len(PINS),runtime_external_allowlist=EXTERNAL,
            actual_numeric_arrays_decoded=False,source_code_imported=False,mutable_central_persistent_pins=0),
        source_review_reused=PREVIOUS,source_review_reused_sha256=PREVIOUS_SHA,
        source_parallel_prediction_evidence=dict(previous['source_parallel_prediction_evidence']),
        source_sha256=dict(sorted(PINS.items())),
        strict_pit=False,observers_modified=False,central_modified=False,frozen_files_modified=False,
        limitations=context['limitations'],host_not_pinned_to_avoid_cycle=True,
        implementation=dict(path=str(Path(__file__).relative_to(R)),sha256=implementation,
            metadata_clock_functions_derived_from=derivation))
    report['source_parallel_prediction_evidence']['npz_member_names_source'] = 'Prior frozen216 receipt only; no NPZ directory decode or numerical values read in226 review'
    report['source_parallel_prediction_evidence']['226_uses_only_raw_local_linear_as_error_prediction'] = True
    with path(OUTPUT).open('x') as f:
        json.dump(report,f,indent=2,sort_keys=True); f.write('\n')
    print(json.dumps(dict(path=OUTPUT,sha256=sha(OUTPUT),implementation_sha256=implementation,
        files=len(PINS),source_prepared_members=7,raw_source_members=64,matched_formal_months=113,source_only=True)))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--context-sha256',required=True)
    args=parser.parse_args()
    source(args.context_sha256)
