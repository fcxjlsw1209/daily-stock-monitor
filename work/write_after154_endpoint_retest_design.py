"""Metadata-only bounded retest design; never loads NPZ arrays or imports models."""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'work/after154_endpoint_retest_design.json'
def read(rel):
    return json.loads((ROOT / rel).read_text())
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()
def rel(path):
    return str(Path(path).relative_to(ROOT))

plan = read('outputs/monthly_paired_target143/PLAN.json')
meta = read('outputs/monthly_local_linear70/META.json')
tm = read(rel(plan['inputs']['training_meta']['path']))['training']
calendar = read('outputs/monthly_paired_target143/CALENDAR.json')
months = {}
for day in calendar['sessions']:
    months.setdefault(day[:7], []).append(day)
def next_month(day):
    y, m = map(int, day[:7].split('-'))
    return '%04d-%02d' % (y + (m == 12), 1 if m == 12 else m + 1)
bad_train = [dict(signal=s, recorded_end=e) for s,e in zip(tm['dates'],tm['ends'])
             if s != months[s[:7]][-1] or e != months[next_month(s)][-1]]
non_monthend_queries = [dict(index=i, signal=s, recorded_end=meta['ends'][i])
    for i,s in enumerate(meta['dates']) if s != months[s[:7]][-1]]
partial_evaluation_ends = [dict(index=i, signal=s, recorded_end=e,
    scheduled_target_end=months[next_month(s)][-1])
    for i,(s,e) in enumerate(zip(meta['dates'],meta['ends']))
    if s == months[s[:7]][-1] and e != months[next_month(s)][-1]]
refs = [
    'outputs/monthly_architecture_restart/residual_model.py',
    'outputs/monthly_target_definition70/labels.py',
    'outputs/monthly_target_definition70/model.py',
    'outputs/monthly_target_definition70/study.py',
    'outputs/monthly_target_definition70/PLAN.json',
    'outputs/monthly_local_linear70/model.py',
    'outputs/monthly_local_linear70/study.py',
    'outputs/monthly_horizon70_research/labels.py',
    'outputs/monthly_earnings_repair/research_calendar.py',
    'work/after142_paired_target_labels.py',
    'work/monthly_paired_target143_model.py',
    'work/monthly_paired_target143_runner.py',
    'work/monthly_paired_target143_proposal.json',
    'outputs/monthly_paired_target143/PLAN.json',
    'outputs/monthly_paired_target143/CALENDAR.json',
    'work/after143_stability_novelty.json',
    'work/after152_additional_mechanism_review.json',
    'work/after148_next_mechanism_review.json',
    'work/write_after154_endpoint_retest_design.py',
]
refs += [rel(plan['inputs'][key]['path']) for key in ('meta','training_meta','train_inputs','prices','policy')]
refs += [rel(path) for path in plan['calendar']['source_hashes']]
hashes = {p: sha(ROOT / p) for p in sorted(set(refs))}
cache_checks = {}
for key in ('meta','training_meta','train_inputs','prices','policy'):
    item = plan['inputs'][key]
    actual = hashes[rel(item['path'])]
    if actual != item['sha256']:
        raise ValueError('Frozen143 input digest changed: ' + key)
    cache_checks[key] = dict(path=rel(item['path']), sha256=actual, matches_frozen143=True,
        contents_inspected=('metadata_only' if key in ('meta','training_meta') else False))
if bad_train:
    raise ValueError('Training dates do not satisfy the proposed full natural-month contract')
record = dict(
    schema_version=1, stage='bounded_retest_design_only', created_utc=datetime.now(timezone.utc).isoformat(),
    registered=False, ready_for_prospective_implementation=True, actual_label_calculations=0,
    actual_feature_calculations=0, actual_model_fits=0, actual_orders=0, actual_returns=0,
    novelty=dict(status='historical_repetition_not_excluded', missing_packages_reported=32,
        historical_scope=116, source='work/after152_additional_mechanism_review.json',
        conclusion='Bounded prospective retest is permitted; no claim of archival novelty or pristine out-of-sample evidence.'),
    hypothesis='At fixed predictors and sample, preferring lower fitted probability-of-negative-total-return rankings may differ from preferring higher fitted mean return. The binary target discards gain/loss magnitudes, so a win is not guaranteed and tail losses remain possible.',
    nearest_verified_methods=[
        dict(path='outputs/monthly_target_definition70/labels.py',
            target='continuous excess - 0.5 * max(0,-absolute_return), clipped and monthly centered',
            difference='Not a binary absolute-loss target. Other arms are simple/log excess and volatility-scaled excess.'),
        dict(path='work/after142_paired_target_labels.py', target='paired continuous close/open log excess, clipped and monthly centered',
            difference='Availability is postponed to the opening after next month-end; model also adds local residual correction. Do not reuse these semantics implicitly.')],
    observed_metadata=dict(training_months=len(tm['dates']), training_rows=sum(map(len,tm['symbols'])),
        evaluation_queries=len(meta['dates']), evaluation_slots=sum(map(len,meta['symbols'])),
        training_calendar_boundary_mismatches=bad_train,
        non_monthend_queries=non_monthend_queries, partial_evaluation_ends=partial_evaluation_ends,
        npz_arrays_loaded=False, price_values_read=False),
    candidate_contract=dict(
        arms=['incumbent_local67','matched_mean_return_ridge','matched_loss_indicator_ridge'],
        labels=dict(signal='s = official last session of a calendar month',
            endpoint='e = official last session of the following calendar month; derive before inspecting prices',
            mean_return='r = adjusted_close[e] / adjusted_close[s] - 1; y_return = r (finite raw simple return, no clipping or monthly centering)',
            loss='y_loss = 1 if adjusted_close[e] < adjusted_close[s], else 0; exact zero return is non-loss',
            availability='e close; an annual fit uses only e <= preceding December31 cutoff',
            forbidden=['SPY-relative loss classification','path-minimum loss','within-month stop/hit label','clipping or centering before binary classification','label endpoint shortened to final cached price']),
        training=dict(predictors='Unchanged cached plain10, train_plain/event_plain; no new source/features',
            source='outputs/monthly_target_definition70/TRAIN_INPUTS.npz',
            shared_rows='frozen common_good & train_eligible & finite plain10 & exact finite positive adjusted endpoints & finite raw return; then annual maturity and five-year bounds; require at least two shared names per training month',
            inherited_mask_caveat='common_good retains older full-path, positive-vol63 and SPY/continuous-target validity restrictions. Preserve explicitly in both arms; do not misdescribe this as a newly endpoint-only cohort.',
            annual_clock='One fit per prediction calendar year; cutoff preceding December31; signal strictly after cutoff minus five calendar years; end <= cutoff',
            minimum_mature_months=18, alpha=10, half_life_months=12,
            weight='Equal aggregate month weight before calendar-age decay, normalize to training row count',
            scaler='Weighted means/scales fit on exact common training rows only; same row/weight/scaler hashes for both arms',
            intercept='Use existing unpenalized weighted target-mean intercept; no class balancing, resampling, calibration or local residual correction',
            implementation='Reuse residual_model._fit_ridge/_predict/month_balanced_weights and adapt target_definition70 annual guard; do not call four-target wrapper unchanged',
            prediction='A Ridge linear-probability score, potentially outside [0,1], used only for ranking. Not a calibrated probability; no 0.5 trading threshold.'),
        query_and_ranking=dict(
            shared_query_mask='Original signal-time eligible and finite event_plain, both forecasts finite; never depend on that query future outcome or future price completeness',
            rank='For each shared query population use average-rank0/(n-1)-0.5 on return prediction and on negative loss prediction; n<=1 and all ties => 0; missing => 0 in both',
            dose=0.01, maximum_absolute_additive_correction=0.005,
            application='Add to the same unchanged pre-score; apply the original episode selector once; preserve original core long-momentum skip42, universe, eligibility and EW3/fallback execution',
            first_partial_query='2016-08-17 is not a month-end: both new arms use exact incumbent fallback for that existing query; do not relabel it as a full natural-month training observation',
            terminal_partial_replay='Existing final evaluation end 2026-09-24 remains a disclosed partial-period diagnostic, identical across arms. A 2026-08-31 label would require scheduled 2026-09-30, never 09-24; absent endpoint is missing, not training data.'),
        selection='First88 only: binary arm must pass the frozen strict same-close guard against both controls; next-open5bp noninferior incumbent and strictly above matched-return control with existing risk/subperiod tolerances. Later34/full diagnostics cannot rescue or retune.',
        interpretation='The two new arms isolate target choice. Their difference from incumbent also includes the added pooled-model correction. Historical repetition and reused evaluation history must remain disclosed.'),
    price_and_calendar_contract=dict(
        adjusted_close='Use frozen143 admitted corporate-action-adjusted series on the same unit scale; no adding dividends again and no raw-close substitution',
        parity='Existing local_linear70.verify_panel_parity checks constant per-security adjusted-close scale and adjusted_open = adjusted_close * raw_open/raw_close. New preparation must reverify source closure and actual parity; this design only inspected code and input digests.',
        calendar='Reuse frozen143 CALENDAR plus its pinned exception/runtime source closure. Existing exchange calendar with verified closure overlay is not a newly certified day-by-day historical official archive.',
        next_open='Execution diagnostics retain the same admitted adjusted opening convention; the close-endpoint target does not pretend to predict a net-of-next-open/cost probability.'),
    minimum_synthetic_and_independent_checks=[
        'Exact positive/negative/zero endpoint labels; no SPY or monthly-mean reclassification',
        'December/January and leap-year boundaries; official month-end only; explicit initial non-month-end fallback',
        'Missing next month-end never shortened to last visible price; missing/zero/nonfinite inputs never fill zero',
        'Annual maturity excludes any label ending after cutoff; five-year lower bound exclusive',
        'Both targets have identical row IDs, weights, scaler, minimum-month and query masks; query availability ignores future labels',
        'Constant-class binary sample gives finite intercept-only score/ties; no fake fit failure, clipping or probability-certification',
        'Common per-security price scaling leaves labels unchanged; adjusted/raw mismatches fail parity',
        'All original score arrays/orders preserved for incumbent; one episode call; both unavailable corrections exact fallback',
        '123 prefix checks and future perturbation; independent label/fit oracle before actual ranking/replay'],
    blockers=dict(technical_source_blocker_found=False,
        before_registration=['Implement new two-target pure adapter and independent oracle plus synthetic review',
            'Freeze exact label/rank/first-partial-query policy and complete source/runtime/hash closure',
            'Keep bounded-retest novelty and inherited sample-mask limitations in proposal']),
    input_hashes=hashes, frozen_cache_digest_checks=cache_checks,
)
with OUT.open('x') as f:
    json.dump(record,f,ensure_ascii=False,indent=2,allow_nan=False)
    f.write('\n')
print(json.dumps(dict(output=rel(OUT),sha256=sha(OUT),training_months=len(tm['dates']),
    training_rows=sum(map(len,tm['symbols'])),non_monthend_queries=non_monthend_queries,
    partial_evaluation_ends=partial_evaluation_ends,reference_hashes=len(hashes)),ensure_ascii=False))
