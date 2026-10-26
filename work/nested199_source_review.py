"""Source-only253-session adjusted-close availability audit for round199.

No strategy imports or NPZ arrays. No horizon gains, horizon ranks, min/mean
features, deltas, orders or portfolio returns. Use run_local.sh.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
O = ROOT / 'outputs'
OUT = ROOT / 'work/nested199_SOURCE_REVIEW.json'
CONTEXT = ROOT / 'work/monthly_nested_trend199_context.json'
CONTEXT_SHA = '2a0b157600acf69ea9325fd3e46bc0a41d3337ff852e34677ee0d58d8b407233'
PRIOR_REVIEW = ROOT / 'work/payoff198_SOURCE_REVIEW.json'
PRIOR_SHA = '40848c92606cff55942492cafaefb0a7367a47435a5e6e3ce23cba4e8d78c84b'
PRIOR = O / 'monthly_event_risk_payoff198'
PRIOR_COMPLETION_SHA = 'a4e03ee8178dcaaf0f13829e0774e44ec72e7ea9c71fbf65f3e196e26f2bb351'
PRIOR_PLAN_SHA = '4ee7b38ed136bcc74230ffd609b842ee82483878d07893549ba77b3e22b25426'
PRIOR_RESULTS_SHA = '7ada48bcb926698b2f3a812c97e3281bbb9b9e582c78fb03217ea1ea7c8f0756'
INTEGRATION_DEPENDENCIES = (
    'outputs/monthly_core_timing70_research/core_adapter.py',
    'outputs/monthly_core_timing70_research/test_core_adapter.py',
    'outputs/monthly_path70_research/incumbent_adapter.py',
    'outputs/monthly_local_linear70/study.py',
    'outputs/monthly_daily_episodes/episodes.py',
    'outputs/monthly_quality_research/explore.py',
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(ok, message):
    if not ok:
        raise ValueError(message)


def quality_counts(values):
    a = np.asarray(values, dtype=float)
    finite = np.isfinite(a)
    return dict(observations=int(a.size), finite_positive=int((finite & (a > 0)).sum()),
        zero=int((finite & (a == 0)).sum()), negative=int((finite & (a < 0)).sum()),
        nan=int(np.isnan(a).sum()), infinite=int(np.isinf(a).sum()))


def window(calendar, day):
    t = calendar.index(day)
    require(t >= 294, 'Official history too short')
    dates = calendar[t-294:t-41]
    require(len(dates) == 253 and dates[-1] == calendar[t-42], 'Wrong253-session window')
    return dates


def self_checks():
    require(quality_counts([1., 0., -1., np.nan, np.inf, -np.inf]) ==
        dict(observations=6, finite_positive=1, zero=1, negative=1, nan=1, infinite=2),
        'Synthetic count partition failed')
    cal = [str(i) for i in range(400)]
    dates = window(cal, '350')
    require(dates[0] == '56' and dates[-1] == '308' and len(dates) == 253, 'Synthetic offsets failed')
    require(quality_counts([1.]*253)['finite_positive'] == 253, 'Positive source coverage failed')
    try:
        window(cal, '293')
    except ValueError:
        pass
    else:
        raise ValueError('Short calendar must fail')
    return 4


def main():
    require(not OUT.exists(), 'Write-once source review already exists')
    checks = self_checks()
    pins = {}
    def pin(path, expected=None):
        path = Path(path).resolve()
        require(path.is_relative_to(ROOT), 'Nonlocal source path')
        digest = sha(path)
        require(expected is None or digest == expected, 'Changed source: ' + str(path))
        require(str(path) not in pins or pins[str(path)] == digest, 'Conflicting source pin')
        pins[str(path)] = digest
        return digest

    pin(Path(__file__))
    pin(CONTEXT, CONTEXT_SHA)
    pin(PRIOR_REVIEW, PRIOR_SHA)
    context, previous = read(CONTEXT), read(PRIOR_REVIEW)
    formation = context['formation']
    require(context['round'] == 199 and context['name'] == 'monthly_nested_trend199'
            and formation['horizons'] == [126, 189, 252]
            and formation['end_offset'] == -42
            and formation['start_offsets'] == [-168, -231, -294]
            and '253 original stored adjusted closes' in formation['prices']
            and 't-294..t-42' in formation['prices'], 'Prospective window differs')
    require(previous['passed'] is True and previous['raw_source_parity']['exact_source_map_equal'] is True,
            'Original loader/source audit unavailable')
    for p, h in previous['source_sha256'].items(): pin(p, h)
    for p, h in context['evidence_sha256'].items(): pin(ROOT/p, h)
    integration_pins = {}
    for p in INTEGRATION_DEPENDENCIES:
        require(p in context['evidence_sha256'], 'Critical integration dependency absent')
        integration_pins[p] = pin(ROOT/p, context['evidence_sha256'][p])
    # Source-text assertions only. Do not import or execute scoring/episode helpers.
    adapter_text = (ROOT/INTEGRATION_DEPENDENCIES[0]).read_text()
    quality_text = (ROOT/INTEGRATION_DEPENDENCIES[5]).read_text()
    require('delta[changed] = blend * (cw[changed] * rank[changed] - core[changed])' in adapter_text
            and 'original.update(baseline=baseline.copy(), core=core.copy(),' in adapter_text
            and 'core_weight=cw.copy())' in adapter_text,
            'Frozen core adapter no longer exposes reviewed support-preserving path')
    require('cw=np.where(calm,.5,.52)' in quality_text
            and "core=np.where(calm,.5*f['252_raw'],d['core'])" in quality_text,
            'Frozen quality source state/core branch changed')
    pin(PRIOR/'COMPLETION.json', PRIOR_COMPLETION_SHA)
    closed = read(PRIOR/'COMPLETION.json')
    require(closed['status'] == 'COMPLETED_INDEPENDENTLY_VERIFIED'
            and closed['protected_files'] == 21444 and closed['observers_modified'] is False
            and closed['plan_sha256'] == PRIOR_PLAN_SHA
            and closed['result_completion_sha256'] == PRIOR_RESULTS_SHA,
            'Prior198 closure incomplete')
    pin(PRIOR/'PLAN.json', PRIOR_PLAN_SHA)
    pin(PRIOR/'results/COMPLETION.json', PRIOR_RESULTS_SHA)
    prior_result = read(PRIOR/'results/COMPLETION.json')
    require(prior_result['status'] == 'COMPLETE' and prior_result['plan_sha256'] == closed['plan_sha256']
            and prior_result['paths'] == 9 and prior_result['slices'] == 45,
            'Prior198 result closure incomplete')
    central = read(O/'monthly_target70_research/CONTINUATION_STATUS.json')
    require(len(central['completed']) == 196 and sum(x['trials'] for x in central['completed']) == 1105
            and central['current'] == 'monthly_event_risk_payoff198' and central['registered_queue'] == [],
            'Source review must precede199 registration')
    reference = central['completed'][-1]['completion']
    require(Path(reference['path']).resolve() == (PRIOR/'COMPLETION.json').resolve()
            and reference['sha256'] == PRIOR_COMPLETION_SHA, 'Central prior198 receipt differs')
    local_plan = read(O/'monthly_local_linear70/PLAN.json')
    raw = local_plan['raw_source_hashes']
    require(raw == previous['raw_source_hashes'] and len(raw) == 64, 'Original64 raw-source closure differs')
    for p, h in raw.items(): pin(p, h)
    meta_path = O/'monthly_local_linear70/META.json'
    calendar_path = O/'monthly_paired_target143/CALENDAR.json'
    pin(meta_path); pin(calendar_path)
    meta, record = read(meta_path), read(calendar_path)
    dates, names_by_event, calendar = meta['dates'], meta['symbols'], record['sessions']
    require(calendar == sorted(set(calendar)), 'Invalid official calendar')
    require(record['provenance'] == read(O/'monthly_paired_target143/PLAN.json')['calendar'], 'Calendar provenance differs')
    for p, h in record['provenance']['source_hashes'].items(): pin(p, h)
    require(len(dates) == len(names_by_event) == 122 and dates == sorted(set(dates)), 'Original event axis differs')
    require(len(meta['etfs']) == len(set(meta['etfs'])) == 17, 'Original ETF identities differ')
    for i, (day, names) in enumerate(zip(dates, names_by_event)):
        require(len(names) == len(set(names)) == 49 and set(meta['etfs']).issubset(names), 'Original candidate identity differs')
        t = calendar.index(day)
        require((i == 0 and day == '2016-08-17') or calendar[t+1][:7] != day[:7], 'Noninitial query is not monthend')
    names = sorted(set(s for row in names_by_event for s in row))
    loader = O/'monthly_stability_monitor/inputs.py'
    require(str(loader) in raw, 'Original loader not pinned')
    spec = importlib.util.spec_from_file_location('nested199_frozen_source_loader', loader)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    panel, info = module.load_panel(meta['etfs'])
    used = dict(info['source_hashes']); used[str(loader)] = sha(loader)
    require(used == raw, 'Current loader used a different source closure')
    adj = panel['adj']
    require(adj.index.is_unique and adj.index.is_monotonic_increasing and adj.columns.is_unique,
            'Invalid original adjusted-close axes')
    require(set(names).issubset(adj.columns), 'Missing original candidate column')
    del panel

    manifest_path = O/'index_union/alias_resolved_inputs/MANIFEST.json'
    manifest = read(manifest_path)
    adj_field = next(x for x in manifest['fields'] if x['field'] == 'adj')
    pin(O/'index_union'/adj_field['file'], adj_field['sha256'])
    # Reuse previously byte-verified input lineage; describe this field accurately.
    lineage = json.loads(json.dumps(previous['source_lineage']))
    require(set(lineage) == set(names), 'Original candidate source identities changed')
    for chain in lineage.values():
        for item in chain:
            operation = item['operation']
            if operation == 'base_alias_panel':
                item.update(path=str(O/'index_union'/adj_field['file']), sha256=adj_field['sha256'],
                    field='adj_close', basis='Frozen signal-file adjusted close; provider total-return convention, not execution raw close')
            elif operation == 'replace_column_then_mask_from_2021_01_01':
                item.pop('not_used', None)
                item.update(field='adjClose', basis='Tiingo adjusted close; dates from2021-01-01 explicitly missing')
            elif operation == 'leading_adj_gap_before_first_overlap':
                item.update(field='Adj Close', basis='Leading gap multiplied by original(existing adjusted close / extension adjusted close) at first overlap; unchanged fixed original splice')
            else:
                item.update(field='Adj Close', basis='Stored CSV adjusted close reindexed onto original panel dates')

    by_query, invalid = [], []
    totals = dict(observations=0, finite_positive=0, zero=0, negative=0, nan=0, infinite=0)
    by_symbol = {s: dict(query_cells=0, source_valid_cells=0, source_invalid_cells=0) for s in names}
    for day, row in zip(dates, names_by_event):
        support = window(calendar, day)
        missing = [d for d in support if pd.Timestamp(d) not in adj.index]
        view = adj.reindex(index=pd.to_datetime(support), columns=row)
        valid = 0
        for s in row:
            c = quality_counts(view[s].to_numpy())
            for k, v in c.items(): totals[k] += v
            ok = not missing and c['finite_positive'] == 253
            valid += int(ok)
            by_symbol[s]['query_cells'] += 1
            by_symbol[s]['source_valid_cells' if ok else 'source_invalid_cells'] += 1
            if not ok:
                a = view[s].to_numpy(dtype=float)
                bad = np.flatnonzero(~(np.isfinite(a) & (a > 0)))
                invalid.append(dict(date=day, symbol=s, counts=c, missing_official_sessions=missing,
                    first_invalid_date=support[int(bad[0])] if len(bad) else None,
                    last_invalid_date=support[int(bad[-1])] if len(bad) else None))
        by_query.append(dict(date=day, start=support[0], end=support[-1], source_valid_n=valid,
            source_invalid_n=49-valid, fewer_than_two_source_valid=valid<2, missing_official_sessions=missing))
    count = 122*49
    require(totals['observations'] == count*253 and sum(totals[k] for k in
        ('finite_positive','zero','negative','nan','infinite')) == totals['observations'], 'Source coverage accounting')
    for s in names:
        a = adj[s].to_numpy(dtype=float); positive = np.flatnonzero(np.isfinite(a) & (a > 0))
        by_symbol[s].update(first_positive_date=str(adj.index[positive[0]].date()) if len(positive) else None,
            last_positive_date=str(adj.index[positive[-1]].date()) if len(positive) else None)
    for n in range(123):
        require(n == 0 or by_query[n-1]['end'] <= dates[n-1], 'Window extends after query')

    result = dict(schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(),
        passed=True, conditional_research_sources_ready=True, source_only=True,
        strict_pit_vintage=False, strict_pit_membership=False, strategy_admitted=False,
        context_sha256=CONTEXT_SHA, review_code_sha256=sha(Path(__file__)),
        raw_source_hashes=raw, source_sha256=pins,
        actual_features_computed=False, actual_orders_computed=False, actual_returns_computed=False,
        actual_block_returns_computed=False, actual_pair_comparisons_computed=False,
        actual_negative_anchors_computed=False, actual_recovery_comparisons_computed=False,
        actual_durability_comparisons_computed=False,
        actual_selected_anchors_computed=False, actual_pre_event_volatility_computed=False,
        actual_payoffs_computed=False, actual_signal_returns_computed=False,
        actual_horizon_gains_computed=False, actual_horizon_ranks_computed=False,
        actual_min_mean_aggregation_computed=False, actual_deltas_computed=False,
        actual_ranks_computed=False, actual_source_coverage_computed=True,
        original_sources_modified=False, observers_accessed=False,
        observed_central_state=dict(completed_rounds=196,cumulative_trials=1105,
            current=central['current'],registered_queue=[],role='Observation only; not a reusable central-file pin.'),
        predecessor_closure=dict(directory=PRIOR.name, completion_sha256=PRIOR_COMPLETION_SHA,
            plan_sha256=closed['plan_sha256'], result_completion_sha256=closed['result_completion_sha256'],
            protected_files=21444, observers_modified=False),
        integration_source_verification=dict(passed=True, hash_bound_dependencies=integration_pins,
            dependency_count=6, context_evidence_pins_verified=len(context['evidence_sha256']),
            method='Read frozen source text and verify context-pinned bytes only; no source strategy imports or numerical execution.',
            facts=[
                'Frozen core adapter supports prefix-limited n/n1/nwidth weights, copies source arrays, adds the same core delta to baseline/core/pre-score copies and supplies original support inputs to one original selector.',
                'Frozen quality source sets cw .50 for calm and .52 otherwise; calm weighted core uses original252_raw. This audit does not reconstruct actual core/state arrays.',
                'Local67 config4 modified_pre_score includes its original local model correction before the adapter; native incumbent choices remain a separate exact-parity obligation for the registered host.',
                'The old core adapter treats nonfinite targets as no-change; the new host must explicitly reject infinity to satisfy the199 contract. This source review does not certify that future numerical gate.'
            ], numerical_integration_verified=False),
        raw_source_parity=dict(passed=True, exact_source_map_equal=True, count=64,
            current_loader_sha256=sha(loader), reused_review_sha256=PRIOR_SHA,
            independent_numeric_price_reconstruction_performed=False,
            method='Execute exact pinned original load_panel; verify all current bytes and original source map. Do not invoke original feature builder.'),
        axes=dict(monthly_queries=122,symbols_per_query=49,checked_cells=count,unique_candidate_symbols=len(names),
            etf_count=17,first_query=dates[0],last_query=dates[-1],inherited_outcome_period=context['evaluation']['period'],
            adjusted_panel_rows=len(adj),adjusted_panel_columns=len(adj.columns),
            adjusted_panel_first=str(adj.index[0].date()),adjusted_panel_last=str(adj.index[-1].date()),
            loader_data_as_of=info['data_as_of'],all_noninitial_queries_monthend=True,
            metadata_prefix_boundaries_checked=123,prefix_check_scope='Only temporal boundaries0..122; no feature or order reruns.'),
        source_valid_cells=count-len(invalid),source_invalid_cells=len(invalid),
        coverage=dict(window_sessions=253,earliest_offset=294,latest_offset=42,
            earliest_window_start=by_query[0]['start'],latest_window_end=by_query[-1]['end'],
            complete_positive_windows=count-len(invalid),incomplete_windows=len(invalid),
            fewer_than_two_source_valid_rows=sum(r['fewer_than_two_source_valid'] for r in by_query),
            observation_counts=totals,repeated_window_observations=True,
            by_query=by_query,unavailable_cells=invalid,by_symbol=by_symbol),
        source_lineage=lineage,alias_rules=previous['alias_rules'],
        normalization_facts=[
            'The feature input is the original stored adjusted-close panel, not raw Close or normalized execution prices. Yahoo source metadata declares split/dividend-adjusted close; Tiingo adjClose is kept separately from raw OHLCV.',
            'Base signal-file adjusted close is retained after the original bar-valid and quarantine mask. Neither raw-share reconstruction nor split-only volume normalization changes the stored adjusted-close column.',
            'Reviewed aliases copy one provider history into a research security ID. Provider choice uses retrospective history coverage, and alias checks do not establish first-public vintages. META maps to research_equity_FB; BRK-B maps to BRK.B; PTR direct override uses PTRCY.',
            'ETF manifest overrides occur in fixed original order, followed by added-security archives. Early extension fills only leading adjusted-price gaps before first overlap, multiplying that segment by the ratio at the first overlap. No interior/later fill is added.',
            'The final CHL/PTR override uses Tiingo adjClose and masks dates from2021-01-01. The original loader does not create sessions or infer membership between monthend observations.',
            'The original price normalization/splice is frozen and reused unchanged. This audit counts all253 finite-positive source prices; it does not compute any horizon gain, horizon rank, min/mean aggregate, delta, order, return or repaired source value.'
        ],
        limitations=[
            'Conditional research readiness is not strict source admission: original alias bundle remains diagnostic_not_admitted, with historicalTop32/theme17 proxies and retrospective vendor/action/class conventions.',
            'Stored adjusted closes can incorporate subsequent vendor revisions and corporate-action anchors; neither the historical first-public vintage nor complete split/dividend/ADR action reconciliation is certified.',
            'A positive common scaling mathematically preserves simple returns; this does not certify stored binary rounding, exactties or consistency across inherited extension splices/vendor/action revisions. No actual gains or ranks were computed to assert such consistency.',
            'Source-validity is not final feature verification or original trading eligibility. Valid positive prices could still overflow in the registered simple-ratio arithmetic; the contract requires a hard numerical failure rather than a silent missing mask. The eligible mask is not inspected or changed; missing histories remain missing. n<2 is source cardinality only, not a new filter; the199 contract retains singleton ranks.',
            'All comparisons, ranks, deltas, original27/order parity and outcomes still require registered production plus independent verification. Prior source pins do not replace those numerical checks.',
        ],synthetic_source_checks_passed=checks)
    require(result['source_valid_cells'] > 0, 'No usable source support')
    for p, h in pins.items(): require(sha(p) == h, 'Source changed during review: '+p)
    with OUT.open('x') as stream:
        json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(passed=True,path=str(OUT),sha256=sha(OUT),review_code_sha256=sha(Path(__file__)),
        source_pins=len(pins),source_valid_cells=count-len(invalid),source_invalid_cells=len(invalid),
        fewer_than_two_rows=result['coverage']['fewer_than_two_source_valid_rows'],
        invalid_symbols={s:r['source_invalid_cells'] for s,r in by_symbol.items() if r['source_invalid_cells']},
        observation_counts=totals,synthetic_checks=checks)))


if __name__ == '__main__':
    main()
