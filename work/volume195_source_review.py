"""Offline source/availability audit only; never construct a strategy feature.

Run with ./run_local.sh work/volume195_source_review.py.  This reads the frozen
original panel loader, verifies its exact 64-input closure, and counts invalid
stored-volume observations in the prospective 273-session windows.  It does not
import a strategy, compute medians/events/ranks, or load any strategy NPZ.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'work/volume195_SOURCE_REVIEW.json'
CONTEXT = ROOT / 'work/monthly_volume_cluster195_context.json'
CONTEXT_SHA = '1a6e06499d35667ae435887f3d2d2e56fce4ea4194531032b2ab4969b046b0a0'
O = ROOT / 'outputs'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(ok, message):
    if not ok:
        raise ValueError(message)


def volume_counts(values):
    """Source-quality counts only; deliberately no rolling/event operations."""
    a = np.asarray(values, dtype=float)
    finite = np.isfinite(a)
    return dict(observations=int(a.size), finite_positive=int((finite & (a > 0)).sum()),
                zero=int((finite & (a == 0)).sum()), negative=int((finite & (a < 0)).sum()),
                nan=int(np.isnan(a).sum()), infinite=int(np.isinf(a).sum()))


def window(calendar, day):
    i = calendar.index(day)
    require(i >= 314, 'Insufficient official calendar before query')
    dates = calendar[i - 314:i - 41]
    require(len(dates) == 273 and dates[-1] == calendar[i - 42], 'Window alignment')
    return dates


def self_checks():
    c = volume_counts([1, 0, -1, np.nan, np.inf, -np.inf])
    require(c == dict(observations=6, finite_positive=1, zero=1, negative=1,
                      nan=1, infinite=2), 'Synthetic quality classification')
    cal = [str(i) for i in range(400)]
    w = window(cal, '350')
    require(w[0] == '36' and w[-1] == '308' and len(w) == 273, 'Synthetic offsets')
    require(volume_counts([1] * 273)['finite_positive'] == 273, 'Positive support')
    try:
        window(cal, '313')
    except ValueError:
        pass
    else:
        raise ValueError('Short calendar must fail')
    return 4


def main():
    require(not OUT.exists(), 'Write-once source review already exists')
    synthetic_checks = self_checks()
    require(sha(CONTEXT) == CONTEXT_SHA, 'Prospective context changed')
    context = read(CONTEXT)
    require(context['formation']['volume_count'] == 273 and
            context['formation']['earliest_offset'] == 314 and
            context['formation']['latest_offset'] == 42, 'Prospective window changed')
    state_path = O / 'monthly_target70_research/CONTINUATION_STATUS.json'
    state = read(state_path)
    require(len(state['completed']) == 192 and
            sum(x['trials'] for x in state['completed']) == 1093 and
            state['current'] == 'monthly_trade_wc194' and
            state['registered_queue'] == [], 'Source review must precede 195 registration')

    pins = {}
    def pin(path, expected=None):
        path = Path(path).resolve()
        require(path.is_relative_to(ROOT), 'Nonlocal source path')
        digest = sha(path)
        require(expected is None or digest == expected, 'Changed source: ' + str(path))
        require(str(path) not in pins or pins[str(path)] == digest, 'Pin conflict')
        pins[str(path)] = digest
        return digest

    pin(Path(__file__))
    pin(CONTEXT, CONTEXT_SHA)
    for relative, digest in context['evidence_sha256'].items():
        pin(ROOT / relative, digest)
    plan_paths = [O / x / 'PLAN.json' for x in (
        'monthly_local_linear70', 'monthly_paired_target143',
        'monthly_activity164', 'monthly_followthrough166')]
    plans = []
    for p in plan_paths:
        pin(p)
        plans.append(read(p))
    raw = plans[0]['raw_source_hashes']
    require(len(raw) == 64, 'Original raw closure must contain 64 sources')
    require(all(p['raw_source_hashes'] == raw for p in plans[1:]), 'Prior raw closures differ')
    for path, digest in raw.items():
        pin(path, digest)

    meta_path = O / 'monthly_local_linear70/META.json'
    calendar_path = O / 'monthly_paired_target143/CALENDAR.json'
    pin(meta_path)
    pin(calendar_path)
    meta = read(meta_path)
    calendar_record = read(calendar_path)
    calendar = calendar_record['sessions']
    require(calendar == sorted(set(calendar)), 'Calendar must be unique and ordered')
    require(calendar_record['provenance'] == plans[1]['calendar'], 'Calendar provenance changed')
    for p, h in calendar_record['provenance']['source_hashes'].items():
        pin(p, h)
    dates, symbols = meta['dates'], meta['symbols']
    require(len(dates) == len(symbols) == 122 and dates == sorted(set(dates)), 'Original event axis')
    require(len(meta['etfs']) == 17, 'Original ETF universe')
    for i, (day, names) in enumerate(zip(dates, symbols)):
        require(len(names) == len(set(names)) == 49, 'Original candidate width')
        require(set(meta['etfs']).issubset(names), 'Original ETF membership')
        j = calendar.index(day)
        require((i == 0 and day == '2016-08-17') or
                calendar[j + 1][:7] != day[:7], 'Noninitial query must be monthend')
    names = sorted(set(s for row in symbols for s in row))

    loader = O / 'monthly_stability_monitor/inputs.py'
    require(str(loader) in raw, 'Original loader must be pinned')
    spec = importlib.util.spec_from_file_location('volume195_original_input_loader', loader)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    panel, info = module.load_panel(meta['etfs'])
    loaded_hashes = dict(info['source_hashes'])
    loaded_hashes[str(loader)] = sha(loader)
    require(loaded_hashes == raw, 'Current frozen loader did not use exact original 64 inputs')
    volume = panel['volume']
    require(volume.index.is_unique and volume.index.is_monotonic_increasing and
            volume.columns.is_unique, 'Volume axes invalid')
    require(volume.index.equals(panel['adj'].index) and
            volume.columns.equals(panel['adj'].columns), 'Volume and original price axes differ')
    require(set(names).issubset(volume.columns), 'Missing original candidate volume column')
    # Only the volume panel is used below.  No price-derived return is calculated.
    del panel

    index_root = O / 'index_union'
    alias_path = index_root / 'alias_resolved_inputs/MANIFEST.json'
    input_path = index_root / 'input_diagnostics/MANIFEST.json'
    for path in (alias_path, input_path):
        pin(path)
    aliases, integrated = read(alias_path), read(input_path)
    for row in aliases['sources']:
        pin(index_root / row['file'], row['sha256'])
    volume_manifest = next(r for r in aliases['fields'] if r['field'] == 'volume')
    pin(index_root / volume_manifest['file'], volume_manifest['sha256'])
    for relative in (
        'outputs/monthly_state_rank_research/state_rank.py',
        'outputs/monthly_architecture_restart/study.py',
        'outputs/monthly_local_linear70/study.py',
        'outputs/index_union/assemble_alias_resolved_inputs.py',
        'outputs/index_union/assemble_input_diagnostics.py',
        'outputs/index_union/consolidate_security_aliases.py',
        'outputs/index_union/consolidate_issuer_classes.py',
        'outputs/index_union/normalize_new_prices.py',
        'outputs/index_union/price_units.py',
        'outputs/index_union/build_execution_candidates.py',
        'outputs/index_union/build_raw_source_candidates.py',
        'work/followthrough166_source_probe.py',
        'work/sector_diffusion167_source_probe.py'):
        pin(ROOT / relative)

    alias_map = {r['security_id']: r for r in aliases['alias_decisions']}
    base_sources = {r['symbol']: r for r in integrated['price_sources']}
    lineage = {}
    for s in names:
        provider_symbol = alias_map.get(s, {}).get('price_provider_symbol', s)
        row = base_sources.get(provider_symbol)
        chain = []
        if row is not None:
            for key, hashkey in (('input_file', 'input_sha256'), ('raw_file', 'raw_sha256')):
                pin(index_root / row[key], row[hashkey])
            chain.append(dict(operation='base_alias_panel',
                              path=str(index_root / volume_manifest['file']),
                              sha256=volume_manifest['sha256'], provider_symbol=provider_symbol,
                              upstream_source=row,
                              basis='frozen signal-file volume; not execution raw-file volume'))
        lineage[s] = chain
    # This records the loader's ordered source rules, without reimplementing it.
    for relative in ('four_factor_rebound/ETF_SOURCES.json',
                     'monthly_reconstruction_v2/ETF_MANIFEST.json',
                     'monthly_reconstruction_v2/foreign/ETF_MANIFEST.json'):
        for row in read(O / relative):
            if row['symbol'] in lineage:
                lineage[row['symbol']].append(dict(operation='overwrite_full_column',
                    manifest=str(O / relative), path=row['file'], sha256=row['sha256'],
                    field='Volume', basis='stored CSV Volume, no loader share-unit conversion'))
    for relative in ('monthly_published44_reconstruction/prices',
                     'monthly_top35_extension/prices', 'monthly_2016_extension/prices'):
        for f in sorted((O / relative).glob('*.csv')):
            if f.stem not in lineage:
                continue
            receipt = read(f.with_suffix('.source.json'))
            lineage[f.stem].append(dict(operation=('leading_adj_gap_before_first_overlap' if
                relative.startswith('monthly_2016') else 'add_security'), path=str(f),
                sha256=raw[str(f)], source_receipt=receipt, field='Volume',
                basis='stored Volume unchanged; adjusted-price splice ratio never applied to volume'))
    for symbol, cache in (('CHL', 'CHL'), ('PTR', 'PTRCY')):
        if symbol in lineage:
            p = O / 'dynamic_top100/additional_sources/tiingo' / (cache + '_prices.json')
            lineage[symbol].append(dict(operation='replace_column_then_mask_from_2021_01_01',
                path=str(p), sha256=raw[str(p)], provider_symbol=cache, field='volume',
                not_used='adjVolume', basis='Tiingo raw volume; no split-only rescaling in this override'))
    require(all(lineage.values()), 'Missing lineage for original candidate')

    coverage_rows = []
    unavailable_cells = []
    totals = dict(observations=0, finite_positive=0, zero=0, negative=0, nan=0, infinite=0)
    by_symbol = {s: dict(query_cells=0, complete_positive_windows=0, incomplete_windows=0)
                 for s in names}
    for day, row_names in zip(dates, symbols):
        support = window(calendar, day)
        missing_sessions = [x for x in support if pd.Timestamp(x) not in volume.index]
        frame = volume.reindex(index=pd.to_datetime(support), columns=row_names)
        row_good = 0
        for s in row_names:
            c = volume_counts(frame[s].to_numpy())
            for key, value in c.items():
                totals[key] += value
            good = not missing_sessions and c['finite_positive'] == 273
            by_symbol[s]['query_cells'] += 1
            by_symbol[s]['complete_positive_windows' if good else 'incomplete_windows'] += 1
            row_good += int(good)
            if not good:
                a = frame[s].to_numpy(dtype=float)
                bad = ~(np.isfinite(a) & (a > 0))
                unavailable_cells.append(dict(date=day, symbol=s, counts=c,
                    first_invalid_date=support[int(np.flatnonzero(bad)[0])] if bad.any() else None,
                    last_invalid_date=support[int(np.flatnonzero(bad)[-1])] if bad.any() else None,
                    missing_official_sessions=missing_sessions))
        coverage_rows.append(dict(date=day, start=support[0], end=support[-1],
            observations_per_symbol=273, complete_positive_windows=row_good,
            incomplete_windows=49-row_good, missing_official_sessions=missing_sessions))
    checked = 122 * 49
    require(totals['observations'] == checked * 273, 'Coverage accounting mismatch')
    require(sum(totals[k] for k in ('finite_positive', 'zero', 'negative', 'nan', 'infinite')) ==
            totals['observations'], 'Coverage categories must partition observations')
    for s in names:
        a = volume[s].to_numpy(dtype=float)
        positive = np.flatnonzero(np.isfinite(a) & (a > 0))
        by_symbol[s].update(first_positive_date=str(volume.index[positive[0]].date()) if len(positive) else None,
            last_positive_date=str(volume.index[positive[-1]].date()) if len(positive) else None)
    # These are metadata/time-slice checks, not 123 feature or order reruns.
    for n in range(123):
        if not n:
            require(len(volume.iloc[:0]) == 0, 'Empty source prefix')
        else:
            require(pd.Timestamp(coverage_rows[n-1]['end']) <= pd.Timestamp(dates[n-1]),
                    'Source support later than query')

    result = dict(schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(),
        passed=True, conditional_research_sources_ready=True,
        interpretation='Source integrity and availability only; no strategy or strict-PIT admission.',
        strict_pit_vintage=False, strict_pit_membership=False, strategy_admitted=False,
        context_sha256=CONTEXT_SHA, review_code_sha256=sha(Path(__file__)),
        raw_source_hashes=raw, source_sha256=pins,
        source_only=True, actual_source_coverage_computed=True,
        actual_features_computed=False, actual_orders_computed=False, actual_returns_computed=False,
        medians_computed=0, volume_event_indicators_computed=0, covariance_computed=0,
        event_frequencies_computed=0, ranks_computed=0, new_model_fits=0,
        raw_sources_changed=False, observers_accessed=False,
        raw_source_parity=dict(passed=True, current_loader_hash=sha(loader),
            exact_source_map_equal=True, count=64,
            identical_prior_raw_maps=[str(p) for p in plan_paths],
            method='Execute exact frozen load_panel and compare every path/hash with Local67; no strategy load.',
            independent_numeric_volume_reconstruction_performed=False),
        observed_central_state=dict(completed_rounds=192, cumulative_trials=1093,
            current=state['current'], registered_queue=[],
            role='Observation only; central file is not a reusable source pin.'),
        axes=dict(monthly_queries=122, symbols_per_query=49, checked_cells=checked,
            unique_candidate_symbols=len(names), etf_count=17, first_query=dates[0], last_query=dates[-1],
            inherited_outcome_period=context['evaluation']['period'],
            volume_panel_rows=len(volume), volume_panel_columns=len(volume.columns),
            volume_panel_first=str(volume.index[0].date()), volume_panel_last=str(volume.index[-1].date()),
            loader_data_as_of=info['data_as_of'], official_calendar_first=calendar[0],
            official_calendar_last=calendar[-1], all_noninitial_queries_monthend=True,
            metadata_prefix_boundaries_checked=123,
            prefix_check_scope='0..122 temporal boundaries only; no feature/order prefixes computed'),
        availability=dict(window_sessions=273, earliest_offset=314, latest_offset=42,
            earliest_window_start=coverage_rows[0]['start'], latest_window_end=coverage_rows[-1]['end'],
            complete_positive_windows=checked-len(unavailable_cells),
            incomplete_windows=len(unavailable_cells), observation_counts=totals,
            repeated_window_observations=True, by_query=coverage_rows,
            unavailable_cells=unavailable_cells, by_symbol=by_symbol),
        source_lineage=lineage,
        source_label_correction=dict(
            original_context_preserved=True,
            context_word='native',
            precise_meaning='Original stored volume: heterogeneous normalized signal-volume and direct raw-volume streams, not a unified raw-share series.',
            numerical_formula_or_input_changes=False,
            within_window_discontinuity_test_performed=False,
            caveat='This is a provenance/coverage audit, not a new corporate-action discontinuity detector. No newly inferred discontinuity is silently masked.'),
        alias_rules=dict(pool_aliases={'META':'research_equity_FB','BRK-B':'BRK.B'},
            direct_vendor_aliases={'PTR':'PTRCY'},
            relevant_base_alias_decisions=[r for r in aliases['alias_decisions'] if r['security_id'] in names],
            rule='One provider history chosen for each reviewed identity; no alias price-provider splice. Retrospective.'),
        normalization_facts=[
            'Alias base volume is signal-file volume after original bar-valid/quarantine mask, not raw execution-file volume.',
            'Yahoo normalization records split-adjusted OHLCV and dividend-adjusted close; it preserves the stored Volume column. This is the archived declared convention, not a new vendor certification.',
            'Documented raw Tiingo/WIKI signal candidates use explicit future split factors through the vendor anchor: signal volume=raw volume*subsequent-split multiplier. Cash-dividend adjustment is not used to infer volume scaling.',
            'Alias consolidation copies all source fields from one provider selected by retrospective close coverage; raw-price overlap tolerance does not independently certify volume overlap.',
            'Original loader overwrites manifests in their fixed order, adds archived securities, and fills only leading adjusted-price gaps from early extensions. Stored volume is not multiplied by the adjusted-price splice ratio.',
            'Final CHL/PTR overrides use Tiingo volume, not adjVolume, and explicitly mask dates from 2021-01-01. These overrides are a different share-unit convention from the split-adjusted signal base.',
            'No live re-fetch, forward fill, zero imputation, new outlier clipping or new split/ADR normalization is performed. Zero, negative and nonfinite source observations fail the prospective complete-positive support rule.'
        ],
        source_facts_vs_assumptions=dict(
            facts='Current bytes, loader transformations, official dates, archived aliases and availability counts are verified.',
            assumptions='Use native stored share-volume conventions as conditional research data. They are not turnover, dollars, buyer-initiated flow or a universally raw-share series.',
            not_verified='First-public historical vintages, full action completeness, every ADR/share-ratio transition, universal cross-provider unit reconciliation, or all-US point-in-time membership.'),
        limitations=[
            'Original alias bundle remains alias_resolution_diagnostic_not_admitted; historical Top32/theme17 is a declared research proxy. Reuse does not promote strict source admission.',
            'Retrospective split anchoring and aliases are inherited. A split adjustment strictly after a whole 273-session window is a constant factor throughout that window and cannot change within-window median comparisons. This does not certify consistent units across a provider transition or corporate action within the window; time-varying conventions and missing actions can change the event.',
            'Complete-positive coverage is source availability, not trading eligibility. No eligibility mask was read or changed; partial histories remain unavailable and are never synthesized.',
            'The 5978 query cells reuse calendar observations; counts are not counts of independent samples.',
            'Prior148 already used binary-event lag-one covariance versus frequency. This prospective input is unsigned own-volume; missing older volume/liquidity originals leave exhaustive novelty UNKNOWN.',
            'This review does not calculate above-median events, feature strengths, identification, holdings changes or performance. A registered implementation and independent checks remain necessary.'
        ],
        synthetic_source_checks_passed=synthetic_checks)
    require(result['availability']['complete_positive_windows'] > 0, 'No usable source support')
    # Every source/metadata byte remains unchanged through this read-only audit.
    for path, digest in pins.items():
        require(sha(path) == digest, 'Source changed during audit: ' + path)
    with OUT.open('x') as f:
        json.dump(result, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')
    print(json.dumps(dict(passed=True, path=str(OUT), sha256=sha(OUT),
        review_code_sha256=sha(Path(__file__)), source_pins=len(pins),
        checked_cells=checked, complete_positive_windows=checked-len(unavailable_cells),
        incomplete_windows=len(unavailable_cells), observation_counts=totals,
        incomplete_symbols={s:v['incomplete_windows'] for s,v in by_symbol.items()
                            if v['incomplete_windows']}, synthetic_checks=synthetic_checks)))


if __name__ == '__main__':
    main()
