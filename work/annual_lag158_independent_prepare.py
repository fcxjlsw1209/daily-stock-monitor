"""Independent annual-lag preparation verification; no producer imports.

The pure oracles accept synthetic objects. Real file access is confined to an
explicit registered-PLAN audit. Dates use ordinal subtraction and average ranks
use Fraction arithmetic; neither calls the new model or runner.
"""
import argparse
from collections import Counter
from datetime import date, datetime, timezone
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
import monthly_cash_quality146_prepare_review as legacy

ROOT = Path(__file__).resolve().parents[1]
OUT = 'outputs/monthly_annual_lag158'
NAMES = ('incumbent_local67', 'negative_current_annual_lag', 'negative_annual_lag_change')
RAW = ('negative_current_lag', 'negative_lag_change')
RANKS = ('current_lag_rank', 'lag_change_rank')
EXTRA = RAW + RANKS + ('common_available',)
ISSUERS = {'KO': '0000021344', 'WMT': '0000104169'}
INPUTS = ('proposal', 'meta', 'policy', 'original_orders', 'prices', 'protected_manifest',
          'design', 'calendar', 'source_overlay', 'source_review', 'source_prior')
CACHE_FILES = ('DATA.npz', 'SCORES.npz', 'META.json', 'PRETRAIN_ORDERS.json',
               'PRICE_PARITY.json', 'FEATURE_AUDIT.json', 'CALENDAR.json', 'RANK_AUDIT.json')
CHECKS = ('original27_arrays_byte_identical', 'raw_prices_independently_verified',
          'calendar_overlay_independently_verified', 'annual_lags_independently_reconstructed',
          'shared_source_eligible_mask_and_centered_ranks_verified',
          'all_three_score_matrices_and_ordered_choices_independently_verified',
          'all_signal_prefixes_verified', 'initial_partial_and_missing_exact_fallback_verified',
          'one_episode_and_top2_lock_verified', 'full_5978_source_rows_and_cutoffs_verified',
          'source_contradictions_hard_failed')
FIXED = {
    'design': ('work/after157_lag_pilot_design.json', '215ef056f8ac8fb3ac19e52acc49bd983802bb13cddec0b103826c85640055d5'),
    'meta': ('outputs/monthly_local_linear70/META.json', '3b46968f486e88fa170ac01f120dd5ef74ebd5a56243c6991c5a4fea506076f4'),
    'calendar': ('outputs/monthly_paired_target143/CALENDAR.json', '8ee4da69fba84db1edee37a7e0f5af5d3ab16c6257e5d37cddbd531f01404622'),
    'source_overlay': ('outputs/after157_walmart_context_admission_v1/ADMISSION.json', '9e90889a1fb4f057438b6ac16a9e234f97cc9ede08c6cb28a6fe190389f3f9d6'),
    'source_review': ('work/after157_walmart_context_independent.json', '16a666e2d72426a9baa92e693fc003d8c107f9908032762d61232c151dc18814'),
    'source_prior': ('work/after157_annual_pair_source_feasibility.json', '4fdeaad7de09625991a2b3603d33799bf1c62dd0b99f560b900ac8928abc9242'),
}
SOURCE_CHECKS = ('all_overlay_input_hashes_verified', 'all_twelve_original_chains_verified',
    'reviewed_originals_match_frozen_metadata_and_raw_source_rows', 'exact_publisher_links_verified',
    'complete_eleven_wmt_pairs_verified', 'ko_exact_pair_replay_and_inheritance_verified',
    'all_5978_exact_rows_independently_reconstructed', 'all_original_fields_and_flags_preserved',
    'stock3904_etf2074_axes_preserved', 'only_specific_context_blocker_resolved',
    'other_timing_identity_amendment_selection_gates_preserved', 'wmt_may2018_still_blocked',
    'segment_summaries_independently_recomputed', 'fy2024_discrepancy_retained_and_citations_corrected',
    'original_tier3_zero_preserved', 'no_producer_imports', 'input_hashes_unchanged_after_audit')
MISSING = 'ACCESSION_SPECIFIC_NORMAL_CONSECUTIVE_ANNUAL_CONTEXT_NOT_ESTABLISHED'
INITIAL = '2016-08-17'
HISTORICAL_STATE = 'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
HISTORICAL_SHA = 'f0f6571a3f4de4be06aeb957fe03fd13eed11f86b288abafc91094db38ac32b7'
HISTORICAL_ARCHIVE = 'outputs/monthly_target70_research/PRE157_' + HISTORICAL_SHA + '.json'
old, require = legacy.old, legacy.require


def independent_imports():
    forbidden = ('monthly_annual_lag158_model', 'monthly_annual_lag158_runner',
                 'annual_lag158_lifecycle', 'monthly_annual_lag158_lifecycle')
    require(not any(name in sys.modules for name in forbidden), 'No new producer/lifecycle imports')


def source_files(root=None):
    root = ROOT if root is None else Path(root).resolve()
    return sorted(set(legacy.source_files(root) + [root / 'work' / name for name in
        ('annual_lag158_independent_prepare.py', 'test_annual_lag158_independent_prepare.py')]))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def day(value):
    require(type(value) is str and re.fullmatch(r'[0-9]{4}-[0-9]{2}-[0-9]{2}', value), 'Strict ISO date')
    result = date.fromisoformat(value)
    require(result.isoformat() == value, 'Canonical date')
    return result


def ns(value):
    require(type(value) is str and re.fullmatch(
        r'[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{1,6})?(?:Z|[+-][0-9]{2}:[0-9]{2})', value),
        'Explicit timezone timestamp')
    stamp = datetime.fromisoformat(value.replace('Z', '+00:00'))
    delta = stamp.astimezone(timezone.utc) - datetime(1970, 1, 1, tzinfo=timezone.utc)
    return (delta.days * 86400 + delta.seconds) * 10**9 + delta.microseconds * 1000


def rank_oracle(values):
    raw = np.asarray(values)
    require(raw.ndim == 1 and raw.dtype.kind in 'fiu' and np.isfinite(raw).all(), 'Finite numeric rank population')
    n = len(raw)
    if n < 2:
        return np.zeros(n, dtype=np.float64)
    order = sorted(range(n), key=lambda i: raw[i])
    result = np.zeros(n, dtype=np.float64)
    start = 0
    while start < n:
        end = start + 1
        while end < n and raw[order[start]] == raw[order[end]]:
            end += 1
        centered = Fraction(start + end - 1, 2 * (n - 1)) - Fraction(1, 2)
        for index in order[start:end]:
            result[index] = float(centered)
        start = end
    return result


def calendar_check(calendar):
    require(type(calendar) is dict and set(calendar) == {'sessions', 'closes'}, 'Explicit calendar shape')
    sessions, closes = calendar['sessions'], calendar['closes']
    require(type(sessions) is list and sessions and type(closes) is dict, 'Explicit sessions/closes')
    require(sessions == sorted(set(sessions)), 'Strict unique session order')
    for session in sessions:
        day(session)
        require(type(closes.get(session)) is int, 'Integer session close')
    return {session: index for index, session in enumerate(sessions)}


def cutoff_check(row, signal, calendar, positions):
    require(signal in positions and positions[signal] >= 42, 'Full t-42 session history')
    cut = calendar['sessions'][positions[signal] - 42]
    close = calendar['closes'][cut]
    require(row.get('information_cutoff_date') == cut and type(row.get('information_cutoff_ns')) is int and
            row['information_cutoff_ns'] == close and ns(row.get('information_cutoff')) == close and
            ns(row.get('original_decision_cutoff')) == calendar['closes'][signal], 'Original t-42/decision cutoff mismatch')
    return cut


def inspected_pair(row, events, calendar, cutoff):
    ids = row.get('selected_event_ids')
    require(type(ids) is list and len(ids) == 2 and all(type(e) is str and e for e in ids) and
            ids[0] != ids[1], 'Exactly two original event identities; no fallback')
    selected = []
    for event_id in ids:
        event = events.get(event_id)
        require(type(event) is dict and event.get('event_id') == event_id, 'Missing selected original')
        require(event.get('cik') == ISSUERS[row['symbol']] and event.get('form') == '10-K', 'Original issuer/form')
        require(type(event.get('accessionNumber')) is str and
                re.fullmatch(r'[0-9]{10}-[0-9]{2}-[0-9]{6}', event['accessionNumber']) and
                type(event.get('primaryDocument')) is str and event['primaryDocument'], 'Original version identity')
        report, filing = day(event.get('reportDate')), day(event.get('filingDate'))
        require(filing >= report, 'Negative annual filing lag')
        require(event['filingDate'] >= calendar['sessions'][0], 'Filing predates calendar')
        # Deliberately scan the calendar instead of using the producer's bisect.
        later = (session for session in calendar['sessions'] if session > event['filingDate'])
        next(later, None)
        available = next(later, None)
        require(available is not None and available <= cutoff, 'Selected original later than t-42; no fallback')
        selected.append((report, event_id, event['accessionNumber'], available, filing))
    selected.sort()
    previous, current = selected
    require(previous[0] < current[0] and previous[2] != current[2], 'Distinct original periods/accessions')
    return previous, current


def selected_pair(row, events, calendar, cutoff):
    previous, current = inspected_pair(row, events, calendar, cutoff)
    prior_lag = previous[4].toordinal() - previous[0].toordinal()
    current_lag = current[4].toordinal() - current[0].toordinal()
    detail = dict(previous_event_id=previous[1], current_event_id=current[1],
                  previous_available_date=previous[3], current_available_date=current[3],
                  previous_lag_days=prior_lag, current_lag_days=current_lag)
    return (-current_lag, prior_lag - current_lag), detail


def features_oracle(meta, rows, event_metadata, source_mask, eligible, calendar, n=None):
    count = 122 if n is None else n
    require(type(count) is int and 0 <= count <= 122, 'Valid prefix 0..122')
    require(len(meta['dates']) == len(meta['symbols']) == 122 and len(rows) == 5978, 'Full original grid')
    require(type(meta['etfs']) in (list, tuple) and len(meta['etfs']) == len(set(meta['etfs'])), 'ETF identity list')
    for mask in (source_mask, eligible):
        require(isinstance(mask, np.ndarray) and mask.dtype == bool and mask.shape == (122, 49), 'Boolean original-grid mask')
    positions = calendar_check(calendar)
    arrays = {key: np.full((count, 49), np.nan) for key in RAW}
    arrays.update({key: np.zeros((count, 49)) for key in RANKS})
    arrays['common_available'] = np.zeros((count, 49), bool)
    cells, monthly = [], []
    for i in range(count):
        signal, names = meta['dates'][i], meta['symbols'][i]
        day(signal)
        require(i == 0 or meta['dates'][i-1] < signal, 'Ordered signal dates')
        require(len(names) == len(set(names)) == 49 and all(type(s) is str and s for s in names), '49 original candidates')
        for j, symbol in enumerate(names):
            overlay, source = rows[i*49+j], bool(source_mask[i, j])
            row = overlay['original_row']
            require(type(row['signal_index']) is int and type(row['candidate_index']) is int and
                    (row['signal_index'], row['candidate_index'], row['symbol'], row['signal_date']) == (i, j, symbol, signal),
                    'Original grid alignment')
            require(type(row['is_etf']) is bool and row['is_etf'] == (symbol in meta['etfs']), 'ETF identity alignment')
            require(type(overlay['conditional_annual_pair_admitted']) is bool and
                    overlay['conditional_annual_pair_admitted'] == source, 'Source mask matches exact overlay')
            for key in ('reasons', 'known_amendment_event_ids'):
                require(type(row[key]) is list and all(type(v) is str and v for v in row[key]), 'Explicit original reason/id list')
            require(type(overlay['remaining_row_reasons']) is list and
                    all(type(v) is str and v for v in overlay['remaining_row_reasons']), 'Explicit remaining reason list')
            cutoff = cutoff_check(row, signal, calendar, positions)
            reasons = []
            if not source:
                reasons.append('SOURCE_NOT_CONDITIONALLY_ADMITTED')
            if overlay.get('body_pair_context_verified') is not True or row.get('tier2') is not True:
                reasons.append('SOURCE_OR_ANNUAL_CONTEXT_NOT_VERIFIED')
            blockers = overlay['remaining_row_reasons'] + [reason for reason in row['reasons'] if not
                (reason == MISSING and overlay.get('body_pair_context_verified') is True)]
            reasons.extend('SOURCE_BLOCKER:' + reason for reason in dict.fromkeys(blockers))
            if row['known_amendment_event_ids']:
                reasons.append('KNOWN_SELECTED_PERIOD_AMENDMENT')
            if overlay.get('old_admission_flags_changed') is not False:
                reasons.append('ORIGINAL_ADMISSION_FLAGS_NOT_PRESERVED')
            if symbol not in ISSUERS or row['is_etf']:
                reasons.append('OUTSIDE_FIXED_KO_WMT_SCOPE')
            elif row.get('cik') != ISSUERS[symbol]:
                reasons.append('ROW_ISSUER_MISMATCH')
            if overlay.get('body_context_source') not in (None, symbol) or (
                    overlay.get('body_pair_context_verified') is True and overlay.get('body_context_source') != symbol):
                reasons.append('BODY_CONTEXT_ISSUER_MISMATCH')
            require(not source or not reasons, 'Contradictory admitted original metadata: ' + ','.join(reasons))
            if source:
                inspected_pair(row, event_metadata, calendar, cutoff)
            if not eligible[i, j]:
                reasons.append('ORIGINAL_INELIGIBLE')
            if signal == INITIAL:
                reasons.append('INITIAL_PARTIAL_EXACT_FALLBACK')
            details = dict(information_cutoff_date=cutoff)
            if not reasons:
                values, selected = selected_pair(row, event_metadata, calendar, cutoff)
                details.update(selected)
                arrays['common_available'][i, j] = True
                for key, value in zip(RAW, values):
                    arrays[key][i, j] = value
            cells.append(dict(signal_index=i, candidate_index=j, symbol=symbol, signal_date=signal,
                source_conditionally_admitted=source, original_eligible=bool(eligible[i, j]),
                available=not reasons, reasons=list(dict.fromkeys(reasons)), **details))
        mask = arrays['common_available'][i]
        for raw, rank in zip(RAW, RANKS):
            arrays[rank][i, mask] = rank_oracle(arrays[raw][i, mask])
        monthly.append(dict(signal_index=i, signal_date=signal, common_candidates=int(mask.sum()),
            current_rank_nonzero=int(np.count_nonzero(arrays[RANKS[0]][i])),
            change_rank_nonzero=int(np.count_nonzero(arrays[RANKS[1]][i])),
            ranks_identical=bool(np.array_equal(arrays[RANKS[0]][i], arrays[RANKS[1]][i]))))
    require(all(np.array_equal(np.isfinite(arrays[key]), arrays['common_available']) for key in RAW), 'Common raw mask')
    return dict(arrays=arrays, cells=cells, monthly=monthly)


def score_oracle(arrays, meta, hands, n=None):
    helper = old.original_helper()
    x = helper.original_prefix(arrays, meta, n)
    shape, count = x['core'].shape, x['n']
    scores, choices, deltas = {}, {}, {}
    for index, name in enumerate(NAMES):
        pre, delta = x['local'].copy(), np.zeros(shape)
        if index:
            hand = np.asarray(hands[RANKS[index-1]])[:count]
            require(hand.shape == shape and hand.dtype.kind == 'f' and np.isfinite(hand).all() and
                    (np.abs(hand) <= .5).all(), 'Finite centered rank bounded at 0.5')
            for i, symbols in enumerate(meta['symbols'][:count]):
                require(all(hand[i, j] == 0 for j, s in enumerate(symbols) if s in meta['etfs']), 'ETF correction zero')
            active = x['eligible'] & (hand != 0)
            delta[active] = .01 * hand[active]
            require((np.abs(delta) <= .005).all(), 'Pre-score addition bounded at 0.005')
            pre[active] += delta[active]
        selected, final, _ = helper.episode(pre, x)
        scores[name], choices[name], deltas[name] = final, selected, delta
    return scores, choices, deltas


def order_oracle(meta, scores, choices):
    return {name: [dict(date=meta['dates'][i], signal_date=meta['dates'][i], end=meta['ends'][i],
        symbols=[meta['symbols'][i][j] for j in selected], scores=[float(scores[name][i, j]) for j in selected])
        for i, selected in enumerate(choices[name])] for name in NAMES}


def verify_original_arrays(original, prepared):
    require(len(original) == 27 and set(prepared) == set(original) | set(EXTRA), 'Original27 plus exact five arrays')
    for key, value in original.items():
        require(value.shape == prepared[key].shape and value.dtype == prepared[key].dtype and
                value.tobytes() == prepared[key].tobytes(), 'Original array changed: ' + key)


def historical_pins(reader, values, registered):
    """Resolve only the explicitly named immutable PRE157 central-state backup."""
    out = {}
    central = reader.path(HISTORICAL_STATE)
    for name, digest in values.items():
        path = reader.path(name)
        if path == central:
            require(digest == HISTORICAL_SHA, 'Only exact historical central alias')
            path = reader.path(HISTORICAL_ARCHIVE)
        require(registered.get(str(path)) == digest, 'Source closure outside registered bytes: ' + str(path))
        require(reader.sha(path) == digest, 'Source closure hash mismatch')
        out[str(path)] = digest
    return out


def source_inputs(reader, plan, pins):
    refs = plan['inputs']
    for key, (name, digest) in FIXED.items():
        require(reader.path(refs[key]['path']) == reader.path(name) and
                refs[key]['sha256'] == reader.sha(name) == digest and pins.get(str(reader.path(name))) == digest,
                'Fixed source/design contract: ' + key)
    meta = reader.json(refs['meta']['path'])
    overlay = reader.json(refs['source_overlay']['path'])
    review = reader.json(refs['source_review']['path'])
    prior = reader.json(refs['source_prior']['path'])
    require(review.get('passed') is True and review.get('source_admission_changed') is False and
            all(review.get('checks', {}).get(key) is True for key in SOURCE_CHECKS), 'Independent source review required')
    closure = historical_pins(reader, overlay['input_hashes'], pins)
    reviewed = historical_pins(reader, review['input_hashes'], pins)
    require(all(reviewed.get(p) == h for p, h in closure.items()) and
            reviewed.get(str(reader.path(refs['source_overlay']['path']))) == refs['source_overlay']['sha256'],
            'Complete independent overlay closure')
    require(overlay['source_admission_changed'] is False and overlay['old_admission_flags_changed'] is False and
            overlay['full_strategy_admission'] is False and overlay['lag_or_feature_values_computed'] is False,
            'Source-only conditional overlay')
    require(len(meta['dates']) == len(meta['symbols']) == 122 and meta['train_months'] == 88 and
            meta['early_months'] == 53 and len(meta['etfs']) == len(set(meta['etfs'])) == 17, 'Original metadata axes')
    rows = overlay['rows']
    require(len(rows) == len(prior['rows']) == 5978, '5978 source rows')
    # Independent base exchange calendar plus the same pinned official closures.
    from exchange_calendars.exchange_calendar_xnys import XNYSExchangeCalendar
    saved = reader.json(refs['calendar']['path'])
    historical_pins(reader, saved['provenance']['source_hashes'], pins)
    manifest = reader.json('outputs/index_union/CALENDAR_EXCEPTIONS.json')
    closed = set()
    for record in manifest['closures']:
        if 'XNYS' in record['markets']:
            path = reader.path('outputs/index_union') / record['source_file']
            require(reader.sha(path) == record['source_sha256'] == pins.get(str(path)), 'Official closure evidence')
            closed.add(record['date'])
    schedule = XNYSExchangeCalendar(start=saved['start'], end=saved['end']).schedule
    sessions = [str(d.date()) for d in schedule.index if str(d.date()) not in closed]
    require(sessions == saved['sessions'], 'Independent full XNYS calendar')
    closes = {str(d.date()): int(c.value) for d, c in zip(schedule.index, schedule['close']) if str(d.date()) not in closed}
    calendar = dict(sessions=sessions, closes=closes)
    positions = calendar_check(calendar)
    source = np.zeros((122, 49), bool)
    events = prior['selected_original_and_amendment_metadata']
    for index, (row, original) in enumerate(zip(rows, prior['rows'])):
        require(canonical(row['original_row']) == canonical(original), 'Frozen original row changed')
        i, j = divmod(index, 49)
        require(type(original['signal_index']) is int and type(original['candidate_index']) is int and
                (original['signal_index'], original['candidate_index'], original['symbol'], original['signal_date']) ==
                (i, j, meta['symbols'][i][j], meta['dates'][i]), 'Full original row order')
        require(original['tier3'] is False and original['normal_period_admission'] is False and
                row['old_admission_flags_changed'] is False, 'Old flags retained')
        cutoff = cutoff_check(original, meta['dates'][i], calendar, positions)
        flag = row['conditional_annual_pair_admitted']
        require(type(flag) is bool, 'Source flag type')
        source[i, j] = flag
        if flag:
            require(original['symbol'] in ISSUERS and original['cik'] == ISSUERS[original['symbol']] and
                    original['is_etf'] is False and original['tier2'] is True and
                    row['body_pair_context_verified'] is True and row['body_context_source'] == original['symbol'] and
                    not row['remaining_row_reasons'] and not original['known_amendment_event_ids'] and
                    not [r for r in original['reasons'] if r != MISSING], 'Contradictory admitted row')
            inspected_pair(original, events, calendar, cutoff)
    require((int(source.sum()), int(source[:88].sum()), int(source[88:].sum())) == (230, 168, 62), 'Fixed source coverage')
    require(sum(r['original_row']['is_etf'] for r in rows) == 2074, 'Stock/ETF denominator')
    return meta, rows, events, source, calendar


def coverage_oracle(rows, expected):
    result = {}
    for name, lo, hi in [('all122', 0, 122), ('first88', 0, 88), ('later34', 88, 122)]:
        cells = [c for c in expected['cells'] if lo <= c['signal_index'] < hi]
        source = [c for c in cells if c['source_conditionally_admitted']]
        usable = [c for c in cells if c['available']]
        def pairs(items):
            return {tuple(sorted(rows[49*c['signal_index']+c['candidate_index']]['original_row']['selected_event_ids'])) for c in items}
        result[name] = dict(candidate_cells=len(cells), source_conditionally_admitted_rows=len(source),
            usable_rows=len(usable), unique_source_pairs=len(pairs(source)), unique_usable_pairs=len(pairs(usable)),
            months_with_two_usable=sum(sum(c['signal_index'] == i for c in usable) == 2 for i in range(lo, hi)))
    return result


def rank_audit_oracle(meta, expected):
    arrays = expected['arrays']
    return dict(months=[dict(signal_index=i, date=meta['dates'][i],
        shared_query_population=int(arrays['common_available'][i].sum()),
        informative_pair=int(arrays['common_available'][i].sum()) >= 2,
        initial_partial_fallback=meta['dates'][i] == INITIAL,
        unchanged_current_correction=not np.any(arrays[RANKS[0]][i]),
        unchanged_change_correction=not np.any(arrays[RANKS[1]][i])) for i in range(len(arrays['common_available']))],
        prefixes_checked=123)


def verify_feature_audit(saved, expected, rows, input_digests):
    for key in ('source_overlay', 'source_review', 'source_prior'):
        require(saved.get(key + '_sha256') == input_digests[key], 'Feature source binding: ' + key)
    arrays = expected['arrays']
    for key in ('new_predictive_model_fits', 'nuisance_estimates'):
        require(type(saved.get(key)) is int and saved[key] == 0, 'No new estimates: ' + key)
    for key in ('source_admission_changed', 'exact_acceptance_certified', 'historical_price_identity_certified', 'strategy_returns_or_nav_computed'):
        require(saved.get(key) is False, 'Feature scope flag: ' + key)
    require(saved.get('source_contradictions_hard_failed') is True and saved.get('numerical_features_computed') is True,
            'Validated actual feature computation')
    require(saved['rows'] == 5978 and saved['months'] == 122 and saved['dose'] == .01 and
            saved['max_absolute_score_correction'] == .005 and saved['fixed_issuers'] == ISSUERS and
            saved['usable_rows'] == saved['lag_pairs_evaluated'] == int(arrays['common_available'].sum()) and
            saved['source_conditionally_admitted_rows'] == 230, 'Feature audit counts/parameters')
    actual_cells = saved['cells']
    require(len(actual_cells) == len(expected['cells']), 'Every cell audited')
    for actual, want in zip(actual_cells, expected['cells']):
        require(set(actual) == set(want), 'Cell audit schema')
        for key in want:
            require(canonical(sorted(actual[key]) if key == 'reasons' else actual[key]) ==
                    canonical(sorted(want[key]) if key == 'reasons' else want[key]), 'Independent cell field: ' + key)
    require(canonical(saved['monthly']) == canonical(expected['monthly']), 'Independent monthly ranks')
    require(saved['reason_counts'] == dict(Counter(reason for cell in expected['cells'] for reason in cell['reasons'])) and
            saved['informative_rank_months'] == sum(m['current_rank_nonzero'] > 0 or m['change_rank_nonzero'] > 0 for m in expected['monthly']),
            'Reason/informative month counts')
    require(canonical(saved['coverage']) == canonical(coverage_oracle(rows, expected)), 'Coverage and repeated pair counts')
    require(type(saved.get('feature_prefixes_checked')) is int and saved['feature_prefixes_checked'] == 123,
            'Producer feature-prefix count')
    definitions = dict(negative_current_lag_definition='-(filingDate_current-reportDate_current).days',
        negative_lag_change_definition='previous_lag_days-current_lag_days',
        rank_definition='(average_rank1-1)/(n-1)-0.5; n<=1 gives0; rank only common original-eligible mask')
    require(all(saved.get(key) == value for key, value in definitions.items()), 'Feature arithmetic declarations')


def verify_change_counts(saved, expected, choices):
    nonzero = {name: int(np.count_nonzero(np.any(expected['arrays'][key] != 0, axis=1)))
               for name, key in zip(NAMES[1:], RANKS)}
    changed = {name: sum(a != b for a, b in zip(choices[name], choices[NAMES[0]])) for name in NAMES[1:]}
    require(canonical(saved.get('nonzero_correction_months')) == canonical(nonzero) and
            canonical(saved.get('ordered_holdings_changed_months')) == canonical(changed), 'Correction/holding change counts')
    require(saved.get('primary_no_incremental_evidence') is (nonzero[NAMES[2]] == 0 or changed[NAMES[2]] == 0),
            'No incremental evidence flag')
    return dict(nonzero_correction_months=nonzero, ordered_holdings_changed_months=changed,
                primary_no_incremental_evidence=nonzero[NAMES[2]] == 0 or changed[NAMES[2]] == 0)


def measurement_contract():
    return dict(design_sha256=FIXED['design'][1], fixed_issuers=ISSUERS.copy(),
        current='-(filingDate_current-reportDate_current).days', change='previous_lag_days-current_lag_days',
        source_cutoff='original_XNYS_close_at_t_minus_42', availability='second_XNYS_session_strictly_after_filingDate',
        common_mask='conditional_overlay_and_original_eligible_and_valid_exact_selected_pair',
        invalid_latest_blocks_fallback=True, contradictory_admitted_metadata='hard_failure_even_if_ineligible',
        rank='ascending_average_rank0_over_n_minus1_minus_half', min_rank_population=2,
        dose=.01, max_absolute_correction=.005, original_arrays=27, extra_arrays=list(EXTRA),
        score_placement='unchanged_local67_pre_plus_dose_rank_then_one_episode',
        first_partial='2016-08-17_exact_incumbent_fallback', additional_breadth_floor=None,
        new_predictive_model_fits=0, nuisance_estimates=0, exact_acceptance_certified=False)


def selection_contract():
    return dict(primary=NAMES[2], controls=list(NAMES[:2]), first_months=88, selection_count=1,
        same_close='strict_cagr_over_both_controls', next_open_5bp='noninferior_incumbent_strict_over_current_lag',
        cagr_epsilon=1e-12, sharpe_tolerance=.05, max_drawdown_tolerance=.03,
        early_cagr_tolerance=.025, recent_cagr_tolerance=.025, prefix_reselection=False,
        later34_diagnostic_only=True, parameter_search=False)


def validate_plan(reader, plan_path, plan_sha, stage='prepare'):
    independent_imports()
    require(stage in ('prepare', 'outcome', 'evaluate'), 'Known audit stage')
    require(type(plan_sha) is str and re.fullmatch('[0-9a-f]{64}', plan_sha), 'Explicit PLAN SHA')
    directory = reader.path(OUT)
    require(reader.path(plan_path) == directory/'PLAN.json' and reader.sha(plan_path) == plan_sha, 'Exact registered PLAN')
    plan = reader.json(plan_path)
    require(plan.get('round') == 158 and plan.get('status') == 'REGISTERED' and
            plan['arms'] == list(NAMES) and plan['primary_challenger'] == NAMES[2] and plan['dose'] == .01,
            'Three fixed registered arms')
    require(plan['runtime'] == dict(python=sys.version, numpy=np.__version__, pandas=pd.__version__), 'Pinned runtime')
    require(reader.path(plan['output_directory']) == directory and set(plan['inputs']) == set(INPUTS), 'Exact output and inputs')
    require(plan['measurement'] == measurement_contract() and plan['selection_contract'] == selection_contract(),
            'Exact prospective measurement/selection')
    require(plan['tolerances']['feature_atol'] == 1e-12 and plan['tolerances']['scores_atol'] == 1e-12 and
            plan['tolerances']['scores_rtol'] == 0, 'Fixed preparation tolerances')
    for key in ('actual_label_calculations_before_registration', 'actual_model_fits_before_registration',
                'actual_feature_calculations_before_registration', 'actual_returns_before_registration',
                'actual_lag_calculations_before_registration', 'actual_nuisance_estimates_before_registration'):
        require(type(plan.get(key)) is int and plan[key] == 0, 'Prospective zero activity: ' + key)
    require(plan.get('no_observer_writes') is True and type(plan.get('new_market_downloads')) is int and
            plan['new_market_downloads'] == 0, 'Offline original-input experiment')
    require(plan.get('historical_pin_aliases') == {HISTORICAL_STATE: dict(original_sha256=HISTORICAL_SHA,
        archive_path=HISTORICAL_ARCHIVE, archive_sha256=HISTORICAL_SHA)}, 'Exact historical state alias')
    if stage == 'prepare':
        require(not any((directory/name).exists() for name in ('TRAIN_TRIALS.json', 'TRAIN_OPEN_TRIALS.json',
                'WALKFORWARD.json', 'FROZEN.json', 'DIAGNOSE_STARTED.json', 'RESULTS.json')), 'Prepare audit must precede outcomes')
    pins = reader.verify(plan['hashes'])
    sources = {str(p): reader.sha(p) for p in source_files(reader.root)}
    require(reader.verify(plan['independent_prepare_source_hashes']) == sources, 'Exact independent preparation code closure')
    inputs = {}
    for key, ref in plan['inputs'].items():
        path = reader.path(ref['path'])
        require(reader.sha(path) == ref['sha256'] == pins.get(str(path)), 'Named input binding: ' + key)
        inputs[key] = path
    for key, filename in [('policy', 'INCUMBENT_POLICY.json'), ('original_orders', 'local_linear_ORDERS.json')]:
        require(inputs[key] == reader.path(old.LOCAL + '/' + filename), 'Original incumbent input')
    require(inputs['prices'] == reader.path(old.PRICES), 'Original price cache')
    proposal = reader.json(inputs['proposal'])
    require(proposal['round'] == 158 and proposal['arms'] == list(NAMES) and proposal['primary_challenger'] == NAMES[2] and
            proposal['status'] == 'FINAL_PROSPECTIVE_SPECIFICATION_BEFORE_REGISTRATION' and
            proposal['measurement'] == measurement_contract() and proposal['selection_contract'] == selection_contract(),
            'Final prospective proposal')
    require(proposal.get('design') == dict(path=FIXED['design'][0], sha256=FIXED['design'][1]) and
            proposal.get('previous_completion') == dict(path='outputs/monthly_filing_association157/COMPLETION.json',
                sha256='f4de303fc011a336865519d82b48219833eb0953cc9a8dbed0849f3be2354549'),
            'Frozen design and completed predecessor')
    for key in ('actual_new_labels', 'actual_new_model_fits', 'actual_new_features', 'actual_new_orders',
                'actual_new_returns', 'actual_new_lag_values'):
        require(type(proposal.get(key)) is int and proposal[key] == 0, 'Proposal zero activity')
    policy = reader.json(inputs['policy'])
    require(all(policy.get(k) == v for k, v in old.original_helper().POLICY.items()), 'Frozen incumbent episode policy')
    return plan, pins, inputs


def audit(root, plan_path, plan_sha):
    reader = old.Reader(root)
    plan, pins, inputs = validate_plan(reader, plan_path, plan_sha)
    directory = reader.path(OUT)
    meta, rows, events, source, calendar = source_inputs(reader, plan, pins)
    cache = reader.json(directory/'CACHE.json')
    require(set(cache) == set(CACHE_FILES), 'Complete preparation cache')
    reader.verify(cache, directory)
    require(canonical(reader.json(directory/'META.json')) == canonical(meta), 'Exact prepared metadata')
    require(canonical(reader.json(directory/'CALENDAR.json')) == canonical(calendar), 'Independent prepared calendar')
    original, arrays = reader.npz(old.LOCAL + '/DATA.npz'), reader.npz(directory/'DATA.npz')
    verify_original_arrays(original, arrays)
    expected = features_oracle(meta, rows, events, source, original['eligible'], calendar)
    for key in EXTRA:
        require(arrays[key].shape == expected['arrays'][key].shape and arrays[key].dtype == expected['arrays'][key].dtype,
                'Feature matrix shape/dtype')
        np.testing.assert_array_equal(arrays[key], expected['arrays'][key])
    feature_audit = reader.json(directory/'FEATURE_AUDIT.json')
    verify_feature_audit(feature_audit, expected, rows,
                         {k: plan['inputs'][k]['sha256'] for k in ('source_overlay', 'source_review', 'source_prior')})
    require(canonical(reader.json(directory/'RANK_AUDIT.json')) == canonical(rank_audit_oracle(meta, expected)), 'Rank audit')
    scores, choices, deltas = score_oracle(arrays, meta, expected['arrays'])
    change_counts = verify_change_counts(feature_audit, expected, choices)
    saved_scores, saved_orders = reader.npz(directory/'SCORES.npz'), reader.json(directory/'PRETRAIN_ORDERS.json')
    require(set(saved_scores) == set(saved_orders) == set(NAMES), 'Three prepared arms')
    wanted = order_oracle(meta, scores, choices)
    for name in NAMES:
        require(scores[name].shape == saved_scores[name].shape and scores[name].dtype == saved_scores[name].dtype, 'Score shape/dtype')
        np.testing.assert_allclose(saved_scores[name], scores[name], rtol=0, atol=1e-12)
        old.compare(saved_orders[name], wanted[name], name)
    require(saved_orders[NAMES[0]] == reader.json(inputs['original_orders']), 'Frozen incumbent orders')
    for n in range(123):
        prefix = features_oracle(meta, rows, events, source, original['eligible'], calendar, n)
        for key in EXTRA:
            np.testing.assert_array_equal(prefix['arrays'][key], expected['arrays'][key][:n])
        ss, cc, _ = score_oracle(arrays, meta, prefix['arrays'], n=n)
        for name in NAMES:
            require(cc[name] == choices[name][:n] and ss[name].tobytes() == scores[name][:n].tobytes(), 'Causal score/order prefix')
    zeros = {key: np.zeros((122, 49)) for key in RANKS}
    neutral_scores, neutral_choices, _ = score_oracle(arrays, meta, zeros)
    for name, rank in zip(NAMES[1:], RANKS):
        require(neutral_choices[name] == choices[NAMES[0]] and neutral_scores[name].tobytes() == scores[NAMES[0]].tobytes(), 'Neutral fallback')
        for i in range(122):
            if not np.any(expected['arrays'][rank][i]):
                require(choices[name][i] == choices[NAMES[0]][i] and
                        scores[name][i].tobytes() == scores[NAMES[0]][i].tobytes(), 'Exact unchanged-month fallback')
    require(reader.sha(old.LOADER) == old.LOADER_SHA, 'Frozen raw price loader')
    spec = importlib.util.spec_from_file_location('lag158_independent_price_loader', reader.path(old.LOADER))
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    panel, info = loader.load_panel(meta['etfs'])
    used = dict(info['source_hashes'])
    used[str(reader.path(old.LOADER))] = old.LOADER_SHA
    raw = reader.verify(plan['raw_source_hashes'])
    require({str(reader.path(p)): h for p, h in used.items()} == raw and len(raw) == 64, 'Exact64 raw sources')
    reader.verify(used)
    parity = old.price_parity(panel, reader.npz(inputs['prices']))
    old.compare(reader.json(directory/'PRICE_PARITY.json'), parity, 'price-parity')
    required = {str(directory/name): reader.sha(directory/name) for name in (*CACHE_FILES, 'PLAN.json', 'CACHE.json')}
    allowed = dict(pins, **required)
    for path, digest in list(reader.seen.items()):
        require(reader.sha(path) == digest == allowed.get(path), 'Unbound or changed input: ' + path)
    independent_imports()
    return dict(stage='prepare', passed=True, plan_sha256=plan_sha, cache_sha256=reader.sha(directory/'CACHE.json'),
        orders_sha256=reader.sha(directory/'PRETRAIN_ORDERS.json'),
        source_hashes={str(p): reader.sha(p) for p in source_files(reader.root)}, artifact_hashes=dict(reader.seen),
        **dict.fromkeys(CHECKS, True), producer_modules_imported=False, returns_or_nav_computed=False,
        orders_checked=366, candidate_cells_checked=5978, months_checked=122, prefixes_checked=123,
        lag_feature_matrices_verified=2, new_predictive_model_fits=0, nuisance_estimates=0,
        available_cells=int(expected['arrays']['common_available'].sum()), source_available_cells=int(source.sum()),
        coverage=coverage_oracle(rows, expected), exact_acceptance_certified=False,
        **change_counts,
        max_absolute_pre_score_correction=max(float(np.max(np.abs(v))) for v in deltas.values()),
        max_score_error=max(float(np.max(np.abs(saved_scores[name]-scores[name]))) for name in NAMES),
        limitations=['Fixed KO/WMT source pilot and reconstructed historical availability; no exact publication or price-identity certification.',
                     'Repeated monthly pairs are counted explicitly; these are not independent annual observations.',
                     'This receipt verifies preparation only and cannot establish return improvement or target achievement.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--plan-sha', required=True)
    args = parser.parse_args()
    result = audit(ROOT, args.plan, args.plan_sha)
    path = ROOT/OUT/'INDEPENDENT_PREPARE.json'
    with path.open('x') as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
    print(json.dumps(dict(passed=True, path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())))


if __name__ == '__main__':
    main()
