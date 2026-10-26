"""Independent round139 prepare review; real inputs require explicit audit().

Inputs are already validated event10/matched_recent10 measurements from the
frozen earnings feature validator. This module neither selects new events nor
recomputes their returns. All pool/peer arithmetic is independently implemented
here; no round139 producer module is imported.
"""
from datetime import date
from pathlib import Path
import argparse
import ast
import hashlib
import importlib.util
import io
import json
import math
import re
import sys

import numpy as np
import pandas as pd


ARMS = ('incumbent_local67', 'peer_recent10', 'negative_own_event10', 'peer_event10')
RAW_KEYS = ('raw_peer_recent10', 'raw_negative_own_event10', 'raw_peer_event10')
HAND_KEYS = ('hand_peer_recent10', 'hand_negative_own_event10', 'hand_peer_event10')
EXTRA_KEYS = RAW_KEYS + HAND_KEYS + ('common_available', 'source_available', 'peer_count')
HELPER_RELATIVE = 'work/monthly_ridge_support70_result_review.py'
HELPER_SHA = 'd0f053881cb94e49e7fff9eaaeaa9e260f0b3c66419f87819391c309cb7cb8ef'
DEFAULT_ROOT = Path('/Users/shiweili/code/monthly-stock-research')
STUDY = 'outputs/monthly_peer_earnings139'
LOCAL = 'outputs/monthly_local_linear70'
PRICES = 'outputs/monthly_2016_extension/PRICES.npz'
RAW_LOADER = 'outputs/monthly_stability_monitor/inputs.py'
RAW_LOADER_SHA = 'dd97fb4a9f7b2102d2c7f497825526a484628fa86ee9891312bfb5abad7c8ded'
MAPPING = 'outputs/monthly_strict_relative/strict_study.py'
MAPPING_SHA = 'ddb11c0149e920de2952daba515ce28625a56df1f5c1039d4b9920c6d58d02c6'
CALENDAR_SHA = '4b8f32ad8a555d6537d7cf3bd0f831e64fbb2de26d033c8daaf68590fcd4a6ea'
PROPOSAL_SHA = '3c379b0246c8cbee19db9ace977dd1f0138e036922658fe63e838f89a9c0b7f4'
REPAIR = 'outputs/monthly_earnings_repair'
EVENT_VALIDATOR = 'outputs/monthly_earnings_factor_research/earnings_features.py'
ISSUER_SOURCE = 'outputs/monthly_top35_extension/CAP_COLLECTION.json'
INPUTS = ('proposal', 'meta', 'policy', 'original_orders', 'prices', 'sessions', 'protected_manifest')
CACHE_FILES = ('DATA.npz', 'SCORES.npz', 'META.json', 'FEATURE_AUDIT.json', 'PRETRAIN_ORDERS.json', 'PRICE_PARITY.json')
CHECKS = ('original27_arrays_byte_identical', 'all_features_independently_reconstructed',
          'all_four_score_matrices_and_ordered_choices_independently_verified',
          'one_episode_and_prefix_isolation_verified', 'zero_and_missing_exact_fallback')
ATOL = 1e-12
SECTOR = {'Basic Materials':'XLB', 'Communication Services':'XLC', 'Energy':'XLE',
          'Financial Services':'XLF', 'Industrials':'XLI', 'Technology':'XLK',
          'Consumer Defensive':'XLP', 'Real Estate':'XLRE', 'Utilities':'XLU',
          'Healthcare':'XLV', 'Consumer Cyclical':'XLY'}
FOREIGN = {'ASML':'XLK', 'ARM':'XLK', 'TSM':'XLK', 'BABA':'XLY', 'NVO':'XLV',
           'SAP':'XLK', 'TM':'XLY', 'SHEL':'XLE', 'SKHY':'XLK', 'BHP':'XLB',
           'HSBC':'XLF', 'SMFG':'XLF', 'NVS':'XLV', 'SNDK':'XLK'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_day(value, label):
    require(isinstance(value, str), label + ' must be YYYY-MM-DD')
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(label + ' must be YYYY-MM-DD') from error
    require(parsed.isoformat() == value, label + ' must be canonical YYYY-MM-DD')
    return value


def numeric_matrix(value, shape, label):
    array = np.asarray(value)
    require(array.shape == shape and array.dtype.kind in 'fiu', label + ' numeric nonboolean shape')
    require(not np.isinf(array).any(), label + ' cannot contain infinity')
    return array


def text_matrix(value, shape, label):
    array = np.asarray(value, dtype=object)
    require(array.shape == shape, label + ' shape')
    require(all(x is None or isinstance(x, str) and bool(x) for x in array.flat),
            label + ' must contain nonempty strings or None')
    return array


def build_peer_features(*, meta, event10, matched_recent10, valid10,
                        anchor_sessions, event_ids, official_sessions,
                        sector_benchmarks, issuer_keys):
    """Reconstruct the nine-array feature payload from normalized source cells.

    No eligibility mask is accepted: peer construction uses the original pool.
    Every known non-ETF issuer must occur at most once in each monthly pool.
    Audit cells are in the original pool order; donor lists sort by issuer/symbol.
    """
    validity = np.asarray(valid10)
    require(validity.ndim == 2 and validity.dtype == np.dtype(bool), 'valid10 strict bool matrix')
    shape = validity.shape
    months, width = shape
    require(width > 0, 'Positive original pool width')
    require(isinstance(meta, dict), 'Metadata object')
    days, pools, etfs = meta['dates'], meta['symbols'], meta['etfs']
    require(len(days) == len(pools) == months, 'Metadata month axis')
    require(len(etfs) == len(set(etfs)) and all(isinstance(x, str) and x for x in etfs), 'Unique ETF identifiers')
    etfs = set(etfs)
    days = [canonical_day(x, 'Signal date') for x in days]
    require(all(a < b for a, b in zip(days, days[1:])), 'Strictly increasing signal dates')
    calendar = [canonical_day(x, 'Official session') for x in official_sessions]
    require(bool(calendar) and all(a < b for a, b in zip(calendar, calendar[1:])), 'Strict official session axis')
    positions = {day: i for i, day in enumerate(calendar)}
    require(all(day in positions for day in days), 'Signal must be an official session')
    for pool in pools:
        require(len(pool) == width and all(isinstance(x, str) and x for x in pool)
                and len(set(pool)) == width, 'Original unique pool axis')

    events = numeric_matrix(event10, shape, 'event10')
    recent = numeric_matrix(matched_recent10, shape, 'matched_recent10')
    require(np.isfinite(events[validity]).all() and np.isfinite(recent[validity]).all(), 'Valid source reactions finite')
    require(np.isnan(events[~validity]).all() and np.isnan(recent[~validity]).all(), 'Invalid source reactions are NaN')
    anchors = text_matrix(anchor_sessions, shape, 'anchor_sessions')
    ids = text_matrix(event_ids, shape, 'event_ids')
    sectors = text_matrix(sector_benchmarks, shape, 'sector_benchmarks')
    issuers = text_matrix(issuer_keys, shape, 'issuer_keys')
    ages = np.empty(shape, dtype=object)
    for i in range(months):
        known_issuers = []
        for j, symbol in enumerate(pools[i]):
            issuer = issuers[i, j]
            if symbol not in etfs and issuer is not None:
                known_issuers.append(issuer)
            anchor = anchors[i, j]
            if anchor is not None:
                canonical_day(anchor, 'Anchor session')
                require(anchor in positions, 'Anchor must be an official session')
                ages[i, j] = positions[days[i]] - positions[anchor]
            else:
                ages[i, j] = None
            if validity[i, j]:
                require(anchor is not None and ids[i, j] is not None, 'Valid source has anchor and event identity')
                require(ages[i, j] >= 0, 'Valid source anchor cannot follow signal')
        require(len(known_issuers) == len(set(known_issuers)), 'Duplicate known non-ETF issuer in original pool')

    arrays = {key: np.full(shape, np.nan, dtype=float) for key in RAW_KEYS + HAND_KEYS}
    arrays['common_available'] = np.zeros(shape, dtype=bool)
    arrays['source_available'] = np.zeros(shape, dtype=bool)
    arrays['peer_count'] = np.zeros(shape, dtype=np.int64)
    reasons = np.empty(shape, dtype=object)
    for i in range(months):
        for j, symbol in enumerate(pools[i]):
            if symbol in etfs:
                reason = 'etf'
            elif issuers[i, j] is None:
                reason = 'unknown_issuer'
            elif sectors[i, j] is None:
                reason = 'unknown_sector'
            elif not validity[i, j]:
                reason = 'missing_event'
            elif ages[i, j] < 10:
                reason = 'immature_event'
            elif ages[i, j] > 63:
                reason = 'stale_event'
            else:
                reason = 'source_available'
                arrays['source_available'][i, j] = True
            reasons[i, j] = reason

    cells = []
    for i in range(months):
        for j, symbol in enumerate(pools[i]):
            peers = []
            if arrays['source_available'][i, j]:
                for k in range(width):
                    if (arrays['source_available'][i, k]
                            and sectors[i, k] == sectors[i, j]
                            and issuers[i, k] != issuers[i, j]):
                        peers.append(k)
                peers.sort(key=lambda k: (issuers[i, k], pools[i][k]))
                arrays['peer_count'][i, j] = len(peers)
                reasons[i, j] = 'available' if len(peers) >= 2 else 'insufficient_peer_issuers'
                if len(peers) >= 2:
                    count = len(peers)
                    values = (math.fsum(float(recent[i, k]) / count for k in peers),
                              -float(events[i, j]) / count,
                              math.fsum(float(events[i, k]) / count for k in peers))
                    require(all(math.isfinite(x) for x in values), 'Finite peer raw features')
                    for raw_key, hand_key, value in zip(RAW_KEYS, HAND_KEYS, values):
                        value = 0. if value == 0. else value
                        arrays[raw_key][i, j] = value
                        arrays[hand_key][i, j] = .5 * math.tanh(value / .05)
                    arrays['common_available'][i, j] = True
            cell = dict(i=i, j=j, signal_date=days[i], symbol=symbol,
                        sector_benchmark=sectors[i, j], issuer_key=issuers[i, j],
                        event_id=ids[i, j], anchor_session=anchors[i, j], age_sessions=ages[i, j],
                        source_available=bool(arrays['source_available'][i, j]),
                        available=bool(arrays['common_available'][i, j]), reason=reasons[i, j],
                        peer_count=len(peers), peer_symbols=[pools[i][k] for k in peers],
                        peer_issuer_keys=[issuers[i, k] for k in peers],
                        peer_event_ids=[ids[i, k] for k in peers])
            for raw_key in RAW_KEYS:
                value = arrays[raw_key][i, j]
                cell[raw_key] = float(value) if math.isfinite(value) else None
            cells.append(cell)
    for key in RAW_KEYS + HAND_KEYS:
        require(np.array_equal(np.isfinite(arrays[key]), arrays['common_available']), 'Exact common feature mask')
    return dict(arrays=arrays, audit=dict(method='fixed_latest_event10_peer_predictive_aggregation_v1',
        candidate_rows=months * width, signal_dates=months, candidates_per_date=width,
        all_three_masks_equal=True, source_peer_trading_eligibility_used=False, cells=cells))


def load_baseline_helper(root=DEFAULT_ROOT):
    """Load only the original frozen independent baseline/accounting source."""
    path = Path(root).resolve() / HELPER_RELATIVE
    require(hashlib.sha256(path.read_bytes()).hexdigest() == HELPER_SHA, 'Frozen independent helper digest')
    spec = importlib.util.spec_from_file_location('peer139_independent_baseline', path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


def score_oracle(original_arrays, meta, hands, n=None, *, helper=None):
    """Independently apply three bounded corrections and exactly one episode/arm.

    ``hands`` maps ARMS[1:] to their matrices. ``helper`` injection is for
    synthetic tests; real preparation should use the digest-bound default.
    """
    helper = load_baseline_helper() if helper is None else helper
    context = helper.original_prefix(original_arrays, meta, n)
    n = context['n']
    shape = context['local'].shape
    require(set(hands) == set(ARMS[1:]), 'Exactly the three registered feature hands')
    checked = {}
    for name in ARMS[1:]:
        hand = np.asarray(hands[name])
        require(hand.ndim == 2 and hand.shape[0] >= n and hand.shape[1] == shape[1]
                and hand.dtype.kind in 'fiu', 'Hand shape and nonboolean numeric type')
        hand = hand[:n]
        require(not np.isinf(hand).any(), 'No infinite hand')
        require((np.abs(hand[np.isfinite(hand)]) <= .5).all(), 'Hand bounded by one half')
        for i, names in enumerate(meta['symbols'][:n]):
            require(all(not np.isfinite(hand[i, j]) for j, symbol in enumerate(names)
                        if symbol in meta['etfs']), 'ETF has no hand')
        checked[name] = hand
    mask = np.isfinite(checked[ARMS[1]])
    require(all(np.array_equal(np.isfinite(value), mask) for value in checked.values()), 'Three hands share exact availability')
    scores, choices, deltas = {}, {}, {}
    for name in ARMS:
        pre = context['local'].copy()
        delta = np.zeros(shape, dtype=float)
        if name != ARMS[0]:
            hand = checked[name]
            for i in range(shape[0]):
                for j in range(shape[1]):
                    value = float(hand[i, j])
                    if not context['eligible'][i, j] or not math.isfinite(value) or value == 0.:
                        continue
                    support = float(context['support'][i, j])
                    delta[i, j] = .025 * (min(.5, max(-.5, support + value)) - support)
            require((np.abs(delta) <= .0125 + 1e-12).all(), 'Bounded support correction')
            pre[delta != 0.] += delta[delta != 0.]
        selected, final, _ = helper.episode(pre, context)
        scores[name], choices[name], deltas[name] = final, selected, delta
    return scores, choices, deltas


def orders_oracle(meta, scores, choices):
    """Render the original frozen order schema without computing any returns."""
    require(set(scores) == set(choices) == set(ARMS), 'Four score/order arms')
    count = len(choices[ARMS[0]])
    require(len(meta['dates']) >= count and len(meta['ends']) >= count, 'Order metadata prefix')
    result = {}
    for name in ARMS:
        require(len(choices[name]) == count and len(scores[name]) == count, 'Common order prefix')
        result[name] = [dict(date=meta['dates'][i], signal_date=meta['dates'][i], end=meta['ends'][i],
            symbols=[meta['symbols'][i][j] for j in selected],
            scores=[float(scores[name][i, j]) for j in selected])
            for i, selected in enumerate(choices[name])]
    return result


def assert_original_arrays_unchanged(original, prepared, extra_keys=EXTRA_KEYS):
    require(len(original) == 27, 'Exactly27 original arrays')
    require(set(prepared) == set(original) | set(extra_keys)
            and not set(original) & set(extra_keys), 'Exact prepared array schema')
    for key, value in original.items():
        other = prepared[key]
        require(other.shape == value.shape and other.dtype == value.dtype
                and other.tobytes() == value.tobytes(), 'Original array byte identity: ' + key)


def assert_incumbent_unchanged(expected_scores, actual_scores, expected_orders, actual_orders, *, helper=None):
    helper = load_baseline_helper() if helper is None else helper
    require(expected_scores.shape == actual_scores.shape and expected_scores.dtype == actual_scores.dtype
            and expected_scores.tobytes() == actual_scores.tobytes(), 'Incumbent score byte identity')
    helper.assert_exact_tree(actual_orders, expected_orders)


class Reader:
    """Bind the exact inspected bytes; all consumed artifacts are rechecked."""
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.seen = {}

    def path(self, value):
        path = Path(value)
        return path.resolve() if path.is_absolute() else (self.root/path).resolve()

    def content(self, value):
        path = self.path(value)
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        require(str(path) not in self.seen or self.seen[str(path)] == digest, 'Artifact changed: ' + str(path))
        self.seen[str(path)] = digest
        return data

    def sha(self, value):
        return hashlib.sha256(self.content(value)).hexdigest()

    def json(self, value):
        def pairs(items):
            result = {}
            for key, child in items:
                require(key not in result, 'Duplicate JSON key')
                result[key] = child
            return result
        def nonfinite(value):
            raise ValueError('Nonfinite JSON token: ' + value)
        return json.loads(self.content(value), object_pairs_hook=pairs, parse_constant=nonfinite)

    def npz(self, value):
        with np.load(io.BytesIO(self.content(value)), allow_pickle=False) as archive:
            return {key: archive[key].copy() for key in archive.files}

    def verify(self, mapping, base=None):
        require(isinstance(mapping, dict) and bool(mapping), 'Nonempty hash mapping')
        result = {}
        for name, digest in mapping.items():
            require(isinstance(digest, str) and re.fullmatch('[0-9a-f]{64}', digest) is not None, 'Valid SHA256')
            if base is not None:
                require(Path(name).name == name and name not in ('', '.', '..'), 'Cache manifest basenames')
                name = Path(base)/name
            path = str(self.path(name))
            require(path not in result or result[path] == digest, 'Conflicting canonical pins')
            require(self.sha(path) == digest, 'Hash mismatch: ' + path)
            result[path] = digest
        return result


def frozen_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def original_tools(root=DEFAULT_ROOT):
    return frozen_module(Path(root)/'work/monthly_latent_state134_prepare_review.py',
                         'peer139_frozen_prepare_utilities')


def source_files(root=None):
    root = DEFAULT_ROOT if root is None else Path(root).resolve()
    tools = original_tools(root)
    return sorted(set([root/'work/monthly_peer_earnings139_prepare_review.py',
                       root/'work/test_monthly_peer_earnings139_prepare_review.py',
                       root/MAPPING, root/'outputs/monthly_path70_research/incumbent_adapter.py',
                       root/EVENT_VALIDATOR, root/REPAIR/'research_calendar.py'] + tools.source_files(root)))


def verify_mapping_source(reader):
    require(reader.sha(MAPPING) == MAPPING_SHA, 'Frozen original mapping source')
    assignments = {}
    for node in ast.parse(reader.content(MAPPING)).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ('SECTOR', 'FOREIGN'):
                    assignments[target.id] = ast.literal_eval(node.value)
    require(assignments == dict(SECTOR=SECTOR, FOREIGN=FOREIGN), 'Original mapping constants')


def benchmark_map(day, pool, sectors, etfs):
    require(isinstance(sectors, pd.DataFrame) and sectors.index.is_unique and sectors.columns.is_unique, 'Sector axes')
    day = pd.Timestamp(day)
    result = {}
    for symbol in pool:
        if symbol in etfs:
            result[symbol] = None
        elif symbol in FOREIGN:
            result[symbol] = FOREIGN[symbol]
        elif symbol in sectors.columns and day in sectors.index:
            result[symbol] = SECTOR.get(sectors.at[day, symbol])
        else:
            result[symbol] = None
    return result


def latest_event_rows(meta, availability, events, document, calendar):
    """Independent latest-known-event identity check, before maturity filtering."""
    expected = [(day, symbol) for day, pool in zip(meta['dates'], meta['symbols']) for symbol in pool]
    require(len(expected) == len(set(expected)), 'Original candidate identities are unique')
    def keyed(rows):
        result = {}
        for row in rows:
            key = (row['signal_date'], row['symbol'])
            require(key not in result, 'Duplicate event candidate')
            result[key] = row
        require(set(result) == set(expected), 'Exact original event candidate axes')
        return result
    rows, old = keyed(document['rows']), keyed(availability)
    identities, histories = {}, {}
    for event in events:
        identity = event['event_id']
        require(isinstance(identity, str) and identity and identity not in identities, 'Unique event IDs')
        identities[identity] = event
        if event['status'] != 'ready':
            require(event.get('known_by_at') is None, 'Unready event has no knowledge timestamp')
            continue
        known = pd.Timestamp(event['known_by_at'])
        require(not pd.isna(known) and known.tz is not None, 'Timezone-aware event knowledge')
        histories.setdefault(event['symbol'], []).append((known, identity))
    for day, symbol in expected:
        row, reference = rows[day, symbol], old[day, symbol]
        require((row['status'], row['event_id']) == (reference['status'], reference['event_id']), 'Frozen availability identity')
        signal_close = calendar.session_close(pd.Timestamp(day))
        require(pd.Timestamp(row['signal_asof']) == signal_close, 'Official signal close')
        if symbol in meta['etfs']:
            require(row['status'] == 'not_applicable_etf' and row['event_id'] is None and not row['reactions'], 'ETF has no event values')
            continue
        visible = [(known, identity) for known, identity in histories.get(symbol, []) if known <= signal_close]
        latest = None
        if visible:
            newest = max(known for known, _ in visible)
            finalists = [identity for known, identity in visible if known == newest]
            require(len(finalists) == 1, 'Ambiguous latest event timestamp')
            latest = finalists[0]
        require(row['event_id'] == latest, 'Not the latest known event')
        if latest is None:
            require(row['status'] != 'verified_event_available' and not row['reactions'], 'Unknown event cannot be scored')
        else:
            require(row['status'] == 'verified_event_available', 'Known event marked missing')
            require(all(reaction['known_by_at'] == identities[latest]['known_by_at']
                        for reaction in row['reactions'].values()), 'Reaction knowledge belongs to selected event')
    return rows


def issuer_map(records):
    aliases = {'META': 'research_equity_FB', 'BRK-B': 'BRK.B'}
    result = {}
    for record in records:
        symbol, identity = record['symbol'], record['url']
        require(isinstance(symbol, str) and isinstance(identity, str) and bool(identity), 'Registered issuer identity')
        symbol = aliases.get(symbol, symbol)
        require(symbol not in result or result[symbol] == identity, 'Conflicting issuer identity')
        result[symbol] = identity
    return result


def normalized_event_inputs(reader, meta, panel, sessions, event_pins):
    """Validate frozen event arrays; do not reconstruct repaired raw event prices."""
    consumed = set()
    def bound(path, json_value=False):
        key = str(reader.path(path))
        require(key in event_pins and reader.sha(path) == event_pins[key], 'Event source closure: ' + key)
        consumed.add(key)
        return reader.json(path) if json_value else reader.path(path)
    readiness = bound(REPAIR+'/SCOPED_READINESS.json', True)
    require(readiness['scoped_post_known_drift_data_ready'] is True and readiness['unresolved_count'] == 0
            and not readiness['unresolved'] and not readiness['invalid_candidate_price_windows']
            and readiness['protected_files_changed'] == 0, 'Frozen event readiness')
    for name, digest in readiness['inputs'].items():
        path = name if Path(name).is_absolute() else REPAIR+'/'+name
        bound(path)
        require(reader.sha(path) == digest, 'Event readiness input binding')
    panel_audit = bound(REPAIR+'/MONTHLY_PANEL_AUDIT.json', True)
    for name, digest in panel_audit['input_sha256'].items():
        path = REPAIR+'/'+name
        bound(path)
        require(reader.sha(path) == digest, 'Event panel input binding')
    document = bound(REPAIR+'/MONTHLY_FEATURES.json', True)
    availability = bound(REPAIR+'/MONTHLY_AVAILABILITY.json', True)
    events = bound(REPAIR+'/EVENTS.json', True)
    path = bound(EVENT_VALIDATOR)
    for path_source in (REPAIR+'/research_calendar.py', 'outputs/index_union/CALENDAR_EXCEPTIONS.json',
                        'outputs/index_union/sources/closure_20250109.html'):
        bound(path_source)
    validator = frozen_module(path, 'peer139_independent_frozen_event_validator')
    calendar = validator._calendar()
    actual_sessions = [str(day.date()) for day in calendar.sessions
                       if sessions[0] <= str(day.date()) <= sessions[-1]]
    require(actual_sessions == list(sessions), 'Event official calendar equals registered calendar')
    rows = latest_event_rows(meta, availability, events, document, calendar)
    event, event_audit = validator.build_features(document, meta, panel['adj'], meta['etfs'])
    identities = issuer_map(bound(ISSUER_SOURCE, True)['records'])
    shape = (len(meta['dates']), len(meta['symbols'][0]))
    anchors, ids, sectors, issuers = [np.full(shape, None, dtype=object) for _ in range(4)]
    for i, (day, pool) in enumerate(zip(meta['dates'], meta['symbols'])):
        mapped = benchmark_map(day, pool, panel['sectors'], meta['etfs'])
        for j, symbol in enumerate(pool):
            row = rows[day, symbol]
            ids[i, j] = row['event_id']
            if symbol not in meta['etfs']:
                sectors[i, j], issuers[i, j] = mapped.get(symbol), identities.get(symbol)
            if event['valid10'][i, j]:
                anchors[i, j] = row['reactions']['10']['anchor_session']
    normalized = dict(meta=meta, event10=event['event10'], matched_recent10=event['matched_recent10'],
        valid10=event['valid10'], anchor_sessions=anchors, event_ids=ids, official_sessions=list(sessions),
        sector_benchmarks=sectors, issuer_keys=issuers)
    provenance = dict(event_validator_audit=event_audit, original_event10_values_preserved=True,
        raw_event_prices_reconstructed=False, latest_known_event_independently_checked=True,
        unknown_issuer_symbols=sorted({symbol for pool in meta['symbols'] for symbol in pool
                                      if symbol not in meta['etfs'] and symbol not in identities}),
        source_semantics='Frozen post-known-close drift; not immediate announcement surprise.',
        mapping_limitation='Inherited sector proxy and issuer URL identity, not new PIT classification proof.')
    return normalized, provenance, consumed


def compare(actual, expected, path=''):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), 'Object schema ' + path)
        for key in expected: compare(actual[key], expected[key], path+'/'+key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), 'List schema ' + path)
        for i, (left, right) in enumerate(zip(actual, expected)): compare(left, right, path+'/'+str(i))
    elif type(expected) is float:
        require(type(actual) in (float, int) and math.isfinite(actual) and math.isfinite(expected)
                and math.isclose(actual, expected, abs_tol=ATOL, rel_tol=0), 'Numeric comparison ' + path)
    else:
        require(type(actual) is type(expected) and actual == expected, 'Discrete comparison ' + path)


def audit(root, plan_path, plan_sha):
    require(re.fullmatch('[0-9a-f]{64}', plan_sha) is not None, 'Explicit PLAN SHA256')
    require(re.fullmatch('[0-9a-f]{64}', PROPOSAL_SHA) is not None, 'Finalize proposal binding before real audit')
    reader = Reader(root)
    directory = reader.path(STUDY)
    require(reader.path(plan_path) == directory/'PLAN.json', 'Dedicated registered PLAN')
    require(reader.sha(plan_path) == plan_sha, 'Exact PLAN digest')
    plan = reader.json(plan_path)
    require(plan.get('round') == 139 and plan.get('status') == 'REGISTERED'
            and plan.get('arms') == list(ARMS) and plan.get('primary_challenger') == ARMS[-1], 'Four-arm139 protocol')
    require(reader.path(plan['output_directory']) == directory, 'Dedicated output directory')
    require(plan['runtime'] == dict(python=sys.version, numpy=np.__version__, pandas=pd.__version__), 'Frozen runtime')
    require(not any((directory/name).exists() for name in ('TRAIN_TRIALS.json', 'FROZEN.json', 'RESULTS.json')), 'Prepare precedes outcomes')
    pins = reader.verify(plan['hashes'])
    require(set(plan['inputs']) == set(INPUTS), 'Exact named inputs')
    inputs = {}
    for name, ref in plan['inputs'].items():
        path = reader.path(ref['path'])
        require(reader.sha(path) == ref['sha256'] == pins.get(str(path)), 'Named input binding '+name)
        inputs[name] = path
    require(reader.sha(inputs['proposal']) == PROPOSAL_SHA, 'Final registered proposal')
    for name in ('actual_model_fits_before_registration', 'actual_feature_calculations_before_registration', 'actual_returns_before_registration'):
        require(type(plan.get(name)) is int and plan[name] == 0, 'Prospective registration '+name)
    for name, wanted in (('feature_atol', ATOL), ('feature_rtol', 0), ('scores_atol', ATOL), ('scores_rtol', 0)):
        require(plan['tolerances'].get(name) == wanted, 'Frozen tolerance '+name)
    sources = {str(path): reader.sha(path) for path in source_files(root)}
    require(reader.verify(plan['independent_prepare_source_hashes']) == sources
            and all(pins.get(key) == value for key, value in sources.items()), 'Independent reviewer source closure')
    old_plan = reader.json(LOCAL+'/PLAN.json')
    raw = reader.verify(plan['raw_source_hashes'])
    require(raw == reader.verify(old_plan['raw_source_hashes']) and len(raw) == 64
            and all(pins.get(key) == value for key, value in raw.items()), 'Original64 raw source identity')
    event_pins = reader.verify(plan['event_input_hashes'])
    require(all(pins.get(key) == value for key, value in event_pins.items()), 'Separate event input closure')
    old_cache = reader.verify(reader.json(LOCAL+'/CACHE_HASHES.json'), reader.path(LOCAL))
    require(all(pins.get(key) == value for key, value in old_cache.items()), 'Original cache artifact pins')
    for name, filename in (('meta','META.json'), ('policy','INCUMBENT_POLICY.json'), ('original_orders','local_linear_ORDERS.json')):
        require(inputs[name] == reader.path(LOCAL+'/'+filename), 'Original input wiring '+name)
    require(inputs['prices'] == reader.path(PRICES), 'Original price input')
    helper = load_baseline_helper(root)
    policy = reader.json(inputs['policy'])
    require(all(policy.get(key) == value for key, value in helper.POLICY.items()), 'Frozen episode policy')
    cache = reader.json(directory/'CACHE.json')
    require(set(cache) == set(CACHE_FILES), 'Six prepared cache files')
    reader.verify(cache, directory)
    meta = reader.json(directory/'META.json')
    require(meta == reader.json(inputs['meta']), 'Original metadata identity')
    require(len(meta['dates']) == len(meta['symbols']) == 122 and meta['train_months'] == 88
            and meta['early_months'] == 53 and len(meta['etfs']) == 17
            and all(len(pool) == 49 for pool in meta['symbols']), 'Original122x49 geometry')
    original, prepared = reader.npz(LOCAL+'/DATA.npz'), reader.npz(directory/'DATA.npz')
    assert_original_arrays_unchanged(original, prepared)
    require(reader.sha(RAW_LOADER) == RAW_LOADER_SHA, 'Frozen raw loader source')
    loader = frozen_module(reader.path(RAW_LOADER), 'peer139_independent_raw_loader')
    panel, info = loader.load_panel(meta['etfs'])
    consumed_raw = dict(info['source_hashes'])
    consumed_raw[str(reader.path(RAW_LOADER))] = RAW_LOADER_SHA
    require({str(reader.path(key)): value for key, value in consumed_raw.items()} == raw, 'Actually consumed raw64 sources')
    reader.verify(consumed_raw)
    verify_mapping_source(reader)
    tools = original_tools(root)
    prices = reader.npz(inputs['prices'])
    require(set(prices) == {'dates', 'symbols', 'close', 'opening'}, 'Original price cache schema')
    parity = tools.price_parity(panel, prices)
    tools.compare(reader.json(directory/'PRICE_PARITY.json'), parity, 'PRICE_PARITY')
    calendar_path = inputs['sessions'].parent/'CALENDAR_PLAN.json'
    require(reader.sha(calendar_path) == CALENDAR_SHA, 'Calendar plan digest')
    calendar_plan = reader.json(calendar_path)
    calendar_pins = reader.verify(calendar_plan['bindings'])
    require(all(pins.get(key) == value for key, value in calendar_pins.items()), 'Calendar input closure')
    require(calendar_plan['start'] == '2015-01-01' and calendar_plan['end'] == '2027-12-31', 'Calendar horizon')
    sessions = tools.official_sessions(reader, calendar_plan['end'])
    sessions = sessions[sessions >= calendar_plan['start']]
    session_days = [str(day.date()) for day in sessions]
    saved_sessions = reader.json(inputs['sessions'])
    require(saved_sessions['plan_sha256'] == CALENDAR_SHA and saved_sessions['sessions'] == session_days, 'Official calendar and overlay')
    normalized, provenance, consumed_events = normalized_event_inputs(reader, meta, panel, session_days, event_pins)
    expected = build_peer_features(**normalized)
    expected_audit = dict(expected['audit'], input_provenance=provenance, new_model_fits=0)
    compare(reader.json(directory/'FEATURE_AUDIT.json'), expected_audit, 'FEATURE_AUDIT')
    feature_errors = {}
    for key in EXTRA_KEYS:
        actual, wanted = prepared[key], expected['arrays'][key]
        require(actual.shape == (122,49) and actual.dtype == wanted.dtype, 'Feature schema '+key)
        if key in RAW_KEYS + HAND_KEYS:
            require(np.array_equal(np.isfinite(actual), np.isfinite(wanted)), 'Feature availability '+key)
            np.testing.assert_allclose(actual, wanted, atol=ATOL, rtol=0, equal_nan=True)
            valid = np.isfinite(wanted)
            feature_errors[key] = float(np.max(np.abs(actual[valid]-wanted[valid]))) if valid.any() else 0.
        else:
            np.testing.assert_array_equal(actual, wanted)
    hands = {name: expected['arrays'][key] for name, key in zip(ARMS[1:], HAND_KEYS)}
    scores, choices, _ = score_oracle(prepared, meta, hands, helper=helper)
    orders = orders_oracle(meta, scores, choices)
    saved_scores, saved_orders = reader.npz(directory/'SCORES.npz'), reader.json(directory/'PRETRAIN_ORDERS.json')
    require(set(saved_scores) == set(saved_orders) == set(ARMS), 'Four prepared arms')
    for name in ARMS:
        require(saved_scores[name].shape == scores[name].shape and saved_scores[name].dtype == scores[name].dtype,
                'Score matrix schema '+name)
        compare(saved_orders[name], orders[name], 'orders/'+name)
        np.testing.assert_allclose(saved_scores[name], scores[name], atol=ATOL, rtol=0)
    assert_incumbent_unchanged(scores[ARMS[0]], saved_scores[ARMS[0]], reader.json(inputs['original_orders']), saved_orders[ARMS[0]], helper=helper)
    for n in (0,1,36,53,60,76,88,122):
        prefix = dict(normalized)
        prefix['meta'] = dict(meta, dates=meta['dates'][:n], symbols=meta['symbols'][:n])
        for key in ('event10', 'matched_recent10', 'valid10', 'anchor_sessions', 'event_ids', 'sector_benchmarks', 'issuer_keys'):
            prefix[key] = normalized[key][:n]
        rebuilt = build_peer_features(**prefix)
        compare(rebuilt['audit']['cells'], expected['audit']['cells'][:n*49], 'Normalized prefix cells')
        for key in EXTRA_KEYS: np.testing.assert_array_equal(rebuilt['arrays'][key], expected['arrays'][key][:n])
        prefix_hands = {name: rebuilt['arrays'][key] for name, key in zip(ARMS[1:], HAND_KEYS)}
        cut_scores, cut_choices, _ = score_oracle(prepared, meta, prefix_hands, n=n, helper=helper)
        for name in ARMS:
            require(cut_scores[name].tobytes() == scores[name][:n].tobytes()
                    and cut_choices[name] == choices[name][:n], 'Score/order prefix '+name)
    for fill in (np.nan, 0.):
        hand = np.full((122,49), np.nan)
        for i, pool in enumerate(meta['symbols']):
            for j, symbol in enumerate(pool):
                if symbol not in meta['etfs']: hand[i,j] = fill
        fallback_scores, fallback_choices, _ = score_oracle(prepared, meta, {name:hand for name in ARMS[1:]}, helper=helper)
        for name in ARMS[1:]:
            require(fallback_scores[name].tobytes() == scores[ARMS[0]].tobytes()
                    and fallback_choices[name] == choices[ARMS[0]], 'Missing/zero exact incumbent fallback')
    artifacts = {str(directory/name): reader.sha(directory/name) for name in (*CACHE_FILES, 'PLAN.json', 'CACHE.json')}
    allowed = dict(pins, **artifacts)
    for path, digest in list(reader.seen.items()):
        require(reader.sha(path) == digest and allowed.get(path) == digest, 'Bound consumed artifact '+path)
    return dict(schema_version=1, passed=True, stage='prepare', plan_sha256=plan_sha,
        cache_sha256=reader.sha(directory/'CACHE.json'), orders_sha256=reader.sha(directory/'PRETRAIN_ORDERS.json'),
        **{key:True for key in CHECKS}, source_hashes=sources, artifact_hashes=dict(reader.seen),
        producer_modules_imported=False, returns_or_nav_computed=False, monthly_label_models_fit=0, new_model_fits=0,
        orders_checked=488, source_cells_checked=5978, raw_prices_independently_verified=True,
        calendar_overlay_independently_verified=True, original_sector_map_independently_reproduced=True,
        all_signal_prefixes_verified=True, original_event10_values_preserved=True, raw_event_prices_reconstructed=False,
        latest_known_event_independently_checked=True, event_sources_checked=len(consumed_events),
        available_stock_cells=int(expected['arrays']['common_available'].sum()),
        max_feature_errors=feature_errors,
        max_score_error=max(float(np.max(np.abs(saved_scores[name]-scores[name]))) for name in ARMS),
        price_parity=parity, limitations=[
            'Frozen event validator reused; repaired event reactions are not independently recomputed from95 raw price files.',
            'Latest-known-event check is limited to the frozen scoped event inventory.',
            'Normalized-source prefix checks do not create additional event history or new event selection.',
            'Inherited proxy-universe, sector mapping and issuer-URL identity limitations remain.',
            'No strategy returns, NAV, monthly models or protected-member scan computed by preparation.'])


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--plan-sha', required=True)
    cli.add_argument('--plan', type=Path, default=DEFAULT_ROOT/STUDY/'PLAN.json')
    cli.add_argument('--prepare-complete', action='store_true')
    cli.add_argument('--output', type=Path, default=DEFAULT_ROOT/STUDY/'INDEPENDENT_PREPARE.json')
    args = cli.parse_args()
    require(args.prepare_complete, 'Explicit registered preparation required')
    result = audit(DEFAULT_ROOT, args.plan, args.plan_sha)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(passed=True, output=str(args.output), sha256=hashlib.sha256(args.output.read_bytes()).hexdigest())))


if __name__ == '__main__': main()
