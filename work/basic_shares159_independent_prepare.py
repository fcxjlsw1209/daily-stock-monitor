"""Draft independent round159 preparation oracles; no producer imports or data I/O.

All financial inputs must be supplied by a subsequently registered host. There
is intentionally no CLI that loads real inputs. The only lazy source read loads
the old, hash-pinned independent Local67 selector/accounting helper.
"""
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import hashlib
import importlib.util
from pathlib import Path
import re
import sys

import numpy as np

NAMES = ('incumbent_local67', 'centered_negative_basic_share_growth_magnitude',
         'centered_negative_basic_share_growth_rank')
HANDS = ('basic_share_magnitude', 'basic_share_rank')
EXTRA = HANDS + ('common_available',)
ISSUERS = ('AAPL', 'AMD', 'MSFT', 'NVDA')
INITIAL = '2016-08-17'
HELPER = 'work/monthly_ridge_support70_result_review.py'
HELPER_SHA = 'd0f053881cb94e49e7fff9eaaeaa9e260f0b3c66419f87819391c309cb7cb8ef'


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def independent_imports():
    forbidden = ('basic_shares159_runner_draft', 'after158_basic_shares_kernel',
                 'monthly_basic_shares159_runner', 'basic_shares159_lifecycle')
    require(not any(name in sys.modules for name in forbidden), 'New producer/kernel imported')


def old_helper():
    path = Path(__file__).resolve().parents[1]/HELPER
    require(hashlib.sha256(path.read_bytes()).hexdigest() == HELPER_SHA, 'Old independent helper changed')
    spec = importlib.util.spec_from_file_location('basic159_old_independent', path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


def ns(text):
    require(type(text) is str and re.fullmatch(
        r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})', text),
        'Explicit timezone timestamp required')
    value = datetime.fromisoformat(text.replace('Z', '+00:00')).astimezone(timezone.utc)
    delta = value-datetime(1970, 1, 1, tzinfo=timezone.utc)
    return (delta.days*86400+delta.seconds)*10**9+delta.microseconds*1000


def day(text):
    value = date.fromisoformat(text)
    require(value.isoformat() == text, 'Canonical ISO day required')
    return value


def decimal_parts(text):
    """Extract integer coefficient/exponent rather than using Decimal division."""
    require(type(text) is str and text == text.strip(), 'Original exact decimal string')
    try:
        value = Decimal(text)
    except InvalidOperation:
        raise ValueError('Invalid share count') from None
    require(value.is_finite() and value > 0, 'Strictly positive finite share count')
    sign, digits, exponent = value.as_tuple()
    coefficient = 0
    for digit in digits:
        coefficient = coefficient*10+digit
    return -coefficient if sign else coefficient, exponent


def negative_change(current, previous):
    a, e = decimal_parts(current)
    b, f = decimal_parts(previous)
    common = min(e, f)
    a *= 10**(e-common)
    b *= 10**(f-common)
    return Fraction(b-a, b)


def hands_oracle(values):
    """Exact 5% bounded centering and rank groups, conversion only at output."""
    available = [j for j, x in enumerate(values) if x is not None]
    require(all(type(values[j]) is Fraction for j in available), 'Exact rational population')
    magnitude, rank = np.zeros(len(values)), np.zeros(len(values))
    n = len(available)
    if n < 2 or len({values[j] for j in available}) == 1:
        return magnitude, rank
    bounded = {j: 20*values[j]/(1+20*abs(values[j])) for j in available}
    center = sum(bounded.values(), Fraction(0))/n
    for j in available:
        magnitude[j] = float((bounded[j]-center)/2)
    order = sorted(available, key=lambda j: values[j])
    lo = 0
    while lo < n:
        hi = lo+1
        while hi < n and values[order[hi]] == values[order[lo]]:
            hi += 1
        centered = Fraction(lo+hi-1, n-1)-1
        for j in order[lo:hi]:
            rank[j] = float(centered)
        lo = hi
    return magnitude, rank


def calendar_positions(calendar):
    sessions, closes = calendar['sessions'], calendar['closes']
    require(type(sessions) is list and sessions == sorted(set(sessions)), 'Ordered unique sessions')
    for session in sessions:
        day(session)
        require(type(closes[session]) is int, 'Close must be integer epoch nanoseconds')
    return {s: i for i, s in enumerate(sessions)}


def second_close(release, calendar):
    day(release)
    require(release >= calendar['sessions'][0], 'Source precedes complete calendar')
    subsequent = (s for s in calendar['sessions'] if s > release)
    next(subsequent, None)
    session = next(subsequent, None)
    require(session is not None, 'Insufficient later sessions')
    return calendar['closes'][session]


def latest_event(events, symbol, cutoff_ns):
    known = [e for e in events if e['symbol'] == symbol and ns(e['known_at']) <= cutoff_ns]
    if not known:
        return None
    latest = max(e['period_end'] for e in known)
    chosen = [e for e in known if e['period_end'] == latest]
    require(len(chosen) == 1, 'Ambiguous latest original annual event')
    return chosen[0]


def source_status(event, symbol, cutoff_day, cutoff_ns, calendar):
    """Checks latest event before counts; no earlier-report fallback is possible."""
    if event is None:
        return 'no_known_original'
    original = event['original_event']
    identity = event['identity']
    require(event['event_id'] == original['event_id'] and event['symbol'] == original['symbol'] == symbol,
            'Original event identity mismatch')
    require(identity == original['identity'], 'Original security identity changed')
    require(ns(event['known_at']) == second_close(original['release_date'], calendar), 'Knowledge clock mismatch')
    filed = event['selected_gaap_contexts']['selected_filed']
    required_usable = max(ns(event['known_at']), second_close(filed, calendar)) if filed else ns(event['known_at'])
    require(ns(event['usable_at']) == required_usable, 'Corroboration clock mismatch')
    if required_usable > cutoff_ns:
        return 'latest_not_yet_usable'
    if event['conditional_source_ready'] is not True or event['unresolved_reasons'] != []:
        return 'latest_source_not_conditionally_ready'
    if event['same_report_annual_pair'] is not True or event['same_share_basis'] is not True:
        return 'source_pair_or_basis_unverified'
    require(event['strict_source_admitted'] is False and event['old_admission_flags_changed'] is False,
            'Conditional source scope changed')
    require(event['inherited_holds'] == original['inherited_holds'] and
            not event['inherited_holds']['inherited_rejection_codes'], 'Inherited source hold was removed')
    if identity.get('security_id') != symbol or not identity.get('issuer_id') or not (
            day(identity['identity_valid_from']) <= day(cutoff_day) <= day(identity['identity_valid_through'])):
        return 'identity_not_valid_at_cutoff'
    age = (day(cutoff_day)-day(event['period_end'])).days
    if age < 0 or age > 550:
        return 'latest_period_outside_age_window'
    periods = original['periods']
    for side in ('current', 'comparison'):
        period = periods[side]
        days = (day(period['period_end'])-day(period['period_start'])).days+1
        require(330 <= days <= 400 and days == period['days_inclusive'], 'Original annual duration')
    require((day(periods['current']['period_start'])-day(periods['comparison']['period_end'])).days == 1,
            'Original annual periods not consecutive')
    require(event['period_end'] == periods['current']['period_end'] and
            event['prior_period_end'] == periods['comparison']['period_end'], 'Original annual endpoints')
    require(event['current_shares'] == original['cells']['current']['value_in_shares'] and
            event['prior_shares'] == original['cells']['comparison']['value_in_shares'], 'Raw count substitution')
    try:
        decimal_parts(event['current_shares']); decimal_parts(event['prior_shares'])
    except ValueError:
        return 'invalid_share_denominator'
    return None


def features_oracle(meta, events, source_rows, original_rows, eligible, calendar, n=None):
    count = 122 if n is None else n
    require(type(count) is int and 0 <= count <= 122, 'Valid prefix')
    require(len(meta['dates']) == len(meta['symbols']) == 122 and len(source_rows) == len(original_rows) == 5978,
            'Full original 122x49 source axis')
    require(len(events) == len({e['event_id'] for e in events}) == 44, 'Fixed unique 44-event axis')
    require(set(e['symbol'] for e in events) == set(ISSUERS), 'Fixed four source issuers')
    require(isinstance(eligible, np.ndarray) and eligible.dtype == bool and eligible.shape == (122, 49),
            'Original boolean eligible array')
    require(len(meta['etfs']) == len(set(meta['etfs'])) == 17, 'Original 17 ETF axis')
    positions = calendar_positions(calendar)
    arrays = {key: np.zeros((count, 49), dtype=bool if key == 'common_available' else float) for key in EXTRA}
    cells = []
    for i in range(count):
        signal, names = meta['dates'][i], meta['symbols'][i]
        require(len(names) == len(set(names)) == 49 and set(meta['etfs']).issubset(names), '49 unique original candidates')
        require(signal in positions and positions[signal] >= 42, 'Complete t-42 calendar')
        cut_day = calendar['sessions'][positions[signal]-42]
        cut_ns = calendar['closes'][cut_day]
        values, seen_issuers = [None]*49, set()
        for j, symbol in enumerate(names):
            source, old = source_rows[49*i+j], original_rows[49*i+j]
            require((source['signal_index'], source['candidate_index'], source['symbol']) == (i, j, symbol),
                    'Source row axis mismatch')
            inherited = source['original_availability_row']
            require(inherited['original_row'] == old, 'Original full source row changed')
            require((old['signal_index'], old['candidate_index'], old['symbol'], old['decision_date']) == (i, j, symbol, signal),
                    'Original row axis mismatch')
            require(ns(old['decision_cutoff']) == old['decision_cutoff_ns'] == calendar['closes'][signal],
                    'Original decision close mismatch')
            require(ns(source['cutoff']) == ns(inherited['cutoff']) == inherited['cutoff_ns'] == cut_ns and
                    inherited['cutoff_date'] == cut_day, 'Exact t-42 source close mismatch')
            require(source['strict_source_admitted'] is False and source['old_admission_flags_changed'] is False,
                    'Source scope flags changed')
            event = latest_event(events, symbol, cut_ns) if symbol in ISSUERS and symbol not in meta['etfs'] else None
            require(source['selected_event_id'] == (event['event_id'] if event else None), 'Latest event or no-fallback mismatch')
            reason = source_status(event, symbol, cut_day, cut_ns, calendar) if event else 'no_known_original'
            ready = reason is None and signal != INITIAL
            require(type(source['conditional_source_ready']) is bool and source['conditional_source_ready'] == ready,
                    'Independent source readiness mismatch')
            if ready:
                issuer = event['identity']['issuer_id']
                require(issuer not in seen_issuers, 'Duplicate issuer in source population')
                seen_issuers.add(issuer)
            available = ready and bool(eligible[i, j])
            if available:
                values[j] = negative_change(event['current_shares'], event['prior_shares'])
                arrays['common_available'][i, j] = True
            raw = values[j]
            model_event = event if bool(eligible[i,j]) and signal != INITIAL else None
            cells.append(dict(signal_index=i, candidate_index=j, date=signal, symbol=symbol,
                event_id=model_event['event_id'] if model_event else None, available=available,
                raw_exact=None if raw is None else dict(numerator=str(raw.numerator), denominator=str(raw.denominator))))
        arrays[HANDS[0]][i], arrays[HANDS[1]][i] = hands_oracle(values)
    return dict(arrays=arrays, cells=cells)


def verify_original_arrays(original, prepared):
    require(len(original) == 27 and set(prepared) == set(original) | set(EXTRA), 'Original27 plus exact3 arrays')
    for key, value in original.items():
        other = prepared[key]
        require(value.shape == other.shape and value.dtype == other.dtype and value.tobytes() == other.tobytes(),
                'Original array changed: '+key)
    require(original['baseline'].shape == (122, 49), 'Original full array axis')


def score_oracle(arrays, meta, hands, n=None):
    helper = old_helper()
    state = helper.original_prefix(arrays, meta, n)
    shape, count = state['local'].shape, state['n']
    scores, choices, deltas, episodes = {}, {}, {}, {}
    for index, name in enumerate(NAMES):
        pre, delta = state['local'].copy(), np.zeros(shape)
        if index:
            hand = np.asarray(hands[HANDS[index-1]])[:count]
            require(hand.shape == shape and hand.dtype.kind == 'f' and np.isfinite(hand).all() and
                    (np.abs(hand) <= 1).all(), 'Bounded hand in [-1,1]')
            for i, symbols in enumerate(meta['symbols'][:count]):
                require(abs(float(hand[i].sum())) <= 1e-12, 'Centered hand')
                require(all(hand[i, j] == 0 for j, s in enumerate(symbols) if s not in ISSUERS or s in meta['etfs']),
                        'Correction outside fixed stock source scope')
                if meta['dates'][i] == INITIAL:
                    require(not hand[i].any(), 'Initial partial correction')
            require((hand[~state['eligible']] == 0).all(), 'Ineligible hand must be zero before scoring')
            active = hand != 0
            delta[active] = .01*hand[active]
            pre[active] += delta[active]
        selected, final, records = helper.episode(pre, state)
        scores[name], choices[name], deltas[name], episodes[name] = final, selected, delta, records
    return dict(scores=scores, choices=choices, deltas=deltas, episodes=episodes)


def order_oracle(meta, result):
    return {name: [dict(date=meta['dates'][i], signal_date=meta['dates'][i], end=meta['ends'][i],
        symbols=[meta['symbols'][i][j] for j in chosen],
        scores=[float(result['scores'][name][i, j]) for j in chosen])
        for i, chosen in enumerate(result['choices'][name])] for name in NAMES}


def verify_feature_output(expected, prepared, feature_audit):
    for key in EXTRA:
        actual, wanted = prepared[key], expected['arrays'][key]
        require(actual.shape == wanted.shape and actual.dtype == wanted.dtype, 'Feature schema '+key)
        if key == 'common_available':
            np.testing.assert_array_equal(actual, wanted)
        else:
            # Exact ordering/ties are checked via raw fractions. The producer
            # renders average ranks with binary arithmetic; allow final-ULP noise.
            np.testing.assert_allclose(actual, wanted, atol=1e-15, rtol=0)
    require(len(feature_audit['cells']) == len(expected['cells']), 'Complete feature audit cells')
    for actual, wanted in zip(feature_audit['cells'], expected['cells']):
        for key in ('signal_index', 'candidate_index', 'date', 'symbol', 'available', 'raw_exact', 'event_id'):
            require(actual[key] == wanted[key], 'Feature cell '+key)
    require(feature_audit['new_predictive_model_fits'] == feature_audit['nuisance_estimates'] == 0,
            'No predictive/nuisance model fits')


def verify_prefixes(arrays, meta, hands, full):
    for n in range(123):
        prefix = score_oracle(arrays, meta, hands, n)
        for name in NAMES:
            require(prefix['choices'][name] == full['choices'][name][:n] and
                    prefix['scores'][name].tobytes() == full['scores'][name][:n].tobytes(), 'Prefix score/order mismatch')
    return 123


def audit_registered(*args, **kwargs):
    raise RuntimeError('Draft only: registered PLAN/hash/sourcegate v2/price-parity adapter not yet connected')
