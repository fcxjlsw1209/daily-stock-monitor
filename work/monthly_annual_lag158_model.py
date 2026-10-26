"""Pure, fixed KO/WMT annual-lag pilot features; no I/O or strategy execution.

The caller must verify and register the source/package/code closure BEFORE
calling ``build`` with real inputs. This module does not establish source,
normal-period, historical-identity or exact-publication admission. It consumes
the reviewed conditional overlay and conservatively retains its blockers.

``meta`` has the original122 ``dates`` and122 lists of49 ``symbols`` plus
``etfs``. ``rows`` is the full5978 overlay list, in signal/candidate order.
``event_metadata`` maps selected IDs to their frozen original records.
``calendar`` has the original ISO ``sessions`` plus an explicit ``closes`` map
from session date to UTC epoch nanoseconds. The caller derives that map from
the pinned full exchange schedule; no weekday/16:00 convention is inferred.
``source_mask`` and ``eligible`` are boolean122x49 matrices. Source true must
agree with the overlay's conditional flag; eligible only narrows that source.

An optional ``n`` computes only the first n months. Shapes are checked globally,
but later row contents, symbols, dates, flags and event records are not read.
Calendar sessions are public schedule metadata, not future research outcomes.
Returned arrays have shape(n,49). NaN means unavailable raw information; zero
rank means neutral score correction and never means a zero-day missing lag.
"""
from bisect import bisect_right
from collections import Counter
from datetime import date, datetime, timezone
import re

import numpy as np

MONTHS, WIDTH, SKIP = 122, 49, 42
DOSE, MAX_ABSOLUTE_CORRECTION = .01, .005
INITIAL_PARTIAL = '2016-08-17'
ISSUERS = {'KO': '0000021344', 'WMT': '0000104169'}
RAW_NAMES = ('negative_current_lag', 'negative_lag_change')
RANK_NAMES = ('current_lag_rank', 'lag_change_rank')
EXTRA = RAW_NAMES + RANK_NAMES + ('common_available',)
MISSING_CONTEXT = 'ACCESSION_SPECIFIC_NORMAL_CONSECUTIVE_ANNUAL_CONTEXT_NOT_ESTABLISHED'
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def iso_day(value):
    """Strict ISO date; booleans, timestamps and relaxed padding are rejected."""
    if type(value) is not str or not re.fullmatch(r'\d{4}-\d{2}-\d{2}', value):
        raise ValueError('Expected YYYY-MM-DD')
    return date.fromisoformat(value)


def timestamp_ns(value):
    if type(value) is not str or not re.fullmatch(
            r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})', value):
        raise ValueError('Expected timezone-aware ISO timestamp')
    stamp = datetime.fromisoformat(value.replace('Z', '+00:00'))
    delta = stamp.astimezone(timezone.utc) - EPOCH
    return ((delta.days * 86400 + delta.seconds) * 1000000 + delta.microseconds) * 1000


def centered_rank(values):
    """Higher is better; average ties, singleton/empty/all-equal gives zero."""
    raw = np.asarray(values)
    require(raw.ndim == 1 and raw.dtype.kind in 'fiu', 'Numeric rank vector required')
    require(np.isfinite(raw).all(), 'Rank population must be finite')
    count = len(raw)
    if count <= 1:
        return np.zeros(count, dtype=np.float64)
    return np.asarray([(np.count_nonzero(raw < value) +
                        (np.count_nonzero(raw == value)-1)/2)/(count-1)-.5
                       for value in raw], dtype=np.float64)


def _calendar(calendar):
    require(type(calendar) is dict and type(calendar.get('sessions')) in (list, tuple),
            'Explicit calendar sessions required')
    sessions = calendar['sessions']
    require(bool(sessions), 'Nonempty calendar required')
    for i, value in enumerate(sessions):
        iso_day(value)
        require(i == 0 or sessions[i-1] < value, 'Sorted unique sessions required')
    closes = calendar.get('closes')
    require(type(closes) is dict, 'Caller-verified close nanoseconds required')
    return sessions, closes


def _close(closes, day):
    value = closes.get(day)
    require(type(value) is int, 'Calendar close must be integer UTC nanoseconds: '+day)
    return value


def _reason_list(value, label):
    require(type(value) is list and all(type(x) is str and x for x in value),
            'Explicit source reason list required: '+label)
    return value


def _cell(overlay, source_on, original_eligible, symbol, signal, position,
          sessions, closes, events):
    row = overlay['original_row']
    reasons = []
    detail = {}
    # All row-level source blockers remain effective even if source_mask is true.
    original_reasons = _reason_list(row.get('reasons'), 'original')
    remaining = _reason_list(overlay.get('remaining_row_reasons'), 'remaining')
    declared = overlay.get('conditional_annual_pair_admitted')
    require(type(declared) is bool, 'Conditional source flag must be boolean')
    require(bool(source_on) == declared, 'Source mask must equal the reviewed overlay flag')
    if not declared:
        reasons.append('SOURCE_NOT_CONDITIONALLY_ADMITTED')
    if overlay.get('body_pair_context_verified') is not True or row.get('tier2') is not True:
        reasons.append('SOURCE_OR_ANNUAL_CONTEXT_NOT_VERIFIED')
    unresolved_original = [r for r in original_reasons if not (
        r == MISSING_CONTEXT and overlay.get('body_pair_context_verified') is True)]
    reasons.extend('SOURCE_BLOCKER:'+r for r in dict.fromkeys(remaining + unresolved_original))
    amendments = _reason_list(row.get('known_amendment_event_ids'), 'known amendments')
    if amendments:
        reasons.append('KNOWN_SELECTED_PERIOD_AMENDMENT')
    if overlay.get('old_admission_flags_changed') is not False:
        reasons.append('ORIGINAL_ADMISSION_FLAGS_NOT_PRESERVED')
    if symbol not in ISSUERS or row.get('is_etf') is not False:
        reasons.append('OUTSIDE_FIXED_KO_WMT_SCOPE')
    elif row.get('cik') != ISSUERS[symbol]:
        reasons.append('ROW_ISSUER_MISMATCH')
    if (overlay.get('body_context_source') not in (None, symbol) or
            (overlay.get('body_pair_context_verified') is True and
             overlay.get('body_context_source') != symbol)):
        reasons.append('BODY_CONTEXT_ISSUER_MISMATCH')
    if not original_eligible:
        reasons.append('ORIGINAL_INELIGIBLE')
    if signal == INITIAL_PARTIAL:
        reasons.append('INITIAL_PARTIAL_EXACT_FALLBACK')

    if position < SKIP:
        reasons.append('INSUFFICIENT_T_MINUS_42_CALENDAR')
    else:
        cutoff = sessions[position-SKIP]
        detail['information_cutoff_date'] = cutoff
        try:
            valid_cutoff = (row.get('information_cutoff_date') == cutoff and
                type(row.get('information_cutoff_ns')) is int and
                row['information_cutoff_ns'] == _close(closes, cutoff) and
                timestamp_ns(row.get('information_cutoff')) == _close(closes, cutoff) and
                timestamp_ns(row.get('original_decision_cutoff')) == _close(closes, signal))
        except (ValueError, TypeError, OverflowError):
            valid_cutoff = False
        if not valid_cutoff:
            reasons.append('ORIGINAL_CUTOFF_MISMATCH')
    if reasons:
        return None, list(dict.fromkeys(reasons)), detail

    ids = row.get('selected_event_ids')
    if (type(ids) is not list or len(ids) != 2 or
            not all(type(key) is str and key for key in ids) or len(set(ids)) != 2):
        return None, ['EXACT_TWO_SELECTED_ORIGINALS_REQUIRED'], detail
    selected = []
    for key in ids:
        event = events.get(key)
        if type(event) is not dict or event.get('event_id') != key:
            return None, ['SELECTED_ORIGINAL_NOT_FOUND'], detail
        if event.get('cik') != ISSUERS[symbol] or event.get('form') != '10-K':
            return None, ['SELECTED_ORIGINAL_IDENTITY_OR_FORM_MISMATCH'], detail
        if (type(event.get('accessionNumber')) is not str or
                not re.fullmatch(r'\d{10}-\d{2}-\d{6}', event['accessionNumber']) or
                type(event.get('primaryDocument')) is not str or not event['primaryDocument']):
            return None, ['SELECTED_ORIGINAL_VERSION_METADATA_INVALID'], detail
        try:
            report, filing = iso_day(event.get('reportDate')), iso_day(event.get('filingDate'))
        except ValueError:
            return None, ['SELECTED_ORIGINAL_DATE_INVALID'], detail
        if filing < report:
            return None, ['NEGATIVE_CALENDAR_LAG'], detail
        if event['filingDate'] < sessions[0]:
            return None, ['FILING_OUTSIDE_CALENDAR_START'], detail
        available = bisect_right(sessions, event['filingDate']) + 1
        if available >= len(sessions) or available > position-SKIP:
            return None, ['SELECTED_ORIGINAL_NOT_AVAILABLE_AT_T_MINUS_42'], detail
        selected.append((report, filing, key, event['accessionNumber'], sessions[available]))
    if selected[0][0] == selected[1][0] or selected[0][3] == selected[1][3]:
        return None, ['SELECTED_ORIGINAL_PERIOD_OR_ACCESSION_COLLISION'], detail
    # Reorder these two IDs only. Never scan the event catalog for an older pair.
    previous, current = sorted(selected, key=lambda item: item[0])
    prior_lag, current_lag = (int((item[1]-item[0]).days) for item in (previous, current))
    detail.update(previous_event_id=previous[2], current_event_id=current[2],
                  previous_available_date=previous[4], current_available_date=current[4],
                  previous_lag_days=prior_lag, current_lag_days=current_lag)
    return (-current_lag, prior_lag-current_lag), [], detail


def build(meta, rows, event_metadata, source_mask, eligible, calendar, n=None):
    """Build only numerical features of an already reviewed/registered pilot.

    Grid misalignment or a forged source mask raises ValueError. Invalid selected
    originals, unsupported identities, source blockers and cutoff failures stay
    unavailable with explicit per-cell reasons. No failed latest is replaced.
    """
    count = MONTHS if n is None else n
    require(type(count) is int and 0 <= count <= MONTHS, 'Prefix n must be integer0..122')
    require(type(meta) is dict and len(meta.get('dates', ())) == MONTHS and
            len(meta.get('symbols', ())) == MONTHS, 'Original122-month metadata required')
    etfs = meta.get('etfs')
    require(type(etfs) in (list, tuple) and all(type(name) is str and name for name in etfs)
            and len(set(etfs)) == len(etfs), 'Original ETF identity list required')
    require(type(rows) in (list, tuple) and len(rows) == MONTHS*WIDTH, 'Full5978 original rows required')
    require(type(event_metadata) is dict, 'Selected-event metadata map required')
    source_mask, eligible = np.asarray(source_mask), np.asarray(eligible)
    for matrix in (source_mask, eligible):
        require(matrix.shape == (MONTHS, WIDTH) and matrix.dtype == bool,
                'Original122x49 boolean masks required')
    sessions, closes = _calendar(calendar)
    positions = {day: i for i, day in enumerate(sessions)}
    arrays = {name: np.full((count, WIDTH), np.nan, dtype=np.float64) for name in RAW_NAMES}
    arrays.update({name: np.zeros((count, WIDTH), dtype=np.float64) for name in RANK_NAMES})
    arrays['common_available'] = np.zeros((count, WIDTH), dtype=bool)
    cells, months = [], []
    prior_signal = None
    for i in range(count):
        signal, names = meta['dates'][i], meta['symbols'][i]
        iso_day(signal)
        require(signal in positions and (prior_signal is None or prior_signal < signal),
                'Ordered original signal sessions required')
        prior_signal = signal
        require(type(names) in (list, tuple) and len(names) == WIDTH and
                all(type(name) is str and name for name in names) and len(set(names)) == WIDTH,
                'Exactly49 unique original candidate names per month')
        for j, symbol in enumerate(names):
            overlay = rows[i*WIDTH+j]
            require(type(overlay) is dict and type(overlay.get('original_row')) is dict,
                    'Original overlay row required')
            row = overlay['original_row']
            require(type(row.get('signal_index')) is int and row['signal_index'] == i and
                    type(row.get('candidate_index')) is int and row['candidate_index'] == j and
                    row.get('symbol') == symbol and row.get('signal_date') == signal,
                    'Original signal/candidate/symbol/date alignment mismatch')
            require(type(row.get('is_etf')) is bool and row['is_etf'] == (symbol in etfs),
                    'Original ETF identity alignment mismatch')
            values, reasons, detail = _cell(overlay, source_mask[i,j], eligible[i,j],
                symbol, signal, positions[signal], sessions, closes, event_metadata)
            if values is not None:
                arrays['common_available'][i,j] = True
                for name, value in zip(RAW_NAMES, values):
                    arrays[name][i,j] = value
            cells.append(dict(signal_index=i, candidate_index=j, symbol=symbol,
                signal_date=signal, source_conditionally_admitted=bool(source_mask[i,j]),
                original_eligible=bool(eligible[i,j]), available=values is not None,
                reasons=reasons, **detail))
        mask = arrays['common_available'][i]
        for raw_name, rank_name in zip(RAW_NAMES, RANK_NAMES):
            arrays[rank_name][i,mask] = centered_rank(arrays[raw_name][i,mask])
        months.append(dict(signal_index=i, signal_date=signal, common_candidates=int(mask.sum()),
            current_rank_nonzero=int(np.count_nonzero(arrays[RANK_NAMES[0]][i])),
            change_rank_nonzero=int(np.count_nonzero(arrays[RANK_NAMES[1]][i])),
            ranks_identical=bool(np.array_equal(arrays[RANK_NAMES[0]][i], arrays[RANK_NAMES[1]][i]))))
    for name in RAW_NAMES:
        require(np.array_equal(np.isfinite(arrays[name]), arrays['common_available']), 'Common raw mask invariant')
    for name in RANK_NAMES:
        require(np.isfinite(arrays[name]).all() and
                not np.any(arrays[name][~arrays['common_available']]) and
                np.all(np.abs(DOSE*arrays[name]) <= MAX_ABSOLUTE_CORRECTION), 'Fixed rank/dose invariant')
    return dict(arrays=arrays, audit=dict(rows=count*WIDTH, months=count,
        usable_rows=int(arrays['common_available'].sum()),
        source_conditionally_admitted_rows=int(source_mask[:count].sum()),
        lag_pairs_evaluated=int(arrays['common_available'].sum()),
        negative_current_lag_definition='-(filingDate_current-reportDate_current).days',
        negative_lag_change_definition='previous_lag_days-current_lag_days',
        rank_definition='(average_rank1-1)/(n-1)-0.5; n<=1 gives0; rank only common original-eligible mask',
        dose=DOSE, max_absolute_score_correction=MAX_ABSOLUTE_CORRECTION,
        fixed_issuers=dict(ISSUERS), source_admission_changed=False,
        exact_acceptance_certified=False, historical_price_identity_certified=False,
        numerical_features_computed=True, strategy_returns_or_nav_computed=False,
        new_predictive_model_fits=0, nuisance_estimates=0,
        reason_counts=dict(Counter(reason for cell in cells for reason in cell['reasons'])),
        informative_rank_months=sum(m['current_rank_nonzero']>0 or m['change_rank_nonzero']>0 for m in months),
        cells=cells, monthly=months))
