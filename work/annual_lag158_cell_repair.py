"""Versioned repair for two unadmitted source rows with no amendment field.

The frozen cell algorithm is copied below. Its only change is the guarded
missing-key branch. No source row, admission flag, rank, dose or date is changed.
"""
from bisect import bisect_right
import re

import monthly_annual_lag158_model as original

require = original.require
_reason_list, _close = original._reason_list, original._close
iso_day, timestamp_ns = original.iso_day, original.timestamp_ns
MISSING_CONTEXT, ISSUERS = original.MISSING_CONTEXT, original.ISSUERS
INITIAL_PARTIAL, SKIP = original.INITIAL_PARTIAL, original.SKIP
ABSENT_REASON = 'AMENDMENT_METADATA_ABSENT_ON_UNADMITTED_ROW'
ETF_REASON = 'ETF_PRESERVED_NO_ANNUAL_FILING_FEATURE'
UNAVAILABLE_REASON = 'SOURCE_EXTENT_OR_CURRENT_CIK_UNAVAILABLE'


def allows_absent_amendments(overlay, source_on):
    """Recognize only the two frozen feasibility early-return row shapes."""
    row = overlay['original_row']
    reasons = row.get('reasons')
    branch = ((reasons == [ETF_REASON] and row.get('is_etf') is True) or
              (reasons == [UNAVAILABLE_REASON] and row.get('is_etf') is False))
    return (not bool(source_on) and branch and
            overlay.get('conditional_annual_pair_admitted') is False and
            overlay.get('body_pair_context_verified') is False and
            'body_context_source' in overlay and overlay['body_context_source'] is None and
            overlay.get('old_admission_flags_changed') is False and
            all(key in row and row[key] is None for key in ('selected_snapshot', 'cik')) and
            all(type(row.get(key)) is list and not row[key]
                for key in ('selected_event_ids', 'period_hint_ids')) and
            all(row.get(key) is False for key in ('tier1', 'tier2', 'tier3', 'normal_period_admission')))


def repaired_cell(overlay, source_on, original_eligible, symbol, signal, position,
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
    if 'known_amendment_event_ids' not in row:
        require(allows_absent_amendments(overlay, source_on),
                'Missing amendment metadata outside frozen unadmitted early-return rows')
        reasons.append(ABSENT_REASON)
    else:
        amendments = _reason_list(row['known_amendment_event_ids'], 'known amendments')
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
