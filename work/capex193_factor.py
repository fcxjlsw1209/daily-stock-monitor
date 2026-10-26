"""Pure same-report annual cash-after-PP&E features. No file or price I/O.

The caller pins the admitted 44-event dataset, source review, calendar and this
module's frozen146 dependency. Latest release is selected before either gate.
The old NI validation is retained for original admission compatibility; NI is
not part of either new ratio. These values are not GAAP free cash flow.
"""
from collections import Counter
from decimal import InvalidOperation
from fractions import Fraction
import hashlib
import re
import numpy as np
import pandas as pd
import monthly_cash_quality146_factor as old

METHOD = 'current_annual_cash_after_ppe_v1'
DEFINITION = 'current_annual_cash_ppe_outflow_v1'
EXTRA = ('candidate_bounded', 'control_bounded', 'common_available')
DEPENDENCY_SHA256 = {'work/monthly_cash_quality146_factor.py':
    '6f5d819488c7952cb32894cf4b5a4f549c73adb570e949fd47c3c6ba2e561eea'}
CONCEPT_SHA256 = '14e69a47bbf3ea9bbc8d60026e1f5092504d28778f52d622b7560b28bc5c1689'


def _detail(reason, event=None):
    return dict(available=False, reason=reason, method=METHOD,
                current_event_id=None if event is None else event.get('event_id'),
                current_available_from=None if event is None else event.get('available_from'),
                current_period_end=None, prior_period_end=None,
                current_days=None, prior_days=None, currency_key=None,
                candidate_bounded=None, control_bounded=None)


def _gate(admission, definition, name):
    if type(admission) is not dict or set(admission) != {'admitted', 'reasons', 'definition'}:
        return name+'_admission_required'
    if admission['admitted'] is not True:
        return 'current_not_'+name+'_admitted'
    if admission['definition'] != definition:
        return 'wrong_'+name+'_definition'
    if type(admission['reasons']) is not list or admission['reasons']:
        return name+'_admitted_event_has_rejection_reasons'
    return None


def _decimal_fraction(text):
    old._require(old._text(text) and re.fullmatch(r'-?\d+(?:\.\d+)?', text),
                 'Explicit plain decimal CAPEX amount required')
    return old._fraction(text)


def _cash_ppe(event):
    """Validate normalized cell against immutable same-report producer evidence."""
    cell, evidence = event.get('cash_ppe'), event.get('cash_ppe_source_evidence')
    old._require(type(cell) is dict and type(evidence) is dict, 'Cash PP&E cell/evidence required')
    for key in ('event_id', 'symbol', 'release_date', 'available_from', 'source'):
        old._require(evidence.get(key) == event.get(key), 'Cash PP&E evidence metadata differs: '+key)
    old._require(evidence.get('inherited_admitted') is True and evidence.get('extraction_verified') is True,
                 'Verified inherited PP&E extraction required')
    old._require(evidence.get('status') == 'verified' and type(evidence.get('hold_reasons')) is list
                 and not evidence['hold_reasons'], 'Held PP&E evidence cannot pass')
    old._require(not ('field' in evidence and 'capex' in evidence), 'Ambiguous producer cell keys')
    original = evidence.get('field', evidence.get('capex'))
    old._require(type(original) is dict and all(k in cell and cell[k] == v for k,v in original.items()),
                 'Normalized PP&E changed original cell evidence')
    old._require(cell.get('source') == event['source'], 'Cash PP&E source mismatch')
    old._require(cell.get('period') == event['periods']['current'], 'Cash PP&E annual period mismatch')
    old._require(cell.get('available_from') == event['available_from'], 'Cash PP&E availability mismatch')
    old._require(cell.get('fact_admitted') is False, 'Raw CAPEX fact flag must remain false')
    row_currency, statement_currency = cell.get('currency_literal'), cell.get('statement_currency_literal')
    currency_ok = row_currency == '$'
    if row_currency is None and statement_currency == '$':
        loc = cell.get('locator')
        cfo_loc = event['fields']['current']['operating_cash_flow'].get('locator')
        old._require(type(loc) is dict and type(cfo_loc) is dict and loc.get('original_cfo_locator') == cfo_loc
                     and loc.get('format') == cfo_loc.get('format'), 'Cash-flow statement locator binding required')
        if loc.get('format') == 'html':
            currency_ok = (cell.get('currency_scope') == 'same formal cash-flow table; ISO unverified'
                           and type(loc.get('table_index_zero_based')) is int and loc['table_index_zero_based'] >= 0
                           and loc['table_index_zero_based'] == cfo_loc.get('table_index_zero_based')
                           and loc.get('statement_segment') == cfo_loc.get('statement_segment'))
        elif loc.get('format') == 'pdf':
            currency_ok = (cell.get('currency_scope') == 'same formal cash-flow page; ISO unverified'
                           and type(loc.get('page_1based')) is int and loc['page_1based'] > 0
                           and loc['page_1based'] == cfo_loc.get('page_1based'))
    old._require(currency_ok and statement_currency in (None, '$') and cell.get('currency_iso') in (None, 'USD'),
                 'Same-report dollar row or explicit statement literal required')
    signed = _decimal_fraction(cell['signed_value_decimal'])
    positive = _decimal_fraction(cell['positive_outflow_decimal'])
    scale = _decimal_fraction(cell['scale'])
    old._require(scale > 0 and positive >= 0 and signed <= 0 and positive == -signed,
                 'Cash PP&E outflow sign/scale mismatch')
    old._require(scale == old._fraction(event['fields']['current']['operating_cash_flow']['scale']),
                 'Paired cash-flow statement units differ')
    old._require(old._text(cell.get('signed_literal')), 'Signed displayed cash cell required')
    row = cell.get('row_text')
    old._require(type(row) is str and hashlib.sha256(row.encode()).hexdigest() == cell.get('row_text_sha256'),
                 'Cash PP&E row text hash mismatch')
    locator = cell.get('locator')
    old._require(type(locator) is dict and locator.get('period_end') == event['periods']['current']['period_end'],
                 'Cash PP&E locator annual end differs')
    label = cell.get('row_label', locator.get('row_label', ''))
    old._require(type(label) is str and bool(re.search(r'propert\w*.*(?:plant|equipment)', label, re.I)),
                 'Explicit PP&E source label required')
    other_scope = re.sub(r'\bacquisition\s+of\s+property(?:,?\s+plant)?\s+and\s+equipment\b', '', label, flags=re.I)
    old._require(not re.search(r'intangible|acquisit|business|companies|subsidiar|principal|finance lease|net investing|free cash flow',other_scope,re.I),
                 'Mixed or noncash PP&E scope excluded')
    return positive*scale


def evaluate_report(event, signal):
    """One already-selected report. Any old/new failure blocks both arms."""
    old._date(signal)
    if type(event) is not dict:
        return _detail('invalid_event')
    detail = _detail('unavailable', event)
    for name, definition in (('cash_quality', old.DEFINITION), ('cash_ppe', DEFINITION)):
        reason = _gate(event.get(name+'_admission'), definition, name)
        if reason:
            detail['reason'] = reason
            return detail
    # Reuse the exact original timing, identity, period, CFO, NI and asset checks.
    legacy = old.evaluate_report(event, signal)
    for key in ('current_period_end','prior_period_end','current_days','prior_days','currency_key'):
        detail[key] = legacy[key]
    if not legacy['available']:
        detail['reason'] = legacy['reason']
        return detail
    try:
        ppe = _cash_ppe(event)
        cash = old._amount(event['fields']['current']['operating_cash_flow'],event,'current')
        assets_now = old._amount(event['fields']['current']['total_assets'],event,'current')
        assets_prior = old._amount(event['fields']['comparison']['total_assets'],event,'comparison')
        average_assets = (assets_now+assets_prior)/2
        candidate, control = (cash-ppe)/average_assets, cash/average_assets
    except (KeyError, TypeError, ValueError, ArithmeticError, InvalidOperation):
        detail['reason'] = 'invalid_or_missing_cash_ppe_evidence'
        return detail
    detail.update(available=True, reason='available', candidate_bounded=old.bounded(candidate),
                  control_bounded=old.bounded(control))
    return detail


def _cell(events, signal):
    day = old._date(signal)
    known = {}
    for event in events:
        old._require(type(event) is dict and old._text(event.get('event_id')), 'Event identity required')
        available = old._date(event.get('available_from'))
        if available > day:
            continue
        key = event['event_id']
        if key in known and known[key] != event:
            return _detail('conflicting_event_id')
        known[key] = event
    if not known:
        return _detail('current_missing_event')
    latest = max(e['available_from'] for e in known.values())
    selected = [e for e in known.values() if e['available_from'] == latest]
    if len(selected) != 1:
        return _detail('current_ambiguous_latest_event')
    return evaluate_report(selected[0], signal)


def build_features(meta, admitted_dataset):
    """Bounded cash-after-PP&E/CFO arrays on the identical latest-report mask."""
    old._require(type(admitted_dataset) is dict and type(admitted_dataset.get('schema_version')) is int
                 and admitted_dataset['schema_version'] == 1 and admitted_dataset.get('method') == METHOD
                 and admitted_dataset.get('conditional_research_only') is True,
                 'Explicit conditional cash-after-PP&E dataset required')
    events = admitted_dataset.get('events')
    old._require(type(events) is list, 'Complete event list required')
    by_symbol = {}
    for event in events:
        old._require(type(event) is dict and old._text(event.get('symbol')), 'Declared event symbol required')
        by_symbol.setdefault(event['symbol'], []).append(event)
    dates = pd.DatetimeIndex(meta['dates'])
    old._require(dates.tz is None and not dates.hasnans and dates.is_unique and dates.is_monotonic_increasing
                 and dates.equals(dates.normalize()), 'Sorted unique naive signal dates required')
    old._require(len(meta['symbols']) == len(dates), 'Original signal/symbol axes required')
    etfs = meta['etfs']
    old._require(len(etfs) == 17 and len(set(etfs)) == 17 and all(old._text(e) for e in etfs),
                 'Original17 ETFs required')
    arrays = {key: np.full((len(dates),49),np.nan) for key in EXTRA[:-1]}
    arrays['common_available'] = np.zeros((len(dates),49),dtype=bool)
    cells=[]
    for i,(day,names) in enumerate(zip(dates,meta['symbols'])):
        old._require(len(names)==49 and len(set(names))==49 and set(etfs)<=set(names)
                     and all(old._text(name) for name in names), 'Original32 stocks plus17 ETFs required')
        signal=str(day.date())
        for j,symbol in enumerate(names):
            detail=_detail('etf_not_applicable') if symbol in etfs else _cell(by_symbol.get(symbol,[]),signal)
            if detail['available']:
                arrays['common_available'][i,j]=True
                for key in EXTRA[:-1]: arrays[key][i,j]=detail[key]
            cells.append(dict(signal_index=i,pool_index=j,signal_date=signal,symbol=symbol,**detail))
    audit=dict(original_arrays_unchanged=True,original_months=len(dates),original_width=49,
               common_available_cells=int(arrays['common_available'].sum()),
               reasons=dict(Counter(c['reason'] for c in cells)),cells=cells,new_model_fits=0,
               returns_or_nav_computed=False,latest_event_before_validity_filter=True,
               same_current_release_for_all_cells=True,both_admissions_required=True,
               original_ni_gate_preserved=True,net_income_used_in_new_ratios=False,
               comparison_flow_values_used=False,annual_duration_normalization=False,
               original_auxiliary_states_and_core_coefficients_preserved=True,
               source_verification='caller_verified_cash_ppe_source_review_and_complete_hash_closure_required',
               arithmetic_method='exact_scaled_decimal_fraction_ratios_then_single_bounded_float',
               currency_scope='same_report_dollar_literal_assumption_not_ISO_certification',
               interpretation='annual_as_reported_CFO_less_cash_PPE_not_GAAP_FCF')
    return dict(arrays=arrays,audit=audit)
