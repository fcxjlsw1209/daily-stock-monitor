"""Pure current-year cash quality features; no I/O, training or portfolio work.

The caller verifies the new field-specific admission and its complete source
closure. Raw historical admission flags remain untouched. All three features
use the same four cells, latest known event, and validity mask.
"""
from collections import Counter
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import re

import numpy as np
import pandas as pd

METHOD = 'current_annual_cash_quality_v1'
DEFINITION = 'consolidated_gaap_cfo_net_income_average_assets_v1'
EXTRA = ('earnings_bounded', 'cash_flow_bounded', 'low_accrual_bounded', 'common_available')
SYMBOLS = frozenset(('AAPL', 'AMD', 'MSFT', 'NVDA'))
SIDES = frozenset(('current', 'comparison'))
FIELDS = frozenset(('operating_cash_flow', 'net_income', 'total_assets'))
REQUIRED = (('current', 'operating_cash_flow'), ('current', 'net_income'),
            ('current', 'total_assets'), ('comparison', 'total_assets'))
MIN_ANNUAL_DAYS, MAX_ANNUAL_DAYS, MAX_AGE_DAYS = 330, 400, 550


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _text(value):
    return type(value) is str and bool(value) and value.strip() == value


def _date(value):
    _require(type(value) is str, 'ISO date string required')
    result = date.fromisoformat(value)
    _require(result.isoformat() == value, 'Canonical ISO date required')
    return result


def _fraction(value):
    _require(_text(value) and re.fullmatch(
        r'-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?', value),
        'Explicit finite decimal string required')
    number = Decimal(value)
    _require(number.is_finite(), 'Finite decimal required')
    return Fraction(number)


def _period(value):
    if type(value) is not dict:
        return None, 'invalid_period_dates'
    try:
        start, end = _date(value['period_start']), _date(value['period_end'])
    except (KeyError, TypeError, ValueError):
        return None, 'invalid_period_dates'
    days = (end-start).days+1
    if not MIN_ANNUAL_DAYS <= days <= MAX_ANNUAL_DAYS:
        return None, 'nonannual_duration'
    if type(value.get('days_inclusive')) is not int or value['days_inclusive'] != days:
        return None, 'duration_mismatch'
    return dict(start=start, end=end, days=days), None


def _detail(reason, event=None):
    return dict(
        available=False, reason=reason, method=METHOD,
        current_event_id=None if event is None else event.get('event_id'),
        current_available_from=None if event is None else event.get('available_from'),
        current_period_end=None, prior_period_end=None,
        current_days=None, prior_days=None, currency_key=None,
        earnings_bounded=None, cash_flow_bounded=None, low_accrual_bounded=None)


def _amount(cell, event, side):
    _require(type(cell) is dict, 'Required cell missing')
    _require(cell.get('source') == event['source'], 'Cell source mismatch')
    _require(cell.get('period') == event['periods'][side], 'Cell period mismatch')
    _require(cell.get('available_from') == event['available_from'], 'Cell availability mismatch')
    _require(cell.get('currency_literal') == '$', 'Same-report dollar literal required')
    value, scale = _fraction(cell['value_decimal']), _fraction(cell['scale'])
    _require(scale > 0, 'Positive declared scale required')
    return value*scale


def bounded(value):
    """Exact rational h(x), rounded only once when producing the float array."""
    _require(type(value) is Fraction, 'Exact rational input required')
    return float(value/(1+abs(value)))


def evaluate_report(event, signal):
    """Evaluate one already selected release under the explicit new admission."""
    day = _date(signal)
    detail = _detail('unavailable', event)

    def fail(reason):
        detail['reason'] = reason
        return detail

    admission = event.get('cash_quality_admission')
    if type(admission) is not dict or set(admission) != {'admitted', 'reasons', 'definition'}:
        return fail('cash_quality_admission_required')
    if admission['admitted'] is not True:
        return fail('current_not_cash_quality_admitted')
    if admission['definition'] != DEFINITION:
        return fail('wrong_cash_quality_definition')
    if type(admission['reasons']) is not list or admission['reasons']:
        return fail('admitted_event_has_rejection_reasons')
    if type(event.get('inherited_rejection_codes')) is not list or event['inherited_rejection_codes']:
        return fail('inherited_rejection_not_cleared')
    if not all(_text(event.get(k)) for k in ('event_id', 'symbol')) or event['symbol'] not in SYMBOLS:
        return fail('unsupported_or_missing_identity')
    identity = event.get('identity')
    if type(identity) is not dict or not all(_text(identity.get(k)) for k in ('issuer_id', 'security_id')):
        return fail('missing_identity')
    if identity['security_id'] != event['symbol']:
        return fail('security_symbol_mismatch')
    source = event.get('source')
    if (type(source) is not dict or not _text(source.get('path')) or
            type(source.get('sha256')) is not str or
            not re.fullmatch('[0-9a-f]{64}', source['sha256'])):
        return fail('missing_source_reference')
    try:
        available, released = _date(event['available_from']), _date(event['release_date'])
        identity_start = _date(identity['identity_valid_from'])
        identity_end = _date(identity['identity_valid_through'])
    except (KeyError, TypeError, ValueError):
        return fail('invalid_dates_or_identity_bounds')
    if 'current_release_available_from' in event and event['current_release_available_from'] != event['available_from']:
        return fail('current_release_availability_mismatch')
    if available > day:
        return fail('report_not_yet_available')
    if released >= available:
        return fail('availability_not_after_release')
    if identity_start != available:
        return fail('identity_start_not_current_availability')
    periods, fields = event.get('periods'), event.get('fields')
    if type(periods) is not dict or set(periods) != SIDES:
        return fail('two_annual_periods_required')
    if (type(fields) is not dict or set(fields) != SIDES or
            any(type(fields[s]) is not dict or set(fields[s]) != FIELDS for s in SIDES)):
        return fail('raw_field_schema_required')
    cur, reason = _period(periods['current'])
    if reason:
        return fail('current_'+reason)
    prev, reason = _period(periods['comparison'])
    if reason:
        return fail('comparison_'+reason)
    detail.update(current_period_end=cur['end'].isoformat(), prior_period_end=prev['end'].isoformat(),
                  current_days=cur['days'], prior_days=prev['days'],
                  currency_key='same-report-dollar:'+event['event_id'])
    if cur['end'] > released:
        return fail('release_before_period_end')
    if prev['end']+timedelta(days=1) != cur['start']:
        return fail('noncontiguous_annual_periods')
    if identity_end != cur['end']+timedelta(days=MAX_AGE_DAYS):
        return fail('identity_end_not_causal_expiry')
    if (day-cur['end']).days > MAX_AGE_DAYS:
        return fail('stale_current_report')
    if not identity_start <= day <= identity_end:
        return fail('identity_outside_interval')
    values = {}
    for side, field in REQUIRED:
        cell = fields[side][field]
        if cell is None:
            return fail('missing_'+side+'_'+field)
        try:
            values[side, field] = _amount(cell, event, side)
        except (KeyError, TypeError, ValueError, ArithmeticError, InvalidOperation):
            return fail('invalid_'+side+'_'+field)
    current_assets = values['current', 'total_assets']
    prior_assets = values['comparison', 'total_assets']
    if current_assets <= 0 or prior_assets <= 0:
        return fail('nonpositive_asset_endpoint')
    average_assets = (current_assets+prior_assets)/2
    earnings = values['current', 'net_income']/average_assets
    cash = values['current', 'operating_cash_flow']/average_assets
    detail.update(available=True, reason='available', earnings_bounded=bounded(earnings),
                  cash_flow_bounded=bounded(cash), low_accrual_bounded=bounded(cash-earnings))
    return detail


def _cell(events, signal):
    day = _date(signal)
    known = {}
    for event in events:
        _require(type(event) is dict and _text(event.get('event_id')), 'Event identity required')
        available = _date(event.get('available_from'))
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
    """Return three bounded arrays and their one common availability mask."""
    _require(type(admitted_dataset) is dict and type(admitted_dataset.get('schema_version')) is int
             and admitted_dataset['schema_version'] == 1 and admitted_dataset.get('method') == METHOD,
             'Explicit new cash-quality dataset schema required')
    events = admitted_dataset.get('events')
    _require(type(events) is list, 'Complete event list required')
    by_symbol = {}
    for event in events:
        _require(type(event) is dict and _text(event.get('symbol')), 'Declared event symbol required')
        by_symbol.setdefault(event['symbol'], []).append(event)
    dates = pd.DatetimeIndex(meta['dates'])
    _require(dates.tz is None and not dates.hasnans and dates.is_unique and dates.is_monotonic_increasing
             and dates.equals(dates.normalize()), 'Sorted unique naive signal dates required')
    _require(len(meta['symbols']) == len(dates), 'Original signal/symbol axes required')
    etfs = meta['etfs']
    _require(len(etfs) == 17 and len(set(etfs)) == 17 and all(_text(e) for e in etfs), 'Original17 ETFs required')
    arrays = {key: np.full((len(dates), 49), np.nan) for key in EXTRA[:-1]}
    arrays['common_available'] = np.zeros((len(dates), 49), dtype=bool)
    cells = []
    for i, (day, names) in enumerate(zip(dates, meta['symbols'])):
        _require(len(names) == 49 and len(set(names)) == 49 and set(etfs) <= set(names)
                 and all(_text(name) for name in names), 'Original32 stocks plus17 ETFs required')
        signal = str(day.date())
        for j, symbol in enumerate(names):
            detail = _detail('etf_not_applicable') if symbol in etfs else _cell(by_symbol.get(symbol, []), signal)
            if detail['available']:
                arrays['common_available'][i, j] = True
                for key in EXTRA[:-1]:
                    arrays[key][i, j] = detail[key]
            cells.append(dict(signal_index=i, pool_index=j, signal_date=signal, symbol=symbol, **detail))
    audit = dict(
        original_arrays_unchanged=True, original_months=len(dates), original_width=49,
        new_model_fits=0, returns_or_nav_computed=False,
        original_auxiliary_states_and_core_coefficients_preserved=True,
        common_available_cells=int(arrays['common_available'].sum()),
        reasons=dict(Counter(cell['reason'] for cell in cells)),
        source_verification='caller_verified_new_field_specific_admission_and_source_hashes_required',
        arithmetic_method='exact_fraction_scaled_amounts_average_assets_ratios_then_float64_bounded_mapping',
        annual_duration_normalization=False, comparison_flow_values_used=False,
        latest_event_before_validity_filter=True, same_current_release_for_all_cells=True,
        currency_scope='same_report_dollar_literal_assumption_not_ISO_certification', cells=cells)
    return dict(arrays=arrays, audit=audit)
