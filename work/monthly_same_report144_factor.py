"""Pure same-current-release annual comparisons; no I/O, models or returns.

Both fiscal years are known only on the current report's availability date.
A rejected latest report blocks fallback. Source-only admission is separately
verified by the caller; this module never upgrades its research assumptions.
"""
from collections import Counter
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation, Inexact, Rounded, localcontext, MAX_EMAX, MIN_EMIN
import numpy as np
import pandas as pd

EXTRA = ('gross_profit_growth_sign', 'gross_profitability_improvement_sign', 'common_available')
METHOD = 'same_current_release_annual_comparison'
MIN_ANNUAL_DAYS, MAX_ANNUAL_DAYS, MAX_AGE_DAYS = 330, 400, 550


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _date(value):
    if not isinstance(value, str):
        raise ValueError('ISO date string required')
    parsed = date.fromisoformat(value)
    if parsed.isoformat() != value:
        raise ValueError('Canonical ISO date required')
    return parsed


def _text(value):
    return isinstance(value, str) and bool(value) and value.strip() == value


def _decimal(value):
    if not _text(value):
        raise ValueError('Explicit decimal string required')
    result = Decimal(value)
    if not result.is_finite():
        raise ValueError('Finite decimal required')
    return result


def _product(*values):
    with localcontext() as ctx:
        ctx.prec = max(28, sum(len(v.as_tuple().digits) for v in values) + 2)
        ctx.Emax, ctx.Emin = MAX_EMAX, MIN_EMIN
        ctx.traps[Inexact] = ctx.traps[Rounded] = True
        result = Decimal(1)
        for value in values:
            result *= value
        if not result.is_finite():
            raise ValueError('Nonfinite exact product')
        return result


def _amount(value):
    if not isinstance(value, dict) or value.get('currency_literal') != '$':
        raise ValueError('Same-report literal dollar amount required')
    amount, scale = _decimal(value['value']), _decimal(value['scale'])
    if scale <= 0:
        raise ValueError('Positive declared scale required')
    return _product(amount, scale)


def _period(record):
    if not isinstance(record, dict):
        return None, 'invalid_period_dates'
    try:
        start, end = _date(record['period_start']), _date(record['period_end'])
    except (KeyError, TypeError, ValueError):
        return None, 'invalid_period_dates'
    days = (end - start).days + 1
    if not MIN_ANNUAL_DAYS <= days <= MAX_ANNUAL_DAYS:
        return None, 'nonannual_duration'
    if type(record.get('days_inclusive')) is not int or record['days_inclusive'] != days:
        return None, 'duration_mismatch'
    try:
        gross_profit, assets = _amount(record['gross_profit']), _amount(record['assets'])
    except (KeyError, TypeError, ValueError, ArithmeticError, InvalidOperation):
        return None, 'invalid_amount_or_scale'
    if assets <= 0:
        return None, 'nonpositive_assets'
    return dict(start=start, end=end, days=days, gross_profit=gross_profit, assets=assets), None


def _detail(reason, current=None):
    eid = None if current is None else current.get('event_id')
    available = None if current is None else current.get('available_from')
    return dict(available=False, reason=reason, method=METHOD,
                current_event_id=eid, prior_event_id=eid,
                current_available_from=available, prior_available_from=available,
                current_period_end=None, prior_period_end=None, current_days=None,
                prior_days=None, currency_key=None, gross_profit_growth_sign=None,
                gross_profitability_improvement_sign=None)


def evaluate_report(record, signal):
    """Return bounded signs for two annual columns of ONE selected release."""
    day = _date(signal)
    detail = _detail('unavailable', record)
    def fail(reason):
        detail['reason'] = reason
        return detail
    if record.get('admitted') is not True:
        return fail('current_not_admitted')
    if record.get('method') != METHOD:
        return fail('wrong_comparison_method')
    if not all(_text(record.get(k)) for k in ('event_id', 'symbol', 'issuer_id', 'security_id')):
        return fail('missing_identity')
    if record['security_id'] != record['symbol']:
        return fail('security_symbol_mismatch')
    if record.get('currency_key') != 'same-report-dollar:'+record['event_id']:
        return fail('invalid_same_report_currency_key')
    try:
        available, released = _date(record['available_from']), _date(record['release_date'])
        identity_start, identity_end = _date(record['identity_valid_from']), _date(record['identity_valid_through'])
    except (KeyError, TypeError, ValueError):
        return fail('invalid_dates_or_identity_bounds')
    if available > day:
        return fail('report_not_yet_available')
    if released >= available:
        return fail('availability_not_after_release')
    if identity_start != available:
        return fail('identity_start_not_current_availability')
    reports = record.get('same_report')
    if not isinstance(reports, dict) or set(reports) != {'current', 'comparison'}:
        return fail('same_report_columns_required')
    cur, reason = _period(reports['current'])
    if reason:
        return fail('current_'+reason)
    prev, reason = _period(reports['comparison'])
    if reason:
        return fail('comparison_'+reason)
    detail.update(current_period_end=cur['end'].isoformat(), prior_period_end=prev['end'].isoformat(),
                  current_days=cur['days'], prior_days=prev['days'], currency_key=record['currency_key'])
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
    try:
        growth_left = _product(cur['gross_profit'], Decimal(prev['days']))
        growth_right = _product(prev['gross_profit'], Decimal(cur['days']))
        profitability_left = _product(growth_left, prev['assets'])
        profitability_right = _product(growth_right, cur['assets'])
    except (ValueError, ArithmeticError):
        return fail('unrepresentable_exact_product')
    def sign(left, right):
        return .5 if left > right else (-.5 if left < right else 0.)
    detail.update(available=True, reason='available',
                  gross_profit_growth_sign=sign(growth_left, growth_right),
                  gross_profitability_improvement_sign=sign(profitability_left, profitability_right))
    return detail


def _cell(events, signal):
    day = _date(signal)
    known = {}
    for event in events:
        _require(isinstance(event, dict) and _text(event.get('event_id')), 'Event identity required')
        available = _date(event.get('available_from'))
        if available > day:
            continue
        eid = event['event_id']
        if eid in known and known[eid] != event:
            return _detail('conflicting_event_id')
        known[eid] = event
    if not known:
        return _detail('current_missing_event')
    latest = max(e['available_from'] for e in known.values())
    selected = [e for e in known.values() if e['available_from'] == latest]
    if len(selected) != 1:
        return _detail('current_ambiguous_latest_event')
    return evaluate_report(selected[0], signal)


def build_features(meta, admitted_dataset):
    """Keep the original pool and return identical missing masks for both arms."""
    _require(isinstance(admitted_dataset, dict) and type(admitted_dataset.get('schema_version')) is int
             and admitted_dataset['schema_version'] == 1 and admitted_dataset.get('method') == METHOD,
             'Explicit same-report dataset schema required')
    events = admitted_dataset.get('events')
    _require(isinstance(events, list), 'Complete event list required')
    by_symbol = {}
    for event in events:
        _require(isinstance(event, dict) and _text(event.get('symbol')), 'Declared event symbol required')
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
    audit = dict(original_arrays_unchanged=True, original_months=len(dates), original_width=49,
                 new_model_fits=0, returns_or_nav_computed=False,
                 original_auxiliary_states_and_core_coefficients_preserved=True,
                 common_available_cells=int(arrays['common_available'].sum()),
                 reasons=dict(Counter(cell['reason'] for cell in cells)),
                 source_verification='caller_verified_conditional_admission_and_source_hashes_required',
                 sign_method='exact_decimal_cross_products_after_declared_scale_normalization',
                 latest_event_before_validity_filter=True, same_current_release_for_both_periods=True, cells=cells)
    return dict(arrays=arrays, audit=audit)
