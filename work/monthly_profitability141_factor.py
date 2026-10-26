"""Pure annual gross-profitability changes; no I/O, ranking, trades or returns.

The caller verifies source hashes, original contexts, security mappings and the
market-calendar availability dates before supplying this admitted dataset:

  {'schema_version': 1, 'events': {symbol: [annual_disclosure_event, ...]}}

Every event, including rejected or amended disclosures, retains event_id and
available_from. A usable event additionally has admitted, original_annual and
measurement_eligible exactly True; issuer_id, security_id, currency_key;
identity_valid_from/identity_valid_through; period_start/period_end; and
 gross_profit/assets = {'value': decimal_string, 'scale': decimal_string}.
Identity intervals are inclusive. The current mapping covers the signal date;
the prior mapping covers that report's availability, with equal issuer/security
identities across the pair. Calendar, source and admission evidence are caller
responsibilities, not inferred from values or a present-day ticker.

All annual events must be supplied, not just valid reports. An invalid latest
report blocks fallback. Same-date competing events require upstream resolution;
this module never selects one by monetary value or by event-id sort order.
"""
from collections import Counter
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation, Inexact, Rounded, localcontext, MAX_EMAX, MIN_EMIN

import numpy as np
import pandas as pd

EXTRA = ('gross_profit_growth_sign', 'gross_profitability_improvement_sign', 'common_available')
MIN_ANNUAL_DAYS = 330
MAX_ANNUAL_DAYS = 400
MAX_AGE_DAYS = 550


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
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError('Explicit decimal string required')
    result = Decimal(value)
    if not result.is_finite():
        raise ValueError('Finite decimal required')
    return result


def _product(*values):
    # Sufficient precision for every coefficient digit; no division, subtraction
    # or ambient Decimal precision can round away a small sign difference.
    with localcontext() as ctx:
        ctx.prec = max(28, sum(len(v.as_tuple().digits) for v in values) + 2)
        ctx.Emax, ctx.Emin = MAX_EMAX, MIN_EMIN
        ctx.traps[Inexact] = True
        ctx.traps[Rounded] = True
        result = Decimal(1)
        for value in values:
            result *= value
        if not result.is_finite():
            raise ValueError('Nonfinite amount product')
        return result


def _amount(value):
    if not isinstance(value, dict):
        raise ValueError('Declared value and scale required')
    amount, scale = _decimal(value['value']), _decimal(value['scale'])
    if scale <= 0:
        raise ValueError('Positive declared scale required')
    return _product(amount, scale)


def _event(record, identity_day):
    """Validate one selected record, never silently select a replacement."""
    if record.get('admitted') is not True:
        return None, 'not_admitted'
    if record.get('original_annual') is not True:
        return None, 'not_original_annual'
    if record.get('measurement_eligible') is not True:
        return None, 'measurement_excluded'
    if not all(_text(record.get(k)) for k in ('issuer_id', 'security_id', 'currency_key')):
        return None, 'missing_identity_or_currency'
    try:
        available = _date(record['available_from'])
        start, end = _date(record['period_start']), _date(record['period_end'])
        identity_start = _date(record['identity_valid_from'])
        identity_end = _date(record['identity_valid_through'])
    except (KeyError, TypeError, ValueError):
        return None, 'invalid_dates_or_identity_bounds'
    if not identity_start <= identity_day <= identity_end:
        return None, 'identity_outside_interval'
    days = (end - start).days + 1
    if not MIN_ANNUAL_DAYS <= days <= MAX_ANNUAL_DAYS:
        return None, 'nonannual_duration'
    if end >= available:
        return None, 'availability_not_after_period'
    try:
        gross_profit = _amount(record['gross_profit'])
        assets = _amount(record['assets'])
    except (KeyError, TypeError, ValueError, ArithmeticError, InvalidOperation):
        return None, 'invalid_amount_or_scale'
    if assets <= 0:
        return None, 'nonpositive_assets'
    return dict(start=start, end=end, available=available, days=days,
                gross_profit=gross_profit, assets=assets,
                issuer_id=record['issuer_id'], security_id=record['security_id'],
                currency_key=record['currency_key']), None


def _detail(reason, current=None, prior=None):
    return dict(available=False, reason=reason,
                current_event_id=None if current is None else current.get('event_id'),
                prior_event_id=None if prior is None else prior.get('event_id'),
                current_available_from=None if current is None else current.get('available_from'),
                prior_available_from=None if prior is None else prior.get('available_from'),
                current_period_end=None, prior_period_end=None,
                current_days=None, prior_days=None, currency_key=None,
                gross_profit_growth_sign=None, gross_profitability_improvement_sign=None)


def evaluate_pair(current, prior, signal):
    """Return bounded signs and audit detail for two explicitly selected reports."""
    day = _date(signal)
    detail = _detail('unavailable', current, prior)
    cur, reason = _event(current, day)
    if reason:
        detail['reason'] = 'current_' + reason
        return detail
    try:
        prior_identity_day = _date(prior['available_from'])
    except (KeyError, TypeError, ValueError):
        detail['reason'] = 'prior_invalid_availability'
        return detail
    prev, reason = _event(prior, prior_identity_day)
    if reason:
        detail['reason'] = 'prior_' + reason
        return detail
    if cur['available'] > day or prev['available'] > day:
        detail['reason'] = 'report_not_yet_available'
        return detail
    if current.get('event_id') == prior.get('event_id'):
        detail['reason'] = 'same_report_not_independent'
        return detail
    detail.update(current_period_end=cur['end'].isoformat(), prior_period_end=prev['end'].isoformat(),
                  current_days=cur['days'], prior_days=prev['days'], currency_key=cur['currency_key'])
    if cur['issuer_id'] != prev['issuer_id'] or cur['security_id'] != prev['security_id']:
        detail['reason'] = 'different_issuer_or_security'
        return detail
    if cur['currency_key'] != prev['currency_key']:
        detail['reason'] = 'different_currency'
        return detail
    if (prev['end'] + timedelta(days=1) != cur['start'] or
            not MIN_ANNUAL_DAYS <= (cur['end'] - prev['end']).days <= MAX_ANNUAL_DAYS):
        detail['reason'] = 'noncontiguous_annual_periods'
        return detail
    if (day - cur['end']).days > MAX_AGE_DAYS:
        detail['reason'] = 'stale_current_report'
        return detail
    try:
        growth_left = _product(cur['gross_profit'], Decimal(prev['days']))
        growth_right = _product(prev['gross_profit'], Decimal(cur['days']))
        profitability_left = _product(growth_left, prev['assets'])
        profitability_right = _product(growth_right, cur['assets'])
    except (ValueError, ArithmeticError):
        detail['reason'] = 'unrepresentable_exact_product'
        return detail
    def sign(left, right):
        return .5 if left > right else (-.5 if left < right else 0.)
    detail.update(available=True, reason='available',
                  gross_profit_growth_sign=sign(growth_left, growth_right),
                  gross_profitability_improvement_sign=sign(profitability_left, profitability_right))
    return detail


def _known(events, day):
    """Read only chronology from future events; their values do not affect past."""
    _require(isinstance(events, list), 'Per-symbol annual events must be a list')
    known = {}
    for event in events:
        _require(isinstance(event, dict) and _text(event.get('event_id')), 'Event identity required')
        # Without a calendar availability date an event cannot safely be sorted.
        available = _date(event.get('available_from'))
        if available > day:
            continue
        identifier = event['event_id']
        if identifier in known and known[identifier] != event:
            return None, 'conflicting_event_id'
        known[identifier] = event
    return list(known.values()), None


def _latest(events):
    if not events:
        return None, 'missing_event'
    latest = max(e['available_from'] for e in events)
    selected = [e for e in events if e['available_from'] == latest]
    if len(selected) != 1:
        return None, 'ambiguous_latest_event'
    return selected[0], None


def _cell(events, signal):
    day = _date(signal)
    known, reason = _known(events, day)
    if reason:
        return _detail(reason)
    current, reason = _latest(known)
    if reason:
        return _detail('current_' + reason)
    cur, reason = _event(current, day)
    if reason:
        return _detail('current_' + reason, current)
    if (day - cur['end']).days > MAX_AGE_DAYS:
        return _detail('stale_current_report', current)
    expected = (cur['start'] - timedelta(days=1)).isoformat()
    candidates = [e for e in known if e['event_id'] != current['event_id'] and e.get('period_end') == expected]
    prior, reason = _latest(candidates)
    if reason:
        return _detail('prior_' + reason, current)
    # Do not skip a later annual event whose fiscal period cannot be identified.
    for event in known:
        if event['event_id'] in (current['event_id'], prior['event_id']):
            continue
        if event['available_from'] >= prior['available_from']:
            try:
                _date(event.get('period_end'))
            except (TypeError, ValueError):
                return _detail('unresolved_intervening_annual_period', current, prior)
    return evaluate_pair(current, prior, signal)


def build_features(meta, admitted_dataset):
    """Preserve the original axes; unusable facts are NaN in BOTH new arms."""
    _require(isinstance(admitted_dataset, dict) and
             type(admitted_dataset.get('schema_version')) is int and admitted_dataset['schema_version'] == 1,
             'Explicit admitted-dataset schema_version=1 required')
    events = admitted_dataset.get('events')
    _require(isinstance(events, dict), 'Per-symbol event mapping required')
    dates = pd.DatetimeIndex(meta['dates'])
    _require(dates.tz is None and not dates.hasnans and dates.is_unique and
             dates.is_monotonic_increasing and dates.equals(dates.normalize()) and len(dates) > 0,
             'Sorted unique naive signal dates required')
    _require(len(meta['symbols']) == len(dates), 'Original signal/symbol axes required')
    etfs = meta['etfs']
    _require(len(etfs) == 17 and len(set(etfs)) == 17 and all(_text(e) for e in etfs),
             'Original17 ETFs required')
    arrays = {key: np.full((len(dates), 49), np.nan) for key in EXTRA[:-1]}
    arrays['common_available'] = np.zeros((len(dates), 49), dtype=bool)
    cells = []
    for i, (day, names) in enumerate(zip(dates, meta['symbols'])):
        _require(len(names) == 49 and len(set(names)) == 49 and set(etfs) <= set(names) and
                 all(_text(name) for name in names), 'Original32 stocks plus17 ETFs required')
        signal = str(day.date())
        for j, symbol in enumerate(names):
            detail = _detail('etf_not_applicable') if symbol in etfs else _cell(events.get(symbol, []), signal)
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
                 source_verification='caller_verified_admission_and_source_hashes_required',
                 sign_method='exact_decimal_cross_products_after_declared_scale_normalization',
                 latest_event_before_validity_filter=True, cells=cells)
    return dict(arrays=arrays, audit=audit)
