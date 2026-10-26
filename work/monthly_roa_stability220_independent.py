"""Independent source-clock, exact arithmetic and account oracle for round220.

Never imports the producer kernel, scoring module or proposed arithmetic helper.
The only dynamically imported dependency is the hash-pinned independent account
adapter, which is not loaded during feature verification.
Pre-registration input hardening: parse every edge-evidence date as canonical
ISO before computing its maximum; malformed metadata is unavailable, not a
lexicographically early date. No registered file or actual source was changed.
"""
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import hashlib
import importlib.util
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FROZEN_HELPER = ROOT / 'work/monthly_sector_recurrence192_independent.py'
FROZEN_HELPER_SHA = 'f78cc005827fd977ec431bdec4af0f4b7160694cb3d382df0115e8e00e0bedf8'
ARMS = ('incumbent_local67', 'roa_stability_penalty', 'three_year_roa_mean_control')
RAW = {ARMS[1]: 'candidate', ARMS[2]: 'control'}
DOSE = .01
VALUE_KEYS = ('available', 'candidate', 'control')
FEATURE_FLAGS = ('passed', 'independent_scalar_rebuild',
    'full_catalog_latest_fiscal_anchor_checked', 'no_older_fallback_checked',
    'exact_identity_checked', 'three_consecutive_original_reports_checked',
    'edge_version_clocks_checked', 'original_fields_only_checked',
    'own_report_assets_checked', 'latest550_and_historical_age_checked',
    'rational_roa_mean_variance_checked', 'sample_ddof1_checked',
    'binary64_sqrt_contract_checked', 'bounded_transform_checked',
    'exact_common_availability', 'no_cross_sectional_ranking',
    'no_predictive_fits_checked', 'audit_checked', 'no_future_or_outcome_inputs')
DELTA_FLAGS = ('passed', 'independent_delta_rebuild', 'fixed_point01_formula_checked',
    'eligible_only_checked', 'zero_or_missing_zero_delta_checked',
    'exact_targets_checked', 'no_second_rank', 'correction_bound_checked',
    'penalty_monotonicity_checked')
FIELDS = ('annual_net_income', 'opening_total_assets', 'closing_total_assets')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def frozen_helper():
    require(hashlib.sha256(FROZEN_HELPER.read_bytes()).hexdigest() == FROZEN_HELPER_SHA,
            'Frozen192 independent accounting helper changed')
    spec = importlib.util.spec_from_file_location('roa220_frozen_independent_accounts', FROZEN_HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dependency_files():
    return tuple(sorted(set(frozen_helper().dependency_files()) | {Path(__file__).resolve()}))


def accounting():
    return frozen_helper().accounting()


def audit_outcomes(px, opening, orders, exports, results, meta):
    helper = frozen_helper()
    helper.ARMS = ARMS
    result = helper.audit_outcomes(px, opening, orders, exports, results, meta)
    require(result['checked_paths'] == 9 and result['checked_slices'] == 45 and
            set(result['paths']) == set(ARMS), 'Incomplete nine-path accounting')
    result['outcome_adapter_sha256'] = FROZEN_HELPER_SHA
    return result


def record_check(actual, expected, label='audit'):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), label + ': keys')
        for key in expected:
            record_check(actual[key], expected[key], label + '/' + key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), label + ': list')
        for i, (a, b) in enumerate(zip(actual, expected)):
            record_check(a, b, label + '/' + str(i))
    else:
        require(type(actual) is type(expected) and actual == expected, label + ': value/type')
        if type(expected) is float:
            require(math.copysign(1., actual) == math.copysign(1., expected), label + ': sign')


def iso(value):
    require(type(value) is str, 'Canonical ISO date required')
    out = date.fromisoformat(value)
    require(out.isoformat() == value, 'Canonical ISO date required')
    return out


def exact_amount(value, scale):
    require(type(value) is str and type(scale) is str, 'Original decimal strings required')
    try:
        v, s = Decimal(value), Decimal(scale)
    except InvalidOperation as exc:
        raise ValueError('Malformed original decimal amount') from exc
    require(v.is_finite() and s.is_finite() and s > 0, 'Finite amount and positive scale required')
    return Fraction(v) * Fraction(s)


def finite_binary64(value):
    require(type(value) is Fraction, 'Exact rational conversion required')
    try:
        number = float(value)
    except OverflowError as exc:
        raise FloatingPointError('Exact amount overflows binary64') from exc
    if not math.isfinite(number) or value != 0 and number == 0.:
        raise FloatingPointError('Nonzero underflow or nonfinite binary64 conversion')
    return number


def arithmetic(amounts):
    require(type(amounts) in (list, tuple) and len(amounts) == 3, 'Exactly three original annual rows')
    ratios = []
    for row in amounts:
        require(type(row) in (list, tuple) and len(row) == 3 and
                all(type(x) is Fraction for x in row), 'Three exact original monetary fields required')
        net, first, last = row
        require(first > 0 and last > 0, 'Positive own-report assets required')
        ratios.append(2 * net / (first + last))
    mean = sum(ratios, Fraction()) / 3
    deviations = [x - mean for x in ratios]
    variance = sum((x*x for x in deviations), Fraction()) / 2
    sigma = math.sqrt(finite_binary64(variance)) if variance else 0.
    penalty = mean - Fraction.from_float(sigma)
    control = finite_binary64(mean / (1 + abs(mean)))
    candidate = control if not variance else finite_binary64(penalty / (1 + abs(penalty)))
    require(candidate <= control, 'Dispersion penalty cannot increase score')
    return dict(roa=tuple(ratios), mean=mean, sample_variance=variance,
        sample_std=sigma, penalized_mean=penalty, candidate=candidate,
        control=control, equal_arms=candidate == control, zero_dispersion=variance == 0)


def feature_arrays(values):
    require(isinstance(values, dict) and set(values) == set(VALUE_KEYS), 'Exact three feature arrays required')
    a = {k: np.asarray(v) for k, v in values.items()}
    mask = a['available']
    require(mask.ndim == 2 and mask.dtype == np.dtype(bool), 'Boolean availability matrix required')
    for key in ('candidate', 'control'):
        x = a[key]
        require(x.shape == mask.shape and x.dtype == np.dtype('float64'), 'Float64 feature shape required')
        require(not np.isinf(x).any() and np.array_equal(np.isfinite(x), mask), 'Exact common source mask')
        require(np.all(np.abs(x[mask]) <= 1.), 'Bounded h outside binary64 range')
    require(np.all(a['candidate'][mask] <= a['control'][mask]), 'Penalty increased h')
    return a


def verify_deltas(values, arrays, targets, deltas):
    a = feature_arrays(values)
    n, w = a['available'].shape
    require(isinstance(arrays, dict) and 'eligible' in arrays, 'Original eligibility required')
    eligible = np.asarray(arrays['eligible'])
    require(eligible.ndim == 2 and eligible.shape[1] == w and len(eligible) >= n and
            eligible.dtype == np.dtype(bool), 'Boolean original eligibility axes')
    require(isinstance(targets, dict) and isinstance(deltas, dict) and
            set(targets) == set(deltas) == set(RAW), 'Exact correction arms required')
    changed, supported, maxima = {}, {}, {}
    for arm, key in RAW.items():
        t, d = np.asarray(targets[arm]), np.asarray(deltas[arm])
        require(t.shape == d.shape == (n,w) and t.dtype == d.dtype == np.dtype('float64'),
                'Exact prefix target/delta float64 matrices required')
        require(np.array_equal(t, a[key], equal_nan=True), 'Target is not original h')
        used = a['available']
        require(np.array_equal(np.signbit(t[used]), np.signbit(a[key][used])), 'Target sign differs')
        wanted = np.zeros((n,w), float)
        for i in range(n):
            for j in range(w):
                if eligible[i,j] and used[i,j] and float(a[key][i,j]) != 0.:
                    wanted[i,j] = .01 * float(a[key][i,j])
        require(np.array_equal(d, wanted) and np.isfinite(d).all(), 'Exact eligible nonzero correction differs')
        require(np.array_equal(np.signbit(d), np.signbit(wanted)), 'Delta exact zero sign differs')
        require(np.all(np.abs(d) <= .01), 'Fixed correction bound exceeded')
        changed[arm], supported[arm] = int(np.count_nonzero(d)), int(used.sum())
        maxima[arm] = float(np.max(np.abs(d), initial=0.))
    return dict(**{k: True for k in DELTA_FLAGS}, checked_events=n, dose=.01,
        max_absolute_delta=.01, actual_max_absolute_delta=maxima,
        changed_cells=changed, target_available_cells=supported)


COUNT_FIELDS = ('checked_cells', 'available_cells', 'unavailable_cells',
    'annual_ratio_evaluations', 'mean_estimates', 'sample_dispersion_estimates', 'bounded_scores')
METHOD = 'as_reported_original_NI_assets_three_annual_reports_v1'
VERIFIED = 'verified_calendar_rule_only'


def known_by(row, signal, catalog=False):
    if signal is None:
        return False
    cutoff = iso(signal)
    if catalog:
        if row.get('clock_status') == VERIFIED:
            return iso(row['assumed_available_session']) <= cutoff
        return row.get('publication_date') is None or iso(row['publication_date']) < cutoff
    return row.get('known_at') is None or iso(row['known_at']) <= cutoff


def indexed(rows, field):
    output = {}
    for item in rows:
        key = item[field]
        require(type(key) is str and key and key not in output, 'Duplicate/invalid ' + field)
        output[key] = item
    return output


def missing(symbol, signal, reason, event=None):
    return dict(symbol=symbol, date=signal, available=False, reason=reason,
        latest_event_id=event, report_version_ids=[], edge_version_ids=[], chain_version_id=None)


def decimal_rational(text):
    require(type(text) is str, 'Reported decimal string required')
    try:
        value = Decimal(text)
    except InvalidOperation as exc:
        raise ValueError('Invalid reported decimal') from exc
    require(value.is_finite(), 'Finite source amount required')
    return Fraction(value)


def original_observation(report, event, signal, calendar, is_latest):
    require(event.get('clock_status') == VERIFIED, 'unverified_report_clock')
    published, released = iso(event['publication_date']), iso(event['assumed_available_session'])
    require(iso(calendar[0]) <= published < iso(calendar[-1]), 'publication_outside_calendar')
    following = next((s for s in calendar if iso(s) > published), None)
    require(following == released.isoformat(), 'wrong_next_session')
    require(event.get('clock', {}).get('next_session_rule_verified') is True, 'missing_clock_receipt')
    require(event.get('original_field_state') == 'admitted' and
        report.get('inherited_single_report_fields_admitted') is True, 'held_report')
    require(all(report[k] == event[k] for k in
        ('report_version_id','event_id','issuer_id','security_id','fiscal_year')), 'report_catalog_binding')
    available = iso(report['known_at'])
    require(report['known_at'] == report['available_from'] == event['assumed_available_session'] and
        available <= iso(signal), 'report_evidence_clock')
    p = report['period']
    start, end = iso(p['period_start']), iso(p['period_end'])
    length = (end - start).days + 1
    require(330 <= length <= 400 and p['days_inclusive'] == length and
        event['period_start'] == start.isoformat() and event['period_end'] == end.isoformat(), 'report_period')
    require(end <= published < available, 'report_release_period')
    require(0 <= ((iso(signal) if is_latest else available) - end).days <= 550, 'report_age')
    require(report.get('corrected_original_values_substituted') is False, 'replaced_original_values')
    source = report['source']
    require(source.get('path') and source.get('sha256'), 'original_source_required')
    fields = report['fields']
    require(set(fields) == set(FIELDS), 'nine_field_schema')
    money, currencies = [], []
    for name in FIELDS:
        description = fields[name]
        cell = description['source_cell']
        expected_day = (start - timedelta(days=1)) if name == FIELDS[1] else end
        expected_text = expected_day.isoformat()
        require(description['as_of'] == expected_text and cell.get('as_of', expected_text) == expected_text,
            'field_period_endpoint')
        require(cell['source']['path'] == source['path'] and cell['source']['sha256'] == source['sha256'],
            'field_original_source')
        require(cell.get('available_from', available.isoformat()) == available.isoformat(), 'field_clock')
        require(description.get('original_field_path') and cell.get('locator'), 'field_locator')
        scale = decimal_rational(cell['scale'])
        require(scale > 0, 'positive_scale')
        amount = decimal_rational(cell['value_decimal']) * scale
        if name != FIELDS[0]:
            require(amount > 0, 'positive_asset_endpoint')
        else:
            require(description.get('ni_attribution') == 'reported_consolidated_total_NI_before_NCI_deduction',
                'NI_attribution')
            period = cell.get('period')
            if period is not None:
                require(period.get('period_start') == start.isoformat() and
                    period.get('period_end') == end.isoformat(), 'NI_annual_period')
        require(cell['currency_literal'] == '$', 'currency_literal')
        currencies.append(cell.get('currency_iso'))
        money.append(amount)
    require(len(set(currencies)) == 1, 'same_report_currency')
    return tuple(money), currencies[0]


def source_cell(dataset, symbol, signal, calendar):
    """Reconstruct the causal join by fiscal-year groups, not publication order."""
    registry = [x for x in dataset['identity_registry'] if symbol in x['symbol_aliases']]
    if len(registry) != 1:
        return missing(symbol, signal, 'unsupported_issuer' if not registry else 'ambiguous_issuer_registry'), None
    identity = registry[0]['issuer_id'], registry[0]['security_id']
    catalog = [row for row in dataset['catalog'] if row['canonical_symbol'] == symbol and known_by(row, signal, True)]
    indexed(catalog, 'event_id')
    if not catalog:
        return missing(symbol, signal, 'no_known_annual_event'), None
    if any(row.get('publication_date') is None for row in catalog):
        return missing(symbol, signal, 'unknown_catalog_timing'), None
    if any(type(row.get('fiscal_year')) is not int for row in catalog):
        return missing(symbol, signal, 'unknown_fiscal_anchor'), None
    yearly, corrections = {}, []
    for row in catalog:
        kind = row.get('kind')
        if kind == 'original_annual_event':
            yearly.setdefault(row['fiscal_year'], []).append(row)
        elif kind == 'correction':
            corrections.append(row)
        else:
            return missing(symbol, signal, 'unknown_event_kind'), None
    if not yearly:
        return missing(symbol, signal, 'no_known_annual_event'), None
    anchors = [(year, iso(row['period_end'])) for year, rows in yearly.items()
        for row in rows if row.get('period_end') is not None]
    for year, ending in anchors:
        if any(older < year and prior >= ending for older, prior in anchors):
            return missing(symbol, signal, 'inconsistent_fiscal_anchors'), None
    final_year = max(yearly)
    if len(yearly[final_year]) != 1:
        return missing(symbol, signal, 'ambiguous_latest_fiscal_anchor'), None
    latest = yearly[final_year][0]
    event_id = latest['event_id']
    def stop(reason):
        return missing(symbol, signal, reason, event_id), None
    if latest.get('clock_status') != VERIFIED:
        return stop('latest_clock_unverified')
    if (latest.get('issuer_id'), latest.get('security_id')) != identity:
        return stop('latest_identity_unverified')
    if latest.get('original_field_state') != 'admitted' or not latest.get('report_version_id'):
        return stop('latest_report_held_or_unprocessed')
    if latest.get('period_end') is None:
        return stop('latest_period_unknown')
    if not 0 <= (iso(signal) - iso(latest['period_end'])).days <= 550:
        return stop('latest_report_stale')
    events = []
    for year in range(final_year-2, final_year+1):
        if len(yearly.get(year, [])) != 1:
            return stop('missing_or_ambiguous_preceding_year')
        row = yearly[year][0]
        if (row.get('issuer_id'), row.get('security_id')) != identity:
            return stop('mixed_or_unknown_issuer')
        events.append(row)
    report_ids = [r.get('report_version_id') for r in events]
    if None in report_ids or len(set(report_ids)) != 3:
        return stop('three_distinct_originals_required')
    event_ids = {r['event_id'] for r in events}
    for row in corrections:
        report_target, event_target = row.get('corrects_report_id'), row.get('corrects_event_id')
        if not report_target and not event_target:
            return stop('unlinked_known_correction')
        if report_target in report_ids or event_target in event_ids:
            return stop('known_correction_requires_new_chain_review')
    reports = indexed([r for r in dataset['reports'] if known_by(r, signal)], 'report_version_id')
    if any(key not in reports for key in report_ids):
        return stop('report_version_not_known')
    possible = [r for r in dataset['chains'] if r['report_version_ids'] == report_ids and known_by(r, signal)]
    if len(possible) != 1:
        return stop('missing_or_ambiguous_chain_evidence')
    chain = possible[0]
    if chain.get('conditional_allowed') is not True or chain.get('known_at') is None:
        return stop('chain_not_conditionally_admitted')
    if (chain['issuer_id'], chain['security_id']) != identity:
        return stop('chain_identity_mismatch')
    if chain['fiscal_years'] != list(range(final_year-2, final_year+1)):
        return stop('chain_fiscal_years_mismatch')
    edges = indexed([r for r in dataset['edges'] if known_by(r, signal)], 'edge_version_id')
    edge_ids = chain['edge_version_ids']
    if len(edge_ids) != 2 or any(key not in edges for key in edge_ids):
        return stop('edge_version_not_known')
    for index, identifier in enumerate(edge_ids):
        edge = edges[identifier]
        if edge.get('conditional_allowed') is not True or edge.get('known_at') is None:
            return stop('edge_not_conditionally_admitted')
        if any(row.get('supersedes_version_id') == identifier for row in edges.values()):
            return stop('edge_version_superseded')
        if ((edge['issuer_id'], edge['security_id']) != identity or
            edge['earlier_report_version_id'] != report_ids[index] or
            edge['later_report_version_id'] != report_ids[index+1]):
            return stop('edge_binding_mismatch')
        references = edge.get('evidence_refs')
        if not isinstance(references, list) or len(references) < 2:
            return stop('edge_original_evidence_binding')
        for report_id in report_ids[index:index+2]:
            links = [r for r in references if r.get('report_version_id') == report_id]
            original = reports[report_id]
            if len(links) != 1:
                return stop('edge_original_evidence_binding')
            link = links[0]
            if (link.get('event_id') != original['event_id'] or
                link.get('source', {}).get('path') != original['source']['path'] or
                link.get('source', {}).get('sha256') != original['source']['sha256'] or
                link.get('assumed_available_from') != original['known_at']):
                return stop('edge_original_evidence_binding')
        if any(not x.get('source', {}).get('path') or not x.get('source', {}).get('sha256') for x in references):
            return stop('edge_original_evidence_binding')
        evidence_dates = [reports[x]['known_at'] for x in report_ids[index:index+2]]
        evidence_dates.extend(x.get('assumed_available_from') for x in references)
        try:
            evidence_days = [iso(value) for value in evidence_dates]
            edge_day, chain_day = iso(edge['known_at']), iso(chain['known_at'])
            supported_clock = bool(evidence_days) and all(day <= edge_day for day in evidence_days) and edge_day <= chain_day
        except (TypeError, ValueError):
            supported_clock = False
        if not supported_clock:
            return stop('edge_evidence_clock')
    try:
        observations, units = [], []
        for i, (identifier, event) in enumerate(zip(report_ids, events)):
            if any(row.get('supersedes_version_id') == identifier for row in reports.values()):
                return stop('report_version_superseded')
            observation, unit = original_observation(reports[identifier], event, signal, calendar, i == 2)
            observations.append(observation)
            units.append(unit)
            if i:
                require(iso(reports[identifier]['period']['period_start']) ==
                    iso(reports[report_ids[i-1]]['period']['period_end']) + timedelta(days=1),
                    'annual_boundaries_not_contiguous')
        require(len({unit for unit in units if unit is not None}) <= 1, 'cross_report_currency')
        references = {(r['report_version_id'], r['field']): r for r in chain['nine_required_fields']}
        require(len(chain['nine_required_fields']) == len(references) == 9, 'nine_field_bindings')
        for identifier in report_ids:
            report = reports[identifier]
            for field in FIELDS:
                ref = references[(identifier, field)]
                require(ref['source_sha256'] == report['source']['sha256'] and ref['event_id'] == report['event_id'] and
                    ref['original_field_path'] == report['fields'][field]['original_field_path'],
                    'nine_field_binding_mismatch')
    except (KeyError, TypeError, ValueError) as exc:
        return stop('invalid_required_source_field:' + str(exc))
    result = missing(symbol, signal, 'available', event_id)
    result.update(available=True, report_version_ids=report_ids, edge_version_ids=edge_ids,
        chain_version_id=chain['chain_version_id'])
    return result, observations


def independent_features(dataset, meta, calendar):
    require(isinstance(dataset, dict) and dataset.get('method') == METHOD and
        dataset.get('conditional_research_only') is True, 'Exact conditional source policy required')
    require(isinstance(calendar, list) and calendar and calendar == sorted(set(calendar)), 'Sorted unique calendar')
    sessions = [iso(s) for s in calendar]
    require(all(x.weekday() < 5 for x in sessions), 'Official sessions cannot be weekends')
    dates, symbols = meta['dates'], meta['symbols']
    require(len(dates) == len(symbols) and dates == sorted(set(dates)), 'Ordered decision metadata')
    require(isinstance(meta['etfs'], (list,tuple,set,frozenset)) and
        all(type(s) is str and s and s.strip() == s for s in meta['etfs']), 'ETF identities required')
    etfs = set(meta['etfs'])
    require(len(etfs) == len(meta['etfs']), 'Unique ETF identities required')
    width = len(symbols[0]) if dates else 0
    require(all(isinstance(row, (list, tuple)) and row and len(row) == width and len(row) == len(set(row)) and
        all(type(s) is str and s and s.strip() == s for s in row) for row in symbols), 'Fixed unique candidate axes')
    values = dict(available=np.zeros((len(dates),width),bool),
        candidate=np.full((len(dates),width),np.nan),control=np.full((len(dates),width),np.nan))
    history = []
    for i, (signal, names) in enumerate(zip(dates, symbols)):
        iso(signal)
        require(signal in calendar, 'Decision outside calendar')
        position = calendar.index(signal)
        if not (i == 0 and signal == '2016-08-17'):
            require(position+1 < len(calendar) and calendar[position+1][:7] != signal[:7], 'Official month-end required')
        cells = []
        for j, symbol in enumerate(names):
            if symbol in etfs:
                cell, observations = missing(symbol, signal, 'ETF_no_financial_feature'), None
            else:
                cell, observations = source_cell(dataset, symbol, signal, calendar)
            if observations is not None:
                x = arithmetic(observations)
                values['available'][i,j] = True
                values['candidate'][i,j], values['control'][i,j] = x['candidate'], x['control']
                cell['measurement'] = dict(ratios_exact=[str(v) for v in x['roa']], mean_exact=str(x['mean']),
                    variance_exact=str(x['sample_variance']), sample_std=x['sample_std'],
                    candidate=x['candidate'],control=x['control'])
            cells.append(cell)
        history.append(dict(date=signal,symbols=list(names),available_count=sum(c['available'] for c in cells),cells=cells))
    count = int(values['available'].sum())
    total = len(dates)*width
    audit = dict(schema_version=1,events=len(dates),symbols_per_event=width,checked_cells=total,
        available_cells=count,unavailable_cells=total-count,annual_ratio_evaluations=3*count,
        mean_estimates=count,sample_dispersion_estimates=count,bounded_scores=2*count,
        new_model_fits=0,predictive_model_fits=0,event_history=history)
    return values,audit


def verify_features(dataset, meta, calendar, values, audit):
    actual = feature_arrays(values)
    expected_values, expected_audit = independent_features(dataset, meta, calendar)
    for name in VALUE_KEYS:
        require(np.array_equal(actual[name], expected_values[name], equal_nan=True), 'Independent source feature differs: '+name)
        if name != 'available':
            mask = expected_values['available']
            require(np.array_equal(np.signbit(actual[name][mask]), np.signbit(expected_values[name][mask])),
                'Independent feature sign differs: '+name)
    record_check(audit, expected_audit)
    return dict(**{name: True for name in FEATURE_FLAGS},
        **{name: expected_audit[name] for name in ('events','symbols_per_event',*COUNT_FIELDS)},
        new_model_fits=0,predictive_model_fits=0,computes_returns=False,
        producer_modules_imported=False,numerical_tolerance=0.,
        numerical_comparison='exact independent fiscal-source join, rational arithmetic, binary64 transform and full audit',
        outcome_adapter_sha256=FROZEN_HELPER_SHA)
