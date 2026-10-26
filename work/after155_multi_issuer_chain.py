"""Source-only annual revenue chain eligibility. No I/O or financial ratios.

New implementation; the previously frozen MSFT policy remains unchanged.
The caller must independently bind source reviews and the decision timestamp.
"""
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
import re


PROFILES = {
    'MSFT': {'calendar': 'june30', 'currency': 'USD'},
    'AAPL': {'calendar': 'september_last_saturday', 'currency': 'USD'},
    'ABBV': {'calendar': 'december31', 'currency': 'USD'},
    'ABT': {'calendar': 'december31', 'currency': 'USD'},
}
MEASURE = 'consolidated_gaap_total_revenue'
DOLLAR_BASIS = 'declared_within_issuer_reported_dollar_v1'


def _timestamp(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _digest(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def _positive(value):
    # Floats could already have rounded the original displayed monetary value.
    if not isinstance(value, str):
        raise ValueError('Expected original exact decimal text')
    result = Decimal(value)
    if not result.is_finite() or result <= 0:
        raise ValueError('Nonpositive or nonfinite source amount')
    return result


def _currency_key(record, issuer, prefix=''):
    currency = record.get(prefix+'currency')
    basis = record.get(prefix+'currency_basis')
    iso = record.get(prefix+'currency_iso')
    certified = record.get(prefix+'currency_iso_certified')
    if currency == 'USD' and basis == 'iso_evidenced' and iso == 'USD' and certified is True:
        return (currency, basis, iso, certified)
    if (issuer != 'MSFT' and currency == 'reported_dollar' and basis == DOLLAR_BASIS
            and iso is None and certified is False):
        return (currency, basis, iso, certified)
    raise ValueError('Unsupported or falsely certified currency basis')


def _end(year, kind):
    if kind == 'june30':
        return date(year, 6, 30)
    if kind == 'december31':
        return date(year, 12, 31)
    end = date(year, 9, 30)
    return end - timedelta(days=(end.weekday() - 5) % 7)


def expected_period(issuer, fiscal_year):
    """Calendar expectation only: observed source dates must match it."""
    if isinstance(fiscal_year, bool) or not isinstance(fiscal_year, int):
        raise ValueError('Fiscal year must be an integer')
    profile = PROFILES[issuer]
    start = _end(fiscal_year - 1, profile['calendar']) + timedelta(days=1)
    end = _end(fiscal_year, profile['calendar'])
    return start.isoformat(), end.isoformat()


def evaluate(issuer, reports, edges, cutoff_ns, decision_date):
    """No replacement values or fallback endpoint are supplied on failure.

    The supplied catalog contains *one issuer only*, including invalid and
    missing-field latest reports, so validity cannot censor endpoint selection.
    Source checks mean an external independent review of the original and its
    currency evidence, not this function's numerical plausibility checks.
    """
    result = dict(issuer=issuer, status='unresolved_source_or_comparability',
                  reasons=[], latest_report_id=None, report_ids=[], edge_ids=[],
                  factor_computed=False, economic_homogeneity_certified=False)
    try:
        return _evaluate(issuer, reports, edges, cutoff_ns, decision_date, result)
    except (KeyError, TypeError, ValueError, InvalidOperation, OverflowError):
        result.update(status='unresolved_source_or_comparability',
                      reasons=['malformed_required_source_record'])
        return result


def _evaluate(issuer, reports, edges, cutoff_ns, decision_date, result):
    profile = PROFILES[issuer]
    day = date.fromisoformat(decision_date)

    def stop(status, reason):
        result.update(status=status, reasons=[reason])
        return result

    def bad(reason):
        return stop('unresolved_source_or_comparability', reason)

    if not _timestamp(cutoff_ns):
        return bad('invalid_decision_timestamp')
    if not isinstance(reports, list) or not isinstance(edges, list):
        return bad('invalid_catalog')
    if any(not isinstance(r, dict) for r in reports) or any(not isinstance(e, dict) for e in edges):
        return bad('invalid_catalog_member')
    # A malformed clock cannot silently disappear from the known-report set.
    if any(r.get('issuer') != issuer or not _timestamp(r.get('known_ns')) for r in reports):
        return bad('unbound_issuer_or_report_clock')
    if any(e.get('issuer') != issuer or not _timestamp(e.get('known_ns')) for e in edges):
        return bad('unbound_issuer_or_edge_clock')
    known = [r for r in reports if r['known_ns'] <= cutoff_ns]
    if not known:
        return stop('missing_required_source', 'no_known_original_annual_report')
    latest_time = max(r['known_ns'] for r in known)
    latest = [r for r in known if r['known_ns'] == latest_time]
    if len(latest) != 1:
        return bad('conflicting_latest_release')
    latest = latest[0]
    result['latest_report_id'] = latest['report_id']
    if latest.get('source_checked') is not True:
        return bad('latest_source_not_checked')
    year = latest['fiscal_year']
    if isinstance(year, bool) or not isinstance(year, int):
        return bad('invalid_fiscal_year')
    # A later correction to an older fiscal year is not an excuse to ignore a
    # newer-year report. Explicitly resolve the correction before admission.
    if any(r['fiscal_year'] > year for r in known):
        return bad('later_release_for_older_year_requires_review')
    age = (day - date.fromisoformat(latest['period_end'])).days
    if age < 0:
        return bad('future_latest_period')
    if age > 550:
        return stop('stale_latest_annual', 'latest_annual_period_older_than_550_days')
    members = []
    currency_key = None
    currency_policy = None
    common_scale = None
    for y in range(year - 3, year + 1):
        options = [r for r in known if r['fiscal_year'] == y]
        if not options:
            return stop('missing_required_source', 'missing_own_year_' + str(y))
        if len(options) != 1:
            return bad('conflicting_own_year_' + str(y))
        r = options[0]
        if (r.get('source_checked') is not True or r.get('currency_basis_checked') is not True
                or r.get('measure') != MEASURE
                or not _digest(r.get('source_sha256')) or not _digest(r.get('source_review_sha256'))):
            return bad('invalid_or_unreviewed_member_' + str(y))
        key = _currency_key(r, issuer)
        if not _digest(r.get('currency_policy_sha256')):
            return bad('unbound_currency_policy_'+str(y))
        if currency_key is None:
            currency_key = key
            currency_policy = r['currency_policy_sha256']
        if key != currency_key or r['currency_policy_sha256'] != currency_policy:
            return bad('mixed_currency_basis_'+str(y))
        if (r.get('period_start'), r.get('period_end')) != expected_period(issuer, y):
            return bad('invalid_period_' + str(y))
        _positive(r['current'])
        scale = _positive(r['scale'])
        if common_scale is None:
            common_scale = scale
        if scale != common_scale:
            return bad('mixed_original_scale_'+str(y))
        if not isinstance(r['report_id'], str) or not r['report_id']:
            return bad('invalid_report_identity')
        result['report_ids'].append(r['report_id'])
        members.append(r)
    if len(set(result['report_ids'])) != 4:
        return bad('duplicate_report_identity')
    for earlier, later in zip(members, members[1:]):
        y = later['fiscal_year']
        options = [e for e in edges if e['earlier_fiscal_year'] == y-1
                   and e['later_fiscal_year'] == y and e['known_ns'] <= cutoff_ns]
        if len(options) != 1:
            return bad('missing_or_conflicting_edge_' + str(y))
        edge = options[0]
        if (not isinstance(edge.get('edge_id'), str) or not edge['edge_id']
                or edge['edge_id'] in result['edge_ids']):
            return bad('invalid_or_duplicate_edge_identity')
        result['edge_ids'].append(edge['edge_id'])
        clocks = edge.get('evidence_known_ns')
        if (edge.get('earlier_report_id') != earlier['report_id']
                or edge.get('later_report_id') != later['report_id']
                or edge['known_ns'] < max(earlier['known_ns'], later['known_ns'])
                or not isinstance(clocks, list) or not clocks
                or any(not _timestamp(t) or t > edge['known_ns'] for t in clocks)
                or not _digest(edge.get('source_review_sha256'))):
            return bad('unbound_or_mistimed_edge_' + str(y))
        if edge['status'] == 'blocked_known_basis_break':
            return stop('blocked_known_basis_break', edge['edge_id'])
        if edge['status'] != 'checked_reported_total_edge':
            return bad('unreviewed_edge_' + str(y))
        if (later.get('comparative_period_start') != earlier['period_start']
                or later.get('comparative_period_end') != earlier['period_end']
                or later.get('comparative_currency') != earlier['currency']
                or later.get('comparative_measure') != earlier['measure']
                or later.get('comparative_source_report_id') != later['report_id']):
            return bad('incompatible_witness_' + str(y))
        if _currency_key(later, issuer, 'comparative_') != currency_key:
            return bad('incompatible_witness_currency_basis_'+str(y))
        amount = _positive(later['comparative'])
        scale = _positive(later['comparative_scale'])
        if scale != _positive(earlier['scale']):
            return bad('incompatible_witness_scale_' + str(y))
        if amount != _positive(earlier['current']):
            return stop('blocked_known_basis_break', 'comparative_mismatch_' + str(y))
    result.update(status='checked_reported_total_chain_with_declared_source_scope', reasons=[])
    return result
