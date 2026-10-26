"""Pure source-eligibility checks for the proposed MSFT four-year revenue chain.

No I/O, prices, revenue growth, ranking, imputation, or strategy calculations.
The caller must provide independently reviewed source records and edge evidence.
"""
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation


def evaluate(reports, edges, cutoff_ns, decision_date):
    """Caller binds the cutoff/date to the frozen exchange calendar.

    Incomplete source records fail closed with an explicit reason. This function
    never supplies replacement fields, sources or report vintages.
    """
    try:
        return _evaluate(reports, edges, cutoff_ns, decision_date)
    except (KeyError, TypeError, ValueError, InvalidOperation, OverflowError):
        return dict(status='unresolved_source_or_comparability',
                    reasons=['malformed_required_source_record'], latest_report_id=None,
                    report_ids=[], edge_ids=[], factor_computed=False)


def _evaluate(reports, edges, cutoff_ns, decision_date):
    day = date.fromisoformat(decision_date)
    known = [r for r in reports if r['known_ns'] <= cutoff_ns]
    result = dict(status='missing_required_source', reasons=[], latest_report_id=None,
                  report_ids=[], edge_ids=[], factor_computed=False)

    def stop(status, reason):
        result['status'] = status
        result['reasons'].append(reason)
        return result

    if not known:
        return stop('missing_required_source', 'no_known_original_annual_report')
    latest_time = max(r['known_ns'] for r in known)
    latest = [r for r in known if r['known_ns'] == latest_time]
    if len(latest) != 1:
        return stop('unresolved_source_or_comparability', 'conflicting_latest_release')
    latest = latest[0]
    result['latest_report_id'] = latest['report_id']
    if not latest.get('source_checked'):
        return stop('unresolved_source_or_comparability', 'latest_source_not_checked')
    if (day-date.fromisoformat(latest['period_end'])).days > 550:
        return stop('stale_latest_annual', 'latest_annual_period_older_than_550_days')
    if date.fromisoformat(latest['period_end']) > day:
        return stop('unresolved_source_or_comparability', 'future_latest_period')
    year = latest['fiscal_year']
    members = []
    for y in range(year-3, year+1):
        candidates = [r for r in known if r['fiscal_year'] == y]
        if not candidates:
            return stop('missing_required_source', 'missing_own_year_'+str(y))
        if len(candidates) != 1:
            return stop('unresolved_source_or_comparability', 'conflicting_own_year_'+str(y))
        r = candidates[0]
        if (not r.get('source_checked') or r.get('issuer') != 'MSFT'
                or r.get('currency') != 'USD' or r.get('measure') != 'consolidated_gaap_total_revenue'):
            return stop('unresolved_source_or_comparability', 'invalid_member_'+str(y))
        if r['period_start'] != f'{y-1}-07-01' or r['period_end'] != f'{y}-06-30':
            return stop('unresolved_source_or_comparability', 'invalid_period_'+str(y))
        try:
            amount, scale = Decimal(r['current']), Decimal(r['scale'])
            if not amount.is_finite() or not scale.is_finite() or amount <= 0 or scale <= 0:
                raise ValueError('Nonpositive amount or scale')
        except (InvalidOperation, ValueError, TypeError):
            return stop('unresolved_source_or_comparability', 'invalid_value_'+str(y))
        result['report_ids'].append(r['report_id'])
        members.append(r)
    for earlier, later in zip(members, members[1:]):
        y = later['fiscal_year']
        if date.fromisoformat(earlier['period_end'])+timedelta(days=1) != date.fromisoformat(later['period_start']):
            return stop('unresolved_source_or_comparability', 'nonadjacent_periods_'+str(y))
        candidates = [e for e in edges if e['earlier_fiscal_year'] == y-1
                      and e['later_fiscal_year'] == y and e['known_ns'] <= cutoff_ns]
        if len(candidates) != 1:
            return stop('unresolved_source_or_comparability', 'missing_or_conflicting_edge_'+str(y))
        edge = candidates[0]
        result['edge_ids'].append(edge['edge_id'])
        if (edge.get('earlier_report_id') != earlier['report_id']
                or edge.get('later_report_id') != later['report_id']
                or edge['known_ns'] < max(earlier['known_ns'], later['known_ns'])
                or not edge.get('evidence_known_ns')
                or any(t > edge['known_ns'] for t in edge['evidence_known_ns'])):
            return stop('unresolved_source_or_comparability', 'unbound_or_mistimed_edge_'+str(y))
        if edge['status'] == 'blocked_known_basis_break':
            return stop('blocked_known_basis_break', edge['edge_id'])
        if edge['status'] != 'checked_reported_total_edge':
            return stop('unresolved_source_or_comparability', edge['edge_id'])
        # A positive prose review cannot override conflicting numerical evidence.
        if (later.get('comparative_period_end') != earlier['period_end']
                or later.get('comparative_period_start') != earlier['period_start']
                or later.get('comparative_currency') != earlier['currency']
                or later.get('comparative_measure') != earlier['measure']
                or later.get('comparative_source_report_id') != later['report_id']):
            return stop('unresolved_source_or_comparability', 'incompatible_witness_'+str(y))
        try:
            witness, witness_scale = Decimal(later['comparative']), Decimal(later['comparative_scale'])
            if (not witness.is_finite() or not witness_scale.is_finite()
                    or witness_scale <= 0 or witness <= 0):
                raise ValueError('Invalid witness')
            if Decimal(earlier['scale']) != witness_scale:
                return stop('unresolved_source_or_comparability', 'incompatible_witness_scale_'+str(y))
            # Equal scales were required above. Comparing the exact Decimals
            # avoids multiplication rounding under the ambient Decimal context.
            if Decimal(earlier['current']) != witness:
                return stop('blocked_known_basis_break', 'comparative_mismatch_'+str(y))
        except (InvalidOperation, ValueError, TypeError):
            return stop('unresolved_source_or_comparability', 'invalid_witness_'+str(y))
    result['status'] = 'checked_reported_total_chain_with_declared_source_scope'
    return result
