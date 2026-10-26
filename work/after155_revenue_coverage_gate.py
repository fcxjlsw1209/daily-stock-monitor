"""Source-only feasibility gate on the complete, unchanged candidate grid.

No prices, financial values, scores or returns are read. Passing this modest
coverage floor is not strategy registration or independent source approval.
"""
from collections import Counter


USABLE = 'checked_reported_total_chain_with_declared_source_scope'


def coverage_gate(meta, rows, plan):
    dates, symbols = meta['dates'], meta['symbols']
    if len(dates) != 122 or len(symbols) != 122 or dates != sorted(set(dates)):
        raise ValueError('Expected complete original 122-month axis')
    if any(len(s) != 49 or len(set(s)) != 49 for s in symbols):
        raise ValueError('Expected 49 unique candidates in every month')
    expected = {(i, j): (dates[i], s) for i, ss in enumerate(symbols)
                for j, s in enumerate(ss)}
    observed = {}
    for row in rows:
        key = (row['signal_index'], row['candidate_index'])
        if key in observed or expected.get(key) != (row['decision_date'], row['symbol']):
            raise ValueError('Missing, duplicated or changed candidate slot')
        observed[key] = row
    if set(expected) != set(observed):
        raise ValueError('Candidate rows may not be dropped for missing source data')
    issuers = plan['fixed_issuer_scope']
    limits = plan['gate_before_strategy_registration']
    if len(issuers) != len(set(issuers)):
        raise ValueError('Duplicate fixed issuer')
    for row in rows:
        if row['status'] == USABLE:
            if row['symbol'] not in issuers or row['symbol'] in meta['etfs']:
                raise ValueError('Usable source outside fixed pilot')
            if not row.get('latest_report_id'):
                raise ValueError('Usable source lacks original endpoint identity')
    windows = {'early53': (0, 53), 'recent35': (53, 88),
               'first88': (0, 88), 'later34': (88, 122), 'all122': (0, 122)}
    summaries = {}
    for name, (start, end) in windows.items():
        by_issuer = {}
        for issuer in issuers:
            selected = [r for r in rows if start <= r['signal_index'] < end
                        and r['symbol'] == issuer]
            usable = [r for r in selected if r['status'] == USABLE]
            by_issuer[issuer] = {
                'candidate_rows': len(selected), 'usable_rows': len(usable),
                'distinct_latest_endpoints': len({r['latest_report_id'] for r in usable}),
                'status_counts': dict(Counter(r['status'] for r in selected)),
            }
        coavailable = []
        for i in range(start, end):
            usable_issuers = {r['symbol'] for r in rows if r['signal_index'] == i
                              and r['status'] == USABLE}
            if len(usable_issuers) >= limits['minimum_issuers']:
                coavailable.append(dates[i])
        summaries[name] = {'issuers': by_issuer,
                           'cutoffs_with_minimum_issuers': coavailable}
    qualifying = [s for s, counts in summaries['first88']['issuers'].items()
                  if counts['usable_rows'] >= limits['minimum_usable_candidate_rows_per_qualifying_issuer_in_first88']
                  and counts['distinct_latest_endpoints'] >= limits['minimum_distinct_latest_original_annual_endpoints_per_qualifying_issuer_in_first88']]
    # Coavailability must involve qualifying issuers, not a succession of
    # undersampled issuers that happen to satisfy the raw per-date count.
    overlap = [dates[i] for i in range(88)
               if len({r['symbol'] for r in rows if r['signal_index'] == i
                       and r['status'] == USABLE and r['symbol'] in qualifying})
               >= limits['minimum_issuers']]
    passed = (len(qualifying) >= limits['minimum_issuers'] and len(overlap)
              >= limits['minimum_first88_cutoffs_with_at_least_two_usable_issuers'])
    return {'source_coverage_floor_passed': passed,
            'qualifying_issuers': qualifying, 'qualifying_joint_cutoffs': overlap,
            'segments': summaries, 'rows_verified': len(rows),
            'source_reviews_verified_by_this_function': False,
            'financial_values_or_returns_computed': False,
            'strategy_registration_authorized_by_this_function': False}
