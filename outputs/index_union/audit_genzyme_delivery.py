"""Validate new primary evidence without changing data, admission or accounts."""
from datetime import date, datetime, timezone
import json
from pathlib import Path

import pandas as pd

from audit_genzyme_price_anchors import checksum, compare_anchor, verify_source

P = Path(__file__).resolve().parent


def assess_receipt(case, liquidation_date, holder_scope):
    """Date-only custody evidence is not an executable opening-price fill."""
    execution = date.fromisoformat(liquidation_date)
    if case['holder_scope'] != holder_scope:
        return 'different_holder_scope_not_applicable'
    exact = case.get('cvr_received_date')
    after = case.get('receipt_after_date_exclusive')
    if exact and after:
        raise ValueError('Receipt cannot have both exact and bound-only precision')
    if exact:
        received = date.fromisoformat(exact)
        if execution < received:
            return 'liquidation_precedes_observed_receipt'
        if execution == received:
            return 'same_day_receipt_time_unknown'
        return 'receipt_date_precedes_liquidation_date_only'
    if after and execution <= date.fromisoformat(after):
        return 'liquidation_precedes_observed_receipt'
    return 'exact_receipt_date_unknown'


def main():
    evidence_path = P / 'sources/oct10_validation/genzyme_delivery_reviewed_facts.json'
    facts = json.loads(evidence_path.read_text())
    if facts.get('retrospective_validation_only') is not True:
        raise ValueError('Later evidence must not become historical signal data')
    sources = []
    for source in facts['sources']:
        path = P / source['source_file']
        verify_source(path, {'sha256': source['source_sha256']})
        sources.append(source)
    quotes = P / 'sources/oct08_validation/GCVRZ_tiingo_prices.json'
    verify_source(quotes, json.loads(quotes.with_suffix('.provenance.json').read_text()))
    frame = pd.DataFrame(json.loads(quotes.read_text()))
    frame['date'] = pd.to_datetime(frame.date, utc=True).dt.strftime('%Y-%m-%d')
    checks = [compare_anchor(frame, anchor) for anchor in facts['anchors']]
    candidate_path = P / 'GENZYME_TERMINAL_CANDIDATE.json'
    candidate = json.loads(candidate_path.read_text())
    receipts = []
    for case in facts['receipt_cases']:
        receipts.append(dict(
            case,
            candidate_liquidation_date=candidate['execution_session'],
            assessment_for_documented_holder=assess_receipt(
                case, candidate['execution_session'], case['holder_scope']),
            assessment_for_generic_strategy_holder=assess_receipt(
                case, candidate['execution_session'], 'generic_strategy_holder')))
    report = dict(
        checked_at=datetime.now(timezone.utc).isoformat(),
        evidence_file=str(evidence_path.relative_to(P)),
        evidence_sha256=checksum(evidence_path), sources=sources,
        quote_sha256=checksum(quotes), candidate_sha256=checksum(candidate_path),
        anchors_checked=len(checks),
        explicitly_identified_price_fields_matched=sum(
            r['price_field_corroborated'] for r in checks),
        price_disagreements=[r for r in checks if not r['value_matches']],
        checks=checks, receipt_cases=receipts,
        documented_holder_receipt_cases=len(receipts),
        general_broker_delivery_certified=False,
        april11_execution_open_independently_corroborated=False,
        march31_quote_overwritten=False,
        candidate_liquidation_policy_still_unverified=True,
        strict_backtest_admitted=False,
        existing_research_results_modified=False,
        original_accounts_modified=False,
        limitations=facts['limitations'])
    (P / 'GENZYME_DELIVERY_AUDIT.json').write_text(
        json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: report[k] for k in (
        'anchors_checked', 'explicitly_identified_price_fields_matched',
        'documented_holder_receipt_cases', 'general_broker_delivery_certified',
        'strict_backtest_admitted')}))
    print(json.dumps({'price_disagreements': report['price_disagreements']}))


if __name__ == '__main__':
    main()
