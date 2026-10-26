"""Review two independent raw-close anchors without changing data admission."""
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

from audit_genzyme_price_anchors import checksum, compare_anchor, verify_source

P = Path(__file__).resolve().parent


def main():
    facts_path = P / 'sources/oct11_validation/kraft_proxy_reviewed_facts.json'
    facts = json.loads(facts_path.read_text())
    if not facts['retrospective_validation_only']:
        raise ValueError('Later filing must remain retrospective verification only')
    quote_path = P / 'sources/pystock/daily_extracted/KRFT.csv'
    verify_source(quote_path, json.loads(quote_path.with_suffix('.provenance.json').read_text()))
    frame = pd.read_csv(quote_path)
    if set(frame.symbol) != {'KRFT'}:
        raise ValueError('Unexpected security in quote archive')
    checks = [compare_anchor(frame, a) for a in facts['anchors']]
    report = dict(
        checked_at=datetime.now(timezone.utc).isoformat(),
        security_id=facts['security_id'], symbol='KRFT', checks=checks,
        anchors_checked=len(checks),
        explicit_raw_close_matches=sum(x['price_field_corroborated'] for x in checks),
        evidence=dict(review_file=str(facts_path.relative_to(P)),
                      review_sha256=checksum(facts_path),
                      quote_file=str(quote_path.relative_to(P)),
                      quote_sha256=checksum(quote_path),
                      primary_url=facts['source_url'],
                      full_primary_document_archived=False),
        entire_series_certified=False, adjustment_factors_certified=False,
        ndx_membership_continuity_verified=False, strict_backtest_admitted=False,
        prices_or_membership_modified=False, new_performance_run=False,
        limitations=facts['limitations'])
    (P / 'KRFT_PRICE_ANCHORS.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: report[k] for k in (
        'anchors_checked', 'explicit_raw_close_matches',
        'ndx_membership_continuity_verified', 'strict_backtest_admitted')}))


if __name__ == '__main__':
    main()
