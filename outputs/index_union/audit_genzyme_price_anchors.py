"""Independent historical price anchors; evidence only, no admission or trading.

An issuer's unspecified 'quoted price' can equal a vendor close without proving
the closing auction, let alone an opening price on a different date.
"""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import math

import pandas as pd

P = Path(__file__).resolve().parent


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_source(path, provenance):
    if checksum(path) != provenance['sha256']:
        raise ValueError('Historical source checksum mismatch: ' + str(path))


def compare_anchor(frame, anchor):
    field = anchor['comparison_field']
    if field not in ('open', 'close'):
        raise ValueError('Only explicit opening/closing field comparisons supported')
    kind = anchor['source_price_kind']
    if kind not in ('opening_price', 'closing_price', 'quoted_price_unspecified'):
        raise ValueError('Unrecognized primary price type')
    part = frame.loc[frame.date.eq(anchor['date'])]
    observed = float(part.iloc[0][field]) if len(part) == 1 else None
    finite = observed is not None and math.isfinite(observed)
    expected = float(anchor['value'])
    matched = bool(finite and math.isfinite(expected)
                   and abs(observed - expected) <= 0.005)
    field_proven = (kind == 'opening_price' and field == 'open'
                    or kind == 'closing_price' and field == 'close')
    return dict(anchor, vendor_value=observed if finite else None,
                observations=len(part), value_matches=matched,
                price_field_corroborated=bool(matched and field_proven))


def main():
    sources = P / 'sources/oct09_validation'
    genz = P.parent / 'dynamic_top100/additional_sources/free_discovery/qstk/GENZ.csv'
    cvr = P / 'sources/oct08_validation/GCVRZ_tiingo_prices.json'
    receipts = []
    checks = []
    for quotes, receipt_file, provenance_file in [
        (genz, sources / 'genzyme_f4a_daily_prices_reviewed_facts.json',
         genz.with_suffix('.csv.provenance.json')),
        (cvr, sources / 'sanofi_cvr_price_reviewed_facts.json',
         cvr.with_suffix('.provenance.json')),
    ]:
        verify_source(quotes, json.loads(provenance_file.read_text()))
        receipt = json.loads(receipt_file.read_text())
        if not receipt['retrospective_validation_only']:
            raise ValueError('These receipts cannot be promoted to signal inputs')
        if 'source_file' in receipt:
            verify_source(P / receipt['source_file'],
                          dict(sha256=receipt['source_sha256']))
        if quotes == genz:
            frame = pd.read_csv(quotes).rename(columns={
                'Date': 'date', 'Open': 'open', 'Close': 'close'})
        else:
            frame = pd.DataFrame(json.loads(quotes.read_text()))
            frame['date'] = pd.to_datetime(frame.date, utc=True).dt.strftime('%Y-%m-%d')
        for anchor in receipt['anchors']:
            checks.append(dict(compare_anchor(frame, anchor),
                               security_id=receipt['security_id'],
                               symbol=receipt['symbol'],
                               evidence_file=str(receipt_file.relative_to(P))))
        receipts.append(dict(file=str(receipt_file.relative_to(P)),
                             sha256=checksum(receipt_file),
                             quote_file=str(quotes), quote_sha256=checksum(quotes)))

    report = dict(
        checked_at=datetime.now(timezone.utc).isoformat(),
        checks=checks, anchors_checked=len(checks),
        value_matches=sum(r['value_matches'] for r in checks),
        explicitly_identified_price_fields_matched=sum(
            r['price_field_corroborated'] for r in checks),
        evidence=receipts,
        cvr_execution_open_independently_corroborated=False,
        cash_and_cvr_delivery_observed=False,
        strict_backtest_admitted=False,
        existing_research_results_modified=False,
        existing_observation_accounts_modified=False,
        limitations=[
            'Matching isolated anchors does not certify the full daily series.',
            'GCVRZ April4 quoted price equals the vendor close; quote type is unspecified.',
            'No independent April11 open or actual cash/CVR delivery evidence.',
            'Later filing facts are validation only, not historical signal inputs.',
        ])
    target = P / 'GENZYME_PRICE_ANCHORS.json'
    target.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: report[k] for k in (
        'anchors_checked', 'value_matches',
        'explicitly_identified_price_fields_matched',
        'cvr_execution_open_independently_corroborated', 'strict_backtest_admitted')}))


if __name__ == '__main__':
    main()
