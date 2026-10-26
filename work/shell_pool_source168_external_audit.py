"""Audit bounded public retrievals without admitting a market-cap input."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/monthly_shell_pool_source168'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    external = OUT / 'external'
    collection = json.loads((external / 'COLLECTION.json').read_text())
    documents = {}
    for row in collection['records']:
        if 'path' not in row:
            continue
        path = ROOT / row['path']
        assert sha(path) == row['sha256']
        documents[row['name']] = (row, path.read_text(errors='replace'))
    wire, wire_body = documents['issuer_wire_20160729_capital']
    assert wire['status'] == 200
    for text in ['2016-07-29T07:22:00-04:00', '4,325,899,655 A shares',
                 '3,745,486,731 B shares', '8,071,386,386',
                 'no ordinary shares in Treasury', 'not yet cancelled']:
        assert text in wire_body, text
    vendor, vendor_body = documents['current_vendor_history']
    assert vendor['status'] == 200 and 'data = [' in vendor_body
    sec, _ = documents['sec_20160919_issuance']
    ratio, ratio_body = documents['issuer_2016q2_ads_ratio']
    original = [r for r in json.loads((ROOT / 'outputs/monthly_top35_extension/CAP_COLLECTION.json').read_text())['records'] if r['symbol'] == 'SHEL']
    assert len(original) == 1
    legacy = original[0]
    missing = []
    for field, expected_field in [('raw_file', 'raw_sha256'), ('data_file', 'sha256')]:
        path = Path(legacy[field])  # Wrapper relocates historical JSON in memory.
        missing.append(dict(path=str(path), exists=path.exists(), expected_sha256=legacy[expected_field]))
    result = dict(
        schema='shell_source168_external_audit_v1', created_utc=datetime.now(timezone.utc).isoformat(),
        collection_sha256=sha(external / 'COLLECTION.json'),
        issuer_wire=dict(source_sha256=wire['sha256'],
                         stated_publication='2016-07-29T07:22:00-04:00',
                         class_A_issued=4325899655, class_B_issued=3745486731,
                         total_issued=8071386386, treasury_ordinary_shares=0,
                         basis='Issued voting-rights capital, including shares repurchased but not yet cancelled; not certified outstanding shares net of employee trusts.',
                         historical_publication_metadata_observed=True,
                         exact_historical_retrieval_vintage_verified=False),
        vendor=dict(source_sha256=vendor['sha256'], current_body_has_history_series=True,
                    matches_original_raw_sha256=vendor['sha256'] == legacy['raw_sha256'],
                    old_single_class_basis_verified=False,
                    note='Current retrieval is a distinct data vintage. No replacement or reconstruction of original CSV was attempted.'),
        original_vendor_files=missing,
        issuer_ratio_page=dict(status=ratio['status'],
                               explicit_ratio_in_retrieved_body='Each ADS represents two ordinary shares' in ratio_body,
                               note='HTTP success alone is not proof that the page body includes the ratio statement.'),
        sec=dict(status=sec['status'], body_admitted=False, stop_host_for_today=sec.get('stop_host_for_today', False)),
        market_cap_computed=False, universe_admission=False, ranks_computed=False,
        strategy_trials=0, frozen_files_modified=False,
        remaining=['Complete independent twenty-report source audit.',
                   'Resolve the legacy vendor series class basis and its missing exact original files.',
                   'Obtain compatible class-specific quotes and reconcile issued versus outstanding share basis before any replacement cap or ranking.',
                   'Register a separate pool-construction comparison only after a defensible data rule is established.'])
    assert result['issuer_wire']['class_A_issued'] + result['issuer_wire']['class_B_issued'] == result['issuer_wire']['total_issued']
    with (OUT / 'EXTERNAL_AUDIT.json').open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
