"""Register a bounded source-only audit; never compute ranks or strategy returns."""
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/monthly_shell_pool_source168'
SOURCE = ROOT / 'outputs/monthly_earnings_repair/batch_health_energy_foreign/raw/SHEL'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    records = []
    for meta in sorted(SOURCE.glob('*.meta.json')):
        record = json.loads(meta.read_text())
        match = re.search(r'q([1-4])[-_](20\d{2})[^/]*qra', record.get('url', ''), re.I)
        if not match:
            continue
        quarter, year = map(int, match.groups())
        if not (2015, 4) <= (year, quarter) <= (2020, 3):
            continue
        stem = meta.name.removesuffix('.meta.json')
        binary = SOURCE / (stem + '.bin')
        assert record['status'] == 200 and binary.read_bytes().startswith(b'%PDF-')
        assert digest(binary) == record['sha256']
        paths = [meta, binary]
        for suffix in ['.txt', '.text.json']:
            path = SOURCE / (stem + suffix)
            if path.exists():
                paths.append(path)
        records.append(dict(year=year, quarter=quarter, url=record['url'],
                            binary=str(binary.relative_to(ROOT)),
                            inputs={str(p.relative_to(ROOT)): digest(p) for p in paths}))
    records.sort(key=lambda row: (row['year'], row['quarter']))
    expected = [(y, q) for y in range(2015, 2021) for q in range(1, 5)
                if (2015, 4) <= (y, q) <= (2020, 3)]
    assert [(r['year'], r['quarter']) for r in records] == expected
    anchors = ['outputs/monthly_2016_extension/extend.py',
               'outputs/monthly_top35_extension/run.py',
               'outputs/monthly_top35_extension/CAP_COLLECTION.json',
               'outputs/dynamic_top100/pilot_identity/SHELL_CLASS_EVIDENCE.json']
    plan = dict(schema='shell_pool_source168_plan_v1',
                registered_utc=datetime.now(timezone.utc).isoformat(),
                scope='Twenty original QRA PDFs, 2015Q4 through 2020Q3; source evidence only.',
                objective='Replace unverified Shell class-cap assumptions only when original evidence supports it; no performance-based choice of pool.',
                records=records, anchors={p: digest(ROOT / p) for p in anchors},
                tasks=['Verify all original bytes and text provenance.',
                       'Extract exact dated A/B issued-share counts and rollforward context, separately from outstanding shares, treasury, employee trusts and weighted-average EPS denominators.',
                       'Record stated report release dates separately from proven historical availability; never backdate late disclosures.',
                       'Identify class-specific quote and original vendor-series evidence still needed.',
                       'Independently review all 20 source decisions; no automatic universe admission.'],
                prohibitions=['No price, cap, rank, factor, holding or return recomputation.',
                              'No replacement of frozen pools, source files, strategies, observation accounts or central experiment queue.',
                              'No network requests inside the source audit.',
                              'Missing or ambiguous evidence stays explicit; no assumed same-class price or invented exact 2.1 multiplier.'],
                strategy_trials_added=0, outcome_selection=False)
    OUT.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(plan, ensure_ascii=False, indent=2) + '\n'
    with (OUT / 'PLAN.json').open('x') as handle:
        handle.write(payload)
    print(json.dumps({'records': len(records), 'plan': str(OUT / 'PLAN.json'),
                      'sha256': digest(OUT / 'PLAN.json')}))


if __name__ == '__main__':
    main()
