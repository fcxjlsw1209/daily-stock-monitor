"""Freeze a source-only balance-sheet audit before extracting new amounts."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/monthly_liquidity162_sources'
BASE = ROOT / 'outputs/monthly_earnings_repair'

def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    state_path = ROOT / 'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
    state = json.loads(state_path.read_text())
    assert state['current'] == 'monthly_rank161'
    assert state['current_status'] == 'completed_independently_verified'
    assert len(state['completed']) == 161 and not state['registered_queue']
    assert sum(x['trials'] for x in state['completed']) == 998
    pins = {str(Path(__file__).resolve()): digest(__file__)}
    records = []
    for folder, ticker in [('microsoft', 'MSFT'), ('nvidia', 'NVDA')]:
        events_path = BASE / folder / 'events.json'
        pins[str(events_path)] = digest(events_path)
        events = json.loads(events_path.read_text())
        rows = [x for x in events if x['fiscal_quarter'] == 4 and 2015 <= x['fiscal_year'] <= 2026]
        assert len(rows) == 12 and sorted(x['fiscal_year'] for x in rows) == list(range(2015, 2027))
        if ticker == 'NVDA':
            manifest_path = BASE / folder / 'manifest.json'
            manifest = json.loads(manifest_path.read_text())
            pins[str(manifest_path)] = digest(manifest_path)
        for row in sorted(rows, key=lambda x:x['fiscal_year']):
            raw = Path(row['evidence_path'])
            if not raw.is_absolute():
                raw = BASE / folder / raw
            assert raw.is_relative_to(ROOT) and raw.is_file() and not raw.is_symlink()
            expected = row.get('sha256', row.get('evidence_sha256'))
            assert digest(raw) == expected
            pins[str(raw)] = expected
            if ticker == 'MSFT':
                mp = raw.with_suffix('.meta.json')
                source = json.loads(mp.read_text())
                pins[str(mp)] = digest(mp)
            else:
                source = manifest[row['source_url']]
            assert source['http_status'] == 200 and source['sha256'] == expected
            assert source['url'] == row['source_url'] and source.get('retrieved_at')
            assert row['precision'] == 'date_only' and row['publication_timestamp'] is None
            assert row['publication_timezone'] == 'America/Los_Angeles'
            records.append(dict(
                ticker=ticker, fiscal_year=row['fiscal_year'], fiscal_quarter=4,
                fiscal_period_end=row['fiscal_period_end_actual'], release_date=row['release_date'],
                precision='date_only', publication_timezone='America/Los_Angeles',
                raw_path=str(raw), raw_sha256=expected, source_url=row['source_url'],
                retrieved_at=source['retrieved_at']))
    calendar = ROOT / 'outputs/monthly_paired_target143/CALENDAR.json'
    pins[str(calendar)] = digest(calendar)
    c = json.loads(calendar.read_text())
    for p,h in c['provenance']['source_hashes'].items():
        assert Path(p).is_relative_to(ROOT) and digest(p) == h
        pins[p] = h
    plan = dict(
        schema='liquidity162_source_scope_v1', status='SOURCE_AUDIT_REGISTERED',
        created_utc=datetime.now(timezone.utc).isoformat(), source_records=records,
        source_hashes=pins, parent_completed_round=161, strategy_trial_registered=False,
        source_scope='Exactly FY2015-FY2026 original Q4 releases for MSFT and NVDA; no outcome-driven expansion.',
        hypothesis='Reported current debt versus liquid financial assets may describe a financing constraint distinct from earnings quality or total indebtedness.',
        required_fields=['current_liquid_financial_assets','explicit_current_debt_rows',
                         'explicit_noncurrent_debt_rows','total_assets'],
        extraction_rules=[
            'Use the balance-sheet current period column matching the recorded fiscal period end, never the comparative prior-period or cash-flow column.',
            'Retain exact table/row/column text, currency, scale and raw tokens alongside parsed values.',
            'Absent or ambiguous rows, blank cells and dash tokens are unavailable, not zero.',
            'Do not count both a total and its components; do not add commercial paper separately if embedded in short-term debt.',
            'Keep lease liabilities and restricted cash separately identified; do not silently blend their definitions into debt or liquid assets.',
            'An issuer-specific reported aggregate may be conditionally usable as a proxy, but cannot certify all debt due in the next year or all freely available cash.',
            'Require independent field/period/unit/provenance review before declaring any record usable.',
            'Do not infer missing current debt from balancing residuals or later filings.',
            'Known-by is no earlier than the close of the first exchange session strictly after the date-only local release day.',
            'Retain all 24 successes and rejects; no score/holdings/returns calculation in this audit.'
        ],
        limitations=['Later-retrieved document vintages; complete historical revision chain is unverified.',
                     'Only two issuers in this source pilot; no claim of full historical universe or factor coverage.',
                     'No exact first-publication clock for these releases.',
                     'Existing estimated historical universe limitations remain unchanged.'],
        numerical_factors_computed=0, orders_computed=0, returns_computed=0,
        no_observer_writes=True, no_network=True)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'SOURCE_PLAN.json'
    with path.open('x') as f:
        json.dump(plan, f, indent=2, sort_keys=True)
        f.write('\n')
    print(json.dumps(dict(registered_sources=len(records), source_plan=str(path), sha256=digest(path))))

if __name__ == '__main__':
    main()
