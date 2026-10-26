from pathlib import Path
import json,hashlib
R=Path.cwd();p=R/'work/profitability141_identity_admission_review.json';d=json.loads(p.read_text());plan=R/'outputs/monthly_profitability141_sources/DATA_PLAN.json';a=json.loads(plan.read_text())
assert {x['allowed_cik'] for x in d['rows']}==set(a['allowed_ciks']) and len(d['rows'])==43
assert d['input_bindings']['data_plan']['sha256']==hashlib.sha256(plan.read_bytes()).hexdigest()
assert len(d['decisions']['dated_primary_issuer_ticker_supported_subset'])==3
assert len(d['decisions']['must_remain_neutral_pending_historical_security_admission'])==40
assert set(d['decisions']['dated_primary_issuer_ticker_supported_subset']).isdisjoint(d['decisions']['must_remain_neutral_pending_historical_security_admission'])
assert len(d['source_checks'])==116 and all(x['migration_manifest_hash_matches'] is True for x in d['source_checks'])
assert all(x['original_receipt_hash_matches'] is True for x in d['source_checks'])
assert all(not x['full_historical_security_admission'] for x in d['rows'])
assert len(d['focused_original_identity_evidence']['annual_issuer_ticker_observations'])==24
assert d['scope']['ratios_scores_orders_returns_computed']==0 and not d['decisions']['financial_facts_or_strategy_candidates_admitted']
print(json.dumps({'passed':True,'allowed_set_exact':43,'all_source_hash_and_receipt_checks':116,'primary_annual_observations':24,'plan_unchanged':True,'report_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}))
