from pathlib import Path
from datetime import datetime, timezone
import hashlib,json
R=Path(__file__).resolve().parent.parent;D=R/'outputs/monthly_profitability_extension_sources_v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads((D/'MANIFEST.json').read_text());raw=json.loads((D/'RAW_FACTS.json').read_text());checks=json.loads((D/'SOURCE_CHECKS.json').read_text());visual=json.loads((D/'VISUAL_REVIEW.json').read_text())
for rel,h in m['output_hashes'].items():assert sha(D/rel)==h
for rel,h in m['source_hashes'].items():assert sha(R/rel)==h
for rel,h in m['pdf_runtime_hashes'].items():assert sha(Path(rel))==h
for row in raw['rows']:
 assert row['fact_admitted'] is False and row['strategy_admitted'] is False
 assert row['availability']['first_public_version_verified'] is False
 assert row['gross_amount']['date']==row['total_assets']['date']==row['annual_period']['end']
 assert row['annual_period']['start_is_direct_quote'] is False
 assert row['availability']['candidate_first_usable_session']>row['availability']['release_date']
 for x in row.get('rendered_statement_artifacts',[]):assert sha(R/x['path'])==x['sha256']
assert len(raw['rows'])==20 and len(checks['checks'])==20 and len(visual['pages'])==20
result=dict(status='PASS_SOURCE_ONLY_POSTWRITE_INTEGRITY',created_utc=datetime.now(timezone.utc).isoformat(),manifest_sha256=sha(D/'MANIFEST.json'),raw_facts_sha256=sha(D/'RAW_FACTS.json'),source_files_checked=len(m['source_hashes']),runtime_files_checked=len(m['pdf_runtime_hashes']),current_annual_pairs_checked=20,pdf_visual_pages=20,admitted_pairs=0,source_errors=0,pdf_parser_warning_rows=[dict(symbol=x['symbol'],fiscal_year=x['fiscal_year'],warnings=x['parser_warnings'])for x in checks['checks']if x['parser_warnings']],gap=raw['missing_events'],independent_semantic_review_pending=True,factor_rank_order_return_computations=0)
with (D/'POSTWRITE_INTEGRITY.json').open('x')as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result,indent=2));print('POSTWRITE_SHA',sha(D/'POSTWRITE_INTEGRITY.json'));print('TESTS_SHA',sha(D/'TESTS.json'))
