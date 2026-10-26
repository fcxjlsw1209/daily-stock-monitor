from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
R=Path(__file__).resolve().parent.parent
D=R/'outputs/monthly_profitability_extension_sources_v1'
I='work/after141_profitability_coverage_options.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(R/I)=='0e845bb922bfa2c44b99260462378bef136fe13aea905d22c4bdcbbc1cfa124f'
i=json.loads((R/I).read_text());specs=[];bindings={I:sha(R/I)}
for option in i['options']:
 if option['symbol'] not in ['AAPL','AMD']:continue
 for s in option['sources']:
  raw=s.get('raw_pdf',s.get('raw_html'))
  keys=['receipt','parent_release','parent_receipt','existing_text_pages'] if option['symbol']=='AAPL' else ['receipt']
  for b in [raw]+[s[k] for k in keys]:
   assert sha(R/b['path'])==b['sha256'];bindings[b['path']]=b['sha256']
  specs.append(dict(symbol=option['symbol'],fiscal_year=s['fiscal_year'],fiscal_year_end=s['fiscal_year_end'],release_date=s['release_date'],source_url=s['source_url'],raw_path=raw['path'],raw_sha256=raw['sha256'],receipt=s['receipt'],parent_release=s.get('parent_release'),parent_receipt=s.get('parent_receipt'),format='pdf' if option['symbol']=='AAPL' else 'html',expected_label=s['numerator_label']))
for rel in ['outputs/monthly_forward_guidance_preflight_v1/CALENDAR_PLAN.json','outputs/monthly_forward_guidance_preflight_v1/SESSIONS.json','outputs/monthly_profitability141_sources/STOPPED_2026-10-19.json','work/SEC_REQUEST_STOP_20261019.json','run_local.sh','local_run.py','local.sandbox.sb']:
 bindings[rel]=sha(R/rel)
cal=json.loads((R/'outputs/monthly_forward_guidance_preflight_v1/CALENDAR_PLAN.json').read_text())
for p,h in cal['bindings'].items():assert sha(R/p)==h;bindings[p]=h
bundle=Path('/Users/shiweili/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/lib/python3.12/site-packages')
runtime={}
for package in ['pypdf','pypdfium2','pypdfium2_raw']:
 for p in sorted((bundle/package).rglob('*')):
  if p.is_file() and (p.suffix in ['.py','.so','.dylib'] or p.name.endswith('.json')):runtime[str(p)]=sha(p)
plan=dict(schema_version=1,created_utc=datetime.now(timezone.utc).isoformat(),status='FROZEN_SOURCE_EXTRACTION_PLAN_NO_FACTOR_ADMISSION',sources=specs,input_hashes=bindings,pdf_runtime_hashes=runtime,pdf_runtime=dict(project_python='3.9.6',pypdf='6.10.0',pypdfium2='5.14.0',bundle_site_packages=str(bundle),execution='./run_local.sh',note='pdfplumber cannot import under pinned Python3.9; use already installed compatible pypdf and PDFium without downloading.'),
 contracts=dict(scope='20 already-local current-annual issuer releases; AAPL2016-2025 and AMD2015-2018,2020-2025. No new requests.',numerator='AAPL monetary Gross margin; AMD2015-18 monetary Gross margin and AMD2020-25 Gross profit; exact current annual GAAP/unadjusted consolidated statement row, never percentages/reconstructed revenue-cost/non-GAAP.',denominator='Same original release consolidated Total assets for precisely current annual end.',amounts='Retain exact displayed text, Decimal string, explicit independent scale, literal currency and source/cell locators. No ratio, normalization for ranking, factor, or return computation.',annual='Require annual header and matching current end. Derive start only as day after explicit prior year end in SAME balance sheet.330-400 inclusive days; mark derived, not quoted.',availability='Preserve issuer-local date-only release metadata and original provenance; next pinned session strictly after release. No historical first-version certification.',identity='Record source issuer/symbol evidence without admitting continuous historical security identity. All new facts remain fact_admitted=false and strategy_admitted=false.',gaps='Retain every registered annual event and explicit parser/semantic gap. AMD2019 missing stays missing, never backfill from later comparisons.',known_semantic_review='AMD2025 has discontinued operations; mandatory explicit review flag. All labels/periods need independent review. Apple old Revenue evidence is not GP/Assets admission.',pdf='Read original .response PDF bytes pinned equal to identity-decoded SHA. Rebuild selected pages and char/word geometry, render and visually inspect relevant statement pages, retain visual artifacts in this new directory.'),
 outputs=['RAW_FACTS.json','SOURCE_CHECKS.json','MANIFEST.json','pdf_artifacts/'],factor_rank_order_return_computations=0,changes_to_141_or_central_state=False)
D.mkdir(exist_ok=False)
p=D/'EXTRACTION_PLAN.json'
with p.open('x') as f:json.dump(plan,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps({'path':str(p.relative_to(R)),'sha256':sha(p),'source_count':len(specs),'source_bindings':len(bindings),'pdf_runtime_bindings':len(runtime)}))
