from pathlib import Path
import hashlib,json
from after141_aapl_source_extract import extract
R=Path(__file__).resolve().parent.parent
plan=json.loads((R/'outputs/monthly_profitability_extension_sources_v1/EXTRACTION_PLAN.json').read_text())
for p,h in plan['pdf_runtime_hashes'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
for s in plan['sources']:
 if s['symbol']=='AAPL':
  try:
   raw=(R/s['raw_path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==s['raw_sha256']
   o=extract(raw,s,plan['pdf_runtime']['bundle_site_packages'])
   print(json.dumps({'year':s['fiscal_year'],'ok':True,'days':o['annual_period']['days_inclusive']}))
  except Exception as e:print(json.dumps({'year':s['fiscal_year'],'ok':False,'error':str(e)}))
