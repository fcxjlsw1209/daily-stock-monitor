from pathlib import Path
import json
from bs4 import BeautifulSoup
R=Path(__file__).resolve().parent.parent
p=json.loads((R/'outputs/monthly_profitability_extension_sources_v1/EXTRACTION_PLAN.json').read_text())
i=json.loads((R/'work/after141_profitability_coverage_options.json').read_text())
print('AMD_INDEX', {k:v for k,v in i['options'][1].items()if k!='sources'})
e=json.loads((R/'outputs/monthly_sec_apple_original_admission_v1/EVIDENCE.json').read_text())
print('AAPL_EVIDENCE_KEYS',list(e))
for acc,r in e['releases'].items():
 if r['publication'].get('fiscal_quarter')==4:
  print('FIRST_APPLE_ANNUAL',acc,r);break
for s in p['sources']:
 if s['symbol']=='AMD':
  print('AMD_RECEIPT',json.loads((R/s['receipt']['path']).read_text()));break
for s in p['sources']:
 if s['symbol']=='AAPL':
  soup=BeautifulSoup((R/s['parent_release']['path']).read_bytes(),'html.parser')
  print('APPLE_META',s['fiscal_year'],[(x.get('property')or x.get('name'),x.get('content')) for x in soup.find_all('meta') if any(z in (x.get('property','')+x.get('name','')).lower()for z in ['date','time'])],[(x.get('class'),x.get_text(' ',strip=True)) for x in soup.select('.category-eyebrow__date')])
