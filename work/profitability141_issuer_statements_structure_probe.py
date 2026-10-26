from pathlib import Path
import json,re
from bs4 import BeautifulSoup
R=Path.cwd();p=json.loads((R/'outputs/monthly_profitability141_issuer_statements/EXTRACTION_PLAN.json').read_text())
for x in p['source_scope']['sources']:
 if x['symbol']!='MSFT' or x['fiscal_year'] not in [2015,2016,2024,2026]:continue
 soup=BeautifulSoup((R/x['raw_path']).read_bytes(),'html.parser');print('SOURCE',x['fiscal_year'])
 for r in x['rows']:
  t=soup.find_all('table')[r['table_index_zero_based']];print(r['kind'],'ANCESTORS',[(a.name,a.attrs) for a in list(t.parents)[:4]],'CAPTION',t.find('caption'))
  print('BEFORE',[(a.name,a.get_text(' ',strip=True)[:180]) for a in list(t.previous_siblings) if getattr(a,'name',None)][-6:])
print('CALENDAR',json.loads((R/'outputs/monthly_forward_guidance_preflight_v1/SESSIONS.json').read_text()).__class__.__name__)
