from pathlib import Path
import json,re
from bs4 import BeautifulSoup
R=Path.cwd();plan=json.loads((R/'outputs/monthly_profitability141_issuer_statements/EXTRACTION_PLAN.json').read_text())
for x in plan['source_scope']['sources']:
 soup=BeautifulSoup((R/x['raw_path']).read_bytes(),'html.parser');alltext=' '.join(soup.stripped_strings)
 item={'symbol':x['symbol'],'year':x['fiscal_year'],'rows':[],'currency_context':re.findall(r'.{0,80}(?:U\.S\. dollar|US dollar|United States dollar|U\.S\. currency).{0,80}',alltext,flags=re.I)[:2]}
 for q in x['rows']:
  table=soup.find_all('table')[q['table_index_zero_based']];rows=table.find_all('tr');row=rows[q['row_index_zero_based']];cells=row.find_all(['td','th'],recursive=False)
  heading=table.find_previous(string=re.compile(r'INCOME STATEMENTS|BALANCE SHEETS|STATEMENTS OF INCOME',re.I))
  item['rows'].append({'kind':q['kind'],'row':row.get_text(' ',strip=True),'cell':cells[q['matching_cells'][0]['source_cell_index']].get_text(' ',strip=True),'heading':str(heading).strip()[:110] if heading else None,'first_headers':[r.get_text(' ',strip=True) for r in rows[:8]] if x['symbol']=='NVDA' else None})
 print(json.dumps(item))
