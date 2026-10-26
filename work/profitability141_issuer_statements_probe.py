from pathlib import Path
import json,re
from bs4 import BeautifulSoup
R=Path.cwd();a=json.loads((R/'work/monthly_profitability141_local_annual_feasibility.json').read_text())
for x in a['annual_sources']:
 if x['fiscal_year'] not in [2015,2016,2026]:continue
 soup=BeautifulSoup((R/x['raw_path']).read_bytes(),'html.parser');text=' '.join(soup.stripped_strings)
 print('SOURCE',x['symbol'],x['fiscal_year'])
 print('WORDS',re.findall(r'.{0,60}(?:U.S. dollars|US dollars|United States dollars|fiscal year begins|fiscal year ended|GAAP|June 30|dollars).{0,70}',text)[:20])
 for row in x['rows']:
  table=soup.find_all('table')[row['table_index_zero_based']]
  prev=table.find_previous(['h1','h2','h3','h4','p']);print('TABLE',row['kind'],'HEADING',prev.get_text(' ',strip=True)[:180] if prev else None)
  print('FIRST', [t.get_text(' ',strip=True) for t in table.find_all('tr')[:4]])
print('ORIGINAL_DIRS',[(p.name,[x.name for x in p.iterdir()][:10]) for p in (R/'outputs').iterdir() if p.is_dir() and any(k in p.name for k in ['nvidia_original','microsoft_original','msft_original'])])
