from bs4 import BeautifulSoup
import json,re
import profitability141_issuer_statements as m
p=json.loads(m.read_pinned({'path':m.PLAN,'sha256':m.PLAN_SHA}))
for s in p['source_scope']['sources']:
 if s['symbol']!='MSFT' or s['fiscal_year'] not in [2018,2021,2026]:continue
 soup=BeautifulSoup(m.read_pinned({'path':s['raw_path'],'sha256':s['raw_sha256']}),'html.parser');text=m.norm(soup.get_text(' ',strip=True));print('YEAR',s['fiscal_year'])
 print('POLICY',[x for x in re.findall(r'.{0,100}GAAP.{0,160}',text) if any(w in x.lower() for w in ['accordance','generally accepted','addition to','reported'])][:5])
 for q in s['rows']:
  table=soup.find_all('table')[q['table_index_zero_based']]
  try:m.extract_cell(soup,q,s['symbol'],s['fiscal_year_end']);print(q['kind'],'pass')
  except Exception as e:
   print(q['kind'],str(e));print('ROWS',[[m.norm(c.get_text(' ',strip=True)) for c in row.find_all(['td','th'],recursive=False)] for row in table.find_all('tr')[:8]])
