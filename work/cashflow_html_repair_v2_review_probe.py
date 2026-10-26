from pathlib import Path
from bs4 import BeautifulSoup
for year,ti,sel in [(2018,9,list(range(8))),(2021,6,list(range(39,49))+[63])]:
 p=Path('outputs/monthly_earnings_repair/microsoft/raw/FY%dQ4.html'%year)
 soup=BeautifulSoup(p.read_bytes(),'html.parser');table=soup.find_all('table')[ti]
 rows=[r for r in table.find_all('tr')if r.find_parent('table')is table]
 print(year,'table',ti,'predecessors',list(table.previous_siblings)[:3])
 for ri in sel:
  print(ri,[(c.get('rowspan'),c.get('colspan'),' '.join(c.get_text(' ',strip=True).split()))for c in rows[ri].find_all(['td','th'],recursive=False)])
