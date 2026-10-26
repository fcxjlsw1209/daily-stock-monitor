from pathlib import Path
from bs4 import BeautifulSoup
s=BeautifulSoup(Path('outputs/monthly_earnings_repair/microsoft/raw/FY2018Q4.html').read_bytes(),'html.parser')
t=s.find_all('table')[9]
for i,n in enumerate(t.parent.contents):
 print(i,getattr(n,'name',None),str(n)[:700])
