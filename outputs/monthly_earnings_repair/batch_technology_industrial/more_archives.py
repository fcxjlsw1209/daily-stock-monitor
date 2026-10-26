from common import *
from urllib.parse import urljoin
from bs4 import BeautifulSoup
# Older Intel URLs do not contain a year; the actual original release body gates admission.
s=BeautifulSoup((HERE/'INTC/raw/archive.bin').read_bytes(),'html.parser')
for a in s.select('a[href]'):
 if a.get_text(' ',strip=True)!='Earnings Release':continue
 u=urljoin(CONFIG['INTC'][1],a['href']).rstrip('/')
 try:b,p,m=fetch('INTC',u);print('INTC',len(b),u,flush=True)
 except Exception as e:print('INTC ERROR',str(e),flush=True)
# Official AMD press archive with explicit year filters exposes pre-2020 releases.
for y in range(2011,2020):
 u='https://ir.amd.com/news-events/press-releases?year='+str(y)
 try:
  b,p,m=fetch('AMD',u,'archive_'+str(y));s=BeautifulSoup(b,'html.parser')
  links={urljoin(u,a['href']).rstrip('/') for a in s.select('a[href]') if 'press-releases/detail/' in a['href'] and re.search(r'\b(?:Reports|Announces)\b.*(?:Quarter|Results)',a.get_text(' ',strip=True),re.I)}
  print('AMD ARCHIVE',y,len(links),flush=True)
  for v in links:
   try:b,p,m=fetch('AMD',v);print('AMD',len(b),v,flush=True)
   except Exception as e:print('AMD ERROR',str(e),flush=True)
 except Exception as e:print('AMD ERROR',str(e),flush=True)
