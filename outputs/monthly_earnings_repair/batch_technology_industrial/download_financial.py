from common import *
from urllib.parse import urljoin
from bs4 import BeautifulSoup
import concurrent.futures

def run(k):
 for y in range(2011,2027):
  u=CONFIG[k][1]+'/news-events/press-releases?year='+str(y)+'&category=financial'
  try:
   b,p,m=fetch(k,u,'financial_'+str(y));s=BeautifulSoup(b,'html.parser')
   links={urljoin(u,a['href']).rstrip('/') for a in s.select('a[href]') if 'press-releases/detail/' in a['href'] and re.search(r'\b(?:Reports|Announces)\b.*(?:Quarter|Results)',a.get_text(' ',strip=True),re.I) and not re.search('to report|date|conference',a.get_text(' ',strip=True),re.I)}
   print(k,'ARCHIVE',y,len(links),flush=True)
   for v in links:
    try:b,p,m=fetch(k,v);print(k,'DOC',len(b),v,flush=True)
    except Exception as e:print(k,'ERROR',str(e),flush=True)
  except Exception as e:print(k,'ERROR',str(e),flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(run,['INTC','AMD']))
s=BeautifulSoup((HERE/'INTC/raw/archive.bin').read_bytes(),'html.parser')
for u in sorted({a['href'] for a in s.select('a[href]') if '/earnings_release/' in a['href']}):
 try:b,p,m=fetch('INTC',u);print('INTC PDF',len(b),flush=True)
 except Exception as e:print('INTC ERROR',str(e),flush=True)
