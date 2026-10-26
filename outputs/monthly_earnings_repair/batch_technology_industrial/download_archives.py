from common import *
from urllib.parse import urljoin
from bs4 import BeautifulSoup
import concurrent.futures

def run(k):
 try:
  b=(HERE/k/'raw/archive.bin').read_bytes();s=BeautifulSoup(b,'html.parser');links={}
  for a in s.select('a[href]'):
   label=a.get_text(' ',strip=True);u=urljoin(CONFIG[k][1]+'/',a['href']).rstrip('/')
   if k in ['INTC','AMD','KLAC'] and 'press-releases/detail/' in u and re.search('reports|announces',u) and re.search('20(?:1[0-9]|2[0-6])',u):links[u]=label
   if k=='LRCX' and label.upper()=='PRESS RELEASE':links[u]=label
  dump(HERE/k/'DISCOVERY.json',dict(source_archive=rel(HERE/k/'raw/archive.bin'),role='Links only; release dates and period ends require original bodies',links=links))
  for u,label in links.items():
   try:
    b,p,m=fetch(k,u);print(k,'DOC',len(b),u,flush=True)
   except Exception as e:print(k,'ERROR',str(e),flush=True)
  print(k,'TOTAL',len(links),flush=True)
 except Exception as e:print(k,'ERROR',str(e),flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(run,['INTC','AMD','KLAC','LRCX']))
for k,u,n in [('PLTR','https://investors.palantir.com/feed/PressRelease.svc/GetPressReleaseList?languageId=1&bodyType=1&year=-1&includeTags=true&pressReleaseDateFilter=1&pageSize=-1','news_feed'),('ACN','https://investor.accenture.com/filings-and-reports/earnings-reports/module-holder/earnings-reports-documents?async=1','async')]:
 try:b,p,m=fetch(k,u,n);print(k,n,len(b),flush=True)
 except Exception as e:print(k,'ERROR',str(e),flush=True)
