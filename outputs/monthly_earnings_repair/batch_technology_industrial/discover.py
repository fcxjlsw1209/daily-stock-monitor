from common import *
from urllib.parse import urljoin
paths={'INTC':'/financial-info/financial-results','AMD':'/financial-information/financial-results','KLAC':'/financial-information/financial-results','PLTR':'/bundle.js','ACN':'/filings-and-reports/earnings-reports','LRCX':'/quarterly-results','SKHY':'/en/wp-json/wp/v2/posts?search=financial%20results&per_page=100'}
for k,p in paths.items():
 try:
  b,f,m=fetch(k,urljoin(CONFIG[k][1],p),'archive');print(k,len(b),flush=True)
 except Exception as e:print(k,str(e),flush=True)
