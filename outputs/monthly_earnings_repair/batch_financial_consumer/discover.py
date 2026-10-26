import json,re
from bs4 import BeautifulSoup
import core as c
for ticker,config in json.loads((c.HERE/'sources.json').read_text()).items():
 try:
  b,p,m=c.fetch(config['url'],ticker,'landing');s=BeautifulSoup(b,'html.parser')
  links=[dict(text=a.get_text(' ',strip=True)[:140],url=a['href']) for a in s.find_all('a',href=True) if any(k in a['href'].lower()+' '+a.get_text().lower() for k in ('earning','quarter','sec-filing','.pdf','press release'))]
  scripts=[a['src'] for a in s.find_all('script',src=True) if any(x in a['src'].lower() for x in ('q4','api','drupal','eqcdn','s3.amazonaws'))]
  c.dump(c.HERE/ticker/'discovery.json',dict(landing_meta=m,links=links,scripts=scripts))
  print(ticker,m['status'],m.get('final_url'),'links',len(links),'scripts',scripts,flush=True)
 except Exception as e:print(ticker,'ERROR',str(e),flush=True)
