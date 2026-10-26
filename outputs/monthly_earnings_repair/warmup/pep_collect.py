"""Collect original issuer-distributed PepsiCo releases; no forecast notices."""
import datetime as dt
import json
import re
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from budvz_io import P, fetch, dump

OUT=P/'pep_supplement'
def main():
    links={}
    for page in range(1,23):
        url='https://www.prnewswire.com/news/pepsico%2C-inc./?page='+str(page)
        try:p,m=fetch('pep_supplement',url)
        except Exception as e:
            print('STOP',type(e).__name__,str(e),flush=True);break
        if m['http_status'] in (403,429):break
        if m['http_status']!=200:continue
        soup=BeautifulSoup(p.read_bytes(),'html.parser')
        for a in soup.select('a[href]'):
            href=a['href'];title=a.get_text(' ',strip=True)
            if re.search(r'/news-releases/pepsico-(reports|delivers)-',href) and re.search(r'(quarter|full-year)',href):
                links[urljoin(url,href)]=title
        print('index',page,'releases',len(links),flush=True)
    dump(OUT/'DISCOVERED.json',links)
    rows=[]
    for url,title in links.items():
        try:p,m=fetch('pep_supplement',url)
        except Exception as e:
            print('STOP',type(e).__name__,str(e),flush=True);break
        if m['http_status'] in (403,429):break
        if m['http_status']!=200:continue
        s=BeautifulSoup(p.read_bytes(),'html.parser');a=s
        h=s.select_one('h1');t=' '.join(a.get_text(' ',strip=True).split())
        p.with_suffix('.txt').write_text(t)
        rows.append(dict(url=url,title=h.get_text(' ',strip=True) if h else title,path=str(p),meta=m))
        dump(OUT/'MANIFEST.json',rows)
        print('release',len(rows),title[:90],flush=True)
    dump(OUT/'MANIFEST.json',rows)
if __name__=='__main__':main()
