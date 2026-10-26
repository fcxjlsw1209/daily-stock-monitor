"""Discover original McDonald's Cision releases; don't use monthly sales notices."""
import json,re
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from budvz_io import P,fetch,dump
OUT=P/'financial_supplement'
def main():
    rows=[];seen=set()
    for y in range(2011,2027):
        u=f'https://mcdonalds.mediaroom.com/index.php?s=2429&year={y}&l=100'
        f,m=fetch('financial_supplement',u)
        if m['http_status'] in (403,429):break
        if m['http_status']!=200:continue
        for a in BeautifulSoup(f.read_bytes(),'html.parser').select('a[href]'):
            title=' '.join(a.stripped_strings);url=urljoin(u,a['href'])
            if not re.search(r'(?:results|quarter earnings)',title,re.I) or re.search(r'(?:plans|schedule|webcast|call|announce.*date)',title,re.I):continue
            if url in seen:continue
            seen.add(url);p,meta=fetch('financial_supplement',url)
            rows.append(dict(ticker='MCD',title=title,url=url,path=str(p),meta=meta));dump(OUT/'MCD_MANIFEST.json',rows)
            print(y,title,meta['http_status'],flush=True)
            if meta['http_status'] in (403,429):return
if __name__=='__main__':main()
