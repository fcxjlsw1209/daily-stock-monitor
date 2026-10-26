"""Discover VZ original release/table links through issuer quarter archive."""
import json
import re
from pathlib import Path
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from budvz_io import fetch, dump, P

BASE='https://www.verizon.com'

def main():
    index=P.parent/'batch_financial_consumer/VZ/raw/quarter_index.html'
    soup=BeautifulSoup(index.read_bytes(),'html.parser')
    urls={}
    for a in soup.select('a[href]'):
        m=re.fullmatch(r'([1-4])Q\s+(20\d\d)',a.get_text(' ',strip=True))
        if m: urls[(int(m[2]),int(m[1]))]=urljoin(BASE,a['href'])
    manifest=[]
    for y in range(2010,2027):
        for q in range(1,5):
            if (y,q)<(2010,4) or (y,q)>(2026,2): continue
            url=urls.get((y,q),BASE+f'/about/investors/quarterly-reports/{q}q-{y}-quarter-earnings-conference-call-webcast')
            row=dict(year=y,quarter=q,archive_url=url)
            try:
                f,m=fetch('vz',url)
                if m['http_status']!=200: raise ValueError('archive HTTP '+str(m['http_status']))
                s=BeautifulSoup(f.read_bytes(),'html.parser')
                main=s.select_one('main') or s
                row.update(archive_path=str(f),archive_sha256=m['sha256'],
                    title=(s.select_one('h1').get_text(' ',strip=True) if s.select_one('h1') else ''),
                    links=[dict(label=a.get_text(' ',strip=True),url=urljoin(BASE,a['href'])) for a in main.select('a[href]') if '/news/' in a['href'] or '/download' in a['href'] or '.pdf' in a['href']])
                for link in row['links']:
                    f,meta=fetch('vz',link['url']);link.update(path=str(f),meta=meta)
            except Exception as exc: row['error']=str(exc)
            manifest.append(row);dump(P/'vz/MANIFEST.json',manifest)
            print(y,q,len(row.get('links',[])),row.get('error',''),flush=True)

if __name__=='__main__':main()
