"""Extract explicit modern SP100 table rows; never certify chain completeness."""
import hashlib,json,re
from io import StringIO
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import pandas as pd
from bs4 import BeautifulSoup
P=Path(__file__).resolve().parent

def parse_effective_date(value):
    text=re.sub(r'\s+', ' ', str(value).strip())
    text=re.sub(r',\s*', ', ', text)
    text=re.sub(r'^Sept\.? ', 'Sep ', text)
    text=re.sub(r'^([A-Za-z]{3})\. ', r'\1 ', text)
    if not re.fullmatch(r'[A-Za-z]+ \d{1,2}, (?:19|20)\d{2}', text):
        raise ValueError('Explicit four-digit effective year required: '+text)
    for fmt in ('%B %d, %Y','%b %d, %Y'):
        try:return datetime.strptime(text,fmt).date().isoformat()
        except ValueError:pass
    raise ValueError('Invalid effective date: '+text)

def extract(html):
    soup=BeautifulSoup(html,'html.parser')
    body=soup.select_one('.wd_news_body') or soup.select_one('.wd_body')
    text=body.get_text(' ',strip=True) if body else ''
    # Only the explicit prior-open convention is accepted automatically.
    if not re.search(r'prior to the open(?:ing)? of trading',text,re.I):return []
    result=[]
    for table in soup.find_all('table'):
        if 'Index Name' not in table.get_text():continue
        frame=pd.read_html(StringIO(str(table)))[0]
        headers=[str(x).strip() for x in frame.iloc[0]]
        if not {'Effective Date','Index Name','Action','Ticker'}.issubset(headers):continue
        frame=frame.iloc[1:].copy();frame.columns=headers
        # Blank date cells in publisher tables continue the preceding date group.
        frame['Effective Date']=frame['Effective Date'].replace(r'^\s*$',float('nan'),regex=True).ffill()
        for _,row in frame.iterrows():
            if str(row['Index Name']).strip()!='S&P 100':continue
            if row['Action'] not in ('Addition','Deletion'):raise ValueError('Unknown action')
            ticker=str(row['Ticker']).strip()
            if not re.fullmatch(r'[A-Z][A-Z0-9.\-]*',ticker):raise ValueError('Bad ticker')
            result.append(dict(effective_session=parse_effective_date(row['Effective Date']),action=row['Action'].lower(),ticker=ticker,company=str(row.get('Company Name','')),sector=str(row.get('GICS Sector','')),effective_convention='prior_open',identity_verified=False))
    return result

def main():
    rows=[];review=[]
    for item in json.loads((P/'sp_release_inventory.json').read_text()):
        file=P/item['file'];raw=file.read_bytes();meta=json.loads(file.with_suffix('.provenance.json').read_text())
        if hashlib.sha256(raw).hexdigest()!=meta['sha256']:raise ValueError('Changed cache')
        events=extract(raw.decode())
        for event in events:rows.append(dict(**event,source_url=meta['url'],source_sha256=meta['sha256'],source_file=item['file'],published_date=pd.Timestamp(item['date']).date().isoformat(),published_time_precision='day',published_at=datetime.fromisoformat(parse_effective_date(item['date'])).replace(hour=23,minute=59,second=59,tzinfo=ZoneInfo('America/New_York')).isoformat(),chain_verified=False))
        review.append(dict(file=item['file'],explicit_modern_rows=len(events),requires_other_change_review=True))
    (P/'sp100_modern_event_rows.json').write_text(json.dumps(rows,indent=2)+'\n')
    (P/'sp100_extraction_audit.json').write_text(json.dumps(dict(releases_checked=len(review),explicit_rows=len(rows),effective_dates=len({r['effective_session'] for r in rows}),complete_chain=False,review=review),indent=2)+'\n')
    print('releases',len(review),'explicit rows',len(rows),'dates',len({r['effective_session'] for r in rows}))
if __name__=='__main__':main()
