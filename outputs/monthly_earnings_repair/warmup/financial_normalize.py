"""Normalize individually evidenced missing financial/consumer releases offline."""
import hashlib,json,re
from pathlib import Path
from bs4 import BeautifulSoup
from dateutil.parser import parse as dateparse
from budvz_io import P,dump
OUT=P/'financial_supplement'
DATE=r'([A-Za-z]+\.?\s+\d{1,2},\s+20\d\d)'
QUARTERS={'first':1,'second':2,'third':3,'fourth':4}

def original(row):
    p=Path(row['path']);m=row['meta']
    if m['http_status']!=200 or hashlib.sha256(p.read_bytes()).hexdigest()!=m['sha256']:
        raise ValueError('Original source checksum/status')
    t=' '.join(p.with_suffix('.txt').read_text().split()) if p.read_bytes().startswith(b'%PDF') else ' '.join(BeautifulSoup(p.read_bytes(),'html.parser').get_text(' ',strip=True).split())
    return p,m,t

def parse(row):
    p,m,t=original(row);symbol=row['ticker']
    if symbol=='MCD':
        pub=re.search(r'(?:OAK BROOK,\s*Ill\.|CHICAGO)\s*,?\s*'+DATE+r'\s*/\s*PRNewswire',t,re.I)
        period=re.search(r'(first|second|third|fourth) quarter(?: and (?:full )?year)? ended\s+'+DATE,t,re.I)
        if not pub or not period:raise ValueError('Original dateline/period absent')
        end=dateparse(period[2]).date();q=QUARTERS[period[1].lower()];year=end.year
        if end.month!=q*3:raise ValueError('Quarter/period mismatch')
    elif symbol=='KO':
        year,q=row['year'],row['quarter']
        pub=re.search(r'ATLANTA,\s*'+DATE,t,re.I)
        period=re.search(r'Consolidated Statements of Income.*?Three Months Ended\s+'+DATE,t,re.I)
        if not pub or not period or f'{year} THIRD QUARTER' not in t.upper() or q!=3:raise ValueError('Coca-Cola quarter evidence absent')
        end=dateparse(period[1]).date()
    elif symbol=='C':
        year,q=row['year'],row['quarter']
        pub=re.search(r'For Immediate Release Citigroup Inc\. \(NYSE:\s*C\)\s*'+DATE,t,re.I)
        # Original fiscal-quarter results plus explicitly dated quarter-end
        # balance sheet/regulatory-capital tables; never the SEC filing date.
        period=re.search(r'(?:December\s+31,\s*'+str(year)+r'|12/31/'+str(year)+r')',t)
        if not pub or not period or f'FOURTH QUARTER {year}' not in t.upper() or q!=4:raise ValueError('Citi quarterly financial-statement evidence absent')
        end=dateparse(period[0]).date()
    elif symbol=='MA':
        year,q=row['year'],row['quarter']
        pub=re.search(r'Purchase,\s*NY\s*[,–−-]\s*'+DATE,t,re.I)
        identity=re.search(r'financial results for the (first|second|third|fourth)[ -]quarter(?: and full.year| of)?\s+'+str(year),t,re.I)
        table=re.search(r'CONSOLIDATED STATEMENTS? OF OPERATIONS.*?(?=CONSOLIDATED BALANCE SHEET|$)',t,re.I)
        if not pub or not identity or QUARTERS[identity[1].lower()]!=q or not table:
            raise ValueError('Mastercard release identity absent')
        period=re.search(r'Three Months Ended ([A-Za-z]+\s+\d{1,2}),',table[0],re.I)
        if not period or str(year) not in table[0]:raise ValueError('Income statement current-year period absent')
        end=dateparse(period[1]+', '+str(year)).date()
        if end.month!=q*3:raise ValueError('Mastercard quarter/period mismatch')
    else:raise ValueError('Unsupported original issuer')
    publication=dateparse(pub[1]).date()
    if not 0<(publication-end).days<110:raise ValueError('Publication/period ordering')
    return dict(ticker=symbol,release_date=str(publication),publication_timestamp=None,precision='date_only',
        publication_timezone='America/Chicago' if symbol=='MCD' else 'America/New_York',
        fiscal_year=year,fiscal_quarter=q,fiscal_period_end_actual=str(end),source_url=row['url'],
        evidence_path=str(p.resolve()),sha256=m['sha256'],retrieved_at=m['retrieved_at'],
        source_kind='issuer_original_earnings_release',publication_version='current_archive',
        first_public_time_verified=False,period_basis=period[0],conference_call_time_used=False,
        training_predecessor_only=(symbol=='KO' and year==2010 and q==3))

def main():
    base=json.loads((P.parent/'integrated_batches/financial_consumer_687/events.json').read_text())
    existing={(r['ticker'],r['fiscal_year'],r['fiscal_quarter']) for r in base}
    rows=[];pending=[]
    for filename in ['MA_MANIFEST.json','MCD_MANIFEST.json']:
        for row in json.loads((OUT/filename).read_text()):
            try:
                event=parse(row);key=(event['ticker'],event['fiscal_year'],event['fiscal_quarter'])
                if key not in existing:rows.append(event);existing.add(key)
            except (ValueError,FileNotFoundError) as e:pending.append(dict(url=row['url'],reason=str(e)))
    for row in json.loads((OUT/'BEGINNING_MANIFEST.json').read_text()):
        # Only original PDFs; incidental HTML and unsuccessful pattern probes
        # stay in the acquisition manifest, not the normalized event stream.
        if not Path(row['path']).read_bytes().startswith(b'%PDF'):continue
        event=parse(row);key=(event['ticker'],event['fiscal_year'],event['fiscal_quarter'])
        if key not in existing:rows.append(event);existing.add(key)
    for filename in ['INDEX_OBSERVATIONS.json','EXTRA_INDEX_OBSERVATIONS.json']:
        p=OUT/filename
        for row in json.loads(p.read_text()):
            key=(row['ticker'],row['fiscal_year'],row['fiscal_quarter'])
            if key in existing:continue
            if row['original_response_available'] is not False:raise ValueError('Indexed evidence mislabeled')
            rows.append(dict(row,publication_timestamp=None,precision='date_only',
                publication_timezone='America/Chicago' if row['ticker']=='ABT' else 'America/New_York',
                evidence_path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                source_kind='public_index_observation_of_original_issuer_release',publication_version='current_archive',
                first_public_time_verified=False,conference_call_time_used=False))
            existing.add(key)
    rows.sort(key=lambda r:(r['ticker'],r['release_date']))
    dump(OUT/'events.json',rows);dump(OUT/'PENDING.json',pending)
    summary=dict(events=len(rows),by_symbol={s:sum(r['ticker']==s for r in rows) for s in sorted(set(r['ticker'] for r in rows))},pending=pending)
    dump(OUT/'STATUS.json',summary);print(json.dumps(summary))
if __name__=='__main__':main()
