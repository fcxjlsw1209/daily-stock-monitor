"""Normalize dated original PepsiCo releases and explicitly labeled index facts."""
import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from dateutil.parser import parse as dateparse
from budvz_io import P,dump
OUT=P/'pep_supplement'
QUARTERS={'first':1,'second':2,'third':3,'fourth':4}

def parse_original(row):
    p=Path(row['path']);meta=row['meta']
    assert hashlib.sha256(p.read_bytes()).hexdigest()==meta['sha256']
    t=p.with_suffix('.txt').read_text();title=row['title']
    if not re.search(r'PepsiCo (Reports|Delivers|Achieves)',title,re.I):raise ValueError('Not actual results title')
    if re.search(r'(announces timing|confirms timing|will (?:issue|release))',title,re.I):raise ValueError('Preannouncement')
    q=re.search(r'(first|second|third|fourth)[ -]quarter',title,re.I)
    y=re.search(r'20\d\d',title)
    if q and y is None:
        y=re.search(r'20\d\d',t[t.find('PURCHASE'):t.find('PURCHASE')+500])
    if 'PepsiCo Achieves 2013 Financial Targets' in title:
        statement=re.search(r'today reported its (fourth)-quarter and full-year (2013) results',t)
        if statement:q=statement;y=re.search('2013',statement[0])
    if not q or not y:raise ValueError('Quarter/year title unclear')
    year=int(y[0]);quarter=QUARTERS[q[1].lower()]
    d=re.search(r'PURCHASE,\s*N\.Y\.?\s*,\s*(.{5,30}?)\s*/\s*PRNewswire(?:-FirstCall)?\s*/',t)
    if not d:raise ValueError('Original dateline absent')
    if re.search(r'20\d\d',d[1]):release=dateparse(d[1]).date()
    else:
        original_wire_date=re.search(r'News provided by PepsiCo, Inc\. ([A-Za-z]+ \d{1,2}, 20\d\d)',t)
        if not original_wire_date:raise ValueError('Dateline year lacks original wire date corroboration')
        release=dateparse(original_wire_date[1]).date()
        partial=dateparse(d[1],default=dt.datetime(release.year,1,1)).date()
        if partial!=release:raise ValueError('Wire date and body dateline disagree')
    # Some issuer-distributed release notices explicitly state results ARE
    # available now and include the actual quarter-end, unlike future notices.
    explicit=re.search(r'(?:quarter|full.year)\s+'+str(year)+r'\s*\((?:quarter )?ended\s+([A-Za-z]+ \d{1,2})(?:st|nd|rd|th)?(?:, '+str(year)+r')?\)',t,re.I)
    period=None;basis=None
    if explicit and re.search('financial results are available',t,re.I):
        period=dateparse(explicit[1]+', '+str(year)).date();basis=explicit[0]
    else:
        statement=re.search(r'(?:Condensed )?Consolidated Statements? of Income.{0,350}',t,re.I)
        if statement:
            m=re.search(r'(?:Weeks Ended(?: \d+ Weeks Ended)?|Quarter Ended(?: Year Ended)?|Year Ended)\s+(\d{1,2}/\d{1,2}/(?:20)?\d{2})',statement[0],re.I)
            if m:period=dateparse(m[1]).date();basis=m[0]
    if period is None:raise ValueError('Actual period end absent')
    if period.year!=year or not 0<(release-period).days<100:raise ValueError('Period/publication inconsistent')
    return make(year,quarter,release,period,row['url'],p,meta['retrieved_at'],
                'issuer_distributed_original_prnewswire_release',basis)

def make(year,quarter,release,period,url,path,retrieved,kind,basis):
    return dict(ticker='PEP',fiscal_year=year,fiscal_quarter=quarter,
      release_date=str(release),publication_timestamp=None,precision='date_only',
      publication_timezone='America/New_York',fiscal_period_end_actual=str(period),
      source_url=url,evidence_path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
      retrieved_at=retrieved,source_kind=kind,period_basis=basis,
      publication_version='current_archive',first_public_time_verified=False,
      evidence_limitation=('Public index observation; original PDF bytes unavailable.' if 'index' in kind else 'Original issuer-distributed release; current archive, not historical capture.'),
      conference_call_time_used=False)

def main():
    rows={};pending=[]
    originals=json.loads((OUT/'MANIFEST.json').read_text())
    for r in json.loads((OUT/'MEDIAROOM.json').read_text()):
        from bs4 import BeautifulSoup
        soup=BeautifulSoup(Path(r['path']).read_bytes(),'html.parser')
        h=soup.select_one('h1') or soup.select_one('title')
        if h:r['title']=h.get_text(' ',strip=True);originals.append(r)
    for r in originals:
        try:e=parse_original(r);rows[(e['fiscal_year'],e['fiscal_quarter'])]=e
        except ValueError as exc:pending.append(dict(url=r['url'],reason=str(exc)))
    for p in sorted(OUT.glob('indexed_*.json'))+[OUT/'INDEXED_PDF_OBSERVATIONS.json']:
        j=json.loads(p.read_text());items=j if isinstance(j,list) else [j]
        for o in items:
            if 'fiscal_year' in o:y,q=o['fiscal_year'],o['fiscal_quarter']
            else:
                m=re.search(r'/q(\d)-(20\d\d)-',o['source_url']);q,y=int(m[1]),int(m[2])
            pub=o.get('observed_publication_date') or o.get('observed_publication_date_text')
            end=o.get('observed_period_end') or o.get('observed_period_end_text')
            if not pub or not end:pending.append(dict(url=o['source_url'],reason='Index dates incomplete'));continue
            release,period=dateparse(pub).date(),dateparse(end).date()
            assert period.year==y and 0<(release-period).days<100
            if (y,q) not in rows:rows[(y,q)]=make(y,q,release,period,o['source_url'],p,o['retrieved_at'],o['evidence_kind'],'Original PDF financial-statement/end-period excerpt in observation JSON.')
    # Four already validated PEP events reside in the financial-consumer batch.
    existing={(2019,1),(2019,2),(2024,4),(2025,4)}
    result=[v for k,v in sorted(rows.items()) if k not in existing and k>=(2010,3)]
    for row in result:
        if (row['fiscal_year'],row['fiscal_quarter'])==(2010,3):row['training_predecessor_only']=True
    complete=set(rows)|existing
    missing=[dict(year=y,quarter=q) for y in range(2010,2027) for q in range(1,5)
             if (2010,4)<=(y,q)<=(2026,2) and (y,q) not in complete]
    dump(OUT/'events.json',result);dump(OUT/'PARSE_EXCLUSIONS.json',pending);dump(OUT/'PENDING.json',missing)
    dump(OUT/'STATUS.json',dict(events=len(result),existing_batch_events=4,combined_quarters=len(result)+4,
        quarters=[str(x['fiscal_year'])+'Q'+str(x['fiscal_quarter']) for x in result],missing_quarters=missing,
        alternative_source_parse_exclusions=len(pending),semantics='No expected preliminary updates certified; regular quarterly sequence only.'))
    print(json.dumps(dict(events=len(result),combined_quarters=len(result)+4,missing_quarters=missing)))
if __name__=='__main__':main()
