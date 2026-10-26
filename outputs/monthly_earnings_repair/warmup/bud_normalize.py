"""Verify publication datelines and actual period ends in original BUD PDFs."""
import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from budvz_io import P,dump
from bs4 import BeautifulSoup

def early_q1():
    # Original May release supplies publication; the later filed report only
    # corroborates the identity of that already-public period, not feature data.
    pub=P/'bud/raw/5d9815660c99df7b7e2d4010.bin'
    proof=P/'bud/raw/9d0a0ce2dc19bfcffbe260fb.bin'
    if not pub.exists() or not proof.with_suffix('.txt').exists():return None
    pm=json.loads(pub.with_suffix('.json').read_text())
    qm=json.loads(proof.with_suffix('.json').read_text())
    for f,m in [(pub,pm),(proof,qm)]:
        assert m['http_status']==200 and hashlib.sha256(f.read_bytes()).hexdigest()==m['sha256']
    raw=pub.read_text();match=re.search(r'oldPostsContent:("(?:[^"\\]|\\.)*")',raw)
    if not match:raise ValueError('Original first-quarter article body not present')
    text=BeautifulSoup(json.loads(match[1]),'html.parser').get_text(' ',strip=True)
    assert 'BRUSSELS (May 4, 2011)' in text and 'first quarter results for 2011' in text
    t=' '.join(proof.with_suffix('.txt').read_text().split())
    phrase='results for the three-month period ended 31 March 2011, which we filed on Form 6-K with the SEC on 4 May 2011'
    assert phrase in t
    return dict(ticker='BUD',release_date='2011-05-04',publication_timestamp=None,
        precision='date_only',publication_timezone='Europe/Brussels',
        publication_timezone_basis='Brussels dateline in original dated issuer newsroom body.',
        fiscal_period_end_actual='2011-03-31',fiscal_year=2011,fiscal_quarter=1,
        source_url=pm['url'],evidence_path=str(pub),evidence_sha256=pm['sha256'],
        source_kind='issuer_original_newsroom_with_period_identity_corroboration',
        retrieved_at=pm['retrieved_at'],first_public_time_verified=False,
        evidence_excerpt='Original issuer body dated May 4, 2011 announces first-quarter results.',
        additional_evidence=[dict(evidence_path=str(proof),sha256=qm['sha256'],source_url=qm['url'],
            excerpt='Original issuer interim report explicitly identifies the earlier May 4 release as covering the period ended March 31, 2011.')],
        period_basis='Retrospective document cross-reference verifies historical period identity only; no later financial values or forecasts are imported.',
        evidence_limitation='Full original May release attachment unavailable. Publication is proven by its original issuer summary; period identity is corroborated by a later original issuer filing archived by FinancialReports. This does not assert a historical first-public timestamp.')

def parse(text, path, meta):
    text=' '.join(text.split())
    # PDF extraction splits typeset digits; only join within date/year patterns.
    text=re.sub(r'\b(20\d)\s+(\d)\b',r'\1\2',text)
    text=re.sub(r'\b([123])\s+(\d)(?= (?:March|June|September|December))',r'\1\2',text)
    date=re.search(r'Brussels\s*[,/]\s*(\d{1,2}\s+[A-Za-z]+\s+20\d\d)',text[:300])
    if not date: raise ValueError('No original Brussels publication dateline')
    release=dt.datetime.strptime(date[1],'%d %B %Y').date()
    title=re.search(r'InBev reports (?:Full Year and )?(first|second|third|fourth) quarter.{0,60}?(20\d\d)',text[:1500],re.I)
    if not title: raise ValueError('Not a regular quarterly results release')
    q=['first','second','third','fourth'].index(title[1].lower())+1;y=int(title[2])
    matches=list(re.finditer(r'(?:m\s*o\s*n\s*t\s*h\s*s|year)\s+e\s*n\s*d\s*e\s*d\s+(\d{1,2}\s+(?:March|June|September|December)\s+20\d\d)',text,re.I))
    ends=[(dt.datetime.strptime(m[1],'%d %B %Y').date(),m) for m in matches]
    ends=[(d,m) for d,m in ends if d.year==y and (d.month-1)//3+1==q]
    if not ends: raise ValueError('Actual fiscal end missing from financial statement notes')
    period,m=ends[-1]
    assert 0 < (release-period).days < 100
    return dict(ticker='BUD',release_date=str(release),publication_timestamp=None,
        precision='date_only',publication_timezone='Europe/Brussels',
        publication_timezone_basis='Original Brussels dateline; conservative issuer-local calendar availability, no exact first-public time claim.',
        fiscal_period_end_actual=str(period),fiscal_year=y,fiscal_quarter=q,
        source_url=meta['url'],evidence_path=str(path),evidence_sha256=meta['sha256'],
        source_kind='issuer_original_earnings_release_pdf',retrieved_at=meta['retrieved_at'],
        evidence_excerpt=text[:160]+' | '+text[max(0,m.start()-100):m.end()+100],
        period_basis='Actual months-ended date in financial-statement notes, not quarter-name imputation',
        first_public_time_verified=False)

def main():
    events={};rejected=[]
    for f in sorted((P/'bud/raw').glob('*.txt')):
        raw=f.with_suffix('.bin');meta=json.loads(f.with_suffix('.json').read_text())
        try:
            assert hashlib.sha256(raw.read_bytes()).hexdigest()==meta['sha256']
            e=parse(f.read_text(),raw,meta)
            key=(e['fiscal_year'],e['fiscal_quarter'])
            if key in events: assert events[key]['release_date']==e['release_date']
            events[key]=e
        except Exception as ex:rejected.append(dict(path=str(f),reason=str(ex)))
    extra=early_q1()
    if extra:events[(extra['fiscal_year'],extra['fiscal_quarter'])]=extra
    dump(P/'bud/events.json',sorted(events.values(),key=lambda r:r['release_date']))
    dump(P/'bud/AUDIT.json',dict(verified=len(events),rejected=rejected,quarters=[f'{y}Q{q}' for y,q in sorted(events)]))
    print('BUD verified',len(events),'rejected',len(rejected))

if __name__=='__main__':main()
