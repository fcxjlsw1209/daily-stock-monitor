"""Disney quarter identity and original release header, not filing/URL dates."""
import hashlib,json,re
from pathlib import Path
from dateutil.parser import parse as dateparse
from budvz_io import P,dump
OUT=P/'dis_supplement'
DATE=r'([A-Za-z]+\s+\d{1,2},\s+20\d\d)'
QUARTERS={1:'FIRST',2:'SECOND',3:'THIRD',4:'FOURTH'}
def parse(row):
    p=Path(row['path']);m=row['meta'];y,q=row['year'],row['quarter']
    if m['http_status']!=200 or not p.read_bytes().startswith(b'%PDF'):raise ValueError('No original PDF')
    if hashlib.sha256(p.read_bytes()).hexdigest()!=m['sha256']:raise ValueError('Source checksum')
    t=' '.join(p.with_suffix('.txt').read_text().split())
    visual=OUT/'VISUAL_FIRST_PAGES.json'
    notes=json.loads(visual.read_text()) if visual.exists() else {}
    note=notes.get(f'{y}Q{q}')
    if note:
        if note['original_pdf_sha256']!=m['sha256']:raise ValueError('Visual evidence PDF checksum')
        image=Path(note['image_path'])
        if hashlib.sha256(image.read_bytes()).hexdigest()!=note['image_sha256']:raise ValueError('Visual page checksum')
        t=note['verified_transcription']+' '+t
    release=re.search(r'FOR IMMEDIATE RELEASE\s*'+DATE,t[:500],re.I)
    if not release:
        release=re.match(DATE+r'\s+THE WALT DISNEY COMPANY REPORTS',t,re.I)
    letter=re.match(DATE+r'\s+To Our Shareholders and the Broader Investment Community',t,re.I)
    if not release and letter:release=letter
    if not release:raise ValueError('Original release dateline absent')
    title=t[:650].upper()
    if letter:
        identity=re.search(r'The following table summarizes '+QUARTERS[q]+r' quarter results for fiscal '+str(y),t,re.I)
        if not identity:raise ValueError('Shareholder letter fiscal quarter mismatch')
    elif not (QUARTERS[q]+' QUARTER' in title and 'FISCAL '+str(y) in t[:2200].upper()):raise ValueError('Fiscal quarter identity mismatch')
    period=re.search(r'(?:quarter|year) ended\s+'+DATE,t[:1800],re.I)
    if not period and letter:
        period=re.search(r'Quarter Ended (?:Six|Nine) Months Ended \(\$ in millions, except per share amounts\) '+DATE,t[identity.end():],re.I)
    if not period:raise ValueError('Actual period end absent')
    pub,end=dateparse(release[1]).date(),dateparse(period[1]).date()
    if not 0<(pub-end).days<100:raise ValueError('Publication/period mismatch')
    return dict(ticker='DIS',release_date=str(pub),publication_timestamp=None,precision='date_only',
        publication_timezone='America/Los_Angeles',fiscal_year=y,fiscal_quarter=q,
        fiscal_period_end_actual=str(end),source_url=row['url'],evidence_path=str(p.resolve()),sha256=m['sha256'],
        retrieved_at=m['retrieved_at'],source_kind='issuer_original_earnings_release_pdf',
        publication_version='current_archive',first_public_time_verified=False,
        period_basis=period[0],conference_call_time_used=False,
        visual_first_page_evidence=note)
def main():
    existing={(r['fiscal_year'],r['fiscal_quarter']) for r in json.loads((P.parent/'integrated_batches/financial_consumer_628/events.json').read_text()) if r['ticker']=='DIS'}
    rows=[];pending=[]
    observed_file=OUT/'INDEXED_RELEASE_OBSERVATIONS.json'
    observed=json.loads(observed_file.read_text())
    indexed={(r['fiscal_year'],r['fiscal_quarter']):r for r in observed}
    for row in json.loads((OUT/'MANIFEST.json').read_text()):
        if (row['year'],row['quarter']) in existing:continue
        try:rows.append(parse(row))
        except (ValueError,FileNotFoundError) as e:
            obs=indexed.get((row['year'],row['quarter']))
            if not obs:pending.append(dict(year=row['year'],quarter=row['quarter'],reason=str(e)));continue
            assert obs['facts_verified'] and obs['original_response_available'] is False
            rows.append(dict(obs,ticker='DIS',publication_timestamp=None,precision='date_only',publication_timezone='America/Los_Angeles',
                evidence_path=str(observed_file.resolve()),sha256=hashlib.sha256(observed_file.read_bytes()).hexdigest(),
                source_kind='public_index_observation_of_original_issuer_release',publication_version='current_archive',
                first_public_time_verified=False,conference_call_time_used=False))
    dump(OUT/'events.json',rows);dump(OUT/'PENDING.json',pending)
    dump(OUT/'STATUS.json',dict(events=len(rows),existing=len(existing),pending=len(pending)))
    print(json.dumps(dict(events=len(rows),existing=len(existing),pending=len(pending))))
if __name__=='__main__':main()
