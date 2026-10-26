"""Offline Merck/Oracle gap normalization; no trading or calendar approximations."""
from pathlib import Path
import datetime as dt
import hashlib,json,re
from dateutil.parser import parse

P=Path(__file__).resolve().parent
ROOT=P.parents[3]
MDY=r'([A-Za-z]+\.?\s+\d{1,2},\s+20\d\d)'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(name,data):(P/name).write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')

def common(ticker,y,q,published,end,url,path,source_kind):
    if not 0<(dt.date.fromisoformat(published)-dt.date.fromisoformat(end)).days<110:
        raise ValueError('Publication/fiscal-end ordering')
    return dict(ticker=ticker,fiscal_year=y,fiscal_quarter=q,release_date=published,
        fiscal_period_end_actual=end,publication_timestamp=None,precision='date_only',
        publication_timezone='America/New_York' if ticker=='MRK' else 'America/Los_Angeles',
        publication_timezone_basis='Conservative issuer-local calendar policy, not inferred CMS clock.',
        source_url=url,evidence_path=str(path.resolve()),sha256=sha(path),source_kind=source_kind,
        first_public_time_verified=False,conference_call_time_used=False)

def main():
    rows=[]
    supplement=P/'PERIOD_OBSERVATIONS.json'
    for r in json.loads((P/'MRK_MANIFEST.json').read_text()):
        f=Path(r['path']);assert r['meta']['http_status']==200 and sha(f)==r['meta']['sha256']
        t=' '.join(f.with_suffix('.txt').read_text().split());y=r['year'];q=r['quarter']
        pub=re.search(r'Save as PDF '+MDY,t)
        if not pub:raise ValueError('Visible original publication header absent')
        published=str(parse(pub[1]).date())
        if (y,q)==(2011,1):
            fact=json.loads(supplement.read_text())[0]
            assert fact['issuer']=='Merck & Co., Inc.' and fact['actual_period_end']=='2011-03-31'
            end=fact['actual_period_end'];period_basis='First Quarter 2011 issuer release, fiscal identity separately verified in original Form10-Q cover.'
        else:
            period=re.search(r'(?:NINE MONTHS ENDED|SIX MONTHS ENDED|Year Ended)\s+'+MDY,t,re.I)
            if not period:raise ValueError('Current-period financial table absent')
            end=str(parse(period[1]).date());period_basis=period[0]
            if not(end.startswith(str(y)) and int(end[5:7])==q*3):raise ValueError('Wrong period year/quarter')
        event=common('MRK',y,q,published,end,r['url'],f,'issuer_original_release_html')
        event.update(period_basis=period_basis,original_response_available=True,training_predecessor_only=(y,q)==(2010,3))
        if (y,q)==(2011,1):
            event['additional_evidence']=[dict(evidence_path=str(supplement.resolve()),sha256=sha(supplement),role='fiscal identity only; no later financial values or publication time used')]
        rows.append(event)
    facts=P/'ORCL_INDEX_OBSERVATIONS.json'
    for r in json.loads(facts.read_text()):
        assert r['original_response_available'] is False and r['publication_basis']=='explicit_original_issuer_release_date_in_body'
        e=common('ORCL',r['fiscal_year'],r['fiscal_quarter'],r['release_date'],r['fiscal_period_end_actual'],r['source_url'],facts,'public_index_observation_of_original_issuer_filing')
        e.update(original_response_available=False,publication_evidence=r['publication_fact'],period_evidence=r['period_fact'],
            limitation='Saved fact observation from publicly indexed issuer filing, not original response bytes. No SEC filing/acceptance date used.')
        rows.append(e)
    # Existing original PDF was missed because extraction splits REDWOOD across
    # lines. Restrict whitespace repair to this exact literal, not arbitrary text.
    folder=ROOT/'outputs/monthly_earnings_repair/batch_health_energy_foreign/raw/ORCL'
    matches=[]
    for m in folder.glob('*.meta.json'):
        meta=json.loads(m.read_text())
        if '1q15-press_release-September.pdf' in meta['url']:matches.append((m,meta))
    assert len(matches)==1
    m,meta=matches[0];f=m.with_name(m.name.replace('.meta.json','.bin'))
    assert meta['status']==200 and sha(f)==meta['sha256']
    t=f.with_suffix('.txt').read_text();t=re.sub(r'RE\s+DWOOD','REDWOOD',t);t=' '.join(t.split())
    pub=re.search(r'REDWOOD SHORES, Calif\.,\s*'+MDY,t)
    identity=re.search(r'fiscal (2015) Q(1)',t)
    period=re.search(r'three months ended (August 31, 2014)',t,re.I)
    if not(pub and identity and period):raise ValueError('Oracle original dateline/current fiscal period absent')
    e=common('ORCL',2015,1,str(parse(pub[1]).date()),str(parse(period[1]).date()),meta['url'],f,'issuer_original_release_pdf')
    e.update(original_response_available=True,period_basis=period[0]);rows.append(e)
    base=json.loads((ROOT/'outputs/monthly_earnings_repair/integrated_batches/health_energy_foreign_900/events.json').read_text())
    old={(r['ticker'],r['fiscal_year'],r['fiscal_quarter']) for r in base}
    ids=[(r['ticker'],r['fiscal_year'],r['fiscal_quarter']) for r in rows]
    if old.intersection(ids) or len(ids)!=len(set(ids)):raise ValueError('Duplicate verified quarter')
    rows.sort(key=lambda r:(r['ticker'],r['release_date']));dump('events.json',rows)
    dump('STATUS.json',dict(events=len(rows),by_symbol={s:sum(r['ticker']==s for r in rows) for s in ('MRK','ORCL')},stable_for_merge=True,event_file_sha256=sha(P/'events.json')))
    print(json.dumps(json.loads((P/'STATUS.json').read_text())))

if __name__=='__main__':main()
