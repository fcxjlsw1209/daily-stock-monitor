"""Index SEC discovery evidence; current metadata never certifies past eligibility."""
import datetime,hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent
D=P/'listing_registry_sec'
def main():
    archives=[];events=[];issuers=[];pending=[]
    for f in sorted(D.glob('*_submissions.json')):
        prov=f.with_suffix('.provenance.json')
        if not prov.exists():pending.append(f.name);continue  # collector may be writing
        raw=f.read_bytes();meta=json.loads(prov.read_text())
        if hashlib.sha256(raw).hexdigest()!=meta['sha256']:raise ValueError('Checksum mismatch '+f.name)
        j=json.loads(raw);cik=str(j['cik']).zfill(10);source=str(f.relative_to(P))
        if f.name!=cik+'_submissions.json':raise ValueError('Identity mismatch')
        issuers.append(dict(cik=cik,sec_name=j.get('name'),former_names=j.get('formerNames',[]),current_tickers=j.get('tickers',[]),current_exchanges=j.get('exchanges',[]),source_file=source,identity_verified=False))
        for a in j.get('filings',{}).get('files',[]):
            if a.get('filingFrom','9999')<='2026-10-05' and a.get('filingTo','')>='2008-01-01':
                archives.append(dict(cik=cik,**a,index_source=source))
        r=j.get('filings',{}).get('recent',{})
        for i,form in enumerate(r.get('form',[])):
            if form.startswith(('8-A','25','15F','15-')):
                events.append(dict(cik=cik,form=form,filed=r['filingDate'][i],accession=r['accessionNumber'][i],document=r['primaryDocument'][i],source_file=source,eligibility_verified=False))
    report=dict(checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),cached_issuers=len(issuers),archive_chunks=len(archives),listing_event_index_hints=len(events),pending_cache_writes=pending,historical_identity_certified=0,complete_universe=False,note='Contemporary metadata is discovery only; historical documents and security type must be checked. No absence-based exclusion.')
    for name,data in [('archive_queue',archives),('listing_event_hints',events),('issuer_hints',issuers),('AUDIT',report)]:
        temp=D/(name+'.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(D/(name+'.json'))
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
