"""Collect archived SEC filing indexes once, retaining historical source identities."""
import datetime,hashlib,json,time
from pathlib import Path
import requests
P=Path(__file__).resolve().parent;D=P/'registry_sec/filing_indexes';D.mkdir(exist_ok=True)
def main():
    progress=json.loads((P/'registry_sec/progress.json').read_text());files={r['cik']:P/r['submissions_file'] for r in progress if r.get('submissions_file')}
    for r in json.loads((P/'legacy_sec/progress.json').read_text()):files[r['cik']]=P/'legacy_sec'/f"{r['symbol']}_submissions.json"
    output=[];session=requests.Session();session.headers['User-Agent']='Personal financial research data quality audit'
    for n,(cik,f) in enumerate(files.items(),1):
        j=json.loads(f.read_text());chunks=[(j['filings']['recent'],str(f.relative_to(P)))]
        for archive in j['filings'].get('files',[]):
            if archive.get('filingFrom','9999')>'2026-10-06' or archive.get('filingTo','')<'2008-01-01':continue
            name=archive['name']
            if '/' in name or not name.startswith('CIK'):raise ValueError('Unexpected SEC archive filename')
            dest=D/name;url='https://data.sec.gov/submissions/'+name
            if not dest.exists():
                r=session.get(url,timeout=40);time.sleep(.4)
                if r.status_code in [403,429]:print('source stopped',r.status_code,flush=True);return
                r.raise_for_status();r.json();dest.write_bytes(r.content);dest.with_suffix('.provenance.json').write_text(json.dumps(dict(source=url,sha256=hashlib.sha256(r.content).hexdigest(),retrieved_at=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2))
            chunks.append((json.loads(dest.read_text()),str(dest.relative_to(P))))
        for chunk,source in chunks:
            for i,form in enumerate(chunk['form']):
                date=chunk['filingDate'][i]
                if '2008-01-01'<=date<='2026-10-06' and form in ['10-K','10-Q','20-F','40-F','25','25-NSE','15-12B','15-12G','15F-12B']:
                    acc=chunk['accessionNumber'][i];doc=chunk['primaryDocument'][i]
                    output.append(dict(cik=cik,form=form,filed=date,report_date=chunk['reportDate'][i],accession=acc,url=f'https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc.replace("-","")}/{doc}',index_source=source))
        (D/'filing_queue.json').write_text(json.dumps(output,indent=2)+'\n');print(n,'/',len(files),'filings',len(output),flush=True)
if __name__=='__main__':main()
