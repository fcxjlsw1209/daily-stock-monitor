"""Cache early-listing SEC identity evidence, never certify vendor name matches."""
import argparse,datetime,hashlib,json,time
from pathlib import Path
import requests
P=Path(__file__).resolve().parent
D=P/'listing_registry_sec'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--limit',type=int,default=100);args=ap.parse_args()
    if args.limit<=0:raise ValueError('positive batch size required')
    D.mkdir(exist_ok=True)
    progress_path=D/'progress.json'
    records=json.loads(progress_path.read_text()) if progress_path.exists() else {}
    hints=json.loads((P/'registry_early_listing_queue.json').read_text())
    ciks=sorted({r['cik'] for r in hints})
    s=requests.Session();s.headers['User-Agent']='Personal investment research historical data quality audit'
    processed=0
    for cik in ciks:
        dest=D/(cik+'_submissions.json');prov=dest.with_suffix('.provenance.json')
        if dest.exists():
            meta=json.loads(prov.read_text())
            if hashlib.sha256(dest.read_bytes()).hexdigest()!=meta['sha256']:raise ValueError('cache checksum mismatch '+cik)
            j=json.loads(dest.read_text())
            if str(j.get('cik')).zfill(10)!=cik:raise ValueError('cache identity mismatch')
            continue
        if records.get(cik,{}).get('status')==404:continue
        url='https://data.sec.gov/submissions/CIK'+cik+'.json'
        now=datetime.datetime.now(datetime.timezone.utc).isoformat()
        try:r=s.get(url,timeout=30)
        except requests.RequestException as exc:
            records[cik]=dict(status='network_error',error=type(exc).__name__,checked_at=now)
            progress_path.write_text(json.dumps(records,indent=2)+'\n');print('stopped',cik,type(exc).__name__,flush=True);return
        time.sleep(.6)
        records[cik]=dict(status=r.status_code,checked_at=now,identity_verified=False,usable_for_rank=False)
        if r.status_code in [403,429]:
            records[cik]['retry_after']=r.headers.get('Retry-After');progress_path.write_text(json.dumps(records,indent=2)+'\n');print('stopped',cik,r.status_code,flush=True);return
        if r.status_code==200:
            j=r.json()
            if str(j.get('cik')).zfill(10)!=cik:raise ValueError('SEC response identity mismatch')
            dest.write_bytes(r.content);prov.write_text(json.dumps(dict(source=url,sha256=hashlib.sha256(r.content).hexdigest(),retrieved_at=now),indent=2)+'\n')
            records[cik].update(sec_name=j.get('name'),file=str(dest.relative_to(P)),archive_files=len(j.get('filings',{}).get('files',[])))
        elif r.status_code!=404:
            progress_path.write_text(json.dumps(records,indent=2)+'\n');print('stopped unexpected status',r.status_code,flush=True);return
        processed+=1;progress_path.write_text(json.dumps(records,indent=2)+'\n')
        if processed%25==0:print('batch',processed,'cached',sum(x.get('status')==200 for x in records.values()),'of',len(ciks),flush=True)
        if processed>=args.limit:break
    print('batch finished',processed,'total responses',len(records),'of',len(ciks),flush=True)
if __name__=='__main__':main()
