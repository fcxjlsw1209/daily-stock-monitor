"""Fetch SEC evidence for reference name candidates, without treating them as identities."""
import datetime,hashlib,json,time
from pathlib import Path
import requests
P=Path(__file__).resolve().parent;D=P/'registry_sec';D.mkdir(exist_ok=True)
def main():
    refs=json.loads((P/'registry_reference_hints.json').read_text());cache={}
    for folder in ['expanded_sec','foreign_sec','priority_sec','pilot_identity']:
        for f in (P/folder).glob('*_submissions.json'):
            j=json.loads(f.read_text())
            if j.get('cik'):cache[str(j['cik']).zfill(10)]=f
    ids=[]
    for row in refs:
        for c in row['candidate_ciks']:
            if c not in ids:ids.append(c)
    results=[];s=requests.Session();s.headers['User-Agent']='Personal financial research data quality audit'
    for cik in ids:
        record={'cik':cik,'identity_verified':False,'usable_for_rank':False}
        for kind,url in [('submissions',f'https://data.sec.gov/submissions/CIK{cik}.json'),('facts',f'https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json')]:
            f=D/f'{cik}_{kind}.json'
            if kind=='submissions' and cik in cache:f=cache[cik]
            elif kind=='facts' and cik in cache:
                trial=cache[cik].with_name(cache[cik].name.replace('_submissions.json','_facts.json'))
                if trial.exists():f=trial
            if not f.exists():
                r=s.get(url,timeout=30);time.sleep(.4)
                if r.status_code in (403,429):
                    record[kind+'_status']=r.status_code;results.append(record);(D/'progress.json').write_text(json.dumps(results,indent=2));print('source unavailable',r.status_code,cik,flush=True);return
                if not r.ok:record[kind+'_status']=r.status_code;continue
                j=r.json()
                if str(j.get('cik','')).zfill(10)!=cik:raise ValueError('SEC identity mismatch')
                f.write_bytes(r.content);f.with_suffix('.provenance.json').write_text(json.dumps(dict(source=url,sha256=hashlib.sha256(r.content).hexdigest(),retrieved_at=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2))
            j=json.loads(f.read_text());record[kind+'_file']=str(f.relative_to(P));record[kind+'_status']='cached_or_downloaded'
            if kind=='submissions':
                record['sec_name']=j.get('name');recent=j['filings']['recent'];record['periodic_filings_before_2010']=sum(d<'2010-01-01' and form in ['10-K','10-Q','20-F','40-F'] for d,form in zip(recent['filingDate'],recent['form']))
            else:
                shares=[dict(v,taxonomy=tax,concept=tag) for tax,concepts in j.get('facts',{}).items() for tag,item in concepts.items() if tag in ['EntityCommonStockSharesOutstanding','CommonStockSharesOutstanding','NumberOfSharesOutstanding'] for v in item.get('units',{}).get('shares',[])]
                record['share_observations']=len(shares);record['earliest_share_filed']=min((x['filed'] for x in shares),default=None)
        results.append(record);(D/'progress.json').write_text(json.dumps(results,indent=2)+'\n')
        print(len(results),'/',len(ids),cik,record.get('sec_name'),record.get('share_observations'),flush=True)
if __name__=='__main__':main()
