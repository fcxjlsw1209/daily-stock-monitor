"""Bounded public-source probes; downloaded data is unverified research evidence."""
from pathlib import Path
import requests,json,hashlib,datetime,time
P=Path(__file__).resolve().parent/'additional_sources';P.mkdir(exist_ok=True)
urls={
'delisted_provenance':'https://raw.githubusercontent.com/YLiu95/delisted-equity-data/main/SOURCES.md',
'delisted_inventory':'https://raw.githubusercontent.com/YLiu95/delisted-equity-data/main/data/delisted_equity_data.json',
'sec_2010_index':'https://www.sec.gov/Archives/edgar/full-index/2010/QTR1/master.idx',
'sec_2010_financials':'https://www.sec.gov/files/dera/data/financial-statement-data-sets/2010q1.zip',
'nasdaq_current':'https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt',
'other_current':'https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt',
'stooq_aapl':'https://stooq.com/q/d/l/?s=aapl.us&i=d',
'stooq_atvi':'https://stooq.com/q/d/l/?s=atvi.us&i=d',
'stooq_twtr':'https://stooq.com/q/d/l/?s=twtr.us&i=d',
'alpha_listing_demo':'https://www.alphavantage.co/query?function=LISTING_STATUS&date=2014-07-10&state=delisted&apikey=demo',
}
records=[]
for name,url in urls.items():
 rec=dict(name=name,url=url,checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
 try:
  r=requests.get(url,headers={'User-Agent':'Personal investment research data-quality audit'},timeout=35)
  path=P/(name+('.zip' if url.endswith('.zip') else '.txt'));path.write_bytes(r.content)
  rec.update(status=r.status_code,bytes=len(r.content),sha256=hashlib.sha256(r.content).hexdigest(),file=str(path),preview=r.text[:160] if not url.endswith('.zip') else '')
 except Exception as e:rec['error']=str(e)
 records.append(rec);(P/'probe_registry.json').write_text(json.dumps(records,indent=2));print(name,rec.get('status'),rec.get('bytes'),rec.get('preview','')[:70],flush=True);time.sleep(.3)
