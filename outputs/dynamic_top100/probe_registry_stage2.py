from pathlib import Path
import requests,json,time,hashlib
P=Path(__file__).resolve().parent/'additional_sources'
urls={
'alpha_current_demo':'https://www.alphavantage.co/query?function=LISTING_STATUS&apikey=demo',
'alpha_delisted_2026':'https://www.alphavantage.co/query?function=LISTING_STATUS&date=2026-10-05&state=delisted&apikey=demo',
'alpha_active_2010':'https://www.alphavantage.co/query?function=LISTING_STATUS&date=2010-01-04&state=active&apikey=demo',
'delist_tree':'https://api.github.com/repos/royelee/delist-detection/git/trees/main?recursive=1',
}
for year in [2010,2015,2020,2025]:urls['sec_frame_'+str(year)]=f'https://data.sec.gov/api/xbrl/frames/dei/EntityCommonStockSharesOutstanding/shares/CY{year}Q1I.json'
records=[]
for name,url in urls.items():
 try:
  r=requests.get(url,headers={'User-Agent':'Personal investment research data-quality audit'},timeout=30);(P/(name+'.txt')).write_bytes(r.content)
  rec=dict(name=name,url=url,status=r.status_code,bytes=len(r.content),sha256=hashlib.sha256(r.content).hexdigest(),preview=r.text[:160]);print(name,r.status_code,len(r.content),r.text[:55],flush=True)
 except Exception as e:rec=dict(name=name,url=url,error=str(e))
 records.append(rec);(P/'stage2_registry.json').write_text(json.dumps(records,indent=2));time.sleep(.4)
