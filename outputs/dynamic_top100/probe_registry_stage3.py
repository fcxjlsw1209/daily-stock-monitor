from pathlib import Path
import requests,json,time,hashlib
P=Path(__file__).resolve().parent/'additional_sources'
urls={
'kaggle_nyse':'https://www.kaggle.com/api/v1/datasets/download/dgawlik/nyse',
'wayback_yhoo':'https://web.archive.org/cdx/search/cdx?url=real-chart.finance.yahoo.com/table.csv%3Fs%3DYHOO&output=json&filter=statuscode:200&collapse=digest&limit=10',
'wayback_atvi':'https://web.archive.org/cdx/search/cdx?url=finance.yahoo.com/quote/ATVI/history&output=json&filter=statuscode:200&collapse=digest&limit=10',
}
for name in ['securities','ticker_history','delistings','payouts','web_verification']:
 urls['delist_'+name]=f'https://raw.githubusercontent.com/royelee/delist-detection/main/output/{name}.csv'
records=[]
for name,url in urls.items():
 try:
  r=requests.get(url,headers={'User-Agent':'Personal investment research data-quality audit'},timeout=45);path=P/(name+('.zip' if name=='kaggle_nyse' else '.txt'));path.write_bytes(r.content)
  rec=dict(name=name,url=url,status=r.status_code,bytes=len(r.content),sha256=hashlib.sha256(r.content).hexdigest());print(name,r.status_code,len(r.content),flush=True)
 except Exception as e:rec=dict(name=name,url=url,error=str(e))
 records.append(rec);(P/'stage3_registry.json').write_text(json.dumps(records,indent=2));time.sleep(.5)
