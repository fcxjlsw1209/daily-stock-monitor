from pathlib import Path
import requests,json,time,pandas as pd
P=Path(__file__).resolve().parent;D=P/'foreign_sec';D.mkdir(exist_ok=True)
# CIKs discovered in SEC frames, with AZN/NVO/SHOP additionally checked against SEC filing indexes.
m={'TSM':1046179,'ASML':937966,'HSBC':1089113,'ARM':1973239,'BABA':1577552,'RY':1000275,'SHEL':1306965,'NVS':1114448,'MUFG':67088,'SAP':1000184,'TM':1094517,'BHP':811809,'SAN':891478,'TTE':879764,'TD':947263,'RIO':863064,'SCCO':1001838,'SMFG':1022837,'AZN':901832,'NVO':353278,'SHOP':1594805}
out=[]
for symbol,c in m.items():
 cik=str(c).zfill(10);rec=dict(symbol=symbol,cik=cik,historical_listing_verified=False,adr_units_verified=False)
 try:
  for kind,url in [('submissions',f'https://data.sec.gov/submissions/CIK{cik}.json'),('facts',f'https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json')]:
   path=D/(symbol+'_'+kind+'.json')
   if not path.exists():
    r=requests.get(url,headers={'User-Agent':'Personal investment research data-quality audit'},timeout=25);r.raise_for_status();path.write_bytes(r.content);time.sleep(.5)
  sub=json.loads((D/(symbol+'_submissions.json')).read_text());data=json.loads((D/(symbol+'_facts.json')).read_text())
  rec.update(name=sub.get('name'),tickers=sub.get('tickers'),current_ticker_match=symbol in sub.get('tickers',[]))
  facts=[];candidates=[]
  for tax,concepts in data['facts'].items():
   for tag,item in concepts.items():
    if 'shares' in item.get('units',{}):
     candidates.append(dict(taxonomy=tax,tag=tag,rows=len(item['units']['shares'])))
    if tag in ['EntityCommonStockSharesOutstanding','CommonStockSharesOutstanding','NumberOfSharesOutstanding']:
     facts.extend(dict(v,taxonomy=tax,concept=tag,cik=cik) for v in item.get('units',{}).get('shares',[]))
  (D/(symbol+'_share_concepts.json')).write_text(json.dumps(candidates,indent=2))
  if facts:pd.DataFrame(facts).to_csv(D/(symbol+'_shares.csv'),index=False)
  rec.update(status='downloaded',facts=len(facts),first_filed=min([x['filed'] for x in facts],default=None))
 except Exception as e:rec.update(status='error',error=str(e))
 out.append(rec);(D/'coverage.json').write_text(json.dumps(out,indent=2));print(symbol,rec.get('status'),rec.get('facts'),rec.get('current_ticker_match'),flush=True)
