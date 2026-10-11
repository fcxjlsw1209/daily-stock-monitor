from pathlib import Path
import json,sys
import pandas as pd
O=Path(__file__).resolve().parent.parent;P=Path(__file__).resolve().parent
sys.path.insert(0,str(O/'monthly_industry_refinement'));import refine as ref
sys.path.insert(0,str(O/'monthly_top35_extension'));import run as cap
px,raw,members,periods,sources,_=ref.e.load_prices()
print('Panel',str(px.index[0].date()),str(px.index[-1].date()),len(px.columns),flush=True)
records=cap.read(O/'monthly_top35_extension/CAP_COLLECTION.json')['records'];ev=cap.read(O/'us_listed_top32_yahoo/LISTING_EVIDENCE.json')
unique={}
for r in records:
 if r['url'] in unique and r['symbol'] not in {'GOOGL','BRK-B'}:continue
 unique[r['url']]=r
caps={r['symbol']:pd.read_csv(r['data_file'],index_col=0,parse_dates=True).market_cap_usd for r in unique.values()}
dates=px.loc['2016-08-17':'2020-11-30'].index.to_series().groupby(px.loc['2016-08-17':'2020-11-30'].index.to_period('M')).last().tolist();dates=[pd.Timestamp('2016-08-17')]+dates
ans=[];missing={};stale=[];conflicts=[]
for date in dates:
 asof=px.index[px.index.get_loc(date)-1];rank=[]
 for s,z in caps.items():
  active,conflict=cap.listing_status(s,str(asof.date()),ev)
  if not active or (s=='LTM' and '2020-06-22'<=str(asof.date())<'2024-07-25'):continue
  symbol=cap.actual(s);v=cap.snapshot_cap(z,asof,raw['close'][symbol] if symbol in raw['close'] else None)
  if v is None:continue
  if s=='SHEL':v['market_cap_usd']*=2.1;v['bounded_not_exact']=True
  rank.append(dict(symbol=s,**v,listing_conflict=bool(conflict)))
 rank.sort(key=lambda z:(-z['market_cap_usd'],z['symbol']))
 for k,z in enumerate(rank[:40]):
  s=cap.actual(z['symbol']);z['rank']=k+1
  if k<32:
   if s not in px or px.loc[:date,s].tail(295).notna().sum()<295:missing.setdefault(z['symbol'],[]).append(str(date.date()))
   if z['age_days']>40:stale.append(dict(date=str(date.date()),**z))
   if z['listing_conflict']:conflicts.append(dict(date=str(date.date()),**z))
 ans.append(dict(date=str(date.date()),ranks=rank[:40]))
result=dict(dates=len(dates),union=sorted({z['symbol'] for r in ans for z in r['ranks'][:32]}),missing=missing,stale=stale,conflicts=conflicts,rankings=ans)
(P/'PROBE.json').write_text(json.dumps(result,indent=2)+'\n')
print('Early dates',len(dates),'union',result['union'],'missing',{s:len(v) for s,v in missing.items()},'stale',len(stale),'conflicts',len(conflicts),flush=True)
for r in ans:
 if r['date'] in ['2016-08-17','2018-12-31','2020-11-30']:print(r['date'],[(z['symbol'],round(z['market_cap_usd']/1e9,1)) for z in r['ranks'][:32]],flush=True)
