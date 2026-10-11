"""Sensitivity to five major NYSE-listed foreign issuers omitted by index union.
Not a full historically certified US market Top32 reconstruction.
"""
from pathlib import Path
import json,re,urllib.request,urllib.error,sys
import numpy as np,pandas as pd
from research import P,O,load,raw_features,Simulator,fixed_rules,dump,metrics,fit_errors,sha
import collect
SYMBOLS=['TSM','BABA','NVO','SAP','TM'];F=P/'foreign';(F/'sources').mkdir(exist_ok=True,parents=True)

def main():
 collect.P=F;collect.SYMBOLS=SYMBOLS;collect.main();prices=json.loads((F/'ETF_MANIFEST.json').read_text());directory=json.loads((O/'four_factor_rebound/vendor_cap/DIRECTORY.json').read_text());caps={};sources=[]
 for s in SYMBOLS:
  if s not in directory:print('missing cap mapping',s,flush=True);continue
  url=directory[s]['url'];file=F/'sources'/f'{s}_cap.html'
  if not file.exists():
   try:
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=30) as r:file.write_bytes(r.read())
   except urllib.error.HTTPError as e:
    print('cap failure',s,e.code,flush=True)
    if e.code==429:dump(F/'RATE_LIMIT.json',dict(symbol=s,status=429,retry_after=e.headers.get('Retry-After')));break
    continue
  found=re.search(r'\bdata\s*=\s*(\[\{"d":.*?\]);',file.read_text(),re.S)
  if not found:print('cap chart missing',s,flush=True);continue
  z=pd.DataFrame(json.loads(found.group(1)));z['date']=pd.to_datetime(z.d,unit='s',utc=True).dt.tz_localize(None).dt.normalize();z['market_cap_usd']=z.m*100000;z=z[['date','market_cap_usd']].sort_values('date').drop_duplicates('date',keep='last');assert z.market_cap_usd.gt(0).all();caps[s]=z
  z.to_csv(F/f'{s}_cap.csv',index=False);sources.append(dict(symbol=s,url=url,file=str(file),sha256=sha(file)))
 dump(F/'CAP_SOURCES.json',sources)
 px,fs,pools=load();origcap=pd.read_csv(O/'daily_rebuild_research/price_scaled_caps.csv.gz',index_col=0,parse_dates=True);oldeligible=pd.read_csv(O/'index_union/alias_resolved_inputs/eligible.csv.gz',index_col=0,parse_dates=True)
 for row in prices:
  s=row['symbol'];d=pd.read_csv(row['file'],index_col=0,parse_dates=True);px[s]=d['Adj Close'].reindex(px.index)
  for k,col in [('close','Close'),('opening','Open'),('high','High'),('low','Low'),('volume','Volume')]:fs[k][s]=d[col].reindex(px.index)
  fs['sector60'][s]=px.SPY.pct_change(60,fill_method=None)
  if s not in caps:continue
  z=caps[s].copy();z['available']=z.date+pd.Timedelta(days=1);join=pd.merge_asof(pd.DataFrame({'session':px.index}),z,left_on='session',right_on='available',direction='backward');close=fs['close'][s];known=pd.DataFrame({'price_date':px.index,'price':close}).dropna().reset_index(drop=True);obs=join.dropna().sort_values('date');den=pd.merge_asof(obs,known,left_on='date',right_on='price_date',direction='backward');den=den.set_index('session').reindex(px.index);rawcap=den.market_cap_usd*close/den.price
  good=(pd.Series(px.index,index=px.index)-den.date).dt.days.le(45)&(den.date-den.price_date).dt.days.le(7);origcap[s]=rawcap.where(good);oldeligible[s]=px[s].notna()
 # Keep existing issuer dedup identities; each added foreign ticker is a distinct company.
 manifest=json.loads((O/'index_union/alias_resolved_inputs/MANIFEST.json').read_text());identities=manifest['issuers'];stockcols=[s for s in oldeligible if s in origcap];member=pd.DataFrame(False,index=px.index,columns=px.columns);count=[]
 for d in px.index:
  ranked=origcap.loc[d,stockcols].where(oldeligible.loc[d,stockcols]).dropna().sort_values(ascending=False,kind='stable');seen=set();chosen=[]
  for s in ranked.index:
   company=identities.get(s,'FOREIGN_'+s)
   if company in seen:continue
   chosen.append(s);seen.add(company)
   if len(chosen)==32:break
  if len(chosen)==32:member.loc[d,chosen]=True
  count.append(dict(date=str(d.date()),foreign_selected=[s for s in chosen if s in SYMBOLS]))
 member.to_csv(F/'membership.csv.gz');dump(F/'MEMBERSHIP_COUNTS.json',count)
 raw=raw_features(px,fs);cal=json.loads((O/'monthly_curve_reverse/CALIBRATION.json').read_text());obs=pd.read_csv(O/'monthly_curve_reverse/DIGITIZED_WEALTH.csv');results={};curves={};orders={}
 from research import POOLS
 for pool in ['core17','theme17']:
  mask=member.copy();mask[POOLS[pool]]=px[POOLS[pool]].notna();rank={k:v.where(mask).rank(axis=1,pct=True) for k,v in raw.items()};rules=fixed_rules(rank,raw);sim=Simulator(px,fs,mask)
  for name in ['m252_42','m294_63','risk','balanced','relative','short_blend']:
   for gate in ['positive','above100']:
    allowed=raw['m252_42']>0
    if gate=='above100':allowed &=px>px.rolling(100).mean()
    nav,order=sim.run(rules[name],allowed);tag=f'{pool}__{name}__{gate}';results[tag]=dict(**metrics(nav),**fit_errors(nav,cal,obs));curves[tag]=nav;orders[tag]=order;print(tag,round(results[tag]['cagr_252']*100,2),round(results[tag]['max_dd']*100,2),flush=True)
 dump(F/'RESULTS.json',results);dump(F/'ORDERS.json',orders);pd.DataFrame(curves).to_csv(F/'CURVES.csv.gz');dump(F/'POLICY.json',dict(additions=SYMBOLS,not_complete_all_US_universe=True,cap_source='Public vendor company-level snapshots; available date+1calendar day, max45days, scale only with past observation price.',limitations=['Does not prove original author pool','Retroactively revised vendor cap data; historical publication timestamp unknown','Original index-union coverage caveats retained','Foreign issuer eligibility approximated by available NYSE ADR price series; all five already listed before comparison period'],trials=len(results)))
if __name__=='__main__':main()
