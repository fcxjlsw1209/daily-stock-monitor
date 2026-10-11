"""Independent, frozen expanded-universe experiment. No monitor imports or writes."""
from pathlib import Path
import json, hashlib, sqlite3, sys
import numpy as np
import pandas as pd
import exchange_calendars as xc
P=Path(__file__).resolve().parent
U=P.parent/'universe_expansion'
sys.path.insert(0,str(P.parent/'swing_research'))
from backtest_swing import summarize
MAP={'Information Technology':'XLK','Communication Services':'XLC','Consumer Discretionary':'XLY','Consumer Staples':'XLP','Financials':'XLF','Health Care':'XLV','Industrials':'XLI','Energy':'XLE','Materials':'XLB','Utilities':'XLU','Real Estate':'XLRE'}
RECIPES={
 'accel30':dict(mom=.3,accel=.2,jump=.2,downside=.15,cmf=.15),
 'accel40':dict(mom=.4,accel=.1,jump=.2,downside=.15,cmf=.15),
 'sector':dict(mom=.3,sector=.2,jump=.2,downside=.15,cmf=.15),
 'breadth':dict(mom=.3,breadth=.2,jump=.2,downside=.15,cmf=.15),
 'activity':dict(mom=.4,accel=.1,jump=.1,downside=.15,cmf=.15,activity=.1),
 'combined':dict(mom=.3,sector=.1,breadth=.1,jump=.1,downside=.15,cmf=.15,activity=.1)}
PERIODS={'all':('2019-01-01','2026-10-05'),'early':('2019-01-01','2021-12-31'),'middle':('2022-01-01','2023-12-31'),'late':('2024-01-01','2026-10-05')}

def load():
 meta=pd.read_csv(U/'sources/current_sp500.csv');meta['Symbol']=meta.Symbol.str.replace('.','-',regex=False);meta=meta.set_index('Symbol')
 dates=xc.get_calendar('XNYS',start='2010-01-01',end='2026-10-05').sessions.tz_localize(None).difference(pd.DatetimeIndex(['2025-01-09']))
 cols=sorted(meta.index);frames={s:pd.read_csv(U/'raw'/f'{s}.csv.gz',index_col=0,parse_dates=True).reindex(dates) for s in cols}
 for path in sorted((P/'etf_raw').glob('*.csv')):
  d=pd.read_csv(path,index_col=0,parse_dates=True);d.index=pd.to_datetime(d.index,utc=True).tz_localize(None).normalize();frames[path.stem]=d.reindex(dates)
 raw={k:pd.DataFrame({s:d[k] for s,d in frames.items()}) for k in ['Open','High','Low','Close','Adj Close','Volume','Dividends']}
 with sqlite3.connect(f'file:{P.parent}/stock_database/stocks.sqlite?mode=ro',uri=True) as db:
  old=set(pd.read_sql_query('SELECT DISTINCT symbol FROM daily_prices',db).symbol)
 return raw,meta,sorted(old&set(cols))

def features(raw,meta,pool):
 cols=list(meta.index);a=raw['Adj Close'][cols];c=raw['Close'][cols];v=raw['Volume'][cols];o=raw['Open'][cols];h=raw['High'][cols];l=raw['Low'][cols]
 good=np.isfinite(a)&(a>0)&np.isfinite(o)&(o>0)&(l>0)&(h>=l)&(o>=l)&(o<=h)&(c>=l)&(c<=h)&(v>0)
 liquid=(c*v).rolling(20).mean()
 valid=good.rolling(253).sum().eq(253)&(liquid>20e6)
 # Deduplicate current issuer identity using only trailing liquidity.
 for _,g in meta.groupby('CIK'):
  if len(g)>1:
   sy=list(g.index);valid[sy] &= liquid[sy].rank(axis=1,method='first',ascending=False).eq(1)
 r=a.pct_change(fill_method=None);ma=a.rolling(100).mean();above=a>ma
 breadth=pd.DataFrame(np.nan,index=a.index,columns=cols);activity=breadth.copy()
 ratio=(c*v).rolling(5).mean()/(c*v).rolling(60).mean()
 for _,g in meta.groupby('GICS Sector'):
  sy=list(g.index)
  for s in sy:
   peers=[q for q in sy if meta.at[q,'CIK']!=meta.at[s,'CIK']]
   eligible=valid[peers];n=eligible.sum(axis=1)
   breadth[s]=(above[peers]&eligible).sum(axis=1).div(n).where(n>=10)
   activity[s]=ratio[peers].where(eligible).median(axis=1).where(n>=10)
 spread=h-l;loc=((2*c-h-l)/spread).where(spread!=0,0)
 vals=dict(mom=a.shift(21)/a.shift(252)-1,accel=a/a.shift(21)-a.shift(21)/a.shift(42),
  sector=pd.DataFrame({s:raw['Adj Close'][MAP[meta.at[s,'GICS Sector']]].pct_change(60,fill_method=None) for s in cols}),
  breadth=breadth,activity=activity,jump=-r.abs().rolling(60).max()/r.abs().rolling(60).sum(),downside=-r.clip(upper=0).rolling(60).std(),cmf=(loc*v).rolling(21).sum()/v.rolling(21).sum())
 bull=raw['Adj Close'].SPY>raw['Adj Close'].SPY.rolling(200).mean()
 bear=raw['Adj Close'].SPY<raw['Adj Close'].SPY.rolling(200).mean()
 result={}
 for name,w in RECIPES.items():
  assert abs(sum(w.values())-1)<1e-9
  assert sum(w.get(k,0) for k in ['mom','accel','sector','breadth'])<=.50000001
  ok=valid[pool].copy()
  for k in w:ok &= np.isfinite(vals[k][pool])
  ranks={k:vals[k][pool].where(ok).rank(axis=1,pct=True) for k in w}
  score=sum(ranks[k]*weight for k,weight in w.items())
  short_score=sum(((-vals[k][pool] if k in ['mom','accel','sector','breadth','cmf'] else vals[k][pool]).where(ok).rank(axis=1,pct=True))*weight for k,weight in w.items())
  longmask=(ok&(vals['mom'][pool]>0)&above[pool]&(score>=.65)).mul(bull,axis=0)
  shortmask=(ok&(vals['mom'][pool]<0)&(a[pool]<ma[pool])&(short_score>=.65)).mul(bear,axis=0)
  result[name]=(longmask,score,shortmask,short_score)
 return result

def simulate(raw,meta,f,cost=10,borrow=0,shorts=False,benchmark=None,start='2019-01-01'):
 dates=raw['Close'].index;half=cost/20000;cash=100000.;equity=100000.;pos={};daily=[];trades=[];orders=[];events=[];fees=0.;borrow_total=0.;income_total=0.;last_day=None
 lm,ls,sm,ss=f
 c,o,div=(raw[k] for k in ['Close','Open','Dividends'])
 def price(frame,day,s):
  x=frame.at[day,s];return float(x) if np.isfinite(x) and x>0 else None
 for i,day in enumerate(dates):
  if day<pd.Timestamp(start):continue
  prev=dates[i-1];prior_equity=equity;exited=set()
  elapsed=(day-last_day).days if last_day is not None else 1
  for s,p in pos.items():
   fee=abs(p['qty'])*p['mark']*borrow*elapsed/365 if p['qty']<0 else 0.
   cash-=fee;p['borrow']+=fee;borrow_total+=fee
   dv=div.at[day,s]
   if not np.isfinite(dv):
    events.append(dict(date=str(day.date()),symbol=s,event='missing_dividend_assumed_zero_unverified'));dv=0.
   income=p['qty']*dv;cash+=income;p['income']+=income;income_total+=income
  for s,p in list(pos.items()):
   if benchmark or i-p['entry_i']<20:continue
   px=price(o,day,s)
   if px is None:
    events.append(dict(date=str(day.date()),symbol=s,event='exit_deferred_missing_open'));continue
   fee=abs(p['qty'])*px*half;cash+=p['qty']*px-fee;fees+=fee
   pnl=p['qty']*(px-p['entry_price'])+p['income']-p['borrow']-p['entry_fee']-fee
   trades.append(dict(symbol=s,side=p['side'],entry_date=p['entry_date'],exit_date=str(day.date()),net_pnl=pnl,net_return=pnl/p['budget'],sessions=i-p['entry_i'],borrow=p['borrow'],dividends=p['income']))
   orders.append(dict(date=str(day.date()),symbol=s,side='exit_'+p['side'],price=px,quantity=abs(p['qty']),fee=fee));del pos[s];exited.add(s)
  if benchmark:
   candidates=[(benchmark,1.,1)] if not pos and not orders else []
  else:
   candidates=[(s,float(v),1) for s,v in ls.loc[prev].where(lm.loc[prev]).dropna().items()]
   if shorts:candidates += [(s,float(v),-1) for s,v in ss.loc[prev].where(sm.loc[prev]).dropna().items()]
   candidates.sort(key=lambda x:(-x[1],x[0]))
  open_marks={s:price(o,day,s) or p['mark'] for s,p in pos.items()}
  open_equity=cash+sum(p['qty']*open_marks[s] for s,p in pos.items())
  gross=sum(abs(p['qty'])*open_marks[s] for s,p in pos.items())
  for s,score,sign in candidates:
   if len(pos)>=(1 if benchmark else 5):break
   if s in pos or s in exited:continue
   if not benchmark:
    if any(meta.at[s,'CIK']==meta.at[t,'CIK'] for t in pos):continue
    if sum(meta.at[s,'GICS Sector']==meta.at[t,'GICS Sector'] for t in pos)>=2:continue
   px=price(o,day,s)
   if px is None:
    # Data failure cancels this order; no substitution using another name's future bar.
    events.append(dict(date=str(day.date()),symbol=s,event='entry_cancelled_missing_bar'));break
   budget=min(prior_equity*(1 if benchmark else .2),max(0,(open_equity-gross)))
   if not shorts:budget=min(budget,cash)
   if budget<1:break
   qty=sign*budget/(1+half)/px;fee=abs(qty)*px*half
   cash-=qty*px+fee;fees+=fee;gross+=abs(qty)*px;open_equity-=fee
   pos[s]=dict(qty=qty,entry_i=i,entry_date=str(day.date()),entry_price=px,entry_fee=fee,budget=budget,income=0.,borrow=0.,mark=px,side='long' if sign>0 else 'short')
   orders.append(dict(date=str(day.date()),symbol=s,side='entry_'+pos[s]['side'],price=px,quantity=abs(qty),fee=fee))
  for s,p in pos.items():
   px=price(c,day,s)
   if px is None:events.append(dict(date=str(day.date()),symbol=s,event='stale_close_mark'))
   else:p['mark']=px
  invested=sum(p['qty']*p['mark'] for p in pos.values());gross=sum(abs(p['qty'])*p['mark'] for p in pos.values());equity=cash+invested
  assert equity>0 and (shorts or cash>=-1e-6)
  daily.append(dict(date=str(day.date()),equity=equity,cash=cash,invested=invested,exposure=gross/equity,net_exposure=invested/equity,positions=len(pos),return_=equity/prior_equity-1,borrow=borrow_total,fees=fees,dividends=income_total))
  last_day=day
 openpnl=sum(p['qty']*(p['mark']-p['entry_price'])+p['income']-p['borrow']-p['entry_fee'] for p in pos.values())
 assert abs(equity-100000-sum(t['net_pnl'] for t in trades)-openpnl)<1e-5
 d=pd.DataFrame(daily).rename(columns={'return_':'return'})
 t=pd.DataFrame(trades,columns=['symbol','side','entry_date','exit_date','net_pnl','net_return','sessions','borrow','dividends'])
 d.attrs['open_positions']={s:{**p,'unrealized_pnl':p['qty']*(p['mark']-p['entry_price'])+p['income']-p['borrow']-p['entry_fee']} for s,p in pos.items()}
 d.attrs['side_pnl']={side:sum(x['net_pnl'] for x in trades if x['side']==side)+sum(x['unrealized_pnl'] for x in d.attrs['open_positions'].values() if x['side']==side) for side in ['long','short']}
 assert abs(sum(d.attrs['side_pnl'].values())-(equity-100000))<1e-5
 return d,t,pd.DataFrame(orders),events

def main():
 raw,meta,old=load();pools={'original_intersection':old,'expanded':sorted(meta.index)}
 (P/'recipes.json').write_text(json.dumps(RECIPES,indent=2))
 rows=[];annual=[];exceptions=[];all_features={};prefix_checks=[]
 for univ,pool in pools.items():
  fs=features(raw,meta,pool);all_features[univ]=fs
  cut='2023-12-29';small=features({k:v.loc[:cut] for k,v in raw.items()},meta,pool)
  for name in fs:
   for j in range(4):pd.testing.assert_frame_equal(fs[name][j].loc[:cut],small[name][j])
  prefix_checks.append(univ)
  print('prefix passed',univ,len(pool),flush=True)
 configs=[(u,n,c,0.,False) for u in pools for n in RECIPES for c in [10,20]]
 configs += [('expanded','combined',c,b,True) for c in [10,20] for b in [.02,.05,.1]]
 def save(tag,d,t,o,ev,info):
  (P/(tag+'_positions.json')).write_text(json.dumps(d.attrs,indent=2));d.to_csv(P/(tag+'_daily.csv'),index=False);t.to_csv(P/(tag+'_trades.csv'),index=False);o.to_csv(P/(tag+'_orders.csv'),index=False)
  for split,(a,b) in PERIODS.items():rows.append(dict(**info,split=split,**summarize(d,t,a,b),borrow_paid=float(d.borrow.iloc[-1]),data_events=len(ev)))
  for y in range(2019,2027):annual.append(dict(**info,year=y,**summarize(d,t,f'{y}-01-01',f'{y}-12-31')))
  exceptions.extend(dict(run=tag,**v) for v in ev)
  pd.DataFrame(rows).to_csv(P/'metrics.csv',index=False);pd.DataFrame(annual).to_csv(P/'yearly.csv',index=False)
  print(tag,'CAGR',round(rows[-4]['cagr'],4),'DD',round(rows[-4]['max_drawdown'],4),'events',len(ev),flush=True)
 for u,n,c,b,sh in configs:
  tag=f'{u}_{n}_{c}_borrow{int(b*100)}';d,t,o,ev=simulate(raw,meta,all_features[u][n],c,b,sh)
  save(tag,d,t,o,ev,dict(run=tag,universe=u,recipe=n,cost_bps=c,borrow_rate=b,shorts=sh))
 for symbol in ['SPY','QQQ']:
  for cost in [10,20]:
   tag=f'{symbol}_buyhold_{cost}';d,t,o,ev=simulate(raw,meta,all_features['expanded']['combined'],cost,benchmark=symbol)
   save(tag,d,t,o,ev,dict(run=tag,universe='benchmark',recipe=symbol,cost_bps=cost,borrow_rate=0.,shorts=False))
 pd.DataFrame(exceptions,columns=['run','date','symbol','event']).to_csv(P/'data_events.csv',index=False)
 source_paths=[P/'run.py',P/'test_run.py',Path(summarize.__code__.co_filename),P/'PROTOCOL.md',U/'raw_hashes.json',U/'sources/current_sp500.csv']+list((P/'etf_raw').glob('*.csv'))
 (P/'validation.json').write_text(json.dumps(dict(runs=len(configs)+4,prefix_invariance=prefix_checks,accounting='passed',original_intersection=len(old),expanded=len(meta),point_in_time=False,untouched_holdout=False,source_hashes={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in source_paths}),indent=2))
if __name__=='__main__':main()
