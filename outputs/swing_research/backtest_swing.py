"""Long-only, capacity-constrained daily swing portfolio research."""
from pathlib import Path
from dataclasses import dataclass
import json,sqlite3
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parent
DATA=P.parent/'overnight_research'/'intraday_validation'
DB=P.parent/'stock_database'/'stocks.sqlite'
MAP={'Technology':'XLK','Communication Services':'XLC','Consumer Cyclical':'XLY','Consumer Defensive':'XLP','Financial Services':'XLF','Healthcare':'XLV','Industrials':'XLI','Energy':'XLE','Basic Materials':'XLB','Utilities':'XLU','Real Estate':'XLRE'}

@dataclass
class Inputs:
    close:pd.DataFrame
    opening:pd.DataFrame
    dividends:pd.DataFrame
    volume:pd.DataFrame
    sector:dict
    sector_prices:pd.DataFrame

def load():
    with sqlite3.connect(f'file:{DB}?mode=ro',uri=True) as db:d=pd.read_sql_query('SELECT * FROM daily_prices',db)
    d.trade_date=pd.to_datetime(d.trade_date)
    def w(k):return d.pivot(index='trade_date',columns='symbol',values=k).sort_index()
    c,o,div,v=map(w,['close','open','dividends','volume'])
    for s,path in [('SPY',P.parent/'overnight_research'/'spy_source.csv'),('QQQ',P/'qqq_source.csv')]:
        h=pd.read_csv(path,index_col=0,parse_dates=True)
        for x,key in [(c,'Close'),(o,'Open'),(div,'Dividends'),(v,'Volume')]:x[s]=h[key].reindex(c.index)
    sector={r['symbol']:r.get('sector') for r in json.loads((DATA/'download_manifest.json').read_text())}
    sector.update(SPY='ETF',QQQ='ETF')
    sp={}
    for e in MAP.values():
        h=pd.read_csv(DATA/'raw'/f'{e}_daily.csv',index_col=0)
        h.index=pd.to_datetime(h.index,utc=True).tz_convert('America/New_York').tz_localize(None).normalize()
        sp[e]=h.Close.reindex(c.index)
    return Inputs(c,o,div,v,sector,pd.DataFrame(sp))

def signals(data):
    c=data.close;stocks=[s for s in c if s not in ['SPY','QQQ']]
    r60=c.pct_change(60,fill_method=None);r3=c.pct_change(3,fill_method=None)
    scale=c.pct_change(fill_method=None).rolling(20).std()*np.sqrt(3);z=r3/scale
    eligible=(c.rolling(200).count()>=200)&((c*data.volume).rolling(20).mean()>20e6)
    rank=r60[stocks].where(eligible[stocks]).rank(axis=1,pct=True)
    common=eligible&(c>c.rolling(100).mean())&(r60>0)&(r3<0)&(z>=-2.5)&(z<=-.75)
    common=common.mul(c.SPY>c.SPY.rolling(200).mean(),axis=0)
    a=common[stocks]&(rank>=.75)
    er=pd.DataFrame({s:data.sector_prices[MAP[data.sector[s]]].pct_change(3,fill_method=None) for s in stocks})
    linked=a&((er<0).add(r3.SPY<0,axis=0)>0)&((r3[stocks]-er)>=-scale[stocks])
    return {'stock_base':(a,-z[stocks]),'stock_industry':(linked,-z[stocks]),
      'SPY_pullback':(common[['SPY']],-z[['SPY']]),'QQQ_pullback':(common[['QQQ']],-z[['QQQ']])}

def positive(x,description):
    if not np.isfinite(x) or x<=0:raise ValueError('Missing/invalid price '+description)
    return float(x)

def simulate(data,mask,score,rule,cost_bps=5,start='2019-01-01',end='2026-09-18',etf=False,initial=100000.,buyhold=False):
    c,o,div=data.close,data.opening,data.dividends
    ma5=c.rolling(5).mean();dates=c.index;cash=initial;pos={};daily=[];trades=[];orders=[]
    half=cost_bps/20000;slots=1 if etf else 5
    prev_equity=initial;holding_costs=0.;dividends_received=0.
    for i,day in enumerate(dates):
        if day<pd.Timestamp(start) or day>pd.Timestamp(end):continue
        if i==0:raise ValueError('Warmup day required')
        prev=dates[i-1];sold=set()
        # Entitlement on ex-date is to holders before the session opens.
        for s,p in pos.items():
            amount=float(div.at[day,s])
            if not np.isfinite(amount):raise ValueError('Missing dividends '+s)
            receipt=p['qty']*amount;cash+=receipt;p['dividends']+=receipt;dividends_received+=receipt
        for s,p in list(pos.items()):
            age=i-p['entry_i']
            due=(rule=='fixed3' and age>=3) or (rule in ['fixed5','ma5_max5'] and age>=5)
            reason='max_hold'
            if rule=='ma5_max5' and age>=1 and positive(c.at[prev,s],s+' prior close')>ma5.at[prev,s]:due=True;reason='above_ma5'
            if buyhold:due=False
            if due:
                price=positive(o.at[day,s],s+' exit');proceeds=p['qty']*price;fee=proceeds*half
                cash+=proceeds-fee;holding_costs+=fee
                pnl=proceeds-fee+p['dividends']-p['cost']
                trades.append({'symbol':s,'entry_date':str(p['entry_date'].date()),'exit_date':str(day.date()),'entry_price':p['entry_price'],
                  'exit_price':price,'quantity':p['qty'],'entry_outlay':p['cost'],'dividends':p['dividends'],
                  'entry_fee':p['entry_fee'],'exit_fee':fee,'net_pnl':pnl,'net_return':pnl/p['cost'],'sessions':age,'reason':reason})
                orders.append({'date':str(day.date()),'symbol':s,'side':'sell','price':price,'quantity':p['qty'],'fee':fee})
                sold.add(s);del pos[s]
        opening_equity=cash+sum(p['qty']*positive(o.at[day,s],s+' opening mark') for s,p in pos.items())
        candidates=score.loc[prev].where(mask.loc[prev]).dropna().sort_values(ascending=False,kind='stable').index
        if buyhold and (pos or orders):candidates=[]
        for s in candidates:
            if len(pos)>=slots:break
            if s in pos or s in sold:continue
            sector=data.sector[s]
            if not etf and sum(data.sector[q]==sector for q in pos)>=2:continue
            budget=min(opening_equity/slots,cash)
            if budget<opening_equity*.001:continue
            price=positive(o.at[day,s],s+' entry');notional=budget/(1+half);qty=notional/price;fee=notional*half
            cash-=notional+fee;holding_costs+=fee
            pos[s]={'entry_i':i,'entry_date':day,'entry_price':price,'qty':qty,'cost':notional+fee,'entry_fee':fee,'dividends':0.}
            orders.append({'date':str(day.date()),'symbol':s,'side':'buy','price':price,'quantity':qty,'fee':fee})
        invested=sum(p['qty']*positive(c.at[day,s],s+' closing mark') for s,p in pos.items());eq=cash+invested
        if cash < -1e-7 or len(pos)>slots:raise AssertionError('Cash/position constraint')
        if not etf:
            for sec in set(data.sector[s] for s in pos):
                assert sum(data.sector[s]==sec for s in pos)<=2
        daily.append({'date':str(day.date()),'equity':eq,'cash':cash,'invested':invested,'exposure':invested/eq,'positions':len(pos),
          'return':eq/prev_equity-1,'fees_cumulative':holding_costs,'dividends_cumulative':dividends_received})
        prev_equity=eq
    last=pd.Timestamp(daily[-1]['date']);unrealized=[]
    for s,p in pos.items():
        value=p['qty']*float(c.at[last,s]);pnl=value+p['dividends']-p['cost']
        unrealized.append({'symbol':s,'entry_date':str(p['entry_date'].date()),'quantity':p['qty'],'mark_price':float(c.at[last,s]),
          'market_value':value,'entry_outlay':p['cost'],'dividends':p['dividends'],'unrealized_pnl_including_dividends':pnl})
    trade_cols=['symbol','entry_date','exit_date','entry_price','exit_price','quantity','entry_outlay','dividends','entry_fee','exit_fee','net_pnl','net_return','sessions','reason']
    t=pd.DataFrame(trades,columns=trade_cols)
    # Closed P&L + open P&L must equal account change, including all fees/dividends.
    residual=daily[-1]['equity']-initial-t.net_pnl.sum()-sum(p['unrealized_pnl_including_dividends'] for p in unrealized)
    assert abs(residual)<1e-6*initial, residual
    return pd.DataFrame(daily),t,pd.DataFrame(orders),unrealized

def summarize(d,t,a,b):
    f=d[(d.date>=a)&(d.date<=b)]
    if f.empty:return {}
    net=(1+f['return']).cumprod();peak=np.maximum.accumulate(np.r_[1.,net])[1:]
    years=((pd.Timestamp(f.date.iloc[-1])-pd.Timestamp(f.date.iloc[0])).days+1)/365.25
    tt=t[(t.exit_date>=a)&(t.exit_date<=b)];r=tt.net_return;pnl=tt.net_pnl
    wins=r[r>0];losses=r[r<0]
    return {'days':len(f),'closed_trades':len(tt),'cagr':float(net.iloc[-1]**(1/years)-1),'total_return':float(net.iloc[-1]-1),
      'max_drawdown':float((net/peak-1).min()),'trade_win':float((r>0).mean()) if len(r) else None,
      'mean_trade_return':float(r.mean()) if len(r) else None,'mean_win':float(wins.mean()) if len(wins) else None,
      'mean_loss':float(losses.mean()) if len(losses) else None,
      'profit_factor_dollars':float(pnl[pnl>0].sum()/-pnl[pnl<0].sum()) if (pnl<0).any() else None,
      'worst_trade':float(r.min()) if len(r) else None,'mean_hold':float(tt.sessions.mean()) if len(tt) else None,
      'median_hold':float(tt.sessions.median()) if len(tt) else None,'average_exposure':float(f.exposure.mean()),
      'days_in_market':int((f.positions>0).sum()),'fraction_days_in_market':float((f.positions>0).mean()),'worst_day':float(f['return'].min())}

def main():
    data=load();models=signals(data);metrics=[];dailies=[];traderecords=[];orderrecords=[];openrecords=[];yearly=[]
    for name,(mask,score) in models.items():
        for rule in ['fixed3','fixed5','ma5_max5']:
            for cost in [0,5,10,20]:
                d,t,orders,unrealized=simulate(data,mask,score,rule,cost,etf=name.startswith(('SPY','QQQ')))
                for split,(a,b) in {'earlier':('2019-01-01','2021-12-31'),'recent':('2022-01-01','2026-09-18'),'all':('2019-01-01','2026-09-18')}.items():
                    metrics.append({'strategy':name,'exit_rule':rule,'cost_bps':cost,'split':split,**summarize(d,t,a,b)})
                if cost in [5,10]:
                    for frame in [d,t,orders]:frame['strategy']=name;frame['exit_rule']=rule;frame['cost_bps']=cost
                    dailies.append(d);traderecords.append(t);orderrecords.append(orders)
                    openrecords.extend([dict(x,strategy=name,exit_rule=rule,cost_bps=cost) for x in unrealized])
                    for year in range(2019,2027):yearly.append({'strategy':name,'exit_rule':rule,'cost_bps':cost,'year':year,**summarize(d,t,f'{year}-01-01',f'{year}-12-31')})
                print(name,rule,cost,'closed trades',len(t),'final equity',round(d.equity.iloc[-1],2),flush=True)
    for s in ['SPY','QQQ']:
        mask=pd.DataFrame(True,index=data.close.index,columns=[s]);score=mask.astype(float)
        for cost in [5,10]:
            d,t,orders,u=simulate(data,mask,score,'buyhold',cost,etf=True,buyhold=True)
            for split,(a,b) in {'earlier':('2019-01-01','2021-12-31'),'recent':('2022-01-01','2026-09-18'),'all':('2019-01-01','2026-09-18')}.items():
                metrics.append({'strategy':s+'_buyhold','exit_rule':'buyhold','cost_bps':cost,'split':split,**summarize(d,t,a,b)})
            d['strategy']=s+'_buyhold';d['exit_rule']='buyhold';d['cost_bps']=cost;dailies.append(d)
            openrecords.extend([dict(x,strategy=s+'_buyhold',exit_rule='buyhold',cost_bps=cost) for x in u])
    pd.DataFrame(metrics).to_csv(P/'metrics.csv',index=False)
    pd.concat(dailies).to_csv(P/'daily_equity.csv.gz',index=False,compression='gzip')
    pd.concat(traderecords).to_csv(P/'trades.csv',index=False)
    pd.concat(orderrecords).to_csv(P/'orders.csv',index=False)
    pd.DataFrame(openrecords).to_csv(P/'open_positions.csv',index=False)
    pd.DataFrame(yearly).to_csv(P/'yearly.csv',index=False)
    print(pd.DataFrame(metrics).query("cost_bps==5 and split=='all'")[['strategy','exit_rule','closed_trades','cagr','trade_win','max_drawdown','average_exposure']].to_string(index=False))
if __name__=='__main__':main()
