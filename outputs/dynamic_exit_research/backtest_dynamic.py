"""Fixed-protocol daily entry/exit study. No live orders."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parent
STRATEGIES={'legacy_fixed5':('legacy','fixed5'),'rebound_fixed5':('rebound','fixed5'),'rebound_partial':('rebound','partial'),
 'meanrev_fixed5':('meanrev','fixed5'),'meanrev_full':('meanrev','meanrev'),'breakout_fixed5':('breakout','fixed5'),'breakout_trailing':('breakout','trailing')}

def load():
    paths={'QQQ':P.parent/'swing_research'/'qqq_source.csv','SPY':P.parent/'overnight_research'/'spy_source.csv'}
    return {s:pd.read_csv(path,index_col=0,parse_dates=True).sort_index() for s,path in paths.items()}

def features(h,spy):
    c=h.Close;f=pd.DataFrame(index=h.index)
    f['ma10']=c.rolling(10).mean();f['ma20']=c.rolling(20).mean()
    f['low5']=h.Low.rolling(5).min()
    tr=pd.concat([h.High-h.Low,(h.High-c.shift(1)).abs(),(h.Low-c.shift(1)).abs()],axis=1).max(axis=1)
    f['atr']=tr.rolling(14).mean()
    common=(c>c.rolling(100).mean())&(spy.Close.reindex(h.index)>spy.Close.rolling(200).mean().reindex(h.index))&(c.rolling(200).count()>=200)&((c*h.Volume).rolling(20).mean()>20e6)
    f['rebound']=common&(c.shift(1)<f.ma20.shift(1))&(c>h.High.shift(1))
    f['meanrev']=common&((c-f.ma20)/c.rolling(20).std()<=-1.5)
    f['breakout']=common&(c>h.High.shift(1).rolling(20).max())
    r3=c.pct_change(3,fill_method=None);z=r3/(c.pct_change(fill_method=None).rolling(20).std()*np.sqrt(3))
    f['legacy']=common&(c.pct_change(60,fill_method=None)>0)&(r3<0)&(z>=-2.5)&(z<=-.75)
    return f

def simulate(h,f,entry,exit_rule,cost_bps=5,start='2011-01-01',end='2026-09-18',initial=100000.):
    cash=initial;pos=None;pending=None;prev_eq=initial;fee_rate=cost_bps/20000
    daily=[];trades=[];orders=[];total_div=0.;total_fee=0.;counter=0
    for i,day in enumerate(h.index):
        if day<pd.Timestamp(start) or day>pd.Timestamp(end):continue
        if i==0:raise ValueError('Warmup required')
        last_processed_i=i
        r=h.loc[day];prev=h.index[i-1];sold=False
        assert np.isfinite(r.Open) and np.isfinite(r.Close) and r.Open>0 and r.Close>0
        if pos:
            income=pos['qty']*r.Dividends
            assert np.isfinite(income)
            cash+=income;pos['dividends']+=income;total_div+=income
        if pending is not None:
            assert pos is not None
            fraction,reason=pending
            qty=pos['qty'] if fraction==1 else pos['initial_qty']/2
            assert qty<=pos['qty']+1e-9
            gross=qty*r.Open;fee=gross*fee_rate;proceeds=gross-fee
            cash+=proceeds;pos['net_sale_proceeds']+=proceeds;pos['exit_fees']+=fee;total_fee+=fee
            pos['qty']-=qty;sold=True
            orders.append({'date':str(day.date()),'episode':pos['id'],'side':'sell','qty':qty,'price':r.Open,'fee':fee,'reason':reason})
            if fraction==1:
                pnl=pos['net_sale_proceeds']+pos['dividends']-pos['entry_outlay']
                trades.append({'episode':pos['id'],'entry_date':str(pos['date'].date()),'exit_date':str(day.date()),'entry_price':pos['entry_price'],
                  'initial_qty':pos['initial_qty'],'entry_outlay':pos['entry_outlay'],'net_sale_proceeds':pos['net_sale_proceeds'],
                  'dividends':pos['dividends'],'entry_fee':pos['entry_fee'],'exit_fees':pos['exit_fees'],
                  'net_pnl':pnl,'net_return':pnl/pos['entry_outlay'],'sessions':i-pos['index'],
                  'partial_taken':pos['partial_taken'],'exit_reason':reason,'worst_close_episode_return':pos['worst_return']})
                pos=None
            else:pos['partial_taken']=True
            pending=None
        if pos is None and not sold and bool(f.at[prev,entry]):
            counter+=1;outlay=cash;notional=outlay/(1+fee_rate);fee=notional*fee_rate;qty=notional/r.Open
            cash-=outlay;total_fee+=fee
            stop=(r.Open-3*f.at[prev,'atr']) if exit_rule=='trailing' else f.at[prev,'low5']
            pos={'id':counter,'index':i,'date':day,'entry_price':r.Open,'initial_qty':qty,'qty':qty,'entry_outlay':outlay,
              'entry_fee':fee,'exit_fees':0.,'dividends':0.,'net_sale_proceeds':0.,'partial_taken':False,
              'stop':stop,'highwater':r.Open,'worst_return':0.}
            orders.append({'date':str(day.date()),'episode':counter,'side':'buy','qty':qty,'price':r.Open,'fee':fee,'reason':entry})
        if pos:
            age=i-pos['index'];pos['worst_return']=min(pos['worst_return'],(pos['qty']*r.Close+pos['net_sale_proceeds']+pos['dividends'])/pos['entry_outlay']-1)
            if exit_rule=='fixed5':
                if age+1>=5:pending=(1,'fixed5')
            elif exit_rule in ['partial','meanrev']:
                if r.Close<pos['stop']:pending=(1,'close_below_initial_low')
                elif exit_rule=='meanrev' and r.Close>=f.at[day,'ma20']:pending=(1,'back_to_ma20')
                elif exit_rule=='partial':
                    if not pos['partial_taken'] and r.Close>f.at[day,'ma20']:pending=(.5,'half_above_ma20')
                    elif pos['partial_taken'] and r.Close<f.at[day,'ma10']:pending=(1,'remainder_below_ma10')
            elif exit_rule=='trailing':
                pos['highwater']=max(pos['highwater'],r.Close)
                pos['stop']=max(pos['stop'],pos['highwater']-3*f.at[day,'atr'])
                if r.Close<pos['stop']:pending=(1,'close_below_trail')
            else:raise ValueError(exit_rule)
        value=pos['qty']*r.Close if pos else 0.;eq=cash+value
        assert cash>=-1e-6
        daily.append({'date':str(day.date()),'equity':eq,'cash':cash,'invested':value,'exposure':value/eq,'return':eq/prev_eq-1,
          'episode':pos['id'] if pos else 0,'positions':int(pos is not None),'qty':pos['qty'] if pos else 0.,
          'stop':pos['stop'] if pos else np.nan,'fees_cumulative':total_fee,'dividends_cumulative':total_div})
        prev_eq=eq
    openrec=[]
    if pos:
        pnl=value+pos['net_sale_proceeds']+pos['dividends']-pos['entry_outlay']
        openrec=[{'episode':pos['id'],'entry_date':str(pos['date'].date()),'remaining_qty':pos['qty'],'market_value':value,
          'partial_taken':pos['partial_taken'],'total_episode_pnl_marked':pnl,'sessions':last_processed_i-pos['index']}]
    cols=['episode','entry_date','exit_date','entry_price','initial_qty','entry_outlay','net_sale_proceeds','dividends','entry_fee','exit_fees','net_pnl','net_return','sessions','partial_taken','exit_reason','worst_close_episode_return']
    t=pd.DataFrame(trades,columns=cols)
    assert abs(eq-initial-t.net_pnl.sum()-sum(p['total_episode_pnl_marked'] for p in openrec))<initial*1e-6
    return pd.DataFrame(daily),t,pd.DataFrame(orders),openrec

def summarize(d,t,start,end):
    dd=d[(d.date>=start)&(d.date<=end)];tt=t[(t.exit_date>=start)&(t.exit_date<=end)];rr=tt.net_return
    wealth=(1+dd['return']).cumprod();peak=np.maximum.accumulate(np.r_[1,wealth])[1:]
    years=((pd.Timestamp(dd.date.iloc[-1])-pd.Timestamp(dd.date.iloc[0])).days+1)/365.25
    return {'days':len(dd),'trades':len(tt),'win_rate':float((rr>0).mean()) if len(rr) else None,
      'cagr':float(wealth.iloc[-1]**(1/years)-1),'total_return':float(wealth.iloc[-1]-1),'max_drawdown':float((wealth/peak-1).min()),
      'mean_win':float(rr[rr>0].mean()),'mean_loss':float(rr[rr<0].mean()),'worst_trade':float(rr.min()),
      'mean_hold':float(tt.sessions.mean()),'median_hold':float(tt.sessions.median()),'max_hold':int(tt.sessions.max()) if len(tt) else None,
      'average_exposure':float(dd.exposure.mean()),'fraction_days_in_market':float((dd.positions>0).mean()),
      'partial_episodes':int(tt.partial_taken.sum()),'profit_factor':float(tt.loc[tt.net_pnl>0,'net_pnl'].sum()/-tt.loc[tt.net_pnl<0,'net_pnl'].sum()) if (tt.net_pnl<0).any() else None}

def main():
    data=load();stats=[];daily=[];trades=[];orders=[];opens=[];yearly=[]
    for symbol,h in data.items():
        f=features(h,data['SPY']);f.to_csv(P/f'{symbol}_features.csv')
        for name,(entry,rule) in STRATEGIES.items():
            for cost in [5,10,20]:
                d,t,o,u=simulate(h,f,entry,rule,cost)
                for split,(a,b) in {'early':('2011-01-01','2018-12-31'),'late':('2019-01-01','2026-09-18'),'full':('2011-01-01','2026-09-18')}.items():
                    stats.append(dict(symbol=symbol,strategy=name,cost_bps=cost,split=split,**summarize(d,t,a,b)))
                for frame in [d,t,o]:frame['symbol']=symbol;frame['strategy']=name;frame['cost_bps']=cost
                daily.append(d);trades.append(t);orders.append(o);opens.extend([dict(p,symbol=symbol,strategy=name,cost_bps=cost) for p in u])
                if cost==5:
                    for year in range(2011,2027):yearly.append(dict(symbol=symbol,strategy=name,year=year,**summarize(d,t,f'{year}-01-01',f'{year}-12-31')))
                print(symbol,name,cost,'episodes',len(t),'equity',round(d.equity.iloc[-1]),flush=True)
    pd.DataFrame(stats).to_csv(P/'metrics.csv',index=False)
    pd.concat(daily).to_csv(P/'daily_equity.csv.gz',index=False,compression='gzip')
    pd.concat(trades).to_csv(P/'trades.csv',index=False);pd.concat(orders).to_csv(P/'orders.csv',index=False)
    pd.DataFrame(opens).to_csv(P/'open_positions.csv',index=False);pd.DataFrame(yearly).to_csv(P/'yearly.csv',index=False)
    print(pd.DataFrame(stats).query("symbol=='QQQ' and cost_bps==5 and split=='full'")[['strategy','trades','win_rate','cagr','max_drawdown','median_hold','max_hold','average_exposure']].to_string(index=False))
if __name__=='__main__':main()
