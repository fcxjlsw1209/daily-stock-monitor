from pathlib import Path
import sys,json
import pandas as pd
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent/'daily_stock_monitor'))
from strategy import load_inputs,features
from engine import initial_state,plan,execute
from backtest_factors import summarize,SPLITS


def simulate(data,f,admission,exit_rule,cost):
    state=initial_state();ds=[];ts=[];os=[];dates=data.close.index
    config={'exit':exit_rule,'allocation':.10,'max_positions':5,'sector_cap':2}
    for day in dates[(dates>='2019-01-01')&(dates<='2026-09-18')]:
        decision=plan(state,day,dates,data.close,f['ma5'],f['atr'],f['masks'][admission],f['score'],data.sector,config)
        needed=set(state['positions'])|{x['symbol'] for x in decision['buys']}
        bars={s:dict(open=data.opening.at[day,s],close=data.close.at[day,s],dividends=data.dividends.at[day,s]) for s in needed}
        state,d,t,o,_=execute(state,decision,bars,dates,cost,data.sector);ds.append(d);ts.extend(t);os.extend(o)
    cols=['symbol','entry_date','exit_date','entry_price','exit_price','quantity','entry_outlay','dividends','entry_fee','exit_fee','net_pnl','net_return','sessions','reason']
    trades=pd.DataFrame(ts,columns=cols)
    unrealized=sum(p['qty']*data.close.at[dates[-1],s]+p['dividends']-p['cost'] for s,p in state['positions'].items())
    assert abs(state['equity']-state['initial']-trades.net_pnl.sum()-unrealized)<1e-5
    return pd.DataFrame(ds),trades,pd.DataFrame(os),state


def run():
    data,raw=load_inputs();f=features(data,raw);metrics=[];daily=[];trades=[];orders=[];years=[]
    for a in ['strict','industry','broad']:
        for e in ['fixed5','rebound5','guarded10']:
            name=a+'_'+e
            for cost in [5,10,20]:
                d,t,o,s=simulate(data,f,a,e,cost)
                for split,(lo,hi) in SPLITS.items():metrics.append(dict(strategy=name,admission=a,exit=e,cost_bps=cost,split=split,**summarize(d,t,lo,hi)))
                for year in range(2019,2027):years.append(dict(strategy=name,cost_bps=cost,year=year,**summarize(d,t,f'{year}-01-01',f'{year}-12-31')))
                for frame in [d,t,o]:frame['strategy']=name;frame['cost_bps']=cost
                daily.append(d);trades.append(t);orders.append(o)
                print(name,cost,len(t),round(s['equity'],2),flush=True)
    m=pd.DataFrame(metrics);m.to_csv(P/'metrics.csv',index=False);pd.DataFrame(years).to_csv(P/'yearly.csv',index=False)
    pd.concat(daily).to_csv(P/'daily_equity.csv.gz',index=False,compression='gzip');pd.concat(trades).to_csv(P/'trades.csv.gz',index=False,compression='gzip');pd.concat(orders).to_csv(P/'orders.csv.gz',index=False,compression='gzip')
    rows=[]
    for name,g in m[m.cost_bps==20].groupby('strategy'):
        d=g.set_index('split');a=d.loc['all'];early=d.loc['earlier'];recent=d.loc['recent']
        ok=bool((d.cagr>0).all() and early.closed_trades>=100 and recent.closed_trades>=100 and a.trade_win>=.60 and recent.trade_win>=.55 and -a.max_drawdown<=.20)
        rank=min(early.cagr,recent.cagr)/max(-a.max_drawdown,.03)
        rows.append(dict(strategy=name,qualified=ok,selection_score=rank,recent_win=recent.trade_win,admission=a.admission,exit=a['exit']))
    ranked=pd.DataFrame(rows).sort_values(['qualified','selection_score','recent_win','strategy'],ascending=[False,False,False,True]);ranked.to_csv(P/'selection.csv',index=False)
    qualified=ranked[ranked.qualified];choice=None if qualified.empty else qualified.iloc[0].to_dict()
    (P/'choice.json').write_text(json.dumps(choice,indent=2))
    print(m.query("cost_bps==10 and split=='all'")[['strategy','closed_trades','cagr','trade_win','max_drawdown','average_exposure']].to_string(index=False));print(ranked.to_string(index=False))
if __name__=='__main__':run()
