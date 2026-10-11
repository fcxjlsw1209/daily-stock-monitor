"""Supplemental earlier-period ETF check; same frozen rules, no tuning."""
import pandas as pd
from backtest_swing import load,signals,simulate,summarize,P

def main():
    data=load();models=signals(data);metrics=[];daily=[];trades=[]
    for name in ['SPY_pullback','QQQ_pullback']:
        mask,score=models[name]
        for rule in ['fixed3','fixed5','ma5_max5']:
            for cost in [5,10]:
                d,t,o,u=simulate(data,mask,score,rule,cost,start='2011-01-01',etf=True)
                for split,(a,b) in {'early_extension':('2011-01-01','2018-12-31'),'later':('2019-01-01','2026-09-18'),'full':('2011-01-01','2026-09-18')}.items():
                    metrics.append(dict(strategy=name,exit_rule=rule,cost_bps=cost,split=split,**summarize(d,t,a,b)))
                for frame in [d,t]:frame['strategy']=name;frame['exit_rule']=rule;frame['cost_bps']=cost
                daily.append(d);trades.append(t)
    m=pd.DataFrame(metrics);m.to_csv(P/'etf_extended_metrics.csv',index=False)
    pd.concat(daily).to_csv(P/'etf_extended_equity.csv.gz',index=False,compression='gzip')
    pd.concat(trades).to_csv(P/'etf_extended_trades.csv',index=False)
    print(m[(m.strategy=='QQQ_pullback')&(m.cost_bps==5)][['exit_rule','split','closed_trades','cagr','trade_win','max_drawdown','fraction_days_in_market']].to_string(index=False))
if __name__=='__main__':main()
