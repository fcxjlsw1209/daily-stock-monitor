"""Clock/universe accounting comparisons, no parameter optimization."""
import rebuild as b
import clock_execution as ce
from repair_actions import patch
import pandas as pd,numpy as np,json,copy
P=b.P/'diagnostics';P.mkdir(exist_ok=True)

def monthly_planner(clock,monthends):
 def plan(name,state,day,dates,close,ma5,atr,mask,score,sectors,reps,issuers):
  signal=day if clock=='same_close_diagnostic' else dates[dates.get_loc(day)-1]
  result=dict(session=str(day.date()),signal_date=str(signal.date()),sells=[],buys=[],equity_basis=state['equity'],candidate_count=0)
  if day not in monthends:return result
  pool=score.loc[signal].where(mask.loc[signal]).dropna().sort_values(ascending=False,kind='stable');chosen=[]
  for s,v in pool.items():
   if s not in reps or issuers[s] in {issuers[t] for t in chosen}:continue
   if sum(sectors[t]==sectors[s] for t in chosen)>=2:continue
   chosen.append(s)
   if len(chosen)==3:break
  if len(chosen)<2:chosen=[]
  result['sells']=[dict(symbol=s,reason='month_end_rebalance') for s in state['positions']]
  result['buys']=[dict(symbol=s,budget=state['equity']/len(chosen),score=float(pool[s]),atr=float(atr.at[signal,s]),reference_close=float(close.at[signal,s])) for s in chosen]
  return result
 return plan

def main():
 # Resume only exact predetermined cases after an isolated module import issue.
 b.dump(P/'CONFIG.json',dict(declared_before_results=True,monthly_recipe='Frozen original base: R12-2 50,riskadjusted20,sectorrelative15,CMF10,high52 5. No model. Positive momentum. Top3, sector2, equal actual liquidation proceeds, fixedshares intramonth.',cases='Original snapshot vs forwardprice-scaled pool; priorclose vs sameclose signal; costs0/10bps. Existing lastyear-selected daily baseline independent2016restart also shown, not refit.',timing='priorclose case fills next session monthend CLOSE; sameclose uses final monthend bar at its own close, diagnostic only. Both price/action clocks correct.',data_certified=False))
 m,d,old=b.prepare();d=patch(d);dates=d['close'].index;monthends=set(dates[:-1][dates[:-1].to_period('M')!=dates[1:].to_period('M')]);summary=json.loads((P/'SUMMARY.json').read_text()) if (P/'SUMMARY.json').exists() else {}; (P/'runs').mkdir(exist_ok=True)
 for pool in ['snapshot','scaled']:
  md=dict(d);md['eligible']=old if pool=='snapshot' else d['eligible'];md['opening']=md['close'];_,mf=b.r.features(md)
  ff=dict(score=mf['base'],mask=mf['valid']&mf['positive'],atr=mf['atr'],ma5=mf['ma5'])
  for clock in ['next_open','same_close_diagnostic']:
   for cost in [0,10]:
    key=f'monthly_{pool}_{"prior_close" if clock=="next_open" else clock}_cost{cost}';oldexec=ce.original_execute
    if key in summary:continue
    try:
     ce.original_execute=b.r.execute_equal_close
     out=ce.run_account(md,ff,monthly_planner(clock,monthends),b.START,b.END,cost,clock)
    finally:ce.original_execute=oldexec
    b.dump(P/'runs'/f'{key}.json',out)
    if out['status']!='complete':raise ValueError({k:out.get(k) for k in ['error','blocked_session']})
    summary[key]=b.metrics(out);b.dump(P/'SUMMARY.json',summary);print(key,summary[key]['full'],flush=True)
 # Original daily research baseline, no re-optimization and unchanged original pool.
 import sys,subprocess
 subprocess.run([sys.executable,str(b.P/'old_control.py')],check=True)
 for cost in [0,10]:
  key=f'old_daily_cap30_cost{cost}';out=json.loads((P/'runs'/f'{key}.json').read_text());summary[key]=b.metrics(out);b.dump(P/'SUMMARY.json',summary);print(key,summary[key]['full'],flush=True)
 # Baseline follows screenshot close-to-close; also disclose calendar vs252day CAGR.
 a=d['adj']['QQQ'].loc[b.START:b.END];wealth=a/a.iloc[0];n=len(a)-1;years=(a.index[-1]-a.index[0]).days/365.25;returns=a.pct_change().dropna()
 b.dump(P/'QQQ.json',dict(start=b.START,end=b.END,wealth=float(wealth.iloc[-1]),calendar_cagr=float(wealth.iloc[-1]**(1/years)-1),cagr_252=float(wealth.iloc[-1]**(252/n)-1),sharpe=float(returns.mean()/returns.std()*np.sqrt(252)),max_dd=float((wealth/wealth.cummax()-1).min()),n_returns=n,zero_cost=True,dividend_reinvestment_approx=True,not_a_new_ledger=True))
 b.dump(P/'CHECKS.json',dict(complete=True,runs=len(summary)))
if __name__=='__main__':main()
