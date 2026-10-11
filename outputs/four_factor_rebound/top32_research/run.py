from pathlib import Path
import sys,json,hashlib,copy
import numpy as np,pandas as pd
from prepare import prepare,P,U
from factors import build,ridge_predictions
sys.path.insert(0,str(U))
import historical_backtest as h
from research_scope import sha

VARIANTS=[dict(name='rules_h5',model=False,cap=5,threshold=.65,tp=2.,sl=2.,replace=.15),dict(name='rules_h10',model=False,cap=10,threshold=.65,tp=3.,sl=2.,replace=.15),dict(name='rules_selective',model=False,cap=10,threshold=.75,tp=3.,sl=2.,replace=.15),dict(name='rules_no_replace',model=False,cap=10,threshold=.65,tp=3.,sl=2.,replace=2.),dict(name='ridge_h5',model=True,cap=5,threshold=.65,tp=2.,sl=2.,replace=.15),dict(name='ridge_h10',model=True,cap=10,threshold=.65,tp=3.,sl=2.,replace=.15),dict(name='ridge_selective',model=True,cap=10,threshold=.75,tp=3.,sl=2.,replace=.15),dict(name='ridge_no_replace',model=True,cap=10,threshold=.65,tp=3.,sl=2.,replace=2.)]

def planner(cfg):
 def plan(name,state,day,dates,close,ma5,atr,mask,score,sectors,reps,issuers):
  i=dates.get_loc(day);prev=dates[i-1];scores=score.loc[prev];pool=scores.where(mask.loc[prev]).dropna().sort_values(ascending=False,kind='stable');pool=pool[[s in set(reps) for s in pool.index]]
  sells=[];remaining=set(state['positions'])
  for s,p in state['positions'].items():
   age=i-dates.get_loc(pd.Timestamp(p['entry_date']));price=close.at[prev,s];why=None
   if age>=cfg['cap']:why='holding_limit'
   elif price<p['entry_price']-cfg['sl']*p['atr']:why='close_stop_atr'
   elif price>p['entry_price']+cfg['tp']*p['atr']:why='close_take_profit_atr'
   elif pd.notna(scores[s]) and scores[s]<.35:why='score_deterioration'
   elif age>=2 and len(pool) and pool.iloc[0]-scores[s]>cfg['replace']:why='better_candidate_gap'
   if len(pool)<2:why='fewer_than_two_qualified_cash'
   if why:sells.append(dict(symbol=s,reason=why));remaining.remove(s)
  sold={r['symbol'] for r in sells};selected=[]
  for s,value in pool.items():
   if len(remaining)+len(selected)>=3:break
   allheld=remaining|{x[0] for x in selected}
   if s in sold or s in allheld or issuers[s] in {issuers[t] for t in allheld}:continue
   if sum(sectors[t]==sectors[s] for t in allheld)>=2:continue
   selected.append((s,value))
  n=len(remaining)+len(selected)
  if n<2:
   for s in remaining:sells.append(dict(symbol=s,reason='fewer_than_two_feasible_cash'))
   selected=[];n=0
  buys=[dict(symbol=s,budget=float(state['equity']/n),score=float(v),atr=float(atr.at[prev,s]),reference_close=float(close.at[prev,s])) for s,v in selected]
  return dict(session=str(day.date()),signal_date=str(prev.date()),sells=sells,buys=buys,equity_basis=state['equity'],candidate_count=len(pool))
 return plan

def period_metrics(daily,a,b):
 x=pd.DataFrame(daily).set_index('date');v=x.loc[a:b]
 if v.empty:return None
 before=x.loc[x.index<a];base=float(before.equity.iloc[-1]) if len(before) else 100000.
 eq=np.r_[base,v.equity.to_numpy()];years=(pd.Timestamp(v.index[-1])-pd.Timestamp(v.index[0])).days/365.25
 return dict(return_=float(eq[-1]/base-1),cagr=float((eq[-1]/base)**(1/years)-1) if years>=1 else None,max_drawdown=float((eq/np.maximum.accumulate(eq)-1).min()))

def main():
 out=P/'validated_round1';out.mkdir(exist_ok=True)
 config=dict(variants=VARIANTS,stocks='existing historical NDX union SP100 available-data subset, not dynamic cap Top32',etfs='17 unlevered ETFs in ETF_SOURCES.json',development='2019-2023',evaluation='2024-2026-10-06; previously examined market history, not pristine holdout',selection='Choose highest 2019-2023 CAGR among complete candidates with development drawdown <=40%; if none, no selected strategy.',recent_windows=['2024','2025','2026 YTD','last 252 sessions'],four_group_weights=[.25]*4,cost_bps=10,leverage=1,entry='prior close signal next open',cash_interest=0,position_rule='2 or 3 holdings; new entries budget equity/N limited by cash, existing share counts unchanged; not daily rebalanced',ridge='NumPy exact L2 solver; alpha10, trailing756 sessions, monthly refit, all labels mature before fit; replaces half of rebound-group rank only',limitations='Underlying available-data study limitations inherited; research variants are newly specified, not replication of original SOTA description.')
 (out/'CONFIG.json').write_text(json.dumps(config,indent=2)+'\n')
 m,d=prepare();print('data ready',flush=True);f=build(d);print('rule factors ready',flush=True);pred,audit=ridge_predictions(f);(out/'TRAINING_AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n');print('ridge ready',len(audit),flush=True)
 modelscore=(f['groups'][0]+.5*f['groups'][1]+.5*pred.where(f['valid']).rank(axis=1,pct=True)+f['groups'][2]+f['groups'][3])/4
 summary={}
 for cfg in VARIANTS:
  s=modelscore if cfg['model'] else f['score'];ff=dict(score=s,mask=(f['event']&(s>=cfg['threshold'])).mul(f['market'],axis=0),ma5=f['ma5'],atr=f['atr'])
  h.plan=planner(cfg)
  r=h.run_account('baseline',d,'2019-01-02',m['end'],{'baseline':ff})
  v=dict(status=r['status'],error=r.get('error'),blocked_session=r.get('blocked_session'))
  if r['status']=='complete':
   v.update(metrics=r['metrics'],development=period_metrics(r['daily'],'2019-01-02','2023-12-31'),evaluation=period_metrics(r['daily'],'2024-01-01',m['end']),year2024=period_metrics(r['daily'],'2024-01-01','2024-12-31'),year2025=period_metrics(r['daily'],'2025-01-01','2025-12-31'),year2026=period_metrics(r['daily'],'2026-01-01',m['end']))
   x=pd.DataFrame(r['daily']);rolling=x.equity/x.equity.shift(252)-1;v['last252_return']=float(rolling.iloc[-1]);v['best_rolling252_return']=float(rolling.max());v['best_rolling252_end']=x.loc[rolling.idxmax(),'date'];v['recent_100pct_target_met']=v['last252_return']>=1
  (out/(cfg['name']+'.json')).write_text(json.dumps(r,allow_nan=False)+'\n');summary[cfg['name']]=v
  (out/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n');print(cfg['name'],json.dumps(v),flush=True)
 eligible=[n for n,v in summary.items() if v['status']=='complete' and v['development']['max_drawdown']>=-.4]
 selected=max(eligible,key=lambda n:summary[n]['development']['cagr']) if eligible else None
 (out/'SELECTION.json').write_text(json.dumps(dict(selected=selected,criterion=config['selection'],not_selected_on_recent_returns=True),indent=2)+'\n')
 print('selected',selected,flush=True)
if __name__=='__main__':main()
