from pathlib import Path
import sys,json,functools,hashlib
import pandas as pd,numpy as np
from prepare import prepare,P,U
from factors import build,ridge_predictions
from run import VARIANTS,planner,period_metrics
import historical_backtest as h

def summarise(r):
 if r['status']!='complete':return {'status':r['status'],'error':r.get('error'),'blocked_session':r.get('blocked_session')}
 x=pd.DataFrame(r['daily']);return dict(status='complete',metrics=r['metrics'],development=period_metrics(r['daily'],'2019-01-02','2023-12-31'),evaluation=period_metrics(r['daily'],'2024-01-01','2026-10-06'),year2024=period_metrics(r['daily'],'2024-01-01','2024-12-31'),year2025=period_metrics(r['daily'],'2025-01-01','2025-12-31'),year2026=period_metrics(r['daily'],'2026-01-01','2026-10-06'),last252_return=float(x.equity.iloc[-1]/x.equity.iloc[-253]-1))
def main():
 out=P/'validated_round1';selected=json.loads((out/'SELECTION.json').read_text())['selected']
 if selected is None:raise ValueError('No development-qualified candidate')
 cfg=next(r for r in VARIANTS if r['name']==selected);m,d=prepare();f=build(d);pred,_=ridge_predictions(f)
 score=(f['groups'][0]+.5*f['groups'][1]+.5*pred.where(f['valid']).rank(axis=1,pct=True)+f['groups'][2]+f['groups'][3])/4 if cfg['model'] else f['score']
 ff=dict(score=score,mask=(f['event']&(score>=cfg['threshold'])).mul(f['market'],axis=0),atr=f['atr'],ma5=f['ma5'])
 original_plan=h.plan;original_exec=h.execute;original_terminal=h.terminal_baskets.apply;original_spin=h.spin_distributions.apply
 h.plan=planner(cfg)
 def expensive(*args,**kwargs):kwargs['cost_bps']=30;return original_exec(*args,**kwargs)
 h.execute=expensive;h.terminal_baskets.apply=functools.partial(original_terminal,cost_bps=30);h.spin_distributions.apply=functools.partial(original_spin,cost_bps=30)
 r=h.run_account('baseline',d,'2019-01-02',m['end'],{'baseline':ff});report={'selected':selected,'cost_30bps':summarise(r)}
 (out/'selected_cost30.json').write_text(json.dumps(r,allow_nan=False)+'\n');print('cost stress',json.dumps(report),flush=True)
 h.plan=original_plan;h.execute=original_exec;h.terminal_baskets.apply=original_terminal;h.spin_distributions.apply=original_spin
 r=h.run_account('accel40',d,'2019-01-02',m['end']);report['incumbent_accel40_same_universe']=summarise(r);(out/'incumbent_accel40.json').write_text(json.dumps(r,allow_nan=False)+'\n')
 # Fully invested, same start, buy cost5bps, total-return adjustment approximation.
 dates=d['adj'].loc['2019-01-02':m['end']].index;start=dates[0];ao=d['opening'].at[start,'QQQ']*d['adj'].at[start,'QQQ']/d['close'].at[start,'QQQ'];eq=100000/(1.0005)*d['adj'].loc[dates,'QQQ']/ao
 daily=[dict(date=str(t.date()),equity=float(v)) for t,v in eq.items()]
 report['qqq_buy_hold']={k:period_metrics(daily,a,b) for k,a,b in [('full','2019-01-02',m['end']),('development','2019-01-02','2023-12-31'),('evaluation','2024-01-01',m['end']),('year2024','2024-01-01','2024-12-31'),('year2025','2025-01-01','2025-12-31'),('year2026','2026-01-01',m['end'])]};report['qqq_buy_hold']['last252_return']=float(eq.iloc[-1]/eq.iloc[-253]-1)
 r=json.loads((out/(selected+'.json')).read_text());tr=pd.DataFrame(r['trades']);recent=tr.loc[tr.exit_date>='2024-01-01'];report['recent_realized_pnl_top_symbols']=recent.groupby('symbol').net_pnl.sum().sort_values(ascending=False).head(10).to_dict();report['recent_closed_trades']=len(recent)
 report['input_manifest_sha256']=hashlib.sha256((U/'alias_resolved_inputs/MANIFEST.json').read_bytes()).hexdigest();report['source_limitations']='Same underlying available-data and identity/sector assumptions, plus self-selected17ETF menu; not exact capTop32 replication.'
 report['code_sha256']={n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in ['prepare.py','factors.py','run.py','validate_selected.py']}
 (out/'VALIDATION.json').write_text(json.dumps(report,indent=2)+'\n');print('validation done',flush=True)
if __name__=='__main__':main()
