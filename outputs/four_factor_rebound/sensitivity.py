import json
from prepare import prepare,P
from factors import build,ridge_predictions
from run import VARIANTS,planner
from validate_selected import summarise
import historical_backtest as h

def main():
 out=P/'validated_round1';cfg=next(x for x in VARIANTS if x['name']==json.loads((out/'SELECTION.json').read_text())['selected']);m,d=prepare();f=build(d);pred,_=ridge_predictions(f)
 s=(f['groups'][0]+.5*f['groups'][1]+.5*pred.where(f['valid']).rank(axis=1,pct=True)+f['groups'][2]+f['groups'][3])/4
 cases=[dict(cfg,threshold=.72,name='threshold72'),dict(cfg,threshold=.78,name='threshold78'),dict(cfg,cap=8,name='hold8'),dict(cfg,cap=12,name='hold12')]
 report={'purpose':'Post-selection robustness only; never reselect the winner from these results.','cases':{}}
 for c in cases:
  h.plan=planner(c);ff=dict(score=s,mask=(f['event']&(s>=c['threshold'])).mul(f['market'],axis=0),atr=f['atr'],ma5=f['ma5'])
  r=h.run_account('baseline',d,'2019-01-02',m['end'],{'baseline':ff});report['cases'][c['name']]=summarise(r)
  (out/'SENSITIVITY.json').write_text(json.dumps(report,indent=2)+'\n');print(c['name'],json.dumps(report['cases'][c['name']]),flush=True)
if __name__=='__main__':main()
