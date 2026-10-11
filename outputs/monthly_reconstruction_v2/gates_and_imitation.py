"""Trend eligibility hypotheses and inverse-path imitation, all isolated.
Imitation uses a hindsight teacher from a later-published chart; never call it
independent OOS or point-in-time available training labels.
"""
import json
import numpy as np,pandas as pd
from research import P,O,load,raw_features,Simulator,fixed_rules,MODEL_FEATURES,dump,metrics,fit_errors

def main():
 px,fs,pools=load();raw=raw_features(px,fs);obs=pd.read_csv(O/'monthly_curve_reverse/DIGITIZED_WEALTH.csv');cal=json.loads((O/'monthly_curve_reverse/CALIBRATION.json').read_text());base_results=json.loads((P/'RESULTS.json').read_text());gates={
 'above50':px>px.rolling(50).mean(),
 'above100':px>px.rolling(100).mean(),
 'above200':px>px.rolling(200).mean(),
 'positive63':raw['r63']>0,
 'positive21':raw['r21']>0,
 'above50_and_positive63':(px>px.rolling(50).mean())&(raw['r63']>0),
 'above100_and_positive21':(px>px.rolling(100).mean())&(raw['r21']>0)}
 cfg=dict(gates=list(gates),base_rules=['m252_42','m294_63','risk','balanced'],pools=['core17','theme17'],goal='Test current-trend qualification implied by strong intermediate trend; comparison selected by development curve-fit error, not target CAGR.',inverse_imitation=dict(teacher=['INVERSE_core17_q0.json','INVERSE_theme17_q0.json','INVERSE_theme17_q0.5.json'],training_end='2021-12-31',alphas=[10,100],features=MODEL_FEATURES,blend=[.3,.6],strict_warning='Teacher holdings use future monthly returns AND full future curve. Training cutoff refers to feature/label months, not true public availability. All imitation results retrospective explanatory diagnostics. Neither original holdings nor deployable alpha verified.'),phase_informed_by_prior_results=True)
 dump(P/'GATES_CONFIG.json',cfg);out={};orders_out={};curves={};coef_out={}
 def save(tag,pool,sc,gate,sim,teacher=False):
  nav,orders=sim.run(sc,gate);result=dict(pool=pool,**metrics(nav),**fit_errors(nav,cal,obs),inverse_teacher=teacher)
  result['development_selection_error']=result['development_global_log_rmse']+2*result['development_month_shape_log_rmse'];out[tag]=result;orders_out[tag]=orders;curves[tag]=nav;dump(P/'GATES_RESULTS.json',out);print(tag,round(result['cagr_252']*100,2),round(result['max_dd']*100,2),flush=True)
 for pool in ['core17','theme17']:
  member=pools[pool];rank={k:v.where(member).rank(axis=1,pct=True) for k,v in raw.items()};rules=fixed_rules(rank,raw);positive=raw['m252_42']>0;sim=Simulator(px,fs,member)
  for name in cfg['base_rules']:
   for gn,gate in gates.items():save(pool+'__'+name+'__'+gn,pool,rules[name],positive&gate,sim)
  # Models selected only on earlier screenshot fit, then add two explicit trend gates.
  candidates=[(k,v) for k,v in base_results.items() if k.startswith(pool+'__ridge')];selected=min(candidates,key=lambda x:x[1]['development_selection_error'])[0];n=selected.split('__')[1];parts=n.split('_');tree='tree'in parts;window=parts[2] if tree else parts[1];target=parts[3] if tree else parts[2];w=float(parts[-1][1:]);fn=f'{pool}_{window}_{target}'+('_tree' if tree else '')+'.csv.gz';pred=pd.read_csv(P/'predictions'/fn,index_col=0,parse_dates=True);sc=(1-w)*rules['balanced']+w*pred.where(member).rank(axis=1,pct=True)
  for gn in ['above50','above100_and_positive21']:save(pool+'__selected_model__'+gn,pool,sc,positive&gates[gn],sim)
  # Infer coarse feature preferences from all three admissible inverse paths.
  teacher_paths=[json.loads((P/f).read_text())['orders'] for f in cfg['inverse_imitation']['teacher']];teacher={}
  for path in teacher_paths:
   for row in path:
    d=pd.Timestamp(row['date']);acc=teacher.setdefault(d,{})
    for s in row['holdings']:acc[s]=acc.get(s,0)+1/len(teacher_paths)
  xx=[];yy=[];ww=[];months=[]
  for d,choices in sorted(teacher.items()):
   if d>pd.Timestamp(cfg['inverse_imitation']['training_end']):continue
   valid=sim.valid.loc[d]&positive.loc[d];frame=pd.DataFrame({k:rank[k].loc[d] for k in MODEL_FEATURES}).loc[valid].dropna();y=np.array([choices.get(s,0) for s in frame.index]);wgt=np.where(y>0,(len(y)-y.sum())/max(y.sum(),1),1.)
   wgt=wgt/wgt.sum();xx.extend(frame.to_numpy());yy.extend(y);ww.extend(wgt);months.extend([str(d.date())]*len(frame))
  X=np.asarray(xx);Y=np.asarray(yy);W=np.asarray(ww);W*=len(W)/W.sum();mu=np.average(X,axis=0,weights=W);sd=np.sqrt(np.average((X-mu)**2,axis=0,weights=W));sd[sd<1e-6]=1;z=(X-mu)/sd;ym=np.average(Y,weights=W)
  for alpha in cfg['inverse_imitation']['alphas']:
   coef=np.linalg.solve(np.einsum('ni,nj,n->ij',z,z,W)+alpha*np.eye(z.shape[1]),np.einsum('ni,n,n->i',z,W,Y-ym));scores=sum(((rank[k]-mu[j])/sd[j])*coef[j] for j,k in enumerate(MODEL_FEATURES));scores=scores.where(member).rank(axis=1,pct=True)
   coef_out[f'{pool}_{alpha}']=dict(coefficients={k:float(v) for k,v in zip(MODEL_FEATURES,coef)},rows=len(Y),train_feature_date_max=max(months),teacher_available_only_after_chart=True)
   for w in cfg['inverse_imitation']['blend']:
    sc=(1-w)*rules['m252_42']+w*scores;tag=f'{pool}__IMI_alpha{alpha}_w{w}';save(tag,pool,sc,positive,sim,True)
    # Separate genuinely later feature dates, still not independent point-in-time training.
    later=curves[tag].loc['2022-01-03':];out[tag]['post2022_diagnostic']=metrics(later)
 dump(P/'GATES_RESULTS.json',out);dump(P/'IMITATION_COEFFICIENTS.json',coef_out);dump(P/'GATES_ORDERS.json',orders_out);pd.DataFrame(curves).to_csv(P/'GATES_CURVES.csv.gz')
 eligible={k:v for k,v in out.items() if not v['inverse_teacher']};selection=min(eligible,key=lambda k:eligible[k]['development_selection_error']);best=max(eligible,key=lambda k:eligible[k]['cagr_252']);dump(P/'GATES_SELECTION.json',dict(development_selected=selection,best_full_cagr_retrospective=best,all_trials=len(out),causal_rule_trials=len(eligible),teacher_imitation_is_not_causal=True));print('SELECTED',selection,'BEST',best)
if __name__=='__main__':main()
