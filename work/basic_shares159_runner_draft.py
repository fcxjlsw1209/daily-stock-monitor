"""Synthetic-ready round159 wiring draft. Real execution is deliberately disabled.

The pure functions below accept supplied arrays/events only. Source admission,
registered PLAN bindings, cache publication and independent phase receipts must
be connected by the final lifecycle before real data may enter this module.
"""
from copy import deepcopy
from pathlib import Path
import argparse
import hashlib
import json
import math
import numpy as np
import pandas as pd
import after158_basic_shares_kernel as kernel
import monthly_forward_guidance132_runner as base
import monthly_annual_lag158_runner as previous

TRAIN, MONTHS, WIDTH, DOSE = 88, 122, 49, .01
NAMES=('incumbent_local67','centered_negative_basic_share_growth_magnitude','centered_negative_basic_share_growth_rank')
HANDS=('basic_share_magnitude','basic_share_rank')
EXTRA=HANDS+('common_available',)
ISSUERS=('AAPL','AMD','MSFT','NVDA')
INITIAL_PARTIAL='2016-08-17'
OUT='outputs/monthly_basic_shares159'
DESIGN='work/after158_basic_shares_design_review.json'
INPUTS=('proposal','meta','policy','original_orders','prices','protected_manifest','design','calendar','source_manifest','source_review','source_events')
CACHE_FILES=('DATA.npz','SCORES.npz','META.json','PRETRAIN_ORDERS.json','PRICE_PARITY.json','FEATURE_AUDIT.json','CALENDAR.json','SUPPORT_AUDIT.json')
require=base.require


def measurement_contract():
 return dict(arms=list(NAMES),source_scope=list(ISSUERS),raw='1-current_basic_annual/same_report_prior_basic_annual',arithmetic='exact_decimal_to_Fraction; bounded score conversion only',magnitude='(x/(0.05+abs(x))-mean_over_common_eligible_population)/2',rank='2*((average_ascending_rank-1)/(n-1)-0.5)',dose=DOSE,max_abs_delta=.01,common_mask='admitted_latest_same_report_pair_intersect_original_eligibility; no ETF',known_at='second_XNYS_session_strictly_after_release_date_close',source_cutoff='original_signal_session_minus42_close',max_age_days=550,initial_partial='exact_incumbent',zero_delta='exact_incumbent',placement='unchanged_Local67_config4_pre_plus_one_delta_then_one_original_episode',original_arrays=27,extra_arrays=list(EXTRA),new_predictive_model_fits=0,nuisance_estimates=0)


def selection_contract():
 return dict(selection_count=1,first88_only=True,early53=True,recent35=True,later34_diagnostic_only=True,prefix_reselection=False,minimum_source_events=12,minimum_source_issuers=3,minimum_multi_source_months=24,minimum_multi_source_months_each_era=8,minimum_close_cagr_gain=.005,cagr_epsilon=1e-12,open_cost_noninferior=True,sharpe_tolerance=.05,drawdown_tolerance=.03,era_cagr_tolerance=.025,minimum_changed_held_set_months=3,minimum_changed_each_era=1,minimum_changed_source_events=3,minimum_changed_source_issuers=2,choice='highest_same_close_first88_CAGR; ties_within1e-12_lower_fixed_arm_index',execution_paths=9)


def integration_contract():
 return dict(status='DRAFT_REAL_EXECUTION_DISABLED',prepare_order=['verify_registered_round159_PLAN_and_all_code_parameter_runtime_pins','verify_independent_field_source_gate_and_all_5978_rows_before_backend_load','backend.load_original_27_arrays_then_independent_price_parity','build_features_then_pre_outcome_support_gate','panels_then_write_once_cache','independent_prepare_receipt_before_any_returns'],select_order=['reverify_PLAN_source_cache_and_independent_prepare','Backend.training_first88_once','select_first88_with_source_and_changed_held_set_support','write_once_all_three_orders_and_FROZEN'],diagnose_order=['reverify_FROZEN_and_prepare_receipt','write_DIAGNOSE_STARTED_with_PLAN_and_FROZEN_SHA','Backend.diagnostics_nine_paths','independent_all9paths_366orders_accounting','formal_lifecycle_closure_and_protected_archive'],required_receipt_counts={'candidate_cells':5978,'orders':366,'prefixes':123,'feature_matrices':2,'paths':9},unimplemented=['final_source_gate_schema_and_exact_hashes','registered_PLAN_loader_and_all_phase_CAS_write_once_guards','independent_prepare_outcome_receipts_and_source_to_cell_validation','descriptive_leave_one_issuer_and_source_event_contribution_summaries_without_reselection'])


def build_features(meta,events,eligible,cutoffs,n=None,source_rows=None):
 """Pure in-memory producer. Caller must verify source/identity before this call.

No raw growth float is persisted: exact rational numerator/denominator strings
remain in cell audit. Only the two bounded corrections become float arrays.
When supplied, source_rows retains the full original axis even for a prefix.
Its final row gate is applied before either cross-sectional transformation.
"""
 total=len(meta['dates']);n=total if n is None else n
 require(type(n)is int and 0<=n<=total and np.asarray(eligible).shape==(total,WIDTH)and eligible.dtype==bool,'Original boolean eligibility/valid prefix required')
 if source_rows is not None:require(isinstance(source_rows,list)and len(source_rows)==total*WIDTH,'Full original source-row axis required')
 result={key:np.zeros((n,WIDTH),dtype=bool if key=='common_available'else float)for key in EXTRA};cells=[];monthly=[]
 for i in range(n):
  names=meta['symbols'][i];require(len(names)==len(set(names))==WIDTH,'49 unique original candidates')
  initial=meta['dates'][i]==INITIAL_PARTIAL
  allowed=[bool(eligible[i,j] and s in ISSUERS and s not in meta['etfs'])for j,s in enumerate(names)]
  sources=None if source_rows is None else source_rows[i*WIDTH:(i+1)*WIDTH]
  if sources is not None:
   for j,(s,source)in enumerate(zip(names,sources)):
    require(isinstance(source,dict)and (source.get('signal_index'),source.get('candidate_index'),source.get('symbol'))==(i,j,s),'Original source-row order required')
    require(type(source.get('conditional_source_ready'))is bool,'Explicit source-row boolean gate required')
    require('selected_event_id'in source and (source['selected_event_id']is None or isinstance(source['selected_event_id'],str)),'Source-row selected event required')
    allowed[j]=allowed[j]and source['conditional_source_ready']
  row=kernel.evaluate_month(events,names,allowed,cutoffs[i],initial_partial=initial)
  for j,(s,audit,x)in enumerate(zip(names,row['audit'],row['negative_growth'])):
   available=x is not None;result['common_available'][i,j]=available
   item=dict(audit,signal_index=i,candidate_index=j,date=meta['dates'][i],cutoff=cutoffs[i],original_eligible=bool(eligible[i,j]),raw_exact=None)
   if sources is not None and not initial and eligible[i,j]and s not in meta['etfs']and not sources[j]['conditional_source_ready']:
    item.update(event_id=sources[j]['selected_event_id'],reason='source_row_not_conditionally_ready')
   if available:item['raw_exact']={'numerator':str(x.numerator),'denominator':str(x.denominator)}
   cells.append(item)
  for key,source in zip(HANDS,('magnitude','rank')):result[key][i]=row[source]
  monthly.append(dict(signal_index=i,date=meta['dates'][i],available_count=int(result['common_available'][i].sum()),initial_partial=initial))
 return dict(arrays=result,audit={'cells':cells,'monthly':monthly,'new_predictive_model_fits':0,'nuisance_estimates':0,'caller_source_gate_required':True})


def validated_hands(arrays,meta,n):
 shape=np.asarray(arrays['baseline']).shape
 require(len(shape)==2 and shape[1]==WIDTH and type(n)is int and 0<=n<=shape[0],'Valid original shape and prefix required')
 use=np.asarray(arrays['common_available']);eligible=np.asarray(arrays['eligible'])
 require(use.shape==eligible.shape==shape and use.dtype==eligible.dtype==bool,'Boolean common/eligible masks required')
 use=use[:n].copy();require(not (use & ~eligible[:n]).any(),'Original eligibility must precede centering')
 require(np.all(use.sum(axis=1)<=4),'At most four fixed issuers')
 for i,names in enumerate(meta['symbols'][:n]):
  require(len(names)==len(set(names))==WIDTH,'Original unique symbols')
  require(all(not use[i,j]or(s in ISSUERS and s not in meta['etfs'])for j,s in enumerate(names)),'Feature outside exact source-stock scope')
  if meta['dates'][i]==INITIAL_PARTIAL:require(not use[i].any(),'Initial partial must be neutral')
 hands={}
 for key in HANDS:
  raw=np.asarray(arrays[key]);require(raw.shape==shape,'Correction shape')
  part=raw[:n];require(all(isinstance(x,(int,float,np.integer,np.floating))and not isinstance(x,(bool,np.bool_))for x in part.flat),'Numeric bounded hand required')
  hand=part.astype(float,copy=True);require(np.isfinite(hand).all()and np.all(np.abs(hand)<=1),'Finite hand in [-1,1]')
  require(np.all(hand[~use]==0),'Uncovered candidates must have exact zero correction')
  for i in range(n):
   count=int(use[i].sum());require(abs(float(hand[i].sum()))<=1e-12,'Common population must be centered')
   if count<2:require(np.all(hand[i]==0),'Singleton/empty population must be neutral')
   if key==HANDS[0]and count:require(np.max(np.abs(hand[i]))<=(count-1)/count+1e-12,'Magnitude population bound')
  hands[key]=hand
 return hands,use


def choices_all(arrays,meta,n,backend,policy):
 """Add .01*hand once to the real Local67 pre-score, then frozen selector."""
 require(all(policy.get(k)==v for k,v in dict(budget=.05,ceiling=5,gap=.02).items()),'Original episode policy required')
 hands,use=validated_hands(arrays,meta,n)
 original,original_scores=backend.baseline(arrays,meta,n,policy)
 pre_original=np.asarray(backend.pre(arrays,n));shape=np.asarray(arrays['baseline']).shape
 require(pre_original.shape==shape,'Full original Local67 pre-score required')
 choices={NAMES[0]:original};scores={NAMES[0]:original_scores}
 for name,key in zip(NAMES[1:],HANDS):
  delta=DOSE*hands[key];changed=delta!=0.;require(np.all(np.abs(delta)<=.01),'Fixed delta cap')
  if not changed.any():
   choices[name]=deepcopy(original);scores[name]=original_scores.copy();continue
  pre=pre_original.copy();view=pre[:n].copy();view[changed]+=delta[changed];pre[:n]=view
  # The factor is only a bounded neutral dummy at budget0. The actual hand
  # has already been applied exactly once, including valid values beyond .5.
  chosen,score=backend.adapter.score_and_choose('incumbent66',0.,np.zeros(shape),arrays,meta,n,pre_score=pre,effects=arrays['effects'],episode_policy=policy,boundary_selector=backend.selector)
  for i in np.flatnonzero(~changed.any(axis=1)):
   require(chosen[i]==original[i]and score[i].tobytes()==original_scores[i].tobytes(),'Exact neutral-row incumbent parity')
  choices[name]=chosen;scores[name]=score
 audit=[dict(signal_index=i,date=meta['dates'][i],available_count=int(use[i].sum()),nonzero_hands={name:bool(np.any(hands[key][i]))for name,key in zip(NAMES[1:],HANDS)})for i in range(n)]
 return choices,scores,audit


def panels(arrays,meta,backend,policy):
 require(len(meta['dates'])==MONTHS,'Full122 pipeline')
 before={k:(v.dtype.str,v.shape,v.tobytes())for k,v in arrays.items()}
 choices,scores,audit=choices_all(arrays,meta,MONTHS,backend,policy)
 for n in range(MONTHS+1):
  cc,ss,_=choices_all(arrays,meta,n,backend,policy)
  require(all(cc[k]==choices[k][:n]and ss[k].tobytes()==scores[k][:n].tobytes()for k in NAMES),'Prefix choices/scores changed')
 require(before=={k:(v.dtype.str,v.shape,v.tobytes())for k,v in arrays.items()},'Original arrays mutated')
 return {k:backend.make_orders(choices[k],scores[k],meta)for k in NAMES},scores,audit


def source_support(cells):
 """Count source observations, not repeated months, before outcome access."""
 require(all(type(c.get('signal_index'))is int for c in cells),'Cell month index')
 first=[c for c in cells if c['signal_index']<TRAIN and c['available']]
 events={c['event_id']for c in first};issuers={c['symbol']for c in first}
 counts={i:sum(c['signal_index']==i for c in first)for i in range(TRAIN)}
 early=sum(counts[i]>=2 for i in range(53));recent=sum(counts[i]>=2 for i in range(53,88));total=early+recent
 return dict(passed=len(events)>=12 and len(issuers)>=3 and total>=24 and early>=8 and recent>=8,event_ids=sorted(events),issuers=sorted(issuers),distinct_events=len(events),distinct_issuers=len(issuers),months_ge2=total,early_months_ge2=early,recent_months_ge2=recent)


def changed_support(orders,cells,name):
 """Only genuinely changed held sets; reorderings do not count."""
 require(name in NAMES[1:]and len(orders[name])>=TRAIN and len(orders[NAMES[0]])>=TRAIN,'First88 orders required')
 by_month={i:[]for i in range(TRAIN)}
 for c in cells:
  if c['signal_index']<TRAIN and c['available']:by_month[c['signal_index']].append(c)
 months=[];events=set();issuers=set()
 for i in range(TRAIN):
  new=set(orders[name][i]['symbols']);old=set(orders[NAMES[0]][i]['symbols'])
  if new==old:continue
  months.append(i)
  for c in by_month[i]:
   if c['symbol']in new.symmetric_difference(old):events.add(c['event_id']);issuers.add(c['symbol'])
 early=sum(i<53 for i in months);recent=len(months)-early
 return dict(passed=len(months)>=3 and early>=1 and recent>=1 and len(events)>=3 and len(issuers)>=2,months=months,early_months=early,recent_months=recent,event_ids=sorted(events),issuers=sorted(issuers),distinct_events=len(events),distinct_issuers=len(issuers),attribution_rule='available source on a stock entering or leaving held set; descriptive, not causal')


def metric_guard(candidate,incumbent,minimum_gain):
 fields=(('training','cagr'),('training','sharpe'),('training','max_drawdown'),('early','cagr'),('recent','cagr'))
 vals=[r.get(a,{}).get(b)for r in (candidate,incumbent)for a,b in fields]
 if not all(type(x)in(int,float)and math.isfinite(x)for x in vals):return False
 x,b=vals[:5],vals[5:]
 return x[0]>=b[0]+minimum_gain-1e-12 and all(v>=w-t for v,w,t in zip(x[1:],b[1:],(.05,.03,.025,.025)))


def select_first88(rows,opened,support,changed):
 require(all([r['name']for r in group]==list(NAMES)for group in(rows,opened)),'Fixed three-arm order')
 require(type(support.get('passed'))is bool and set(changed)==set(NAMES[1:]),'Pre-outcome source and changed-support receipts required')
 eligible=[];guards={}
 for i,name in enumerate(NAMES[1:],1):
  g=dict(source_support=support['passed'],changed_holdings=changed[name].get('passed')is True,close=metric_guard(rows[i],rows[0],.005),cost_open=metric_guard(opened[i],opened[0],0.))
  guards[name]=g
  if all(g.values()):eligible.append(i)
 winner=0
 for i in eligible:
  if winner==0 or rows[i]['training']['cagr']>rows[winner]['training']['cagr']+1e-12:winner=i
 return dict(primary=NAMES[winner],admitted=winner!=0,selection_count=1,guards=guards,later34_used_for_selection=False)


class Backend(base.Local67Backend):
 """Final host can reuse original baseline/pre/selector and unchanged engines."""
 price_parity=previous.Backend.price_parity
 # No inherited training: old helpers have different NAMES/selection policies.
 def training(self,orders,meta,px,op):
  rp=self.local.rp;blocks,_,bounds,valid=rp.paths(px,meta,TRAIN);rows=[];opened=[]
  for name in NAMES:
   indices=[[meta['symbols'][i].index(s)for s in row['symbols']]for i,row in enumerate(orders[name][:TRAIN])]
   rr=rp.fast_returns(indices,blocks,valid);rows.append(dict(name=name,**self.local.old.session.measure(rr,meta,bounds)))
   _,eq,_,_=self.local.ex.e.engine.replay(px,op,orders[name][:TRAIN],cost=.0005,execution='open',start=pd.Timestamp(meta['dates'][0]),end=pd.Timestamp(meta['ends'][87]))
   item={'name':name}
   for key,lo,hi in(('training',0,87),('early',0,52),('recent',53,87)):
    start,end=meta['dates'][lo],meta['ends'][hi];part=eq.loc[start:end];require(len(part)>=2,'Training interval too short');item[key]=rp.stats_rr(part.to_numpy()[1:]/part.to_numpy()[:-1]-1,start,end)
   opened.append(item)
  return rows,opened
 def diagnostics(self,orders,meta,px,op,writer):
  results={}
  for name in NAMES:
   results[name]={}
   for mode,cost,key in(('close',0.,'same_close'),('open',0.,'next_open_zero'),('open',.0005,'next_open_cost')):
    met,eq,trades,holds=self.local.ex.e.engine.replay(px,op,orders[name],cost=cost,execution=mode,start=pd.Timestamp(meta['dates'][0]),end=pd.Timestamp(meta['ends'][-1]))
    for year in met['annual_returns']:
     if year['year']in(2016,2026):year['partial']=True
    results[name][key]=dict(metrics=met,slices=self.local.arch.old.prior.old.previous.cuts(eq,meta))
    writer(name+'_'+key+'_DAILY.json',[dict(date=str(t.date()),equity=float(v))for t,v in eq.items()]);writer(name+'_'+key+'_TRADES.json',trades);writer(name+'_'+key+'_HOLDINGS.json',holds)
  return results


class Runner:
 def __init__(self,*args,**kwargs):
  raise RuntimeError('Draft159 cannot execute real phases: field source gate, parameter/code hashes, registered PLAN and independent lifecycle are not finalized.')


def source_files(root):
 root=Path(root);paths=[Path(__file__),root/'work/test_basic_shares159_runner_draft.py',Path(kernel.__file__),root/'work/test_after158_basic_shares_kernel.py',Path(previous.__file__)]
 return dict(previous.source_files(root),**{str(p.resolve()):base.sha(p)for p in paths})


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--contracts',action='store_true');a=p.parse_args()
 if a.contracts:print(json.dumps({'measurement':measurement_contract(),'selection':selection_contract(),'integration':integration_contract()},indent=2))
 else:p.print_help()
if __name__=='__main__':main()
