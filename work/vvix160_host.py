"""Fixed three-arm VVIX experiment. Registration precedes all real features."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import stat
import tempfile
import sys
import numpy as np
import pandas as pd
import vvix160_kernel as kernel
import vvix160_runner as r
import after159_vvix_validate as source
import basic_shares159_host as common

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/monthly_vvix160'
STATE=ROOT/'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
SELF=Path(__file__).resolve()
require=common.require
sha,read,write,write_equal=common.sha,common.read,common.write,common.write_equal
verify,combined,resolve=common.verify,common.combined,common.resolve
now,runtime=common.now,common.runtime
CACHE=('DATA.npz','SCORES.npz','META.json','PRETRAIN_ORDERS.json','PRICE_PARITY.json',
       'FEATURE_AUDIT.json','STATE_VALUES.json','SUPPORT_AUDIT.json')
PROTECTED_SHA='89e82ac5cbec3b998f1d8aab93f1f8fd7c5e19522321b2dd1da2f88dfff922cb'

def validate_prior_state(status,before):
 require(status['completed']==before['completed']and len(status['completed'])==159 and sum(x['trials']for x in status['completed'])==992,'Original completed history must remain exact')
 require(status.get('best_training_selected_candidate')==before.get('best_training_selected_candidate'),'Prior incumbent metadata changed during experiment')

def state_commit(expected,value):
 require(sha(STATE)==expected,'Concurrent central state change')
 backup=STATE.parent/('PRE160_'+expected+'.json')
 if not backup.exists():
  with backup.open('xb')as f:f.write(STATE.read_bytes());f.flush();os.fsync(f.fileno())
 require(sha(backup)==expected,'Central backup mismatch')
 with tempfile.NamedTemporaryFile(dir=STATE.parent,delete=False)as f:
  temp=Path(f.name);f.write((json.dumps(value,indent=2,sort_keys=True)+'\n').encode());f.flush();os.fsync(f.fileno())
 try:
  require(sha(STATE)==expected,'Concurrent central commit');os.replace(temp,STATE)
 finally:temp.unlink(missing_ok=True)

def paths():
 d={k:ROOT/'outputs/monthly_local_linear70'/v for k,v in
    dict(meta='META.json',policy='INCUMBENT_POLICY.json',original_orders='local_linear_ORDERS.json').items()}
 d.update(prices=ROOT/'outputs/monthly_2016_extension/PRICES.npz',
  vol_feature=ROOT/'outputs/monthly_2016_extension/FEATURES.npz',
  vol_meta=ROOT/'outputs/monthly_2016_extension/INPUTS.json',
  protected_manifest=ROOT/'work/trade_reconcile_20261015_close/NEXT_EXPECTED_PROTECTED.json',
  proposal=ROOT/'work/vvix160_proposal.json',calendar=ROOT/'outputs/monthly_paired_target143/CALENDAR.json',
  source_validation=ROOT/'outputs/after159_vvix_source/VALIDATION.json',
  source_review=ROOT/'work/after159_vvix_admission_review.json',
  source_collection=ROOT/'outputs/after159_vvix_source/COLLECTION.json',
  source_vix=ROOT/'outputs/after159_vvix_source/VIX_History.csv',
  source_vvix=ROOT/'outputs/after159_vvix_source/VVIX_History.csv')
 return d

def source_gate(p):
 require(sha(p['protected_manifest'])==PROTECTED_SHA,'Exact original protected manifest required')
 val,review=read(p['source_validation']),read(p['source_review'])
 require(val.get('passed')is True and val['signals_checked']==122 and val['noninitial_signals']==121,'Full fixed source windows required')
 require(review.get('passed')is True and review.get('strict_pit_vintage')is False and review.get('strategy_admitted')is False and review.get('conditional_research_sources_ready')is True,'Independent conditional source review required')
 require(review.get('validation_sha256')==sha(p['source_validation']),'Current validation review required')
 require(review['summary']=={k:val[k]for k in('signals_checked','noninitial_signals','first_required_date','last_required_date','distinct_required_sessions')},'Independent coverage totals match')
 verify(val['input_hashes']);verify(review['input_hashes'])
 require(sha(p['vol_feature'])=='1d44b505ddcfd2979e30ba4e73ff528cd1d72efc5ca7e745c3f1c51401fbf5b5','Original volatility cache')
 require(sha(p['vol_meta'])==sha(p['meta'])=='3b46968f486e88fa170ac01f120dd5ef74ebd5a56243c6991c5a4fea506076f4','Original axes must be byte identical')
 return val,review

def validation_paths():
 import vvix160_independent as independent
 return sorted(set([SELF,ROOT/'work/vvix160_kernel.py',ROOT/'work/vvix160_runner.py',
  ROOT/'work/test_vvix160_kernel.py',ROOT/'work/test_vvix160_runner.py',
  ROOT/'work/test_vvix160_host.py',*independent.source_files(ROOT)]))

def validate_tests(receipt):
 require(receipt.get('passed')is True and receipt.get('synthetic_only')is True,'Synthetic validation required')
 pins=combined(receipt.get('implementation_hashes',{}))
 require(all(pins.get(str(p))==sha(p)for p in validation_paths()),'All current implementations/tests must be validated')
 verify(pins)

def validate_plan(plan):
 require(plan.get('round')==160 and plan.get('status')=='REGISTERED' and plan.get('arms')==list(r.NAMES),'Fixed round160 registration')
 require(plan.get('runtime')==runtime()and plan.get('measurement')==r.measurement_contract()and plan.get('selection_contract')==r.selection_contract(),'Frozen runtime and numeric contracts')
 require(plan.get('output_directory')==str(OUT)and plan.get('no_observer_writes')is True,'Fixed research boundary')
 require(plan.get('inputs',{}).get('protected_manifest',{}).get('sha256')==PROTECTED_SHA,'Frozen protected manifest identity')
 for key in ('actual_features_before_registration','actual_orders_before_registration','actual_returns_before_registration'):
  require(type(plan.get(key))is int and plan[key]==0,'No real numeric work before registration')

def queue_entry(plan):
 return dict(round=160,directory=OUT.name,plan=dict(path=str(OUT/'PLAN.json'),sha256=sha(OUT/'PLAN.json')),registered_utc=plan['registered_utc'])

def enqueue(plan):
 validate_plan(plan);verify(plan['hashes']);status=read(STATE);current=sha(STATE);entry=queue_entry(plan)
 if status['registered_queue']==[entry]:return print(json.dumps({'already_registered':True,'plan_sha256':sha(OUT/'PLAN.json')}))
 require(current==plan['registration_precondition']['sha256']and not status['registered_queue'],'Exact empty preregistration state')
 require(read(OUT/'STATE_BEFORE.json')==status,'Original central state copy')
 status.update(current=OUT.name,current_status='registered_before_actual_preparation',registered_queue=[entry],updated_utc=now(),last_goal_turn_classification='progress',next='159 completed rounds / 992 trials; fixed three-arm VVIX experiment registered. Preserve dominant skip42 momentum, monthly EW3 and fixed intramonth holdings. Current-vintage source and historical universe limitations remain.',next_step='Prepare fixed VVIX/VIX three-arm experiment, then independent audit before first88 selection.')
 state_commit(current,status);print(json.dumps({'registered_round':160,'plan_sha256':sha(OUT/'PLAN.json')}))

def register():
 if (OUT/'PLAN.json').exists():return enqueue(read(OUT/'PLAN.json'))
 require(not OUT.exists()or set(p.name for p in OUT.iterdir())<={'STATE_BEFORE.json'},'Fresh experiment directory')
 status=read(STATE);state_sha=sha(STATE)
 require(len(status['completed'])==159 and sum(x['trials']for x in status['completed'])==992 and not status['registered_queue'],'Expected159 completed/992trials/empty queue')
 p=paths();val,review=source_gate(p)
 collector=common.Collector(p);r.Backend(ROOT,collector) # metadata/closure only
 nested=[read(q)for q in collector.pins if q.endswith('/INPUT_HASHES.json')]
 validation=ROOT/'work/vvix160_registration_tests.json';validate_tests(read(validation))
 audit=ROOT/'work/vvix160_registration_review.json';ar=read(audit)
 require(ar.get('passed')is True and ar.get('host_sha256')==sha(SELF),'Current independent host review required')
 price_plan=ROOT/'outputs/monthly_paired_target143/PLAN.json';pp=read(price_plan)
 extra=validation_paths()+[validation,audit,price_plan,ROOT/'work/vvix160_proposal.json',Path(common.__file__).resolve()]
 pins=combined(r.source_files(ROOT),collector.pins,*nested,val['input_hashes'],review['input_hashes'],
               {str(q):sha(q)for q in extra},pp['raw_source_hashes'],{str(q):sha(q)for q in p.values()})
 verify(pins)
 plan=dict(round=160,status='REGISTERED',registered_utc=now(),output_directory=str(OUT),arms=list(r.NAMES),runtime=runtime(),
  measurement=r.measurement_contract(),selection_contract=r.selection_contract(),inputs={k:dict(path=str(v),sha256=sha(v))for k,v in p.items()},
  hashes=pins,raw_source_hashes=pp['raw_source_hashes'],registration_precondition=dict(path=str(STATE),sha256=state_sha),
  source_limitations=read(p['proposal'])['source_limitations'],actual_features_before_registration=0,
  actual_orders_before_registration=0,actual_returns_before_registration=0,no_observer_writes=True)
 OUT.mkdir(parents=True,exist_ok=True)
 if not(OUT/'STATE_BEFORE.json').exists():
  with(OUT/'STATE_BEFORE.json').open('xb')as f:f.write(STATE.read_bytes())
 require(sha(OUT/'STATE_BEFORE.json')==state_sha,'Exact-byte preregistration central snapshot')
 write(OUT/'PLAN.json',plan);enqueue(plan)

class Host:
 def __init__(self,expected):
  require(sha(OUT/'PLAN.json')==expected,'Exact PLAN SHA required');self.plan=read(OUT/'PLAN.json');self.plan_sha=expected
  validate_plan(self.plan);self.pins=self.plan['hashes'];verify(self.pins)
  require(sha(OUT/'STATE_BEFORE.json')==self.plan['registration_precondition']['sha256'],'Exact original central state snapshot')
  self.inputs={k:self.bound_path(v['path'])for k,v in self.plan['inputs'].items()}
  require(all(sha(self.inputs[k])==v['sha256']for k,v in self.plan['inputs'].items()),'Named input bindings')
  require(self.inputs==paths(),'Fixed named input paths');source_gate(self.inputs)
  status=read(STATE);registered=status['registered_queue']==[queue_entry(self.plan)]
  completed=not status['registered_queue']and len(status['completed'])==160 and status['completed'][-1]['directory']==OUT.name and (OUT/'COMPLETION.json').exists()and read(OUT/'COMPLETION.json')['plan_sha256']==expected
  require(registered or completed,'Current central registration')
  if registered:validate_prior_state(status,read(OUT/'STATE_BEFORE.json'))
 def bound_path(self,p):
  p=resolve(p);require(self.pins.get(str(p))==sha(p),'Unbound dependency '+str(p));return p
 def refresh(self):require(sha(OUT/'PLAN.json')==self.plan_sha,'PLAN changed');verify(self.pins)
 def write(self,name,value):self.refresh();write(OUT/name,value)
 def prepare(self):
  require(not(OUT/'PREPARE_STARTED.json').exists(),'Preparation already started; preserve any partial output')
  self.write('PREPARE_STARTED.json',dict(plan_sha256=self.plan_sha,utc=now()))
  backend=r.Backend(ROOT,self);meta,a,px,op=backend.load();require(len(a)==27,'Original27 arrays')
  before={k:v.tobytes()for k,v in a.items()}
  parity=backend.price_parity(meta,px,op,self.plan['raw_source_hashes'])
  require(meta==read(self.inputs['vol_meta']),'Exact original volatility axes')
  with np.load(self.inputs['vol_feature'],allow_pickle=False)as f:vol=f['vol126'].copy()
  require(vol.shape==(122,49)and vol.dtype==np.float64,'Original vol126 shape/dtype')
  calendar=read(self.inputs['calendar'])['sessions']
  required=set()
  for day in meta['dates'][1:]:
   t=calendar.index(day);required.update(calendar[max(0,t-252):t])
  values={}
  for name,key in (('VIX','source_vix'),('VVIX','source_vvix')):
   parsed=source.parse_csv(name,self.inputs[key].read_bytes())
   require(parsed['schema_and_chronology_passed'],'Original source row schema')
   values[name]={v['date']:float(v['cells'][4 if name=='VIX'else 1])for v in parsed['rows']if v['date']in required}
  built=kernel.build_states(meta['dates'],calendar,values)
  deltas=kernel.build_deltas(built,vol)
  require(set(deltas)==set(r.EXTRA),'Two score delta arrays')
  for n in range(123):
   prefix=kernel.build_states(meta['dates'],calendar,values,limit=n)
   part=kernel.build_deltas(prefix,vol,limit=n)
   require(all(part[k].tobytes()==deltas[k][:n].tobytes()for k in r.EXTRA),'Exact state/delta prefixes')
  support=r.source_support(built['audit']['monthly'],meta);require(support['passed']is True,'Complete paired source support')
  a.update(deltas);orders,scores,panel_audit=r.panels(a,meta,backend,read(self.inputs['policy']))
  require(orders[r.NAMES[0]]==read(self.inputs['original_orders']),'Full original incumbent orders')
  require(all(a[k].tobytes()==v for k,v in before.items()),'Original arrays changed')
  changed={name:r.changed_support(orders,name)for name in r.NAMES[1:]}
  self.write('SUPPORT_AUDIT.json',dict(source=support,changed=changed))
  self.write('FEATURE_AUDIT.json',built['audit'])
  self.write('STATE_VALUES.json',dict(states={k:v.tolist()for k,v in built['states'].items()},common_available=built['common_available'].tolist()))
  self.write('PRETRAIN_ORDERS.json',orders);self.write('META.json',meta);self.write('PRICE_PARITY.json',parity)
  self.refresh()
  with(OUT/'DATA.npz').open('xb')as f:np.savez_compressed(f,**a)
  with(OUT/'SCORES.npz').open('xb')as f:np.savez_compressed(f,**scores)
  self.write('CACHE.json',{name:sha(OUT/name)for name in CACHE})
  print(json.dumps(dict(prepared=True,source_support=support,changed_support=changed)))
 def receipt(self,review):
  import vvix160_independent as independent
  expected={str(resolve(p)):sha(p)for p in independent.source_files(ROOT)}
  require(combined(review['source_hashes'])==expected,'Complete independent source closure')
  require(all(self.pins.get(p)==h for p,h in expected.items()),'Independent sources must be preregistered')
  verify(review['source_hashes']);verify(review['artifact_hashes'])
  require(review.get('producer_modules_imported')is False,'Verifier imported producer')
 def cached(self):
  import vvix160_independent as independent
  self.refresh();cache=read(OUT/'CACHE.json');require(set(cache)==set(CACHE),'Exact prepared cache')
  verify({str(OUT/k):h for k,h in cache.items()});review=read(OUT/'INDEPENDENT_PREPARE.json')
  require(review.get('passed')is True and review.get('plan_sha256')==self.plan_sha and review.get('cache_sha256')==sha(OUT/'CACHE.json'),'Current independent prepare receipt')
  require(all(review.get('checks',{}).get(k)is True for k in independent.PREPARE_CHECKS),'All independent preparation checks')
  self.receipt(review);backend=r.Backend(ROOT,self);m,a,px,op=backend.load();return backend,m,px,op
 def select(self):
  require(not(OUT/'SELECT_STARTED.json').exists(),'Selection already started')
  backend,m,px,op=self.cached();self.write('SELECT_STARTED.json',dict(plan_sha256=self.plan_sha,utc=now()))
  orders=read(OUT/'PRETRAIN_ORDERS.json');support=read(OUT/'SUPPORT_AUDIT.json')
  rows,opened=backend.training(orders,m,px,op);selection=r.select_first88(rows,opened,support['source'],support['changed'])
  self.write('TRAIN_TRIALS.json',rows);self.write('TRAIN_OPEN_TRIALS.json',opened)
  self.write('FROZEN.json',dict(selection=selection,plan_sha256=self.plan_sha,selection_count=1,later34_diagnostic_only=True,
   hashes={x:sha(OUT/x)for x in('PLAN.json','CACHE.json','PRETRAIN_ORDERS.json','INDEPENDENT_PREPARE.json','TRAIN_TRIALS.json','TRAIN_OPEN_TRIALS.json','SUPPORT_AUDIT.json')}))
  print(json.dumps(selection))
 def diagnose(self):
  require(not(OUT/'DIAGNOSE_STARTED.json').exists(),'Diagnostics already started')
  backend,m,px,op=self.cached();frozen=read(OUT/'FROZEN.json');verify({str(OUT/k):h for k,h in frozen['hashes'].items()})
  self.write('DIAGNOSE_STARTED.json',dict(plan_sha256=self.plan_sha,frozen_sha256=sha(OUT/'FROZEN.json'),utc=now()))
  results=backend.diagnostics(read(OUT/'PRETRAIN_ORDERS.json'),m,px,op,self.write)
  self.write('RESULTS.json',dict(plan_sha256=self.plan_sha,selection=frozen['selection'],arms=results))
  print(json.dumps({name:results[name]['same_close']['metrics']for name in r.NAMES}))
 def close(self):
  import vvix160_independent as independent
  self.cached();review=read(OUT/'INDEPENDENT_OUTCOME.json');results=read(OUT/'RESULTS.json');frozen=read(OUT/'FROZEN.json')
  require(review.get('passed')is True and review.get('plan_sha256')==self.plan_sha and review['artifact_hashes'].get(str(OUT/'RESULTS.json'))==sha(OUT/'RESULTS.json'),'Current independent outcome')
  require(review.get('paths_checked')==9 and review.get('orders_checked')==366,'All9paths/366decisions')
  require(all(review.get('checks',{}).get(k)is True for k in independent.OUTCOME_CHECKS),'Complete outcome checks');self.receipt(review)
  require(results['selection']==frozen['selection'],'Frozen selection unchanged')
  protected=read(self.inputs['protected_manifest']);require(len(protected)==21444,'Full archive manifest')
  for p,h in protected.items():
   q=resolve(p);require(q==Path(p)and q.is_relative_to(ROOT)and stat.S_ISREG(q.stat().st_mode),'Protected identity')
   require(sha(q)==h,'Changed protected file '+str(q))
  self.refresh();write_equal(OUT/'PROTECTED_AFTER.json',dict(passed=True,checked=21444,verified=21444,changed=[],missing=[],read_or_identity_errors=[],manifest_sha256=sha(self.inputs['protected_manifest'])))
  selected=results['selection']['primary'];metric=results['arms'][selected]['same_close']['metrics']
  done=dict(directory=OUT.name,completed_utc=now(),primary=selected,primary_cagr=metric['cagr'],primary_sharpe=metric['sharpe'],trials=3,new_admitted=int(results['selection']['admitted']),new_model_fits=0,independent_execution_paths=9,protected_files=21444,target_reached=metric['cagr']>=.70,plan_sha256=self.plan_sha,results_sha256=sha(OUT/'RESULTS.json'),source_conditional_only=True,independent_prepare_sha256=sha(OUT/'INDEPENDENT_PREPARE.json'),independent_outcome_sha256=sha(OUT/'INDEPENDENT_OUTCOME.json'),frozen_sha256=sha(OUT/'FROZEN.json'),protected_after_sha256=sha(OUT/'PROTECTED_AFTER.json'))
  if(OUT/'COMPLETION.json').exists():done['completed_utc']=read(OUT/'COMPLETION.json')['completed_utc']
  write_equal(OUT/'COMPLETION.json',done);status=read(STATE);current=sha(STATE)
  keys=('directory','completed_utc','primary','primary_cagr','primary_sharpe','trials','new_admitted','new_model_fits','independent_execution_paths','protected_files','target_reached')
  entry={k:done[k]for k in keys};entry['completion']=dict(path=str(OUT/'COMPLETION.json'),sha256=sha(OUT/'COMPLETION.json'))
  if len(status['completed'])==160:
   require(status['completed'][-1]==entry and not status['registered_queue'],'Completed central state differs');return print(json.dumps(done))
  require(len(status['completed'])==159 and status['registered_queue']==[queue_entry(self.plan)],'Exact prior central state')
  validate_prior_state(status,read(OUT/'STATE_BEFORE.json'))
  status['completed'].append(entry);status.update(current=OUT.name,current_status='completed_independently_verified',registered_queue=[],updated_utc=done['completed_utc'],last_goal_turn_classification='progress',latest_round_receipt=entry['completion'],next='160 completed rounds / 995 trials. Review the independently verified fixed VVIX/VIX contrast and source limitations before declaring progress toward70%. Frozen observation accounts remain unchanged.',next_step='Evaluate the completed three-arm evidence without retuning its window, sign or dose after observing outcomes.')
  state_commit(current,status);print(json.dumps(done))

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['register','prepare','select','diagnose','close']);parser.add_argument('--plan-sha');args=parser.parse_args()
 with common.state_lock():
  if args.phase=='register':register()
  else:getattr(Host(args.plan_sha),args.phase)()
