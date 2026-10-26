"""Explicit registration and phase boundaries for the fixed basic-shares pilot.

No real experiment may start until source review and synthetic validation are
bound into PLAN. This file is a draft until that registration is published.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import stat
import os
import tempfile
import fcntl
from contextlib import contextmanager
import numpy as np
import pandas as pd
import basic_shares159_runner_draft as r
import monthly_forward_guidance132_runner as base

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/monthly_basic_shares159'
STATE=ROOT/'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
SOURCE=ROOT/'outputs/after158_basic_shares_admission_v2'
REVIEW=ROOT/'work/after158_basic_shares_admission_independent.json'
SELF=Path(__file__).resolve()
require=base.require
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):
 d=hashlib.sha256()
 with Path(p).open('rb')as f:
  for part in iter(lambda:f.read(1<<20),b''):d.update(part)
 return d.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x')as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def write_equal(p,v):
 if Path(p).exists():require(read(p)==v,'Conflicting preexisting artifact: '+str(p))
 else:write(p,v)
@contextmanager
def state_lock():
 with (STATE.parent/'.lifecycle.lock').open('a')as f:
  fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:yield
  finally:fcntl.flock(f,fcntl.LOCK_UN)
def state_commit(expected,value):
 require(sha(STATE)==expected,'Concurrent state change')
 backup=STATE.parent/('PRE159_'+expected+'.json')
 if not backup.exists():
  with backup.open('xb')as f:f.write(STATE.read_bytes());f.flush();os.fsync(f.fileno())
 require(sha(backup)==expected,'State backup differs')
 with tempfile.NamedTemporaryFile(dir=STATE.parent,delete=False)as f:
  tmp=Path(f.name);f.write((json.dumps(value,indent=2,sort_keys=True)+'\n').encode());f.flush();os.fsync(f.fileno())
 try:
  require(sha(STATE)==expected,'Concurrent state commit');os.replace(tmp,STATE)
 finally:tmp.unlink(missing_ok=True)
def resolve(p):
 p=Path(p);return (p if p.is_absolute()else ROOT/p).resolve()
def verify(pins):
 for p,h in pins.items():require(sha(resolve(p))==h,'Changed dependency: '+str(p))
def combined(*maps):
 out={}
 for d in maps:
  for p,h in d.items():
   q=str(resolve(p));require(q not in out or out[q]==h,'Conflicting input '+q);out[q]=h
 return out
def runtime():return dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__)
SOURCE_CHECKS=('all_input_output_and_implementation_hashes_verified','independent_earliest_cached_accession_selection',
 'all_88_original_counts_and_native_intervals_verified','all_44_original_periods_and_holds_verified','max_of_two_second_session_close_clocks',
 'all_5978_grid_rows_and_original_nested_records_preserved','latest_known_before_validity_no_fallback','original_t_minus_42_and_identity_constraints',
 'six_original_rejections_not_lifted','initial_partial_unchanged','all_five_coverage_splits_recomputed','strict_admission_flags_remain_false',
 'all_34_general_units_have_independent_original_table_evidence','v1_failure_preserved_and_only_v2_unit_guard_changed')
PREPARE_CHECKS=('original27_arrays_byte_identical','original5978_rows_preserved','two_clock_tminus42_latest_invalid_no_fallback',
 'original_eligible_and_distinct_issuer_before_transform','exact_fraction_raw_and_rank','centered_five_percent_magnitude_and_rank_hands',
 'once_point01_delta_then_original_episode','all123_feature_and_score_prefixes','initial_and_zero_delta_exact_incumbent',
 'baseline_full_orders_unchanged','source_support_independently_verified','raw_price_parity_independently_verified','sourcegate_v2_review_bound')
def source_review_binding(manifest,review):
 require(review.get('passed')is True and all(review.get('checks',{}).get(k)is True for k in SOURCE_CHECKS),'Complete independent source review required')
 require(review.get('source_manifest_sha256')==sha(SOURCE/'MANIFEST.json') and review.get('plan_sha256')==sha(SOURCE/'PLAN.json'),'Independent review must bind current v2 source')
 require(review['summary']==manifest['summary'],'Source review coverage mismatch')
 required=combined(manifest['input_hashes'],manifest['outputs'],{str(SOURCE/'MANIFEST.json'):sha(SOURCE/'MANIFEST.json')})
 bound=combined(review['input_hashes']);require(all(bound.get(k)==v for k,v in required.items()),'Incomplete source review input closure')
def measurement():
 d=r.measurement_contract();d['usable_at']='max(second_XNYS_close_after_release,second_XNYS_close_after_selected_corroborating_filing)'
 d['identity']='exact source symbol/security and distinct issuer in each original 49-name row, under preserved historical identity limitations'
 d['conditional_source_policy']='after158_basic_shares_admission_v2';return d

def validation_paths():
 import basic_shares159_independent as independent
 return sorted(set([SELF,Path(r.__file__).resolve(),Path(r.kernel.__file__).resolve(),
  ROOT/'work/test_basic_shares159_host.py',ROOT/'work/test_basic_shares159_runner_draft.py',
  ROOT/'work/test_after158_basic_shares_kernel.py',*independent.source_files(ROOT)]))

def validate_tests(receipt):
 require(receipt.get('passed')is True and receipt.get('synthetic_only')is True,'Synthetic validation required')
 pins=combined(receipt.get('implementation_hashes',{}))
 require(all(pins.get(str(p))==sha(p)for p in validation_paths()),'Complete current implementation/test validation required')
 verify(pins)

def validate_plan(plan):
 require(plan.get('round')==159 and plan.get('status')=='REGISTERED' and plan.get('arms')==list(r.NAMES),'Wrong registration')
 require(plan.get('runtime')==runtime() and plan.get('measurement')==measurement() and plan.get('selection_contract')==r.selection_contract(),'Changed experiment contract')
 require(plan.get('output_directory')==str(OUT) and plan.get('no_observer_writes')is True,'Fixed output and observer protection required')
 for key in ('actual_features_before_registration','actual_orders_before_registration','actual_returns_before_registration'):
  require(type(plan.get(key))is int and plan[key]==0,'No numerical work before registration')

class Collector:
 def __init__(self,inputs):self.inputs=inputs;self.pins={}
 def bound_path(self,p):
  p=resolve(p);self.pins[str(p)]=sha(p);return p

def register():
 if (OUT/'PLAN.json').exists():return enqueue(read(OUT/'PLAN.json'))
 require(not OUT.exists() or set(p.name for p in OUT.iterdir())<= {'STATE_BEFORE.json'},'New registration directory required')
 status=read(STATE);state_sha=sha(STATE)
 require(len(status['completed'])==158 and sum(x['trials']for x in status['completed'])==989 and not status['registered_queue'],'Expected completed158,989trials,empty queue')
 manifest=read(SOURCE/'MANIFEST.json');review=read(REVIEW)
 source_review_binding(manifest,review)
 # The review schema and exact input bindings are additionally preregistered.
 paths={k:ROOT/'outputs/monthly_local_linear70'/v for k,v in dict(meta='META.json',policy='INCUMBENT_POLICY.json',original_orders='local_linear_ORDERS.json').items()}
 paths.update(prices=ROOT/'outputs/monthly_2016_extension/PRICES.npz',protected_manifest=ROOT/'work/trade_reconcile_20261015_close/NEXT_EXPECTED_PROTECTED.json',design=ROOT/r.DESIGN,source_manifest=SOURCE/'MANIFEST.json',source_events=SOURCE/'EVENTS.json',source_rows=SOURCE/'ROWS.json',source_review=REVIEW,calendar=ROOT/'outputs/monthly_paired_target143/CALENDAR.json')
 collector=Collector(paths);backend=r.Backend(ROOT,collector) # metadata/closure only, no .load()
 source_files=r.source_files(ROOT)
 nested=[read(p)for p in collector.pins if p.endswith('/INPUT_HASHES.json')]
 import basic_shares159_independent as independent
 independent_paths=independent.source_files(ROOT)
 extra=[SELF,*independent_paths,ROOT/'work/test_basic_shares159_host.py',ROOT/'work/test_basic_shares159_runner_draft.py',ROOT/'work/test_after158_basic_shares_kernel.py']
 validation=ROOT/'work/basic_shares159_registration_tests.json';v=read(validation)
 validate_tests(v);extra.append(validation)
 registration_review=ROOT/'work/basic_shares159_host_registration_review_v2.json';rv=read(registration_review)
 require(rv.get('passed')is True and rv.get('host_sha256')==sha(SELF),'Current independent registration review required')
 extra.append(registration_review)
 price_plan=ROOT/'outputs/monthly_paired_target143/PLAN.json';pp=read(price_plan)
 inputs={k:dict(path=str(p),sha256=sha(p))for k,p in paths.items()}
 pins=combined(source_files,collector.pins,*nested,manifest['input_hashes'],manifest['outputs'],review['input_hashes'],{str(p):sha(p)for p in extra+[price_plan]},pp['raw_source_hashes'],{str(p):sha(p)for p in paths.values()})
 verify(pins)
 plan=dict(round=159,status='REGISTERED',registered_utc=now(),output_directory=str(OUT),arms=list(r.NAMES),runtime=runtime(),
  measurement=measurement(),selection_contract=r.selection_contract(),inputs=inputs,hashes=pins,raw_source_hashes=pp['raw_source_hashes'],
  registration_precondition=dict(path=str(STATE),sha256=state_sha),source_limitations=manifest['limitations'],
  actual_features_before_registration=0,actual_orders_before_registration=0,actual_returns_before_registration=0,no_observer_writes=True)
 OUT.mkdir(parents=True,exist_ok=True);write_equal(OUT/'STATE_BEFORE.json',status);write(OUT/'PLAN.json',plan)
 return enqueue(plan)

def queue_entry(plan):
 return dict(round=159,directory='monthly_basic_shares159',plan=dict(path=str(OUT/'PLAN.json'),sha256=sha(OUT/'PLAN.json')),registered_utc=plan['registered_utc'])
def enqueue(plan):
 validate_plan(plan)
 verify(plan['hashes']);status=read(STATE);current=sha(STATE);entry=queue_entry(plan);digest=sha(OUT/'PLAN.json')
 if status['registered_queue']==[entry]:return print(json.dumps({'already_registered':True,'plan_sha256':digest}))
 require(current==plan['registration_precondition']['sha256'] and not status['registered_queue'],'Stale registration state')
 require(read(OUT/'STATE_BEFORE.json')==status,'Exact pre-registration state required')
 status.update(current='monthly_basic_shares159',current_status='registered_before_actual_preparation',registered_queue=[entry],updated_utc=now(),last_goal_turn_classification='progress')
 state_commit(current,status)
 print(json.dumps({'plan_sha256':digest,'registered_round':159}))

class Host:
 def __init__(self,expected):
  require(sha(OUT/'PLAN.json')==expected,'Exact PLAN SHA required');self.plan=read(OUT/'PLAN.json');self.plan_sha=expected
  p=self.plan;validate_plan(p)
  self.pins=p['hashes'];verify(self.pins);self.inputs={k:self.bound_path(v['path'])for k,v in p['inputs'].items()}
  require(all(sha(self.inputs[k])==v['sha256']for k,v in p['inputs'].items()),'Input binding')
  self.out=OUT
  status=read(STATE)
  registered=status['registered_queue']==[queue_entry(p)]
  completed=not status['registered_queue'] and len(status['completed'])==159 and status['completed'][-1]['directory']=='monthly_basic_shares159' and (OUT/'COMPLETION.json').exists() and read(OUT/'COMPLETION.json')['plan_sha256']==expected
  require(registered or completed,'PLAN not registered in current central state')
 def bound_path(self,p):
  p=resolve(p);require(self.pins.get(str(p))==sha(p),'Unbound '+str(p));return p
 def write(self,name,obj):self.refresh();write(OUT/name,obj)
 def refresh(self):require(sha(OUT/'PLAN.json')==self.plan_sha,'PLAN changed');verify(self.pins)
 def source(self):
  meta=read(self.inputs['meta']);base.validate_meta(meta)
  events=read(self.inputs['source_events'])['events'];rows=read(self.inputs['source_rows']);require(len(events)==44 and len(rows)==5978,'Full source axes')
  require(len({e['event_id']for e in events})==44,'Distinct original source events required')
  cutoffs=[]
  for i,names in enumerate(meta['symbols']):
   issuers=[]
   for j,name in enumerate(names):
    row=rows[49*i+j];require((row['signal_index'],row['candidate_index'],row['symbol'])==(i,j,name),'Source row identity mismatch')
    old=row['original_availability_row'];require(old['decision_date']==meta['dates'][i],'Source decision date')
    if j==0:cutoffs.append(row['cutoff'])
    require(row['cutoff']==cutoffs[i],'Month cutoff mismatch')
    if row['conditional_source_ready']:
     ee=[e for e in events if e['event_id']==row['selected_event_id']];require(len(ee)==1,'Source event reference')
     e=ee[0];ident=e['identity'];day=row['cutoff'][:10]
     require(ident['security_id']==name and ident['identity_valid_from']<=day<=ident['identity_valid_through'],'Identity/report-use window')
     issuers.append(ident['issuer_id']);require(e['symbol']==name and name not in meta['etfs'],'Issuer or ETF mismatch')
   require(len(issuers)==len(set(issuers)),'Duplicate source issuer')
  return meta,events,rows,cutoffs
 def prepare(self):
  require(not (OUT/'PREPARE_STARTED.json').exists(),'Prepare already started; preserve failure')
  self.write('PREPARE_STARTED.json',dict(plan_sha256=self.plan_sha,utc=now()))
  meta,events,rows,cutoffs=self.source();backend=r.Backend(ROOT,self);m,a,px,op=backend.load()
  require(m==meta and len(a)==27,'Original metadata /27arrays changed');before={k:v.tobytes()for k,v in a.items()}
  parity=backend.price_parity(meta,px,op,self.plan['raw_source_hashes'])
  built=r.build_features(meta,events,a['eligible'],cutoffs,source_rows=rows);f=built['arrays'];audit=built['audit']
  for k,row in enumerate(rows):
   i,j=divmod(k,49);expected=bool(row['conditional_source_ready'] and a['eligible'][i,j])
   require(bool(f['common_available'][i,j])==expected,'Feature differs from independent source gate')
  support=r.source_support(audit['cells'])
  if not support['passed']:
   self.write('SOURCE_SUPPORT_FAILURE.json',dict(plan_sha256=self.plan_sha,source=support,returns_computed=False))
   raise ValueError('Preregistered source support failed; no orders or returns computed')
  a.update(f)
  for n in range(123):
   part=r.build_features(meta,events,a['eligible'],cutoffs,n,source_rows=rows)
   require(all(part['arrays'][k].tobytes()==f[k][:n].tobytes()for k in r.EXTRA),'Feature prefix changed')
  orders,scores,panel_audit=r.panels(a,meta,backend,read(self.inputs['policy']))
  require(orders[r.NAMES[0]]==read(self.inputs['original_orders']),'Original incumbent orders changed')
  require(all(a[k].tobytes()==v for k,v in before.items()),'Original arrays changed')
  changed={name:r.changed_support(orders,audit['cells'],name)for name in r.NAMES[1:]}
  self.write('SUPPORT_AUDIT.json',dict(source=support,changed=changed));self.write('FEATURE_AUDIT.json',audit)
  self.write('PRETRAIN_ORDERS.json',orders);self.write('META.json',meta);self.write('PRICE_PARITY.json',parity)
  self.refresh()
  with (OUT/'DATA.npz').open('xb')as out:np.savez_compressed(out,**a)
  with (OUT/'SCORES.npz').open('xb')as out:np.savez_compressed(out,**scores)
  self.refresh();names=['SUPPORT_AUDIT.json','FEATURE_AUDIT.json','PRETRAIN_ORDERS.json','META.json','PRICE_PARITY.json','DATA.npz','SCORES.npz']
  self.write('CACHE.json',{n:sha(OUT/n)for n in names});print(json.dumps({'prepared':True,'source_support':support,'changed_support':changed}))
 def cached(self):
  self.refresh();cache=read(OUT/'CACHE.json');verify({str(OUT/k):h for k,h in cache.items()})
  review=read(OUT/'INDEPENDENT_PREPARE.json');require(review.get('passed')is True and review.get('plan_sha256')==self.plan_sha and review.get('cache_sha256')==sha(OUT/'CACHE.json'),'Current independent preparation required')
  require(all(review.get('checks',{}).get(k)is True for k in PREPARE_CHECKS),'Incomplete independent preparation checks')
  self.receipt(review)
  backend=r.Backend(ROOT,self);m,a,px,op=backend.load();return backend,m,px,op
 def receipt(self,review):
  import basic_shares159_independent as independent
  expected={str(resolve(p)):sha(p)for p in independent.source_files(ROOT)}
  require(combined(review['source_hashes'])==expected,'Independent verifier source closure differs')
  require(all(self.pins.get(p)==h for p,h in expected.items()),'Verifier not bound to original PLAN')
  verify(review['source_hashes']);verify(review['artifact_hashes'])
  require(review.get('producer_modules_imported')is False,'Reviewer imported producer')
 def select(self):
  require(not (OUT/'SELECT_STARTED.json').exists(),'Selection already started')
  backend,m,px,op=self.cached();self.write('SELECT_STARTED.json',dict(plan_sha256=self.plan_sha,utc=now()))
  orders=read(OUT/'PRETRAIN_ORDERS.json');support=read(OUT/'SUPPORT_AUDIT.json')
  require(support['source']['passed']is True,'Source support failed; no outcome calculation allowed')
  rows,opened=backend.training(orders,m,px,op)
  selection=r.select_first88(rows,opened,support['source'],support['changed'])
  self.write('TRAIN_TRIALS.json',rows);self.write('TRAIN_OPEN_TRIALS.json',opened)
  self.write('FROZEN.json',dict(selection=selection,plan_sha256=self.plan_sha,selection_count=1,later34_diagnostic_only=True,
   hashes={x:sha(OUT/x)for x in ('PLAN.json','CACHE.json','PRETRAIN_ORDERS.json','INDEPENDENT_PREPARE.json','TRAIN_TRIALS.json','TRAIN_OPEN_TRIALS.json','SUPPORT_AUDIT.json')}))
  print(json.dumps(selection))
 def diagnose(self):
  require(not (OUT/'DIAGNOSE_STARTED.json').exists(),'Diagnostics already started')
  backend,m,px,op=self.cached();frozen=read(OUT/'FROZEN.json');verify({str(OUT/k):h for k,h in frozen['hashes'].items()})
  self.write('DIAGNOSE_STARTED.json',dict(plan_sha256=self.plan_sha,frozen_sha256=sha(OUT/'FROZEN.json'),utc=now()))
  results=backend.diagnostics(read(OUT/'PRETRAIN_ORDERS.json'),m,px,op,self.write);self.refresh()
  self.write('RESULTS.json',dict(plan_sha256=self.plan_sha,selection=frozen['selection'],arms=results))
  print(json.dumps({name:results[name]['same_close']['metrics']for name in r.NAMES}))
 def close(self):
  self.cached();review=read(OUT/'INDEPENDENT_OUTCOME.json');results=read(OUT/'RESULTS.json');frozen=read(OUT/'FROZEN.json')
  import basic_shares159_independent as independent
  require(review.get('passed')is True and review.get('plan_sha256')==self.plan_sha and review['artifact_hashes'].get(str(OUT/'RESULTS.json'))==sha(OUT/'RESULTS.json'),'Current independent outcome review required')
  require(review.get('paths_checked')==9 and review.get('orders_checked')==366,'All nine paths and366 monthly decisions required')
  require(all(review.get('checks',{}).get(k)is True for k in independent.OUTCOME_CHECKS),'Incomplete outcome checks')
  self.receipt(review)
  require(results['selection']==frozen['selection'],'Frozen selection changed')
  protected=read(self.inputs['protected_manifest']);require(len(protected)==21444,'Full historical protection inventory required')
  for p,h in protected.items():
   q=resolve(p);require(q==Path(p) and q.is_relative_to(ROOT) and stat.S_ISREG(q.stat().st_mode),'Invalid protected file identity')
   require(sha(q)==h,'Protected file changed: '+str(q))
  self.refresh();write_equal(OUT/'PROTECTED_AFTER.json',dict(passed=True,checked=21444,verified=21444,changed=[],missing=[],read_or_identity_errors=[],manifest_sha256=sha(self.inputs['protected_manifest'])))
  selected=results['selection']['primary'];metric=results['arms'][selected]['same_close']['metrics']
  done=dict(directory='monthly_basic_shares159',completed_utc=now(),primary=selected,primary_cagr=metric['cagr'],primary_sharpe=metric['sharpe'],trials=3,
   new_admitted=int(results['selection']['admitted']),new_model_fits=0,independent_execution_paths=9,protected_files=21444,target_reached=metric['cagr']>=.70,
   plan_sha256=self.plan_sha,results_sha256=sha(OUT/'RESULTS.json'),source_conditional_only=True,
   independent_prepare_sha256=sha(OUT/'INDEPENDENT_PREPARE.json'),independent_outcome_sha256=sha(OUT/'INDEPENDENT_OUTCOME.json'),
   frozen_sha256=sha(OUT/'FROZEN.json'),protected_after_sha256=sha(OUT/'PROTECTED_AFTER.json'))
  if (OUT/'COMPLETION.json').exists():done['completed_utc']=read(OUT/'COMPLETION.json')['completed_utc']
  self.refresh();write_equal(OUT/'COMPLETION.json',done)
  status=read(STATE);current=sha(STATE)
  entry={k:done[k]for k in ('directory','completed_utc','primary','primary_cagr','primary_sharpe','trials','new_admitted','new_model_fits','independent_execution_paths','protected_files','target_reached')}
  entry['completion']=dict(path=str(OUT/'COMPLETION.json'),sha256=sha(OUT/'COMPLETION.json'))
  if len(status['completed'])==159:
   require(status['completed'][-1]==entry and not status['registered_queue'],'Completed central state differs')
   return print(json.dumps({'already_completed':True,**done}))
  require(len(status['completed'])==158 and status['registered_queue']==[queue_entry(self.plan)],'Central completion precondition')
  status['completed'].append(entry);status.update(current='monthly_basic_shares159',current_status='completed_independently_verified',registered_queue=[],updated_utc=done['completed_utc'],last_goal_turn_classification='progress',latest_round_receipt=entry['completion'])
  state_commit(current,status)
  print(json.dumps(done))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['register','prepare','select','diagnose','close']);p.add_argument('--plan-sha');args=p.parse_args()
 with state_lock():
  if args.phase=='register':register()
  else:getattr(Host(args.plan_sha),args.phase)()
