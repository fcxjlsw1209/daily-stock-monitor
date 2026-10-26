"""Fixed volume-conditioned continuation experiment with matched return-persistence control. Registration precedes all real features."""
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
import followthrough166_kernel as kernel
import followthrough166_runner as r
import basic_shares159_host as common

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/monthly_followthrough166'
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
 require(status['completed']==before['completed']and len(status['completed'])==165 and sum(x['trials']for x in status['completed'])==1010,'Original completed history must remain exact')
 require(status.get('best_training_selected_candidate')==before.get('best_training_selected_candidate'),'Prior incumbent metadata changed during experiment')

def validate_completed_state(status,before,expected_best=None):
 require(len(status['completed'])==166 and sum(x['trials']for x in status['completed'])==1013,
         'Completed round166 must have166rounds/1013trials')
 require(status.get('best_training_selected_candidate') == (before.get('best_training_selected_candidate') if expected_best is None else expected_best), 'Completed incumbent metadata must match the verified completion')
 prior=dict(status,completed=status['completed'][:-1],best_training_selected_candidate=before.get('best_training_selected_candidate'))
 validate_prior_state(prior,before)

def best_after(before,results):
 """Promote only a first88-admitted candidate that improves the full-period reference."""
 old=before.get('best_training_selected_candidate')
 selected=results['selection']['primary']
 if not results['selection']['admitted']:
  return old
 require(selected==r.NAMES[1],'Only the preregistered volume-followthrough arm can be promoted')
 value=results['arms'][selected]
 metric=value['same_close']['metrics']
 if old is not None and metric['cagr']<=old['cagr']:
  return old
 return dict(directory=OUT.name,spec=dict(base='incumbent_local67',round=166,arm=selected),
  cagr=metric['cagr'],sharpe=metric['sharpe'],max_drawdown=metric['max_drawdown'],
  next_open_5bp_cagr=value['next_open_cost']['metrics']['cagr'],
  selected_on_first88=True,observer_promotion=False,
  limitations='Reused development sample; estimated historical pool; later34 diagnostic only.')

def state_commit(expected,value):
 require(sha(STATE)==expected,'Concurrent central state change')
 backup=STATE.parent/('PRE166_'+expected+'.json')
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
  protected_manifest=ROOT/'work/trade_reconcile_20261015_close/NEXT_EXPECTED_PROTECTED.json',
  proposal=ROOT/'work/followthrough166_proposal.json',calendar=ROOT/'outputs/monthly_paired_target143/CALENDAR.json',
  training_meta=ROOT/'outputs/monthly_architecture_restart/META.json',
  source_review=ROOT/'work/followthrough166_design_review.json',kernel_source=ROOT/'work/followthrough166_kernel.py')
 return d

def source_gate(p):
 """Metadata admission only; actual volume-conditioned continuation coverage is measured in prepare."""
 require(sha(p['protected_manifest'])==PROTECTED_SHA,'Exact original protected manifest required')
 proposal,review=read(p['proposal']),read(p['source_review'])
 require(proposal.get('round')==166 and proposal.get('arms')==list(r.NAMES),'Fixed volume-conditioned continuation proposal required')
 require(review.get('passed')is True and review.get('strict_pit_vintage')is False and
         review.get('strict_pit_membership')is False and review.get('strategy_admitted')is False and
         review.get('conditional_research_sources_ready')is True,'Independent conditional design review required')
 require(review.get('proposal_sha256')==sha(p['proposal']),'Current proposal review required')
 for item in (proposal,review):
  for key in ('actual_features_computed','actual_orders_computed','actual_returns_computed'):
   require(type(item.get(key))is int and item[key]==0,'Design review must precede real numerical work')
 pins=combined(review['input_hashes'],review['source_hashes']);verify(pins)
 require(all(pins.get(str(p[k]))==sha(p[k])for k in ('meta','training_meta','prices','calendar','proposal')),
         'Design review must bind original axes, historical membership, prices, calendar and proposal')
 require(sha(p['meta'])=='3b46968f486e88fa170ac01f120dd5ef74ebd5a56243c6991c5a4fea506076f4','Original axes must be byte identical')
 summary=review.get('summary',{})
 require(summary.get('all_overlap_sets_equal')is True and summary.get('noninitial_dates_verified_monthend')is True
         and summary.get('event_months')==122 and summary.get('historical_names_absent_from_price_panel')==[] and summary.get('market_proxy_present')is True,
         'Reviewed historical pool identity, month ends and price-column coverage required')
 return review

def calendar_pins(p):
 calendar=read(p['calendar'])
 provenance=calendar['provenance']
 pins=combined(provenance['source_hashes'])
 overlay=provenance['provenance']
 references={str(resolve(overlay['manifest'])):overlay['manifest_sha256']}
 references.update({str(resolve(x['source_file'])):x['source_sha256']for x in overlay['closures']})
 require(all(pins.get(k)==v for k,v in references.items()),'Calendar closure manifest/evidence must be bound')
 verify(pins)
 return pins

def validation_paths():
 import followthrough166_independent as independent
 return sorted(set([SELF,ROOT/'work/followthrough166_kernel.py',ROOT/'work/followthrough166_runner.py',
  ROOT/'work/test_followthrough166_kernel.py',ROOT/'work/test_followthrough166_runner.py',
  ROOT/'work/test_followthrough166_host.py',ROOT/'work/followthrough166_validate_code.py',*independent.source_files(ROOT)]))

def validate_tests(receipt):
 require(receipt.get('passed')is True and receipt.get('synthetic_only')is True,'Synthetic validation required')
 pins=combined(receipt.get('implementation_hashes',{}))
 require(all(pins.get(str(p))==sha(p)for p in validation_paths()),'All current implementations/tests must be validated')
 verify(pins)

def validate_registration_review(review,validation):
 require(review.get('passed')is True and review.get('host_sha256')==sha(SELF),'Current independent host review required')
 require(review.get('registration_tests_sha256')==sha(validation),'Current synthetic receipt must be independently reviewed')
 pins=combined(review.get('implementation_hashes',{}))
 expected={str(p):sha(p)for p in validation_paths()}
 require(pins==expected,'Complete current implementation/test review required')
 require(pins==combined(read(validation)['implementation_hashes']),'Review and synthetic test implementation bindings must agree')
 verify(pins)

PREPARE_COUNTS=dict(candidate_cells_checked=5978,orders_checked=366,prefixes_checked=123,
                    feature_matrices_checked=2,original_array_count=27,prepared_array_count=29)

def validate_prepare_counts(review):
 require(all(type(review.get(k))is int and review[k]==v for k,v in PREPARE_COUNTS.items()),
         'All5978cells/366orders/123prefixes/2features/27original/29prepared required')

def validate_plan(plan):
 require(plan.get('round')==166 and plan.get('status')=='REGISTERED' and plan.get('arms')==list(r.NAMES),'Fixed round166 registration')
 require(plan.get('runtime')==runtime()and plan.get('measurement')==r.measurement_contract()and plan.get('selection_contract')==r.selection_contract(),'Frozen runtime and numeric contracts')
 require(plan.get('output_directory')==str(OUT)and plan.get('no_observer_writes')is True,'Fixed research boundary')
 require(plan.get('inputs',{}).get('protected_manifest',{}).get('sha256')==PROTECTED_SHA,'Frozen protected manifest identity')
 for key in ('actual_features_before_registration','actual_orders_before_registration','actual_returns_before_registration'):
  require(type(plan.get(key))is int and plan[key]==0,'No real numeric work before registration')

def queue_entry(plan):
 return dict(round=166,directory=OUT.name,plan=dict(path=str(OUT/'PLAN.json'),sha256=sha(OUT/'PLAN.json')),registered_utc=plan['registered_utc'])

def enqueue(plan):
 validate_plan(plan);verify(plan['hashes']);status=read(STATE);current=sha(STATE);entry=queue_entry(plan)
 if status['registered_queue']==[entry]:return print(json.dumps({'already_registered':True,'plan_sha256':sha(OUT/'PLAN.json')}))
 require(current==plan['registration_precondition']['sha256']and not status['registered_queue'],'Exact empty preregistration state')
 require(read(OUT/'STATE_BEFORE.json')==status,'Original central state copy')
 status.update(current=OUT.name,current_status='registered_before_actual_preparation',registered_queue=[entry],updated_utc=now(),last_goal_turn_classification='progress',next='165 completed rounds / 1010 trials; fixed three-arm volume-conditioned continuation experiment registered. Preserve dominant skip42 momentum, monthly EW3 and fixed intramonth holdings. Historical price-vintage and proxy-universe limitations remain.',next_step='Prepare fixed partial-correlation volume-conditioned continuation experiment, then independent audit before first88 selection.')
 state_commit(current,status);print(json.dumps({'registered_round':166,'plan_sha256':sha(OUT/'PLAN.json')}))

def register():
 if (OUT/'PLAN.json').exists():return enqueue(read(OUT/'PLAN.json'))
 require(not OUT.exists()or set(p.name for p in OUT.iterdir())<={'STATE_BEFORE.json'},'Fresh experiment directory')
 status=read(STATE);state_sha=sha(STATE)
 require(len(status['completed'])==165 and sum(x['trials']for x in status['completed'])==1010 and not status['registered_queue'],'Expected165 completed/1010trials/empty queue')
 p=paths();review=source_gate(p)
 collector=common.Collector(p);r.Backend(ROOT,collector) # metadata/closure only
 nested=[read(q)for q in collector.pins if q.endswith('/INPUT_HASHES.json')]
 validation=ROOT/'work/followthrough166_registration_tests.json';validate_tests(read(validation))
 audit=ROOT/'work/followthrough166_registration_review.json';ar=read(audit)
 validate_registration_review(ar,validation)
 price_plan=ROOT/'outputs/monthly_paired_target143/PLAN.json';pp=read(price_plan)
 extra=validation_paths()+[validation,audit,price_plan,ROOT/'work/followthrough166_proposal.json',ROOT/'work/followthrough166_validate_code.py',Path(common.__file__).resolve()]
 pins=combined(r.source_files(ROOT),collector.pins,*nested,review['input_hashes'],review['source_hashes'],calendar_pins(p),
               {str(q):sha(q)for q in extra},pp['raw_source_hashes'],{str(q):sha(q)for q in p.values()})
 verify(pins)
 plan=dict(round=166,status='REGISTERED',registered_utc=now(),output_directory=str(OUT),arms=list(r.NAMES),runtime=runtime(),
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
  require(all(self.pins.get(k)==v for k,v in calendar_pins(self.inputs).items()),'Calendar sources must be preregistered')
  status=read(STATE);registered=status['registered_queue']==[queue_entry(self.plan)]
  completed=not status['registered_queue']and len(status['completed'])==166 and status['completed'][-1]['directory']==OUT.name and (OUT/'COMPLETION.json').exists()and read(OUT/'COMPLETION.json')['plan_sha256']==expected
  require(registered or completed,'Current central registration')
  if registered:validate_prior_state(status,read(OUT/'STATE_BEFORE.json'))
  else:validate_completed_state(status,read(OUT/'STATE_BEFORE.json'),read(OUT/'COMPLETION.json')['best_training_selected_candidate'])
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
  training_meta=read(self.inputs['training_meta'])['training']
  calendar=read(self.inputs['calendar'])['sessions']
  events=list(zip(meta['dates'],meta['symbols']))
  panel={'adj':px}
  for key in ('volume',):
   require(px.index.isin(backend.panel[key].index).all() and px.columns.isin(backend.panel[key].columns).all(),'Missing source axes '+key)
   panel[key]=backend.panel[key].loc[px.index,px.columns].copy()
  deltas,audit=kernel.build_features(panel,training_meta,events,calendar)
  require(set(deltas)==set(r.EXTRA),'Two score delta arrays')
  for n in range(123):
   part,prefix_audit=kernel.build_features(panel,training_meta,events,calendar,limit=n)
   require(all(part[k].tobytes()==deltas[k][:n].tobytes()for k in r.EXTRA),'Exact volume-conditioned continuation/delta prefixes')
   require(prefix_audit['event_history']==audit['event_history'][:n],'Exact source-history prefixes')
  support=r.source_support(audit,meta,deltas);require(support['passed']is True,'Valid common volume-conditioned continuation support')
  a.update(deltas);orders,scores,panel_audit=r.panels(a,meta,backend,read(self.inputs['policy']))
  require(orders[r.NAMES[0]]==read(self.inputs['original_orders']),'Full original incumbent orders')
  require(all(a[k].tobytes()==v for k,v in before.items()),'Original arrays changed')
  changed={name:r.changed_support(orders,name)for name in r.NAMES[1:]}
  self.write('SUPPORT_AUDIT.json',dict(source=support,changed=changed))
  self.write('FEATURE_AUDIT.json',audit)
  self.write('STATE_VALUES.json',{key:[x[key]for x in audit['event_history']]for key in ('raw_followthrough','raw_control','ranks_followthrough','ranks_control','common_support')})
  self.write('PRETRAIN_ORDERS.json',orders);self.write('META.json',meta);self.write('PRICE_PARITY.json',parity)
  self.refresh()
  with(OUT/'DATA.npz').open('xb')as f:np.savez_compressed(f,**a)
  with(OUT/'SCORES.npz').open('xb')as f:np.savez_compressed(f,**scores)
  self.write('CACHE.json',{name:sha(OUT/name)for name in CACHE})
  print(json.dumps(dict(prepared=True,source_support=support,changed_support=changed)))
 def receipt(self,review):
  import followthrough166_independent as independent
  expected={str(resolve(p)):sha(p)for p in independent.source_files(ROOT)}
  require(combined(review['source_hashes'])==expected,'Complete independent source closure')
  require(all(self.pins.get(p)==h for p,h in expected.items()),'Independent sources must be preregistered')
  verify(review['source_hashes']);verify(review['artifact_hashes'])
  require(review.get('producer_modules_imported')is False,'Verifier imported producer')
 def cached(self):
  import followthrough166_independent as independent
  self.refresh();cache=read(OUT/'CACHE.json');require(set(cache)==set(CACHE),'Exact prepared cache')
  verify({str(OUT/k):h for k,h in cache.items()});review=read(OUT/'INDEPENDENT_PREPARE.json')
  require(review.get('passed')is True and review.get('plan_sha256')==self.plan_sha and review.get('cache_sha256')==sha(OUT/'CACHE.json'),'Current independent prepare receipt')
  require(all(review.get('checks',{}).get(k)is True for k in independent.PREPARE_CHECKS),'All independent preparation checks')
  validate_prepare_counts(review)
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
  import followthrough166_independent as independent
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
  done['best_training_selected_candidate']=best_after(read(OUT/'STATE_BEFORE.json'),results)
  if(OUT/'COMPLETION.json').exists():done['completed_utc']=read(OUT/'COMPLETION.json')['completed_utc']
  write_equal(OUT/'COMPLETION.json',done);status=read(STATE);current=sha(STATE)
  keys=('directory','completed_utc','primary','primary_cagr','primary_sharpe','trials','new_admitted','new_model_fits','independent_execution_paths','protected_files','target_reached')
  entry={k:done[k]for k in keys};entry['completion']=dict(path=str(OUT/'COMPLETION.json'),sha256=sha(OUT/'COMPLETION.json'))
  if len(status['completed'])==166:
   validate_completed_state(status,read(OUT/'STATE_BEFORE.json'),done['best_training_selected_candidate'])
   require(status['completed'][-1]==entry and not status['registered_queue'],'Completed central state differs');return print(json.dumps(done))
  require(len(status['completed'])==165 and status['registered_queue']==[queue_entry(self.plan)],'Exact prior central state')
  validate_prior_state(status,read(OUT/'STATE_BEFORE.json'))
  status['completed'].append(entry);status.update(best_training_selected_candidate=done['best_training_selected_candidate'],current=OUT.name,current_status='completed_independently_verified',registered_queue=[],updated_utc=done['completed_utc'],last_goal_turn_classification='progress',latest_round_receipt=entry['completion'],next='166 completed rounds / 1013 trials. Review the independently verified fixed volume-conditioned continuation contrast and source limitations before declaring progress toward70%. Frozen observation accounts remain unchanged.',next_step='Evaluate the completed three-arm evidence without retuning its window, sign or dose after observing outcomes.')
  state_commit(current,status);print(json.dumps(done))

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['register','prepare','select','diagnose','close']);parser.add_argument('--plan-sha');args=parser.parse_args()
 with common.state_lock():
  if args.phase=='register':register()
  else:getattr(Host(args.plan_sha),args.phase)()
