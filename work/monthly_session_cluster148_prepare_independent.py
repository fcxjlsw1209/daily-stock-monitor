"""Independent round148 preparation oracle; no producer imports or import-time data I/O.

Scalar log returns, integer event counts and pairwise tied-rank comparisons
independently check the producer. Real-file auditing requires an
explicit registered PLAN SHA and completed preparation. Pure oracles import only
the pinned old independent selector helper, which never loads a market cache.
"""
from pathlib import Path
import hashlib
import importlib.util
import sys
from datetime import date, timedelta
from collections import Counter
import argparse
import json
import math
import re
import numpy as np
import pandas as pd
import monthly_latent_state134_prepare_review as old
import monthly_session_cluster148_review_contract as contract

ROOT=Path(__file__).resolve().parents[1]
STUDY='outputs/monthly_session_cluster148'
LOCAL=old.LOCAL
NAMES=('incumbent_local67','event_frequency_control','low_event_clustering')
METHOD=contract.METHOD
EXTRA=contract.EXTRA
HANDS=contract.HANDS
features_oracle=contract.features_oracle
CHECKS=('original27_arrays_byte_identical','all_features_independently_reconstructed',
 'all_three_score_matrices_and_ordered_choices_independently_verified',
 'one_episode_and_prefix_isolation_verified','zero_and_missing_exact_fallback')
ATOL=1e-12
INPUTS=('proposal','meta','policy','original_orders','prices','sessions','protected_manifest')
CACHE_FILES=('DATA.npz','SCORES.npz','META.json','FEATURE_AUDIT.json','PRETRAIN_ORDERS.json','PRICE_PARITY.json')
CALENDAR_SHA='4b8f32ad8a555d6537d7cf3bd0f831e64fbb2de26d033c8daaf68590fcd4a6ea'
require=old.require


def source_files(root=None):
 root=ROOT if root is None else Path(root).resolve()
 return sorted(set([root/'work/monthly_session_cluster148_prepare_independent.py',root/'work/test_monthly_session_cluster148_prepare_independent.py',root/'work/monthly_session_cluster148_review_contract.py']+old.source_files(root)))


def score_oracle(arrays,meta,frequency_rank,cluster_rank,n=None):
 helper=old.original_helper();x=helper.original_prefix(arrays,meta,n);n=x['n'];shape=x['core'].shape
 hands=[np.asarray(h)[:n] for h in (frequency_rank,cluster_rank)]
 for hand in hands:
  require(hand.shape==shape and hand.dtype==np.dtype('float64') and np.isfinite(hand).all(),'Finite rank hand')
  require((np.abs(hand)<=.5).all(),'Half-rank bound')
 scores={};choices={};deltas={}
 for k,name in enumerate(NAMES):
  pre=x['local'].copy();delta=np.zeros(shape)
  if k:
   hand=hands[k-1];active=hand!=0
   delta[active]=.01*hand[active]
   pre[active]+=delta[active]
  choice,final,_=helper.episode(pre,x)
  scores[name]=final;choices[name]=choice;deltas[name]=delta
 return scores,choices,deltas


def verify_original_arrays(original,prepared):
 require(len(original)==27 and set(prepared)==set(original)|set(EXTRA),'Original 27 plus five new arrays')
 for key,value in original.items():
  candidate=prepared[key]
  require(value.shape==candidate.shape and value.dtype==candidate.dtype and value.tobytes()==candidate.tobytes(),'Original array changed: '+key)


def verify_audit(saved,expected,meta):
 contract.feature_checks(saved,meta)
 require(saved['months']==expected['months'],'Exact skip42 window metadata')
 require(saved['cells']==expected['cells'],'Exact independent event counts and ranks')


def ordered_orders(meta,scores,choices):
 return {name:[dict(date=day,signal_date=day,end=meta['ends'][i],symbols=[meta['symbols'][i][j] for j in choice],
  scores=[float(scores[name][i,j]) for j in choice]) for i,(day,choice) in enumerate(zip(meta['dates'],choices[name]))] for name in NAMES}


def verify_scores_and_orders(actual_scores,actual_orders,meta,expected_scores,expected_choices):
 require(set(actual_scores)==set(actual_orders)==set(NAMES),'Exactly three saved arms')
 orders=ordered_orders(meta,expected_scores,expected_choices)
 for name in NAMES:
  require(actual_scores[name].shape==expected_scores[name].shape and actual_scores[name].dtype==expected_scores[name].dtype,'Full score matrix axes and dtype')
  np.testing.assert_allclose(actual_scores[name],expected_scores[name],rtol=0,atol=ATOL)
  require(len(actual_orders[name])==len(orders[name]),'Every month ordered')
  for actual,wanted in zip(actual_orders[name],orders[name]):
   require(set(actual)==set(wanted) and all(actual[k]==wanted[k] for k in ('date','signal_date','end','symbols')),'Ordered holding mismatch')
   np.testing.assert_allclose(actual['scores'],wanted['scores'],rtol=0,atol=ATOL)


def synthetic_receipt(result,root=None):
 root=ROOT if root is None else Path(root).resolve()
 reviewed=[root/'work'/name for name in ('monthly_session_cluster148_runner.py','monthly_session_cluster148_factor.py',
  'monthly_session_cluster148_proposal.json','monthly_session_cluster148_concept.json')]
 return dict(schema_version=1,stage='synthetic_prepare_review_only',passed=result.wasSuccessful(),
  tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
  runtime=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__),
  source_hashes={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files(root)},
  reviewed_contract_hashes={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in reviewed},
  actual_preparation_checked=False,actual_factors_calculated=0,actual_returns_calculated=0,
  producer_modules_imported=False,arms=list(NAMES),features=list(EXTRA),required_checks=list(CHECKS),
  actual_api='./run_local.sh work/monthly_session_cluster148_prepare_independent.py --plan-sha <SHA256> --prepare-complete')


def verify_prefixes(arrays,meta,sessions,close,opening,expected,scores,choices):
 for n in range(len(meta['dates'])+1):
  subset=dict(meta,dates=meta['dates'][:n],symbols=meta['symbols'][:n])
  last=pd.Timestamp(meta['dates'][n-1]) if n else None
  calendar=[d for d in sessions if last is not None and pd.Timestamp(d)<=last]
  prefix=features_oracle(subset,calendar,close.loc[calendar],opening.loc[calendar])
  require(prefix['cells']==expected['cells'][:n*49],'Feature prefix isolation')
  for key in EXTRA:np.testing.assert_array_equal(prefix['arrays'][key],expected['arrays'][key][:n])
  ss,cc,_=score_oracle(arrays,meta,*(prefix['arrays'][k] for k in HANDS),n=n)
  for name in NAMES:require(cc[name]==choices[name][:n] and ss[name].tobytes()==scores[name][:n].tobytes(),'Score and holding prefix isolation')
 neutral=np.zeros((len(meta['dates']),49))
 ss,cc,_=score_oracle(arrays,meta,neutral,neutral)
 for name in NAMES[1:]:require(ss[name].tobytes()==scores[NAMES[0]].tobytes() and cc[name]==choices[NAMES[0]],'Zero/missing rank exact fallback')


def audit(root,plan_path,plan_sha):
 require(type(plan_sha)is str and re.fullmatch('[0-9a-f]{64}',plan_sha),'Explicit PLAN SHA required before any file read')
 reader=old.Reader(root);directory=reader.path(STUDY)
 require(reader.path(plan_path)==directory/'PLAN.json','Dedicated round148 PLAN')
 require(reader.sha(plan_path)==plan_sha,'Frozen PLAN hash');plan=reader.json(plan_path)
 require(plan.get('round')==148 and plan.get('status')=='REGISTERED' and plan.get('arms')==list(NAMES) and plan.get('primary_challenger')==NAMES[2],'Round148 frozen arms')
 require(reader.path(plan['output_directory'])==directory and plan['dose']==.01,'Output and dose policy')
 require(plan['runtime']==dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__),'Pinned runtime')
 require(not any((directory/name).exists() for name in ('TRAIN_TRIALS.json','TRAIN_OPEN_TRIALS.json','WALKFORWARD.json','FROZEN.json','RESULTS.json')),'Preparation audit must precede outcomes')
 pins=reader.verify(plan['hashes']);require(set(plan['inputs'])==set(INPUTS),'Named inputs')
 inputs={}
 for name,ref in plan['inputs'].items():
  path=reader.path(ref['path']);require(reader.sha(path)==ref['sha256']==pins.get(str(path)),'Bound input: '+name);inputs[name]=path
 contract.verify_proposal(reader,plan,pins)
 for key in ('actual_model_fits_before_registration','actual_feature_calculations_before_registration','actual_returns_before_registration'):
  require(type(plan.get(key))is int and plan[key]==0,'Prospective registration')
 require(plan['tolerances']['feature_atol']==ATOL and plan['tolerances']['scores_atol']==ATOL and plan['tolerances']['scores_rtol']==0,'Exact preparation tolerances')
 sources={str(path):reader.sha(path) for path in source_files(root)}
 require(reader.verify(plan['independent_prepare_source_hashes'])==sources and all(pins.get(k)==v for k,v in sources.items()),'Independent source closure')
 oldplan=reader.json(LOCAL+'/PLAN.json');raw=reader.verify(plan['raw_source_hashes'])
 require(raw==reader.verify(oldplan['raw_source_hashes']) and len(raw)==64 and all(pins.get(k)==v for k,v in raw.items()),'Original raw64 unchanged')
 original_cache=reader.verify(reader.json(LOCAL+'/CACHE_HASHES.json'),reader.path(LOCAL))
 require(all(pins.get(k)==v for k,v in original_cache.items()),'Original cache bindings')
 for name,file in (('meta','META.json'),('policy','INCUMBENT_POLICY.json'),('original_orders','local_linear_ORDERS.json')):
  require(inputs[name]==reader.path(LOCAL+'/'+file),'Original local67 input: '+name)
 require(inputs['prices']==reader.path(old.PRICES),'Original prices')
 helper=old.original_helper();policy=reader.json(inputs['policy'])
 require(all(policy.get(k)==v for k,v in helper.POLICY.items()),'Frozen episode policy')
 cache=reader.json(directory/'CACHE.json');require(set(cache)==set(CACHE_FILES),'Six preparation cache artifacts');reader.verify(cache,directory)
 meta=reader.json(directory/'META.json');require(meta==reader.json(inputs['meta']),'Exact original metadata')
 require(len(meta['dates'])==len(meta['symbols'])==122 and meta['train_months']==88 and meta['early_months']==53 and len(meta['etfs'])==17 and all(len(row)==49 for row in meta['symbols']),'Original122x49 shape')
 original=reader.npz(LOCAL+'/DATA.npz');arrays=reader.npz(directory/'DATA.npz');verify_original_arrays(original,arrays)
 require(reader.sha(old.LOADER)==old.LOADER_SHA,'Frozen raw price loader')
 spec=importlib.util.spec_from_file_location('cash_quality146_independent_price_loader',reader.path(old.LOADER));loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader)
 panel,info=loader.load_panel(meta['etfs']);used=dict(info['source_hashes']);used[str(reader.path(old.LOADER))]=old.LOADER_SHA
 require({str(reader.path(k)):v for k,v in used.items()}==raw,'Exactly the registered raw64 consumed');reader.verify(used)
 prices=reader.npz(inputs['prices']);require(set(prices)=={'dates','symbols','close','opening'},'Original price cache schema')
 parity=old.price_parity(panel,prices);old.compare(reader.json(directory/'PRICE_PARITY.json'),parity,'PRICE_PARITY')
 calendar_path=inputs['sessions'].parent/'CALENDAR_PLAN.json';require(reader.sha(calendar_path)==CALENDAR_SHA,'Pinned source calendar')
 calendar=reader.json(calendar_path);require(calendar['start']=='2015-01-01' and calendar['end']=='2027-12-31','Calendar coverage')
 require(all(pins.get(k)==v for k,v in reader.verify(calendar['bindings']).items()),'Calendar provenance')
 sessions=old.official_sessions(reader,calendar['end']);sessions=[str(d.date()) for d in sessions if str(d.date())>=calendar['start']]
 saved=reader.json(inputs['sessions']);require(saved['plan_sha256']==CALENDAR_SHA and saved['sessions']==sessions,'Complete official session overlay')
 close,opening=contract.price_frames(prices,sessions,meta)
 expected=features_oracle(meta,sessions,close,opening)
 verify_audit(reader.json(directory/'FEATURE_AUDIT.json'),expected,meta)
 for key in EXTRA:
  require(arrays[key].shape==(122,49) and arrays[key].dtype==expected['arrays'][key].dtype,'New feature dtype/shape')
  np.testing.assert_array_equal(arrays[key],expected['arrays'][key])
 scores,choices,deltas=score_oracle(arrays,meta,*(expected['arrays'][key] for key in HANDS))
 saved_scores=reader.npz(directory/'SCORES.npz');saved_orders=reader.json(directory/'PRETRAIN_ORDERS.json')
 verify_scores_and_orders(saved_scores,saved_orders,meta,scores,choices)
 helper.assert_exact_tree(saved_orders[NAMES[0]],reader.json(inputs['original_orders']))
 require(saved_scores[NAMES[0]].tobytes()==scores[NAMES[0]].tobytes(),'Incumbent score bytes')
 verify_prefixes(arrays,meta,sessions,close,opening,expected,scores,choices)
 required={str(directory/name):reader.sha(directory/name) for name in (*CACHE_FILES,'PLAN.json','CACHE.json')};allowed=dict(pins,**required)
 for path,digest in list(reader.seen.items()):require(reader.sha(path)==digest and allowed.get(path)==digest,'Unbound consumed artifact: '+path)
 return dict(schema_version=1,stage='prepare',passed=True,plan_sha256=plan_sha,cache_sha256=reader.sha(directory/'CACHE.json'),orders_sha256=reader.sha(directory/'PRETRAIN_ORDERS.json'),
  **{key:True for key in CHECKS},source_hashes=sources,artifact_hashes=dict(reader.seen),
  producer_modules_imported=False,returns_or_nav_computed=False,monthly_label_models_fit=0,new_model_fits=0,
  orders_checked=366,source_cells_checked=5978,stock_cells_checked=3904,
  available_feature_cells=int(expected['arrays']['available'].sum()),
  session_event_arithmetic_verified=True,common_valid49_ranking_before_eligibility_verified=True,
  raw_prices_independently_verified=True,calendar_overlay_independently_verified=True,all_signal_prefixes_verified=True,
  max_score_error=max(float(np.max(np.abs(saved_scores[name]-scores[name]))) for name in NAMES),
  limitations=['Adjusted overnight includes dividend effects; original cached price vintage retained.',
   'Event frequency and lag1 covariance are associations, not evidence of investor identity.',
   'Only new ranks change; original27 arrays and one-episode selector remain unchanged.'])



def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--describe',action='store_true');parser.add_argument('--plan-sha')
 parser.add_argument('--plan',type=Path,default=ROOT/STUDY/'PLAN.json');parser.add_argument('--prepare-complete',action='store_true')
 parser.add_argument('--output',type=Path,default=ROOT/STUDY/'INDEPENDENT_PREPARE.json');args=parser.parse_args()
 if args.describe:
  print(json.dumps(dict(stage='oracles_and_registered_audit_ready',method=METHOD,arms=NAMES,features=EXTRA,checks=CHECKS,actual_checks_run=False,producer_modules_imported=False)));return
 require(args.prepare_complete,'Explicit completed preparation required')
 result=audit(ROOT,args.plan,args.plan_sha)
 require(args.output.resolve()==ROOT/STUDY/'INDEPENDENT_PREPARE.json','Dedicated independent output')
 with args.output.open('x') as stream:json.dump(result,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
 print(json.dumps(dict(passed=True,output=str(args.output),sha256=hashlib.sha256(args.output.read_bytes()).hexdigest())))

if __name__=='__main__':main()
