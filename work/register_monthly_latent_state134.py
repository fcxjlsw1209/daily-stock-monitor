"""Metadata/hash-only registration; never calculates a state, score or return."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,sys
import monthly_latent_state134_runner as r

ROOT=Path(__file__).resolve().parents[1]

def build(root,prepare_review,outcome_review):
 root=Path(root).resolve();out=root/'outputs/monthly_latent_state134'
 r.require(not (out/'PLAN.json').exists(),'Registration already exists')
 hashes={}
 def add(p,expected=None):
  p=Path(p);p=(p if p.is_absolute()else root/p).resolve();h=r.sha(p)
  r.require(expected is None or h==expected,'Changed input '+str(p))
  r.require(str(p)not in hashes or hashes[str(p)]==h,'Concurrent input change')
  hashes[str(p)]=h;return p
 def mapping(m):
  for p,h in m.items():add(p,h)
 proposal=add('work/monthly_latent_state134_proposal.json',r.PROPOSAL_SHA)
 add('work/monthly_latent_state134_numerics.json')
 local=r.module('latent134_register_local',root/'outputs/monthly_local_linear70/study.py')
 local.registered();local.verify_source()
 raw=r.read(local.P/'PLAN.json')['raw_source_hashes'];mapping(raw)
 for p in set(local.implementations())|set(local.source_files()):add(p)
 for name in ('PLAN.json','DESIGN.json','DESIGN.sha256','GRID.json','DATA.npz','META.json','CACHE_HASHES.json','INPUT_HASHES.json','INCUMBENT_POLICY.json','local_linear_ORDERS.json'):add(local.P/name)
 for name,h in r.read(local.P/'CACHE_HASHES.json').items():add(local.P/name,h)
 mapping(local.calendar_provenance()['source_hashes'])
 for p,h in r.source_files(root).items():add(p,h)
 # All imported legacy pure-helper implementations, including the attribution helper.
 mapping(r.base.source_files(root))
 tr=r.module('latent134_register_transition',root/'outputs/monthly_target70_research/transition.py')
 add(tr.ADAPTER_PATH)
 for p in tr.BASELINE.SOURCE_PATHS:add(p)
 add('outputs/monthly_target70_research/TRANSITION.npz')
 add('outputs/monthly_target70_research/PLAN.json')
 add('outputs/monthly_target70_research/INPUT_HASHES.json')
 # Bind cache bytes before any fit; prepare independently rebuilds every array
 # from the exact original raw64 sources rather than relying on an old result.
 tp=root/'outputs/monthly_target70_research'
 inputs=dict(proposal=proposal,meta=local.P/'META.json',policy=local.P/'INCUMBENT_POLICY.json',original_orders=local.P/'local_linear_ORDERS.json',prices=root/'outputs/monthly_2016_extension/PRICES.npz',transition=tp/'TRANSITION.npz',protected_manifest=root/'work/trade_reconcile_20261015_close/NEXT_EXPECTED_PROTECTED.json')
 refs={}
 for k,p in inputs.items():
  p=add(p,r.PROTECTED_SHA if k=='protected_manifest'else None);refs[k]=dict(path=str(p),sha256=hashes[str(p)])
 independent={}
 for name,reviewfile in (('prepare',prepare_review),('outcome',outcome_review)):
  reviewfile=add(reviewfile);review=r.read(reviewfile);r.require(review.get('passed')is True,'Prereview incomplete')
  mapping(review['source_hashes'])
  for key in ('input_hashes','implementation_hashes'):
   if review.get(key):mapping(review[key])
  independent[name]=review['source_hashes']
 return dict(round=134,status='REGISTERED',registered_utc=datetime.now(timezone.utc).isoformat(),arms=list(r.NAMES),primary_challenger=r.NAMES[2],output_directory=str(out),inputs=refs,hashes=hashes,raw_source_hashes=raw,independent_prepare_source_hashes=independent['prepare'],independent_outcome_source_hashes=independent['outcome'],runtime=dict(python=sys.version,numpy=r.np.__version__,pandas=r.pd.__version__),tolerances=r.TOLERANCES,actual_model_fits_before_registration=0,actual_feature_calculations_before_registration=0,actual_returns_before_registration=0,no_observer_writes=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--prepare-review',required=True);p.add_argument('--outcome-review',required=True);a=p.parse_args()
 plan=build(ROOT,a.prepare_review,a.outcome_review);out=Path(plan['output_directory']);out.mkdir(exist_ok=True,parents=True);r.dump(out/'PLAN.json',plan)
 print({'path':str(out/'PLAN.json'),'sha256':r.sha(out/'PLAN.json'),'dependencies':len(plan['hashes'])})
if __name__=='__main__':main()
