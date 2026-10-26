"""Register hash closure only; no new probabilities, holdings, fits or returns."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,sys
import monthly_absolute_state135_runner as r
ROOT=Path(__file__).resolve().parents[1]

def build(root,prepare_review,outcome_review):
    root=Path(root).resolve(); out=root/'outputs/monthly_absolute_state135'
    r.require(not (out/'PLAN.json').exists(),'Registration already exists')
    hashes={}
    def add(p,expected=None):
        p=Path(p); p=(p if p.is_absolute()else root/p).resolve(); h=r.sha(p)
        r.require(expected is None or h==expected,'Changed input '+str(p))
        r.require(str(p)not in hashes or hashes[str(p)]==h,'Concurrent input change')
        hashes[str(p)]=h; return p
    def mapping(m):
        for p,h in m.items(): add(p,h)
    old=root/'outputs/monthly_latent_state134'
    add(old/'COMPLETION.json',r.REUSED_COMPLETION_SHA)
    for v in r.read(old/'COMPLETION.json')['artifacts'].values():add(v['path'],v['sha256'])
    oldplan=r.read(old/'PLAN.json'); mapping(oldplan['hashes'])
    mapping(r.source_files(root))
    inputs={k:Path(v['path'])for k,v in oldplan['inputs'].items()}
    inputs['proposal']=add('work/monthly_absolute_state135_proposal.json',r.PROPOSAL_SHA)
    inputs.update({k:old/name for k,name in r.REUSED.items()})
    refs={k:dict(path=str(add(p)),sha256=r.sha(p))for k,p in inputs.items()}
    independent={}
    for name,path in (('prepare',prepare_review),('outcome',outcome_review)):
        path=add(path); review=r.read(path); r.require(review.get('passed')is True,'Prereview incomplete')
        mapping(review['source_hashes'])
        for key in ('input_hashes','implementation_hashes'):
            if review.get(key):mapping(review[key])
        independent[name]=review['source_hashes']
    r.verify_reused(root,hashes,inputs)
    return dict(round=135,status='REGISTERED',registered_utc=datetime.now(timezone.utc).isoformat(),
        arms=list(r.NAMES),primary_challenger=r.NAMES[2],output_directory=str(out),
        inputs=refs,hashes=hashes,raw_source_hashes=oldplan['raw_source_hashes'],
        independent_prepare_source_hashes=independent['prepare'],independent_outcome_source_hashes=independent['outcome'],
        runtime=dict(python=sys.version,numpy=r.np.__version__,pandas=r.pd.__version__),tolerances=r.TOLERANCES,
        actual_model_fits_before_registration=0,actual_feature_calculations_before_registration=0,
        actual_returns_before_registration=0,no_observer_writes=True)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--prepare-review',required=True); p.add_argument('--outcome-review',required=True); a=p.parse_args()
    plan=build(ROOT,a.prepare_review,a.outcome_review); out=Path(plan['output_directory']); out.mkdir(exist_ok=True,parents=True)
    r.dump(out/'PLAN.json',plan)
    print({'path':str(out/'PLAN.json'),'sha256':r.sha(out/'PLAN.json'),'dependencies':len(plan['hashes'])})
if __name__=='__main__':main()
