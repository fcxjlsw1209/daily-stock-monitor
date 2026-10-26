"""Prospective absolute-state interpretation; reuse frozen134 without new fits."""
from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd
import monthly_latent_state134_runner as prior
import monthly_absolute_state135_model as state

require, sha, read, dump, module = prior.require, prior.sha, prior.read, prior.dump, prior.module
TRAIN, MONTHS, WIDTH = prior.TRAIN, prior.MONTHS, prior.WIDTH
NAMES, CACHE_FILES, EXTRA = prior.NAMES, prior.CACHE_FILES, prior.EXTRA
PROTECTED_SHA, TOLERANCES = prior.PROTECTED_SHA, prior.TOLERANCES
select, Backend = prior.select, prior.Backend
PROPOSAL_SHA = '5bf4f3c14d7fceceb65c46ef29b5861058fa89d456b23a0e80fe8a339b3530a7'
REUSED_COMPLETION_SHA = 'c3ce897a194112ac3b840f620bc9f7d48bb09a54d26fd0b1b69437a039df4cc6'
REUSED = dict(reused_plan='PLAN.json', reused_cache='CACHE.json',
    reused_state='STATE_AUDIT.json', reused_data='DATA.npz',
    reused_prepare='INDEPENDENT_PREPARE.json', reused_completion='COMPLETION.json',
    reused_outcome='INDEPENDENT_OUTCOME.json')
INPUTS = prior.INPUTS + tuple(REUSED)


def verify_reused(root, pins, inputs):
    """Validate original completed evidence, not merely an unanchored cache."""
    folder = Path(root).resolve()/'outputs/monthly_latent_state134'
    for key, name in REUSED.items():
        require(Path(inputs[key]).resolve() == folder/name, 'Wrong reused path')
    require(sha(inputs['reused_completion']) == REUSED_COMPLETION_SHA, 'Reused completion changed')
    completion = read(inputs['reused_completion'])
    require(completion['status'] == 'completed_independently_verified', 'Uncompleted prior round')
    for ref in completion['artifacts'].values():
        p = (Path(root)/ref['path']).resolve()
        require(pins.get(str(p)) == ref['sha256'] == sha(p), 'Prior artifact not pinned')
    oldplan = read(inputs['reused_plan'])
    for p, h in oldplan['hashes'].items():
        require(pins.get(str(Path(p).resolve())) == h == sha(p), 'Prior source not pinned')
    for name, h in read(inputs['reused_cache']).items():
        require(pins.get(str(folder/name)) == h == sha(folder/name), 'Prior cache changed')
    for key in ('reused_prepare', 'reused_outcome'):
        review = read(inputs[key])
        require(review['passed'] is True and review['plan_sha256'] == sha(inputs['reused_plan']), 'Prior review invalid')


class Runner(prior.Runner):
    def __init__(self, root, plan_path, plan_sha):
        self.root=Path(root).resolve(); self.plan_path=Path(plan_path).resolve(); self.plan_sha=plan_sha
        require(sha(self.plan_path)==plan_sha, 'PLAN hash mismatch')
        self.plan=read(self.plan_path); p=self.plan
        require(p.get('round')==135 and p.get('status')=='REGISTERED' and p.get('arms')==list(NAMES), 'Wrong round')
        require(p.get('primary_challenger')==NAMES[2] and p.get('tolerances')==TOLERANCES, 'Selection contract mismatch')
        require(p['runtime']==dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__), 'Runtime mismatch')
        self.pins=self.verify(p['hashes'])
        require(set(p['inputs'])==set(INPUTS), 'Input schema')
        self.inputs={k:self.bound_path(v['path']) for k,v in p['inputs'].items()}
        require(all(sha(self.inputs[k])==v['sha256'] for k,v in p['inputs'].items()), 'Input digest')
        require(sha(self.inputs['proposal'])==PROPOSAL_SHA and sha(self.inputs['protected_manifest'])==PROTECTED_SHA, 'Immutable policy mismatch')
        require(all(self.pins.get(k)==v for k,v in source_files(self.root).items()), 'Implementation not bound')
        verify_reused(self.root,self.pins,self.inputs)
        self.out=self.path(p['output_directory'])
        require(self.out==self.root/'outputs/monthly_absolute_state135' and self.plan_path==self.out/'PLAN.json', 'Output path')

    def prepare(self):
        self.refresh(); require(not any((self.out/x).exists() for x in (*CACHE_FILES,'CACHE.json')), 'Prepare already started')
        backend=Backend(self.root,self); m,old,px,op=backend.load()
        require(len(old)==27 and m==read(self.inputs['meta']), 'Original metadata/arrays')
        with np.load(self.inputs['reused_data'],allow_pickle=False) as z:
            require(set(z.files)==set(old)|set(EXTRA), 'Prior array schema')
            for k,v in old.items():
                require(z[k].shape==v.shape and z[k].dtype==v.dtype and z[k].tobytes()==v.tobytes(), 'Original array changed')
            a={k:z[k].copy() for k in z.files}
        result=state.transform(read(self.inputs['reused_state']),a['latent_filtered'],a['latent_memoryless'])
        a.update(latent_filtered=result['filtered'],latent_memoryless=result['memoryless'])
        require(result['audit']['new_model_fits']==0, 'Unexpected model fit')
        orders,scores=prior.panels(a,m,backend,read(self.inputs['policy']))
        require(orders[NAMES[0]]==read(self.inputs['original_orders']), 'Baseline changed')
        with (self.out/'DATA.npz').open('xb') as f: np.savez_compressed(f,**a)
        with (self.out/'SCORES.npz').open('xb') as f: np.savez_compressed(f,**scores)
        self.write('META.json',m); self.write('PRETRAIN_ORDERS.json',orders)
        # Same original raw-price proof, all bytes transitively bound by completed134.
        self.write('PRICE_PARITY.json',read(self.root/'outputs/monthly_latent_state134/PRICE_PARITY.json'))
        self.write('STATE_AUDIT.json',result['audit'])
        self.write('FEATURE_AUDIT.json',dict(original_arrays_unchanged=True,
            reused_completed_round134=True, original_months=MONTHS,original_width=WIDTH,
            new_model_fits=0,reused_model_fits=result['audit']['reused_model_fits'],
            returns_or_nav_computed=False,original_auxiliary_states_and_core_coefficients_preserved=True))
        self.write('CACHE.json',{x:sha(self.out/x) for x in CACHE_FILES})

    def write(self,name,value):
        if name=='VALIDATION.json':
            value=dict(value,reused_model_fits=read(self.out/'STATE_AUDIT.json')['reused_model_fits'])
        return super().write(name,value)


def source_files(root):
    root=Path(root)
    paths=[Path(__file__),Path(state.__file__),root/'work/test_monthly_absolute_state135_model.py',
        root/'work/test_monthly_absolute_state135_runner.py',root/'work/register_monthly_absolute_state135.py']
    return dict(prior.source_files(root),**{str(p.resolve()):sha(p) for p in paths})


def main():
    p=argparse.ArgumentParser(); p.add_argument('phase',choices=('prepare','select_first88','diagnose_later34'))
    p.add_argument('--plan',required=True); p.add_argument('--plan-sha',required=True)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]); args=p.parse_args()
    getattr(Runner(args.root,args.plan,args.plan_sha),args.phase)()

if __name__=='__main__': main()
