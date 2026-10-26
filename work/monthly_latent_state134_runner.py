"""Registered latent-state core experiment. No market work on import."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, sys
import numpy as np
import pandas as pd
import monthly_forward_guidance132_runner as base
import monthly_latent_state134_model as state
require, sha, read, dump, module = base.require, base.sha, base.read, base.dump, base.module
TRAIN, MONTHS, WIDTH = 88, 122, 49
PROTECTED_SHA, LOCAL_SHA, TOLERANCES = base.PROTECTED_SHA, base.LOCAL_SHA, dict(base.TOLERANCES)
TOLERANCES.update(scores_atol=1e-10, state_atol=1e-10, state_rtol=1e-9, state_ll_atol=1e-7)
NAMES=('incumbent_local67','matched_memoryless_state','filtered_latent_state')
INPUTS=('proposal','meta','policy','original_orders','prices','transition','protected_manifest')
CACHE_FILES=('DATA.npz','SCORES.npz','META.json','FEATURE_AUDIT.json','PRETRAIN_ORDERS.json','PRICE_PARITY.json','STATE_AUDIT.json')
FROZEN_FILES=('PLAN.json','CACHE.json','PRETRAIN_ORDERS.json','TRAIN_TRIALS.json','WALKFORWARD.json','INDEPENDENT_PREPARE.json')
CHECKS=('original27_arrays_byte_identical','all_probabilities_and_core_ranks_independently_verified','all_three_score_matrices_and_ordered_choices_independently_verified','one_episode_and_prefix_isolation_verified','zero_and_missing_exact_fallback')
EXTRA=('latent_filtered','latent_memoryless','latent_raw_rank','latent_risk_rank')
PROPOSAL_SHA='d752e5c823241b033a8bb4e05f49fb39224a6763eeb7de204d6075102d7fb778'

def select(rows):
    require([r['name'] for r in rows] == list(NAMES), 'Three fixed arms in order required')
    fields = (('training', 'cagr'), ('training', 'sharpe'), ('training', 'max_drawdown'),
              ('early', 'cagr'), ('recent', 'cagr'))
    values = [r.get(k, {}).get(s) for r in (rows[2], rows[0]) for k, s in fields]
    numeric = all(isinstance(v, (int, float, np.integer, np.floating)) and
                  not isinstance(v, (bool, np.bool_)) for v in values)
    good = False
    if numeric and np.isfinite(values).all():
        v, b = values[:5], values[5:]
        good = bool(v[0] > b[0] + 1e-12 and all(x >= y - t for x, y, t in
                    zip(v, b, (1e-12, .05, .03, .025, .025))))
    return dict(primary=NAMES[2] if good else NAMES[0], admitted=good, guards=good,
                guards_against={NAMES[0]: good})

class Backend(base.Local67Backend):
    def training(self, orders, m, px):
        local, rp = self.local, self.local.rp
        blocks, _, bounds, valid = rp.paths(px, m, TRAIN)
        returns, rows = [], []
        for name in NAMES:
            choices = [[m['symbols'][i].index(s) for s in row['symbols']]
                       for i, row in enumerate(orders[name][:TRAIN])]
            rr = rp.fast_returns(choices, blocks, valid)
            returns.append(rr)
            rows.append(dict(name=name, **local.old.session.measure(rr, m, bounds)))
        folds, stitched, control = [], [], []
        for n, end in ((36, 60), (60, 76), (76, 88)):
            prefix = [dict(name=name, **local.old.session.measure(rr, m, bounds, n))
                      for name, rr in zip(NAMES, returns)]
            pick = select(prefix)
            stitched.extend(returns[NAMES.index(pick['primary'])][bounds[n]:bounds[end]])
            control.extend(returns[0][bounds[n]:bounds[end]])
            folds.append(dict(n=n, end=end, selection=pick))
        return rows, dict(folds=folds, pristine_oos=False,
            metrics=rp.stats_rr(np.asarray(stitched), m['dates'][36], m['ends'][87]),
            incumbent=rp.stats_rr(np.asarray(control), m['dates'][36], m['ends'][87]))

    def diagnostics(self, orders, m, px, op, writer):
        results = {}
        for name in NAMES:
            results[name] = {}
            for mode, cost, key in (('close', 0., 'same_close'), ('open', 0., 'next_open_zero'),
                                    ('open', .0005, 'next_open_cost')):
                met, eq, tr, hold = self.local.ex.e.engine.replay(px, op, orders[name], cost=cost,
                    execution=mode, start=pd.Timestamp(m['dates'][0]), end=pd.Timestamp(m['ends'][-1]))
                for year in met['annual_returns']:
                    if year['year'] in (2016, 2026):
                        year['partial'] = True
                results[name][key] = dict(metrics=met,
                    slices=self.local.arch.old.prior.old.previous.cuts(eq, m))
                writer(name+'_'+key+'_DAILY.json', [dict(date=str(t.date()), equity=float(v)) for t, v in eq.items()])
                writer(name+'_'+key+'_TRADES.json', tr)
                writer(name+'_'+key+'_HOLDINGS.json', hold)
        periods = []
        for i, day in enumerate(m['dates']):
            held = {name: orders[name][i]['symbols'] for name in NAMES}
            returns = {name: float((px.loc[m['ends'][i], ss]/px.loc[day, ss]-1).mean()) if ss else 0.
                       for name, ss in held.items()}
            periods.append(dict(date=day, end=m['ends'][i], split='early53' if i < 53 else
                'recent35' if i < TRAIN else 'later34', holdings=held, returns=returns))
        return results, periods


def choose(name, arrays, meta, n, backend, policy):
    require(name in NAMES, 'Unregistered arm')
    if name == NAMES[0]:
        return backend.baseline(arrays, meta, n, policy)
    p = arrays['latent_memoryless' if name == NAMES[1] else 'latent_filtered'][:n]
    require(p.shape == (n,) and not np.isinf(p).any() and
            ((p[np.isfinite(p)] >= 0) & (p[np.isfinite(p)] <= 1)).all(), 'Probability bounds')
    raw, risk = arrays['latent_raw_rank'][:n], arrays['latent_risk_rank'][:n]
    rank = np.full((MONTHS, WIDTH), np.nan)
    rank[:n] = p[:, None]*raw+(1-p[:, None])*risk
    adapter = module('latent134_core', backend.root/'outputs/monthly_core_timing70_research/core_adapter.py')
    return adapter.score_and_choose('incumbent66', 1., rank, arrays, meta, n,
        pre_score=backend.pre(arrays, n), effects=arrays['effects'],
        episode_policy=policy, boundary_selector=backend.selector)


def panels(a, m, backend, policy):
    original = {k: v.tobytes() for k, v in a.items()}
    orders, scores = {}, {}
    for name in NAMES:
        choices, score = choose(name, a, m, MONTHS, backend, policy)
        orders[name] = backend.make_orders(choices, score, m)
        scores[name] = score
        for n in (0, 1, 36, 60, 76, 88):
            c, s = choose(name, a, m, n, backend, policy)
            require(c == choices[:n] and s.tobytes() == score[:n].tobytes(), 'Prefix mismatch')
    missing = dict(a, latent_filtered=np.full(MONTHS, np.nan), latent_memoryless=np.full(MONTHS, np.nan))
    for name in NAMES[1:]:
        c, s = choose(name, missing, m, MONTHS, backend, policy)
        require(backend.make_orders(c,s,m)==orders[NAMES[0]] and s.tobytes()==scores[NAMES[0]].tobytes(), 'Missing exact fallback')
    require(original == {k:v.tobytes() for k,v in a.items()}, 'Input mutation')
    return orders, scores


class Runner(base.Runner):
    def __init__(self, root, plan_path, plan_sha):
        self.root=Path(root).resolve(); self.plan_path=Path(plan_path).resolve(); self.plan_sha=plan_sha
        require(sha(self.plan_path)==plan_sha, 'PLAN hash mismatch')
        self.plan=read(self.plan_path); p=self.plan
        require(p.get('round')==134 and p.get('status')=='REGISTERED' and p.get('arms')==list(NAMES), 'Wrong round')
        require(p.get('primary_challenger')==NAMES[2] and p.get('tolerances')==TOLERANCES, 'Selection contract mismatch')
        require(p['runtime']==dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__), 'Runtime mismatch')
        self.pins=self.verify(p['hashes'])
        require(set(p['inputs'])==set(INPUTS), 'Input schema')
        self.inputs={k:self.bound_path(v['path']) for k,v in p['inputs'].items()}
        require(all(sha(self.inputs[k])==v['sha256'] for k,v in p['inputs'].items()), 'Input digest')
        require(sha(self.inputs['proposal'])==PROPOSAL_SHA and sha(self.inputs['protected_manifest'])==PROTECTED_SHA, 'Immutable policy mismatch')
        require(all(self.pins.get(k)==v for k,v in source_files(self.root).items()), 'Implementation not bound')
        self.out=self.path(p['output_directory'])
        require(self.out==self.root/'outputs/monthly_latent_state134' and self.plan_path==self.out/'PLAN.json', 'Output path')

    def prepare(self):
        self.refresh(); require(not any((self.out/x).exists() for x in (*CACHE_FILES,'CACHE.json')), 'Prepare already started')
        backend=Backend(self.root,self); m,a,px,op=backend.load()
        require(len(a)==27 and m==read(self.inputs['meta']), 'Original metadata/arrays')
        panel, info=backend.local.arch.old.panel_load(m['etfs'])
        require(base.normalized_hashes(backend.local.raw_source_hashes(info),self.root)==base.normalized_hashes(self.plan['raw_source_hashes'],self.root), 'Raw provenance')
        parity=backend.local.verify_panel_parity(panel,px,op)
        tr=module('latent134_transition',self.root/'outputs/monthly_target70_research/transition.py')
        actual, _=tr.prepare(panel,list(zip(m['dates'],m['symbols'])),set(m['etfs']))
        with np.load(self.inputs['transition'],allow_pickle=False) as z:
            require(set(z.files)==set(actual), 'Transition schema')
            for k in z.files: np.testing.assert_array_equal(actual[k],z[k])
        np.testing.assert_array_equal(actual['core'],a['core'])
        np.testing.assert_array_equal(actual['eligible'],a['eligible'])
        cal=module('latent134_calendar',self.root/'outputs/monthly_earnings_repair/research_calendar.py')
        sessions=cal.market_calendar('2008-01-01',m['ends'][-1]).sessions
        if sessions.tz is not None: sessions=sessions.tz_localize(None)
        result=state.annual_probabilities(px['SPY'],sessions,m['dates'])
        a=dict(a,latent_filtered=result['filtered'],latent_memoryless=result['memoryless'],
            latent_raw_rank=actual['calm_core']/.5,latent_risk_rank=actual['noncalm_core']/.52)
        orders,scores=panels(a,m,backend,read(self.inputs['policy']))
        require(orders[NAMES[0]]==read(self.inputs['original_orders']), 'Baseline changed')
        with (self.out/'DATA.npz').open('xb') as f: np.savez_compressed(f,**a)
        with (self.out/'SCORES.npz').open('xb') as f: np.savez_compressed(f,**scores)
        self.write('META.json',m); self.write('PRETRAIN_ORDERS.json',orders); self.write('PRICE_PARITY.json',parity)
        self.write('STATE_AUDIT.json',result['audit'])
        self.write('FEATURE_AUDIT.json',dict(original_arrays_unchanged=True,transition_cache_rebuilt_exact=True,
            original_months=MONTHS,original_width=WIDTH,new_model_fits=result['audit']['new_model_fits'],
            returns_or_nav_computed=False,original_auxiliary_states_and_core_coefficients_preserved=True))
        self.write('CACHE.json',{x:sha(self.out/x) for x in CACHE_FILES})

    def cache(self):
        data=read(self.out/'CACHE.json'); require(set(data)==set(CACHE_FILES),'Cache schema')
        return self.verify({str(self.out/k):v for k,v in data.items()})

    def require_prepare_review(self):
        cache=self.cache(); r=read(self.out/'INDEPENDENT_PREPARE.json')
        require(r.get('passed') is True and r.get('stage')=='prepare' and all(r.get(k) is True for k in CHECKS),'Prepare review incomplete')
        require(r.get('producer_modules_imported') is False and r.get('returns_or_nav_computed') is False,'Independent isolation')
        require(type(r.get('monthly_label_models_fit')) is int and r['monthly_label_models_fit']==0,'No monthly label refits')
        require(self.verify(r['source_hashes'])==self.verify(self.plan['independent_prepare_source_hashes']),'Review code mismatch')
        required=dict(cache,**{str(self.out/x):sha(self.out/x) for x in ('PLAN.json','CACHE.json')})
        got=self.verify(r['artifact_hashes']); allowed=dict(self.pins,**required)
        require(all(got.get(k)==v for k,v in required.items()) and all(allowed.get(k)==v for k,v in got.items()),'Review artifact closure')
        for key,file in (('plan_sha256','PLAN.json'),('cache_sha256','CACHE.json'),('orders_sha256','PRETRAIN_ORDERS.json')):
            require(r.get(key)==sha(self.out/file),'Review receipt binding')

    def loaded(self):
        self.cache(); backend=Backend(self.root,self); m,old,px,op=backend.load()
        require(m==read(self.out/'META.json'),'Metadata changed')
        with np.load(self.out/'DATA.npz',allow_pickle=False) as z:
            require(set(z.files)==set(old)|set(EXTRA),'Array schema')
            require(all(z[k].dtype==v.dtype and z[k].shape==v.shape and z[k].tobytes()==v.tobytes() for k,v in old.items()),'Original arrays changed')
        return backend,m,px,op

    def select_first88(self):
        self.refresh(); require(not any((self.out/x).exists() for x in ('TRAIN_TRIALS.json','WALKFORWARD.json','FROZEN.json')),'Already selected')
        self.require_prepare_review(); backend,m,px,_=self.loaded()
        rows,folds=backend.training(read(self.out/'PRETRAIN_ORDERS.json'),m,px)
        self.write('TRAIN_TRIALS.json',rows); self.write('WALKFORWARD.json',folds)
        self.write('FROZEN.json',dict(selection=select(rows),pristine_oos=False,all_diagnostic_orders_frozen=True,
            hashes={x:sha(self.out/x) for x in FROZEN_FILES}))

    def diagnose_later34(self):
        self.refresh(); require(not (self.out/'RESULTS.json').exists(),'Already diagnosed')
        frozen=read(self.out/'FROZEN.json'); require(set(frozen['hashes'])==set(FROZEN_FILES),'Frozen schema')
        self.verify({str(self.out/k):v for k,v in frozen['hashes'].items()})
        require(frozen['selection']==select(read(self.out/'TRAIN_TRIALS.json')),'Selection changed')
        self.require_prepare_review(); backend,m,px,op=self.loaded()
        results,periods=backend.diagnostics(read(self.out/'PRETRAIN_ORDERS.json'),m,px,op,self.write)
        for row in read(self.out/'TRAIN_TRIALS.json'):
            require(abs(results[row['name']]['same_close']['slices']['training2016_2023']['cagr']-row['training']['cagr'])<TOLERANCES['training_cagr'],'Replay training parity')
        comparisons={label:backend.old130.attribution(periods,results,new=new,old=old) for label,new,old in
            (('filtered_vs_incumbent',NAMES[2],NAMES[0]),('memoryless_vs_incumbent',NAMES[1],NAMES[0]),('filtered_vs_memoryless',NAMES[2],NAMES[1]))}
        self.write('RESULTS.json',results); self.write('MONTH_COMPARISON.json',periods); self.write('ATTRIBUTION.json',dict(comparisons=comparisons))
        self.write('VALIDATION.json',dict(primary=frozen['selection']['primary'],new_model_fits=read(self.out/'STATE_AUDIT.json')['new_model_fits'],
            original_arrays_unchanged=True,independent_outcome_review_pending=True,no_observer_writes=True,no_intramonth_trades=True,pristine_oos=False))


def source_files(root):
    root=Path(root)
    paths=[Path(__file__),Path(state.__file__),Path(base.__file__),root/'work/monthly_forward_guidance_factor132.py',
        root/'outputs/monthly_core_timing70_research/core_adapter.py',root/'outputs/monthly_target70_research/transition.py',
        root/'work/test_monthly_latent_state134_model.py',root/'work/test_monthly_latent_state134_runner.py',
        root/'work/register_monthly_latent_state134.py']
    return {str(p.resolve()):sha(p) for p in paths}


def main():
    p=argparse.ArgumentParser(); p.add_argument('phase',choices=('prepare','select_first88','diagnose_later34'))
    p.add_argument('--plan',required=True); p.add_argument('--plan-sha',required=True)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]); args=p.parse_args()
    getattr(Runner(args.root,args.plan,args.plan_sha),args.phase)()

if __name__=='__main__': main()
