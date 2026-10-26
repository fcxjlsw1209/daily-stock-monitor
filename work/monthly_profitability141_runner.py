"""Registered source-admitted annual gross-profitability experiment. No market work on import."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, sys
import numpy as np
import pandas as pd
import monthly_forward_guidance132_runner as base
import monthly_profitability141_factor as factor
require, sha, read, dump, module = base.require, base.sha, base.read, base.dump, base.module
TRAIN, MONTHS, WIDTH = 88, 122, 49
PROTECTED_SHA, LOCAL_SHA, TOLERANCES = base.PROTECTED_SHA, base.LOCAL_SHA, dict(base.TOLERANCES)
TOLERANCES.update(scores_atol=1e-12, feature_atol=1e-12, feature_rtol=0.)
NAMES=('incumbent_local67','annual_gross_profit_growth_sign','annual_gross_profitability_improvement_sign')
INPUTS=('proposal','meta','policy','original_orders','prices','sessions','protected_manifest','admitted_dataset','source_admission')
CACHE_FILES=('DATA.npz','SCORES.npz','META.json','FEATURE_AUDIT.json','PRETRAIN_ORDERS.json','PRICE_PARITY.json')
FROZEN_FILES=('PLAN.json','CACHE.json','PRETRAIN_ORDERS.json','TRAIN_TRIALS.json','WALKFORWARD.json','INDEPENDENT_PREPARE.json')
CHECKS=('original27_arrays_byte_identical','all_features_independently_reconstructed','all_three_score_matrices_and_ordered_choices_independently_verified','one_episode_and_prefix_isolation_verified','zero_and_missing_exact_fallback')
EXTRA=factor.EXTRA
PROPOSAL_SHA='80901aab35840451f289e58ca90caefefa751c717561e8cb1175d39012fd7a58'
ADMISSION_CHECKS=('semantic_review_passed','availability_review_passed',
                  'identity_review_passed','period_review_passed')
HAND_KEYS=('gross_profit_growth_sign','gross_profitability_improvement_sign')


def validate_source_admission(receipt, *, dataset_sha, meta_sha, sessions_sha, proposal_sha):
    """Check an independent pre-factor receipt; this is not source adjudication."""
    require(isinstance(receipt,dict) and receipt.get('round')==141 and
            receipt.get('stage')=='source_admission' and receipt.get('passed') is True,
            'Independent source admission required')
    require(all(receipt.get(k) is True for k in ADMISSION_CHECKS),
            'Incomplete source admission checks')
    for key,value in (('dataset_sha256',dataset_sha),('meta_sha256',meta_sha),
                      ('sessions_sha256',sessions_sha),('proposal_sha256',proposal_sha)):
        require(receipt.get(key)==value,'Source admission binding mismatch: '+key)
    require(type(receipt.get('factor_or_return_calculations')) is int and
            receipt['factor_or_return_calculations']==0,
            'Source admission must precede factor/outcome work')
    require(isinstance(receipt.get('source_hashes'),dict) and bool(receipt['source_hashes']),
            'Source admission requires its evidence hashes')



def validate_factor_output(result,meta):
    """Check the declared sign payload before any order is constructed."""
    require(isinstance(result,dict) and set(result)=={'arrays','audit'},'Factor result schema')
    arrays=result['arrays'];audit=result['audit'];shape=(len(meta['dates']),WIDTH)
    require(isinstance(arrays,dict) and set(arrays)==set(EXTRA),'Exact new profitability array schema')
    common=np.asarray(arrays['common_available'])
    require(common.shape==shape and common.dtype==np.dtype(bool),'Common availability schema')
    for key in HAND_KEYS:
        hand=np.asarray(arrays[key])
        require(hand.shape==shape and hand.dtype==np.dtype(float),'Sign feature schema: '+key)
        require(np.array_equal(np.isfinite(hand),common) and not np.isinf(hand).any(),
                'Exact common sign availability')
        require(np.isin(hand[common],[-.5,0.,.5]).all(),'Only declared sign corrections allowed')
    for i,names in enumerate(meta['symbols']):
        require(not any(common[i,j] for j,s in enumerate(names) if s in meta['etfs']),
                'ETFs cannot receive profitability corrections')
    require(isinstance(audit,dict) and type(audit.get('new_model_fits')) is int and
            audit['new_model_fits']==0,'No new model fits permitted')
    cells=audit.get('cells')
    require(isinstance(cells,list) and len(cells)==shape[0]*shape[1],'Every original source cell required')
    for i,(day,names) in enumerate(zip(meta['dates'],meta['symbols'])):
        for j,symbol in enumerate(names):
            cell=cells[i*WIDTH+j]
            require(isinstance(cell,dict) and cell.get('signal_index')==i and cell.get('pool_index')==j
                    and cell.get('signal_date')==day and cell.get('symbol')==symbol,
                    'Factor audit axes differ from original pool')

def select(rows):
    require([r['name'] for r in rows]==list(NAMES),'Three fixed arms in order required')
    fields=(('training','cagr'),('training','sharpe'),('training','max_drawdown'),
            ('early','cagr'),('recent','cagr'))
    guards={}
    for index in (0,1):
        values=[r.get(k,{}).get(s) for r in (rows[2],rows[index]) for k,s in fields]
        numeric=all(isinstance(v,(int,float,np.integer,np.floating)) and
                    not isinstance(v,(bool,np.bool_)) for v in values)
        good=False
        if numeric and np.isfinite(values).all():
            v,b=values[:5],values[5:]
            cagr=v[0]>b[0]+1e-12
            good=bool(cagr and all(x>=y-t for x,y,t in zip(v[1:],b[1:],(.05,.03,.025,.025))))
        guards[NAMES[index]]=good
    good=all(guards.values())
    return dict(primary=NAMES[2] if good else NAMES[0],admitted=good,guards=good,guards_against=guards)

class Backend(base.Local67Backend):
    def price_parity(self,m,px,op,expected_raw):
        panel,info=self.local.arch.old.panel_load(m['etfs'])
        require(base.normalized_hashes(self.local.raw_source_hashes(info),self.root)==
                base.normalized_hashes(expected_raw,self.root),'Raw price/source mismatch')
        self.panel=panel
        return self.local.verify_panel_parity(panel,px,op)

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


def choose(name,arrays,meta,n,backend,policy):
    require(name in NAMES,'Unregistered arm')
    if name==NAMES[0]:return backend.baseline(arrays,meta,n,policy)
    key=HAND_KEYS[NAMES.index(name)-1]
    return base.corrected_choices(backend.pre(arrays,n),arrays[key],arrays,meta,n,
        backend.adapter,backend.selector,policy)


def panels(a,m,backend,policy):
    original={k:(v.dtype.str,v.shape,v.tobytes()) for k,v in a.items()}
    orders,scores={},{}
    for name in NAMES:
        choices,score=choose(name,a,m,MONTHS,backend,policy)
        orders[name]=backend.make_orders(choices,score,m);scores[name]=score
        for n in (0,1,36,60,76,88):
            c,s=choose(name,a,m,n,backend,policy)
            require(c==choices[:n] and s.tobytes()==score[:n].tobytes(),'Prefix mismatch')
    for fill in (np.nan,0.):
        missing=dict(a,gross_profit_growth_sign=np.full((MONTHS,WIDTH),np.nan),gross_profitability_improvement_sign=np.full((MONTHS,WIDTH),np.nan))
        for i,names in enumerate(m['symbols']):
            for j,s in enumerate(names):
                if s not in m['etfs']:
                    missing['gross_profit_growth_sign'][i,j]=fill;missing['gross_profitability_improvement_sign'][i,j]=fill
        for name in NAMES[1:]:
            c,s=choose(name,missing,m,MONTHS,backend,policy)
            require(backend.make_orders(c,s,m)==orders[NAMES[0]] and s.tobytes()==scores[NAMES[0]].tobytes(),'Missing/zero exact fallback')
    require(original=={k:(v.dtype.str,v.shape,v.tobytes()) for k,v in a.items()},'Input mutation')
    return orders,scores


class Runner(base.Runner):
    def __init__(self, root, plan_path, plan_sha, backend_factory=Backend):
        self.root=Path(root).resolve(); self.plan_path=Path(plan_path).resolve(); self.plan_sha=plan_sha
        require(sha(self.plan_path)==plan_sha, 'PLAN hash mismatch')
        self.plan=read(self.plan_path); p=self.plan
        require(p.get('round')==141 and p.get('status')=='REGISTERED' and p.get('arms')==list(NAMES), 'Wrong round')
        require(p.get('primary_challenger')==NAMES[2] and p.get('tolerances')==TOLERANCES, 'Selection contract mismatch')
        require(p['runtime']==dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__), 'Runtime mismatch')
        self.pins=self.verify(p['hashes'])
        require(set(p['inputs'])==set(INPUTS), 'Input schema')
        self.inputs={k:self.bound_path(v['path']) for k,v in p['inputs'].items()}
        require(all(sha(self.inputs[k])==v['sha256'] for k,v in p['inputs'].items()), 'Input digest')
        require(sha(self.inputs['proposal'])==PROPOSAL_SHA and sha(self.inputs['protected_manifest'])==PROTECTED_SHA, 'Immutable policy mismatch')
        require(all(self.pins.get(k)==v for k,v in source_files(self.root).items()), 'Implementation not bound')
        for section in ('raw_source_hashes','fundamental_source_hashes','independent_prepare_source_hashes'):
            require(all(self.pins.get(k)==v for k,v in self.verify(p[section]).items()),'Unbound '+section)
        self.admitted_dataset=self.require_source_admission()
        self.backend_factory=backend_factory
        self.out=self.path(p['output_directory'])
        require(self.out==self.root/'outputs/monthly_profitability141' and self.plan_path==self.out/'PLAN.json', 'Output path')

    def require_source_admission(self):
        receipt=read(self.inputs['source_admission'])
        validate_source_admission(receipt,
            dataset_sha=sha(self.inputs['admitted_dataset']),
            meta_sha=sha(self.inputs['meta']),sessions_sha=sha(self.inputs['sessions']),
            proposal_sha=PROPOSAL_SHA)
        admitted=self.verify(receipt['source_hashes'])
        require(admitted==self.verify(self.plan['fundamental_source_hashes']),
                'Admission source closure differs from PLAN')
        require(all(self.pins.get(k)==v for k,v in admitted.items()),'Unbound admitted source')
        dataset=read(self.inputs['admitted_dataset'])
        require(isinstance(dataset,dict) and type(dataset.get('schema_version')) is int
                and dataset['schema_version']==1 and isinstance(dataset.get('events'),dict),
                'Admitted dataset schema')
        return dataset

    def prepare(self):
        self.refresh(); require(not any((self.out/x).exists() for x in (*CACHE_FILES,'CACHE.json')), 'Prepare already started')
        dataset=self.require_source_admission()
        backend=self.backend_factory(self.root,self); m,a,px,op=backend.load()
        require(len(a)==27 and m==read(self.inputs['meta']), 'Original metadata/arrays')
        base.validate_meta(m)
        parity=backend.price_parity(m,px,op,self.plan['raw_source_hashes'])
        sessions=read(self.inputs['sessions'])
        require(isinstance(sessions,dict) and isinstance(sessions.get('sessions'),list),'Explicit official sessions')
        result=factor.build_features(m,dataset)
        validate_factor_output(result,m)
        require(m==read(self.inputs['meta']),'Factor mutated original metadata')
        result['audit']['source_admission_sha256']=sha(self.inputs['source_admission'])
        result['audit']['admitted_dataset_sha256']=sha(self.inputs['admitted_dataset'])
        a=dict(a,**result['arrays'])
        orders,scores=panels(a,m,backend,read(self.inputs['policy']))
        require(orders[NAMES[0]]==read(self.inputs['original_orders']), 'Baseline changed')
        with (self.out/'DATA.npz').open('xb') as f: np.savez_compressed(f,**a)
        with (self.out/'SCORES.npz').open('xb') as f: np.savez_compressed(f,**scores)
        self.write('META.json',m); self.write('PRETRAIN_ORDERS.json',orders); self.write('PRICE_PARITY.json',parity)
        self.write('FEATURE_AUDIT.json',result['audit'])
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
        self.cache(); backend=self.backend_factory(self.root,self); m,old,px,op=backend.load()
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
        self.refresh()
        outputs=['DIAGNOSE_STARTED.json','RESULTS.json','MONTH_COMPARISON.json',
                 'ATTRIBUTION.json','VALIDATION.json']
        outputs.extend(name+'_'+mode+'_'+kind+'.json' for name in NAMES
                       for mode in ('same_close','next_open_zero','next_open_cost')
                       for kind in ('DAILY','TRADES','HOLDINGS'))
        require(not any((self.out/x).exists() for x in outputs),
                'Diagnostic stage already started; preserve partial outputs for review')
        frozen=read(self.out/'FROZEN.json'); require(set(frozen['hashes'])==set(FROZEN_FILES),'Frozen schema')
        self.verify({str(self.out/k):v for k,v in frozen['hashes'].items()})
        require(frozen['selection']==select(read(self.out/'TRAIN_TRIALS.json')) and
                frozen.get('pristine_oos') is False and frozen.get('all_diagnostic_orders_frozen') is True,'Selection changed')
        self.require_prepare_review(); backend,m,px,op=self.loaded()
        self.write('DIAGNOSE_STARTED.json',dict(plan_sha256=self.plan_sha,
            frozen_sha256=sha(self.out/'FROZEN.json'),
            started_utc=datetime.now(timezone.utc).isoformat()))
        results,periods=backend.diagnostics(read(self.out/'PRETRAIN_ORDERS.json'),m,px,op,self.write)
        for row in read(self.out/'TRAIN_TRIALS.json'):
            require(abs(results[row['name']]['same_close']['slices']['training2016_2023']['cagr']-row['training']['cagr'])<TOLERANCES['training_cagr'],'Replay training parity')
        cells=read(self.out/'FEATURE_AUDIT.json')['cells']
        require(len(cells)==MONTHS*WIDTH and len(periods)==MONTHS,'All original cell/month rows required')
        for i,row in enumerate(periods):
            group=cells[i*WIDTH:(i+1)*WIDTH]
            require(all(c['signal_index']==i and c['pool_index']==j and c['signal_date']==row['date'] and c['symbol']==m['symbols'][i][j] for j,c in enumerate(group)),'Feature/month axes')
            row['profitability']=group
        comparisons={label:backend.old130.attribution(periods,results,new=new,old=old) for label,new,old in
            (('profitability_vs_incumbent',NAMES[2],NAMES[0]),('growth_vs_incumbent',NAMES[1],NAMES[0]),('profitability_vs_growth',NAMES[2],NAMES[1]))}
        by_date={row['date']:row['profitability'] for row in periods}
        for comparison in comparisons.values():
            for change in comparison['changes']:change['profitability']=by_date[change['date']]
        self.write('RESULTS.json',results); self.write('MONTH_COMPARISON.json',periods); self.write('ATTRIBUTION.json',dict(comparisons=comparisons))
        self.write('VALIDATION.json',dict(primary=frozen['selection']['primary'],new_model_fits=read(self.out/'FEATURE_AUDIT.json')['new_model_fits'],
            original_arrays_unchanged=True,independent_outcome_review_pending=True,no_observer_writes=True,no_intramonth_trades=True,pristine_oos=False))


def source_files(root):
    root=Path(root)
    paths=[Path(__file__),Path(factor.__file__),root/'work/test_monthly_profitability141_factor.py',
        root/'work/test_monthly_profitability141_runner.py']
    return dict(base.source_files(root),**{str(p.resolve()):sha(p) for p in paths})


def main():
    p=argparse.ArgumentParser(); p.add_argument('phase',choices=('prepare','select_first88','diagnose_later34'))
    p.add_argument('--plan',required=True); p.add_argument('--plan-sha',required=True)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]); args=p.parse_args()
    getattr(Runner(args.root,args.plan,args.plan_sha),args.phase)()

if __name__=='__main__': main()
