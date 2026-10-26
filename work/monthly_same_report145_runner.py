"""Registered source-admitted annual gross-profitability experiment. No market work on import."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, sys
import numpy as np
import pandas as pd
import monthly_forward_guidance132_runner as base
import monthly_same_report144_factor as factor
require, sha, read, dump, module = base.require, base.sha, base.read, base.dump, base.module
TRAIN, MONTHS, WIDTH = 88, 122, 49
PROTECTED_SHA, LOCAL_SHA, TOLERANCES = base.PROTECTED_SHA, base.LOCAL_SHA, dict(base.TOLERANCES)
TOLERANCES.update(scores_atol=1e-12, feature_atol=1e-12, feature_rtol=0.)
NAMES=('incumbent_local67','expanded_same_report_gross_growth','expanded_same_report_profitability')
INPUTS=('proposal','meta','policy','original_orders','prices','sessions','protected_manifest','admitted_dataset','source_admission','reference144_plan','reference144_orders','reference144_training_open')
CACHE_FILES=('DATA.npz','SCORES.npz','META.json','FEATURE_AUDIT.json','PRETRAIN_ORDERS.json','PRICE_PARITY.json')
FROZEN_FILES=('PLAN.json','CACHE.json','PRETRAIN_ORDERS.json','TRAIN_TRIALS.json','TRAIN_OPEN_TRIALS.json','TRAIN_REFERENCE_OPEN.json','WALKFORWARD.json','INDEPENDENT_PREPARE.json')
CHECKS=('original27_arrays_byte_identical','all_features_independently_reconstructed','all_three_score_matrices_and_ordered_choices_independently_verified','one_episode_and_prefix_isolation_verified','zero_and_missing_exact_fallback')
EXTRA=factor.EXTRA
PROPOSAL_SHA='0e6fa70e1406234b4957a5b996f6a4551e28c6a4cf988f10d823a171231061b2'
ADMISSION_CHECKS=('complete_packages_preserved','unique_event_ids','disjoint_issuer_sets',
                  'rejected_events_preserved','availability_unmodified','original_package_hashes_verified')
HAND_KEYS=('gross_profit_growth_sign','gross_profitability_improvement_sign')
REFERENCE_NAME='same_report_profitability'
PACKAGE_NAMES=('aapl_amd','msft_nvda')
PACKAGE_SYMBOLS={'aapl_amd':frozenset(('AAPL','AMD')),'msft_nvda':frozenset(('MSFT','NVDA'))}


def _source_only_flags(receipt):
    for key in ('financial_calculations','model_fits','ranks_computed','returns_computed','network_requests'):
        require(type(receipt.get(key)) is int and receipt[key]==0,'Source-only counter: '+key)
    for key in ('source_hashes','source_code_hashes'):
        require(isinstance(receipt.get(key),dict) and bool(receipt[key]),'Missing evidence: '+key)
    for key in ('first_public_version_verified','complete_correction_history_verified','certified_daily_security_master'):
        require(receipt.get(key) is False,'Source assumptions must remain explicit: '+key)


def validate_source_admission(receipt, *, dataset_sha):
    require(isinstance(receipt,dict) and receipt.get('schema_version')==1
            and type(receipt.get('schema_version')) is int
            and receipt.get('stage')=='conditional_union_source_admission'
            and receipt.get('passed') is True,'Conditional union admission required')
    require(receipt.get('method')==factor.METHOD,'Wrong union measurement method')
    require(isinstance(receipt.get('checks'),dict) and set(receipt['checks'])==set(ADMISSION_CHECKS)
            and all(receipt['checks'][k] is True for k in ADMISSION_CHECKS),'Incomplete union source checks')
    require(receipt.get('data_sha256')==dataset_sha,'Source union dataset mismatch')
    require(isinstance(receipt.get('packages'),dict) and set(receipt['packages'])==set(PACKAGE_NAMES),
            'Exactly two complete source packages required')
    _source_only_flags(receipt)


def validate_union_dataset(dataset, packages):
    """Prove complete, disjoint source preservation without computing factors."""
    require(isinstance(dataset,dict) and type(dataset.get('schema_version')) is int
            and dataset['schema_version']==1 and dataset.get('method')==factor.METHOD
            and isinstance(dataset.get('events'),list),'Union dataset schema')
    require(set(packages)==set(PACKAGE_NAMES),'Two source packages required')
    expected={};seen_symbols=set()
    for name in PACKAGE_NAMES:
        data=packages[name]
        require(isinstance(data,dict) and type(data.get('schema_version')) is int
                and data['schema_version']==1 and data.get('method')==factor.METHOD
                and isinstance(data.get('events'),list) and data['events'],'Package dataset schema')
        symbols=set()
        for event in data['events']:
            require(isinstance(event,dict) and isinstance(event.get('event_id'),str)
                    and bool(event['event_id']) and isinstance(event.get('symbol'),str),'Package event identity')
            require(event['event_id'] not in expected,'Source event collision')
            expected[event['event_id']]=event;symbols.add(event['symbol'])
        require(symbols==PACKAGE_SYMBOLS[name] and not symbols & seen_symbols,'Disjoint declared issuer sets')
        seen_symbols.update(symbols)
    actual={}
    for event in dataset['events']:
        require(isinstance(event,dict) and isinstance(event.get('event_id'),str)
                and event['event_id'] not in actual,'Duplicate union event identity')
        actual[event['event_id']]=event
    require(actual==expected,'Union must preserve every original event and field')
    return dataset


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

def guard(candidate, control, strict=True):
    fields = (('training', 'cagr'), ('training', 'sharpe'), ('training', 'max_drawdown'),
              ('early', 'cagr'), ('recent', 'cagr'))
    values = [row.get(k, {}).get(s) for row in (candidate, control) for k, s in fields]
    if not all(isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_)) for v in values):
        return False
    if not np.isfinite(values).all():
        return False
    v, b = values[:5], values[5:]
    first = v[0] > b[0]+1e-12 if strict else v[0] >= b[0]-1e-12
    return bool(first and all(x >= y-t for x, y, t in zip(v[1:], b[1:], (.05, .03, .025, .025))))


def select(rows, open_rows, reference_open):
    for group in (rows, open_rows):
        require([r['name'] for r in group] == list(NAMES), 'Three fixed arms in order required')
    close_guards = {NAMES[i]: guard(rows[2], rows[i]) for i in (0, 1)}
    open_guards = {NAMES[i]: guard(open_rows[2], open_rows[i], strict=(i == 1)) for i in (0, 1)}
    require(reference_open.get('name')==REFERENCE_NAME,'Wrong fixed144 comparison arm')
    reference_guard=guard(open_rows[2],reference_open,strict=False)
    passed = all(close_guards.values()) and all(open_guards.values()) and reference_guard
    return dict(primary=NAMES[2] if passed else NAMES[0], admitted=passed, guards=passed,
                guards_against=close_guards, next_open_guards_against=open_guards,
                mechanism_confirmed=all(open_guards.values()),reference144_next_open_noninferior=reference_guard)

class Backend(base.Local67Backend):
    def price_parity(self,m,px,op,expected_raw):
        panel,info=self.local.arch.old.panel_load(m['etfs'])
        require(base.normalized_hashes(self.local.raw_source_hashes(info),self.root)==
                base.normalized_hashes(expected_raw,self.root),'Raw price/source mismatch')
        self.panel=panel
        return self.local.verify_panel_parity(panel,px,op)

    def open_training(self, orders, meta, px, op, n, names=NAMES):
        split = 53 if n >= 88 else n//2
        rows = []
        for name in names:
            _, eq, _, _ = self.local.ex.e.engine.replay(px, op, orders[name][:n], cost=.0005,
                execution='open', start=pd.Timestamp(meta['dates'][0]), end=pd.Timestamp(meta['ends'][n-1]))
            intervals = (('training', meta['dates'][0], meta['ends'][n-1]),
                         ('early', meta['dates'][0], meta['ends'][split-1]),
                         ('recent', meta['dates'][split], meta['ends'][n-1]))
            row = dict(name=name)
            for key, lo, hi in intervals:
                part = eq.loc[lo:hi]
                require(len(part) >= 2, 'Training open interval too short')
                row[key] = self.local.rp.stats_rr(part.to_numpy()[1:]/part.to_numpy()[:-1]-1, lo, hi)
            rows.append(row)
        return rows

    def training(self, orders, meta, px, op, reference_orders):
        rp = self.local.rp
        blocks, _, bounds, valid = rp.paths(px, meta, TRAIN)
        returns, rows = [], []
        for name in NAMES:
            indices = [[meta['symbols'][i].index(s) for s in row['symbols']] for i, row in enumerate(orders[name][:TRAIN])]
            rr = rp.fast_returns(indices, blocks, valid); returns.append(rr)
            rows.append(dict(name=name, **self.local.old.session.measure(rr, meta, bounds)))
        open_rows = self.open_training(orders, meta, px, op, TRAIN)
        reference_rows={REFERENCE_NAME:reference_orders}
        reference_open=self.open_training(reference_rows,meta,px,op,TRAIN,(REFERENCE_NAME,))[0]
        folds, stitched, control = [], [], []
        for n, end in ((36, 60), (60, 76), (76, 88)):
            prefix = [dict(name=name, **self.local.old.session.measure(rr, meta, bounds, n)) for name, rr in zip(NAMES, returns)]
            oprefix = self.open_training(orders, meta, px, op, n)
            reference_prefix=self.open_training(reference_rows,meta,px,op,n,(REFERENCE_NAME,))[0]
            selection = select(prefix, oprefix, reference_prefix)
            stitched.extend(returns[NAMES.index(selection['primary'])][bounds[n]:bounds[end]])
            control.extend(returns[0][bounds[n]:bounds[end]])
            folds.append(dict(n=n, end=end, selection=selection, open_training=oprefix, reference_open_training=reference_prefix))
        return rows, open_rows, reference_open, dict(folds=folds, pristine_oos=False,
            metrics=rp.stats_rr(np.asarray(stitched), meta['dates'][36], meta['ends'][87]),
            incumbent=rp.stats_rr(np.asarray(control), meta['dates'][36], meta['ends'][87]))

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
        for n in range(MONTHS+1):
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
        require(p.get('round')==145 and p.get('status')=='REGISTERED' and p.get('arms')==list(NAMES), 'Wrong round')
        require(p.get('primary_challenger')==NAMES[2] and p.get('tolerances')==TOLERANCES, 'Selection contract mismatch')
        require(p['runtime']==dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__), 'Runtime mismatch')
        self.pins=self.verify(p['hashes'])
        require(set(p['inputs'])==set(INPUTS), 'Input schema')
        self.inputs={k:self.bound_path(v['path']) for k,v in p['inputs'].items()}
        require(all(sha(self.inputs[k])==v['sha256'] for k,v in p['inputs'].items()), 'Input digest')
        require(sha(self.inputs['proposal'])==PROPOSAL_SHA and sha(self.inputs['protected_manifest'])==PROTECTED_SHA, 'Immutable policy mismatch')
        require(all(self.pins.get(k)==v for k,v in source_files(self.root).items()), 'Implementation not bound')
        for section in ('raw_source_hashes','fundamental_source_hashes','independent_prepare_source_hashes','independent_outcome_source_hashes'):
            require(all(self.pins.get(k)==v for k,v in self.verify(p[section]).items()),'Unbound '+section)
        self.admitted_dataset=self.require_source_admission()
        self.require_reference()
        self.backend_factory=backend_factory
        self.out=self.path(p['output_directory'])
        require(self.out==self.root/'outputs/monthly_same_report145' and self.plan_path==self.out/'PLAN.json', 'Output path')

    def require_source_admission(self):
        receipt=read(self.inputs['source_admission'])
        validate_source_admission(receipt,dataset_sha=sha(self.inputs['admitted_dataset']))
        closure=self.verify(self.plan['fundamental_source_hashes'])
        def bound_sources(item):
            require(all(k not in item['source_hashes'] or item['source_hashes'][k]==v
                        for k,v in item['source_code_hashes'].items()),'Conflicting source/code evidence')
            sources=self.verify(dict(item['source_hashes'],**item['source_code_hashes']))
            require(all(closure.get(k)==v for k,v in sources.items()),'Source package absent from PLAN closure')
            require(sources.get(str(self.inputs['sessions']))==sha(self.inputs['sessions']),
                    'Source availability calendar differs from strategy calendar')
        bound_sources(receipt)
        packages={}
        for name in PACKAGE_NAMES:
            ref=receipt['packages'][name]
            require(isinstance(ref,dict) and set(ref)=={'data_path','data_sha256','audit_path','audit_sha256'},'Package binding schema')
            data_path=self.bound_path(ref['data_path']);audit_path=self.bound_path(ref['audit_path'])
            require(sha(data_path)==ref['data_sha256'] and sha(audit_path)==ref['audit_sha256'],'Package digest mismatch')
            require(closure.get(str(data_path))==ref['data_sha256'] and closure.get(str(audit_path))==ref['audit_sha256'],
                    'Complete original package must be bound')
            audit=read(audit_path)
            require(audit.get('stage')=='conditional_source_admission' and audit.get('passed') is True
                    and audit.get('data_sha256')==sha(data_path),'Package admission invalid')
            require(isinstance(audit.get('checks'),dict) and bool(audit['checks'])
                    and all(v is True for v in audit['checks'].values()),'Incomplete original source checks')
            _source_only_flags(audit);bound_sources(audit)
            for key in ('plan_sha256','policy_sha256'):
                require(audit.get(key) in closure.values(),'Original source policy/plan unbound')
            packages[name]=read(data_path)
        for key in ('admitted_dataset','source_admission'):
            require(closure.get(str(self.inputs[key]))==sha(self.inputs[key]),'Union artifact absent from closure')
        require(all(self.pins.get(k)==v for k,v in closure.items()),'Unbound union source')
        return validate_union_dataset(read(self.inputs['admitted_dataset']),packages)

    def require_reference(self):
        plan=read(self.inputs['reference144_plan'])
        require(plan.get('round')==144 and plan.get('status')=='REGISTERED','Fixed144 plan required')
        refs=self.plan['reference144_hashes']
        require(set(refs)=={'PLAN.json','PRETRAIN_ORDERS.json','TRAIN_OPEN_TRIALS.json'},'Fixed144 reference closure')
        for key,file in (('reference144_plan','PLAN.json'),('reference144_orders','PRETRAIN_ORDERS.json'),
                         ('reference144_training_open','TRAIN_OPEN_TRIALS.json')):
            require(sha(self.inputs[key])==refs[file],'Fixed144 reference digest')
        orders=read(self.inputs['reference144_orders'])
        require(REFERENCE_NAME in orders and len(orders[REFERENCE_NAME])==MONTHS,'Complete fixed144 orders required')
        rows=read(self.inputs['reference144_training_open'])
        selected=[row for row in rows if row.get('name')==REFERENCE_NAME]
        require(len(selected)==1,'Fixed144 training reference required')
        return orders[REFERENCE_NAME],selected[0]

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
        # Source expansion must reduce exactly to frozen144 when the new two
        # issuers are removed, before looking at any return.
        old_dataset=dict(dataset,events=[e for e in dataset['events'] if e['symbol'] in PACKAGE_SYMBOLS['aapl_amd']])
        old_result=factor.build_features(m,old_dataset)
        old_arrays=dict(a,**old_result['arrays'])
        reference_orders=read(self.inputs['reference144_orders'])
        for new_name,old_name in zip(NAMES,('incumbent_local67','same_report_gross_growth','same_report_profitability')):
            old_choices,old_scores=choose(new_name,old_arrays,m,MONTHS,backend,read(self.inputs['policy']))
            require(backend.make_orders(old_choices,old_scores,m)==reference_orders[old_name],
                    'Removing new source must reproduce fixed144 ordered holdings')
        result['audit']['fixed144_orders_reproduced_without_new_sources']=True
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
        self.refresh(); require(not any((self.out/x).exists() for x in ('TRAIN_TRIALS.json','TRAIN_OPEN_TRIALS.json','TRAIN_REFERENCE_OPEN.json','WALKFORWARD.json','FROZEN.json')),'Already selected')
        self.require_prepare_review(); backend,m,px,op=self.loaded()
        reference_orders,expected_reference=self.require_reference()
        rows,open_rows,reference_open,folds=backend.training(read(self.out/'PRETRAIN_ORDERS.json'),m,px,op,reference_orders)
        for group,fields in (('training',('cagr','sharpe','max_drawdown')),('early',('cagr',)),('recent',('cagr',))):
            for field in fields:
                require(abs(reference_open[group][field]-expected_reference[group][field])<TOLERANCES['training_cagr'],
                        'Replayed fixed144 reference training mismatch')
        self.write('TRAIN_TRIALS.json',rows); self.write('TRAIN_OPEN_TRIALS.json',open_rows); self.write('TRAIN_REFERENCE_OPEN.json',reference_open); self.write('WALKFORWARD.json',folds)
        self.write('FROZEN.json',dict(selection=select(rows,open_rows,reference_open),pristine_oos=False,all_diagnostic_orders_frozen=True,
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
        require(frozen['selection']==select(read(self.out/'TRAIN_TRIALS.json'),read(self.out/'TRAIN_OPEN_TRIALS.json'),read(self.out/'TRAIN_REFERENCE_OPEN.json')) and
                frozen.get('pristine_oos') is False and frozen.get('all_diagnostic_orders_frozen') is True,'Selection changed')
        self.require_prepare_review(); backend,m,px,op=self.loaded()
        self.write('DIAGNOSE_STARTED.json',dict(plan_sha256=self.plan_sha,
            frozen_sha256=sha(self.out/'FROZEN.json'),
            started_utc=datetime.now(timezone.utc).isoformat()))
        results,periods=backend.diagnostics(read(self.out/'PRETRAIN_ORDERS.json'),m,px,op,self.write)
        for filename,mode in (('TRAIN_TRIALS.json','same_close'),('TRAIN_OPEN_TRIALS.json','next_open_cost')):
            for row in read(self.out/filename):
                require(abs(results[row['name']][mode]['slices']['training2016_2023']['cagr']-row['training']['cagr'])<TOLERANCES['training_cagr'],'Replay training parity')
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
        self.write('VALIDATION.json',dict(primary=frozen['selection']['primary'],selection=frozen['selection'],new_model_fits=read(self.out/'FEATURE_AUDIT.json')['new_model_fits'],
            original_arrays_unchanged=True,independent_outcome_review_pending=True,no_observer_writes=True,no_intramonth_trades=True,pristine_oos=False))


def source_files(root):
    root=Path(root)
    paths=[Path(__file__),Path(factor.__file__),root/'work/test_monthly_same_report144_factor.py',
        root/'work/test_monthly_same_report145_runner.py']
    return dict(base.source_files(root),**{str(p.resolve()):sha(p) for p in paths})


def main():
    p=argparse.ArgumentParser(); p.add_argument('phase',choices=('prepare','select_first88','diagnose_later34'))
    p.add_argument('--plan',required=True); p.add_argument('--plan-sha',required=True)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]); args=p.parse_args()
    getattr(Runner(args.root,args.plan,args.plan_sha),args.phase)()

if __name__=='__main__': main()
