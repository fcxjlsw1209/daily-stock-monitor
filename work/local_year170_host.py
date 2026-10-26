"""Predeclared local-only year-block stability experiment. Offline, write once.

No pooled refit per deletion, no new score budget, no intramonth trades.
Run through run_local.sh. Register precedes any new real fit or return.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import copy
import hashlib
import importlib.util
import io
import json
import os
import tempfile
import unittest
import numpy as np
import pandas as pd
import ptr169_rebuild_evaluate as ev

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/monthly_local_year170'
CENTRAL = ROOT/'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
LOCAL = ROOT/'outputs/monthly_local_linear70'
REBUILD = ROOT/'outputs/monthly_ptr_pool_source169/fixed_incumbent_rebuild/control'
PRIOR = ROOT/'outputs/monthly_sector_diffusion167/PLAN.json'
NAMES = ('incumbent_local67', 'local_year_stability', 'local_half_control')
RAW = dict(local_year_stability='local_year_stability', local_half_control='local_half')
TRAIN = 88
sha, read, require, dump = ev.sha, ev.read, ev.require, ev.dump


def now():
    return datetime.now(timezone.utc).isoformat()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def local():
    return module('local_year170_original', LOCAL/'study.py')


def contract():
    return dict(arms=list(NAMES), candidate=NAMES[1], control_nonpromotable=NAMES[2],
        period=['2016-08-17', '2026-09-24'], training_months=88, later_months=34,
        hypothesis='Only same-security local residual correction is shrunk by sensitivity to deleting a calendar year of its own mature history. Pooled fit stays fixed.',
        training='Original annual prior-December cutoff, five-year mature cohort; original own18 threshold. Delete occupied own-security signal calendar-year blocks only; no fold-specific minimum; original pooled fit, residuals, scales and unnormalized local weights stay fixed.',
        shrink='For nonzero full correction, p=sum(original local weight mass of deleted years whose remaining-sample correction is strictly same-sign)/sum(positive year masses); g=max(0,2*p-1). Fewer than two positive-mass occupied years => g=0. Full correction zero => g=1. Numerical failures abort.',
        predictions='incumbent=pooled+c; stability=pooled+g*c; half=pooled+0.5*c. No own history preserves pooled and original availability.',
        rank='Original complete forecast_available & eligible U, average centered percentile; same row readiness; no local-history filter. Initial partial date2016-08-17 has zero new delta and preserves incumbent.',
        slot='Replace only original local-quarter adjustment: coefficient=.25*(.05*weak+.01*(calm&~weak)*extension); pre=original Local67 pre+coefficient*(new_local_rank-original_local_rank); one original episode.',
        portfolio='Original dominant skip42 long momentum, unchanged eligibility and historical proxy Top32+theme17, EW3/two halves/under2 cash, fixed adjusted units intramonth, difference rebalance, no forced terminal liquidation.',
        selection=dict(first88_only=True, minimum_cagr_gain=.005, open_5bp_noninferior=True,
            sharpe_tolerance=.05, drawdown_tolerance=.03, era_cagr_tolerance=.025,
            changed_set_months_min=3, changed_set_months_each_era_min=1,
            require_noninferior_to_half=True, later34_for_selection=False),
        limitation='Reused development sample, historical price vintages and estimated membership, not pristine OOS/PIT; deleting local rows is sensitivity analysis, not independent validation because pooled fit includes those rows.',
        novelty='Local-only extension; older pooled cluster shrink and geometric support differ. Historical honest-local model source missing, so exhaustive novelty is not certified.',
        original_pool_preserved_for_control=True, no_new_valuation_choices=True,
        historical_best_tracking='FROZEN selection and latest_training_selected_candidate use first88 only. Legacy best_training_selected_candidate tracks the posthoc maximum full-period CAGR across previously first88-selected candidates; it is descriptive, not independent validation or promotion.',
        new_arms=2, registered_trials=3, execution_paths=9, no_observer_writes=True)


def code_files():
    return [ROOT/'work'/n for n in (
        'local_year170_host.py', 'local_year170_model.py', 'test_local_year170_model.py',
        'local_year170_independent.py', 'test_local_year170_independent.py',
        'test_local_year170_host.py',
        'ptr169_rebuild_evaluate.py', 'ptr169_results_independent.py')]


def verify(pins):
    require(isinstance(pins,dict) and pins,'Nonempty source pins required')
    for name,digest in pins.items():
        path=Path(name)
        if not path.is_absolute(): path=ROOT/path
        path=path.resolve()
        require('CloudStorage' not in path.parts,'Cloud input forbidden')
        require(path.is_relative_to(ROOT) or str(path).startswith('/Library/Developer/CommandLineTools/'),
                'Unapproved external dependency: '+str(path))
        require(path.is_file() and sha(path)==digest,'Changed or missing source: '+str(path))


def bind(pins,path):
    key=str(path);digest=sha(path)
    require(key not in pins or pins[key]==digest,'Conflicting inherited source digest: '+key)
    pins[key]=digest


def state_check(state):
    require(len(state['completed']) == 167 and sum(r['trials'] for r in state['completed']) == 1016,
            'Expected 167 completed rounds / 1016 trials')
    require(state['best_training_selected_candidate']['directory'] == 'monthly_local_linear70', 'Incumbent changed')


def commit_state(before, after, tag):
    expected = hashlib.sha256(before).hexdigest()
    require(CENTRAL.read_bytes() == before, 'Concurrent central update')
    backup = CENTRAL.parent/('PRE_LOCAL_YEAR170_'+tag+'_'+expected+'.json')
    if not backup.exists():
        with backup.open('xb') as stream: stream.write(before)
    require(sha(backup) == expected, 'Central backup mismatch')
    with tempfile.NamedTemporaryFile(dir=CENTRAL.parent, delete=False) as stream:
        tmp = Path(stream.name)
        stream.write((json.dumps(after, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode())
        stream.flush(); os.fsync(stream.fileno())
    try:
        require(CENTRAL.read_bytes() == before, 'Concurrent central update before commit')
        os.replace(tmp, CENTRAL)
    finally:
        tmp.unlink(missing_ok=True)


def tests():
    paths=[ROOT/'work'/n for n in ('test_local_year170_model.py',
        'test_local_year170_independent.py','test_local_year170_host.py')]
    modules = [module('test_ly170_'+str(i), p) for i, p in enumerate(paths)]
    suite = unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(m) for m in modules])
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(HostTests))
    stream = io.StringIO(); result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    print(stream.getvalue(), flush=True)
    require(result.wasSuccessful(), 'Synthetic validation failed')
    receipt = ROOT/'work/local_year170_TESTS.json'
    require(not receipt.exists(), 'Do not overwrite test receipt')
    dump(receipt, dict(passed=True, synthetic_only=True, tests=result.testsRun,
        runtime=ev.runtime(), code_sha256={str(p):sha(p) for p in code_files()}))


def register():
    require(not OUT.exists(), 'Fresh experiment directory required')
    before = CENTRAL.read_bytes(); state = json.loads(before); state_check(state)
    require(not state['registered_queue'], 'Existing registered experiment')
    require(state.get('latest_pool_data_progress', {}).get('status') ==
        'FIXED_INCUMBENT_DATA_SENSITIVITY_COMPLETED_INDEPENDENTLY_VERIFIED', 'Close prior data sensitivity first')
    validation = ROOT/'work/local_year170_TESTS.json'; t = read(validation)
    require(t['passed'] and t['synthetic_only'] and t['runtime'] == ev.runtime(), 'Synthetic tests required')
    require(t['code_sha256'] == {str(p):sha(p) for p in code_files()}, 'Code changed since tests')
    pins = dict(read(PRIOR)['hashes'])
    for p in code_files()+[PRIOR,validation]: bind(pins,p)
    for name in ('TRAIN_INPUTS.npz','MONTHLY_META.json','PARITY.json'):
        bind(pins,REBUILD/name)
    bind(pins,REBUILD.parent/'PLAN.json')
    for name in ('DATA.npz','META.json','local_linear_ORDERS.json','INCUMBENT_POLICY.json'): bind(pins,LOCAL/name)
    for name in ev.ENGINE_HASHES: bind(pins,ev.ENGINE_DIR/name)
    verify(pins)
    protected=ROOT/'work/trade_reconcile_20261015_close/NEXT_EXPECTED_PROTECTED.json'
    verify(read(protected))
    OUT.mkdir()
    with (OUT/'STATE_BEFORE.json').open('xb') as f: f.write(before)
    plan = dict(schema='local_year170_v1', created_utc=now(), runtime=ev.runtime(),
        contract=contract(), input_sha256=pins, prior_central_sha256=hashlib.sha256(before).hexdigest(),
        actual_new_fits_before_registration=0, actual_new_returns_before_registration=0)
    dump(OUT/'PLAN.json',plan)
    state.update(current=OUT.name,current_status='registered_before_new_model_fits',
        registered_queue=[dict(directory=OUT.name,plan_sha256=sha(OUT/'PLAN.json'))],
        updated_utc=now(),last_goal_turn_classification='progress',
        next_step='Prepare local-only year-block shrink, independent numerical audit, then select on first88 only.')
    commit_state(before,state,'register')
    print(json.dumps(dict(status='REGISTERED',plan_sha256=sha(OUT/'PLAN.json'))),flush=True)


def gate(expected):
    require(sha(OUT/'PLAN.json') == expected, 'Explicit PLAN SHA required')
    p = read(OUT/'PLAN.json')
    require(p['contract'] == contract() and p['runtime'] == ev.runtime(), 'Frozen contract/runtime changed')
    verify(p['input_sha256'])
    require(sha(OUT/'STATE_BEFORE.json') == p['prior_central_sha256'], 'Changed pre-registration state')
    state=read(CENTRAL)
    require(state['registered_queue'] == [dict(directory=OUT.name,plan_sha256=expected)], 'Central registration mismatch')
    state_check(state)
    return p


def rank_and_delta(study, arrays, forecast, n):
    shape=arrays['baseline'].shape
    rank_original,_=study.ranked_forecasts({k:arrays['raw_'+k] for k in study.ARMS}, arrays['model_available'], arrays,n)
    old=rank_original['local_linear']
    ranks,deltas={},{}
    _,use=study.original_rank_inputs(arrays,n)
    coef=.25*(.05*arrays['state_weak_narrow'][:n,None]+.01*(arrays['state_calm10'][:n]&~arrays['state_weak_narrow'][:n])[:,None]*arrays['extension_available'][:n])
    for name,key in RAW.items():
        raw=forecast['raw'][key].reshape(shape)
        rr,ready=study.ranked_forecasts({k:raw for k in study.ARMS}, forecast['available'].reshape(shape), arrays,n)
        ranks[name]=rr['local_linear']
        delta=np.zeros((n,shape[1])); active=use&ready[:,None]
        delta[active]=(coef*(ranks[name]-old))[active]
        if n: delta[0]=0.  # Preserve the inherited midmonth initial allocation.
        deltas[name]=delta
    return ranks,deltas


def decisions(study,a,m,forecast,n):
    policy=read(LOCAL/'INCUMBENT_POLICY.json')
    choices,scores=study.choices(study.configs()[4],a,m,n,policy=policy)
    orders={NAMES[0]:study.rp.make_orders(choices,scores,m)}
    ranks,deltas=rank_and_delta(study,a,forecast,n)
    for name in NAMES[1:]:
        pre=study.modified_pre_score(study.configs()[4],a,n)
        view=pre[:n].copy(); changed=deltas[name]!=0.; view[changed]+=deltas[name][changed]; pre[:n]=view
        ch,sc=study.adapter.score_and_choose('incumbent66',0.,np.full(a['baseline'].shape,np.nan),a,m,n,
            pre_score=pre,effects=a['effects'],episode_policy=policy,boundary_selector=study.ep.select_boundary)
        orders[name]=study.rp.make_orders(ch,sc,m)
    return orders,ranks,deltas


def require_model_audit(check,forecast):
    for key in ('passed','independent_augmented_least_squares','original_source_and_local_parity',
                'availability_unchanged','not_held_out_validation'):
        require(check.get(key) is True,'Independent model audit missing '+key)
    require(check.get('computes_returns') is False,'Preparation must not compute returns')
    audit=forecast['audit']
    require(check.get('annual_cohorts')==len(audit['annual'])==11,'Annual audit coverage')
    require(check.get('local_queries')==len(audit['local'])==int(forecast['available'].sum()),
            'Local-query audit coverage')
    require(check.get('deletion_folds')==audit['counts']['deletion_evaluations']==
            sum(len(row.get('folds',[])) for row in audit['local']),'Deletion-fold audit coverage')


def require_slot_audit(check,deltas):
    for key in ('passed','full_original_universe','includes_eligible_etfs',
                'no_local_history_filter','replacement_not_additive_local_slot'):
        require(check.get(key) is True,'Independent slot audit missing '+key)
    require(set(check.get('ready_rows',{}))==set(RAW),'Missing readiness coverage')
    require(len(set(check['ready_rows'].values()))==1,'Candidate readiness differs')
    require(check.get('changed_cells')=={k:int(np.count_nonzero(v)) for k,v in deltas.items()},
            'Changed-cell audit coverage')


def require_outcome_audit(check):
    for key in ('passed','independent_accounts','independent_of_production_model_and_engine'):
        require(check.get(key) is True,'Independent outcome audit missing '+key)
    require(check.get('checked_paths')==9 and check.get('checked_slices')==45,
            'Independent outcome coverage')
    require(check.get('terminal_liquidation') is False,'Unexpected terminal liquidation')
    require(check.get('accounting_source_sha256')==sha(ROOT/'work/ptr169_results_independent.py'),
            'Independent accounting source changed')
    require(set(check.get('paths',{}))==set(NAMES),'Missing independent arms')
    for paths in check['paths'].values():
        require(set(paths)=={p[0] for p in ev.PATHS},'Missing independent paths')
        for path in paths.values():
            require(path.get('passed') is True and path.get('monthly_executions')==122,
                    'Incomplete month-by-month independent accounting')
            require(set(path.get('slices',{}))=={'full122','first88','early53','recent35','later34'},
                    'Missing independent slices')


def prepare(expected):
    gate(expected)
    require(not (OUT/'prepared').exists(), 'Preparation is write-once')
    import local_year170_model as model
    import local_year170_independent as independent
    study=local(); m,a,px,op=study.load()
    train=ev.load_arrays(REBUILD/'TRAIN_INPUTS.npz'); meta=read(REBUILD/'MONTHLY_META.json')
    require(meta['monthly']==m,'Original monthly identity changed')
    require(meta['training']==read(study.arch.P/'META.json')['training'], 'Original training identity changed')
    widths=np.array([len(s) for s in meta['training']['symbols']])
    np.testing.assert_array_equal(train['offsets'],np.r_[0,np.cumsum(widths)])
    for key in ('dates','ends'):
        np.testing.assert_array_equal(train[key].astype('datetime64[D]'),np.repeat(
            np.array(meta['training'][key],dtype='datetime64[D]'),widths))
    original_train=ev.load_arrays(study.target.P/'TRAIN_INPUTS.npz')
    ev.equal_arrays(train,original_train)
    before={k:(v.dtype.str,v.shape,v.tobytes()) for k,v in a.items()}
    symbols=np.array([s for ss in meta['training']['symbols'] for s in ss])
    event_symbols=np.array([s for ss in m['symbols'] for s in ss])
    forecast=model.annual_forecasts(train['dates'],train['ends'],symbols,train['train_plain'],
        {k:train['target_'+k] for k in study.target.TARGETS},train['common_good'],train['event_dates'],
        event_symbols,train['event_plain'],a['eligible'].reshape(-1)&np.isfinite(train['event_plain']).all(axis=1))
    check=independent.check_prepare(train,meta,a,forecast)
    require_model_audit(check,forecast)
    orders,ranks,deltas=decisions(study,a,m,forecast,122)
    require(orders[NAMES[0]]==read(LOCAL/'local_linear_ORDERS.json'),'Original full order parity failed')
    slot=independent.verify_rank_slot(a,forecast,ranks,deltas)
    require_slot_audit(slot,deltas)
    for n in range(123):
        oo,rr,dd=decisions(study,a,m,forecast,n)
        require(all(oo[k]==orders[k][:n] for k in NAMES),'Prefix order parity failed')
        for k in NAMES[1:]:
            np.testing.assert_array_equal(rr[k],ranks[k][:n]); np.testing.assert_array_equal(dd[k],deltas[k][:n])
    require(before=={k:(v.dtype.str,v.shape,v.tobytes()) for k,v in a.items()},'Original arrays modified')
    gate(expected)
    dest=OUT/'prepared'; dest.mkdir()
    np.savez_compressed(dest/'FORECASTS.npz',**{'raw_'+k:v for k,v in forecast['raw'].items()},
        available=forecast['available'],local_available=forecast['local_available'])
    dump(dest/'MODEL_AUDIT.json',forecast['audit']);dump(dest/'ORDERS.json',orders)
    dump(dest/'META.json',m);dump(dest/'INDEPENDENT_PREPARE.json',dict(model=check,slot=slot,
        prefixes=123, original_orders_exact=True,original_arrays_unchanged=True))
    np.savez_compressed(dest/'RANKS_DELTAS.npz',**{'rank_'+k:v for k,v in ranks.items()},**{'delta_'+k:v for k,v in deltas.items()})
    files={p.name:sha(p) for p in dest.iterdir()}
    dump(dest/'COMPLETION.json',dict(status='COMPLETE',plan_sha256=expected,created_utc=now(),artifacts_sha256=files,new_returns_computed=False))
    print(json.dumps(dict(status='PREPARED_INDEPENDENTLY_VERIFIED',completion_sha256=sha(dest/'COMPLETION.json'))),flush=True)


def prepared(expected):
    gate(expected); dest=OUT/'prepared'; c=read(dest/'COMPLETION.json')
    require(c['status']=='COMPLETE' and c['plan_sha256']==expected,'Preparation incomplete')
    for n,h in c['artifacts_sha256'].items(): require(sha(dest/n)==h,'Changed prepared artifact '+n)
    return read(dest/'META.json'),read(dest/'ORDERS.json')


def train_metrics(engine,eq,m):
    ans={}
    for name,lo,hi in (('training',0,87),('early',0,52),('recent',53,87)):
        part=eq.loc[m['dates'][lo]:m['ends'][hi]]
        ans[name]=engine.stats(part/part.iloc[0]*100000.)
    return ans


def guard(candidate,reference,gain):
    return (candidate['training']['cagr']>=reference['training']['cagr']+gain-1e-12 and
        candidate['training']['sharpe']>=reference['training']['sharpe']-.05 and
        candidate['training']['max_drawdown']>=reference['training']['max_drawdown']-.03 and
        all(candidate[e]['cagr']>=reference[e]['cagr']-.025 for e in ('early','recent')))


def select_first88(metrics,orders):
    changed=[i for i in range(TRAIN) if set(orders[NAMES[1]][i]['symbols'])!=set(orders[NAMES[0]][i]['symbols'])]
    guards=dict(close_vs_incumbent=guard(metrics[NAMES[1]]['same_close_zero'],metrics[NAMES[0]]['same_close_zero'],.005),
        open_vs_incumbent=guard(metrics[NAMES[1]]['next_open_cost'],metrics[NAMES[0]]['next_open_cost'],0.),
        close_vs_half=guard(metrics[NAMES[1]]['same_close_zero'],metrics[NAMES[2]]['same_close_zero'],0.),
        open_vs_half=guard(metrics[NAMES[1]]['next_open_cost'],metrics[NAMES[2]]['next_open_cost'],0.),
        changed_support=len(changed)>=3 and any(i<53 for i in changed) and any(i>=53 for i in changed))
    return dict(primary=NAMES[1] if all(guards.values()) else NAMES[0],admitted=all(guards.values()),
        guards=guards,changed_months=changed,first88_only=True,later34_used=False)


def select(expected):
    m,orders=prepared(expected);require(not (OUT/'selected').exists(),'Selection write-once')
    px,op=ev.prices(); engine=ev.engine();metrics={}
    for name in NAMES:
        metrics[name]={}
        for path,mode,cost in (ev.PATHS[0],ev.PATHS[2]):
            _,eq,_,_=engine.replay(px,op,orders[name][:TRAIN],cost=cost,execution=mode,
                start=pd.Timestamp(m['dates'][0]),end=pd.Timestamp(m['ends'][87]))
            require(str(eq.index[-1].date())==m['ends'][87],'Training escaped cutoff')
            metrics[name][path]=train_metrics(engine,eq,m)
    result=select_first88(metrics,orders); gate(expected)
    dest=OUT/'selected';dest.mkdir();dump(dest/'TRAIN_RESULTS.json',metrics)
    dump(dest/'FROZEN.json',dict(created_utc=now(),plan_sha256=expected,
        prepared_sha256=sha(OUT/'prepared/COMPLETION.json'),orders_sha256=sha(OUT/'prepared/ORDERS.json'),
        training_sha256=sha(dest/'TRAIN_RESULTS.json'),selection=result))
    print(json.dumps(result),flush=True)


def diagnose(expected):
    m,orders=prepared(expected); frozen=read(OUT/'selected/FROZEN.json')
    require(frozen['plan_sha256']==expected and frozen['orders_sha256']==sha(OUT/'prepared/ORDERS.json') and
        frozen['training_sha256']==sha(OUT/'selected/TRAIN_RESULTS.json') and frozen['prepared_sha256']==sha(OUT/'prepared/COMPLETION.json'),'Frozen selection changed')
    require(select_first88(read(OUT/'selected/TRAIN_RESULTS.json'),orders)==frozen['selection'],'Selection mismatch')
    require(not (OUT/'results').exists(),'Diagnostics write-once')
    px,op=ev.prices();engine=ev.engine();dest=OUT/'results';dest.mkdir();results={};exports={}
    for name in NAMES:
        results[name]={};exports[name]={}
        for path,mode,cost in ev.PATHS:
            result,eq,tr,holds,positions=ev.run_path(engine,px,op,orders[name],m,mode,cost)
            result['slices']=dict(full122=result['metrics'],**ev.split_stats(engine,eq,m))
            results[name][path]=result
            data=dict(DAILY=[dict(date=str(d.date()),equity=float(v)) for d,v in eq.items()],
                TRADES=tr,HOLDINGS=holds,DAILY_POSITIONS=positions)
            exports[name][path]=data
            for kind,value in data.items():dump(dest/(name+'_'+path+'_'+kind+'.json'),value)
    import local_year170_independent as independent
    check=independent.audit_outcomes(px,op,orders,exports,results,m)
    require_outcome_audit(check)
    dump(dest/'INDEPENDENT_OUTCOME.json',check)
    dump(dest/'RESULTS.json',dict(selection=frozen['selection'],arms=results,
        reused_sample=True,strict_pit=False,observer_promotion=False))
    gate(expected)
    dump(dest/'COMPLETION.json',dict(status='COMPLETE',created_utc=now(),plan_sha256=expected,
        frozen_sha256=sha(OUT/'selected/FROZEN.json'),paths=9,slices=45,
        artifacts_sha256={p.name:sha(p) for p in dest.iterdir()}))
    print(json.dumps(dict(status='COMPLETE',primary=frozen['selection']['primary'],
        full_cagr={n:results[n]['same_close_zero']['metrics']['cagr'] for n in NAMES},
        completion_sha256=sha(dest/'COMPLETION.json'))),flush=True)


def close(expected):
    gate(expected); c=read(OUT/'results/COMPLETION.json')
    require(c['status']=='COMPLETE' and c['plan_sha256']==expected and c['paths']==9 and c['slices']==45,'Incomplete diagnostics')
    for n,h in c['artifacts_sha256'].items():require(sha(OUT/'results'/n)==h,'Changed result '+n)
    require_outcome_audit(read(OUT/'results/INDEPENDENT_OUTCOME.json'))
    m,orders=prepared(expected);frozen=read(OUT/'selected/FROZEN.json')
    require(c['frozen_sha256']==sha(OUT/'selected/FROZEN.json'),'Frozen selection receipt changed')
    require(frozen['plan_sha256']==expected and frozen['orders_sha256']==sha(OUT/'prepared/ORDERS.json') and
        frozen['training_sha256']==sha(OUT/'selected/TRAIN_RESULTS.json') and frozen['prepared_sha256']==sha(OUT/'prepared/COMPLETION.json'),'Selection dependencies changed')
    results=read(OUT/'results/RESULTS.json')
    require(results['selection']==frozen['selection']==select_first88(read(OUT/'selected/TRAIN_RESULTS.json'),orders),'Closure selection mismatch')
    before=CENTRAL.read_bytes();state=json.loads(before);state_check(state)
    protected=ROOT/'work/trade_reconcile_20261015_close/NEXT_EXPECTED_PROTECTED.json'
    verify(read(protected))
    require(not (OUT/'COMPLETION.json').exists(),'Closure write-once')
    best=state['best_training_selected_candidate'];pick=results['selection']['primary'];arm=results['arms'][pick]
    met=arm['same_close_zero']['metrics']
    if results['selection']['admitted'] and met['cagr']>best['cagr']:
        best=dict(directory=OUT.name,spec=dict(arm=pick),cagr=met['cagr'],sharpe=met['sharpe'],
            max_drawdown=met['max_drawdown'],next_open_5bp_cagr=arm['next_open_cost']['metrics']['cagr'],
            selected_on_first88=True,observer_promotion=False,limitations=contract()['limitation'])
    receipt=dict(status='COMPLETED_INDEPENDENTLY_VERIFIED',created_utc=now(),plan_sha256=expected,
        result_completion_sha256=sha(OUT/'results/COMPLETION.json'),new_trials=3,
        protected_files=len(read(protected)),best_training_selected_candidate=best,
        target_reached=best['cagr']>=.7,observers_modified=False)
    dump(OUT/'COMPLETION.json',receipt)
    state['completed'].append(dict(directory=OUT.name,trials=3,completed_utc=now(),primary=pick,
        primary_cagr=met['cagr'],primary_sharpe=met['sharpe'],new_admitted=int(results['selection']['admitted']),
        independent_execution_paths=9,protected_files=receipt['protected_files'],
        target_reached=receipt['target_reached'],completion=dict(path=str(OUT/'COMPLETION.json'),sha256=sha(OUT/'COMPLETION.json'))))
    selected_record=dict(directory=OUT.name,arm=pick,selection=results['selection'],
        full_period_cagr=met['cagr'],full_period_sharpe=met['sharpe'],selected_on_first88=True)
    state.update(best_training_selected_candidate=best,latest_training_selected_candidate=selected_record,
        best_tracking_note=contract()['historical_best_tracking'],
        registered_queue=[],current_status='completed_independently_verified',
        updated_utc=now(),last_goal_turn_classification='progress',
        next_step='Review verified local-year stability result before choosing a new distinct hypothesis; preserve frozen observers.')
    require(len(state['completed'])==168 and sum(r['trials'] for r in state['completed'])==1019,'Wrong final counters')
    commit_state(before,state,'close');print(json.dumps(receipt),flush=True)


class HostTests(unittest.TestCase):
    def sample(self):
        return {e:dict(cagr=.5,sharpe=1.5,max_drawdown=-.3) for e in ('training','early','recent')}

    def test_guard_requires_gain_and_era(self):
        b=self.sample();self.assertFalse(guard(b,b,.005));a=copy.deepcopy(b);a['training']['cagr']+=.006
        self.assertTrue(guard(a,b,.005));a['early']['cagr']-=.026;self.assertFalse(guard(a,b,.005))

    def test_later_orders_cannot_select(self):
        orders={n:[dict(symbols=['A','B','C']) for _ in range(122)] for n in NAMES}
        metrics={n:{p:self.sample() for p in ('same_close_zero','next_open_cost')} for n in NAMES}
        for p in metrics[NAMES[1]]:metrics[NAMES[1]][p]['training']['cagr']+=.01
        for i in (0,1,53):orders[NAMES[1]][i]['symbols']=['A','B','D']
        a=select_first88(metrics,orders);self.assertEqual(a['primary'],NAMES[1])
        for i in range(88,122):orders[NAMES[1]][i]['symbols']=['X']
        self.assertEqual(a,select_first88(metrics,orders))

    def test_half_never_promoted(self):
        orders={n:[dict(symbols=['A','B','C']) for _ in range(122)] for n in NAMES}
        metrics={n:{p:self.sample() for p in ('same_close_zero','next_open_cost')} for n in NAMES}
        metrics[NAMES[2]]['same_close_zero']['training']['cagr']=2.
        self.assertEqual(select_first88(metrics,orders)['primary'],NAMES[0])


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['test','register','prepare','select','diagnose','close'])
    ap.add_argument('--plan-sha256');args=ap.parse_args()
    if args.phase=='test':tests()
    elif args.phase=='register':register()
    else:require(bool(args.plan_sha256),'Explicit plan hash required');globals()[args.phase](args.plan_sha256)
