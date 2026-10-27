"""Prospectively registered monthly pooled-Ridge clock replication.

All actual inputs/models/orders/returns require the PLAN SHA. Research only.
"""
from pathlib import Path
from datetime import datetime, timezone
from copy import deepcopy
import argparse, hashlib, io, json, unittest
import numpy as np
import pandas as pd
import ptr169_rebuild_evaluate as ev
import monthly_ownsector_gate238_host as old
import monthly_forecast_spacing269_host as source
import monthly_weak_tree261_host as training_source
from monthly_same_report145_runner import guard

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/monthly_matured_clock273'
PRIOR=ROOT/'outputs/monthly_roa_increment272'
REFERENCE=ROOT/'outputs/monthly_forecast_spacing269/prepared'
REBUILD=training_source.REBUILD
CENTRAL=old.CENTRAL
TESTS=ROOT/'work/monthly_matured_clock273_TESTS.json'
REVIEW=ROOT/'work/monthly_matured_clock273_CODE_REVIEW.json'
RESULT_REVIEW=ROOT/'work/monthly_matured_clock273_RESULT_REVIEW.json'
ARMS=('incumbent_local67','monthly_pooled_clock')
CODE=tuple('work/'+name for stem in ('host','kernel','inputs','independent') for name in ('monthly_matured_clock273_'+stem+'.py','test_monthly_matured_clock273_'+stem+'.py'))
DESIGN={
 'objective':'2016-08-17..2026-09-24 same-close ZERO-cost CAGR>=70%; dominant skip42 long momentum, monthly EW3/2half/<2cash and fixed intramonth units.',
 'hypothesis':'Matured monthly observations may adapt the small pooled Ridge ranking correction sooner than the prior-Dec31 annual model. Test the full rolling training clock, not a larger ML allocation.',
 'classification':'conditional replication on current Local67; global novelty UNKNOWN; round33 monthly_monthly_update70 four trials existed but exact source/parameters are missing. No new information-family claim.',
 'arms':list(ARMS),'trials':2,
 'model':'Unchanged plain10 inputs, original clipped month-centered log_excess, common_good complete month cohort, at least18 mature months,5calendar-year lower exclusive,alpha10,12month half-life, month-balanced then rowcount-normalized weights, training-only standardization. Monthly candidate uses cutoff=query close and end<=query, rolling lowerbound/weights/scaler together. Annual calibration uses priorDec31 and must exactly reproduce original raw_source_log/availability/ranks before any candidate fit. Solver failure aborts, no lstsq rescue.',
 'scope':'Only original weak AND paired_month_ready AND not initial2016-08-17. Original forecast_available&eligible rank population fixed. Change pre-score by .0375*(monthly_pool_rank-original_pool_rank) before the original single episode selection. Original local .0125 rank, calm extension, nonready fallback, .50/.52 core, auxiliary factors, universe, signals, eligibility and exit policy unchanged. No new availability expansion; insufficient needed history is an error.',
 'inputs':'Preserve original169 training180months/8820rows byte-for-byte. Append Jan..Jul2026 seven full cohorts/343identity rows using matching event_plain/E and same-source complete-path label admission and original four targets/common_good. As-of label computation at prefix query excludes later endpoints and Aug partial Sep24 label. Numeric source data can be loaded globally; decisions/training row selection remain as-of. No claim of zero future-file byte reads.',
 'prefixes':'All123 order/score prefixes from authenticated forecasts without refitting; fresh physical0/1/53/88 model/decision recomputations and full122 main reuse. Future-label poisoning synthetic checks. Count production/independent canonical/SVD fits separately.',
 'selection':'First88 months only, same-close ZERO. Single candidate strict145 guard: CAGR>inc+1e-12,Sharpe>=inc-.05,MDD>=inc-.03,early53/recent35 CAGR>=inc-.025. Freeze selection before later34/full122 and all next-open performance. No fee or later result selects. No parameter tuning after seeing outcomes.',
 'diagnostics':'Two arms x sameclose0,nextopen0,nextopen.0005/side; five slices including full122,first88,early53,recent35,later34. Independent cash/units/fees/accounting. Monthly calendar end==query labels allowed as same-close idealization; next-open separate.',
 'limitations':['Repeated historical development, not pristine OOS.','Estimated historical Top32 and assumed17ETF; strict_pit=false.','Annual local correction remains frozen; only pooled component updates.','No frozen observer, source, prior PLAN, source hashes, or ledger modifications.'],
 'previous_goal_turn_classification':'progress: completed source272 validation and GitHub sync; audit found broad prior-factor overlap; no actual273 data yet.'}
sha,read,dump,require=ev.sha,ev.read,ev.dump,ev.require
exact=source.exact_array

def now():return datetime.now(timezone.utc).isoformat()
def codes():return {s:sha(ROOT/s) for s in CODE}
def checkpoint(folder,expected):
    c=read(folder/'COMPLETION.json');require(c['plan_sha256']==expected and c['status']=='COMPLETE','Phase completion')
    for name,h in c['artifacts_sha256'].items():require(sha(folder/name)==h,'Changed phase artifact '+name)
    return c

def tests():
    require(not TESTS.exists(),'Tests write once')
    mods=[old.module('clock273_test'+str(i),ROOT/p) for i,p in enumerate(CODE) if p.startswith('work/test_')]
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromModule(m) for m in mods)
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite);print(stream.getvalue(),flush=True)
    require(result.wasSuccessful(),'Synthetic tests failed')
    dump(TESTS,dict(passed=True,synthetic_only=True,tests=result.testsRun,failures=0,errors=0,code_sha256=codes(),runtime=ev.runtime()))

def register():
    import monthly_matured_clock273_kernel as kernel
    import monthly_matured_clock273_inputs as inputs
    import monthly_matured_clock273_independent as independent
    require(not OUT.exists(),'Fresh experiment required')
    before=CENTRAL.read_bytes();state=json.loads(before)
    require(state['registered_queue']==[] and state['completed'][-1]['directory']==PRIOR.name,'Unqueued closed272 predecessor')
    require(len(state['completed'])==267 and sum(x['trials'] for x in state['completed'])==1312,'Exact preceding counters')
    done=read(PRIOR/'COMPLETION.json');require(done['status']=='COMPLETED_INDEPENDENTLY_VERIFIED' and done['observers_modified'] is False,'Completed prior')
    require(state['completed'][-1]['completion']['sha256']==sha(PRIOR/'COMPLETION.json'),'Predecessor central binding')
    t=read(TESTS);r=read(REVIEW)
    require(t['passed'] and t['synthetic_only'] and t['code_sha256']==codes() and t['runtime']==ev.runtime(),'Current formal tests')
    require(r['passed'] is True and r['unresolved_blockers']==[] and r['code_sha256']==codes() and r['tests_sha256']==sha(TESTS),'Prospective independent code review')
    pins=source.canonical_pins(read(PRIOR/'PLAN.json')['input_sha256']);closure=source.bind_source_closure(pins)
    study=old.local()
    files=[ROOT/p for p in CODE]+[TESTS,REVIEW,PRIOR/'COMPLETION.json',PRIOR/'PLAN.json',old.PROTECTED,old.CALENDAR]
    files+=list(study.implementations())+list(study.source_files())+list(kernel.dependency_files())+list(inputs.dependency_files())+list(independent.dependency_files())
    files += [REFERENCE/n for n in ('COMPLETION.json','META.json','ORDERS.json','SCORES.npz','PRE_SCORES.npz')]
    files += [Path(ev.__file__),Path(old.__file__),Path(source.__file__),Path(training_source.__file__),ev.PRICE]
    for p in files:old.bind(pins,Path(p))
    raw=read(old.LOCAL/'PLAN.json')['raw_source_hashes'];require(len(raw)==64,'Original64 source map')
    for p,h in raw.items():old.bind(pins,Path(p),h)
    old.verify(pins);protected=read(old.PROTECTED);old.verify(protected);require(len(protected)==21444,'Protection inventory')
    OUT.mkdir();(OUT/'STATE_BEFORE.json').write_bytes(before)
    dump(OUT/'PLAN.json',dict(schema='monthly_matured_clock273_v1',created_utc=now(),design=DESIGN,runtime=ev.runtime(),code_sha256=codes(),input_sha256=pins,original169_source_closure=closure,raw_source_hashes=raw,prior_central_sha256=hashlib.sha256(before).hexdigest(),actual_features_before_registration=0,actual_models_before_registration=0,actual_returns_before_registration=0,observers_modified=False))
    expected=sha(OUT/'PLAN.json');state.update(current=OUT.name,current_status='registered_before_features',registered_queue=[dict(directory=OUT.name,plan_sha256=expected)],updated_utc=now(),last_goal_turn_classification='progress')
    old.commit_state(before,state,'matured_clock273_register');print(json.dumps(dict(status='REGISTERED',plan_sha256=expected)),flush=True)

def gate(expected):
    require(sha(OUT/'PLAN.json')==expected,'Wrong PLAN SHA');plan=read(OUT/'PLAN.json')
    require(plan['design']==DESIGN and plan['runtime']==ev.runtime() and plan['code_sha256']==codes(),'Frozen design/runtime/code')
    old.verify(plan['input_sha256']);require(source.source_closure_check(plan['input_sha256'])==plan['original169_source_closure'],'Original169 closure')
    state=read(CENTRAL);require(state['current']==OUT.name and state['registered_queue']==[dict(directory=OUT.name,plan_sha256=expected)],'Active registration')
    return plan

def physical(arrays,meta,n):
    require(type(n)is int and 0<=n<=len(meta['dates']),'Prefix')
    require(len(arrays)==27,'Original27')
    return {k:v[:n].copy() for k,v in arrays.items()},old.meta_prefix(meta,n)

def decisions(study,arrays,meta,delta,*,reference=None):
    import monthly_selector_scope268_independent as independent
    n=len(meta['dates']);require(delta.shape==arrays['baseline'].shape and np.isfinite(delta).all(),'Delta shape/finite')
    P=study.modified_pre_score(study.configs()[4],arrays,n)[:n].copy()
    source.forecast_inputs(study,P,arrays,meta,n)
    orders={};scores={};pres={};proofs={}
    for arm in ARMS:
        p=P.copy()
        if arm==ARMS[1]:
            active=delta!=0.;p[active]+=delta[active]
        args=source.selector_inputs(study,p,arrays,meta,n)
        original=dict(baseline=args[4],core=args[5],core_weight=args[6],state_calm10=args[7],state_weak_narrow=args[8])
        choices,F=study.ep.select_boundary(p,args[1],args[2],args[3],original,dict(budget=.05,ceiling=5,gap=.02))
        proof=source.selector_receipt(independent.verify(*args,choices,F,'incumbent_local67'),n,p.shape[1])
        orders[arm]=study.rp.make_orders(choices,F,meta);scores[arm]=F;pres[arm]=p;proofs[arm]=proof
    if reference is not None:
        require(orders[ARMS[0]]==reference['orders'][:n],'Exact incumbent orders before candidate fits')
        exact(scores[ARMS[0]],reference['scores'][:n],'Exact incumbent finalscore');exact(P,reference['pre'][:n],'Exact incumbent pre')
    return orders,scores,pres,proofs

def prepare(expected):
    import monthly_matured_clock273_kernel as kernel
    import monthly_matured_clock273_inputs as inputs
    import monthly_matured_clock273_independent as independent
    plan=gate(expected);require(not (OUT/'prepared').exists(),'Prepare write once')
    study=old.local();study.verify_source();meta,arrays,px,opening=study.load()
    require(len(meta['dates'])==122 and len(arrays)==27 and arrays['baseline'].shape==(122,49),'Source axes')
    reference=dict(orders=read(REFERENCE/'ORDERS.json')[ARMS[0]],scores=ev.load_arrays(REFERENCE/'SCORES.npz')[ARMS[0]],pre=ev.load_arrays(REFERENCE/'PRE_SCORES.npz')[ARMS[0]])
    require(read(REFERENCE/'META.json')==meta and reference['orders']==read(old.LOCAL/'local_linear_ORDERS.json'),'Reference metadata/orders')
    decisions(study,arrays,meta,np.zeros((122,49)),reference=reference)
    train=ev.load_arrays(REBUILD/'TRAIN_INPUTS.npz');combined=read(REBUILD/'MONTHLY_META.json');require(combined['monthly']==meta,'Training/event identities')
    before=old.array_snapshot(arrays);tb=old.array_snapshot(train)
    panel,info=study.arch.old.panel_load(meta['etfs']);require(study.raw_source_hashes(info)==plan['raw_source_hashes'],'Original panel hashes')
    priceproof=study.verify_panel_parity(panel,px,opening);sessions=read(old.CALENDAR)['sessions']
    extended,append_audit=inputs.append_completed(train,combined,panel,sessions,asof=meta['dates'][-1])
    augmented=deepcopy(combined);augmented['training']=append_audit['training_metadata']
    controls=[]
    def control_validator(view):
        require(view['candidate'] is None and view['audit']['annual_control_exact'] is True,'Annual control precedes candidate')
        controls.append(view['n']);return dict(passed=True,original_orders_verified=True)
    bundle=kernel.fit_models(extended,augmented,arrays,n=122,control_validator=control_validator)
    modelproof=independent.verify_models(extended,augmented,arrays,bundle,n=122)
    ranks,delta,delta_audit=kernel.rank_and_delta(arrays,meta,bundle,n=122)
    deltapr=independent.verify_delta(arrays,meta,bundle,ranks,delta,delta_audit,n=122)
    orders,scores,pres,selectorproof=decisions(study,arrays,meta,delta,reference=reference)
    prefixes=[]
    for n in range(123):
        aa,mm=physical(arrays,meta,n)
        oo,ss,pp,proof=decisions(study,aa,mm,delta[:n].copy(),reference=reference)
        for arm in ARMS:
            require(oo[arm]==orders[arm][:n],'Cached order prefix');exact(ss[arm],scores[arm][:n],'Cached scoreprefix');exact(pp[arm],pres[arm][:n],'Cached preprefix')
        prefixes.append(dict(n=n,exact=True,selector=proof))
    fresh=[]
    for n in (0,1,53,88):
        pt,pm,pa,scope=kernel.physical_inputs(extended,augmented,arrays,n)
        small=kernel.fit_models(pt,pm,pa,n=n,control_validator=control_validator)
        mp=independent.verify_models(pt,pm,pa,small,n=n)
        rr,dd,da=kernel.rank_and_delta(pa,pm['monthly'],small,n=n)
        dp=independent.verify_delta(pa,pm['monthly'],small,rr,dd,da,n=n)
        for role in ('control','candidate'):
            for key in ('raw','available','rank'):exact(small[role][key],bundle[role][key][:n],'Fresh modelprefix '+role+'/'+key)
        exact(dd,delta[:n],'Fresh deltaprefix')
        oo,ss,pp,sp=decisions(study,pa,pm['monthly'],dd,reference=reference)
        for arm in ARMS:require(oo[arm]==orders[arm][:n],'Fresh orderprefix');exact(ss[arm],scores[arm][:n],'Freshscoreprefix')
        fresh.append(dict(n=n,model=mp,delta=dp,selector=sp,scope=scope))
        print(json.dumps(dict(stage='273_fresh_prefix',n=n)),flush=True)
    require(before==old.array_snapshot(arrays) and tb==old.array_snapshot(train),'Original inputs mutated')
    gate(expected);dest=OUT/'prepared';dest.mkdir()
    np.savez_compressed(dest/'FORECASTS.npz',**{role+'_'+key:bundle[role][key] for role in ('control','candidate') for key in ('raw','available','rank')},delta=delta)
    np.savez_compressed(dest/'APPENDED_INPUTS.npz',**extended)
    for name,value in [('SCORES',scores),('PRE_SCORES',pres)]:np.savez_compressed(dest/(name+'.npz'),**value)
    records={'META.json':meta,'ORDERS.json':orders,'APPEND_AUDIT.json':append_audit,'MODEL_AUDIT.json':{'control':bundle['control']['audit'],'candidate':bundle['candidate']['audit'],'audit':bundle['audit']},'PREP_AUDIT.json':dict(model=modelproof,delta=deltapr,selector=selectorproof,cached_prefixes=prefixes,fresh_prefixes=fresh,full122_reuses_main=True,annual_control_calls=controls),'SOURCE_PARITY.json':dict(original_arrays_unchanged=True,original_training_unchanged=True,original_orders_exact=True,original_pre_and_scores_exact=True,original_array_fingerprints=old.array_fingerprints(arrays),original_training_fingerprints=old.array_fingerprints(train),price_parity=priceproof,raw_source_hashes=plan['raw_source_hashes'],strict_pit=False)}
    for name,value in records.items():dump(dest/name,value)
    dump(dest/'COMPLETION.json',dict(status='COMPLETE',created_utc=now(),plan_sha256=expected,events=122,artifacts_sha256={p.name:sha(p) for p in dest.iterdir()},portfolio_returns_computed=False))
    print(json.dumps(dict(status='PREPARED',completion_sha256=sha(dest/'COMPLETION.json'))),flush=True)

def prepared(expected):
    gate(expected);checkpoint(OUT/'prepared',expected)
    return read(OUT/'prepared/META.json'),read(OUT/'prepared/ORDERS.json')

def select_first88(metrics):
    require(set(metrics)==set(ARMS),'Two fixed arms')
    for groups in metrics.values():
        require(set(groups)=={'training','early','recent'},'Only first88 groups')
        for group,fields in (('training',('cagr','sharpe','max_drawdown')),('early',('cagr',)),('recent',('cagr',))):
            for field in fields:
                v=groups[group][field];require(isinstance(v,(float,int,np.floating,np.integer)) and not isinstance(v,(bool,np.bool_)) and np.isfinite(v),'Finite required first88 metric')
    admitted=guard(metrics[ARMS[1]],metrics[ARMS[0]],strict=True)
    return dict(primary=ARMS[1] if admitted else ARMS[0],admitted=admitted,first88_only=True,later34_used=False,cost_used=False)

def select(expected):
    meta,orders=prepared(expected);require(not (OUT/'selected').exists(),'Selection write once')
    px,op=ev.prices();engine=ev.engine();metrics={}
    for arm in ARMS:
        _,eq,_,_=engine.replay(px,op,orders[arm][:88],cost=0.,execution='close',start=pd.Timestamp(meta['dates'][0]),end=pd.Timestamp(meta['ends'][87]))
        require(str(eq.index[-1].date())==meta['ends'][87],'First88 endpoint');metrics[arm]=old.train_metrics(engine,eq,meta)
    pick=select_first88(metrics);gate(expected);dest=OUT/'selected';dest.mkdir();dump(dest/'TRAIN_RESULTS.json',metrics)
    dump(dest/'FROZEN.json',dict(created_utc=now(),plan_sha256=expected,prepared_sha256=sha(OUT/'prepared/COMPLETION.json'),training_sha256=sha(dest/'TRAIN_RESULTS.json'),selection=pick))
    print(json.dumps(pick),flush=True)

def frozen(expected):
    f=read(OUT/'selected/FROZEN.json');require(f['plan_sha256']==expected and f['prepared_sha256']==sha(OUT/'prepared/COMPLETION.json') and f['training_sha256']==sha(OUT/'selected/TRAIN_RESULTS.json'),'Selection bindings')
    require(f['selection']==select_first88(read(OUT/'selected/TRAIN_RESULTS.json')),'Recomputed frozen selection');return f

# audit_outcomes follows as an adapted two-arm copy of immutable 271 accounting glue.
def audit_outcomes(px,opening,orders,exports,results,meta):
    """Two arms, three paths and five slices; reuse only independent accounting."""
    import monthly_basket_core251_independent as source_ind
    helper=source_ind.accounting()
    require(set(orders)==set(exports)==set(results)==set(ARMS),'Outcome arm names differ')
    require(len(meta['dates'])==len(meta['ends'])==122 and meta['ends'][:-1]==meta['dates'][1:],'Outcome interval axis differs')
    require(meta['dates'][0]=='2016-08-17' and meta['ends'][-1]=='2026-09-24' and
        meta['ends'][52]==meta['dates'][53]=='2020-12-31' and meta['ends'][87]==meta['dates'][88]=='2023-11-30',
        'Outcome fixed slices differ')
    reports={}
    for arm in ARMS:
        require(len(orders[arm])==122 and [o['date'] for o in orders[arm]]==meta['dates'] and
                [o['end'] for o in orders[arm]]==meta['ends'],'Order axis differs')
        require(set(exports[arm])==set(results[arm])=={p[0] for p in ev.PATHS},'Execution paths differ')
        reports[arm]={}
        for name,execution,cost in ev.PATHS:
            data=exports[arm][name]
            require(set(data)=={'DAILY','TRADES','HOLDINGS','DAILY_POSITIONS'},'Incomplete execution exports')
            if cost==0.:
                require(all(float(row['cost'])==0. for row in data['TRADES']), 'Zero-cost trade fees')
                require(all(float(row['fee'])==0. for row in data['HOLDINGS']), 'Zero-cost period fees')
            audited=helper.audit_path(px,opening,orders[arm],*[data[k] for k in ('DAILY','TRADES','HOLDINGS','DAILY_POSITIONS')],execution,cost)
            slices=helper.statistics_by_split(audited['days'],audited['values'],meta);wanted=results[arm][name]
            require(set(wanted['slices'])==set(slices),'Fixed five slices differ')
            helper.check_metrics(slices['full122'],wanted['metrics'],arm+'/'+name)
            for split,value in slices.items():helper.check_metrics(value,wanted['slices'][split],arm+'/'+name+'/'+split)
            for key in ('fees_paid','dollar_turnover','trade_rows'):helper.near(audited[key],wanted['metrics'][key],key)
            annual=[]
            for year in sorted({d[:4] for d in audited['days']}):
                ids=[i for i,day in enumerate(audited['days']) if day.startswith(year)]
                if len(ids)<2:continue
                prior=audited['values'][ids[0]-1] if ids[0] else 100000.
                annual.append(dict(year=int(year),return_=audited['values'][ids[-1]]/prior-1.,partial=int(year) in (2016,2026)))
            # Preserve the frozen metric comparison tolerance; feature/selection checks stay exact.
            require(len(wanted['metrics']['annual_returns'])==len(annual),'Annual metric count')
            for got,expected in zip(wanted['metrics']['annual_returns'],annual):
                require(set(got)==set(expected) and got['year']==expected['year'] and got['partial']==expected['partial'],'Annual metric identity')
                helper.near(got['return_'],expected['return_'],'Annual return')
            require(wanted['accounting']['exported_fill_ledger_matches'] is True and
                    wanted['accounting']['terminal_liquidation'] is False,'Account declaration differs')
            for key in ('days','values','daily_cash_nav'):audited.pop(key)
            reports[arm][name]=dict(audited,passed=True,slices=slices)
    return dict(passed=True,checked_paths=6,checked_slices=30,checked_training_groups=6,independent_accounts=True,
        independent_of_production_model_and_engine=True,terminal_liquidation=False,
        accounting_source_sha256=source_ind.ACCOUNTING_SHA,paths=reports)

def diagnose(expected):
    meta,orders=prepared(expected);f=frozen(expected);require(not (OUT/'results').exists(),'Results write once')
    px,op=ev.prices();engine=ev.engine();dest=OUT/'results';dest.mkdir();results={};exports={}
    for arm in ARMS:
        results[arm]={};exports[arm]={}
        for path,mode,cost in ev.PATHS:
            r,eq,trades,holds,pos=ev.run_path(engine,px,op,orders[arm],meta,mode,cost)
            r['slices']=dict(full122=r['metrics'],**ev.split_stats(engine,eq,meta));results[arm][path]=r
            exports[arm][path]=dict(DAILY=[dict(date=str(d.date()),equity=float(v)) for d,v in eq.items()],TRADES=trades,HOLDINGS=holds,DAILY_POSITIONS=pos)
            for kind,value in exports[arm][path].items():dump(dest/(arm+'_'+path+'_'+kind+'.json'),value)
    proof=audit_outcomes(px,op,orders,exports,results,meta)
    import monthly_basket_core251_independent as account
    helper=account.accounting();training=read(OUT/'selected/TRAIN_RESULTS.json')
    for arm in ARMS:
        for short,long in (('training','first88'),('early','early53'),('recent','recent35')):helper.check_metrics(proof['paths'][arm]['same_close_zero']['slices'][long],training[arm][short],arm+'/'+short)
    dump(dest/'INDEPENDENT_OUTCOME.json',proof);dump(dest/'RESULTS.json',dict(selection=f['selection'],arms=results,reused_sample=True,strict_pit=False,same_close_zero_primary=True,next_open_diagnostic_only=True,observer_promotion=False))
    gate(expected);dump(dest/'COMPLETION.json',dict(status='COMPLETE',created_utc=now(),plan_sha256=expected,frozen_sha256=sha(OUT/'selected/FROZEN.json'),paths=6,slices=30,artifacts_sha256={p.name:sha(p) for p in dest.iterdir()}))
    print(json.dumps(dict(status='COMPLETE',full_cagr={a:results[a]['same_close_zero']['metrics']['cagr'] for a in ARMS})),flush=True)

def close(expected):
    gate(expected);checkpoint(OUT/'prepared',expected);c=checkpoint(OUT/'results',expected);f=frozen(expected)
    require(c['frozen_sha256']==sha(OUT/'selected/FROZEN.json') and c['paths']==6 and c['slices']==30,'Result scope')
    review=read(RESULT_REVIEW)
    require(review['passed'] is True and review['unresolved_blockers']==[] and review['plan_sha256']==expected and review['code_sha256']==codes(),'Independent saved review')
    for directory in ('prepared','results'):
        for p in (OUT/directory).iterdir():require(review['input_sha256'].get(str(p))==sha(p),'Review coverage '+p.name)
    for p in (OUT/'PLAN.json',OUT/'selected/FROZEN.json',OUT/'selected/TRAIN_RESULTS.json'):require(review['input_sha256'].get(str(p))==sha(p),'Review selection coverage')
    old.verify(review['input_sha256']);protected=read(old.PROTECTED);old.verify(protected);require(len(protected)==21444,'Protected archive')
    before=CENTRAL.read_bytes();state=json.loads(before);require(len(state['completed'])==267 and sum(x['trials'] for x in state['completed'])==1312,'Counters before close')
    r=read(OUT/'results/RESULTS.json');require(r['selection']==f['selection'],'Frozen primary')
    pick=f['selection']['primary'];a=r['arms'][pick];m=a['same_close_zero']['metrics'];best=state['best_training_selected_candidate']
    if f['selection']['admitted'] and m['cagr']>best['cagr']:
        best=dict(directory=OUT.name,spec={'arm':pick},cagr=m['cagr'],sharpe=m['sharpe'],max_drawdown=m['max_drawdown'],next_open_5bp_cagr=a['next_open_cost']['metrics']['cagr'],selected_on_first88=True,observer_promotion=False,limitations=DESIGN['limitations'])
    dump(OUT/'COMPLETION.json',dict(status='COMPLETED_INDEPENDENTLY_VERIFIED',created_utc=now(),plan_sha256=expected,result_completion_sha256=sha(OUT/'results/COMPLETION.json'),result_review_sha256=sha(RESULT_REVIEW),new_trials=2,protected_files=21444,best_training_selected_candidate=best,target_reached=best['cagr']>=.7,observers_modified=False))
    state['completed'].append(dict(directory=OUT.name,trials=2,completed_utc=now(),primary=pick,primary_cagr=m['cagr'],primary_sharpe=m['sharpe'],new_admitted=int(f['selection']['admitted']),independent_execution_paths=6,protected_files=21444,target_reached=best['cagr']>=.7,completion=dict(path=str(OUT/'COMPLETION.json'),sha256=sha(OUT/'COMPLETION.json'))))
    state.update(best_training_selected_candidate=best,latest_training_selected_candidate=dict(directory=OUT.name,arm=pick,selection=f['selection'],full_period_cagr=m['cagr'],full_period_sharpe=m['sharpe'],selected_on_first88=True),registered_queue=[],current_status='completed_independently_verified',updated_utc=now(),last_goal_turn_classification='progress',next_step='Review monthly matured pooled clock evidence; observers remain frozen.')
    old.commit_state(before,state,'matured_clock273_close');print(json.dumps(read(OUT/'COMPLETION.json')),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('test','register','prepare','select','diagnose','close'));p.add_argument('--plan-sha256');a=p.parse_args()
    if a.phase=='test':tests()
    elif a.phase=='register':register()
    else:require(bool(a.plan_sha256),'PLAN SHA required');globals()[a.phase](a.plan_sha256)
