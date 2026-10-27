"""PLAN-first monthly EW3 log-growth comparison; no automatic observer promotion."""
from datetime import datetime,timezone
from pathlib import Path
import argparse,hashlib,io,json,unittest
import numpy as np
import pandas as pd
import monthly_ownsector_gate238_host as old
import monthly_basket_core251_host as source_host
import ptr169_rebuild_evaluate as ev

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/monthly_joint_growth271'
PRIOR=ROOT/'outputs/monthly_forecast_spacing269'
SOURCE=ROOT/'outputs/monthly_basket_core251'
CENTRAL=old.CENTRAL
LOCAL=old.LOCAL
CONTEXT=ROOT/'work/monthly_joint_growth271_context.json'
TESTS=ROOT/'work/monthly_joint_growth271_TESTS.json'
REVIEW=ROOT/'work/monthly_joint_growth271_CODE_REVIEW.json'
RESULT_REVIEW=ROOT/'work/monthly_joint_growth271_RESULT_REVIEW.json'
SOURCE_REVIEW=ROOT/'work/basket_core251_SOURCE_CONTRACT.json'
ARMS=('incumbent_local67','member_log_growth','joint_log_growth')
PREFIXES=(0,1,53,88)
CODE=tuple('work/'+name for stem in ('host','adapter','kernel','independent') for name in ('monthly_joint_growth271_'+stem+'.py','test_monthly_joint_growth271_'+stem+'.py'))
PREP_FILES={'SOURCE_FEATURES.json','INPUTS.json','ORDERS.json','META.json','AUDITS.json','PROOFS.json','SOURCE_PARITY.json','SCORES.npz','PRE_SCORES.npz'}
RESULT_FILES={a+'_'+p[0]+'_'+k+'.json' for a in ARMS for p in ev.PATHS for k in ('DAILY','TRADES','HOLDINGS','DAILY_POSITIONS')}|{'RESULTS.json','INDEPENDENT_OUTCOME.json'}
sha,read,dump,require=ev.sha,ev.read,ev.dump,ev.require
train_metrics=source_host.train_metrics
required_metrics=source_host.required_metrics

def now():return datetime.now(timezone.utc).isoformat()
def codes():return {p:sha(ROOT/p) for p in CODE}
def pins_check(pins):
    require(isinstance(pins,dict),'Input map')
    for p,h in pins.items():
        path=Path(p);path=path if path.is_absolute() else ROOT/path
        require(path.resolve()!=CENTRAL.resolve(),'Mutable central cannot be dependency')
        require(sha(path)==h,'Input changed: '+str(path))

def tests():
    require(not TESTS.exists(),'Write-once test receipt')
    stream=io.StringIO();suite=unittest.TestSuite()
    for i in (1,3,5,7):
        m=old.module('joint271_tests_'+str(i),ROOT/CODE[i]);suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(m))
    r=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite);print(stream.getvalue(),flush=True)
    require(r.wasSuccessful(),'Synthetic tests failed')
    dump(TESTS,dict(passed=True,synthetic_only=True,tests=r.testsRun,code_sha256=codes(),runtime=ev.runtime()))

def context_check():
    c=read(CONTEXT)
    require(c['schema']=='monthly_joint_growth271_context_v1' and c['frozen'] is True and c['actual_data_started'] is False,'Prospective frozen context')
    require(c['arms']==list(ARMS) and c['registered_trials']==3 and c['new_predictive_fits']==0,'Fixed arms')
    require(c['code_sha256']==codes() and c['fresh_physical_prefixes']==list(PREFIXES) and c['full_prefix_reused']==122,'Bound complete implementation/prefix scope')
    pins_check(c['input_sha256'])
    return c

def code_review():
    r=read(REVIEW)
    require(r['passed'] is True and r['unresolved_blockers']==[] and r['code_sha256']==codes(),'Independent code review')
    require(r['context_sha256']==sha(CONTEXT) and r['actual_returns_computed'] is False,'Prospective review scope')
    pins_check(r['input_sha256']);return r

def initial_state(state):
    require(len(state['completed'])==265 and sum(x['trials'] for x in state['completed'])==1306,'Prior counters')
    require(state['completed'][-1]['directory']==PRIOR.name,'Immediate closed269 predecessor')
    r=state['completed'][-1]['completion']
    require(Path(r['path'])==PRIOR/'COMPLETION.json' and r['sha256']==sha(PRIOR/'COMPLETION.json'),'Predecessor closure reference')

def register():
    require(not OUT.exists(),'Fresh experiment directory')
    c=context_check();r=code_review();t=read(TESTS)
    require(t['passed'] is True and t['synthetic_only'] is True and t['code_sha256']==codes() and t['runtime']==ev.runtime(),'Passing current synthetic suite')
    before=CENTRAL.read_bytes();s=json.loads(before);initial_state(s);require(s['registered_queue']==[],'No pending research')
    done=read(PRIOR/'COMPLETION.json')
    require(done['status']=='COMPLETED_INDEPENDENTLY_VERIFIED' and done['protected_files']==21444 and done['observers_modified'] is False,'Closed predecessor')
    require(done['plan_sha256']==sha(PRIOR/'PLAN.json') and done['target_reached'] is False,'Predecessor PLAN/target')
    pins={}
    for parent in (PRIOR,SOURCE):
        for name,h in read(parent/'PLAN.json')['input_sha256'].items():old.bind(pins,Path(name),h)
        for name in ('PLAN.json','COMPLETION.json'):old.bind(pins,parent/name)
    for paper in (c,r):
        for name,h in paper['input_sha256'].items():old.bind(pins,Path(name) if Path(name).is_absolute() else ROOT/name,h)
    for path in [ROOT/p for p in CODE]+[CONTEXT,REVIEW,TESTS,SOURCE_REVIEW,source_host.CALENDAR,old.PROTECTED,ROOT/'work/monthly_basket_core251_host.py',ROOT/'work/monthly_forecast_spacing269_RESULT_REVIEW.json']:
        old.bind(pins,path)
    old.verify(pins);old.verify(read(old.PROTECTED));require(len(read(old.PROTECTED))==21444,'Complete protected archive')
    source_host.source_review_check(read(SOURCE_REVIEW))
    OUT.mkdir();(OUT/'STATE_BEFORE.json').write_bytes(before)
    dump(OUT/'PLAN.json',dict(schema='monthly_joint_growth271_plan_v1',created_utc=now(),context_sha256=sha(CONTEXT),context=c,runtime=ev.runtime(),
      input_sha256=pins,raw_source_hashes=read(LOCAL/'PLAN.json')['raw_source_hashes'],prior_central_sha256=hashlib.sha256(before).hexdigest(),
      actual_features_before_registration=0,actual_orders_before_registration=0,actual_returns_before_registration=0,new_predictive_fits=0,protected_files=21444))
    h=sha(OUT/'PLAN.json');s.update(current=OUT.name,current_status='registered_before_features',registered_queue=[dict(directory=OUT.name,plan_sha256=h)],updated_utc=now(),last_goal_turn_classification='progress',next_step='Verify source and incumbent before joint-growth decisions; first88-only ZERO selection.')
    old.commit_state(before,s,'joint_growth271_register');print(json.dumps(dict(status='REGISTERED',plan_sha256=h)),flush=True)

def gate(expected):
    require(sha(OUT/'PLAN.json')==expected,'Explicit immutable PLAN digest')
    p=read(OUT/'PLAN.json');require(p['context_sha256']==sha(CONTEXT) and p['context']==context_check() and p['runtime']==ev.runtime(),'Frozen context/runtime')
    old.verify(p['input_sha256']);code_review()
    require(sha(OUT/'STATE_BEFORE.json')==p['prior_central_sha256'],'Pre-state changed')
    s=read(CENTRAL);initial_state(s)
    require(s['current']==OUT.name and s['registered_queue']==[dict(directory=OUT.name,plan_sha256=expected)],'Active registration')
    return p

def exact(value,other,label):
    require(type(value) is type(other),label+' type')
    if isinstance(value,dict):
        require(set(value)==set(other),label+' keys')
        for k in value:exact(value[k],other[k],label+'/'+k)
    elif isinstance(value,list):
        require(len(value)==len(other),label+' length')
        for i,(a,b) in enumerate(zip(value,other)):exact(a,b,label+'/'+str(i))
    elif isinstance(value,float):require(value.hex()==other.hex(),label+' binary64')
    else:require(value==other,label+' value')

def verify_decisions(events,orders,audits,meta):
    import monthly_joint_growth271_independent as ind
    require(set(orders)==set(audits)==set(ARMS),'Three decision roles')
    checks={}
    for arm in ARMS:
        require(len(orders[arm])==len(events),'Full ordered decision axis')
        choices=[]
        for i,row in enumerate(orders[arm]):
            require(row['date']==row['signal_date']==meta['dates'][i] and row['end']==meta['ends'][i],'Order clocks')
            names=events[i]['symbols'];chosen=[names.index(s) for s in row['symbols']]
            require(len(chosen)==len(set(chosen)) and len(chosen) in (0,2,3),'EW3/cardinality')
            choices.append(chosen)
        checks[arm]=ind.verify_selection(events,arm,choices,audits[arm],n=len(events))
        require(checks[arm]['passed'] is True and checks[arm]['checked_events']==len(events),'Complete independent pure selection')
    return checks

def prepare(expected):
    plan=gate(expected);require(not (OUT/'prepared').exists(),'Write-once preparation')
    import monthly_joint_growth271_adapter as adapter
    import monthly_joint_growth271_kernel as kernel
    study=old.local();meta,arrays,px,op=study.load()
    require(len(meta['dates'])==122 and arrays['baseline'].shape==(122,49) and len(arrays)==27,'Original axes')
    before=old.array_snapshot(arrays);panel,info=study.arch.old.panel_load(meta['etfs']);raw=study.raw_source_hashes(info)
    require(raw==plan['raw_source_hashes'],'Original64 source identity');old.verify(raw)
    parity=study.verify_panel_parity(panel,px,op);panel={'adj':panel['adj']}
    calendar=read(source_host.CALENDAR)['sessions'];review=read(SOURCE_REVIEW);source_host.source_review_check(review)
    expected_control=read(LOCAL/'local_linear_ORDERS.json')
    print(json.dumps(dict(stage='271_full_source_and_incumbent_before_selection')),flush=True)
    result=adapter.prepare(study,arrays,meta,panel,calendar,review,122,expected_incumbent_orders=expected_control)
    exact(result['orders'][ARMS[0]],expected_control,'Frozen incumbent')
    independent=verify_decisions(result['events'],result['orders'],result['audits'],meta)
    # All prefixes invoke the pure selector again; new features are already bounded source windows.
    cached=[]
    for n in range(123):
        for arm in ARMS:
            ch,audit=kernel.select(result['events'],arm,n=n)
            exact(ch,result['choices'][arm][:n],'Cached prefix choices')
            exact(audit['events'],result['audits'][arm]['events'][:n],'Cached prefix audit')
        cached.append(n)
    fresh=[]
    for n in PREFIXES:
        print(json.dumps(dict(stage='271_fresh_physical_prefix',events=n)),flush=True)
        aa=old.cache_prefix(arrays,n);mm=old.meta_prefix(meta,n)
        pp={'adj':panel['adj'].loc[:pd.Timestamp(meta['dates'][n-1])].copy() if n else panel['adj'].iloc[:0].copy()}
        cc=old.feature_prefix_calendar(calendar,meta,n)
        q=adapter.prepare(study,aa,mm,pp,cc,review,n,expected_incumbent_orders=expected_control[:n])
        for key in ('source_features','events'):
            exact(q[key],result[key][:n],'Physical '+key)
        for arm in ARMS:
            exact(q['orders'][arm],result['orders'][arm][:n],'Physical orders')
            exact(q['choices'][arm],result['choices'][arm][:n],'Physical choices')
            exact(q['audits'][arm]['events'],result['audits'][arm]['events'][:n],'Physical audit')
            np.testing.assert_array_equal(q['scores'][arm],result['scores'][arm][:n]);np.testing.assert_array_equal(q['pre_scores'][arm],result['pre_scores'][arm][:n])
        check=verify_decisions(q['events'],q['orders'],q['audits'],mm)
        fresh.append(dict(prefix=n,source_proof=q['source_proof'],incumbent_proof=q['incumbent_proof'],selection_proofs=check,exact=True,work=q['work']))
    require(old.array_snapshot(arrays)==before,'Original arrays mutated');gate(expected)
    dest=OUT/'prepared';dest.mkdir()
    for filename,value in [('SOURCE_FEATURES.json',result['source_features']),('INPUTS.json',result['events']),('ORDERS.json',result['orders']),('META.json',meta),('AUDITS.json',result['audits']),
      ('PROOFS.json',dict(source_proof=result['source_proof'],incumbent_proof=result['incumbent_proof'],selection_proofs=independent,cached_prefixes=cached,fresh_physical_prefixes=fresh,full_prefix_reused=122,work=result['work'],original_arrays_unchanged=True)),
      ('SOURCE_PARITY.json',dict(raw_source_hashes=raw,execution_prices=parity,original_array_fingerprints=old.array_fingerprints(arrays),source_review_sha256=sha(SOURCE_REVIEW)))]:dump(dest/filename,value)
    np.savez_compressed(dest/'SCORES.npz',**result['scores']);np.savez_compressed(dest/'PRE_SCORES.npz',**result['pre_scores'])
    dump(dest/'COMPLETION.json',dict(status='COMPLETE',created_utc=now(),plan_sha256=expected,events=122,new_returns_computed=False,new_predictive_fits=0,artifacts_sha256={p.name:sha(p) for p in dest.iterdir()}))
    print(json.dumps(dict(status='PREPARED',completion_sha256=sha(dest/'COMPLETION.json'))),flush=True)

def prepared(expected):
    gate(expected);c=read(OUT/'prepared/COMPLETION.json')
    require(c['status']=='COMPLETE' and c['plan_sha256']==expected and c['events']==122 and c['new_returns_computed'] is False and set(c['artifacts_sha256'])==PREP_FILES,'Complete prepared inventory')
    for name,h in c['artifacts_sha256'].items():require(sha(OUT/'prepared'/name)==h,'Changed preparation '+name)
    proof=read(OUT/'prepared/PROOFS.json')
    require(proof['cached_prefixes']==list(range(123)) and [x['prefix'] for x in proof['fresh_physical_prefixes']]==list(PREFIXES) and proof['full_prefix_reused']==122 and proof['original_arrays_unchanged'] is True,'Prefix/array preservation')
    import monthly_basket_core251_independent as source_ind
    source_host.source_gate_check(proof['source_proof'],122,source_ind);source_host.incumbent_gate_check(proof['incumbent_proof'],122,source_ind)
    for p in proof['fresh_physical_prefixes']:
        require(p['exact'] is True,'Exact physical prefix');source_host.source_gate_check(p['source_proof'],p['prefix'],source_ind);source_host.incumbent_gate_check(p['incumbent_proof'],p['prefix'],source_ind)
    meta=read(OUT/'prepared/META.json');orders=read(OUT/'prepared/ORDERS.json')
    checks=verify_decisions(read(OUT/'prepared/INPUTS.json'),orders,read(OUT/'prepared/AUDITS.json'),meta)
    exact(checks,proof['selection_proofs'],'Saved pure selection proof before any outcome')
    return meta,orders

def select_first88(metrics,orders):
    require(set(metrics)==set(orders)==set(ARMS),'Fixed roles')
    required_metrics(metrics[ARMS[0]],True);guards={};changes={};accepted=[];undefined={}
    require(all(len(orders[a])>=88 for a in ARMS),'First88 orders')
    for a in ARMS[1:]:
        undefined[a]=required_metrics(metrics[a],False)
        changes[a]=[i for i in range(88) if set(orders[a][i]['symbols'])!=set(orders[ARMS[0]][i]['symbols'])]
        guards[a]=dict(close='same_close_zero' not in undefined[a] and old.guard(metrics[a]['same_close_zero'],metrics[ARMS[0]]['same_close_zero'],.005),
          open='next_open_zero' not in undefined[a] and old.guard(metrics[a]['next_open_zero'],metrics[ARMS[0]]['next_open_zero'],0.),
          support=len(changes[a])>=3 and any(i<53 for i in changes[a]) and any(i>=53 for i in changes[a]))
        if all(guards[a].values()):accepted.append(a)
    def key(a):return metrics[a]['same_close_zero']['training']['cagr'],metrics[a]['same_close_zero']['training']['sharpe'],a=='member_log_growth'
    pick=max(accepted,key=key) if accepted else ARMS[0]
    return dict(primary=pick,admitted=bool(accepted),admissible_arms=accepted,guards=guards,changed_months=changes,undefined_sharpe=undefined,first88_only=True,later34_used=False,cost_used=False)

def select(expected):
    meta,orders=prepared(expected);require(not (OUT/'selected').exists(),'Write-once first88 selection');px,op=ev.prices();engine=ev.engine();metrics={}
    for arm in ARMS:
        metrics[arm]={}
        for path,mode,cost in ev.PATHS[:2]:
            _,eq,_,_=engine.replay(px,op,orders[arm][:88],cost=cost,execution=mode,start=pd.Timestamp(meta['dates'][0]),end=pd.Timestamp(meta['ends'][87]))
            require(str(eq.index[-1].date())==meta['ends'][87],'Training cutoff');metrics[arm][path]=train_metrics(engine,eq,meta)
    choice=select_first88(metrics,orders);gate(expected);dest=OUT/'selected';dest.mkdir();dump(dest/'TRAIN_RESULTS.json',metrics)
    dump(dest/'FROZEN.json',dict(plan_sha256=expected,created_utc=now(),prepared_sha256=sha(OUT/'prepared/COMPLETION.json'),orders_sha256=sha(OUT/'prepared/ORDERS.json'),training_sha256=sha(dest/'TRAIN_RESULTS.json'),selection=choice))
    print(json.dumps(choice),flush=True)

def frozen(expected,orders):
    f=read(OUT/'selected/FROZEN.json');require(f['plan_sha256']==expected and f['prepared_sha256']==sha(OUT/'prepared/COMPLETION.json') and f['orders_sha256']==sha(OUT/'prepared/ORDERS.json') and f['training_sha256']==sha(OUT/'selected/TRAIN_RESULTS.json'),'Frozen selection bindings')
    require(f['selection']==select_first88(read(OUT/'selected/TRAIN_RESULTS.json'),orders),'Frozen selection unchanged');return f


def audit_outcomes(px,opening,orders,exports,results,meta):
    """Three arms, three paths and five slices; reuse only independent accounting."""
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
    return dict(passed=True,checked_paths=9,checked_slices=45,checked_training_groups=18,independent_accounts=True,
        independent_of_production_model_and_engine=True,terminal_liquidation=False,
        accounting_source_sha256=source_ind.ACCOUNTING_SHA,paths=reports)

def zero_cost_check(exports,results):
    for a in ARMS:
        for path in ('same_close_zero','next_open_zero'):
            for row in exports[a][path]['TRADES']:require(type(row['cost']) in (int,float) and row['cost']==0.,'ZERO fill fee')
            for row in exports[a][path]['HOLDINGS']:require(type(row['fee']) in (int,float) and row['fee']==0.,'ZERO period fee')
            for m in (results[a][path]['metrics'],results[a][path]['slices']['full122']):
                require(type(m['cost_per_side']) in (int,float) and m['cost_per_side']==0. and type(m['fees_paid']) in (int,float) and m['fees_paid']==0.,'ZERO configured and total fees')

def train_check(training,audit):
    import monthly_basket_core251_independent as source_ind
    helper=source_ind.accounting();require(set(training)==set(ARMS),'Training roles')
    for a in ARMS:
        require(set(training[a])=={'same_close_zero','next_open_zero'},'Only ZERO training')
        for path,groups in training[a].items():
            require(set(groups)=={'training','early','recent'},'Training split roles')
            for short,long in [('training','first88'),('early','early53'),('recent','recent35')]:helper.check_metrics(audit['paths'][a][path]['slices'][long],groups[short],a+'/'+path+'/'+short)

def outcome_check(check):
    require(check['passed'] is True and check['checked_paths']==9 and check['checked_slices']==45 and check['checked_training_groups']==18 and check['independent_accounts'] is True and check['independent_of_production_model_and_engine'] is True and check['terminal_liquidation'] is False,'Independent accounting scope')
    require(set(check['paths'])==set(ARMS),'Accounting roles')
    for paths in check['paths'].values():
        require(set(paths)=={x[0] for x in ev.PATHS},'Accounting paths')
        for proof in paths.values():require(proof['passed'] is True and proof['monthly_executions']==122 and set(proof['slices'])=={'full122','first88','early53','recent35','later34'},'Full account slices')

def diagnose(expected):
    meta,orders=prepared(expected);f=frozen(expected,orders);require(not (OUT/'results').exists(),'Write-once diagnostics')
    px,op=ev.prices();engine=ev.engine();dest=OUT/'results';dest.mkdir();results={};exports={}
    for a in ARMS:
        results[a]={};exports[a]={}
        for path,mode,cost in ev.PATHS:
            r,eq,fills,holds,pos=ev.run_path(engine,px,op,orders[a],meta,mode,cost)
            r['slices']=dict(full122=r['metrics'],**ev.split_stats(engine,eq,meta));results[a][path]=r
            data=dict(DAILY=[dict(date=str(d.date()),equity=float(v)) for d,v in eq.items()],TRADES=fills,HOLDINGS=holds,DAILY_POSITIONS=pos);exports[a][path]=data
            for k,v in data.items():dump(dest/(a+'_'+path+'_'+k+'.json'),v)
    zero_cost_check(exports,results);check=audit_outcomes(px,op,orders,exports,results,meta);outcome_check(check);train_check(read(OUT/'selected/TRAIN_RESULTS.json'),check)
    dump(dest/'INDEPENDENT_OUTCOME.json',check)
    dump(dest/'RESULTS.json',dict(selection=f['selection'],arms=results,reused_sample=True,strict_pit=False,zero_cost_selection=True,next_open_cost_diagnostic_only=True,observer_promotion=False))
    gate(expected);dump(dest/'COMPLETION.json',dict(status='COMPLETE',plan_sha256=expected,created_utc=now(),frozen_sha256=sha(OUT/'selected/FROZEN.json'),paths=9,slices=45,artifacts_sha256={p.name:sha(p) for p in dest.iterdir()}))
    print(json.dumps(dict(status='DIAGNOSTICS_COMPLETE',full_cagr={a:results[a]['same_close_zero']['metrics']['cagr'] for a in ARMS},completion_sha256=sha(dest/'COMPLETION.json'))),flush=True)

def phase_terminals_check(expected):
    r=read(OUT/'PHASE_TERMINALS.json');require(r['plan_sha256']==expected and set(r['phases'])=={'prepare','select','diagnose'},'Owned actual phase terminals')
    for phase,name in [('prepare','prepared/COMPLETION.json'),('select','selected/FROZEN.json'),('diagnose','results/COMPLETION.json')]:
        x=r['phases'][phase]
        require(type(x['exit_code']) is int and x['exit_code']==0 and type(x['chunk_id']) is str and bool(x['chunk_id']),'Clean actual terminal')
        require(x['session_id'] is None or (type(x['session_id']) is int and x['session_id']>0),'Session identity')
        path=Path(x['artifact_path']);path=path if path.is_absolute() else ROOT/path
        require(path==OUT/name and x['artifact_sha256']==sha(path),'Terminal binds final artifact')

def result_review_check(expected):
    phase_terminals_check(expected);r=read(RESULT_REVIEW)
    require(r['passed'] is True and r['unresolved_blockers']==[] and r['plan_sha256']==expected,'Independent saved review')
    for key,path in [('context_sha256',CONTEXT),('prepared_completion_sha256',OUT/'prepared/COMPLETION.json'),('results_completion_sha256',OUT/'results/COMPLETION.json')]:require(r[key]==sha(path),'Saved review bindings')
    pins_check(r['input_sha256']);pinmap={str(Path(k) if Path(k).is_absolute() else ROOT/k):v for k,v in r['input_sha256'].items()}
    need=[OUT/'PLAN.json',CONTEXT,OUT/'PHASE_TERMINALS.json',OUT/'prepared/COMPLETION.json',OUT/'selected/FROZEN.json',OUT/'selected/TRAIN_RESULTS.json',OUT/'results/COMPLETION.json']
    need += [OUT/'prepared'/x for x in PREP_FILES]+[OUT/'results'/x for x in RESULT_FILES]
    require(all(pinmap.get(str(path))==sha(path) for path in need),'Complete saved artifact review coverage')


def close(expected):
    require(not (OUT/'COMPLETION.json').exists(),'Write-once final closure')
    meta,orders=prepared(expected);f=frozen(expected,orders);c=read(OUT/'results/COMPLETION.json')
    require(c['status']=='COMPLETE' and c['plan_sha256']==expected and c['frozen_sha256']==sha(OUT/'selected/FROZEN.json') and c['paths']==9 and c['slices']==45 and set(c['artifacts_sha256'])==RESULT_FILES,'Result inventory')
    for name,h in c['artifacts_sha256'].items():require(sha(OUT/'results'/name)==h,'Changed result export')
    result=read(OUT/'results/RESULTS.json');require(result['selection']==f['selection'],'Frozen result selection')
    accounting=read(OUT/'results/INDEPENDENT_OUTCOME.json');outcome_check(accounting);train_check(read(OUT/'selected/TRAIN_RESULTS.json'),accounting)
    verify_decisions(read(OUT/'prepared/INPUTS.json'),orders,read(OUT/'prepared/AUDITS.json'),meta)
    result_review_check(expected);protected=read(old.PROTECTED);require(len(protected)==21444,'Full protected scope');old.verify(protected)
    before=CENTRAL.read_bytes();s=json.loads(before);initial_state(s)
    require(s['current']==OUT.name and s['registered_queue']==[dict(directory=OUT.name,plan_sha256=expected)],'Close active registration')
    pick=f['selection']['primary'];arm=result['arms'][pick];m=arm['same_close_zero']['metrics'];best=s['best_training_selected_candidate']
    if f['selection']['admitted'] and m['cagr']>best['cagr']:
        best=dict(directory=OUT.name,spec=dict(arm=pick),cagr=m['cagr'],sharpe=m['sharpe'],max_drawdown=m['max_drawdown'],next_open_5bp_cagr=arm['next_open_cost']['metrics']['cagr'],selected_on_first88=True,observer_promotion=False,limitations=read(CONTEXT)['limitations'])
    done=dict(status='COMPLETED_INDEPENDENTLY_VERIFIED',created_utc=now(),plan_sha256=expected,result_completion_sha256=sha(OUT/'results/COMPLETION.json'),result_review_sha256=sha(RESULT_REVIEW),new_trials=3,new_predictive_fits=0,protected_files=21444,best_training_selected_candidate=best,target_reached=best['cagr']>=.7,observers_modified=False)
    dump(OUT/'COMPLETION.json',done)
    s['completed'].append(dict(directory=OUT.name,trials=3,completed_utc=now(),primary=pick,primary_cagr=m['cagr'],primary_sharpe=m['sharpe'],new_admitted=int(f['selection']['admitted']),independent_execution_paths=9,protected_files=21444,target_reached=best['cagr']>=.7,completion=dict(path=str(OUT/'COMPLETION.json'),sha256=sha(OUT/'COMPLETION.json'))))
    require(len(s['completed'])==266 and sum(x['trials'] for x in s['completed'])==1309,'Counter advance')
    s.update(best_training_selected_candidate=best,latest_training_selected_candidate=dict(directory=OUT.name,arm=pick,selection=f['selection'],full_period_cagr=m['cagr'],full_period_sharpe=m['sharpe'],selected_on_first88=True),registered_queue=[],current_status='completed_independently_verified',updated_utc=now(),last_goal_turn_classification='progress',next_step='Assess long-momentum member versus joint geometric objectives; frozen observers unchanged.')
    old.commit_state(before,s,'joint_growth271_close');print(json.dumps(done),flush=True)


def prepare_recorded(expected):
    try:return prepare(expected)
    except Exception as exc:
        if (OUT/'PLAN.json').is_file() and sha(OUT/'PLAN.json')==expected and not (OUT/'PREPARE_FAILURE.json').exists():
            dump(OUT/'PREPARE_FAILURE.json',dict(status='FAILED',created_utc=now(),plan_sha256=expected,error_type=type(exc).__name__,error=str(exc),source_failure_not_relaxed=True,successful_preparation_claim=False,observers_modified=False))
        raise


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('test','register','prepare','select','diagnose','close'));p.add_argument('--plan-sha256');a=p.parse_args()
    if a.phase=='test':tests()
    elif a.phase=='register':register()
    elif a.phase=='prepare':require(bool(a.plan_sha256),'Explicit PLAN SHA required');prepare_recorded(a.plan_sha256)
    else:
        require(bool(a.plan_sha256),'Explicit PLAN SHA required');globals()[a.phase](a.plan_sha256)

# All actual work requires the separately frozen context, tests and independent review.
if __name__=='__main__':main()
