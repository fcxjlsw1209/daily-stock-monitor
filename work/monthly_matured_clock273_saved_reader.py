"""Read-only273 saved evidence review. No price reads, fits or portfolio replay.

Saved source/forecast NPZs are decoded. Cohort clocks, weights, fit equations,
stored-beta predictions, ranks, pre-score deltas and supplied selections are
checked. Canonical/SVD fitting happened in bound preparation receipts; this
reader does not repeat those fits. Same author as the independent model module.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import math
from pathlib import Path
import platform
import numpy as np
import monthly_monotone_tree266_saved_reader as shared
from monthly_boundary_pair267_saved_reader import Evidence
import monthly_matured_clock273_independent as oracle
import monthly_selector_scope268_independent as selector

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/monthly_matured_clock273'
PRIOR=ROOT/'outputs/monthly_roa_increment272'
LOCAL=ROOT/'outputs/monthly_local_linear70'
REBUILD=ROOT/'outputs/monthly_ptr_pool_source169/fixed_incumbent_rebuild/control'
REFERENCE=ROOT/'outputs/monthly_forecast_spacing269/prepared'
ARMS=('incumbent_local67','monthly_pooled_clock')
PATHS=('same_close_zero','next_open_zero','next_open_cost')
KINDS=('DAILY','TRADES','HOLDINGS','DAILY_POSITIONS')
SLICES=('full122','first88','early53','recent35','later34')
CODE=tuple('work/'+name for stem in ('host','kernel','inputs','independent','saved_reader')
           for name in ('monthly_matured_clock273_'+stem+'.py','test_monthly_matured_clock273_'+stem+'.py'))
PREP={'FORECASTS.npz','APPENDED_INPUTS.npz','SCORES.npz','PRE_SCORES.npz','META.json','ORDERS.json',
      'APPEND_AUDIT.json','MODEL_AUDIT.json','DELTA_AUDIT.json','PREP_AUDIT.json','SOURCE_PARITY.json'}
RESULT={'RESULTS.json','INDEPENDENT_OUTCOME.json'}|{a+'_'+p+'_'+k+'.json' for a in ARMS for p in PATHS for k in KINDS}
FIXED={'work/monthly_monotone_tree266_saved_reader.py':'eb117fa95c4d7940b96797b1a5a2bd08320b64f3663660a9569915f624e56a8d',
 'work/monthly_boundary_pair267_saved_reader.py':'29eb059828074123afdf6365c914b5ccb004fa3e0b1636a77dbef4050f73a06d',
 'work/monthly_selector_scope268_independent.py':'12ec572b1378bbf16fc698890b689523904dd4b7858060a5c4d35619d348d8d9',
 'work/ptr169_results_independent.py':'6264571ad6ddacbc408449180c74a32b37659757c9bdfc5ce0e2672467607d59',
 'work/monthly_gain_concentration265_saved_reader.py':'8990f2e15e423808546da661ae36717a6dd83dd71fee377f918d55599a2decd8'}
need,exact=shared.need,shared.exact


def dependency_files():
    """Return the frozen helper sources required by prospective registration."""
    files=[]
    for name,want in FIXED.items():
        path=ROOT/name
        need(shared.digest(path)==want,'Pinned saved-reader dependency '+name)
        files.append(path)
    return tuple(files)


def arrays(ev,path,want=None):
    ev.pin(path,want)
    with np.load(path,allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}


def fingerprint(value):
    return dict(dtype=value.dtype.str,shape=list(value.shape),sha256=hashlib.sha256(value.tobytes()).hexdigest())


def saved_selection(training):
    need(type(training) is dict and set(training)==set(ARMS),'Two first88 arms')
    fields=(('training','cagr'),('training','sharpe'),('training','max_drawdown'),('early','cagr'),('recent','cagr'))
    for groups in training.values():
        need(type(groups) is dict and set(groups)=={'training','early','recent'},'No later/cost selection groups')
        for group,key in fields:need(shared.finite(groups[group][key]),'Finite first88 metric')
    c,b=([training[arm][g][k] for g,k in fields] for arm in (ARMS[1],ARMS[0]))
    passed=bool(c[0]>b[0]+1e-12 and all(x>=y-t for x,y,t in zip(c[1:],b[1:],(.05,.03,.025,.025))))
    return dict(primary=ARMS[int(passed)],admitted=passed,first88_only=True,later34_used=False,cost_used=False)


def _record_check(record,inp,kind,queries,cutoff):
    """No solve: rebuild the saved equation and apply its stored coefficient."""
    need(type(record) is dict and set(record)==set(oracle.RECORD_KEYS),'Exact fit record')
    lower=oracle.lower_bound(cutoff);ids=np.flatnonzero(inp['valid']&(inp['td']>lower)&(inp['te']<=cutoff))
    qi=[i*inp['width']+int(j) for i in queries for j in np.flatnonzero(inp['use'][i])]
    months=len(np.unique(inp['td'][ids].astype('datetime64[M]')))
    status='cold_start' if months<18 else 'fit' if qi else 'no_prediction_rows'
    need(kind!='monthly' or status=='fit','Required monthly mature support')
    expected=dict(kind=kind,query_indices=queries,cutoff=str(cutoff),lower_bound=str(lower),training_row_ids=ids.tolist(),
        query_row_ids=qi,training_months=months,status=status,signal_dates=inp['td'][ids].astype(str).tolist(),
        label_end_dates=inp['te'][ids].astype(str).tolist(),symbols=[inp['names'][j] for j in ids])
    for key,value in expected.items():exact(record[key],value,'Saved clock/cohort '+key)
    if kind=='annual':
        for i in queries:
            wanted=inp['arrays']['eligible'][i]&np.isfinite(inp['Q'][i*inp['width']:(i+1)*inp['width']]).all(axis=1) if months>=18 else np.zeros(inp['width'],bool)
            oracle.exact_array(inp['arrays']['forecast_available'][i],wanted,'Annual availability support')
    if status!='fit':
        for key in ('mean','scale','beta','intercept','gram','rhs','weights_sha256'):need(record[key] is None,'Skipped model contains numeric fit')
        exact(record['weights'],[],'Skipped weights');exact(record['raw_predictions'],[],'Skipped predictions')
        return qi,np.empty(0,np.float64)
    weights=oracle.month_weights(inp['td'][ids],cutoff)
    exact(record['weights'],weights.tolist(),'Exact saved recency weights')
    need(record['weights_sha256']==hashlib.sha256(weights.tobytes()).hexdigest(),'Weight byte digest')
    X=inp['X'][ids];y=inp['targets']['log_excess'][ids];Q=inp['Q'][qi]
    need(np.isfinite(Q).all() and (np.abs(Q)<=.5+1e-12).all() and (np.abs(X)<=.5+1e-12).all(),'Saved plain10 bounded')
    with np.errstate(over='raise',invalid='raise',divide='raise'):
        mean=np.average(X,axis=0,weights=weights);centered=X-mean
        scale=np.sqrt(np.average(centered*centered,axis=0,weights=weights));scale=np.where(scale>1e-12,scale,1.)
        Z=centered/scale;intercept=float(np.average(y,weights=weights))
        gram=np.einsum('ni,n,nj->ij',Z,weights,Z,optimize=False)+10.*np.eye(10)
        rhs=np.einsum('ni,n,n->i',Z,weights,y-intercept,optimize=False)
        for key,value in [('mean',mean.tolist()),('scale',scale.tolist()),('intercept',intercept),('gram',gram.tolist()),('rhs',rhs.tolist())]:
            exact(record[key],value,'Saved equation '+key)
        beta=np.asarray(record['beta']);need(beta.shape==(10,) and beta.dtype==np.dtype('float64') and np.isfinite(beta).all(),'Finite saved coefficient')
        pred=np.einsum('ni,i->n',(Q-mean)/scale,beta,optimize=False)+intercept
    exact(record['raw_predictions'],pred.tolist(),'Stored coefficient to prediction exact')
    return qi,pred


def model_counts(inp,annual,monthly):
    c=dict.fromkeys(oracle.COUNT_FIELDS,0)
    c.update(events=inp['n'],width=inp['width'],training_rows=inp.get('stop',0),valid_training_rows=int(inp.get('valid',np.empty(0,bool)).sum()),annual_groups=len(annual))
    for records,prefix in ((annual,'annual'),(monthly,'monthly')):
        c[prefix+'_ridge_fits']=sum(r['status']=='fit' for r in records)
        c[prefix+'_prediction_rows']=sum(len(r['query_row_ids']) for r in records if r['status']=='fit')
    c['actual_model_fits']=c['annual_ridge_fits']+c['monthly_ridge_fits'];return c


def model_receipt(record,kind,inp,counts,delta_audit=None):
    n,w=inp['n'],inp['width'];work=oracle._work()
    if kind in ('control','models'):
        fits=counts['annual_ridge_fits']+(counts['monthly_ridge_fits'] if kind=='models' else 0)
        rows=counts['annual_prediction_rows']+(counts['monthly_prediction_rows'] if kind=='models' else 0)
        work.update(training_rows_scanned=2*counts['training_rows'],canonical_solves=fits,svd_solves=fits,
            canonical_prediction_rows=rows,svd_prediction_rows=rows,rank_evaluations=int(inp['use'].sum())*(2 if kind=='models' else 1),output_cells_compared=n*w*(6 if kind=='models' else 3))
        wanted=oracle._receipt(kind,n,w,work,oracle.MODEL_FLAGS)
    else:
        work.update(rank_evaluations=2*int(inp['use'].sum()),delta_evaluations=sum(map(len,delta_audit['active_indices'])),output_cells_compared=8*n*w)
        wanted=oracle._receipt('delta',n,w,work,oracle.DELTA_FLAGS,model_fit_provenance_checked=False,original_pre_score_addition_checked=False)
    exact(record,wanted,'Exact bound independent '+kind+' scope and actual work')


def saved_models(train,meta,original,forecasts,ma,delta_audit,evidence):
    inp=oracle._header(train,meta,original,None);n,w=inp['n'],inp['width']
    need(set(ma)=={'control','candidate','audit'} and set(forecasts)=={'delta'}|{r+'_'+k for r in ('control','candidate') for k in ('raw','available','rank')},'Saved model members')
    annual,monthly=ma['control']['fits'],ma['candidate']['fits']
    expected_raw=original['raw_source_log'].copy()
    if inp['formal'].any():inp['valid']=oracle._usable(inp)
    years=sorted(set(str(d)[:4] for d in inp['dates'][inp['formal']]))
    need(len(annual)==len(years),'One saved annual record per consumed year')
    for row,year in zip(annual,years):
        queries=[i for i,d in enumerate(inp['dates']) if inp['formal'][i] and str(d).startswith(year)]
        ids,pred=_record_check(row,inp,'annual',queries,np.datetime64(str(int(year)-1)+'-12-31'))
        if row['status']=='fit':oracle.exact_array(pred,expected_raw.reshape(-1)[ids],'Annual calibration source predictions')
    active=np.flatnonzero(inp['active']);need(len(monthly)==len(active),'Only active monthly fits')
    for row,i in zip(monthly,active):
        ids,pred=_record_check(row,inp,'monthly',[int(i)],inp['dates'][i]);expected_raw.reshape(-1)[ids]=pred
    oracle.exact_array(forecasts['candidate_raw'],expected_raw,'Saved complete candidate forecast')
    counts=model_counts(inp,annual,monthly);exact(ma['audit']['counts'],counts,'Model counts')
    # Build declarations without any solve. Coefficients remain bound to the
    # separately executed preparation proof, not certified anew by this reader.
    targets=inp.get('targets',{k:np.empty(0,np.float64) for k in oracle.TARGETS});good=inp.get('good',np.empty(0,bool))
    labels=dict(target_keys=list(oracle.TARGETS),target_sha256={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in targets.items()},common_good_sha256=hashlib.sha256(good.tobytes()).hexdigest(),labels_unchanged=True,common_good_unchanged=True,original_row_order_preserved=True,recentered=False,raw_target_reconstructed=False)
    scope=dict(events=n,width=w,consumed_training_rows=inp.get('stop',0),training_months=inp.get('training_months',0),latest_needed_cutoff=str(inp['max_cutoff']) if 'max_cutoff' in inp else None,future_training_numeric_accessed=False,future_queries_accessed=False)
    wanted=dict(schema='monthly_matured_clock273_model_audit_v1',counts=counts,annual_control_exact=True,original_ranking_population=True,paired_ready_preserved=True,local_models_reused=True,other_targets_not_fitted=True,full_clock_change=True,strict_pit=False,computes_returns=False,control_gate=evidence['control_gate'],label_preservation=labels,source_scope=scope)
    exact(ma['audit'],wanted,'Full saved model audit');exact(evidence['producer_counts'],counts,'Producer counts duplicate')
    model_receipt(evidence['control_gate'],'control',inp,counts);model_receipt(evidence['model'],'models',inp,counts)
    bundle=dict(n=n,width=w,audit=ma['audit'],**{r:dict(raw=forecasts[r+'_raw'],available=forecasts[r+'_available'],rank=forecasts[r+'_rank'],audit=ma[r]) for r in ('control','candidate')})
    check=oracle.verify_delta(original,meta['monthly'],bundle,forecasts['candidate_rank'],forecasts['delta'],delta_audit)
    exact(evidence['delta'],check,'Saved delta proof');model_receipt(evidence['delta'],'delta',inp,counts,delta_audit)
    return inp,bundle,dict(annual_records=len(annual),monthly_records=len(monthly),stored_prediction_equations_exact=True,coefficient_solutions_refitted=False,preparation_canonical_and_svd_receipts_bound=True,reader_new_model_fits=0)


def selector_check(original,meta,pre,scores,orders,n):
    a={k:v[:n] for k,v in original.items()};shape=(n,a['baseline'].shape[1]);cw=a['core_weight']
    if cw.ndim==1:cw=cw[:,None]
    cw=np.broadcast_to(cw,shape).copy()
    oracle.exact_array(cw,np.broadcast_to(np.where(a['state_calm10'],.5,.52)[:,None],shape).copy(),'Original core allocation')
    choices=[]
    for i,row in enumerate(orders[:n]):
        need(row['date']==meta['dates'][i] and row['end']==meta['ends'][i],'Order interval')
        choices.append([meta['symbols'][i].index(s) for s in row['symbols']])
    return selector.verify(pre[:n],a['eligible'],meta['symbols'][:n],a['effects'],a['baseline'],a['core'],cw,a['state_calm10'],a['state_weak_narrow'],meta['dates'][:n],choices,scores[:n],'incumbent_local67')


def preparation_work(e,train,meta,original,forecasts,delta_audit,scores,pres,orders):
    need(e['full122_reuses_main'] is True and e['annual_control_calls']==[122,0,1,53,88],'Physical model invocation scope')
    need([r['n'] for r in e['cached_prefixes']]==list(range(123)) and [r['n'] for r in e['fresh_prefixes']]==[0,1,53,88],'All cached and fresh prefixes')
    model_work=dict.fromkeys(oracle.WORK_FIELDS,0);selector_work={};producer={k:0 for k in oracle.COUNT_FIELDS}
    entries=[dict(n=122,model=e['model'],control_gate=e['control_gate'],delta=e['delta'],delta_audit=delta_audit,producer_counts=e['producer_counts'],selector=e['selector'])]+e['fresh_prefixes']
    for entry in entries:
        n=entry['n'];inp=oracle._header(train,meta,original,n)
        if inp['formal'].any():inp['valid']=oracle._usable(inp)
        c=entry['producer_counts'];need(set(c)==set(oracle.COUNT_FIELDS),'Physical model counts')
        for value in c.values():need(type(value) is int and value>=0,'Native model count')
        need(c['events']==n and c['width']==inp['width'] and c['training_rows']==inp.get('stop',0) and c['valid_training_rows']==int(inp.get('valid',np.empty(0,bool)).sum()),'Physical source row counts')
        need(c['local_model_fits']==c['other_target_model_fits']==0 and c['actual_model_fits']==c['annual_ridge_fits']+c['monthly_ridge_fits'],'Physical model count identities')
        years=sorted(set(str(d)[:4] for d in inp['dates'][inp['formal']]))
        fits=predictions=0
        for year in years:
            cutoff=np.datetime64(str(int(year)-1)+'-12-31')
            ids=inp['valid']&(inp['td']>oracle.lower_bound(cutoff))&(inp['te']<=cutoff)
            supported=len(np.unique(inp['td'][ids].astype('datetime64[M]')))>=18
            rows=sum(int(inp['use'][i].sum()) for i,d in enumerate(inp['dates']) if inp['formal'][i] and str(d).startswith(year))
            fits+=int(supported and rows>0);predictions+=rows if supported else 0
        need(c['annual_groups']==len(years) and c['annual_ridge_fits']==fits and c['annual_prediction_rows']==predictions,'Physical annual count reconstruction')
        need(c['monthly_ridge_fits']==int(inp['active'].sum()) and c['monthly_prediction_rows']==int((inp['use']&inp['active'][:,None]).sum()),'Physical active query fits')
        expected_da=dict(delta_audit,events=n,ranking_cells=int(inp['use'].sum()),active_indices=delta_audit['active_indices'][:n],changed_cells=int(np.count_nonzero(forecasts['delta'][:n])))
        exact(entry['delta_audit'],expected_da,'Physical delta audit')
        for kind,key in (('control','control_gate'),('models','model'),('delta','delta')):
            model_receipt(entry[key],kind,inp,c,expected_da)
            for k,v in entry[key]['work'].items():model_work[k]+=v
        for k,v in c.items():producer[k]+=v
    for entry in entries+e['cached_prefixes']:
        n=entry['n']
        if 'exact' in entry:need(entry['exact'] is True,'Cached exact declaration')
        need(set(entry['selector'])==set(ARMS),'Two selector proofs')
        for arm in ARMS:
            actual=selector_check(original,meta['monthly'],pres[arm],scores[arm],orders[arm],n)
            exact(entry['selector'][arm],actual,'Saved selector receipt')
            for k,v in actual['work'].items():selector_work[k]=selector_work.get(k,0)+v
    need(set(e['early_selector'])==set(ARMS),'Early zero-delta reference proofs')
    early=selector_check(original,meta['monthly'],pres[ARMS[0]],scores[ARMS[0]],orders[ARMS[0]],122)
    for arm in ARMS:
        exact(e['early_selector'][arm],early,'Early original selector proof')
        for k,v in early['work'].items():selector_work[k]=selector_work.get(k,0)+v
    totals=dict(prescore_calls=129,selector_calls=258,independent_selector_calls=258,make_orders_calls=258,
        producer_fits=producer['actual_model_fits'],independent_canonical_solves=model_work['canonical_solves'],independent_svd_solves=model_work['svd_solves'])
    exact(e['work_totals'],totals,'Whole prepare actual work totals')
    return dict(producer_model_counts_summed=producer,independent_model_work_summed=model_work,verified_saved_selector_work_summed=selector_work,
        model_invocations=5,cached_selector_invocations=246,full122_counted_once=True,cached_model_fits=0,
        early_original_gate_work_included=True,totals=totals,reader_refits=0)


def accounting(ev,result,ind,training):
    need(ind['passed'] is True and ind['checked_paths']==6 and ind['checked_slices']==30 and ind['checked_training_groups']==6,'Six paths/30 slices/six training groups')
    need(ind['independent_accounts'] is True and ind['independent_of_production_model_and_engine'] is True and ind['terminal_liquidation'] is False,'Independent accounting scope')
    need(set(result['arms'])==set(ind['paths'])==set(training)==set(ARMS),'Accounting arms')
    residuals=[];exports={};zero_fields=0
    for arm in ARMS:
        need(set(result['arms'][arm])==set(ind['paths'][arm])==set(PATHS),'Accounting paths')
        for path in PATHS:
            saved=result['arms'][arm][path];cert=ind['paths'][arm][path]
            need(cert['passed'] is True and cert['monthly_executions']==122 and set(saved['slices'])==set(cert['slices'])==set(SLICES),'Five slices and122 executions')
            shared.compare_metrics(cert['slices']['full122'],saved['metrics'],arm+'/'+path,residuals)
            for split in SLICES:shared.compare_metrics(cert['slices'][split],saved['slices'][split],arm+'/'+path+'/'+split,residuals)
            for key in ('fees_paid','dollar_turnover','trade_rows'):shared.near(cert[key],saved['metrics'][key],key,residuals)
            need(saved['accounting']['exported_fill_ledger_matches'] is True and saved['accounting']['terminal_liquidation'] is False,'Account declarations')
            data={k:ev.read(OUT/'results'/(arm+'_'+path+'_'+k+'.json')) for k in KINDS};exports[arm,path]=data
            need(len(data['HOLDINGS'])==122 and len(data['TRADES'])==cert['trade_rows'],'Saved export rowcounts')
            need([r['date'] for r in data['DAILY']]==[r['date'] for r in data['DAILY_POSITIONS']],'Daily axes')
            if path!='next_open_cost':
                for row in (saved['metrics'],saved['slices']['full122']):need(row['cost_per_side']==row['fees_paid']==0.,'Four exact ZERO headlines')
                need(cert['fees_paid']==0.,'Independent zero fees')
                for key,field in (('TRADES','cost'),('HOLDINGS','fee')):
                    for row in data[key]:need(shared.finite(row[field]) and row[field]==0.,'Every ZERO fee');zero_fields+=1
            if path=='same_close_zero':
                for short,long in (('training','first88'),('early','early53'),('recent','recent35')):
                    shared.compare_metrics(cert['slices'][long],training[arm][short],arm+'/'+short,residuals)
    return dict(paths=6,slices=30,training_groups=6,zero_streams=4,zero_fee_fields=zero_fields,metric_residuals=residuals,portfolio_replay=False),exports


def terminal_check(record,plan_sha,hashes):
    need(record['plan_sha256']==plan_sha and set(record['phases'])=={'prepare','select','diagnose'},'Root phase terminals')
    for key,relative in (('prepare','prepared/COMPLETION.json'),('select','selected/FROZEN.json'),('diagnose','results/COMPLETION.json')):
        row=record['phases'][key]
        need(type(row['exit_code']) is int and row['exit_code']==0 and type(row['chunk_id']) is str and bool(row['chunk_id']),'Actual terminal success')
        need(row['session_id'] is None or type(row['session_id']) is int and row['session_id']>0,'Terminal session')
        need('owner' not in row or row['owner']=='/root','Terminal owner')
        need(shared.local(row['artifact_path'])==OUT/relative and row['artifact_sha256']==hashes[key],'Terminal artifact hash')


def run(args):
    need(platform.python_version()=='3.9.6','Pinned wrapper Python')
    for key in ('plan','prepared','frozen','results','terminals'):need(shared.sha_value(getattr(args,key+'_sha256')),'Explicit phase SHA')
    ev=Evidence();plan=ev.read(OUT/'PLAN.json',args.plan_sha256)
    need(plan['schema']=='monthly_matured_clock273_v1' and plan['design']['arms']==list(ARMS) and plan['design']['trials']==2,'Fixed two-arm PLAN')
    need(set(plan['code_sha256'])==set(CODE),'Ten formal source files')
    for name,h in {**FIXED,**plan['code_sha256']}.items():need(ev.registered(plan,name)==h,'Registered code dependency');ev.pin(name,h)
    tests=ev.read(ROOT/'work/monthly_matured_clock273_TESTS.json',ev.registered(plan,'work/monthly_matured_clock273_TESTS.json'))
    review=ev.read(ROOT/'work/monthly_matured_clock273_CODE_REVIEW.json',ev.registered(plan,'work/monthly_matured_clock273_CODE_REVIEW.json'))
    need(tests['passed'] is True and tests['synthetic_only'] is True and type(tests['tests']) is int and tests['tests']>0 and tests['failures']==tests['errors']==0,'Formal synthetic pass')
    exact(tests['code_sha256'],plan['code_sha256'],'Formal code map');exact(review['code_sha256'],plan['code_sha256'],'Reviewed code map')
    need(review['passed'] is True and review['unresolved_blockers']==[] and review['tests_sha256']==shared.digest(ROOT/'work/monthly_matured_clock273_TESTS.json'),'Prospective review')
    need(tests['runtime']==plan['runtime']==dict(python='3.9.6',numpy='2.0.2',pandas='2.3.3'),'Pinned numerical runtime')
    pc=ev.phase(OUT/'prepared',args.prepared_sha256,args.plan_sha256,PREP)
    rc=ev.phase(OUT/'results',args.results_sha256,args.plan_sha256,RESULT)
    frozen=ev.read(OUT/'selected/FROZEN.json',args.frozen_sha256);training=ev.read(OUT/'selected/TRAIN_RESULTS.json',frozen['training_sha256'])
    need(frozen['plan_sha256']==args.plan_sha256 and frozen['prepared_sha256']==args.prepared_sha256 and rc['frozen_sha256']==args.frozen_sha256,'Phase chain')
    exact(frozen['selection'],saved_selection(training),'Independent first88 selection')
    terminal=ev.read(OUT/'PHASE_TERMINALS.json',args.terminals_sha256);terminal_check(terminal,args.plan_sha256,dict(prepare=args.prepared_sha256,select=args.frozen_sha256,diagnose=args.results_sha256))
    closure=plan['original169_source_closure']
    for name,h in closure['chain_sha256'].items():need(ev.registered(plan,name)==h,'169 chain registered');ev.pin(name,h)
    parity=ev.read(REBUILD/'PARITY.json',ev.registered(plan,REBUILD/'PARITY.json'))
    need(parity['passed'] is True and parity['returns_computed'] is False and parity['arm']=='control' and len(parity['artifacts_sha256'])==12,'Full169 control parity')
    exact(closure['artifacts_sha256'],parity['artifacts_sha256'],'Twelve169 artifact manifest')
    for name,h in parity['artifacts_sha256'].items():need(Path(name).name==name and ev.registered(plan,REBUILD/name)==h,'169 artifact binding');ev.pin(REBUILD/name,h)
    orig=arrays(ev,LOCAL/'DATA.npz',ev.registered(plan,LOCAL/'DATA.npz'));base=arrays(ev,REBUILD/'TRAIN_INPUTS.npz',ev.registered(plan,REBUILD/'TRAIN_INPUTS.npz'))
    metadata=ev.read(REBUILD/'MONTHLY_META.json',ev.registered(plan,REBUILD/'MONTHLY_META.json'));m=ev.read(OUT/'prepared/META.json')
    exact(metadata['monthly'],m,'Original monthly identities');need(len(orig)==27 and orig['baseline'].shape==(122,49),'Original27 and fixed axes')
    extended=arrays(ev,OUT/'prepared/APPENDED_INPUTS.npz');aa=ev.read(OUT/'prepared/APPEND_AUDIT.json')
    need(set(extended)==set(base) and aa['asof']==m['dates'][-1] and aa['counts']['original_rows']==8820 and aa['counts']['appended_rows']==343 and aa['counts']['total_rows']==9163,'Fixed seven completed cohorts')
    for key,value in base.items():oracle.exact_array(extended[key][:len(value)],value,'Original training prefix '+key)
    for key in ('event_plain','event_dates','eligible'):oracle.exact_array(extended[key],base[key],'Original event cache '+key)
    exact(aa['original_arrays'],{k:fingerprint(v) for k,v in base.items()},'Append original fingerprints')
    for key in ('dates','ends','symbols'):exact(aa['training_metadata'][key][:180],metadata['training'][key],'Original training metadata '+key)
    need(len(aa['training_metadata']['dates'])==187,'Exact extended monthly axis')
    expected_appends=[dict(signal_date=m['dates'][j],label_end=m['ends'][j],event_index=j,first_row=8820+i*49,rows=49) for i,j in enumerate(range(114,121))]
    exact(aa['appended_months'],expected_appends,'Exactly Jan-Jul2026 mature extension')
    for i,j in enumerate(range(114,121)):
        lo=8820+i*49
        oracle.exact_array(extended['train_plain'][lo:lo+49],base['event_plain'][j*49:(j+1)*49],'New feature source is signal-time query snapshot')
        oracle.exact_array(extended['train_eligible'][lo:lo+49],base['eligible'][j],'New eligibility source exact')
    for key in ('original_arrays_unchanged','original_prefix_bytes_exact','event_arrays_unchanged','independent_label_oracle_passed'):need(aa[key] is True,'Append proof '+key)
    for key in ('source_files_authenticated','portfolio_returns_computed','original_labels_recomputed','trading_eligibility_modified'):need(aa[key] is False,'Append scope '+key)
    combined=dict(monthly=m,training=aa['training_metadata'])
    fc=arrays(ev,OUT/'prepared/FORECASTS.npz');ma=ev.read(OUT/'prepared/MODEL_AUDIT.json');da=ev.read(OUT/'prepared/DELTA_AUDIT.json');prep=ev.read(OUT/'prepared/PREP_AUDIT.json')
    inp,bundle,models=saved_models(extended,combined,orig,fc,ma,da,prep)
    scores=arrays(ev,OUT/'prepared/SCORES.npz');pres=arrays(ev,OUT/'prepared/PRE_SCORES.npz');orders=ev.read(OUT/'prepared/ORDERS.json')
    need(set(scores)==set(pres)==set(orders)==set(ARMS),'Saved strategy roles')
    reference=ev.read(REFERENCE/'ORDERS.json',ev.registered(plan,REFERENCE/'ORDERS.json'))
    exact(orders[ARMS[0]],reference[ARMS[0]],'Incumbent ordered holdings unchanged')
    for path,value in [('SCORES',scores),('PRE_SCORES',pres)]:
        ref=arrays(ev,REFERENCE/(path+'.npz'),ev.registered(plan,REFERENCE/(path+'.npz')))
        oracle.exact_array(value[ARMS[0]],ref[ARMS[0]],'Incumbent '+path+' bytes')
    wanted=pres[ARMS[0]].copy();active=fc['delta']!=0.;wanted[active]+=fc['delta'][active]
    oracle.exact_array(pres[ARMS[1]],wanted,'Only one nonzero delta addition to original P')
    work=preparation_work(prep,extended,combined,orig,fc,da,scores,pres,orders)
    sp=ev.read(OUT/'prepared/SOURCE_PARITY.json')
    for key in ('original_arrays_unchanged','original_training_unchanged','original_orders_exact','original_pre_and_scores_exact'):need(sp[key] is True,'Source parity '+key)
    exact(sp['original_array_fingerprints'],{k:fingerprint(v) for k,v in orig.items()},'Original27 fingerprints')
    exact(sp['original_training_fingerprints'],{k:fingerprint(v) for k,v in base.items()},'Original15 training fingerprints')
    raw=shared.canonical_source_map(plan['raw_source_hashes']);exact(shared.canonical_source_map(sp['raw_source_hashes']),raw,'Original64 raw sources')
    for name,h in raw.items():need(ev.registered(plan,name)==h,'Registered original raw source');ev.pin(name,h)
    result=ev.read(OUT/'results/RESULTS.json');ind=ev.read(OUT/'results/INDEPENDENT_OUTCOME.json')
    exact(result['selection'],frozen['selection'],'Frozen primary');account,export=accounting(ev,result,ind,training)
    prior_c=ev.read(PRIOR/'results/COMPLETION.json',ev.registered(plan,PRIOR/'results/COMPLETION.json'))
    old_result=ev.read(PRIOR/'results/RESULTS.json',ev.registered(plan,PRIOR/'results/RESULTS.json'))
    need(prior_c['artifacts_sha256']['RESULTS.json']==shared.digest(PRIOR/'results/RESULTS.json'),'Prior summary manifest')
    prior_exports={}
    for path in PATHS:
        for kind in KINDS:
            name=ARMS[0]+'_'+path+'_'+kind+'.json';h=prior_c['artifacts_sha256'][name]
            need(ev.registered(plan,PRIOR/'results'/name)==h,'Prior economic export pin');value=ev.read(PRIOR/'results'/name,h);prior_exports[path,kind]=value
            if not(path=='next_open_cost' and kind=='HOLDINGS'):exact(export[ARMS[0],path][kind],value,'Incumbent economic export exact')
    fee=shared.diagnostic_summary_parity(result['arms'][ARMS[0]],old_result['arms'][ARMS[0]],export[ARMS[0],'next_open_cost']['HOLDINGS'],prior_exports['next_open_cost','HOLDINGS'])
    ev.finish()
    return dict(schema='monthly_matured_clock273_saved_review_v1',passed=True,unresolved_blockers=[],created_utc=datetime.now(timezone.utc).isoformat(),
        plan_sha256=args.plan_sha256,prepared_completion_sha256=args.prepared_sha256,results_completion_sha256=args.results_sha256,
        frozen_sha256=args.frozen_sha256,terminals_sha256=args.terminals_sha256,code_sha256=plan['code_sha256'],input_sha256={str(ROOT/k):v for k,v in ev.pins.items()},
        model_review=models,preparation_work=work,accounting=account,incumbent_summary_fee_display_review=fee,
        scope=dict(saved_npz_decoded=True,raw_price_decoded=False,model_fits_performed=0,svd_performed=False,orders_generated=False,portfolio_returns_recomputed=False,
            stored_beta_prediction_equations_checked=True,preparation_model_verifier_same_author=True,label_returns_reconstructed=False,all_plan_inputs_rehashed=False,
            used_artifacts_and_original64_source_bytes_hashed=True,early_gate_work_receipts_checked=True))


if __name__=='__main__':
    import json
    p=argparse.ArgumentParser()
    for name in ('plan','prepared','frozen','results','terminals'):p.add_argument('--'+name+'-sha256',required=True)
    print(json.dumps(run(p.parse_args()),indent=2,allow_nan=False))
