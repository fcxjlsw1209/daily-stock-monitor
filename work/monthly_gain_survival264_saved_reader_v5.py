"""Root saved-evidence review; no prices/arrays, model fit or portfolio replay.

Reuses the pinned261 standard-library saved-accounting comparison only. The
separately authored264 verifier supplies numerical independence. Prints JSON;
root adds the actually observed reader terminal before saving RESULT_REVIEW.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from fractions import Fraction
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/monthly_gain_survival264'
PARENT=ROOT/'work/monthly_weak_tree261_saved_reader.py'
PARENT_SHA='0f72873c7eb253cb11d364d9b1a4c206885059c5ce5174e73d1df97ad0900190'
ARMS=('incumbent_local67','endpoint_gain_age_control','continuous_gain_survival_candidate')
PREP={'FEATURES.json','SCORES.npz','PRE_SCORES.npz','DECISION_AUDIT.json','ORDERS.json','META.json','SOURCE_PARITY.json','INDEPENDENT_PREPARE.json'}
PATHS=('same_close_zero','next_open_zero','next_open_cost')
RESULT={'RESULTS.json','INDEPENDENT_OUTCOME.json'}|{a+'_'+p+'_'+k+'.json' for a in ARMS for p in PATHS for k in ('DAILY','TRADES','HOLDINGS','DAILY_POSITIONS')}
WORK_FIELDS=('full_source_slot_inspections','cached_slot_validations','momentum_evaluations','return_evaluations','statistic_evaluations','ratio_evaluations','rank_evaluations','scalar_rank_evaluations','original_episode_calls','original_source_price_observations','original_source_cells','positive_gain_anchors','survival_comparisons','endpoint_comparisons','positive_raw_multiplications','original_source_snapshot_price_observations','original_source_snapshot_hashes')
SNAPSHOT_FIELDS=('original_source_snapshot_sha256','original_source_snapshot_price_observations','original_source_snapshot_hashes')
ZERO_SOURCE_COUNTS=('endpoint_nonzero_deltas','survival_nonzero_deltas','endpoint_changed_holdings','survival_changed_holdings','candidate_control_changed_holdings')

def need(ok,msg):
    if not ok:raise ValueError(msg)

def canonical_digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def integer(value,label,expected=None):
    need(type(value) is int and value>=0,'Native nonnegative '+label)
    if expected is not None:need(value==expected,'Exact '+label)
    return value

def sha_value(value,label):
    need(type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value),'SHA256 '+label)
    return value

def _receipt_work(value,n,label):
    need(type(value) is dict and value.get('passed') is True,'Passed '+label)
    integer(value.get('checked_events'),label+' prefix',n)
    need(value.get('producer_modules_imported') is False and value.get('computes_returns') is False,'Independent scope '+label)
    return {k:integer(value.get(k),label+'/'+k) for k in WORK_FIELDS}

def _new_work(counts,full,selecting):
    """Gain routines are grouped work; weighted sums are not primitive counters."""
    work=dict.fromkeys(WORK_FIELDS,0)
    work.update(full_source_slot_inspections=counts['price_observations'] if full else 0,
        cached_slot_validations=0 if full else counts['price_observations'],
        return_evaluations=counts['gain_return_observations'],statistic_evaluations=counts['gain_statistics'],
        ratio_evaluations=counts['quality_ratio_evaluations'],rank_evaluations=2*counts['available_events'] if selecting else 0,
        scalar_rank_evaluations=counts['endpoint_ranked_cells']+counts['survival_ranked_cells'] if selecting else 0,
        original_episode_calls=3 if selecting else 0)
    for key in ('positive_gain_anchors','survival_comparisons','endpoint_comparisons','positive_raw_multiplications'):work[key]=counts[key]
    return work

def source_evidence_review(evidence,width=49):
    """Reconcile saved receipts only. Never reconstruct or hash source prices.

    Original rolling reconstruction is reused. The original, raw and full gates
    each separately scan/hash the consumed source matrix; cached gates do neither.
    A saved digest is verified here as a binding, not reissued as a capability.
    """
    integer(width,'source width');need(width>0,'Positive source width')
    main=evidence['main'];physical=evidence['physical_prefixes'];cached=evidence['cached_prefixes']
    need(type(main) is dict and main.get('prefix')==122 and main.get('reused_full') is False and main.get('source_recomputed') is True,'Main full source execution')
    need(type(physical) is list and [x['prefix'] for x in physical]==[0,1,53,88,122],'All physical prefixes')
    need(type(cached) is list and [x['prefix'] for x in cached]==list(range(123)),'All cached prefixes')
    need(evidence['full_original_source_certificate_sha256']==canonical_digest(main['original_source']),'Full original certificate binding')
    need(evidence['full_source_certificate_sha256']==canonical_digest(main['gates']['source']),'Full raw source receipt binding')
    actual=[main]+physical[:-1];producer=dict.fromkeys(WORK_FIELDS,0);independent=dict.fromkeys(WORK_FIELDS,0)
    def add(target,work):
        for key in WORK_FIELDS:target[key]+=work[key]
    def projection(cert,n):
        return dict(schema='monthly_gain_survival264_original_source_prefix_v1',checked_events=n,original_events=cert['original_events'][:n],u0_indices=cert['u0_indices'][:n])
    for item in [main]+physical+cached:
        n=integer(item['prefix'],'source prefix');need(n<=122 and item.get('exact') is True,'Exact source prefix scope')
        counts=item['counts'];need(type(counts) is dict,'Saved decision counts')
        for key,value in counts.items():integer(value,'count/'+key)
        integer(counts['events'],'count events',n);integer(counts['symbols_per_event'],'count width',width)
        need(counts['new_model_fits']==counts['predictive_model_fits']==0 and counts['original_episode_calls']==3,'No fits and three episodes')
        need(counts['price_observations']==253*counts['required_cells'] and counts['gain_return_observations']==252*counts['required_cells'] and counts['gain_statistics']==counts['required_cells'],'Source/gain routine counts')
        need(counts['endpoint_ranked_cells']==counts['survival_ranked_cells']==counts['required_cells'],'Complete two rank domains')
        need(counts['quality_ratio_evaluations']==2*(counts['required_cells']-counts['no_positive_gain_cells']) and counts['positive_raw_multiplications']==2*(counts['required_cells']-counts['nonpositive_original_cells']),'Quality and signed-transform counts')
        cached_item='selection' in item
        kind='selection' if cached_item else 'full'
        need(item[kind]['counts']==counts,'Selection/count receipt binding')
        raw_counts=dict(counts)
        for key in ZERO_SOURCE_COUNTS:raw_counts[key]=0
        need(item['gates']['source']['feature_counts']==raw_counts,'Unranked raw-source count binding')
        gates=[('incumbent',item['gates']['incumbent']),('source',item['gates']['source']),(kind,item[kind])]
        for label,receipt in gates:
            work=_receipt_work(receipt,n,label)
            expected=dict.fromkeys(WORK_FIELDS,0) if label=='incumbent' else _new_work(counts,not cached_item,label==kind)
            if label=='incumbent':expected['original_episode_calls']=1
            for key,value in expected.items():
                if not cached_item and label!='incumbent' and key in SNAPSHOT_FIELDS[1:]:continue
                integer(work[key],label+' work/'+key,value)
            if label=='source':need(receipt.get('source_rebuilt') is (not cached_item),'Source reconstruction scope')
        if cached_item:
            cert=main['original_source']
            need(item['original_source_certificate_sha256']==canonical_digest(cert),'Cached original certificate binding')
        else:
            cert=item['original_source'];work=item['producer_original_source_work']
            _receipt_work(cert,n,'original')
            need(type(cert['original_events']) is list and type(cert['u0_indices']) is list and len(cert['original_events'])==len(cert['u0_indices'])==n,'Complete original certificate prefix')
            formal=0
            for q,event in enumerate(cert['original_events']):
                integer(event['query_index'],'original query',q);need(event['u0_indices']==cert['u0_indices'][q],'Original U0 duplicate binding')
                ids=event['u0_indices'];need(type(ids) is list and all(type(j) is int and 0<=j<width for j in ids) and ids==sorted(set(ids)),'Original U0 native indices')
                need(type(event['initial_partial']) is bool and type(event['calm']) is bool,'Original event states')
                if event['initial_partial']:need(q==0 and event['date']=='2016-08-17' and ids==[],'Named partial identity')
                else:
                    formal+=1
                    for key in ('original_raw','original_rank','frozen_core','rebuilt_core','core_abs_error','original_sigma63'):need(type(event[key]) is list and len(event[key])==width,'Original full-width evidence')
            integer(cert['original_checked_cells'],'original checked cells',width*formal)
            integer(cert['original_source_cells'],'original reconstruction cells',width*formal)
            integer(counts['original_core_cells'],'original full-width count',width*formal)
            need(cert['original_source_price_observations']>=width*formal,'Original reconstruction observations')
            if not formal:integer(cert['original_source_price_observations'],'No initial/empty reconstruction',0)
            allowed={'original_source_price_observations','original_source_cells',*SNAPSHOT_FIELDS[1:]}
            for key in WORK_FIELDS:
                if key not in allowed:integer(cert[key],'Original gate no new gain work/'+key,0)
            expected_anchor=dict(schema='monthly_gain_survival264_original_source_anchor_v1',checked_events=n,certificate_sha256=canonical_digest(cert),original_event_sha256=[canonical_digest(e) for e in cert['original_events']])
            need(item['original_source_anchor']==expected_anchor,'Physical original source anchor exact')
            need(type(work) is dict and set(work)=={'fullpast_rows','union_symbols','original_source_price_observations','original_source_cells'},'Producer original-work schema')
            rows=integer(work['fullpast_rows'],'consumed rows');names=work['union_symbols']
            need(type(names) is list and all(type(s) is str and s for s in names) and names==sorted(set(names)),'Consumed symbol union')
            observations=integer(work['original_source_price_observations'],'consumed observations',rows*len(names))
            integer(work['original_source_cells'],'producer original cells',width*formal)
            need(bool(rows)==bool(names)==bool(formal),'No source numeric work for initial/empty prefixes')
            snapshot_sha=sha_value(cert.get('original_source_snapshot_sha256'),'consumed source snapshot')
            for label,proof in [('original',cert),('source',item['gates']['source']),('full',item['full'])]:
                need(proof.get('original_source_snapshot_sha256')==snapshot_sha,'Same consumed source snapshot/'+label)
                integer(proof['original_source_snapshot_price_observations'],'Snapshot observations/'+label,observations)
                integer(proof['original_source_snapshot_hashes'],'Numeric snapshot passes/'+label,int(observations>0))
                if label!='original':
                    need(proof['original_source_proof_sha256']==canonical_digest(cert),'Exact issued original certificate/'+label)
                    integer(proof['original_source_proof_events'],'Original proof events/'+label,n)
            reuse=item.get('reused_full');need(type(reuse) is bool and item.get('source_recomputed') is (not reuse),'Physical reuse flags')
        need(item['consumed_original_source_sha256']==canonical_digest(projection(cert,n)),'Consumed original prefix commitment')
    for item in physical:
        reuse=item['prefix']==122;need(item['reused_full'] is reuse,'Only full122 is reused')
        if reuse:need(item==dict(main,reused_full=True,source_recomputed=False),'Full122 exact reuse')
    for item in actual+cached:
        full='full' in item;work=_new_work(item['counts'],full,True)
        if full:
            work['original_source_price_observations']=item['producer_original_source_work']['original_source_price_observations'];work['original_source_cells']=item['counts']['original_core_cells']
            add(independent,{key:item['original_source'][key] for key in WORK_FIELDS})
        add(producer,work)
        for receipt in (item['gates']['incumbent'],item['gates']['source'],item['full' if full else 'selection']):add(independent,{key:receipt[key] for key in WORK_FIELDS})
    total={key:producer[key]+independent[key] for key in WORK_FIELDS};execution=evidence['execution_accounting']
    for owner,expected in (('producer',producer),('independent',independent),('total',total)):
        got=execution[owner];need(type(got) is dict and set(got)==set(WORK_FIELDS),'Complete '+owner+' execution counters')
        for key,value in expected.items():integer(got[key],owner+'/'+key,value)
    for key,value in dict(physical_preparations=5,cached_decisions=123,original_source_gate_calls=5,original_core_checked_cells=sum(x['original_source']['original_checked_cells'] for x in actual),return_vector_steps_computed=total['return_evaluations'],new_model_fits=0,predictive_model_fits=0).items():integer(execution[key],key,value)
    need(execution['physical_122_reused'] is True and execution['source_and_cached_statistic_recomputation_disclosed'] is True,'Reuse/recomputation disclosure')
    return dict(passed=True,physical_preparations=5,cached_decisions=123,physical_122_reused=True,
        original_raw_full_snapshot_bindings_checked=True,source_snapshot_sha_recomputed_by_reader=False,
        original_pandas_reconstruction_reused=True,cached_source_price_reads=0,
        independent_original_reconstruction_observations=independent['original_source_price_observations'],
        independent_source_snapshot_observations=independent['original_source_snapshot_price_observations'],
        independent_numeric_snapshot_passes=independent['original_source_snapshot_hashes'],
        execution_owner_totals_checked=True)

def load_parent():
    need(hashlib.sha256(PARENT.read_bytes()).hexdigest()==PARENT_SHA,'Saved comparison helper changed')
    spec=importlib.util.spec_from_file_location('gain_survival264_saved_comparison261',PARENT)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.OUT=OUT;module.ARMS=ARMS
    return module

def selection(p,metrics,orders):
    need(set(metrics)==set(orders)==set(ARMS),'All three arms')
    invalid={a:p.required_metrics(metrics[a],a==ARMS[0]) for a in ARMS}
    need(all(len(orders[a])==122 for a in ARMS),'Full122 saved orders')
    candidate=ARMS[2];guards={};changes={}
    for reference in ARMS[:2]:
        changed=[i for i in range(88) if set(orders[candidate][i]['symbols'])!=set(orders[reference][i]['symbols'])]
        changes[reference]=changed
        admissible=not invalid[candidate] and not invalid[reference]
        close=admissible and p.guard(metrics[candidate][PATHS[0]],metrics[reference][PATHS[0]],.005 if reference==ARMS[0] else 0.)
        opening=admissible and p.guard(metrics[candidate][PATHS[1]],metrics[reference][PATHS[1]],0.)
        guards[reference]=dict(close=bool(close),open=bool(opening),support=len(changed)>=3 and any(i<53 for i in changed) and any(i>=53 for i in changed))
    admitted=all(all(g.values()) for g in guards.values())
    return dict(primary=candidate if admitted else ARMS[0],admitted=admitted,guards=guards,changed_months_by_reference=changes,undefined_sharpe=invalid,first88_only=True,later34_used=False,cost_used=False,diagnostic_arm_never_promoted=True)


def diagnostic_holdings_parity(current, previous):
    """Check display-only fee rounding; no tolerance for economic exports.

    Frozen engine sums at most six nonnegative turnover terms in set order,
    then scales by cost. gamma_6 bounds six binary64 rounding operations.
    This establishes compatibility with that representation error bound; it
    does not reconstruct the engine's unstored intermediate post value.
    """
    need(type(current) is list and type(previous) is list and len(current)==len(previous), 'Complete displayed holdings')
    gamma=Fraction(6,2**53-6)
    relative_bound=2*gamma/(1-gamma)
    differences=[]
    for i,(a,b) in enumerate(zip(current,previous)):
        need(type(a) is dict and type(b) is dict and set(a)==set(b) and 'fee' in a,'Holdings field schema')
        need({k:v for k,v in a.items() if k!='fee'}=={k:v for k,v in b.items() if k!='fee'},'Non-fee holdings fields must be exact')
        av,bv=a['fee'],b['fee']
        need(type(av) in (int,float) and type(bv) in (int,float) and math.isfinite(av) and math.isfinite(bv) and av>=0 and bv>=0,'Finite nonnegative display fees')
        if av==bv:continue
        need(av>0 and bv>0,'Zero display fee must remain exact')
        terms=set(a['symbols']) | (set(current[i-1]['symbols']) if i else set())
        need(len(terms)<=6,'At most six turnover terms for EW3')
        difference=abs(Fraction(av)-Fraction(bv))
        bound=relative_bound*min(Fraction(av),Fraction(bv))
        need(difference<=bound,'Display fee difference exceeds positive-sum binary64 bound')
        differences.append(dict(index=i,signal_date=a['signal_date'],execution_date=a['execution_date'],previous_fee=bv,current_fee=av,
            absolute_difference=float(difference),larger_fee_ulp=math.ulp(max(av,bv)),ulp_difference=float(difference)/math.ulp(max(av,bv)),
            exact_bound_numerator=bound.numerator,exact_bound_denominator=bound.denominator,turnover_terms_upper_bound=6))
    return dict(exact=not differences,display_fee_only=True,nonfee_fields_exact=True,differences=differences,
        gamma_operations=6,unit_roundoff='2**-53',roundoff_bound='2*gamma_6/(1-gamma_6)*min(fee_a,fee_b)',
        bound_checked_with_exact_fractions=True,intermediate_post_reconstructed=False,mechanism_compatible_not_exact_intermediate_proof=True,
        frozen_engine_path='outputs/monthly_factor_search_v2/validate.py',frozen_engine_sha256='b5c6bbbfa5f4db0821f6b0dacb33a43c974cceb8a150578803eef73132cd30e4',
        frozen_engine_lines=[86,89,95],actual_portfolio_replay=False)

def diagnostic_summary_parity(current, previous, current_holdings, previous_holdings):
    """Allow only bound-checked displayed fees, with exact own-ledger totals.

    This does not recompute returns, NAV, fills or costs. Each saved total must
    equal the frozen engine's chronological binary64 ``total = 0.; total += fee``
    over its own already-saved HOLDINGS. No aggregate epsilon is used.
    """
    need(type(current) is dict and type(previous) is dict and set(current)==set(previous)==set(PATHS),'Exact summary path keys')
    display=diagnostic_holdings_parity(current_holdings,previous_holdings)
    totals={};stripped=[]
    for label,summary,holdings in (('current',current,current_holdings),('previous',previous,previous_holdings)):
        cost=summary['next_open_cost']
        need(type(cost) is dict and type(cost.get('metrics')) is dict and type(cost.get('slices')) is dict and type(cost['slices'].get('full122')) is dict,'Cost summary fee containers')
        metrics=cost['metrics'];full=cost['slices']['full122']
        need('fees_paid' in metrics and 'fees_paid' in full,'Both saved cost fee totals required')
        need(type(metrics.get('cost_per_side')) in (int,float) and metrics['cost_per_side']==.0005,'Diagnostic cost rate remains exact0.05percent per side')
        total=0.;last_signal=None;last_execution=None
        for holding in holdings:
            signal=holding.get('signal_date');execution=holding.get('execution_date')
            for value in (signal,execution):
                need(type(value) is str,'Native holdings date strings')
                try:canonical=datetime.strptime(value,'%Y-%m-%d').date().isoformat()
                except (ValueError,OverflowError):raise ValueError('Canonical holdings dates') from None
                need(canonical==value,'Canonical holdings dates')
            need(signal<=execution and (last_signal is None or last_signal<signal) and (last_execution is None or last_execution<execution),'Strict chronological holdings dates')
            last_signal=signal;last_execution=execution
            total+=holding['fee']
            need(math.isfinite(total),'Finite chronological display fee total')
        for location,container in (('metrics',metrics),('slices.full122',full)):
            value=container['fees_paid']
            need(type(value) in (int,float) and math.isfinite(value) and value>=0,'Finite nonnegative summary fee '+label+'/'+location)
            need(value==total,'Exact own chronological display fee sum '+label+'/'+location)
        totals[label]=total
        # Remove precisely these two leaves, retaining every other key/value.
        clean=dict(summary);clean_cost=dict(cost);clean['next_open_cost']=clean_cost
        clean_cost['metrics']={k:v for k,v in metrics.items() if k!='fees_paid'}
        clean_cost['slices']=dict(cost['slices'])
        clean_cost['slices']['full122']={k:v for k,v in full.items() if k!='fees_paid'}
        stripped.append(clean)
    need(stripped[0]==stripped[1],'Every non-display-fee summary field must be exact')
    return dict(exact=current==previous,economic_summary_metrics_exact=True,nonfee_fields_exact=True,
        allowed_difference_paths=['next_open_cost.metrics.fees_paid','next_open_cost.slices.full122.fees_paid'],
        own_chronological_fee_totals=totals,own_totals_exact=True,full122_duplicate_exact=True,
        accumulation='binary64 total=0.0; chronological total+=holding[fee]',aggregate_tolerance_used=False,
        display_fee_review=display,actual_portfolio_replay=False,saved_nav_metrics_recomputed=False)

def run(args):
    p=load_parent();ev=p.Evidence();ev.pin(PARENT,PARENT_SHA);ev.pin(Path(__file__))
    ev.pin(ROOT/'work/monthly_tracking_risk262_saved_reader_v2.py','e32e157ad16b0d49a32a700b18f150efb1d2d9aeb6f189fca7f2a69201707283')
    ev.pin(ROOT/'outputs/monthly_factor_search_v2/validate.py','b5c6bbbfa5f4db0821f6b0dacb33a43c974cceb8a150578803eef73132cd30e4')
    plan=ev.read(OUT/'PLAN.json',args.plan_sha256)
    context=ROOT/'work/monthly_gain_survival264_context.json'
    contract=ROOT/'work/monthly_gain_survival264_IMPLEMENTATION_CONTRACT.json'
    c=ev.read(context,plan['context_sha256']);d=ev.read(contract,plan['implementation_contract_sha256'])
    need(c['frozen'] is True and d['frozen'] is True and c['arms']==list(ARMS),'Frozen prospective scope')
    need(c==plan['context'] and d['context']['sha256']==plan['context_sha256'],'Context binding')
    for name in ('TESTS','CODE_REVIEW'):
        path=ROOT/('work/monthly_gain_survival264_'+{'TESTS':'TESTS_V2','CODE_REVIEW':'CODE_REVIEW'}[name]+'.json')
        value=ev.read(path,ev.registered(plan,path));need(value['passed'] is True,'Formal '+name)
        for file,want in value['code_sha256'].items():ev.pin(file,want)
        if name=='TESTS':need(value['synthetic_only'] is True and value['runtime']==plan['runtime'],'Formal synthetic/runtime evidence')
        else:need(value['unresolved_blockers']==[] and value['actual_returns_computed'] is False,'Prospective code review')
    prep=ev.phase(OUT/'prepared',args.prepared_sha256,args.plan_sha256,PREP)
    result_completion=ev.phase(OUT/'results',args.results_sha256,args.plan_sha256,RESULT)
    need(result_completion['frozen_sha256']==args.frozen_sha256 and result_completion['paths']==9 and result_completion['slices']==45,'Results linked to exact frozen training and complete scope')
    frozen=ev.read(OUT/'selected/FROZEN.json',args.frozen_sha256)
    train=ev.read(OUT/'selected/TRAIN_RESULTS.json',frozen['training_sha256'])
    orders=ev.read(OUT/'prepared/ORDERS.json',frozen['orders_sha256'])
    need(frozen['plan_sha256']==args.plan_sha256 and frozen['prepared_sha256']==args.prepared_sha256,'Frozen training lineage')
    terminals=ev.read(OUT/'PHASE_TERMINALS.json',args.terminals_sha256)
    p.terminals_check(terminals,args.plan_sha256,dict(prepare=args.prepared_sha256,select=args.frozen_sha256,diagnose=args.results_sha256))
    results=ev.read(OUT/'results/RESULTS.json');independent=ev.read(OUT/'results/INDEPENDENT_OUTCOME.json')
    need(results['zero_cost_selection'] is True and results['next_open_cost_diagnostic_only'] is True and results['observer_promotion'] is False,'Zero-cost primary research scope')
    for arm in ARMS:
        for path in PATHS[:2]:
            rate=results['arms'][arm][path]['metrics']['cost_per_side']
            need(type(rate) in (int,float) and rate==0.,'Every ZERO path cost rate exactly zero')
    pick=selection(p,train,orders)
    need(pick==frozen['selection']==results['selection'],'First88-only dual-reference selection exact')
    accounting,exports=p.accounting(ev,results,independent,train)
    evidence=ev.read(OUT/'prepared/INDEPENDENT_PREPARE.json')
    decision=ev.read(OUT/'prepared/DECISION_AUDIT.json')
    source=ev.read(OUT/'prepared/SOURCE_PARITY.json')
    need(evidence['all123_order_prefixes'] is True and evidence['original_arrays_unchanged'] is True and evidence['original_eligible_unchanged'] is True,'Complete prefix/preservation evidence')
    need([x['prefix'] for x in evidence['cached_prefixes']]==list(range(123)),'All123 cached prefixes')
    source_work_review=source_evidence_review(evidence)
    need(evidence['new_model_fits']==evidence['predictive_model_fits']==0 and prep['new_returns_computed'] is False,'Preparation scope')
    need(source['original_orders_exact'] is True and source['new253_source_and_gain_checks_passed'] is True,'Original and new source checks')
    prior_dir=ROOT/'outputs/monthly_risk_clock263'
    prior_close=ev.read(prior_dir/'COMPLETION.json',ev.registered(plan,prior_dir/'COMPLETION.json'))
    need(prior_close['status']=='COMPLETED_INDEPENDENTLY_VERIFIED','Registered predecessor closed')
    prior_result=ev.read(prior_dir/'results/COMPLETION.json',prior_close['result_completion_sha256'])
    need(prior_result['status']=='COMPLETE' and prior_result['plan_sha256']==prior_close['plan_sha256'],'Prior result/close chain')
    prior_frozen=ev.read(prior_dir/'selected/FROZEN.json',prior_result['frozen_sha256'])
    prior_prep=ev.read(prior_dir/'prepared/COMPLETION.json',prior_frozen['prepared_sha256'])
    need(prior_frozen['plan_sha256']==prior_prep['plan_sha256']==prior_close['plan_sha256'],'Prior selection/preparation chain')
    prior_orders=ev.read(prior_dir/'prepared/ORDERS.json',prior_prep['artifacts_sha256']['ORDERS.json'])
    need(orders[ARMS[0]]==prior_orders[ARMS[0]],'Original incumbent ordered decisions unchanged')
    prior=ev.read(prior_dir/'results/RESULTS.json',prior_result['artifacts_sha256']['RESULTS.json'])
    prior_cost_name=ARMS[0]+'_next_open_cost_HOLDINGS.json'
    prior_cost_holdings=ev.read(prior_dir/'results'/prior_cost_name,prior_result['artifacts_sha256'][prior_cost_name])
    summary_fee_display_review=diagnostic_summary_parity(results['arms'][ARMS[0]],prior['arms'][ARMS[0]],exports[ARMS[0],'next_open_cost']['HOLDINGS'],prior_cost_holdings)
    exact_export_files=[];fee_display_review=None
    for path in PATHS:
        for kind in ('DAILY','TRADES','HOLDINGS','DAILY_POSITIONS'):
            name=ARMS[0]+'_'+path+'_'+kind+'.json'
            previous=ev.read(prior_dir/'results'/name,prior_result['artifacts_sha256'][name])
            current=exports[ARMS[0],path][kind]
            if path=='next_open_cost' and kind=='HOLDINGS':
                need(results['arms'][ARMS[0]][path]['metrics']['cost_per_side']==.0005,'Diagnostic cost rate remains exact0.05percent per side')
                fee_display_review=diagnostic_holdings_parity(current,previous)
                if current==previous:exact_export_files.append(name)
            else:
                need(current==previous,'Original incumbent complete economic export unchanged: '+name)
                exact_export_files.append(name)
    need(len(exact_export_files)>=11 and fee_display_review is not None,'All twelve incumbent export comparisons')
    return dict(schema='monthly_gain_survival264_root_saved_review_v5',created_utc=datetime.now(timezone.utc).isoformat(),passed=True,unresolved_blockers=[],
       reviewer_role='Root saved-evidence reviewer; independent264 verifier supplies separately authored numerical and fixed-unit accounting proof.',
       plan_sha256=args.plan_sha256,context_sha256=plan['context_sha256'],implementation_contract_sha256=plan['implementation_contract_sha256'],prepared_completion_sha256=args.prepared_sha256,results_completion_sha256=args.results_sha256,phase_terminals_sha256=args.terminals_sha256,
       root_owned_terminals=terminals,selection=pick,saved_accounting_evidence=accounting,decision_counts=decision['counts'],execution_accounting=evidence['execution_accounting'],saved_source_work_review=source_work_review,
       full_results={a:{path:results['arms'][a][path]['metrics'] for path in PATHS} for a in ARMS},
       segmented_results={a:{path:results['arms'][a][path]['slices'] for path in PATHS} for a in ARMS},
       changed_holdings={r:p.changes(orders,r,ARMS[2]) for r in ARMS[:2]},
       original_core_certificate_pinned_across_cached_prefixes=True,original_incumbent_saved_paths_exact=len(exact_export_files)==12,original_incumbent_economic_exports_exact=True,original_incumbent_summary_metrics_exact=summary_fee_display_review['exact'],economic_summary_metrics_exact=True,summary_fee_display_review=summary_fee_display_review,exact_incumbent_export_files=exact_export_files,diagnostic_fee_display_review=fee_display_review,actual_models_computed_by_reviewer=False,npz_decoded=False,source_prices_loaded=False,actual_portfolio_replay=False,saved_nav_metrics_recomputed=False,phase_rerun=False,
       strict_pit=False,reused_sample=True,observers_modified=False,
       limitations=c['limitations'],input_sha256=ev.pins)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('plan','prepared','frozen','results','terminals'):parser.add_argument('--'+key+'-sha256',required=True)
    args=parser.parse_args()
    print(json.dumps(run(args),indent=2,allow_nan=False))
