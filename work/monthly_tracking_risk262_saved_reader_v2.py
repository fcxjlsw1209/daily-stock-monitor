"""Root saved-evidence review; no prices/arrays, model fit or portfolio replay.

Reuses the pinned261 standard-library saved-accounting comparison only. The
separately authored262 verifier supplies numerical independence. Prints JSON;
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
OUT=ROOT/'outputs/monthly_tracking_risk262'
PARENT=ROOT/'work/monthly_weak_tree261_saved_reader.py'
PARENT_SHA='0f72873c7eb253cb11d364d9b1a4c206885059c5ce5174e73d1df97ad0900190'
ARMS=('incumbent_local67','sector_total_risk_control','sector_tracking_risk_candidate')
PREP={'FEATURES.json','SCORES.npz','PRE_SCORES.npz','DECISION_AUDIT.json','ORDERS.json','META.json','SOURCE_PARITY.json','INDEPENDENT_PREPARE.json'}
PATHS=('same_close_zero','next_open_zero','next_open_cost')
RESULT={'RESULTS.json','INDEPENDENT_OUTCOME.json'}|{a+'_'+p+'_'+k+'.json' for a in ARMS for p in PATHS for k in ('DAILY','TRADES','HOLDINGS','DAILY_POSITIONS')}

def need(ok,msg):
    if not ok:raise ValueError(msg)

def load_parent():
    need(hashlib.sha256(PARENT.read_bytes()).hexdigest()==PARENT_SHA,'Saved comparison helper changed')
    spec=importlib.util.spec_from_file_location('tracking262_saved_comparison261',PARENT)
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

def run(args):
    p=load_parent();ev=p.Evidence();ev.pin(PARENT,PARENT_SHA);ev.pin(Path(__file__))
    ev.pin(ROOT/'work/monthly_tracking_risk262_saved_reader.py','a1cabb1c1add631d38bc57fe4010edcaf5a8a4d218a84f6c6ee81e84531e4d92')
    ev.pin(ROOT/'outputs/monthly_factor_search_v2/validate.py','b5c6bbbfa5f4db0821f6b0dacb33a43c974cceb8a150578803eef73132cd30e4')
    plan=ev.read(OUT/'PLAN.json',args.plan_sha256)
    context=ROOT/'work/monthly_tracking_risk262_context.json'
    contract=ROOT/'work/monthly_tracking_risk262_IMPLEMENTATION_CONTRACT.json'
    c=ev.read(context,plan['context_sha256']);d=ev.read(contract,plan['implementation_contract_sha256'])
    need(c['frozen'] is True and d['frozen'] is True and c['arms']==list(ARMS),'Frozen prospective scope')
    need(c==plan['context'] and d['context']['sha256']==plan['context_sha256'],'Context binding')
    for name in ('TESTS','CODE_REVIEW'):
        path=ROOT/('work/monthly_tracking_risk262_'+name+'.json')
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
    pick=selection(p,train,orders)
    need(pick==frozen['selection']==results['selection'],'First88-only dual-reference selection exact')
    accounting,exports=p.accounting(ev,results,independent,train)
    evidence=ev.read(OUT/'prepared/INDEPENDENT_PREPARE.json')
    decision=ev.read(OUT/'prepared/DECISION_AUDIT.json')
    source=ev.read(OUT/'prepared/SOURCE_PARITY.json')
    need(evidence['all123_order_prefixes'] is True and evidence['original_arrays_unchanged'] is True and evidence['original_eligible_unchanged'] is True,'Complete prefix/preservation evidence')
    need([x['prefix'] for x in evidence['cached_prefixes']]==list(range(123)),'All123 cached prefixes')
    need([x['prefix'] for x in evidence['physical_prefixes']]==[0,1,53,88,122],'All physical prefixes')
    need(all(x['exact'] is True for x in evidence['cached_prefixes']+evidence['physical_prefixes']),'Exact prefix results')
    need(evidence['new_model_fits']==evidence['predictive_model_fits']==0 and prep['new_returns_computed'] is False,'Preparation scope')
    need(source['original_orders_exact'] is True and source['new65_source_and_canonical_checks_passed'] is True,'Original and new source checks')
    prior_dir=ROOT/'outputs/monthly_weak_tree261'
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
    need(results['arms'][ARMS[0]]==prior['arms'][ARMS[0]],'Original67.67 all saved path summary objects unchanged')
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
    return dict(schema='monthly_tracking_risk262_root_saved_review_v2',created_utc=datetime.now(timezone.utc).isoformat(),passed=True,unresolved_blockers=[],
       reviewer_role='Root saved-evidence reviewer; independent262 verifier supplies separately authored numerical and fixed-unit accounting proof.',
       plan_sha256=args.plan_sha256,context_sha256=plan['context_sha256'],implementation_contract_sha256=plan['implementation_contract_sha256'],prepared_completion_sha256=args.prepared_sha256,results_completion_sha256=args.results_sha256,phase_terminals_sha256=args.terminals_sha256,
       root_owned_terminals=terminals,selection=pick,saved_accounting_evidence=accounting,decision_counts=decision['counts'],execution_accounting=evidence['execution_accounting'],
       full_results={a:{path:results['arms'][a][path]['metrics'] for path in PATHS} for a in ARMS},
       segmented_results={a:{path:results['arms'][a][path]['slices'] for path in PATHS} for a in ARMS},
       changed_holdings={r:p.changes(orders,r,ARMS[2]) for r in ARMS[:2]},
       original_incumbent_saved_paths_exact=len(exact_export_files)==12,original_incumbent_economic_exports_exact=True,original_incumbent_summary_metrics_exact=True,exact_incumbent_export_files=exact_export_files,diagnostic_fee_display_review=fee_display_review,actual_models_computed_by_reviewer=False,npz_decoded=False,source_prices_loaded=False,actual_portfolio_replay=False,saved_nav_metrics_recomputed=False,phase_rerun=False,
       strict_pit=False,reused_sample=True,observers_modified=False,
       limitations=c['limitations'],input_sha256=ev.pins)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('plan','prepared','frozen','results','terminals'):parser.add_argument('--'+key+'-sha256',required=True)
    args=parser.parse_args()
    print(json.dumps(run(args),indent=2,allow_nan=False))
