"""Read-only268 reconciliation of frozen artifacts, phase receipts and accounts.

No NPZ decode, fitting, prediction, source-price reading or portfolio replay.
The numerical selector verifier was separately authored; this root review is
not blind. Reused accounting rules are role-adapted without global mutation.
"""
import argparse
from datetime import datetime, timezone
import json
import hashlib
import math
from pathlib import Path
import platform
import monthly_monotone_tree266_saved_reader as old
from monthly_boundary_pair267_saved_reader import Evidence

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/monthly_selector_scope268'
PRIOR=ROOT/'outputs/monthly_boundary_pair267'
CONTEXT=ROOT/'work/monthly_selector_scope268_context.json'
CONTRACT=ROOT/'work/monthly_selector_scope268_IMPLEMENTATION_CONTRACT.json'
TESTS=ROOT/'work/monthly_selector_scope268_TESTS.json'
REVIEW=ROOT/'work/monthly_selector_scope268_CODE_REVIEW.json'
ARMS=('incumbent_local67','unlocked_top5_gap','unlocked_all')
PATHS=('same_close_zero','next_open_zero','next_open_cost')
KINDS=('DAILY','TRADES','HOLDINGS','DAILY_POSITIONS')
PREP={'SCORES.npz','PRE_SCORES.npz','ORDERS.json','META.json','DECISION_AUDIT.json','SOURCE_PARITY.json','INDEPENDENT_PREPARE.json','SOURCE_AUDIT.json'}
RESULT={'RESULTS.json','INDEPENDENT_OUTCOME.json'}|{a+'_'+p+'_'+k+'.json' for a in ARMS for p in PATHS for k in KINDS}
FORMAL=tuple('work/'+n for stem in ('host','kernel','independent','admission') for n in ('monthly_selector_scope268_'+stem+'.py','test_monthly_selector_scope268_'+stem+'.py'))
SUPPLEMENTAL=('work/monthly_selector_scope268_saved_reader.py','work/test_monthly_selector_scope268_saved_reader.py')
PRODUCER=('original_pre_score_calls','original_selector_calls','experimental_selector_calls','selector_events','new_model_fits','new_label_values_computed','new_price_values_read')
INTEGRITY=('checks','cache_values_compared','reference_values_compared','reference_order_events_compared','metadata_events_compared')
WORK=('selector_calls','event_rows','eligible_cells','source_rank_calls','support_evaluations','delta_evaluations','selection_rank_calls','partial_events','two_eligible_events','cash_events','new_model_fits')
FLAGS=('original_rank_and_symbol_ties_exact','scope_and_gap_exact','existing_support_budget_arithmetic_exact','partial_and_small_universe_identity_exact','scores_float64_bytes_exact','ordered_choices_exact')
FIXED={'work/monthly_monotone_tree266_saved_reader.py':'eb117fa95c4d7940b96797b1a5a2bd08320b64f3663660a9569915f624e56a8d','work/monthly_boundary_pair267_saved_reader.py':'29eb059828074123afdf6365c914b5ccb004fa3e0b1636a77dbef4050f73a06d','work/monthly_monotone_tree266_admission.py':'1b4199a384ac9ffaa43eb34775dd2c3ae094b743b55a3622ee6ac4f472050baf','work/ptr169_results_independent.py':'6264571ad6ddacbc408449180c74a32b37659757c9bdfc5ce0e2672467607d59','work/monthly_gain_concentration265_saved_reader.py':'8990f2e15e423808546da661ae36717a6dd83dd71fee377f918d55599a2decd8'}
need,exact,digest,local=old.need,old.exact,old.digest,old.local


def structured_digest(value):
    """Original typed digest for JSON-only ledger, no arrays decoded."""
    h=hashlib.sha256()
    def walk(v):
        need(type(v) in (dict,list,tuple,str,int,float,bool,type(None)),'JSON-only digest')
        h.update(type(v).__name__.encode()+b':')
        if type(v) is dict:
            for k in sorted(v):walk(k);walk(v[k])
        elif type(v) in (list,tuple):
            for x in v:walk(x)
        else:h.update(repr(v).encode())
    walk(value);return h.hexdigest()


def roles(x,to_old=True):
    a,b=(ARMS,old.ARMS) if to_old else (old.ARMS,ARMS)
    need(type(x) is dict and set(x)==set(a),'Exact three roles')
    return {j:x[i] for i,j in zip(a,b)}


class Accounts:
    def __init__(self,ev):
        self.ev=ev
        self.paths={old.OUT/'results'/(a+'_'+p+'_'+k+'.json'):OUT/'results'/(b+'_'+p+'_'+k+'.json') for a,b in zip(old.ARMS,ARMS) for p in PATHS for k in KINDS}
    def read(self,path):
        need(path in self.paths,'Explicit account export path')
        return self.ev.read(self.paths[path])


def accounting(ev,result,ind,training):
    result=dict(result,arms=roles(result['arms']));ind=dict(ind,paths=roles(ind['paths']))
    report,exports=old.accounting(Accounts(ev),result,ind,roles(training))
    names=dict(zip(old.ARMS,ARMS))
    return report,{(names[a],p):v for (a,p),v in exports.items()}


def selection(metrics,orders):
    need(set(metrics)==set(orders)==set(ARMS),'Three selection roles')
    invalid={a:old.required_metrics(metrics[a],a==ARMS[0]) for a in ARMS}
    need(all(len(orders[a])>=88 for a in ARMS),'First88 orders')
    for a in ARMS:
        for i in range(88):
            symbols=orders[a][i]['symbols']
            need(type(symbols) is list and len(symbols) in (0,2,3) and all(type(s) is str and s for s in symbols) and len(set(symbols))==len(symbols),'Well-formed first88 holdings')
    changes={a:[i for i in range(88) if set(orders[a][i]['symbols'])!=set(orders[ARMS[0]][i]['symbols'])] for a in ARMS[1:]}
    guards={};qualified=[]
    for a in ARMS[1:]:
        guards[a]=dict(close=bool(not invalid[a] and old.guard(metrics[a]['same_close_zero'],metrics[ARMS[0]]['same_close_zero'],.005)),open=bool(not invalid[a] and old.guard(metrics[a]['next_open_zero'],metrics[ARMS[0]]['next_open_zero'],0.)),support=len(changes[a])>=3 and any(i<53 for i in changes[a]) and any(i>=53 for i in changes[a]))
        if all(guards[a].values()):qualified.append(a)
    primary=ARMS[0]
    if qualified:
        best=max(metrics[a]['same_close_zero']['training']['cagr'] for a in qualified)
        primary=next(a for a in ARMS[1:] if a in qualified and best-metrics[a]['same_close_zero']['training']['cagr']<=1e-12)
    return dict(primary=primary,admitted=bool(qualified),qualified=qualified,guards=guards,changed_months_by_candidate=changes,undefined_sharpe=invalid,first88_only=True,later34_used=False,cost_used=False,tie_rule='fixed_arm_order_with_1e-12_cagr_tolerance')


def counts(row,fields):
    need(type(row) is dict and set(row)==set(fields),'Exact count fields')
    for k,v in row.items():old.integer(v,'Counter '+k)


def summed(rows,fields):
    result={k:0 for k in fields}
    for r in rows:
        counts(r,fields)
        for k,v in r.items():result[k]+=v
    return result


def entry(e,n,w,*,cached=False):
    need(set(e)=={'prefix','recomputed_original_pre','reuses_main_verified_full','source_audit','independent','producer_counts'},'Exact entry fields')
    need(e['prefix']==n and e['recomputed_original_pre'] is (not cached),'Prefix/fresh clock')
    p=e['producer_counts'];counts(p,PRODUCER)
    exact(p,dict(original_pre_score_calls=int(not cached),original_selector_calls=1,experimental_selector_calls=2,selector_events=3*n,new_model_fits=0,new_label_values_computed=0,new_price_values_read=0),'Actual producer work')
    s=e['source_audit']
    need(s['schema']=='selector_scope268_source_audit_v1' and s['events']==n and s['width']==w and s['cached_original_pre'] is cached and s['original_pre_score_calls']==int(not cached),'Source scope')
    for k in ('original_pre_scores_exact','original_eligibility_exact','original_scores_exact','original_orders_exact','original_gate_before_candidates'):need(s[k] is True,'Source flag '+k)
    for k in ('source_files_authenticated','new_features_computed','new_labels_computed','new_price_windows_read'):need(s[k] is False,'Source scope limit '+k)
    old.integer(s['new_model_fits'],'Zero new model fits',0)
    sw=s['source_integrity_work'];counts(sw,INTEGRITY)
    need(sw['checks']==4 and sw['reference_values_compared']==12*n*w and sw['reference_order_events_compared']==sw['metadata_events_compared']==4*n,'Source integrity actual work')
    need(sw['cache_values_compared']>=4*n*w,'Source cache cells')
    need(set(e['independent'])==set(ARMS),'Independent roles')
    for a,r in e['independent'].items():
        need(r['schema']=='selector_scope268_independent_v1' and r['passed'] is True and r['mode']==a and r['events']==n and r['width']==w,'Independent receipt')
        need(set(r['flags'])==set(FLAGS) and all(v is True for v in r['flags'].values()),'Independent flags')
        need(r['numerical_tolerance']==0. and r['source_provenance_verified'] is False and r['actual_coverage_claim'] is False,'Independent exact arithmetic / no provenance claim')
        z=r['work'];counts(z,WORK)
        need(z['selector_calls']==1 and z['event_rows']==z['source_rank_calls']==n and z['new_model_fits']==0,'Independent invocation work')
        need(z['support_evaluations']==z['delta_evaluations']<=z['eligible_cells']<=n*w,'Independent consumed-cell counts')
        need(z['selection_rank_calls']+z['two_eligible_events']+z['cash_events']==n and z['partial_events']<=min(1,n),'Independent event accounting')


def preparation(proof,audit,source,meta,*,total=122,width=49,prefixes=(0,1,53,88,122),orders=None):
    n=len(meta['dates']);w=len(meta['symbols'][0]);need(n==total and w==width,'Full original dimensions')
    need(proof['schema']=='selector_scope268_independent_prepare_v1','Preparation schema')
    for k in ('all123_order_prefixes','source_arrays_unchanged','original_reference_unchanged'):need(proof[k] is True,'Preparation flag '+k)
    entry(proof['main'],n,w);need(proof['main']['reuses_main_verified_full'] is False,'Main genuinely computed')
    need([e['prefix'] for e in proof['cached_prefixes']]==list(range(n+1)),'All cached prefixes')
    for e in proof['cached_prefixes']:entry(e,e['prefix'],w,cached=True);need(e['reuses_main_verified_full'] is False,'Cached not full alias')
    need([e['prefix'] for e in proof['physical_prefixes']]==list(prefixes) and prefixes[-1]==n,'Physical prefix set')
    for e in proof['physical_prefixes']:
        entry(e,e['prefix'],w)
        need(e['reuses_main_verified_full'] is (e['prefix']==n),'Physical full alias')
    expected=dict(proof['main'],reuses_main_verified_full=True);exact(proof['physical_prefixes'][-1],expected,'Full reused without new counts')
    entries=[proof['main']]+proof['cached_prefixes']+proof['physical_prefixes'][:-1]
    work=proof['execution_accounting'];need(work['actual_prepare_entries']==len(entries)==n+1+len(prefixes) and work['physical_full_reuses_main'] is True and work['new_model_fits']==0,'Actual entry count')
    exact(work['producer'],summed([e['producer_counts'] for e in entries],PRODUCER),'Producer totals')
    exact(work['independent'],summed([r['work'] for e in entries for r in e['independent'].values()],WORK),'Independent totals')
    exact(work['source_integrity_work'],summed([e['source_audit']['source_integrity_work'] for e in entries],INTEGRITY),'Source totals')
    exact(source,dict(main=proof['main']['source_audit'],physical=[dict(prefix=e['prefix'],audit=e['source_audit']) for e in proof['physical_prefixes']],cached=[dict(prefix=e['prefix'],audit=e['source_audit']) for e in proof['cached_prefixes']]),'Source separate artifact')
    need(audit['schema']=='selector_scope268_decision_audit_v1' and audit['events']==n and audit['width']==w and set(audit['arms'])==set(ARMS),'Decision audit')
    exact(audit['producer_counts'],proof['main']['producer_counts'],'Decision actual counts')
    exact(audit['arms'][ARMS[0]],dict(source='frozen_original_select_boundary',selector_calls=1,events=n),'Original selector called')
    for a in ARMS[1:]:
        decision(audit['arms'][a],proof['main']['independent'][a],meta,a,None if orders is None else orders[a])
    events=audit['arms'][ARMS[1]]['events']
    for e in entries+[proof['physical_prefixes'][-1]]:
        for a in ARMS:
            exact(e['independent'][a]['work'],prefix_work(events,a,e['prefix']),'Every prefix/arm actual primitive count')
    return work


def prefix_work(events,arm,n):
    need(arm in ARMS and 0<=n<=len(events),'Prefix work scope')
    out={k:0 for k in WORK};out.update(selector_calls=1,event_rows=n,source_rank_calls=n)
    for i in range(n):
        e=events[i];rank=e['original_rank'];out['eligible_cells']+=len(rank);out['partial_events']+=int(e['initial_partial'])
        if len(rank)<3:
            out['two_eligible_events' if len(rank)==2 else 'cash_events']+=1
        else:
            out['selection_rank_calls']+=1
            mode=ARMS[0] if e['initial_partial'] else arm
            count=len(e['original_boundary_indices']) if mode==ARMS[0] else 2+len(e['original_boundary_indices']) if mode==ARMS[1] else len(rank)
            out['support_evaluations']+=count;out['delta_evaluations']+=count
    return out


def decision(audit,ind,meta,arm,orders=None):
    n=len(meta['dates']);w=len(meta['symbols'][0]);c=audit['counts']
    need(audit['schema']=='monthly_selector_scope268_decisions_v1' and audit['mode']==arm and len(audit['events'])==n,'Candidate audit identity')
    for key in ('original_inputs_preserved','long_core_unchanged','original_eligibility_unchanged','existing_effects_only','initial_partial_exact'):need(audit[key] is True,'Candidate decision '+key)
    need(audit['computes_returns'] is False,'No selection returns')
    for key,value in c.items():old.integer(value,'Producer counter '+key)
    need(set(c)=={'events','symbols_per_event','eligible_cells','initial_partial_events','regular_events','three_or_more_events','two_name_events','cash_events','original_boundary_cells','adjusted_cells','nonzero_delta_cells','changed_score_cells','top2_exit_events','original_rankings','final_rankings','boundary_winner_reductions','selector_calls','new_model_fits','predictive_model_fits'},'Exact producer count fields')
    if orders is not None:need(len(orders)==n,'Decision order count')
    need(c['events']==n and c['symbols_per_event']==w and c['selector_calls']==1 and c['new_model_fits']==c['predictive_model_fits']==0,'Candidate totals')
    need(c['eligible_cells']==ind['work']['eligible_cells'] and c['adjusted_cells']==ind['work']['support_evaluations'],'Cross independent consumed cells')
    counts={k:0 for k in ('eligible_cells','adjusted_cells','initial_partial_events','three_or_more_events','two_name_events','cash_events','original_boundary_cells','top2_exit_events','nonzero_delta_cells','changed_score_cells','boundary_winner_reductions')}
    for i,e in enumerate(audit['events']):
        rank=e['original_rank'];boundary=e['original_boundary_indices'];adj=e['adjustment_indices'];domain=e['selection_indices'];selected=e['selected_indices']
        need(e['query_index']==i and e['date']==meta['dates'][i] and e['initial_partial'] is (meta['dates'][i]=='2016-08-17'),'Event clock')
        effective=ARMS[0] if e['initial_partial'] else arm;need(e['effective_mode']==effective,'Initial incumbent scope')
        for ids in (rank,boundary,adj,domain,selected,e['eligible_indices'],e['top2_exited_indices']):need(type(ids) is list and len(set(ids))==len(ids) and all(type(j) is int and 0<=j<w for j in ids),'Event indices')
        need(set(rank)==set(e['eligible_indices']) and set(boundary)<=set(rank[2:5]),'Original domain subset')
        if orders is not None:exact([meta['symbols'][i][j] for j in selected],orders[i]['symbols'],'Decision indices map to ordered holdings')
        need(len(set(e['changed_score_indices']))==len(e['changed_score_indices']) and set(e['changed_score_indices'])<=set(adj),'Changed cells within consumed domain')
        need(all(old.finite(v) for key in ('adjustment_support','adjustment_effects','adjustment_targets','adjustment_deltas') for v in e[key]),'Finite saved consumed arithmetic')
        counts['nonzero_delta_cells']+=sum(v!=0. for v in e['adjustment_deltas']);counts['changed_score_cells']+=len(e['changed_score_indices']);counts['boundary_winner_reductions']+=int(effective==ARMS[0] and len(rank)>=3)
        counts['eligible_cells']+=len(rank);counts['adjusted_cells']+=len(adj);counts['initial_partial_events']+=int(e['initial_partial']);counts['original_boundary_cells']+=len(boundary);counts['top2_exit_events']+=bool(e['top2_exited_indices'])
        if len(rank)<3:
            need(not adj and not domain and not boundary and selected==(rank if len(rank)==2 else []),'Small domain identity')
            counts['two_name_events' if len(rank)==2 else 'cash_events']+=1
        else:
            counts['three_or_more_events']+=1
            wanted=boundary if effective==ARMS[0] else rank[:2]+boundary if effective==ARMS[1] else rank
            wanted_domain=rank[:2]+boundary if effective!=ARMS[2] else rank
            need(adj==wanted and domain==wanted_domain and len(selected)==3 and set(selected)<=set(domain),'Recorded selector domain')
            if effective==ARMS[0]:need(set(rank[:2])<=set(selected),'Incumbent top2 retained')
        need(e['top2_exited_indices']==[j for j in rank[:2] if j not in selected] if len(rank)>=3 else e['top2_exited_indices']==[],'Top2 exit evidence')
        need(all(len(e[k])==len(adj) for k in ('adjustment_support','adjustment_effects','adjustment_targets','adjustment_deltas')),'Consumed arithmetic evidence')
    for key,value in counts.items():need(c[key]==value,'Recorded total '+key)
    need(c['regular_events']+c['initial_partial_events']==n and c['original_rankings']==n and c['final_rankings']==c['three_or_more_events'],'Event totals')
    for source,target in [('initial_partial_events','partial_events'),('three_or_more_events','selection_rank_calls'),('two_name_events','two_eligible_events'),('cash_events','cash_events'),('original_rankings','source_rank_calls')]:need(c[source]==ind['work'][target],'Independent exact event counter '+target)


def terminals(record,sha,hashes):
    need(record['plan_sha256']==sha and set(record['phases'])==set(hashes),'Root phase terminals')
    for phase,name in [('prepare','prepared/COMPLETION.json'),('select','selected/FROZEN.json'),('diagnose','results/COMPLETION.json')]:
        r=record['phases'][phase]
        need(set(r)=={'session_id','exit_code','chunk_id','artifact_path','artifact_sha256'},'Terminal fields')
        need(r['session_id'] is None or type(r['session_id']) is int and r['session_id']>0,'Session identity')
        old.integer(r['exit_code'],'Actual exit',0)
        need(type(r['chunk_id']) is str and bool(r['chunk_id'].strip()) and local(r['artifact_path'])==OUT/name and r['artifact_sha256']==hashes[phase],'Terminal digest')


def formal(ev,plan):
    for name,h in FIXED.items():need(ev.registered(plan,name)==h,'Shared frozen helper');ev.pin(name,h)
    context=ev.read(CONTEXT,ev.registered(plan,CONTEXT));contract=ev.read(CONTRACT,ev.registered(plan,CONTRACT))
    exact(plan['context'],context,'Embedded context');exact(plan['implementation_contract'],contract,'Embedded contract')
    need(plan['context_sha256']==digest(CONTEXT)==contract['context_sha256'] and plan['implementation_contract_sha256']==digest(CONTRACT),'Paper digests')
    need(contract['code_files']==list(FORMAL) and contract['supplemental_input_files']==list(SUPPLEMENTAL),'8+2 code inventory')
    for paper in (context,contract):
        need(paper['frozen'] is True and paper['registration_authorized'] is True and paper['actual_data_started'] is False,'Prospective paper')
        for name in FORMAL+SUPPLEMENTAL:need(paper['input_sha256'][name]==ev.registered(plan,name)==ev.pin(name),'Paper code binding')
    exact(context['arms'],list(ARMS),'Fixed roles');exact(context['promotable_arms'],list(ARMS[1:]),'Two eligible candidates')
    need(context['registered_trials']==3 and context['actual_models_fitted']==0 and context['actual_orders_computed'] is False and context['actual_returns_computed'] is False,'Prospective scope')
    tests=ev.read(TESTS,ev.registered(plan,TESTS));review=ev.read(REVIEW,ev.registered(plan,REVIEW))
    expected={str(ROOT/p):digest(ROOT/p) for p in FORMAL}
    need(tests['passed'] is True and tests['synthetic_only'] is True and tests['tests']>0 and tests['code_sha256']==expected,'Actual formal tests')
    need(tests['runtime']==plan['runtime']==dict(python='3.9.6',numpy='2.0.2',pandas='2.3.3'),'Pinned numerical runtime')
    need(review['passed'] is True and review['unresolved_blockers']==[] and review['code_sha256']==expected and review['actual_returns_computed'] is False,'Pre-registration code review')
    need(review['context_sha256']==digest(CONTEXT) and review['implementation_contract_sha256']==digest(CONTRACT),'Review bound papers')
    for path,h in review['input_sha256'].items():need(ev.registered(plan,path)==h,'Review pin');ev.pin(path,h)
    return context,tests


def source_proof(ev,plan,source,meta):
    original=ev.read(old.LOCAL/'PLAN.json',ev.registered(plan,old.LOCAL/'PLAN.json'))
    maps=[old.canonical_source_map(v) for v in (source['raw_source_hashes'],plan['raw_source_hashes'],original['raw_source_hashes'])]
    exact(maps[0],maps[1],'Source raw64 plan');exact(maps[1],maps[2],'Source raw64 original')
    for key,name in [('original_training_hash','TRAIN_INPUTS.npz'),('original_identity_hash','MONTHLY_META.json'),('saved169_parity_hash','PARITY.json')]:
        need(source[key]==ev.registered(plan,old.REBUILD/name),'Registered source pin');ev.pin(old.REBUILD/name,source[key])
    for key in ('original_arrays_unchanged','original_orders_exact','original_scores_exact','original_pre_scores_exact','original_eligibility_exact','original_gate_before_candidates','whole_original_cache_decoded'):need(source[key] is True,'Exact source '+key)
    for key in ('training_arrays_decoded','new_features_computed','new_labels_computed','new_price_windows_read','strict_pit'):need(source[key] is False,'Source limited scope '+key)
    old.integer(source['new_model_fits'],'Zero new fits',0)
    closure=source['original169_source_closure'];exact(closure,plan['original169_source_closure'],'Registered169 closure')
    for p,h in closure['chain_sha256'].items():need(ev.registered(plan,p)==h,'169 chain registered');ev.pin(p,h)
    parity=ev.read(old.REBUILD/'PARITY.json');p=ev.read(old.REBUILD.parent/'PLAN.json')
    need(parity['passed'] is True and parity['returns_computed'] is False and parity['source_plan_sha256']==digest(old.REBUILD.parent/'PLAN.json'),'Original169 source parity')
    need(len(parity['artifacts_sha256'])==12 and parity['artifacts_sha256']==closure['artifacts_sha256'],'All12 source artifacts')
    for name,h in parity['artifacts_sha256'].items():need(Path(name).name==name and ev.registered(plan,old.REBUILD/name)==h,'Registered source member');ev.pin(old.REBUILD/name,h)
    need(closure['source_input_count']==len(p['input_sha256']),'169 source inventory')
    for path,h in p['input_sha256'].items():need(ev.registered(plan,path)==h,'Inherited169 input binding')
    old.full_original_chain(ev,plan)
    need(source['prior_prepared_sha256']==ev.registered(plan,PRIOR/'prepared/COMPLETION.json'),'Prior original cache')
    exact(meta,ev.read(old.REBUILD/'MONTHLY_META.json')['monthly'],'Original169 identity')
    fingerprints=source['original_array_fingerprints'];need(type(fingerprints) is dict and len(fingerprints)==27,'Original27 inventory')
    for item in fingerprints.values():
        need(set(item)=={'dtype','shape','sha256'} and item['shape'][0]==122 and old.sha_value(item['sha256']),'Original array fingerprint')
    calendar=ev.read(old.CALENDAR,ev.registered(plan,old.CALENDAR));need(source['calendar_sha256']==plan['calendar_sha256']==digest(old.CALENDAR),'Official calendar binding')
    axis=calendar['sessions'];need(axis==sorted(set(axis)),'Official sessions');positions={d:i for i,d in enumerate(axis)}
    for i,d in enumerate(meta['dates']):need(d in positions and (i==0 or positions[d]+1<len(axis) and axis[positions[d]+1][:7]!=d[:7]),'Monthend signal calendar')
    return axis,sum(math.prod(item['shape'][1:]) for item in fingerprints.values())


def run(args):
    need(platform.python_version()=='3.9.6','Pinned runtime')
    for name in ('plan','prepared','frozen','results','terminals'):need(old.sha_value(getattr(args,name+'_sha256')),'Explicit digest '+name)
    ev=Evidence();plan=ev.read(OUT/'PLAN.json',args.plan_sha256);context,tests=formal(ev,plan)
    runtime=ev.read(old.RUNTIME,ev.registered(plan,old.RUNTIME));need(len(runtime['input_sha256'])==3309,'Pinned numeric package inventory')
    for path,h in runtime['input_sha256'].items():need(ev.registered(plan,path)==h,'Registered runtime closure')
    protected=ev.read(old.PROTECTED,ev.registered(plan,old.PROTECTED));need(len(protected)==plan['protected_files']==21444,'Protected inventory')
    before=ev.read(OUT/'STATE_BEFORE.json',plan['prior_central_sha256']);need(len(before['completed'])==263 and sum(x['trials'] for x in before['completed'])==1300 and before['registered_queue']==[],'Central pre-state')
    prior=ev.read(PRIOR/'COMPLETION.json','b429eebc44946764bd1731383332ae1c33911b40550f376311cfedd9230e828e')
    need(prior['status']=='COMPLETED_INDEPENDENTLY_VERIFIED' and prior['protected_files']==21444 and prior['observers_modified'] is False,'Closed predecessor')
    need(before['completed'][-1]['directory']==PRIOR.name and before['completed'][-1]['completion']['sha256']==digest(PRIOR/'COMPLETION.json')==context['predecessor']['completion_sha256'],'Immediate central predecessor')
    need(prior['plan_sha256']==ev.pin(PRIOR/'PLAN.json',ev.registered(plan,PRIOR/'PLAN.json')),'Prior PLAN binding')
    priorreview=ROOT/'work/monthly_boundary_pair267_RESULT_REVIEW.json'
    need(prior['result_review_sha256']==context['predecessor']['result_review_sha256']==ev.pin(priorreview,ev.registered(plan,priorreview)),'Prior saved review binding')
    terminal=ev.read(OUT/'PHASE_TERMINALS.json',args.terminals_sha256);terminals(terminal,args.plan_sha256,dict(prepare=args.prepared_sha256,select=args.frozen_sha256,diagnose=args.results_sha256))
    pc=ev.phase(OUT/'prepared',args.prepared_sha256,args.plan_sha256,PREP)
    need(pc['events']==122 and pc['new_returns_computed'] is False,'Prepared scope')
    need(pc['new_labels_computed'] is False,'No new labels');old.integer(pc['new_model_fits'],'Prepared zero fits',0)
    frozen=ev.read(OUT/'selected/FROZEN.json',args.frozen_sha256);training=ev.read(OUT/'selected/TRAIN_RESULTS.json',frozen['training_sha256'])
    need(frozen['plan_sha256']==args.plan_sha256 and frozen['prepared_sha256']==args.prepared_sha256 and frozen['orders_sha256']==pc['artifacts_sha256']['ORDERS.json'],'Frozen first88')
    need({p.name for p in (OUT/'selected').iterdir()}=={'FROZEN.json','TRAIN_RESULTS.json'},'Selected artifact names')
    rc=ev.phase(OUT/'results',args.results_sha256,args.plan_sha256,RESULT);need(rc['frozen_sha256']==args.frozen_sha256 and rc['paths']==9 and rc['slices']==45,'Nine complete paths')
    meta=ev.read(OUT/'prepared/META.json');orders=ev.read(OUT/'prepared/ORDERS.json')
    need(len(meta['dates'])==len(meta['ends'])==len(meta['symbols'])==122 and meta['ends'][:-1]==meta['dates'][1:] and meta['dates'][0]=='2016-08-17' and meta['ends'][-1]=='2026-09-24','Full clock')
    need(meta['ends'][52]=='2020-12-31' and meta['ends'][87]=='2023-11-30' and all(len(s)==49 and len(set(s))==49 for s in meta['symbols']),'Segments and universe')
    need(set(orders)==set(ARMS),'Order roles')
    for arm in ARMS:
        need(len(orders[arm])==122 and [o['date'] for o in orders[arm]]==meta['dates'] and [o['end'] for o in orders[arm]]==meta['ends'],'Order dates')
        for o in orders[arm]:need(o['signal_date']==o['date'] and len(o['symbols']) in (0,2,3) and len(o['symbols'])==len(set(o['symbols'])) and len(o['scores'])==len(o['symbols']) and all(old.finite(x) for x in o['scores']),'Order decisions')
    priorpc=ev.read(PRIOR/'prepared/COMPLETION.json',ev.registered(plan,PRIOR/'prepared/COMPLETION.json'))
    priororders=ev.read(PRIOR/'prepared/ORDERS.json',priorpc['artifacts_sha256']['ORDERS.json']);exact(orders[ARMS[0]],priororders[ARMS[0]],'Incumbent source')
    exact(meta,ev.read(PRIOR/'prepared/META.json',priorpc['artifacts_sha256']['META.json']),'Source metadata')
    source=ev.read(OUT/'prepared/SOURCE_PARITY.json');need(source['original_arrays_unchanged'] is True and source['original_orders_exact'] is True and source['original_scores_exact'] is True and source['original_pre_scores_exact'] is True and source['original_gate_before_candidates'] is True,'Source parity flags')
    axis,rowwidth=source_proof(ev,plan,source,meta)
    exact(orders[ARMS[0]],ev.read(old.LOCAL/'local_linear_ORDERS.json',ev.registered(plan,old.LOCAL/'local_linear_ORDERS.json')),'Original Local67 holdings')
    proof=ev.read(OUT/'prepared/INDEPENDENT_PREPARE.json')
    work=preparation(proof,ev.read(OUT/'prepared/DECISION_AUDIT.json'),ev.read(OUT/'prepared/SOURCE_AUDIT.json'),meta,orders=orders)
    for e in [proof['main']]+proof['cached_prefixes']+proof['physical_prefixes']:
        need(e['source_audit']['source_integrity_work']['cache_values_compared']==4*e['prefix']*rowwidth,'All27 prefix source values')
    need(pc['execution_accounting_sha256']==structured_digest(work),'Prepared actual-work binding')
    selected=selection(training,orders);exact(selected,frozen['selection'],'Independent first88 selection')
    result=ev.read(OUT/'results/RESULTS.json');ind=ev.read(OUT/'results/INDEPENDENT_OUTCOME.json');exact(result['selection'],selected,'Result frozen choice')
    need(result['strict_pit'] is False and result['reused_sample'] is True and result['observer_promotion'] is False and result['zero_cost_selection'] is True and result['next_open_cost_diagnostic_only'] is True,'Declared research limits')
    need(ind['accounting_source_sha256']==FIXED['work/ptr169_results_independent.py'],'Independent accountant source')
    for arm in ARMS:
        for path in PATHS[:2]:
            for m in (result['arms'][arm][path]['metrics'],result['arms'][arm][path]['slices']['full122']):need(m['cost_per_side']==0. and m['fees_paid']==0.,'Exact ZERO headline')
    account,exports=accounting(ev,result,ind,training)
    for arm in ARMS:
        for path in PATHS:
            exact([r['date'] for r in exports[arm,path]['DAILY']],[d for d in axis if meta['dates'][0]<=d<=meta['ends'][-1]],'Official daily account calendar')
    prc=ev.read(PRIOR/'results/COMPLETION.json',prior['result_completion_sha256']);pr=ev.read(PRIOR/'results/RESULTS.json',prc['artifacts_sha256']['RESULTS.json']);priorfees=None
    for path in PATHS:
        for kind in KINDS:
            name=ARMS[0]+'_'+path+'_'+kind+'.json';previous=ev.read(PRIOR/'results'/name,prc['artifacts_sha256'][name]);current=exports[ARMS[0],path][kind]
            if path=='next_open_cost' and kind=='HOLDINGS':priorfees=previous
            else:exact(current,previous,'Incumbent economic '+path+'/'+kind)
    fee_review=old.diagnostic_summary_parity(result['arms'][ARMS[0]],pr['arms'][ARMS[0]],exports[ARMS[0],'next_open_cost']['HOLDINGS'],priorfees)
    ev.finish()
    return dict(schema='selector_scope268_saved_reader_v1',created_utc=datetime.now(timezone.utc).isoformat(),passed=True,unresolved_blockers=[],plan_sha256=args.plan_sha256,context_sha256=digest(CONTEXT),implementation_contract_sha256=digest(CONTRACT),prepared_completion_sha256=args.prepared_sha256,results_completion_sha256=args.results_sha256,phase_terminals_sha256=args.terminals_sha256,root_owned_terminals=terminal,formal_synthetic_tests=tests['tests'],execution_accounting=work,selection=selected,saved_accounting_evidence=account,incumbent_economic_exports_exact=True,incumbent_summary_exact=fee_review['exact'],summary_fee_display_review=fee_review,full_results={a:{p:result['arms'][a][p]['metrics'] for p in PATHS} for a in ARMS},segmented_results={a:{p:result['arms'][a][p]['slices'] for p in PATHS} for a in ARMS},npz_decoded=False,actual_models_computed_by_reviewer=False,actual_portfolio_replay=False,source_prices_loaded=False,phase_rerun=False,frozen_files_modified=False,observers_modified=False,strict_pit=False,reused_sample=True,protection_and_close=dict(status='pending_root_close',protected_members_rehashed=False,expected_post_close_completed=264,expected_post_close_trials=1303),limitations=['Root source/host review and separately authored selector verifier are non-blind.','Reader reconciles saved independent results; no new source-array decode, model fit or NAV recomputation.','Expanded scoring scope reuses existing forecasts without retraining; historical data-vintage/universe limitations remain.','Full protected-file verification and central closure still belong to root close.'],input_sha256=ev.pins)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','prepared','frozen','results','terminals'):p.add_argument('--'+name+'-sha256',required=True)
    print(json.dumps(run(p.parse_args()),indent=2,sort_keys=True,allow_nan=False))
