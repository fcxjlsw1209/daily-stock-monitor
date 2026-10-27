"""Read-only272 saved-evidence audit; no price loading, NAV replay or source admission.

Saved financial arrays and P/F are decoded for separately authored oracle checks.
The reader and producer runner share an author: independent mathematical/selector
proof comes from pinned separately authored modules, not that authorship claim.
"""
import argparse
from datetime import datetime, timezone
import json
import hashlib
import math
from pathlib import Path
import platform
import numpy as np
import monthly_monotone_tree266_saved_reader as old
from monthly_boundary_pair267_saved_reader import Evidence

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/monthly_roa_increment272'
PRIOR=ROOT/'outputs/monthly_joint_growth271'
CONTEXT=ROOT/'work/monthly_roa_increment272_context.json'
CONTRACT=ROOT/'work/monthly_roa_increment272_IMPLEMENTATION_CONTRACT.json'
TESTS=ROOT/'work/monthly_roa_increment272_TESTS.json'
REVIEW=ROOT/'work/monthly_roa_increment272_CODE_REVIEW.json'
ARMS=('incumbent_local67','final154_expanded_earnings_control','exact3_roa_increment_candidate')
PATHS=('same_close_zero','next_open_zero','next_open_cost')
KINDS=('DAILY','TRADES','HOLDINGS','DAILY_POSITIONS')
PREP={'SCORES.npz','PRE_SCORES.npz','ORDERS.json','META.json','DECISION_AUDIT.json','SOURCE_PARITY.json','INDEPENDENT_PREPARE.json','SOURCE_AUDIT.json','FINANCIAL_INPUTS.npz','OVERLAY_AUDIT.json'}
RESULT={'RESULTS.json','INDEPENDENT_OUTCOME.json'}|{a+'_'+p+'_'+k+'.json' for a in ARMS for p in PATHS for k in KINDS}
FORMAL=tuple('work/'+n for stem in ('runner','factor','independent') for n in ('monthly_roa_increment272_'+stem+'.py','test_monthly_roa_increment272_'+stem+'.py'))
SUPPLEMENTAL=('work/monthly_roa_increment272_saved_reader.py','work/test_monthly_roa_increment272_saved_reader.py')
PRODUCER=('original_pre_score_calls','original_selector_calls','financial_correction_calls','overlay_calls','selector_audit_rank_calls','new_model_fits','new_price_windows_read')
INTEGRITY=('checks','cache_values_compared','reference_values_compared','reference_order_events_compared','metadata_events_compared','financial_values_compared')
WORK=('selector_calls','event_rows','eligible_cells','source_rank_calls','support_evaluations','delta_evaluations','selection_rank_calls','partial_events','two_eligible_events','cash_events','new_model_fits')
FLAGS=('original_rank_and_symbol_ties_exact','scope_and_gap_exact','existing_support_budget_arithmetic_exact','partial_and_small_universe_identity_exact','scores_float64_bytes_exact','ordered_choices_exact')
OVERLAY_COUNTS=('events','symbols_per_event','baseline_available_cells','catalog_headers_checked','latest_lookups','selected_increment_cells','event_bindings','required_cell_checks','ratio_evaluations','bounded_evaluations','filled_cells','zero_fills','new_model_fits')
PRE_WORK=('pre_score_verifications','cells_compared','eligible_delta_cells','hand_bound_cells','new_model_fits')
LEGACY=ROOT/'outputs/monthly_financial_coverage154'
INCREMENT=ROOT/'outputs/monthly_roa_field_scope270_increment_v1'
FIXED={'work/monthly_monotone_tree266_saved_reader.py':'eb117fa95c4d7940b96797b1a5a2bd08320b64f3663660a9569915f624e56a8d','work/monthly_boundary_pair267_saved_reader.py':'29eb059828074123afdf6365c914b5ccb004fa3e0b1636a77dbef4050f73a06d','work/monthly_monotone_tree266_admission.py':'1b4199a384ac9ffaa43eb34775dd2c3ae094b743b55a3622ee6ac4f472050baf','work/ptr169_results_independent.py':'6264571ad6ddacbc408449180c74a32b37659757c9bdfc5ce0e2672467607d59','work/monthly_gain_concentration265_saved_reader.py':'8990f2e15e423808546da661ae36717a6dd83dd71fee377f918d55599a2decd8'}
FIXED.update({'work/monthly_selector_scope268_host.py': '91a8a894b50db26e67f47834dac4d4a4a0c5d3ce79583551225ec77a2ea22192', 'work/monthly_selector_scope268_independent.py': '12ec572b1378bbf16fc698890b689523904dd4b7858060a5c4d35619d348d8d9', 'work/monthly_selector_scope268_saved_reader.py': 'ee9d371a0bb144c5072c62acb08e8200e7c18d9db9f7f8dddfdd7591f019b7d2'})
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

def counts(row,fields):
    need(type(row) is dict and set(row)==set(fields),'Exact count fields')
    for k,v in row.items():old.integer(v,'Counter '+k)

def summed(rows,fields):
    result={k:0 for k in fields}
    for r in rows:
        counts(r,fields)
        for k,v in r.items():result[k]+=v
    return result

def indices(xs,w,label,*,ordered=False):
    need(type(xs) is list and all(type(j) is int and 0<=j<w for j in xs) and len(xs)==len(set(xs)),label+' indices')
    if ordered:need(xs==sorted(xs),label+' ascending indices')

def selector_work(events,meta,n,w,orders=None):
    need(0<=n<=len(events),'Selector prefix length')
    out=dict.fromkeys(WORK,0);out['selector_calls']=1
    for i in range(n):
        e=events[i]
        need(type(e) is dict and set(e)=={'query_index','date','eligible_indices','original_rank','boundary_indices','selected_indices'},'Selector event schema')
        old.integer(e['query_index'],'Selector event index',i);need(e['date']==meta['dates'][i],'Selector date')
        eligible,rank,boundary,chosen=(e[k] for k in ('eligible_indices','original_rank','boundary_indices','selected_indices'))
        for k in ('eligible_indices','original_rank','boundary_indices','selected_indices'):indices(e[k],w,k,ordered=k=='eligible_indices')
        need(set(rank)==set(eligible),'Selector original population')
        need(boundary==[j for j in rank[2:5] if j in boundary],'Ordered original boundary subset')
        if len(rank)>=3:
            need(boundary and boundary[0]==rank[2] and len(chosen)==3 and set(rank[:2])<=set(chosen) and len(set(chosen)&set(boundary))==1,'Original locked top2 and one boundary')
        else:need(boundary==[] and chosen==(rank if len(rank)==2 else []),'Small universe selector identity')
        if orders is not None:exact([meta['symbols'][i][j] for j in chosen],orders[i]['symbols'],'Selector indices map to ordered holdings')
        out['event_rows']+=1;out['eligible_cells']+=len(rank);out['source_rank_calls']+=1
        out['support_evaluations']+=len(boundary);out['delta_evaluations']+=len(boundary)
        out['selection_rank_calls']+=int(len(rank)>=3);out['partial_events']+=int(e['date']=='2016-08-17')
        out['two_eligible_events']+=int(len(rank)==2);out['cash_events']+=int(len(rank)<2)
    return out

def zero_headlines(result):
    """Six zero-cost strategy/path pairs; cost diagnostics stay separate."""
    for arm in ARMS:
        for path in PATHS[:2]:
            record=result['arms'][arm][path]
            for item in (record['metrics'],record['slices']['full122']):
                for name in ('cost_per_side','fees_paid'):
                    need(old.finite(item[name]) and item[name]==0.,'Exact ZERO headline '+name)

def terminals(record,sha,hashes):
    need(record['plan_sha256']==sha and set(record['phases'])==set(hashes),'Root phase terminals')
    for phase,name in [('prepare','prepared/COMPLETION.json'),('select','selected/FROZEN.json'),('diagnose','results/COMPLETION.json')]:
        r=record['phases'][phase]
        need(set(r) in ({'session_id','exit_code','chunk_id','artifact_path','artifact_sha256'},{'session_id','exit_code','chunk_id','artifact_path','artifact_sha256','owner'}),'Terminal fields')
        need('owner' not in r or (type(r['owner']) is str and r['owner']=='/root'),'Optional root owner')
        need(r['session_id'] is None or type(r['session_id']) is int and r['session_id']>0,'Session identity')
        old.integer(r['exit_code'],'Actual exit',0)
        need(type(r['chunk_id']) is str and bool(r['chunk_id'].strip()) and local(r['artifact_path'])==OUT/name and r['artifact_sha256']==hashes[phase],'Terminal digest')

def formal(ev,plan):
    for name,h in FIXED.items():need(ev.registered(plan,name)==h,'Shared frozen helper');ev.pin(name,h)
    context=ev.read(CONTEXT,ev.registered(plan,CONTEXT));contract=ev.read(CONTRACT,ev.registered(plan,CONTRACT))
    exact(plan['context'],context,'Embedded context');exact(plan['implementation_contract'],contract,'Embedded contract')
    need(plan['context_sha256']==digest(CONTEXT)==contract['context_sha256'] and plan['implementation_contract_sha256']==digest(CONTRACT),'Paper digests')
    need(contract['code_files']==list(FORMAL) and contract['supplemental_input_files']==list(SUPPLEMENTAL),'6+2 code inventory')
    for paper in (context,contract):
        need(paper['frozen'] is True and paper['registration_authorized'] is True and paper['actual_data_started'] is False,'Prospective paper')
        for name in FORMAL+SUPPLEMENTAL:need(paper['input_sha256'][name]==ev.registered(plan,name)==ev.pin(name),'Paper code binding')
    exact(context['arms'],list(ARMS),'Fixed roles');need(context['only_promotable']==ARMS[2] and context['diagnostic_only']==[ARMS[1]],'Only exact3 candidate eligible')
    need(context['registered_trials']==3 and context['actual_models_fitted']==0 and context['actual_orders_computed'] is False and context['actual_returns_computed'] is False,'Prospective scope')
    tests=ev.read(TESTS,ev.registered(plan,TESTS));review=ev.read(REVIEW,ev.registered(plan,REVIEW))
    expected={str(ROOT/p):digest(ROOT/p) for p in FORMAL}
    old.integer(tests['tests'],'Actual formal test count')
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
    for key in ('original_arrays_unchanged','original_orders_exact','original_scores_exact','original_pre_scores_exact','original_eligibility_exact','both_reference_gates_before_overlay','whole_original_cache_decoded','legacy154_scores_exact','legacy154_orders_exact'):need(source[key] is True,'Exact source '+key)
    for key in ('training_arrays_decoded','new_price_windows_read','strict_pit','legacy154_runtime_reverified','full270_census_passed'):need(source[key] is False,'Source limited scope '+key)
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

def arrays(ev,path):
    ev.pin(path)
    with np.load(path,allow_pickle=False) as archive:result={k:archive[k].copy() for k in archive.files}
    ev.pin(path)
    return result


def array_exact(a,b,label):
    need(type(a) is np.ndarray and type(b) is np.ndarray and a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes(),label+' exact typed bytes')


def selection(training):
    """Independent literal145 guard; next-open/later are never inputs."""
    need(type(training) is dict and set(training)==set(ARMS),'First88 three arms')
    for a in ARMS:need(set(training[a])=={'same_close_zero'},'Only same-close ZERO selection')
    fields=(('training','cagr'),('training','sharpe'),('training','max_drawdown'),('early','cagr'),('recent','cagr'))
    def guard(candidate,reference):
        values=[r.get(g,{}).get(k) for r in (candidate,reference) for g,k in fields]
        if not all(type(x) in (int,float) and math.isfinite(x) for x in values):return False
        v,b=values[:5],values[5:]
        return bool(v[0]>b[0]+1e-12 and all(x>=y-t for x,y,t in zip(v[1:],b[1:],(.05,.03,.025,.025))))
    checks={a:guard(training[ARMS[2]]['same_close_zero'],training[a]['same_close_zero']) for a in ARMS[:2]}
    passed=all(checks.values())
    return dict(schema='monthly_roa_increment272_first88_v1',primary=ARMS[2] if passed else ARMS[0],admitted=passed,guards_against=checks,selection_execution='close',cost_per_side=0.,training_months=88,early_months=53,recent_months=35,guard_evaluations=2,metric_scalar_reads=20,later_used=False,next_open_used=False,cost_diagnostics_used=False,control_promotable=False)


def accounting(ev,result,ind,training):
    # Extra next-open groups are reconstructed from already-saved diagnostics;
    # they never feed selection. The frozen accountant independently checks them.
    diagnostic={a:dict(training[a],next_open_zero={k:result['arms'][a]['next_open_zero']['slices'][v] for k,v in [('training','first88'),('early','early53'),('recent','recent35')]}) for a in ARMS}
    report,exports=old.accounting(Accounts(ev),dict(result,arms=roles(result['arms'])),dict(ind,paths=roles(ind['paths'])),roles(diagnostic))
    names=dict(zip(old.ARMS,ARMS))
    return report,{(names[a],p):v for (a,p),v in exports.items()}


def selector_receipt(r,n,w):
    need(type(r) is dict and set(r)=={'schema','passed','mode','events','width','flags','work','numerical_tolerance','source_provenance_verified','actual_coverage_claim'},'Selector receipt fields')
    need(r['schema']=='selector_scope268_independent_v1' and r['passed'] is True and r['mode']==ARMS[0],'Selector receipt role')
    old.integer(r['events'],'Selector events',n);old.integer(r['width'],'Selector width',w)
    need(set(r['flags'])==set(FLAGS) and all(v is True for v in r['flags'].values()),'Selector exact flags')
    exact(r['numerical_tolerance'],0.,'No selector tolerance');need(r['source_provenance_verified'] is False and r['actual_coverage_claim'] is False,'Pure selector proof')
    counts(r['work'],WORK)


def pre_receipt(r,n,w,hand,eligible):
    need(set(r)=={'schema','passed','events','width','P_modified_once','eligible_only','adapter_bound_checked','source_files_authenticated','computes_returns','work'},'P receipt fields')
    need(r['schema']=='monthly_roa_increment272_pre_score_independent_v1' and r['passed'] is True and r['events']==n and r['width']==w,'P receipt scope')
    for k in ('P_modified_once','eligible_only','adapter_bound_checked'):need(r[k] is True,'P proof '+k)
    need(r['source_files_authenticated'] is False and r['computes_returns'] is False,'P pure scope')
    expected=dict(pre_score_verifications=1,cells_compared=n*w,eligible_delta_cells=int((eligible[:n]&np.isfinite(hand[:n])&(hand[:n]!=0)).sum()),hand_bound_cells=int(np.isfinite(hand[:n]).sum()),new_model_fits=0)
    counts(r['work'],PRE_WORK);exact(r['work'],expected,'Actual P verification domain')


def overlay_prefix_counts(overlay,baseline_mask,meta,n,w):
    c=dict.fromkeys(OVERLAY_COUNTS,0);c.update(events=n,symbols_per_event=w,baseline_available_cells=int(baseline_mask[:n].sum()),catalog_headers_checked=44 if n else 0)
    ids={'AAPL:FY2016:2016-10-25','AMD:FY2015:2016-01-19','MSFT:FY2015:2015-07-21'}
    cells=overlay['cells'][:n*w];need(len(cells)==n*w,'Saved overlay cell domain')
    for position,row in enumerate(cells):
        i,j=divmod(position,w);need(row['signal_index']==i and row['pool_index']==j and row['date']==meta['dates'][i] and row['symbol']==meta['symbols'][i][j],'Financial cell axes')
        c['latest_lookups']+=int(row['symbol'] not in meta['etfs']);c['selected_increment_cells']+=int(row['event_id'] in ids)
        if row['filled']:
            c['filled_cells']+=1;c['zero_fills']+=int(row['hand_value']==0.)
    for k in ('event_bindings','ratio_evaluations','bounded_evaluations'):c[k]=c['filled_cells']
    c['required_cell_checks']=3*c['filled_cells']
    return c


def entry(e,n,w,meta,audit,overlay,financial,eligible,*,cached=False,orders=None,rowwidth=None):
    need(set(e)=={'prefix','recomputed_original_pre','reuses_main_verified_full','source_audit','independent','overlay_proof','overlay_counts','producer_counts'},'Preparation entry fields')
    old.integer(e['prefix'],'Prefix',n);need(e['recomputed_original_pre'] is (not cached) and type(e['reuses_main_verified_full']) is bool,'Prefix execution mode')
    wanted=dict.fromkeys(PRODUCER,0);wanted.update(original_pre_score_calls=int(not cached),original_selector_calls=3,financial_correction_calls=2,overlay_calls=1,selector_audit_rank_calls=3*n)
    counts(e['producer_counts'],PRODUCER);exact(e['producer_counts'],wanted,'Producer invocations')
    need(set(e['independent'])==set(ARMS),'Independent three roles')
    for a in ARMS:
        r=e['independent'][a];need(set(r)=={'pre_score','selector'},'Independent role fields');selector_receipt(r['selector'],n,w)
        exact(r['selector']['work'],selector_work(audit['arms'][a]['selector_events'],meta,n,w,None if orders is None else orders[a]),'Prefix selector work')
        if a==ARMS[0]:need(r['pre_score'] is None,'No financial incumbent correction')
        else:pre_receipt(r['pre_score'],n,w,financial['control_hand' if a==ARMS[1] else 'candidate_hand'],eligible)
    expected=overlay_prefix_counts(overlay,financial['control_mask'],meta,n,w);counts(e['overlay_counts'],OVERLAY_COUNTS);exact(e['overlay_counts'],expected,'Prefix overlay work')
    r=e['overlay_proof'];need(set(r)=={'schema','passed','events','width','counts','overlay_verifications','audit_matched','baseline_bytes_preserved','availability_independent_of_E','source_files_authenticated','computes_returns'},'Overlay proof fields')
    need(r['schema']=='monthly_roa_increment272_overlay_independent_v1' and r['passed'] is True and r['events']==n and r['width']==w,'Overlay proof domain')
    old.integer(r['overlay_verifications'],'One independent overlay',1)
    for k in ('audit_matched','baseline_bytes_preserved','availability_independent_of_E'):need(r[k] is True,'Overlay proof '+k)
    need(r['source_files_authenticated'] is False and r['computes_returns'] is False,'Pure overlay proof');exact(r['counts'],expected,'Independent prefix overlay work')
    sa=e['source_audit'];need(sa['schema']=='roa_increment272_source_audit_v1' and sa['events']==n and sa['width']==w and sa['cached_original_pre'] is cached,'Source prefix scope')
    for k in ('original_pre_scores_exact','original_eligibility_exact','original_scores_exact','original_orders_exact','legacy154_scores_exact','legacy154_orders_exact','both_reference_gates_before_overlay'):need(sa[k] is True,'Both reference gates '+k)
    need(sa['source_files_authenticated'] is False and sa['new_price_windows_read'] is False,'Source pure scope');old.integer(sa['new_model_fits'],'Zero source fits',0)
    q=sa['source_integrity_work'];counts(q,INTEGRITY)
    need(q['checks']==4 and q['reference_values_compared']==12*n*w and q['reference_order_events_compared']==8*n and q['metadata_events_compared']==4*n and q['financial_values_compared']==8*n*w,'Four consumed integrity checks')
    if rowwidth is not None:need(q['cache_values_compared']==4*n*rowwidth,'All27 consumed prefix array values')


def preparation(proof,audit,source,meta,overlay,financial,eligible,*,total=122,width=49,prefixes=(0,1,53,88,122),orders=None,rowwidth=None):
    need(proof['schema']=='roa_increment272_independent_prepare_v1','Preparation schema')
    need(audit['schema']=='roa_increment272_decision_audit_v1' and audit['events']==total and audit['width']==width and set(audit['arms'])==set(ARMS),'Master decision scope')
    main=proof['main'];entry(main,total,width,meta,audit,overlay,financial,eligible,orders=orders,rowwidth=rowwidth);need(main['reuses_main_verified_full'] is False,'Main executed')
    cached,physical=proof['cached_prefixes'],proof['physical_prefixes'];need([x['prefix'] for x in cached]==list(range(total+1)) and [x['prefix'] for x in physical]==list(prefixes),'All cached and physical prefixes')
    for e in cached:entry(e,e['prefix'],width,meta,audit,overlay,financial,eligible,cached=True,orders=orders,rowwidth=rowwidth);need(e['reuses_main_verified_full'] is False,'Cached actual entry')
    for e in physical:
        entry(e,e['prefix'],width,meta,audit,overlay,financial,eligible,orders=orders,rowwidth=rowwidth)
        if e['prefix']==total:exact(e,dict(main,reuses_main_verified_full=True),'Full physical alias')
        else:need(e['reuses_main_verified_full'] is False,'Fresh physical entry')
    for k in ('all123_order_prefixes','source_arrays_unchanged','original_reference_unchanged','financial_source_unchanged'):need(proof[k] is True,'Preservation '+k)
    exact(source,dict(main=main['source_audit'],physical=[dict(prefix=e['prefix'],audit=e['source_audit']) for e in physical],cached=[dict(prefix=e['prefix'],audit=e['source_audit']) for e in cached]),'Source audit projections')
    entries=[main]+cached+[e for e in physical if not e['reuses_main_verified_full']]
    claimed=proof['execution_accounting'];keys={'actual_prepare_entries','producer','producer_overlay','independent_overlay','independent_pre_score','independent_selector','source_integrity_work','physical_full_reuses_main','new_model_fits','work_definition'}
    need(set(claimed)==keys and claimed['actual_prepare_entries']==len(entries) and claimed['physical_full_reuses_main'] is True,'Work ledger scope');old.integer(claimed['new_model_fits'],'Ledger zero fits',0)
    expected=dict(producer=summed([e['producer_counts'] for e in entries],PRODUCER),producer_overlay=summed([e['overlay_counts'] for e in entries],OVERLAY_COUNTS),independent_overlay=summed([e['overlay_proof']['counts'] for e in entries],OVERLAY_COUNTS),independent_pre_score=summed([e['independent'][a]['pre_score']['work'] for e in entries for a in ARMS[1:]],PRE_WORK),independent_selector=summed([e['independent'][a]['selector']['work'] for e in entries for a in ARMS],WORK),source_integrity_work=summed([e['source_audit']['source_integrity_work'] for e in entries],INTEGRITY))
    for k,v in expected.items():exact(claimed[k],v,'Summed actual work '+k)
    need(type(claimed['work_definition']) is str and bool(claimed['work_definition']),'Disclosed work scope')
    return claimed


def math_check(meta,financial,pre,scores,orders,source_arrays,catalog,records,sessions,overlay,main):
    """One saved full-value verification, not123 original executions or a replay."""
    import monthly_roa_increment272_independent as independent
    import monthly_selector_scope268_independent as selector
    n,w=source_arrays['baseline'].shape
    proof=independent.verify_overlay(meta,financial['control_hand'],financial['control_mask'],catalog,records,sessions,financial['candidate_hand'],financial['candidate_mask'],overlay)
    exact(proof,main['overlay_proof'],'Independently recomputed saved overlay proof')
    E=source_arrays['eligible'];cw=source_arrays['core_weight']
    if cw.ndim==1:cw=cw[:,None]
    cw=np.broadcast_to(cw,(n,w)).copy()
    for arm in ARMS:
        if arm!=ARMS[0]:
            hand=financial['control_hand' if arm==ARMS[1] else 'candidate_hand']
            exact(independent.verify_pre_score(pre[ARMS[0]],hand,E,pre[arm]),main['independent'][arm]['pre_score'],'Saved independent .01 correction')
        choices=[[meta['symbols'][i].index(s) for s in row['symbols']] for i,row in enumerate(orders[arm])]
        inputs=(pre[arm],E,meta['symbols'],source_arrays['effects'],source_arrays['baseline'],source_arrays['core'],cw,source_arrays['state_calm10'],source_arrays['state_weak_narrow'],meta['dates'])
        exact(selector.verify(*inputs,choices,scores[arm],ARMS[0]),main['independent'][arm]['selector'],'Saved independent episode')
        for i,ids in enumerate(choices):exact([float(scores[arm][i,j]) for j in ids],orders[arm][i]['scores'],'Order displayed scores bind F')
    return dict(saved_overlay_verifications=1,saved_pre_score_verifications=2,saved_selector_verifications=3,saved_financial_cells=n*w,original_prefix_executions_repeated=0,raw_panel_read=False,portfolio_replay=False,source_admission_repeated=False)

def run(args):
    need(platform.python_version()=='3.9.6','Pinned runtime')
    for name in ('plan','prepared','frozen','results','terminals'):need(old.sha_value(getattr(args,name+'_sha256')),'Explicit digest '+name)
    ev=Evidence();plan=ev.read(OUT/'PLAN.json',args.plan_sha256);context,tests=formal(ev,plan)
    # Verify only registered active pins. Old154 metadata hash groups are not recursively opened.
    for path,h in plan['input_sha256'].items():ev.pin(path,h)
    runtime=ev.read(old.RUNTIME,ev.registered(plan,old.RUNTIME));need(len(runtime['input_sha256'])==3309,'Pinned numerical package inventory')
    for path,h in runtime['input_sha256'].items():need(ev.registered(plan,path)==h,'Registered runtime closure')
    protected=ev.read(old.PROTECTED,ev.registered(plan,old.PROTECTED));need(len(protected)==plan['protected_files']==21444,'Protected inventory')
    before=ev.read(OUT/'STATE_BEFORE.json',plan['prior_central_sha256']);need(len(before['completed'])==266 and sum(x['trials'] for x in before['completed'])==1309 and before['registered_queue']==[],'Central pre-state')
    prior=ev.read(PRIOR/'COMPLETION.json','64fb4bda4307f20f2d98cfc1dd0500646a601ce9245671fcf3c5e700a0c16bd0')
    need(prior['status']=='COMPLETED_INDEPENDENTLY_VERIFIED' and prior['protected_files']==21444 and prior['observers_modified'] is False,'Closed predecessor')
    need(before['completed'][-1]['directory']==PRIOR.name and before['completed'][-1]['completion']['sha256']==digest(PRIOR/'COMPLETION.json')==context['predecessor']['completion_sha256'],'Immediate predecessor')
    need(prior['plan_sha256']==ev.pin(PRIOR/'PLAN.json',ev.registered(plan,PRIOR/'PLAN.json')),'Prior PLAN')
    prior_review=ROOT/'work/monthly_joint_growth271_RESULT_REVIEW.json'
    need(prior['result_review_sha256']==context['predecessor']['result_review_sha256']==ev.pin(prior_review,ev.registered(plan,prior_review)),'Prior saved review')
    terminal=ev.read(OUT/'PHASE_TERMINALS.json',args.terminals_sha256);terminals(terminal,args.plan_sha256,dict(prepare=args.prepared_sha256,select=args.frozen_sha256,diagnose=args.results_sha256))
    pc=ev.phase(OUT/'prepared',args.prepared_sha256,args.plan_sha256,PREP)
    need(pc['events']==122 and pc['new_returns_computed'] is False,'Prepared scope');old.integer(pc['new_model_fits'],'No new fits',0)
    frozen=ev.read(OUT/'selected/FROZEN.json',args.frozen_sha256);training=ev.read(OUT/'selected/TRAIN_RESULTS.json',frozen['training_sha256'])
    need(frozen['plan_sha256']==args.plan_sha256 and frozen['prepared_sha256']==args.prepared_sha256 and frozen['orders_sha256']==pc['artifacts_sha256']['ORDERS.json'],'Frozen first88')
    need({p.name for p in (OUT/'selected').iterdir()}=={'FROZEN.json','TRAIN_RESULTS.json'},'Selected artifact names')
    rc=ev.phase(OUT/'results',args.results_sha256,args.plan_sha256,RESULT);need(rc['frozen_sha256']==args.frozen_sha256 and rc['paths']==9 and rc['slices']==45,'Nine complete paths')
    meta=ev.read(OUT/'prepared/META.json');orders=ev.read(OUT/'prepared/ORDERS.json')
    need(len(meta['dates'])==len(meta['ends'])==len(meta['symbols'])==122 and meta['ends'][:-1]==meta['dates'][1:] and meta['dates'][0]=='2016-08-17' and meta['ends'][-1]=='2026-09-24','Full clock')
    need(meta['ends'][52]=='2020-12-31' and meta['ends'][87]=='2023-11-30' and all(len(s)==49 and len(set(s))==49 for s in meta['symbols']),'Segments/universe')
    need(set(orders)==set(ARMS),'Three order roles')
    for arm in ARMS:
        need(len(orders[arm])==122 and [o['date'] for o in orders[arm]]==meta['dates'] and [o['end'] for o in orders[arm]]==meta['ends'],'Order dates')
        for o in orders[arm]:need(o['signal_date']==o['date'] and len(o['symbols']) in (0,2,3) and len(o['symbols'])==len(set(o['symbols'])) and len(o['scores'])==len(o['symbols']) and all(old.finite(x) for x in o['scores']),'Order shape')
    priorpc=ev.read(PRIOR/'prepared/COMPLETION.json',ev.registered(plan,PRIOR/'prepared/COMPLETION.json'))
    exact(orders[ARMS[0]],ev.read(PRIOR/'prepared/ORDERS.json',priorpc['artifacts_sha256']['ORDERS.json'])[ARMS[0]],'Incumbent ordered reference')
    exact(meta,ev.read(PRIOR/'prepared/META.json',priorpc['artifacts_sha256']['META.json']),'Original metadata')
    source=ev.read(OUT/'prepared/SOURCE_PARITY.json');axis,rowwidth=source_proof(ev,plan,source,meta)
    exact(orders[ARMS[0]],ev.read(old.LOCAL/'local_linear_ORDERS.json',ev.registered(plan,old.LOCAL/'local_linear_ORDERS.json')),'Original Local67 holdings')
    exact(source['legacy154'],plan['legacy154'],'154 historical binding');legacy=source['legacy154']
    need(legacy['historical_runtime_reverified'] is False and legacy['original149_SBC_common_mask_inherited'] is True and legacy['shared_session_calendar_exact'] is True,'Historical scope retained')
    for name,h in legacy['artifacts_sha256'].items():need(ev.registered(plan,LEGACY/name)==h,'154 registered anchor');ev.pin(LEGACY/name,h)
    exact(meta,ev.read(LEGACY/'META.json'),'154 identity');exact(orders[ARMS[1]],ev.read(LEGACY/'PRETRAIN_ORDERS.json')['expanded_earnings'],'154 ordered control')
    oldsessions=ev.read(local(legacy['session_reference']['path']),legacy['session_reference']['sha256'])['sessions']
    exact([d for d in axis if oldsessions[0]<=d<=oldsessions[-1]],oldsessions,'Shared original session clock')
    exact(source['source_increment'],plan['source_increment'],'Source increment binding');ref=context['source_increment']
    manifest=ev.read(INCREMENT/'MANIFEST.json',ref['manifest_sha256']);increment=ev.read(INCREMENT/'INCREMENT.json',ref['increment_sha256'])
    sr=ev.read(ROOT/ref['actual_review']['path'],ref['actual_review']['sha256'])
    need(sr['passed'] is True and sr['increment_accepted'] is True and sr['source_only'] is True and sr['full_census_passed'] is False and sr['ratio_or_chain_admitted'] is False,'Exact3 source scope independent review')
    need(manifest['increment_accepted'] is True and manifest['full_census_accepted'] is False and manifest['output_sha256']=={'INCREMENT.json':ref['increment_sha256']},'Increment complete not full census')
    for pins in (manifest['input_sha256'],sr['input_sha256']):
        for path,h in pins.items():need(ev.registered(plan,path)==h,'Source active pin registered');ev.pin(path,h)
    for k in ('full_census_passed','old_admissions_recertified','ratio_or_chain_admitted','strategy_registration_authorized'):need(increment[k] is False,'Increment limitation '+k)
    financial=arrays(ev,OUT/'prepared/FINANCIAL_INPUTS.npz');pre=arrays(ev,OUT/'prepared/PRE_SCORES.npz');scores=arrays(ev,OUT/'prepared/SCORES.npz')
    need(set(financial)=={'control_hand','control_mask','candidate_hand','candidate_mask'} and set(pre)==set(scores)==set(ARMS),'Saved numerical roles')
    baseline=arrays(ev,LEGACY/'DATA.npz');base_scores=arrays(ev,LEGACY/'SCORES.npz');prior_pre=arrays(ev,PRIOR/'prepared/PRE_SCORES.npz');prior_scores=arrays(ev,PRIOR/'prepared/SCORES.npz')
    array_exact(financial['control_hand'],baseline['earnings_expanded'],'Frozen154 hand');array_exact(financial['control_mask'],baseline['expanded_available'],'Frozen154 mask')
    array_exact(pre[ARMS[0]],prior_pre[ARMS[0]],'Original saved P');array_exact(scores[ARMS[0]],prior_scores[ARMS[0]],'Original saved F');array_exact(scores[ARMS[1]],base_scores['expanded_earnings'],'154 saved F')
    source_arrays={k:baseline[k] for k in source['original_array_fingerprints']}
    for k,a in source_arrays.items():
        fp=source['original_array_fingerprints'][k];need(a.dtype.str==fp['dtype'] and list(a.shape)==fp['shape'] and hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()==fp['sha256'],'Original27 fingerprint '+k)
    proof=ev.read(OUT/'prepared/INDEPENDENT_PREPARE.json');audit=ev.read(OUT/'prepared/DECISION_AUDIT.json');overlay=ev.read(OUT/'prepared/OVERLAY_AUDIT.json')
    work=preparation(proof,audit,ev.read(OUT/'prepared/SOURCE_AUDIT.json'),meta,overlay,financial,source_arrays['eligible'],orders=orders,rowwidth=rowwidth)
    need(pc['execution_accounting_sha256']==structured_digest(work),'Prepared work digest')
    math_report=math_check(meta,financial,pre,scores,orders,source_arrays,increment['full_catalog'],increment['increment_records'],axis,overlay,proof['main'])
    selected=selection(training);exact(selected,frozen['selection'],'Independent first88 selection')
    result=ev.read(OUT/'results/RESULTS.json');ind=ev.read(OUT/'results/INDEPENDENT_OUTCOME.json');exact(result['selection'],selected,'Immutable frozen selection')
    need(result['strict_pit'] is False and result['reused_sample'] is True and result['observer_promotion'] is False and result['zero_cost_selection'] is True and result['next_open_zero_diagnostic_only'] is True and result['next_open_cost_diagnostic_only'] is True,'Declared research limitations')
    need(ind['accounting_source_sha256']==FIXED['work/ptr169_results_independent.py'],'Pinned accountant');zero_headlines(result)
    account,exports=accounting(ev,result,ind,training)
    for arm in ARMS:
        for path in PATHS:exact([r['date'] for r in exports[arm,path]['DAILY']],[d for d in axis if meta['dates'][0]<=d<=meta['ends'][-1]],'Official daily account clock')
    prc=ev.read(PRIOR/'results/COMPLETION.json',prior['result_completion_sha256']);pr=ev.read(PRIOR/'results/RESULTS.json',prc['artifacts_sha256']['RESULTS.json']);priorfees=None
    for path in PATHS:
        for kind in KINDS:
            name=ARMS[0]+'_'+path+'_'+kind+'.json';previous=ev.read(PRIOR/'results'/name,prc['artifacts_sha256'][name]);current=exports[ARMS[0],path][kind]
            if path=='next_open_cost' and kind=='HOLDINGS':priorfees=previous
            else:exact(current,previous,'Incumbent economic '+path+'/'+kind)
    fee_review=old.diagnostic_summary_parity(result['arms'][ARMS[0]],pr['arms'][ARMS[0]],exports[ARMS[0],'next_open_cost']['HOLDINGS'],priorfees)
    ev.finish()
    return dict(schema='roa_increment272_saved_reader_v1',created_utc=datetime.now(timezone.utc).isoformat(),passed=True,unresolved_blockers=[],plan_sha256=args.plan_sha256,context_sha256=digest(CONTEXT),implementation_contract_sha256=digest(CONTRACT),prepared_completion_sha256=args.prepared_sha256,results_completion_sha256=args.results_sha256,phase_terminals_sha256=args.terminals_sha256,root_owned_terminals=terminal,formal_synthetic_tests=tests['tests'],execution_accounting=work,saved_numeric_verification=math_report,selection=selected,saved_accounting_evidence=account,selection_metric_groups=9,diagnostic_training_groups_checked=18,incumbent_economic_exports_exact=True,incumbent_summary_exact=fee_review['exact'],summary_fee_display_review=fee_review,
        full_results={a:{p:result['arms'][a][p]['metrics'] for p in PATHS} for a in ARMS},segmented_results={a:{p:result['arms'][a][p]['slices'] for p in PATHS} for a in ARMS},npz_decoded=True,saved_npz_archives_decoded=7,actual_models_computed_by_reviewer=False,actual_portfolio_replay=False,source_prices_loaded=False,phase_rerun=False,frozen_files_modified=False,observers_modified=False,strict_pit=False,reused_sample=True,
        protection_and_close=dict(status='pending_root_close',protected_members_rehashed=False,expected_post_close_completed=267,expected_post_close_trials=1312),
        limitations=['Runner and saved reader share an author; mathematical and selector modules are separately authored, with immutable source pins.','Saved financial/P/F/legacy154 arrays decoded for exact arithmetic and selector verification; this is no new price/replay/source-admission run.','Historical154 cache preserves its original SBC common mask; exact3 fills use a separately accepted NI/assets scope. The failed44 census remains failed, with every hold and legacy runtime limitation retained.','Only first88 same-closeZERO selected; next-open/later/full diagnostic results cannot rescue a failed candidate.','Full21444 protection scan and central close belong to root.'],input_sha256=ev.pins)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','prepared','frozen','results','terminals'):p.add_argument('--'+name+'-sha256',required=True)
    print(json.dumps(run(p.parse_args()),indent=2,sort_keys=True,allow_nan=False))
