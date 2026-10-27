"""Read-only269 reconciliation of frozen artifacts, phase receipts and accounts.

No NPZ decode, fitting, prediction, source-price reading or portfolio replay.
The numerical spacing verifier is separately authored. This source review
is non-blind and the selector verifier is inherited unchanged. Reused accounting rules are role-adapted without global mutation.
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
OUT=ROOT/'outputs/monthly_forecast_spacing269'
PRIOR=ROOT/'outputs/monthly_selector_scope268'
CONTEXT=ROOT/'work/monthly_forecast_spacing269_context.json'
CONTRACT=ROOT/'work/monthly_forecast_spacing269_IMPLEMENTATION_CONTRACT.json'
TESTS=ROOT/'work/monthly_forecast_spacing269_TESTS.json'
REVIEW=ROOT/'work/monthly_forecast_spacing269_CODE_REVIEW.json'
ARMS=('incumbent_local67','pooled_affine','mixed_affine')
PATHS=('same_close_zero','next_open_zero','next_open_cost')
KINDS=('DAILY','TRADES','HOLDINGS','DAILY_POSITIONS')
PREP={'SCORES.npz','PRE_SCORES.npz','ORDERS.json','META.json','DECISION_AUDIT.json','SOURCE_PARITY.json','INDEPENDENT_PREPARE.json','SOURCE_AUDIT.json'}
RESULT={'RESULTS.json','INDEPENDENT_OUTCOME.json'}|{a+'_'+p+'_'+k+'.json' for a in ARMS for p in PATHS for k in KINDS}
FORMAL=tuple('work/'+n for stem in ('host','kernel','independent','admission') for n in ('monthly_forecast_spacing269_'+stem+'.py','test_monthly_forecast_spacing269_'+stem+'.py'))
SUPPLEMENTAL=('work/monthly_forecast_spacing269_saved_reader.py','work/test_monthly_forecast_spacing269_saved_reader.py')
PRODUCER=('original_pre_score_calls','original_selector_calls','experimental_transform_calls','selector_events','selector_audit_rank_calls','source_pooled_rank_calls','source_joint_rank_calls','new_model_fits','new_label_values_computed','new_price_values_read')
INTEGRITY=('checks','cache_values_compared','reference_values_compared','reference_order_events_compared','metadata_events_compared')
WORK=('selector_calls','event_rows','eligible_cells','source_rank_calls','support_evaluations','delta_evaluations','selection_rank_calls','partial_events','two_eligible_events','cash_events','new_model_fits')
FLAGS=('original_rank_and_symbol_ties_exact','scope_and_gap_exact','existing_support_budget_arithmetic_exact','partial_and_small_universe_identity_exact','scores_float64_bytes_exact','ordered_choices_exact')
TRANSFORM_WORK=('transform_calls','event_rows','use_cells','active_rows','consumed_raw_cells','source_rank_checks','affine_evaluations','delta_evaluations','output_cells_compared','audit_event_checks','new_model_fits')
TRANSFORM_FLAGS=('source_rank_formula_exact','fixed_slot_weights_exact','active_scope_and_fallback_exact','affine_spacing_and_endpoints_exact','pre_float64_bytes_exact','audit_schema_counts_events_exact','input_arrays_unmodified')
TRANSFORM_COUNTS=('events','symbols_per_event','initial_partial_events','weak_events','ready_events','use_cells','active_events','active_cells','pooled_rank_checks','local_rank_checks','pooled_mapped_cells','local_mapped_cells','pooled_identity_rows','local_identity_rows','pooled_affine_rows','local_affine_rows','pooled_changed_rank_cells','local_changed_rank_cells','pooled_nonzero_delta_cells','local_nonzero_delta_cells','nonzero_delta_cells','changed_pre_cells','transform_calls','new_model_fits','predictive_model_fits')
TRANSFORM_TRUE=('original_inputs_preserved','nonconsumed_pre_preserved','initial_partial_exact','nonweak_exact','not_ready_exact','consumed_source_ranks_checked','local_weight_fixed','no_rerank','no_new_budget')
EVENT_KEYS=('query_index','date','initial_partial','weak','ready','use_indices','branch','pooled','local','delta','changed_pre_indices')
MAP_KEYS=('raw','old_rank','new_rank','distinct_levels','branch','minimum','maximum','rank_minimum','rank_maximum','delta')
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


def receipt(r,arm,n,w,*,transform=False):
    fields={'schema','passed','mode','events','width','flags','work','numerical_tolerance','source_provenance_verified','actual_coverage_claim'}
    need(type(r) is dict and set(r)==fields,'Exact independent receipt fields')
    schema='forecast_spacing269_independent_v1' if transform else 'selector_scope268_independent_v1'
    need(r['schema']==schema and r['passed'] is True and r['mode']==(arm if transform else ARMS[0]),'Independent receipt identity')
    old.integer(r['events'],'Independent events',n);old.integer(r['width'],'Independent width',w)
    flags=TRANSFORM_FLAGS if transform else FLAGS
    need(set(r['flags'])==set(flags) and all(v is True for v in r['flags'].values()),'Independent flags')
    exact(r['numerical_tolerance'],0.,'Exact numerical tolerance')
    need(r['source_provenance_verified'] is False and r['actual_coverage_claim'] is False,'No receipt provenance claim')
    fields=TRANSFORM_WORK if transform else WORK;counts(r['work'],fields)
    z=r['work'];old.integer(z['new_model_fits'],'Zero fits',0)
    if transform:
        need(z['transform_calls']==1 and z['event_rows']==z['audit_event_checks']==n and z['output_cells_compared']==n*w,'Transform invocation work')
    else:
        need(z['selector_calls']==1 and z['event_rows']==z['source_rank_calls']==n,'Selector invocation work')
        need(z['support_evaluations']==z['delta_evaluations']<=z['eligible_cells']<=n*w,'Selector consumed-cell counts')
        need(z['selection_rank_calls']+z['two_eligible_events']+z['cash_events']==n and z['partial_events']<=min(1,n),'Selector event counts')


def entry(e,n,w,*,cached=False):
    need(type(e) is dict and set(e)=={'prefix','recomputed_original_pre','reuses_main_verified_full','source_audit','independent','producer_counts'},'Exact entry fields')
    old.integer(e['prefix'],'Prefix',n)
    need(e['recomputed_original_pre'] is (not cached) and type(e['reuses_main_verified_full']) is bool,'Prefix/fresh clock')
    p=e['producer_counts'];counts(p,PRODUCER)
    expected=dict.fromkeys(PRODUCER,0);expected.update(original_pre_score_calls=int(not cached),original_selector_calls=3,experimental_transform_calls=2,selector_events=3*n,selector_audit_rank_calls=3*n,source_pooled_rank_calls=1,source_joint_rank_calls=1)
    exact(p,expected,'Actual producer work')
    s=e['source_audit']
    need(s['schema']=='forecast_spacing269_source_audit_v1' and s['events']==n and s['width']==w and s['cached_original_pre'] is cached and s['original_pre_score_calls']==int(not cached),'Source scope')
    for k in ('original_pre_scores_exact','original_eligibility_exact','original_scores_exact','original_orders_exact','original_gate_before_candidates','source_rank_parity'):need(s[k] is True,'Source flag '+k)
    exact(s['source_rank_work'],dict(pooled_rank_calls=1,joint_rank_calls=1),'Prefix source rank rechecks, not new forecasts')
    for k in ('source_files_authenticated','new_features_computed','new_labels_computed','new_price_windows_read'):need(s[k] is False,'Source scope limit '+k)
    old.integer(s['new_model_fits'],'Zero new model fits',0)
    sw=s['source_integrity_work'];counts(sw,INTEGRITY)
    need(sw['checks']==4 and sw['reference_values_compared']==12*n*w and sw['reference_order_events_compared']==sw['metadata_events_compared']==4*n,'Source integrity actual work')
    need(sw['cache_values_compared']>=4*n*w,'Source cache cells')
    need(set(e['independent'])==set(ARMS),'Independent roles')
    for a,r in e['independent'].items():
        need(type(r) is dict and set(r)=={'transform','selector'},'Separate transform/selector proof')
        receipt(r['selector'],a,n,w)
        if a==ARMS[0]:need(r['transform'] is None,'No incumbent transform')
        else:receipt(r['transform'],a,n,w,transform=True)


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


def numeric(value,label):
    need(type(value) is float and math.isfinite(value),'Finite binary64 '+label)
    return value


def saved_map(m,u,weight):
    """Reconcile saved forecast arithmetic only, never read raw/cache NPZ."""
    need(type(m) is dict and set(m)==set(MAP_KEYS),'Exact saved map fields')
    for k in ('raw','old_rank','new_rank','delta'):
        need(type(m[k]) is list and len(m[k])==u,'Map vector length '+k)
        for v in m[k]:numeric(v,k)
    xs,rs=m['raw'],m['old_rank'];lo=min(xs);hi=max(xs);rl=min(rs);rh=max(rs);levels=len(set(xs))
    expected=[((float(2*sum(y<x for y in xs)+sum(y==x for y in xs)+1)/2.)/float(u))-.5 for x in xs]
    exact(rs,expected,'Saved original percentile ranks')
    old.integer(m['distinct_levels'],'Distinct raw levels',levels)
    for k,v in [('minimum',lo),('maximum',hi),('rank_minimum',rl),('rank_maximum',rh)]:exact(m[k],v,'Saved map extrema '+k)
    identity=levels<=2;need(m['branch']==('old_rank_copy' if identity else 'affine'),'Map branch')
    mapped=[];interior=0
    if not identity:
        span=numeric(hi-lo,'raw span');rspan=numeric(rh-rl,'rank span');need(span>0.,'Positive raw span')
    for x,r in zip(xs,rs):
        if identity or x==lo or x==hi:target=r
        else:
            numerator=numeric(x-lo,'map numerator');fraction=numeric(numerator/span,'map fraction')
            product=numeric(fraction*rspan,'map product');target=numeric(product+rl,'mapped rank');interior+=1
        need(rl<=target<=rh,'Mapped endpoint range');mapped.append(target)
    exact(m['new_rank'],mapped,'Saved affine arithmetic and exact endpoints')
    delta=[numeric(weight*numeric(a-b,'rank difference'),'weighted delta') for a,b in zip(mapped,rs)]
    exact(m['delta'],delta,'Saved slot weight arithmetic')
    return dict(identity=identity,interior=interior,changed=sum(a.hex()!=b.hex() for a,b in zip(mapped,rs)),nonzero=sum(x!=0. for x in delta))


def transform_evidence(audit,arm,meta,n,w,*,selector_events=None):
    need(arm in ARMS[1:],'Candidate transform arm')
    keys={'schema','mode','counts','events','source_files_authenticated','computes_returns'}|set(TRANSFORM_TRUE)
    need(type(audit) is dict and set(audit)==keys and audit['schema']=='monthly_forecast_spacing269_transform_v1' and audit['mode']==arm,'Transform audit schema')
    for k in TRANSFORM_TRUE:need(audit[k] is True,'Transform preservation '+k)
    need(audit['source_files_authenticated'] is False and audit['computes_returns'] is False,'Transform limited scope')
    need(type(audit['events']) is list and n==len(audit['events']),'Transform full event axis')
    c=dict.fromkeys(TRANSFORM_COUNTS,0);c.update(events=n,symbols_per_event=w,transform_calls=1)
    work=dict.fromkeys(TRANSFORM_WORK,0);work.update(transform_calls=1,event_rows=n,output_cells_compared=n*w,audit_event_checks=n)
    prefix=[dict(work, event_rows=0,output_cells_compared=0,audit_event_checks=0)]
    for i,e in enumerate(audit['events']):
        need(type(e) is dict and set(e)==set(EVENT_KEYS),'Transform event schema')
        old.integer(e['query_index'],'Transform event index',i);need(e['date']==meta['dates'][i],'Transform date')
        partial=e['date']=='2016-08-17';need(e['initial_partial'] is partial and (not partial or i==0),'Named initial partial')
        need(type(e['weak']) is bool and type(e['ready']) is bool,'Native source states')
        ids=e['use_indices'];indices(ids,w,'Use',ordered=True);u=len(ids)
        if selector_events is not None:need(set(ids)<=set(selector_events[i]['eligible_indices']),'Source U within original E')
        branch='initial_partial' if partial else 'nonweak' if not e['weak'] else 'not_ready' if not e['ready'] else 'empty_use' if not ids else 'active'
        need(e['branch']==branch,'Exact active scope precedence')
        c['initial_partial_events']+=int(partial);c['weak_events']+=int(e['weak']);c['ready_events']+=int(e['ready']);c['use_cells']+=u;work['use_cells']+=u
        changed=e['changed_pre_indices'];indices(changed,w,'Changed P',ordered=True)
        if branch!='active':need(e['pooled'] is None and e['local'] is None and e['delta']==[] and changed==[],'Inactive values not consumed')
        else:
            c['active_events']+=1;c['active_cells']+=u;work['active_rows']+=1
            slots=[('pooled',.0375)]+([('local',.0125)] if arm==ARMS[2] else [])
            if arm==ARMS[1]:need(e['local'] is None,'Pooled-only does not consume local map')
            for role,weight in slots:
                proof=saved_map(e[role],u,weight)
                c[role+'_rank_checks']+=1;c[role+'_mapped_cells']+=u
                c[role+('_identity_rows' if proof['identity'] else '_affine_rows')]+=1
                c[role+'_changed_rank_cells']+=proof['changed'];c[role+'_nonzero_delta_cells']+=proof['nonzero']
                work['consumed_raw_cells']+=u;work['source_rank_checks']+=1;work['delta_evaluations']+=u;work['affine_evaluations']+=proof['interior']
            total=e['pooled']['delta']
            if arm==ARMS[2]:total=[numeric(a+b,'total delta') for a,b in zip(total,e['local']['delta'])]
            exact(e['delta'],total,'Saved two-product addition order')
            nonzero=[j for j,d in zip(ids,total) if d!=0.]
            need(set(changed)<=set(nonzero),'Changed P only nonzero delta; exact P bytes bound separately')
            c['nonzero_delta_cells']+=len(nonzero);c['changed_pre_cells']+=len(changed)
        work['event_rows']=i+1;work['output_cells_compared']=(i+1)*w;work['audit_event_checks']=i+1
        prefix.append(dict(work))
    counts(audit['counts'],TRANSFORM_COUNTS);exact(audit['counts'],c,'All transform counts derived from saved events')
    return prefix


def preparation(proof,audit,source,meta,*,total=122,width=49,prefixes=(0,1,53,88,122),orders=None):
    n=len(meta['dates']);w=len(meta['symbols'][0]);need(n==total and w==width,'Full original dimensions')
    need(proof['schema']=='forecast_spacing269_independent_prepare_v1','Preparation schema')
    for k in ('all123_order_prefixes','source_arrays_unchanged','original_reference_unchanged'):need(proof[k] is True,'Preparation flag '+k)
    entry(proof['main'],n,w);need(proof['main']['reuses_main_verified_full'] is False,'Main genuinely computed')
    need([e['prefix'] for e in proof['cached_prefixes']]==list(range(n+1)),'All cached prefixes')
    for e in proof['cached_prefixes']:entry(e,e['prefix'],w,cached=True);need(e['reuses_main_verified_full'] is False,'Cached not full alias')
    need([e['prefix'] for e in proof['physical_prefixes']]==list(prefixes) and prefixes[-1]==n,'Physical prefix set')
    for e in proof['physical_prefixes']:
        entry(e,e['prefix'],w);need(e['reuses_main_verified_full'] is (e['prefix']==n),'Physical full alias')
    exact(proof['physical_prefixes'][-1],dict(proof['main'],reuses_main_verified_full=True),'Full reused without new counts')
    entries=[proof['main']]+proof['cached_prefixes']+proof['physical_prefixes'][:-1]
    work=proof['execution_accounting'];need(work['actual_prepare_entries']==len(entries)==n+1+len(prefixes) and work['physical_full_reuses_main'] is True and work['new_model_fits']==0,'Actual entry count')
    exact(work['producer'],summed([e['producer_counts'] for e in entries],PRODUCER),'Producer totals')
    exact(work['independent_transform'],summed([e['independent'][a]['transform']['work'] for e in entries for a in ARMS[1:]],TRANSFORM_WORK),'Transform totals')
    exact(work['independent_selector'],summed([e['independent'][a]['selector']['work'] for e in entries for a in ARMS],WORK),'Selector totals')
    exact(work['source_integrity_work'],summed([e['source_audit']['source_integrity_work'] for e in entries],INTEGRITY),'Source totals')
    exact(source,dict(main=proof['main']['source_audit'],physical=[dict(prefix=e['prefix'],audit=e['source_audit']) for e in proof['physical_prefixes']],cached=[dict(prefix=e['prefix'],audit=e['source_audit']) for e in proof['cached_prefixes']]),'Source separate artifact')
    need(audit['schema']=='forecast_spacing269_decision_audit_v1' and audit['events']==n and audit['width']==w and set(audit['arms'])==set(ARMS),'Decision audit')
    exact(audit['producer_counts'],proof['main']['producer_counts'],'Decision actual counts')
    for a in ARMS:
        d=audit['arms'][a];want={'selector_calls','events','selector_events'}|({'source'} if a==ARMS[0] else {'transform'})
        need(type(d) is dict and set(d)==want and d['selector_calls']==1 and d['events']==n,'Arm decision fields')
        if a==ARMS[0]:need(d['source']=='frozen_original_select_boundary','Original selector source')
        events=d['selector_events'];need(type(events) is list and len(events)==n,'Selector event inventory')
        selector_work(events,meta,n,w,None if orders is None else orders[a])
        transforms=None if a==ARMS[0] else transform_evidence(d['transform'],a,meta,n,w,selector_events=events)
        for e in entries+[proof['physical_prefixes'][-1]]:
            exact(e['independent'][a]['selector']['work'],selector_work(events,meta,e['prefix'],w),'Every selector prefix/arm actual work')
            if transforms is not None:exact(e['independent'][a]['transform']['work'],transforms[e['prefix']],'Every transform prefix/arm actual work')
    one,two=(audit['arms'][a]['transform']['events'] for a in ARMS[1:])
    for i in range(n):
        for k in ('query_index','date','initial_partial','weak','ready','use_indices','branch','pooled'):exact(one[i][k],two[i][k],'Matched source spacing roles '+k)
        for a in ARMS[1:]:exact(audit['arms'][a]['selector_events'][i]['eligible_indices'],audit['arms'][ARMS[0]]['selector_events'][i]['eligible_indices'],'Unchanged original E')
    return work


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
    for key in ('original_arrays_unchanged','original_orders_exact','original_scores_exact','original_pre_scores_exact','original_eligibility_exact','original_gate_before_candidates','whole_original_cache_decoded','original_forecast_ranks_exact','paired_readiness_exact'):need(source[key] is True,'Exact source '+key)
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
    before=ev.read(OUT/'STATE_BEFORE.json',plan['prior_central_sha256']);need(len(before['completed'])==264 and sum(x['trials'] for x in before['completed'])==1303 and before['registered_queue']==[],'Central pre-state')
    prior=ev.read(PRIOR/'COMPLETION.json','e55987c61bcce4d5e029472e768ec87d89b242d3d887c3cffddecca8c783154b')
    need(prior['status']=='COMPLETED_INDEPENDENTLY_VERIFIED' and prior['protected_files']==21444 and prior['observers_modified'] is False,'Closed predecessor')
    need(before['completed'][-1]['directory']==PRIOR.name and before['completed'][-1]['completion']['sha256']==digest(PRIOR/'COMPLETION.json')==context['predecessor']['completion_sha256'],'Immediate central predecessor')
    need(prior['plan_sha256']==ev.pin(PRIOR/'PLAN.json',ev.registered(plan,PRIOR/'PLAN.json')),'Prior PLAN binding')
    priorreview=ROOT/'work/monthly_selector_scope268_RESULT_REVIEW.json'
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
    zero_headlines(result)
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
    return dict(schema='forecast_spacing269_saved_reader_v1',created_utc=datetime.now(timezone.utc).isoformat(),passed=True,unresolved_blockers=[],plan_sha256=args.plan_sha256,context_sha256=digest(CONTEXT),implementation_contract_sha256=digest(CONTRACT),prepared_completion_sha256=args.prepared_sha256,results_completion_sha256=args.results_sha256,phase_terminals_sha256=args.terminals_sha256,root_owned_terminals=terminal,formal_synthetic_tests=tests['tests'],execution_accounting=work,selection=selected,saved_accounting_evidence=account,incumbent_economic_exports_exact=True,incumbent_summary_exact=fee_review['exact'],summary_fee_display_review=fee_review,full_results={a:{p:result['arms'][a][p]['metrics'] for p in PATHS} for a in ARMS},segmented_results={a:{p:result['arms'][a][p]['slices'] for p in PATHS} for a in ARMS},npz_decoded=False,actual_models_computed_by_reviewer=False,actual_portfolio_replay=False,source_prices_loaded=False,phase_rerun=False,frozen_files_modified=False,observers_modified=False,strict_pit=False,reused_sample=True,protection_and_close=dict(status='pending_root_close',protected_members_rehashed=False,expected_post_close_completed=265,expected_post_close_trials=1306),limitations=['Root source/host review and separately authored selector verifier are non-blind.','Reader recomputes saved forecast-map arithmetic and prefix counters, not source NPZ values or new forecasts; exact P additions/byte preservation remain bound to the actual independent transform proof and immutable PRE_SCORES NPZ.','Within-slot spacing reuses existing forecasts without retraining; historical data-vintage/universe limitations remain.','Full protected-file verification and central closure still belong to root close.'],input_sha256=ev.pins)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','prepared','frozen','results','terminals'):p.add_argument('--'+name+'-sha256',required=True)
    print(json.dumps(run(p.parse_args()),indent=2,sort_keys=True,allow_nan=False))
