"""Read terminal238 saved evidence; no producer, oracle, engine or array decoding.
Only saved hashes, metadata, scalar audit records and accounting certificates are
reviewed. No source momentum, eligibility, order stream or portfolio return is rebuilt.
The inherited incumbent-comparison policy permits ONLY cost-path period
HOLDINGS.fee differences <=1e-10; all other incumbent fields remain exact.
Run only after root authorizes terminal COMPLETION and PLAN digests.
"""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import math

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/monthly_ownsector_gate238'
PRIOR=ROOT/'outputs/monthly_allstate_gate237'
CENTRAL=ROOT/'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
CONTEXT=ROOT/'work/monthly_ownsector_gate238_context.json'
CONTEXT_SHA='c6d4db0f570f41bc44a906589fd8a37b437ab5c5cf1a9db6f4111d0b83e18f69'
TESTS=ROOT/'work/monthly_ownsector_gate238_TESTS.json'
CODE_REVIEW=ROOT/'work/ownsector_gate238_CODE_REVIEW.json'
LOCAL=ROOT/'outputs/monthly_local_linear70'
RUNTIME={'/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/bin/python3.9':'6b24be8ba00c4abc8b3e7e2620bd710e2cf7a4312d00c147f8f05fbb6f3d297a'}
ARMS=('incumbent_local67','ownsector_relative_gate')
PATHS={'same_close_zero':('close',0.),'next_open_zero':('open',0.),'next_open_cost':('open',.0005)}
BOUNDS={'full122':(0,122),'first88':(0,88),'early53':(0,53),'recent35':(53,88),'later34':(88,122)}
PREP={'FEATURES.npz','FEATURE_AUDIT.json','ORDERS.json','META.json','SOURCE_PARITY.json','INDEPENDENT_PREPARE.json'}
RESULT={'RESULTS.json','INDEPENDENT_OUTCOME.json'}|{a+'_'+p+'_'+k+'.json' for a in ARMS for p in PATHS for k in ('DAILY','TRADES','HOLDINGS','DAILY_POSITIONS')}
STAT_KEYS={'start','end','years','final_equity','cagr','sharpe','max_drawdown','trading_sessions','total_return'}
FEATURE_PREFIXES=(0,1,53,88,122)
NUMERICAL_FILES=('monthly_ownsector_gate238_host.py','test_monthly_ownsector_gate238_host.py','monthly_ownsector_gate238_kernel.py','test_monthly_ownsector_gate238_kernel.py','monthly_ownsector_gate238_independent.py','test_monthly_ownsector_gate238_independent.py')
PERIOD_FEE_ATOL=1e-10
ACCOUNTING_RESIDUALS=[]
# Formal test count must match the explicit CLI and immutable registered receipt.
# Explicit PLAN and terminal digests are mandatory. Optional code-review digest defaults to the registered immutable pin.


CALENDAR=ROOT/'outputs/monthly_paired_target143/CALENDAR.json'

COUNT_FIELDS=('events', 'symbols_per_event', 'checked_cells', 'original_eligible_cells', 'weak_events', 'active_events', 'required_cells', 'stock_required_cells', 'etf_required_cells', 'source_price_observations', 'benchmark_price_observations', 'spy_price_observations', 'rejected_cells', 'candidate_eligible_cells', 'new_model_fits', 'predictive_model_fits')
FEATURE_FLAGS=('passed', 'independent_scalar_rebuild', 'original_eligibility_exact', 'weak_state_descriptive_only', 'initial_partial_unchanged', 'official_endpoint_offsets_checked', 'monthend_queries_checked', 'positive_nonboolean_endpoints_checked', 'required_source_only_checked', 'missing_required_source_fails', 'original_positive_momentum_parity_checked', 'strict_relative_comparison_checked', 'max_zero_benchmark_threshold_checked', 'subset_only_checked', 'nonpositive_benchmark_redundancy_checked', 'no_eligibility_rerank', 'no_model_or_score_changes', 'audit_checked', 'descriptive_counts_checked', 'no_predictive_fits_checked', 'no_future_or_outcome_inputs', 'admitted_report_identity_checked', 'official_report_endpoint_binding_checked', 'etf_spy_dispatch_checked', 'dedicated_vox_source_checked', 'no_additional_spy_gate')
SELECTION_FLAGS=('passed', 'independent_original_pre_checked', 'original_pre_unchanged', 'original_model_rank_scope_preserved', 'original_support_checked', 'original_budget_checked', 'original_effects_unchanged', 'final_eligibility_only_checked', 'subset_only_checked', 'one_episode_per_arm', 'initial_partial_unchanged', 'exact_scores_and_choices_checked')
MODEL_FLAGS=FEATURE_FLAGS
HOST_COUNT_FIELDS=COUNT_FIELDS
REPORT=ROOT/'work/ownsector238_saved_result_review.json'
SOURCE=ROOT/'outputs/monthly_ownsector238_repaired_source_audit'


def need(ok,message):
    if not ok:raise ValueError(message)


def finite(value):
    return type(value) in (int,float) and math.isfinite(value)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()


def path(value):
    p=Path(value);return (p if p.is_absolute() else ROOT/p).resolve()


def read_json(value):
    def nonfinite(x):raise ValueError('Nonfinite JSON constant: '+x)
    p=path(value);need(p.is_relative_to(ROOT),'Only local JSON evidence may be read')
    return json.loads(p.read_text(),parse_constant=nonfinite)


class Evidence:
    def __init__(self):self.pins={}
    def pin(self,value,digest=None):
        p=path(value)
        need(p!=CENTRAL.resolve(),'Mutable central must not enter persistent input closure')
        need('CloudStorage' not in p.parts and (p.is_relative_to(ROOT) or RUNTIME.get(str(p))==digest),'Unapproved external input')
        value=sha(p);need(digest is None or value==digest,'Changed evidence: '+str(p))
        key=str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
        need(key not in self.pins or self.pins[key]==value,'Evidence changed within review: '+key)
        self.pins[key]=value;return value
    def load(self,value,digest=None):
        self.pin(value,digest);return read_json(value)
    def phase(self,folder,digest,plan_sha,names):
        c=self.load(folder/'COMPLETION.json',digest)
        need(c.get('status')=='COMPLETE' and c.get('plan_sha256')==plan_sha and set(c.get('artifacts_sha256',{}))==names,'Incomplete exact phase manifest: '+str(folder))
        for filename,h in c['artifacts_sha256'].items():
            need(Path(filename).name==filename,'Phase artifact path escape');self.pin(folder/filename,h)
        return c


def registered_digest(plan,value):
    p=path(value);matches=[digest for name,digest in plan['input_sha256'].items() if path(name)==p]
    need(matches and len(set(matches))==1,'Missing/conflicting registered dependency: '+str(p))
    return matches[0]


def numerical_close(actual,expected,label):
    # Exactly the already frozen independent accounting tolerance; never used
    # for comparing the incumbent with its prior run or for zero-cost fees.
    need(finite(actual) and finite(expected) and abs(actual-expected)<=1e-7+2e-11*abs(expected),'Independent recorded accounting differs: '+label)
    if actual!=expected:
        ACCOUNTING_RESIDUALS.append(dict(label=label,production=actual,independent=expected,
            signed_difference=actual-expected,absolute_difference=abs(actual-expected),absolute_tolerance=1e-7,relative_tolerance=2e-11))


def metric_evidence(independent,actual,label):
    need(set(independent)==STAT_KEYS and STAT_KEYS<=set(actual),'All nine independent statistics required: '+label)
    for key,wanted in independent.items():
        value=actual[key]
        if key in ('start','end','trading_sessions') or wanted is None:
            need(value==wanted,'Independent exact statistic differs: '+label+'/'+key)
        else:numerical_close(value,wanted,label+'/'+key)


def guard(candidate,reference,gain):
    return (candidate['training']['cagr']>=reference['training']['cagr']+gain-1e-12 and
        candidate['training']['sharpe']>=reference['training']['sharpe']-.05 and
        candidate['training']['max_drawdown']>=reference['training']['max_drawdown']-.03 and
        all(candidate[era]['cagr']>=reference[era]['cagr']-.025 for era in ('early','recent')))


def changed(orders,reference,candidate):
    rows=[]
    for i,(old,new) in enumerate(zip(orders[reference],orders[candidate])):
        if set(old['symbols'])!=set(new['symbols']):
            rows.append(dict(index=i,date=old['date'],end=old['end'],reference=list(old['symbols']),candidate=list(new['symbols']),
                removed=sorted(set(old['symbols'])-set(new['symbols'])),added=sorted(set(new['symbols'])-set(old['symbols']))))
    return dict(counts={name:sum(lo<=r['index']<hi for r in rows) for name,(lo,hi) in BOUNDS.items()},months=rows)


def selection_from_saved(training,changes):
    ids=[x['index'] for x in changes[ARMS[1]+'_vs_'+ARMS[0]]['months'] if x['index']<88]
    baseline,candidate=training[ARMS[0]],training[ARMS[1]]
    checks=dict(close_vs_incumbent=guard(candidate['same_close_zero'],baseline['same_close_zero'],.005),
        open_vs_incumbent=guard(candidate['next_open_zero'],baseline['next_open_zero'],0.),
        changed_support=len(ids)>=3 and any(i<53 for i in ids) and any(i>=53 for i in ids))
    admitted=all(checks.values())
    return dict(primary=ARMS[1] if admitted else ARMS[0],admitted=admitted,guards=checks,
        changed_months=ids,first88_only=True,later34_used=False)


def compare_incumbent(evidence,prior_completion,results,prior_results):
    report={}
    for name in PATHS:
        old=prior_results['arms'][ARMS[0]][name];new=results['arms'][ARMS[0]][name]
        need(old['metrics']==new['metrics'] and old['slices']==new['slices'],'Every incumbent metric/slice, including cumulative fees and turnover, must be exact')
        exports={}
        for kind in ('TRADES','DAILY','DAILY_POSITIONS','HOLDINGS'):
            filename=ARMS[0]+'_'+name+'_'+kind+'.json'
            old_rows=evidence.load(PRIOR/'results'/filename,prior_completion['artifacts_sha256'][filename]);new_rows=evidence.load(OUT/'results'/filename)
            if kind!='HOLDINGS':need(old_rows==new_rows,'Incumbent exact '+name+'/'+kind)
            exports[kind]=(old_rows,new_rows)
        oh,nh=exports['HOLDINGS'];need(len(oh)==len(nh)==122,'Incumbent full holdings')
        residuals=[]
        for a,b in zip(oh,nh):
            need(set(a)==set(b) and {k:v for k,v in a.items() if k!='fee'}=={k:v for k,v in b.items() if k!='fee'},'Incumbent nonfee holdings exact')
            need(finite(a['fee']) and finite(b['fee']),'Finite fees')
            if a['fee']!=b['fee']:
                difference=b['fee']-a['fee'];need(name=='next_open_cost' and abs(difference)<=PERIOD_FEE_ATOL,'Only cost period HOLDINGS.fee may differ <=1e-10')
                residuals.append(dict(signal_date=b['signal_date'],execution_date=b['execution_date'],previous237=a['fee'],current238=b['fee'],difference=difference))
        for rows,metric in ((oh,old['metrics']),(nh,new['metrics'])):
            total=0.
            for row in rows:total+=row['fee']
            need(total==metric['fees_paid'],'Saved chronological fees exactly equal cumulative fee metric')
        report[name]=dict(all_metrics_and_slices_exact=True,total_fees_exact=True,all_fills_exact=True,daily_equity_exact=True,daily_positions_exact=True,holding_nonfee_fields_exact=True,period_fee_differences=residuals,max_absolute_period_fee_difference=max([abs(x['difference']) for x in residuals],default=0.))
    return dict(paths=report,period_fee_tolerance=PERIOD_FEE_ATOL,tolerance_scope='ONLY individual next_open_cost HOLDINGS.fee <=1e-10; all metrics/slices including cumulative fees, turnover, trades, DAILY, positions and holding nonfee fields EXACT; zero-cost fee fields EXACT0.',prospective_policy='Inherited frozen232 comparison scope: no accumulated-fee exception and no result-driven widening.')


def model_check(check,counts,n):
    need(len(FEATURE_FLAGS)==26 and all(check.get(k) is True for k in FEATURE_FLAGS),'All26 independent feature flags')
    need(check.get('counts')==counts and all(type(check.get(k)) is int and check[k]==v for k,v in counts.items()),'Exact16 independent feature counters')
    need(counts['events']==n and check.get('numerical_tolerance')==0.,'Feature coverage/tolerance')
    need(check.get('producer_modules_imported') is False and check.get('computes_returns') is False,'Independent scope')


def holdings_counts(orders,n):
    return {a:{str(k):sum(len(row['symbols'])==k for row in orders[a][:n]) for k in (0,2,3)} for a in ARMS}


def selection_check(check,n,orders):
    need(all(check.get(k) is True for k in SELECTION_FLAGS),'All12 selection flags')
    need(check['ordered_choices_checked']==2*n and check['checked_events']==n and check['numerical_tolerance']==0. and check['original_source_parity_tolerance']==1e-12,'Selection coverage/tolerance')
    need(check['producer_modules_imported'] is False and check['computes_returns'] is False and check['full_scores_supplied'] is True and check['full_pre_scores_supplied'] is True,'Pre/full-score scope')
    need(check['holdings_counts']==holdings_counts(orders,n),'Independent3/2/cash counts')


def aggregate_counts(per_event,n):
    counts={k:sum(row[k] for row in per_event[:n]) for k in COUNT_FIELDS if k not in ('events','symbols_per_event')}
    counts.update(events=n,symbols_per_event=49)
    return counts


def feature_evidence(audit,prepared,pc,closure,meta,calendar,orders,source_report):
    need(len(COUNT_FIELDS)==16 and audit['schema']=='monthly_ownsector_gate238_features_v1','Feature schema')
    need([audit[k] for k in ('start_offset','end_offset','return_count')]==[252,42,210],'Frozen window')
    for k in ('rankings_recomputed','source_missing_filled','portfolio_returns_calculated'):need(audit[k] is False,'Source scope '+k)
    axis=calendar['sessions'];need(axis==sorted(set(axis)),'Unique official calendar')
    positions={day:i for i,day in enumerate(axis)};monthends={day[:7]:day for day in axis}
    events=audit['event_history'];need(len(events)==122,'Full feature events')
    keys={'query_index','date','symbols','session_position','weak_narrow','initial_partial','active','branch','source_start','source_end','source_positions','original_eligible_indices','required_indices','stock_required_indices','etf_required_indices','rejected_indices','candidate_eligible_indices'}
    cell_keys={'query_index','column_index','date','symbol','is_etf','benchmark','benchmark_source','original_benchmark','mapping_repair','mapping_source','mapping_evidence','source_start','source_end','momentum','benchmark_momentum','threshold','accepted'}
    records={(c['query_index'],c['column_index']):c for c in source_report['records']}
    need(len(records)==len(source_report['records']),'Unique admitted source identities')
    cells=audit['cell_history'];cell_map={}
    for c in cells:
        key=(c['query_index'],c['column_index'])
        need(set(c)==cell_keys and key not in cell_map and key in records,'Unique saved cell identity')
        cell_map[key]=c
        r=records[key]
        for k in ('query_index','column_index','date','symbol','is_etf','benchmark','original_benchmark','mapping_repair','mapping_source','mapping_evidence'):
            need(c[k]==r[k],'Admitted mapping metadata differs: '+k)
        need(c['benchmark_source']==('vox' if c['benchmark']=='VOX' else 'adj'),'Saved source dispatch')
        need(type(c['accepted']) is bool and all(finite(c[k]) for k in ('momentum','benchmark_momentum','threshold')),'Finite saved feature scalars')
        # Scalar gate mathematics is not recomputed here. Registered independent
        # preparation supplies that check; this reader checks saved identities.
    per_event=[];status=[];expected_cells=set()
    for i,e in enumerate(events):
        day=meta['dates'][i];names=meta['symbols'][i];t=positions[day];partial=(i==0 and day=='2016-08-17')
        need(set(e)==keys and e['query_index']==i and e['date']==day and e['symbols']==names and e['session_position']==t,'Event identity/schema')
        need(partial or monthends[day[:7]]==day,'Official month end')
        need(type(e['weak_narrow']) is bool and type(e['active']) is bool and e['active']==(not partial) and e['initial_partial']==partial,'All-regular activation')
        need(e['branch']==('initial_partial' if partial else 'ownsector_relative_gate'),'Exact branch')
        groups={k:e[k] for k in keys if k.endswith('_indices')}
        for k,ids in groups.items():need(isinstance(ids,list) and all(type(j) is int and 0<=j<49 for j in ids) and ids==sorted(set(ids)),'Sorted unique index evidence '+k)
        old=e['original_eligible_indices'];req=e['required_indices'];new=e['candidate_eligible_indices'];rej=e['rejected_indices']
        stocks=e['stock_required_indices'];etfs=e['etf_required_indices']
        need(req==(old if not partial else []) and set(new)<=set(old) and set(rej)==set(old)-set(new),'Saved required/subset identities')
        need(stocks==[j for j in req if names[j] not in meta['etfs']] and etfs==[j for j in req if names[j] in meta['etfs']],'Explicit ETF partition')
        if partial:need(new==old and e['source_start'] is None and e['source_end'] is None and e['source_positions']==[],'Partial bypass')
        else:need(e['source_start']==axis[t-252] and e['source_end']==axis[t-42] and e['source_positions']==[t-252,t-42],'Saved official endpoint clocks')
        for j in req:
            key=(i,j);expected_cells.add(key);c=cell_map[key]
            need(c['date']==day and c['symbol']==names[j] and c['is_etf']==(j in etfs),'Per-cell metadata')
            need(c['source_start']==e['source_start'] and c['source_end']==e['source_end'] and c['accepted']==(j in new),'Per-cell saved status/clock')
            if j in etfs:need(c['benchmark']=='SPY','ETF benchmark dispatch')
        for arm,ids in ((ARMS[0],old),(ARMS[1],new)):
            selected=orders[arm][i]['symbols'];expected_size=3 if len(ids)>=3 else 2 if len(ids)==2 else 0
            need(len(selected)==expected_size and set(selected)<={names[j] for j in ids},'Saved3/2/cash membership')
        if old==new:need(orders[ARMS[0]][i]==orders[ARMS[1]][i],'Same eligibility full orders must match')
        c=dict.fromkeys(COUNT_FIELDS,0)
        c.update(events=1,symbols_per_event=49,checked_cells=49,original_eligible_cells=len(old),weak_events=int(e['weak_narrow']),active_events=int(not partial),required_cells=len(req),stock_required_cells=len(stocks),etf_required_cells=len(etfs),source_price_observations=2*len(req),benchmark_price_observations=2*len(req),spy_price_observations=2*len(etfs),rejected_cells=len(rej),candidate_eligible_cells=len(new))
        per_event.append(c);status.append(dict(index=i,date=day,original_eligible=len(old),candidate_eligible=len(new),rejected=len(rej)))
    need(expected_cells==set(cell_map)==set(records),'Exact complete required cell set')
    counts=aggregate_counts(per_event,122)
    for record in (audit,prepared,pc,closure):need(all(type(record.get(k)) is int and record[k]==v for k,v in counts.items()),'All16 flat counts agree')
    for record in (prepared,pc):need(record['model_counts']==counts and record['original_array_count']==27 and record['new_predictive_model_fits']==0,'Nested counts/original27/fits')
    need(pc['new_returns_computed'] is False,'Prepare scope')
    model_check(prepared['feature'],counts,122);selection_check(prepared['selection'],122,orders)
    need(prepared['prefixes']==123 and prepared['physical_cache_meta_order_prefixes'] is True and prepared['feature_prefixes']==list(FEATURE_PREFIXES),'123order/5source prefixes')
    prefixes=prepared['physical_prefix_evidence'];need([p['prefix'] for p in prefixes]==list(FEATURE_PREFIXES),'Exact source prefixes')
    for p in prefixes:
        n=p['prefix'];c=aggregate_counts(per_event,n)
        need(p['counts']==c and p['report_records']==c['required_cells'],'Prefix counts/report size')
        model_check(p['feature'],c,n);selection_check(p['selection'],n,orders)
        need(set(p['source_rows'])==set(p['source_last_session'])=={'adj','vox'},'Physical source keys')
        for k in ('adj','vox'):need(type(p['source_rows'][k]) is int and (p['source_rows'][k]>0 if n else p['source_rows'][k]==0),'Physical rows')
        need(p['source_last_session']['adj']==(meta['dates'][n-1] if n else None),'Original price cutoff')
        last=p['source_last_session']['vox'];need((isinstance(last,str) and last<=meta['dates'][n-1]) if n else last is None,'VOX cutoff')
        size=positions[meta['dates'][n-1]]+2 if n else 0
        need(p['calendar_rows']==size and p['calendar_last_session']==(axis[size-1] if n else None),'Next official session witness cutoff')
    need(prefixes[-1]['feature']==prepared['feature'] and prefixes[-1]['selection']==prepared['selection'],'Full prefix receipts agree')
    for k in ('original_orders_exact','original_arrays_unchanged','original_pre_preserved','original_core_preserved','original_E_for_pre','candidate_E_only_at_episode','single_original_episode','no_added_score_budget','feature_prefix_sources_truncated','calendar_prefixes_truncated','selection_uses_zero_cost_paths_only'):need(prepared[k] is True,'Prepare invariant '+k)
    return dict(counts=counts,physical_source_prefixes=list(FEATURE_PREFIXES),cache_meta_order_prefixes=123,cell_records_checked=len(cells),event_records_checked=len(events),eligibility_by_period={label:{k:sum(e[k] for e in per_event[lo:hi]) for k in ('original_eligible_cells','required_cells','rejected_cells','candidate_eligible_cells')} for label,(lo,hi) in BOUNDS.items()},holdings_counts=holdings_counts(orders,122),gate_math_recomputed=False)


def main(expected,expected_plan_sha,test_count):
    need(not REPORT.exists(),'Write-once report already exists')
    ev=Evidence();closure=ev.load(OUT/'COMPLETION.json',expected);plan=ev.load(OUT/'PLAN.json',expected_plan_sha)
    need(closure['status']=='COMPLETED_INDEPENDENTLY_VERIFIED' and closure['plan_sha256']==expected_plan_sha and closure['new_trials']==2 and closure['protected_files']==21444 and closure['observers_modified'] is False,'Terminal closure')
    context=ev.load(CONTEXT,CONTEXT_SHA)
    need(registered_digest(plan,CONTEXT)==CONTEXT_SHA==plan['contract']['prospective_context_sha256'] and plan['contract']['context']==context and context['arms']==list(ARMS),'Frozen context')
    for k in ('original_pre_unchanged','original_eligibility_for_models','eligibility_injected_only_at_episode','original_episode_preserved','original_pool_preserved','no_parameter_search','no_observer_writes'):need(plan['contract'][k] is True,'Registered contract '+k)
    need(plan['contract']['additional_score_budget']==0. and plan['contract']['selection_paths']==['same_close_zero','next_open_zero'] and plan['contract']['next_open_cost_diagnostic_only'] is True,'Frozen zero-only selection')
    for k in ('actual_new_feature_evaluations_before_registration','actual_new_model_fits_before_registration','actual_new_parameters_before_registration','actual_new_orders_before_registration','actual_new_returns_before_registration'):need(plan[k]==0,'Prospective zero '+k)
    tests=ev.load(TESTS,registered_digest(plan,TESTS));review=ev.load(CODE_REVIEW,registered_digest(plan,CODE_REVIEW))
    need(tests['passed'] is True and tests['synthetic_only'] is True and tests['tests']==test_count==87 and tests['runtime']==plan['runtime'],'Formal87 synthetic/runtime')
    need(review['passed'] is True and not review['unresolved_blockers'] and review['context_sha256']==CONTEXT_SHA,'Independent code review')
    receipt=review['formal_test_receipt'];need(receipt['sha256']==sha(TESTS) and receipt['tests']==test_count and receipt['passed'] is True and receipt['synthetic_only'] is True and receipt['runtime']==tests['runtime'],'Code-review test receipt')
    wanted={str((ROOT/'work'/name).relative_to(ROOT)):registered_digest(plan,ROOT/'work'/name) for name in NUMERICAL_FILES}
    need(review['code_sha256']==wanted,'Six reviewed code pins')
    for name,digest in wanted.items():ev.pin(name,digest);need(registered_digest({'input_sha256':tests['code_sha256']},name)==digest,'Tested code pin')
    # Do not traverse the PLAN's inherited full closure, raw prices or protected archive.
    before=ev.load(OUT/'STATE_BEFORE.json',plan['prior_central_sha256'])
    need(len(before['completed'])==233 and sum(x['trials'] for x in before['completed'])==1213 and not before['registered_queue'],'Pre-registration central')
    frozen=ev.load(OUT/'selected/FROZEN.json');training=ev.load(OUT/'selected/TRAIN_RESULTS.json',frozen['training_sha256'])
    need(frozen['plan_sha256']==expected_plan_sha,'Frozen selection PLAN')
    pc=ev.phase(OUT/'prepared',frozen['prepared_sha256'],expected_plan_sha,PREP)
    rc=ev.phase(OUT/'results',closure['result_completion_sha256'],expected_plan_sha,RESULT)
    need(rc['paths']==6 and rc['slices']==30 and rc['frozen_sha256']==sha(OUT/'selected/FROZEN.json'),'Results complete/selection lineage')
    need(frozen['orders_sha256']==sha(OUT/'prepared/ORDERS.json'),'Frozen orders')
    meta=ev.load(OUT/'prepared/META.json');orders=ev.load(OUT/'prepared/ORDERS.json')
    need(meta==ev.load(LOCAL/'META.json',registered_digest(plan,LOCAL/'META.json')),'Original metadata reproduction')
    need(len(meta['dates'])==len(meta['ends'])==len(meta['symbols'])==122 and all(len(set(s))==49 for s in meta['symbols']),'Original122x49')
    need(meta['dates'][0]=='2016-08-17' and meta['ends'][-1]=='2026-09-24' and meta['ends'][:-1]==meta['dates'][1:],'Frozen overall period')
    need(set(orders)==set(ARMS) and all(len(v)==122 for v in orders.values()),'Two complete order streams')
    need(orders[ARMS[0]]==ev.load(LOCAL/'local_linear_ORDERS.json',registered_digest(plan,LOCAL/'local_linear_ORDERS.json')),'Incumbent orders exact')
    for rows in orders.values():
        for i,row in enumerate(rows):need(row['date']==row['signal_date']==meta['dates'][i] and row['end']==meta['ends'][i] and len(row['symbols']) in (0,2,3) and len(set(row['symbols']))==len(row['symbols']),'Saved order axes/support')
    prepared=ev.load(OUT/'prepared/INDEPENDENT_PREPARE.json');audit=ev.load(OUT/'prepared/FEATURE_AUDIT.json');parity=ev.load(OUT/'prepared/SOURCE_PARITY.json')
    calendar=ev.load(CALENDAR,registered_digest(plan,CALENDAR))
    binding=plan['price_source_binding'];need(parity['price_source_binding']==binding,'Source binding equality')
    admission=binding['source_admission'];need(admission['source_admitted'] is True and admission['strict_pit'] is False and admission['raw_original_map_preserved'] is True,'Prior source admission')
    source_plan=ev.load(SOURCE/'SOURCE_PLAN.json',admission['source_plan_sha256']);source_report=ev.load(SOURCE/'REPORT.json',admission['source_report_sha256']);source_close=ev.load(SOURCE/'COMPLETION.json',admission['source_completion_sha256'])
    need(source_report['status']=='ADMITTED' and source_report['admitted'] is True and source_report['gaps']==[] and source_close['admitted'] is True and source_close['report_sha256']==sha(SOURCE/'REPORT.json') and source_close['plan_sha256']==sha(SOURCE/'SOURCE_PLAN.json'),'Admitted source closure binding')
    need(source_close['counts']==source_report['counts'] and source_close['repair_counts']==source_report['repair_counts'] and source_close['source_only'] is True and source_close['strategy_trials']==0,'Source-only closure counts')
    for k in ('missing_endpoints','invalid_endpoints','mapping_missing_cells','gap_cells','market_gap_events'):need(source_report['counts'][k]==0,'Source evidence gaps '+k)
    need(datetime.fromisoformat(source_close['created_utc'])<datetime.fromisoformat(plan['created_utc'])<datetime.fromisoformat(pc['created_utc'])<datetime.fromisoformat(frozen['created_utc'])<datetime.fromisoformat(rc['created_utc'])<datetime.fromisoformat(closure['created_utc']),'Registration/preparation/selection/diagnostic/close chronology')
    source=feature_evidence(audit,prepared,pc,closure,meta,calendar,orders,source_report)
    for k in ('required_cells','stock_required_cells','etf_required_cells'):need(source['counts'][k]==source_report['counts'][k],'Source/gate counts '+k)
    local_plan=ev.load(LOCAL/'PLAN.json',registered_digest(plan,LOCAL/'PLAN.json'))
    need(plan['raw_source_hashes']==parity['raw_source_hashes']==binding['raw_source_hashes']==local_plan['raw_source_hashes'] and len(plan['raw_source_hashes'])==64,'Original64 raw-map identity; no raw rehash')
    need(len(parity['original_array_fingerprints'])==27,'Saved original27 fingerprints')
    for k in ('original_arrays_unchanged','original_orders_exact','original_pre_preserved','original_E_for_pre','candidate_E_only_at_episode'):need(parity[k] is True,'Source parity '+k)
    for k in ('adjusted_levels_constant_scale','adjusted_opening_parity','missing_price_masks_exact','positive_price_masks_exact'):need(parity['execution_prices'][k] is True,'Saved execution parity '+k)
    need(prepared['model_fit_accounting']==pc['model_fit_accounting']==closure['model_fit_accounting']==plan['contract']['model_fit_accounting'],'Descriptor accounting')
    results=ev.load(OUT/'results/RESULTS.json');independent=ev.load(OUT/'results/INDEPENDENT_OUTCOME.json')
    need(results['selection']==frozen['selection'] and results['strict_pit'] is False and results['reused_sample'] is True and results['observer_promotion'] is False,'Saved result qualification')
    for k in ('own_sector_relative_qualification','zero_cost_selection','next_open_cost_diagnostic_only','original_pre_unchanged','candidate_eligibility_only_at_episode'):need(results[k] is True,'Result mechanism '+k)
    need(independent['passed'] is True and independent['independent_accounts'] is True and independent['independent_of_production_model_and_engine'] is True and independent['checked_paths']==6 and independent['checked_slices']==30 and independent['terminal_liquidation'] is False,'Independent6/30 accounting')
    need(independent['accounting_source_sha256']==registered_digest(plan,ROOT/'work/ptr169_results_independent.py'),'Accounting code pin')
    ev.pin(ROOT/'work/ptr169_results_independent.py',independent['accounting_source_sha256'])
    need(set(results['arms'])==set(independent['paths'])==set(training)==set(ARMS),'Exact two arms')
    metrics={};fees={};training_groups=0;zero_streams=0
    for arm in ARMS:
        need(set(results['arms'][arm])==set(independent['paths'][arm])==set(PATHS) and set(training[arm])=={'same_close_zero','next_open_zero'},'Paths/training onlyzero')
        metrics[arm]={};fees[arm]={}
        for name,(execution,cost) in PATHS.items():
            result=results['arms'][arm][name];m=result['metrics'];proof=independent['paths'][arm][name]
            need(m['execution']==execution and m['cost_per_side']==cost and m['start']==meta['dates'][0] and m['end']==meta['ends'][-1],'Execution identities')
            need(proof['passed'] is True and proof['monthly_executions']==122 and set(proof['slices'])==set(result['slices'])==set(BOUNDS),'Five saved slices')
            need(result['accounting']['exported_fill_ledger_matches'] is True and result['accounting']['terminal_liquidation'] is False,'Ledger certificate')
            metric_evidence(proof['slices']['full122'],m,arm+'/'+name+'/metrics');metrics[arm][name]={}
            for label,(lo,hi) in BOUNDS.items():
                v=result['slices'][label];need(v['start']==meta['dates'][lo] and v['end']==meta['ends'][hi-1],'Split boundaries')
                metric_evidence(proof['slices'][label],v,arm+'/'+name+'/'+label)
                metrics[arm][name][label]={k:v[k] for k in ('start','end','cagr','sharpe','max_drawdown')}
            for k in ('fees_paid','dollar_turnover','trade_rows'):numerical_close(m[k],proof[k],arm+'/'+name+'/'+k)
            if name in training[arm]:
                need(set(training[arm][name])=={'training','early','recent'},'Training subsets')
                for subset,label in (('training','first88'),('early','early53'),('recent','recent35')):
                    need(set(training[arm][name][subset])==STAT_KEYS,'Training nine statistics')
                    metric_evidence(proof['slices'][label],training[arm][name][subset],arm+'/'+name+'/'+subset);training_groups+=1
            trades=ev.load(OUT/'results'/f'{arm}_{name}_TRADES.json');holdings=ev.load(OUT/'results'/f'{arm}_{name}_HOLDINGS.json')
            need(len(holdings)==122 and len(trades)==m['trade_rows'],'Saved fee stream coverage')
            need(all(finite(t['cost']) and t['cost']>=0 for t in trades) and all(finite(h['fee']) and h['fee']>=0 for h in holdings),'Finite nonnegative fees')
            for order,h in zip(orders[arm],holdings):need(h['signal_date']==order['date'] and h['symbols']==['META' if s=='research_equity_FB' else s for s in order['symbols']] and h['scores']==order['scores'],'Executed holdings use frozen export alias and ordered scores')
            if cost==0.:
                need(m['fees_paid']==proof['fees_paid']==0. and all(t['cost']==0. for t in trades) and all(h['fee']==0. for h in holdings),'All zero-cost fields exactly0');zero_streams+=1
            fees[arm][name]=dict(execution=execution,cost_per_side=cost,fees_paid=m['fees_paid'],trade_rows=len(trades),holding_rows=len(holdings),all_zero_fee_fields_checked=cost==0.)
    need(training_groups==12 and zero_streams==4,'12training groups/4zero streams')
    prior_done=ev.load(PRIOR/'results/COMPLETION.json',registered_digest(plan,PRIOR/'results/COMPLETION.json'))
    prior_results=ev.load(PRIOR/'results/RESULTS.json',prior_done['artifacts_sha256']['RESULTS.json'])
    incumbent=compare_incumbent(ev,prior_done,results,prior_results)
    changes={ARMS[1]+'_vs_'+ARMS[0]:changed(orders,ARMS[0],ARMS[1])}
    selection=selection_from_saved(training,changes);need(selection==frozen['selection'],'Saved first88 selection reproduced')
    need(selection['admitted'] is False and selection['primary']==ARMS[0],'Recorded rejection/retention expected by review task')
    guard_details={}
    for name,gain in (('same_close_zero',.005),('next_open_zero',0.)):
        a=training[ARMS[1]][name];b=training[ARMS[0]][name]
        guard_details[name]={key:dict(candidate=value,minimum=minimum,passed=value>=minimum) for key,value,minimum in (
            ('training_cagr',a['training']['cagr'],b['training']['cagr']+gain-1e-12),('training_sharpe',a['training']['sharpe'],b['training']['sharpe']-.05),('training_max_drawdown',a['training']['max_drawdown'],b['training']['max_drawdown']-.03),('early_cagr',a['early']['cagr'],b['early']['cagr']-.025),('recent_cagr',a['recent']['cagr'],b['recent']['cagr']-.025))}
    expected_best=before['best_training_selected_candidate'];need(closure['best_training_selected_candidate']==expected_best and closure['target_reached']==(expected_best['cagr']>=.7),'Retained best/target')
    central_bytes=CENTRAL.read_bytes();state=json.loads(central_bytes)
    need(len(state['completed'])==234 and sum(x['trials'] for x in state['completed'])==1215 and state['registered_queue']==[],'Central234/1215/emptyqueue')
    need(state['completed'][-1]['directory']==OUT.name and state['completed'][-1]['completion']['sha256']==expected,'Central terminal binding')
    need(state['best_training_selected_candidate']==expected_best and state['latest_training_selected_candidate']['selection']==selection,'Central retention')
    rows=changes[ARMS[1]+'_vs_'+ARMS[0]]['months'];by_year=Counter(r['date'][:4] for r in rows)
    clusters=dict(by_signal_year=dict(sorted(by_year.items())),removed_symbol_counts=dict(Counter(s for r in rows for s in r['removed']).most_common()),added_symbol_counts=dict(Counter(s for r in rows for s in r['added']).most_common()),scope='Descriptive saved held-set dates only; not performance contribution or causal attribution.')
    ev.pin(Path(__file__))
    report=dict(schema='ownsector238_saved_result_review_v1',passed=True,created_utc=datetime.now(timezone.utc).isoformat(),reviewer_role='Round238 kernel and kernel-test author performing saved evidence review; independent mathematical reconstruction supplied by separately authored asset oracle and saved independent preparation/accounting receipts. This reader imports no producer/oracle/engine.',scope='Bounded saved JSON, hashes, scalar statistical comparisons, recorded identity/count consistency and holdings-set changes. No source momentum, eligibility, score, order or return replay. No NPZ decoding.',plan_sha256=expected_plan_sha,completion_sha256=expected,prepared_completion_sha256=sha(OUT/'prepared/COMPLETION.json'),result_completion_sha256=sha(OUT/'results/COMPLETION.json'),input_sha256=ev.pins,reader_attempts=[dict(attempt=1,exit_code=1,reader_sha256='ca890b7e13bcd14de401fea7f01562aaf30d8068fb69c1f483a77021401079a1',failure='Executed holdings recorded from frozen orders',cause='Reader incorrectly compared internal research_equity_FB key with frozen META display alias.',repair='Apply the existing ptr169_results_independent.ticker display alias only; ordered symbols and scores still checked exactly. No production/receipt/result changed; no tolerance widened.'),dict(attempt=2,all_checks_passed=True)],formal_synthetic_tests=test_count,manifests=dict(prepared_artifacts=6,result_artifacts=26,all_new_artifact_sha256_verified=True),source_evidence=source,source_admission=dict(source_plan_sha256=admission['source_plan_sha256'],source_report_sha256=admission['source_report_sha256'],source_completion_sha256=admission['source_completion_sha256'],counts=source_report['counts'],repair_counts=source_report['repair_counts'],admitted=True,raw_original_map_count=64,raw_source_files_rehashed=False),selection=selection,first88_guard_details=guard_details,changed_holdings=changes,change_clusters=clusters,metrics=metrics,fee_evidence=fees,incumbent_versus237=incumbent,independent_paths=6,independent_slices=30,independent_training_groups=12,zero_cost_streams=4,independent_metric_tolerance=dict(absolute=1e-7,relative=2e-11,scope='Only saved production versus independent accounting statistic comparisons; no widening. Incumbent cross-round metrics exact, only inherited per-period cost HOLDINGS.fee <=1e-10.'),independent_metric_residual_count=len(ACCOUNTING_RESIDUALS),max_absolute_recorded_metric_residual=max((x['absolute_difference'] for x in ACCOUNTING_RESIDUALS),default=0),protected_archive_reread=False,protected_files_verified_by_registered_closer=21444,strict_pit=False,reused_sample=True,same_close_zero_idealized=True,limitations=context['limitations'],saved_outcomes_read=True,raw_prices_read=False,npz_decoded=False,new_features_computed=False,new_models_fitted=False,new_orders_computed=False,new_returns_computed=False,observers_modified=False,completed_entries=234,trials=1215,central_snapshot_sha256=hashlib.sha256(central_bytes).hexdigest(),best=expected_best,target_reached=closure['target_reached'])
    need(CENTRAL.read_bytes()==central_bytes and sha(OUT/'PLAN.json')==expected_plan_sha and sha(OUT/'COMPLETION.json')==expected,'Terminal evidence changed during review')
    with REPORT.open('x') as f:f.write(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(passed=True,review_sha256=sha(REPORT),selection=selection,changed_counts=changes[ARMS[1]+'_vs_'+ARMS[0]]['counts'],clusters=clusters,same_close={a:metrics[a]['same_close_zero']['full122'] for a in ARMS}),ensure_ascii=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--completion-sha256',required=True);parser.add_argument('--plan-sha256',required=True);parser.add_argument('--tests',type=int,required=True)
    args=parser.parse_args();main(args.completion_sha256,args.plan_sha256,args.tests)
