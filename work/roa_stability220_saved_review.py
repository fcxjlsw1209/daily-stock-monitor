"""Read closed round220 evidence; never decode NPZ, select reports or replay returns.

This reviewer also authored the host. Numerical feature independence is supplied
by the separately authored, pinned oracle and its completed preparation receipt.
Run only after explicit authorization of the terminal root COMPLETION SHA.
"""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import math

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/monthly_roa_stability220'
PRIOR=ROOT/'outputs/monthly_risk_consensus219'
CENTRAL=ROOT/'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
CONTEXT=ROOT/'work/monthly_roa_stability220_context.json'
CONTEXT_SHA='673aa44502d61490fef26a16a426cefa94469cdae9f059948bb471458e7370e5'
SOURCE_REVIEW=ROOT/'work/roa_stability220_SOURCE_REVIEW.json'
SOURCE_REVIEW_SHA='8dfbfd87bb91d703d5712024ee79edb8a695645e12da098274b93e15c9793a19'
SOURCE_MANIFEST_SHA='328b19f5e30406dcec2094b0302f83d97a7c303fae5fa9ca0f8500ee861dffac'
SOURCE_DATASET_SHA='f69dca48d6b5aba713dfa94c5c9435b5eedb90ed6503752cdd077b4b1afc5ad6'
SOURCE_DIR=ROOT/'outputs/monthly_roa_stability_source220_v1'
TESTS=ROOT/'work/monthly_roa_stability220_TESTS.json'
CODE_REVIEW=ROOT/'work/roa_stability220_CODE_REVIEW.json'
LOCAL=ROOT/'outputs/monthly_local_linear70'
PROTECTED=ROOT/'work/trade_reconcile_20261015_close/NEXT_EXPECTED_PROTECTED.json'
RUNTIME={'/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/bin/python3.9':'6b24be8ba00c4abc8b3e7e2620bd710e2cf7a4312d00c147f8f05fbb6f3d297a'}
ARMS=('incumbent_local67','roa_stability_penalty','three_year_roa_mean_control')
PATHS={'same_close_zero':('close',0.),'next_open_zero':('open',0.),'next_open_cost':('open',.0005)}
BOUNDS={'full122':(0,122),'first88':(0,88),'early53':(0,53),'recent35':(53,88),'later34':(88,122)}
PREP={'FEATURES.npz','TRANSFORMS_DELTAS.npz','FEATURE_AUDIT.json','ORDERS.json','META.json','SOURCE_PARITY.json','INDEPENDENT_PREPARE.json'}
RESULT={'RESULTS.json','INDEPENDENT_OUTCOME.json'}|{a+'_'+p+'_'+k+'.json' for a in ARMS for p in PATHS for k in ('DAILY','TRADES','HOLDINGS','DAILY_POSITIONS')}
STAT_KEYS={'start','end','years','final_equity','cagr','sharpe','max_drawdown','trading_sessions','total_return'}
ESTIMATES=('annual_ratio_evaluations','mean_estimates','sample_dispersion_estimates','bounded_scores')
FEATURE_FLAGS=('passed','independent_scalar_rebuild','full_catalog_latest_fiscal_anchor_checked','no_older_fallback_checked','exact_identity_checked','three_consecutive_original_reports_checked','edge_version_clocks_checked','original_fields_only_checked','own_report_assets_checked','latest550_and_historical_age_checked','rational_roa_mean_variance_checked','sample_ddof1_checked','binary64_sqrt_contract_checked','bounded_transform_checked','exact_common_availability','no_cross_sectional_ranking','no_predictive_fits_checked','audit_checked','no_future_or_outcome_inputs')
DELTA_FLAGS=('passed','independent_delta_rebuild','fixed_point01_formula_checked','eligible_only_checked','zero_or_missing_zero_delta_checked','exact_targets_checked','no_second_rank','correction_bound_checked','penalty_monotonicity_checked')
NUMERICAL_FILES=('monthly_roa_stability220_host.py','test_monthly_roa_stability220_host.py','monthly_roa_stability220_kernel.py','test_monthly_roa_stability220_kernel.py','monthly_roa_stability220_independent.py','test_monthly_roa_stability220_independent.py','monthly_roa_stability220_scoring.py','test_monthly_roa_stability220_scoring.py','roa_stability_preflight_math.py','test_roa_stability_preflight_math.py')
PERIOD_FEE_ATOL=1e-10
EXPECTED_TESTS=119
EXPECTED_PLAN_SHA='dc17ecb6c74f4bb0dfae4b37b6fc1b276e7e123adb831493cde18a74194170a4'
EXPECTED_PREPARED_SHA='7a3a10b90d9dbbe83cf8330357563fd14165af0207c0e7a458dd432114637aa9'
EXPECTED_RESULTS_SHA='6a2510ebd6fd0831d11916b6395e30463900c33e6def5e993d8af5964d25e7aa'
EXPECTED_CODE_REVIEW_SHA='9b8510d96c255b1eb9399f5d8b532d8ea6f1bc576a5079e03483f24ef9e21d2c'
ACCOUNTING_RESIDUALS=[]
REVIEW_ATTEMPTS=[{'attempt': 1, 'reader_path': '/private/tmp/roa_stability220_saved_review.py', 'reader_sha256': '91c8467ced5e4e4243856d4f2e39f45883b371bb23a5e150769604dda67e465c', 'log_path': '/private/tmp/roa_stability220_saved_review_attempt1.log', 'log_sha256': 'd9ce27b203c2c768bd02da16f248cc9041f72ffc8d64799fcd40ea35ae143f94', 'exit_code': 1, 'failure': 'Exact next_open_cost metrics/full122 fees_paid parity; no report written.'}, {'attempt': 2, 'reader_path': '/private/tmp/roa_stability220_saved_review_attempt2.py', 'reader_sha256': '382dd87e46fba3925b8d514295cbf32d41fd5b17a4c35e11c23a411eea163a9d', 'log_path': '/private/tmp/roa_stability220_saved_review_attempt2.log', 'log_sha256': '4204b8f073d0e2d1bd36b742e60dc00a4417863cd2f68096d366ecb9dc976244', 'exit_code': 1, 'failure': 'Exact next_open_cost metrics/full122 fees_paid parity; no report written.'}]


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
    ids2=[x['index'] for x in changes[ARMS[1]+'_vs_'+ARMS[2]]['months'] if x['index']<88]
    def support(rows):return len(rows)>=3 and any(i<53 for i in rows) and any(i>=53 for i in rows)
    # Explicit names prevent positional reuse from changing the candidate.
    b,c,m=training[ARMS[0]],training[ARMS[1]],training[ARMS[2]]
    checks=dict(close_vs_incumbent=guard(c['same_close_zero'],b['same_close_zero'],.005),
        open_vs_incumbent=guard(c['next_open_cost'],b['next_open_cost'],0.),
        close_vs_mean_control=guard(c['same_close_zero'],m['same_close_zero'],0.),
        open_vs_mean_control=guard(c['next_open_cost'],m['next_open_cost'],0.),
        changed_support=support(ids),changed_support_vs_control=support(ids2))
    admitted=all(checks.values())
    return dict(primary=ARMS[1] if admitted else ARMS[0],admitted=admitted,guards=checks,
        changed_months=ids,changed_months_vs_control=ids2,first88_only=True,later34_used=False)


def feature_coverage(feature,prepared,completion,closure,meta):
    need(feature.get('schema_version')==1 and feature.get('events')==122 and feature.get('symbols_per_event')==49 and feature.get('checked_cells')==5978,'Full financial cell audit required')
    history=feature.get('event_history');need(isinstance(history,list) and len(history)==122,'Missing financial event audit')
    reasons=Counter();security={};selected_reports=set();selected_edges=set();selected_chains=set();available=[];differences=[]
    support_by_era={name:0 for name in BOUNDS};reasons_by_era={name:Counter() for name in BOUNDS}
    for i,event in enumerate(history):
        need(event['date']==meta['dates'][i] and event['symbols']==meta['symbols'][i] and len(event['cells'])==49,'Financial event identity')
        count=0;different=0
        for j,cell in enumerate(event['cells']):
            symbol=meta['symbols'][i][j]
            need(cell['date']==event['date'] and cell['symbol']==symbol and type(cell['available']) is bool,'Financial cell identity/status')
            reason=cell.get('reason');need(type(reason) is str and reason,'Missing financial exclusion reason')
            reasons[reason]+=1
            item=security.setdefault(symbol,dict(available_cells=0,dates=[],reasons=Counter()))
            item['reasons'][reason]+=1
            for name,(lo,hi) in BOUNDS.items():
                if lo<=i<hi:reasons_by_era[name][reason]+=1
            if cell['available']:
                need(symbol not in meta['etfs'] and reason=='available','Financial ETF or false available reason')
                ids=cell.get('report_version_ids');edges=cell.get('edge_version_ids')
                need(isinstance(ids,list) and len(ids)==len(set(ids))==3 and isinstance(edges,list) and len(edges)==len(set(edges))==2,'Three reports/two edges required')
                need(type(cell.get('latest_event_id')) is str and type(cell.get('chain_version_id')) is str,'Selected source identity missing')
                measurement=cell.get('measurement',{})
                need(set(measurement)=={'ratios_exact','mean_exact','variance_exact','sample_std','candidate','control'} and len(measurement['ratios_exact'])==3,'Stored measurement schema')
                need(all(type(x) is str for x in measurement['ratios_exact']) and type(measurement['mean_exact']) is str and type(measurement['variance_exact']) is str,'Stored exact measurement identities')
                c,m=measurement['candidate'],measurement['control']
                need(finite(c) and finite(m) and -1<=c<=m<=1 and finite(measurement['sample_std']) and measurement['sample_std']>=0,'Saved bounded measurement')
                if measurement['variance_exact']=='0':need(c==m,'Zero dispersion must copy control')
                different+=int(c!=m);count+=1;item['available_cells']+=1;item['dates'].append(event['date'])
                selected_reports.update(ids);selected_edges.update(edges);selected_chains.add(cell['chain_version_id'])
            else:
                need('measurement' not in cell and cell.get('report_version_ids')==[] and cell.get('edge_version_ids')==[] and cell.get('chain_version_id') is None,'Unavailable source carries feature/chain')
        need(type(event['available_count']) is int and event['available_count']==count,'Event financial support total')
        available.append(count);differences.append(different)
        for name,(lo,hi) in BOUNDS.items():
            if lo<=i<hi:support_by_era[name]+=count
    a=sum(available);expected={'annual_ratio_evaluations':3*a,'mean_estimates':a,'sample_dispersion_estimates':a,'bounded_scores':2*a}
    for record in (feature,prepared['feature']):
        for name,value in dict(events=122,symbols_per_event=49,checked_cells=5978,available_cells=a,unavailable_cells=5978-a).items():
            need(type(record.get(name)) is int and record[name]==value,'Financial coverage lineage: '+name)
    for record in (feature,prepared['feature'],prepared,completion,closure):
        for key,value in expected.items():need(type(record.get(key)) is int and record[key]==value,'Financial estimate lineage: '+key)
        for key in ('new_model_fits','predictive_model_fits'):need(type(record.get(key)) is int and record[key]==0,'Unexpected new model fits')
    for item in security.values():item['reasons']=dict(sorted(item['reasons'].items()))
    return dict(checked_cells=5978,available_cells=a,unavailable_cells=5978-a,available_by_event=available,
        source_supported_months=sum(x>0 for x in available),support_by_era=support_by_era,
        reasons=dict(sorted(reasons.items())),reasons_by_era={k:dict(sorted(v.items())) for k,v in reasons_by_era.items()},
        by_security=dict(sorted(security.items())),selected_report_versions=sorted(selected_reports),selected_edge_versions=sorted(selected_edges),selected_chain_versions=sorted(selected_chains),
        saved_candidate_control_different_cells=sum(differences),saved_difference_cells_by_era={k:sum(differences[lo:hi]) for k,(lo,hi) in BOUNDS.items()},estimate_counts=expected)


def compare_incumbent(evidence,prior_completion,results,prior_results):
    report={}
    for name in PATHS:
        old=prior_results['arms'][ARMS[0]][name];new=results['arms'][ARMS[0]][name]
        accumulated_fee_residuals=[]
        def exact_metrics_with_declared_fee_exception(previous,current,location,allow_fee):
            need(set(previous)==set(current),'Incumbent metric schema differs: '+location)
            for key in previous:
                old_value=previous[key];new_value=current[key]
                if old_value==new_value:continue
                need(name=='next_open_cost' and allow_fee and key=='fees_paid','Nonfee incumbent metric differs: '+location+'/'+key)
                need(finite(old_value) and finite(new_value),'Nonfinite accumulated incumbent fee')
                delta=new_value-old_value;bound=1e-7+2e-11*abs(old_value)
                need(abs(delta)<=bound,'Accumulated fee exceeds frozen219 accounting tolerance')
                accumulated_fee_residuals.append(dict(field=location+'/'+key,previous219=old_value,current220=new_value,difference=delta,absolute_difference=abs(delta),allowed_absolute_difference=bound))
        exact_metrics_with_declared_fee_exception(old['metrics'],new['metrics'],'metrics',True)
        need(set(old['slices'])==set(new['slices'])==set(BOUNDS),'Incumbent slice schema differs')
        for label in BOUNDS:
            exact_metrics_with_declared_fee_exception(old['slices'][label],new['slices'][label],'slices/'+label,label=='full122')
        exported={}
        for kind in ('TRADES','DAILY','DAILY_POSITIONS','HOLDINGS'):
            filename=ARMS[0]+'_'+name+'_'+kind+'.json'
            previous=evidence.load(PRIOR/'results'/filename,prior_completion['artifacts_sha256'][filename])
            current=evidence.load(OUT/'results'/filename)
            if kind!='HOLDINGS':need(current==previous,'Incumbent export must be EXACT: '+name+'/'+kind)
            exported[kind]=(previous,current)
        prior_holds,current_holds=exported['HOLDINGS'];need(len(prior_holds)==len(current_holds)==122,'Incumbent periods')
        residuals=[]
        for old_row,new_row in zip(prior_holds,current_holds):
            need(set(old_row)==set(new_row) and {k:v for k,v in old_row.items() if k!='fee'}=={k:v for k,v in new_row.items() if k!='fee'},'Incumbent nonfee HOLDINGS field differs')
            need(finite(old_row['fee']) and finite(new_row['fee']),'Nonfinite incumbent fee')
            if old_row['fee']!=new_row['fee']:
                delta=new_row['fee']-old_row['fee']
                need(name=='next_open_cost' and abs(delta)<=PERIOD_FEE_ATOL,'Only existing period-fee1e-10 allowance is permitted')
                residuals.append(dict(signal_date=new_row['signal_date'],execution_date=new_row['execution_date'],previous219=old_row['fee'],current220=new_row['fee'],difference=delta))
        totals={}
        for label,holds,metrics in (('previous219',prior_holds,old['metrics']),('current220',current_holds,new['metrics'])):
            total=0.
            for row in holds:total+=row['fee']
            need(total==metrics['fees_paid'],'Saved chronological fees do not exactly sum to metric')
            totals[label]=total
        # Each saved chronological sum must equal its own metric exactly; the only
        # permitted cross-round difference is the metrics.fees_paid exception above.
        report[name]=dict(all_metrics_and_slices_exact=not accumulated_fee_residuals,all_nonfee_metrics_and_slices_exact=True,accumulated_fee_differences=accumulated_fee_residuals,all_fills_exact=True,daily_equity_exact=True,daily_positions_exact=True,
            holding_nonfee_fields_exact=True,total_fees_exact=totals['previous219']==totals['current220'],fee_totals=totals,period_fee_differences=residuals,
            max_absolute_period_fee_difference=max([abs(x['difference']) for x in residuals],default=0.))
    return dict(paths=report,period_fee_tolerance=PERIOD_FEE_ATOL,
        tolerance_scope='Only individual next_open_cost HOLDINGS.fee may differ by at most1e-10. Only next_open_cost metrics.fees_paid and slices.full122.fees_paid use the existing219 accumulated-accounting tolerance1e-7+2e-11*abs(previous). All other metrics/slices including turnover, fills, equity, positions and all nonfee holding fields are exact.',accumulated_fee_tolerance_source='work/risk_consensus219_saved_review.py:34-35,124',
        explanation='Frozen engine iterates a set when summing turnover; per-period fee summation tails may vary between processes. No new replay or relaxed total/metric comparison is used.')


def main(expected,test_count):
    need(type(test_count) is int and test_count==EXPECTED_TESTS,'Final119 formal synthetic count required')
    ACCOUNTING_RESIDUALS.clear()
    need(not (OUT/'RESEARCH_REVIEW.json').exists() and not (OUT/'RESEARCH_REVIEW.md').exists(),'Saved review is write-once')
    evidence=Evidence();closure=evidence.load(OUT/'COMPLETION.json',expected)
    plan=evidence.load(OUT/'PLAN.json',EXPECTED_PLAN_SHA);plan_sha=EXPECTED_PLAN_SHA
    need(closure.get('status')=='COMPLETED_INDEPENDENTLY_VERIFIED' and closure['plan_sha256']==plan_sha and closure['protected_files']==21444 and closure['observers_modified'] is False,'Terminal closure required')
    need(isinstance(plan.get('input_sha256'),dict) and plan['input_sha256'],'Registered input closure missing')
    for filename,digest in plan['input_sha256'].items():evidence.pin(filename,digest)
    context=evidence.load(CONTEXT,CONTEXT_SHA)
    need(registered_digest(plan,CONTEXT)==CONTEXT_SHA==plan['contract']['prospective_context_sha256'],'Context not registered exactly')
    need(plan['contract']['context']==context and context['arms']==list(ARMS) and context['dose']==.01,'Contract/source theory differs')
    need(plan['protected_files']==21444 and len(evidence.load(PROTECTED,registered_digest(plan,PROTECTED)))==21444,'Registered protection manifest')
    before=evidence.load(OUT/'STATE_BEFORE.json',plan['prior_central_sha256'])
    need(len(before['completed'])==217 and sum(r['trials'] for r in before['completed'])==1168 and not before['registered_queue'],'Pre-registration state differs')
    prior_binding=plan['predecessor_binding'];need(prior_binding['directory']==PRIOR.name,'Wrong predecessor')
    for filename,digest in prior_binding['input_sha256'].items():evidence.pin(filename,digest)
    prior_plan=evidence.load(PRIOR/'PLAN.json',registered_digest(plan,PRIOR/'PLAN.json'))
    prior_plan_sha=sha(PRIOR/'PLAN.json')
    prior_root=evidence.load(PRIOR/'COMPLETION.json',registered_digest(plan,PRIOR/'COMPLETION.json'))
    need(before['completed'][-1]['directory']==PRIOR.name and before['completed'][-1]['completion']['sha256']==sha(PRIOR/'COMPLETION.json'),'Prior central identity')
    prior_done=evidence.load(PRIOR/'results/COMPLETION.json',registered_digest(plan,PRIOR/'results/COMPLETION.json'))
    need(prior_root['plan_sha256']==prior_done['plan_sha256']==prior_plan_sha and prior_root['result_completion_sha256']==sha(PRIOR/'results/COMPLETION.json') and prior_done['status']=='COMPLETE','Prior outcome lineage')
    prior_results=evidence.load(PRIOR/'results/RESULTS.json',prior_done['artifacts_sha256']['RESULTS.json'])
    tests=evidence.load(TESTS,registered_digest(plan,TESTS))
    need(tests['passed'] is True and tests['synthetic_only'] is True and tests['tests']==test_count and tests['runtime']==plan['runtime'],'Formal tests/runtime differ')
    for filename,digest in tests['code_sha256'].items():
        need(registered_digest(plan,filename)==digest,'Tested source not registered');evidence.pin(filename,digest)
    need(registered_digest(plan,CODE_REVIEW)==EXPECTED_CODE_REVIEW_SHA,'Final code review SHA differs')
    review=evidence.load(CODE_REVIEW,EXPECTED_CODE_REVIEW_SHA)
    wanted_codes={str((ROOT/'work'/name).relative_to(ROOT)):registered_digest(plan,ROOT/'work'/name) for name in NUMERICAL_FILES}
    need(review['passed'] is True and review['unresolved_blockers']==[] and review['context_sha256']==CONTEXT_SHA and review['code_sha256']==wanted_codes,'Full ten-file code review differs')
    for key in ('actual_features_computed','actual_model_fits_computed','actual_portfolio_returns_computed','production_files_modified','strategy_registered_by_reviewer'):
        need(review.get(key) is False,'Code-review scope differs: '+key)
    for filename,digest in review['input_sha256'].items():evidence.pin(filename,digest)
    need(plan['numerical_code_review']['sha256']==sha(CODE_REVIEW) and path(plan['numerical_code_review']['path'])==CODE_REVIEW,'Registered numerical review binding')
    frozen=evidence.load(OUT/'selected/FROZEN.json');training=evidence.load(OUT/'selected/TRAIN_RESULTS.json')
    need(frozen['plan_sha256']==plan_sha and frozen['training_sha256']==sha(OUT/'selected/TRAIN_RESULTS.json'),'Frozen first88 source')
    need(frozen['prepared_sha256']==EXPECTED_PREPARED_SHA and closure['result_completion_sha256']==EXPECTED_RESULTS_SHA,'Confirmed phase hashes differ')
    pc=evidence.phase(OUT/'prepared',EXPECTED_PREPARED_SHA,plan_sha,PREP)
    rc=evidence.phase(OUT/'results',closure['result_completion_sha256'],plan_sha,RESULT)
    need(rc['paths']==9 and rc['slices']==45 and rc['frozen_sha256']==sha(OUT/'selected/FROZEN.json'),'Outcome/frozen lineage')
    need(frozen['orders_sha256']==sha(OUT/'prepared/ORDERS.json'),'Frozen prepared orders')
    meta=evidence.load(OUT/'prepared/META.json');orders=evidence.load(OUT/'prepared/ORDERS.json')
    need(meta==evidence.load(LOCAL/'META.json',registered_digest(plan,LOCAL/'META.json')),'Original monthly metadata differs')
    need(len(meta['dates'])==len(meta['ends'])==len(meta['symbols'])==122 and all(len(set(s))==49 for s in meta['symbols']),'Original122x49 axes')
    need(meta['ends'][:-1]==meta['dates'][1:] and meta['dates'][0]=='2016-08-17' and meta['ends'][-1]=='2026-09-24','Original period/calendar')
    need(meta['ends'][52]==meta['dates'][53]=='2020-12-31' and meta['ends'][87]==meta['dates'][88]=='2023-11-30','Frozen split identities')
    need(set(orders)==set(ARMS) and all(len(v)==122 for v in orders.values()),'Three complete order streams')
    need(orders[ARMS[0]]==evidence.load(LOCAL/'local_linear_ORDERS.json',registered_digest(plan,LOCAL/'local_linear_ORDERS.json')),'Incumbent orders differ')
    for rows in orders.values():
        for i,row in enumerate(rows):
            need(row['date']==row['signal_date']==meta['dates'][i] and row['end']==meta['ends'][i],'Order timestamps')
            need(len(row['symbols']) in (0,2,3) and len(set(row['symbols']))==len(row['symbols']) and set(row['symbols'])<=set(meta['symbols'][i]),'Order targets outside original universe')
    prepared=evidence.load(OUT/'prepared/INDEPENDENT_PREPARE.json');feature=evidence.load(OUT/'prepared/FEATURE_AUDIT.json')
    for key in ('original_orders_exact','original_arrays_unchanged','single_original_episode','original_core_metadata_and_support_preserved','financial_source_selection_exact','feature_prefix_sources_truncated','feature_prefix_metadata_truncated','feature_prefix_independently_verified'):
        need(prepared.get(key) is True,'Independent prepare invariant missing: '+key)
    need(prepared['prefixes']==123 and prepared['feature_prefixes']==[0,1,53,88,122] and prepared['original_array_count']==pc['original_array_count']==closure['original_array_count']==27,'Prefix/original-array coverage')
    need(pc['new_returns_computed'] is False,'Preparation must not calculate returns')
    for key in FEATURE_FLAGS:need(prepared['feature'].get(key) is True,'Independent financial flag missing: '+key)
    for key in DELTA_FLAGS:need(prepared['delta'].get(key) is True,'Independent delta flag missing: '+key)
    need(prepared['feature']['computes_returns'] is False and prepared['feature']['producer_modules_imported'] is False and prepared['feature']['numerical_tolerance']==0.,'Independent numeric scope differs')
    need(prepared['delta']['dose']==prepared['delta']['max_absolute_delta']==.01 and set(prepared['delta']['changed_cells'])==set(ARMS[1:]),'Independent fixed score budget differs')
    for record in (prepared,pc,closure):
        need(type(record['new_predictive_model_fits']) is int and record['new_predictive_model_fits']==0 and record['model_fit_accounting']==plan['model_fit_accounting'],'Predictive fit/accounting scope differs')
    coverage=feature_coverage(feature,prepared,pc,closure,meta)
    source=evidence.load(OUT/'prepared/SOURCE_PARITY.json');binding=plan['financial_source_binding']
    need(source['raw_source_hashes']==plan['raw_source_hashes'] and len(plan['raw_source_hashes'])==64 and source['financial_source_binding']==binding,'Source parity lineage differs')
    for key in ('financial_features_from_versioned_original_reports','source_selection_independently_checked','no_fresh_raw_pit_claim'):need(source.get(key) is True,'Source scope missing: '+key)
    sr=evidence.load(SOURCE_REVIEW,SOURCE_REVIEW_SHA)
    need(path(binding['path'])==SOURCE_REVIEW and binding['sha256']==SOURCE_REVIEW_SHA and sr['passed'] is True and sr['context_sha256']==CONTEXT_SHA,'Registered source review')
    need(binding['manifest_sha256']==source['financial_source_manifest_sha256']==SOURCE_MANIFEST_SHA and binding['dataset_sha256']==source['financial_source_dataset_sha256']==SOURCE_DATASET_SHA,'Financial source package identity')
    for key in ('source_evidence_sha256','context_evidence_sha256'):
        for filename,digest in binding[key].items():evidence.pin(filename,digest)
    for key in ('actual_new_features_computed','actual_new_orders_computed','actual_new_returns_computed'):need(sr[key] is False,'Source review crossed preregistration computation boundary')
    source_validation=evidence.load(SOURCE_DIR/'VALIDATION.json',registered_digest(plan,SOURCE_DIR/'VALIDATION.json'))
    need(source_validation['passed'] is True and source_validation['source_only'] is True,'Source-only validation incomplete')
    results=evidence.load(OUT/'results/RESULTS.json');independent=evidence.load(OUT/'results/INDEPENDENT_OUTCOME.json')
    need(results['selection']==frozen['selection'] and results['strict_pit'] is False and results['reused_sample'] is True and results['observer_promotion'] is False and results['roa_stability'] is True and results['same_support_both_arms'] is True,'Outcome interpretation or selection differs')
    need(independent['passed'] is True and independent['independent_accounts'] is True and independent['independent_of_production_model_and_engine'] is True and independent['checked_paths']==9 and independent['checked_slices']==45 and independent['terminal_liquidation'] is False,'Independent execution scope')
    need(independent['accounting_source_sha256']==registered_digest(plan,ROOT/'work/ptr169_results_independent.py'),'Independent accounting source')
    need(set(results['arms'])==set(independent['paths'])==set(training)==set(ARMS),'Outcome/training arms')
    metrics={};fee_evidence={};training_checks=0;zero_streams=0
    for arm in ARMS:
        need(set(results['arms'][arm])==set(independent['paths'][arm])==set(PATHS),'Three execution paths required')
        need(set(training[arm])=={'same_close_zero','next_open_cost'},'Training uses undeclared execution path')
        metrics[arm]={};fee_evidence[arm]={}
        for name,(execution,cost) in PATHS.items():
            result=results['arms'][arm][name];m=result['metrics'];proof=independent['paths'][arm][name]
            need(m['execution']==execution and m['cost_per_side']==cost and m['start']==meta['dates'][0] and m['end']==meta['ends'][-1],'Execution identity/price timing')
            need(proof['passed'] is True and proof['monthly_executions']==122 and set(proof['slices'])==set(result['slices'])==set(BOUNDS),'Complete independent slices')
            need(result['accounting']['exported_fill_ledger_matches'] is True and result['accounting']['terminal_liquidation'] is False,'Daily ledger verification absent')
            metric_evidence(proof['slices']['full122'],m,arm+'/'+name+'/metrics')
            metrics[arm][name]={}
            for label,(lo,hi) in BOUNDS.items():
                value=result['slices'][label]
                need(value['start']==meta['dates'][lo] and value['end']==meta['ends'][hi-1],'Saved split boundary')
                metric_evidence(proof['slices'][label],value,arm+'/'+name+'/'+label)
                metrics[arm][name][label]={k:value[k] for k in ('start','end','cagr','sharpe','max_drawdown')}
            for key in ('fees_paid','dollar_turnover','trade_rows'):numerical_close(m[key],proof[key],arm+'/'+name+'/'+key)
            if name in training[arm]:
                need(set(training[arm][name])=={'training','early','recent'},'Three training periods required')
                for subset,label in (('training','first88'),('early','early53'),('recent','recent35')):
                    need(set(training[arm][name][subset])==STAT_KEYS,'Training metric set')
                    metric_evidence(proof['slices'][label],training[arm][name][subset],arm+'/'+name+'/'+subset);training_checks+=1
            trades=evidence.load(OUT/'results'/f'{arm}_{name}_TRADES.json')
            holdings=evidence.load(OUT/'results'/f'{arm}_{name}_HOLDINGS.json')
            need(len(holdings)==122 and m['trade_rows']==len(trades),'Fee stream coverage')
            need(all(finite(x['cost']) and x['cost']>=0 for x in trades) and all(finite(x['fee']) and x['fee']>=0 for x in holdings),'Invalid saved fees')
            if cost==0.:
                need(m['fees_paid']==0. and proof['fees_paid']==0. and all(x['cost']==0. for x in trades) and all(x['fee']==0. for x in holdings),'Nonzero fee in a zero-cost stream')
                zero_streams+=1
            fee_evidence[arm][name]=dict(execution=execution,cost_per_side=cost,fees_paid=m['fees_paid'],trade_rows=len(trades),holding_rows=len(holdings),all_zero_fee_fields_checked=cost==0.)
    need(training_checks==18 and zero_streams==6,'Incomplete training/zero-cost checks')
    incumbent=compare_incumbent(evidence,prior_done,results,prior_results)
    changes={a+'_vs_'+b:changed(orders,b,a) for a,b in ((ARMS[1],ARMS[0]),(ARMS[2],ARMS[0]),(ARMS[1],ARMS[2]))}
    selection=selection_from_saved(training,changes);need(selection==frozen['selection'],'Frozen first88 guards differ')
    expected_best=before['best_training_selected_candidate'];pick=selection['primary'];picked=results['arms'][pick]
    if selection['admitted'] and picked['same_close_zero']['metrics']['cagr']>expected_best['cagr']:
        m=picked['same_close_zero']['metrics'];expected_best=dict(directory=OUT.name,spec=dict(arm=pick),cagr=m['cagr'],sharpe=m['sharpe'],max_drawdown=m['max_drawdown'],next_open_5bp_cagr=picked['next_open_cost']['metrics']['cagr'],selected_on_first88=True,observer_promotion=False,limitations=plan['contract']['limitation'])
    need(closure['best_training_selected_candidate']==expected_best and closure['target_reached']==(expected_best['cagr']>=.7),'Best/target update differs from frozen admission')
    central_bytes=CENTRAL.read_bytes();state=json.loads(central_bytes)
    need(len(state['completed'])==218 and sum(x['trials'] for x in state['completed'])==1171 and state['registered_queue']==[],'Central postclosure counters')
    need(state['completed'][-1]['directory']==OUT.name and state['completed'][-1]['completion']['sha256']==expected and path(state['completed'][-1]['completion']['path'])==OUT/'COMPLETION.json','Central closure reference')
    need(state['best_training_selected_candidate']==expected_best and state['latest_training_selected_candidate']['selection']==selection,'Central selected/best mismatch')
    evidence.pin(Path(__file__))
    report=dict(passed=True,created_utc=datetime.now(timezone.utc).isoformat(),reviewer_role='Saved-evidence reviewer is also the220 host author; independent numerical feature rebuild is the separately authored pinned oracle.',
        scope='Saved JSON/hash/source-coverage/selection/accounting evidence only; no NPZ decoded, financial feature recomputed, order generated or portfolio return recalculated.',
        completion_sha256=expected,plan_sha256=plan_sha,prepared_completion_sha256=sha(OUT/'prepared/COMPLETION.json'),result_completion_sha256=sha(OUT/'results/COMPLETION.json'),input_sha256=evidence.pins,
        formal_synthetic_tests=test_count,reviewed_numerical_files=10,prior_failed_review_attempts=REVIEW_ATTEMPTS,source_coverage=coverage,selection=selection,changed_holdings=changes,metrics=metrics,fee_evidence=fee_evidence,
        incumbent_versus219=incumbent,independent_recorded_metric_residuals=ACCOUNTING_RESIDUALS,
        independent_metric_tolerance=dict(absolute=1e-7,relative=2e-11,scope='Comparisons with separately recorded independent accounting statistics inherit pinned ptr169_results_independent.check_metrics. The same preexisting219 bound is separately restricted to the two next_open_cost accumulated fees_paid fields; all residuals listed. Never used for zero-cost fees or any other incumbent metric.'),independent_paths=9,independent_slices=45,independent_training_groups=18,zero_cost_streams=6,
        order_prefixes=123,source_prefixes=[0,1,53,88,122],original_array_count=27,original_arrays_checked_by_independent_prepare=True,protected_files_verified_by_registered_host_and_closer=21444,
        source_package_counts=source_validation['counts'],limitations=context['limitations'],strict_pit=False,reused_sample=True,observers_modified=False,
        actual_new_features_computed=False,actual_new_orders_computed=False,actual_portfolio_returns_computed=False,source_npz_decoded=False,
        best=expected_best,target_reached=closure['target_reached'],completed_entries=218,trials=1171,
        central_snapshot_sha256=hashlib.sha256(central_bytes).hexdigest())
    lines=['# 第220轮：三年原报ROA稳定性','',
        '候选通过前88期筛选。' if selection['admitted'] else '候选未通过前88期筛选，保留原Local67。','',
        '候选使用三份连续原年度报告的ROA均值减样本标准差，对照仅使用均值；二者经h(x)=x/(1+|x|)映射，以0.01系数进入原pre-score。原长期动量、股票池、月末等权及单次episode选择保留。','',
        '|执行口径（全122期）|原Local67：年化 / Sharpe / 回撤|稳定性候选|三年均值对照|','|---|---:|---:|---:|']
    def fmt(value):return f"{value['cagr']:.2%} / {value['sharpe']:.3f} / {value['max_drawdown']:.2%}"
    for name,label in (('same_close_zero','同收盘零成本'),('next_open_zero','次开盘零成本'),('next_open_cost','次开盘每边0.05%')):
        lines.append('|'+label+'|'+'|'.join(fmt(metrics[a][name]['full122']) for a in ARMS)+'|')
    lines+=['','|零成本同收盘区间|原Local67|稳定性候选|三年均值对照|','|---|---:|---:|---:|']
    for label in BOUNDS:lines.append('|'+label+'|'+'|'.join(fmt(metrics[a]['same_close_zero'][label]) for a in ARMS)+'|')
    lines+=['','|持仓集合变化比较|全期|前88|前53|后35训练期|后34诊断期|','|---|---:|---:|---:|---:|---:|']
    for label,change in changes.items():lines.append('|'+label+'|'+'|'.join(str(change['counts'][k]) for k in BOUNDS)+'|')
    lines+=['',f"5978个证券月份中，财务共同可用{coverage['available_cells']}格、不可用{coverage['unavailable_cells']}格；有可用来源的月份{coverage['source_supported_months']}个。",'', '|来源状态|证券月份数|','|---|---:|']
    for reason,count in coverage['reasons'].items():lines.append('|'+reason.replace('|','/')+'|'+str(count)+'|')
    lines+=['',f'核对{test_count}项正式合成测试、10份数值源码审查、原27数组保留收据、5组物理来源前缀、123订单前缀、9路径45区间与18组冻结训练核算。六条零成本流的总费用、逐笔费用和逐月费用全部严格为零。','',
        'Local67三路径的收益、Sharpe、回撤、净值、周转额及五区间非费用指标与219精确一致；逐笔成交、每日权益与仓位精确一致。收费路径的metrics.fees_paid及full122.fees_paid仅沿用219既有累计会计容差，逐项披露实际舍入差；逐月fee仍限既有1e-10。两次更强exact检查失败的源码及日志哈希保留于JSON，未修改冻结策略或证据。','',
        '来源限制：文档日加下一交易日是声明的可用时点假设；尚未认证首发版本、完整更正历史、永久证券身份及统一经营边界。40条条件准入报告链不等于40条月度投资机会。','',
        '后34期和全期已反复用于研究，属于诊断样本。审查作者同时负责宿主，数值独立性来自另行编写的oracle。本脚本未重算特征、订单或收益，也未修改观察账户。','']
    need(CENTRAL.read_bytes()==central_bytes,'Central changed during final review')
    with (OUT/'RESEARCH_REVIEW.json').open('x') as stream:stream.write(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    with (OUT/'RESEARCH_REVIEW.md').open('x') as stream:stream.write('\n'.join(lines))
    print(json.dumps(dict(passed=True,review_sha256=sha(OUT/'RESEARCH_REVIEW.json'),markdown_sha256=sha(OUT/'RESEARCH_REVIEW.md'),selection=selection,available_cells=coverage['available_cells'],same_close={a:metrics[a]['same_close_zero']['full122'] for a in ARMS}),ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--completion-sha256',required=True)
    parser.add_argument('--tests',type=int,required=True)
    args=parser.parse_args();main(args.completion_sha256,args.tests)
