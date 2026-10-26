"""Read terminal231 saved evidence; no model imports, array decoding or replay.
Prospectively fixed before231 actual outputs: ONLY cost-path period HOLDINGS.fee
may differ <=1e-10 from230. Cumulative fees and all other incumbent fields exact.
Production-vs-independent nine-statistic accounting retains its fixed bound.
Run only after root authorization of explicit terminal COMPLETION/PLAN/review SHAs.
"""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import math
import zipfile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/monthly_market_drift231'
PRIOR=ROOT/'outputs/monthly_separate_history230'
CENTRAL=ROOT/'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
CONTEXT=ROOT/'work/monthly_market_drift231_context.json'
CONTEXT_SHA='bfa2fc65db4a3e1a50cda8979eef65e513049f196a70cc0647864e0d50b66de9'
SOURCE_REVIEW=ROOT/'work/market_drift231_SOURCE_REVIEW.json'
SOURCE_REVIEW_SHA='908ecdab164621bb7dd4571654cc5e05dcb87511c626b78f9a691ab20a43539a'
TESTS=ROOT/'work/monthly_market_drift231_TESTS.json'
CODE_REVIEW=ROOT/'work/market_drift231_CODE_REVIEW.json'
LOCAL=ROOT/'outputs/monthly_local_linear70'
PROTECTED=ROOT/'work/trade_reconcile_20261015_close/NEXT_EXPECTED_PROTECTED.json'
RUNTIME={'/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/bin/python3.9':'6b24be8ba00c4abc8b3e7e2620bd710e2cf7a4312d00c147f8f05fbb6f3d297a'}
ARMS=('incumbent_local67','market_conditioned_drift','unconditional_drift_control')
PATHS={'same_close_zero':('close',0.),'next_open_zero':('open',0.),'next_open_cost':('open',.0005)}
BOUNDS={'full122':(0,122),'first88':(0,88),'early53':(0,53),'recent35':(53,88),'later34':(88,122)}
PREP={'FEATURES.npz','SCORES.npz','MODEL_AUDIT.json','ORDERS.json','META.json','SOURCE_PARITY.json','INDEPENDENT_PREPARE.json'}
RESULT={'RESULTS.json','INDEPENDENT_OUTCOME.json'}|{a+'_'+p+'_'+k+'.json' for a in ARMS for p in PATHS for k in ('DAILY','TRADES','HOLDINGS','DAILY_POSITIONS')}
STAT_KEYS={'start','end','years','final_equity','cagr','sharpe','max_drawdown','trading_sessions','total_return'}
FEATURE_PREFIXES=(0,1,53,88,122)
NUMERICAL_FILES=('monthly_market_drift231_host.py','test_monthly_market_drift231_host.py','monthly_market_drift231_model.py','test_monthly_market_drift231_model.py','monthly_market_drift231_independent.py','test_monthly_market_drift231_independent.py')
PERIOD_FEE_ATOL=1e-10
ACCOUNTING_RESIDUALS=[]
# Formal test count must match the explicit CLI and immutable registered receipt.
# Final PLAN and code-review digests must be provided explicitly at authorized execution.


CALENDAR=ROOT/'outputs/monthly_paired_target143/CALENDAR.json'

MODEL_FLAGS=('passed', 'independent_scalar_rebuild', 'original_calm_window_checked', 'official_session_windows_checked', 'skip42_stock_window_checked', 'current_market_state_checked', 'previous_day_market_states_checked', 'strict_spy200_comparison_checked', 'mean_log_drift_checked', 'same_window_unconditional_control_checked', 'fixed_63_shrinkage_checked', 'minimum_42_matches_checked', 'complete_eligible_universe_checked', 'whole_month_common_fallback_checked', 'symmetric_eligible_effect_ranks_checked', 'constant_drift_zero_effect_checked', 'initial_partial_unchanged', 'no_forward_labels', 'no_future_inputs', 'audit_checked', 'counts_checked', 'no_predictive_fits_checked')
SELECTION_FLAGS=('passed', 'independent_original_pre_checked', 'original_support_checked', 'original_budget_checked', 'eligible_universe_reranked_checked', 'replacement_not_additive_checked', 'one_episode_per_arm', 'paired_fallback_checked', 'constant_effect_not_fallback_checked', 'exact_scores_and_choices_checked')
COUNT_FIELDS=('query_months', 'eligible_cells', 'supported_months', 'fallback_months', 'initial_partial_months', 'domain_fallback_months', 'source_fallback_months', 'state_support_fallback_months', 'checked_stock_cells', 'source_valid_stock_cells', 'stock_price_observations', 'stock_log_returns', 'drift_estimates', 'ranked_observations', 'market_state_evaluations', 'market_price_observations')
ARRAY_KEYS=('candidate_raw', 'control_raw', 'candidate_effect', 'control_effect', 'ready')
PROPOSAL=ROOT/'work/market_drift231_PROPOSAL.json'
PROPOSAL_SHA='46556aa1b5d6176c6960ae08fbfe5e1190b7042c15d8f321eaa4d1f7e6463b73'
CLARIFICATIONS=ROOT/'work/market_drift231_CLARIFICATIONS.json'
CLARIFICATIONS_SHA='40675e741285248f15abc47114ba4105268451d6b1f9eb883af7f35b271dbf9c'

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
        open_vs_incumbent=guard(c['next_open_zero'],b['next_open_zero'],0.),
        close_vs_unconditional_control=guard(c['same_close_zero'],m['same_close_zero'],0.),
        open_vs_unconditional_control=guard(c['next_open_zero'],m['next_open_zero'],0.),
        changed_support=support(ids),changed_support_vs_control=support(ids2))
    admitted=all(checks.values())
    return dict(primary=ARMS[1] if admitted else ARMS[0],admitted=admitted,guards=checks,
        changed_months=ids,changed_months_vs_control=ids2,first88_only=True,later34_used=False)



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
                residuals.append(dict(signal_date=b['signal_date'],execution_date=b['execution_date'],previous230=a['fee'],current231=b['fee'],difference=difference))
        for rows,metric in ((oh,old['metrics']),(nh,new['metrics'])):
            total=0.
            for row in rows:total+=row['fee']
            need(total==metric['fees_paid'],'Saved chronological fees exactly equal cumulative fee metric')
        report[name]=dict(all_metrics_and_slices_exact=True,total_fees_exact=True,all_fills_exact=True,daily_equity_exact=True,daily_positions_exact=True,holding_nonfee_fields_exact=True,period_fee_differences=residuals,max_absolute_period_fee_difference=max([abs(x['difference']) for x in residuals],default=0.))
    return dict(paths=report,period_fee_tolerance=PERIOD_FEE_ATOL,tolerance_scope='ONLY individual next_open_cost HOLDINGS.fee <=1e-10; all metrics/slices including cumulative fees, turnover, trades, DAILY, positions and holding nonfee fields EXACT; zero-cost fee fields EXACT0.',prospective_policy='Inherited frozen230 comparison scope: no accumulated-fee exception and no result-driven widening.')


def model_check(check,counts,n):
    need(all(check.get(k) is True for k in MODEL_FLAGS) and len(MODEL_FLAGS)==22,'All22 independent feature flags')
    need(check['counts']==counts and check['query_rows']==n and check['numerical_tolerance']==0.,'Independent feature counters/zero tolerance')
    need(check['producer_modules_imported'] is False and check['computes_returns'] is False,'Independent feature scope')
    for k in ('new_model_fits','predictive_model_fits','new_ridge_fits','new_tree_fits'):need(type(check[k]) is int and check[k]==0,'No predictive fit '+k)


def selection_check(check,n):
    need(len(SELECTION_FLAGS)==10 and all(check.get(k) is True for k in SELECTION_FLAGS),'All10 independent replacement selection flags')
    need(check['ordered_choices_checked']==3*n and check['numerical_tolerance']==0. and check['original_source_parity_tolerance']==1e-12,'Exact independent selection proof')
    need(check['producer_modules_imported'] is False and check['computes_returns'] is False,'Independent selection scope')


def model_coverage(audit,prepared,completion,closure,meta,calendar,orders):
    """Recount saved support and clocks, never rebuild source means or ranks."""
    axis=calendar['sessions'];need(axis==sorted(set(axis)),'Official ordered calendar')
    positions={d:i for i,d in enumerate(axis)};monthends={d[:7]:d for d in axis}
    expected=dict(method='current_market_conditioned_skip42_daily_drift',arms=list(ARMS),events=122,
        symbols_per_event=49,skip_sessions=42,calm_returns=210,other_returns=252,market_symbol='SPY',
        market_mean_sessions=200,min_matching_days=42,shrinkage_days=63,
        market_price_observation_counting='200 per evaluated state; overlapping windows counted repeatedly')
    need(all(audit[k]==v for k,v in expected.items()),'Fixed drift method/axes')
    events=audit['event_history'];counts=audit['counts']
    need(len(events)==122 and set(counts)==set(COUNT_FIELDS) and len(COUNT_FIELDS)==16,'122 events/16 counters')
    statuses={'initial_partial','fewer_than_two_eligible','insufficient_calendar','missing_market_symbol',
        'missing_market_session','nonnumeric_market_price','invalid_market_price','invalid_stock_source',
        'insufficient_matching_state','supported'}
    total=dict.fromkeys(COUNT_FIELDS,0);by_year={};supported=[];constant={a:[] for a in ARMS[1:]};sources=[]
    for i,e in enumerate(events):
        day=meta['dates'][i];names=meta['symbols'][i];ids=e['eligible_indices'];cells=e['cells'];status=e['status']
        need(e['row']==i and e['date']==day and e['symbols']==names and len(names)==len(set(names))==49,'Exact event identity')
        need(day in positions and (i==0 and day=='2016-08-17' or monthends[day[:7]]==day),'Formal query day or sole initial exception')
        need(type(e['calm']) is bool and type(e['ready']) is bool and status in statuses,'State/status types')
        L=210 if e['calm'] else 252;t=positions[day];ready=status=='supported'
        need(e['formation_returns']==L and e['ready']==ready,'Original calm branch/ready')
        need(ids==sorted(set(ids)) and all(type(j) is int and 0<=j<49 for j in ids),'Original eligible indices')
        need(len(cells)==len(ids) and [c['column'] for c in cells]==ids and [c['symbol'] for c in cells]==[names[j] for j in ids],'Complete original eligible cell axis')
        c=dict.fromkeys(COUNT_FIELDS,0);c.update(query_months=1,eligible_cells=len(ids),supported_months=int(ready),fallback_months=int(not ready),
            initial_partial_months=int(status=='initial_partial'),domain_fallback_months=int(status=='fewer_than_two_eligible'),
            state_support_fallback_months=int(status=='insufficient_matching_state'),
            source_fallback_months=int(status not in ('initial_partial','fewer_than_two_eligible','insufficient_matching_state','supported')))
        if status=='initial_partial':need(i==0 and day=='2016-08-17','Only initial partial fallback')
        elif status=='fewer_than_two_eligible':need(len(ids)<2 and day!='2016-08-17','Domain fallback')
        else:need(len(ids)>=2 and day!='2016-08-17','Feature source requires original eligible comparison')
        if e['formation_start'] is not None:
            need(t>=L+42+199 and e['formation_start']==axis[t-42-L] and e['formation_end']==axis[t-42] and
                 e['historical_state_first']==axis[t-42-L] and e['historical_state_last']==axis[t-43],'210/252 stock and previous-day state clock')
        else:
            need(status in ('initial_partial','fewer_than_two_eligible','insufficient_calendar') and all(e[k] is None for k in ('formation_end','historical_state_first','historical_state_last')),'No invented source window')
        market_ok=e['query_state_up'] is not None
        if market_ok:
            need(type(e['query_state_up']) is bool and finite(e['query_market_close']) and finite(e['query_market_mean']) and e['query_market_close']>0 and e['query_market_mean']>0,'Saved market state source proof')
            need(e['query_state_up']==(e['query_market_close']>e['query_market_mean']),'Saved strict market comparison')
            need(type(e['historical_up_days']) is int and 0<=e['historical_up_days']<=L,'Historical state count')
            want=e['historical_up_days'] if e['query_state_up'] else L-e['historical_up_days']
            need(e['matching_state_days']==want,'Same current-state matches count')
            c['market_state_evaluations']=L+1;c['market_price_observations']=200*(L+1)
        else:
            need(all(e[k] is None for k in ('query_market_close','query_market_mean','historical_up_days','matching_state_days')),'No incomplete invented market values')
        for cell in cells:
            need(type(cell['source_checked']) is bool and type(cell['source_available']) is bool,'Source flags exact boolean')
            need(not cell['source_available'] or cell['source_checked'],'Available source was checked')
            if cell['source_checked']:c['checked_stock_cells']+=1
            if cell['source_available']:
                c['source_valid_stock_cells']+=1;c['stock_price_observations']+=L+1
                need(cell['reason']=='source_valid' and all(finite(cell[k]) and cell[k]>0 for k in ('price_start','price_end')),'Positive saved source endpoint evidence')
            else:need(cell['price_start'] is None and cell['price_end'] is None,'No filled missing stock endpoint')
            if not cell['source_checked']:need(cell['reason']=='not_checked','Unexamined source reason')
            if ready:
                need(cell['source_available'] and all(finite(cell[k]) for k in ('unconditional_mean','conditional_mean','candidate_raw','control_raw')),'Common source before finite raw drift')
                need(cell['control_raw']==cell['unconditional_mean'],'Same-window control uses recorded unconditional mean')
                if e['matching_state_days']==L or cell['conditional_mean']==cell['unconditional_mean']:
                    need(cell['candidate_raw']==cell['control_raw'],'Exact identity branch preserved')
                need(all(finite(cell[k]) and -.5<=cell[k]<=.5 for k in ('candidate_effect','control_effect')),'Bounded saved effects')
            else:
                need(all(cell[k] is None for k in ('unconditional_mean','conditional_mean','candidate_raw','control_raw')) and cell['candidate_effect']==cell['control_effect']==0.,'Fallback has no drift and zero new effect')
        need(c['source_valid_stock_cells']<=c['checked_stock_cells']<=len(ids),'Source check counts')
        if market_ok:need(c['checked_stock_cells']==len(ids),'Market-valid event checks all original eligible stocks')
        else:need(c['checked_stock_cells']==0,'Market-invalid event reads no stock source')
        if ready:
            need(market_ok and e['matching_state_days']>=42 and c['source_valid_stock_cells']==len(ids),'Ready means common complete sources and42matches')
            c.update(stock_log_returns=L*len(ids),drift_estimates=2*len(ids),ranked_observations=2*len(ids));supported.append(i)
            for arm,key in zip(ARMS[1:],('candidate','control')):
                need(len(orders[arm][i]['symbols'])==min(3,len(ids)) and set(orders[arm][i]['symbols'])<=set(names[j] for j in ids),'Full eligible order domain')
                if len({x[key+'_raw'] for x in cells})==1:
                    need(all(x[key+'_effect']==0. for x in cells),'Constant supported drift has zero effect, not fallback');constant[arm].append(i)
        else:
            for arm in ARMS[1:]:need(orders[arm][i]==orders[ARMS[0]][i],'Both arms exact whole-month incumbent fallback')
            if status=='insufficient_matching_state':need(market_ok and e['matching_state_days']<42 and c['source_valid_stock_cells']==len(ids),'Matching-state fallback only after complete source')
            if status=='invalid_stock_source':need(market_ok and c['source_valid_stock_cells']<len(ids),'Invalid source evidence')
        need(e['counts']==c,'Recounted event counters')
        for key in total:total[key]+=c[key]
        yr=by_year.setdefault(day[:4],dict(year=int(day[:4]),months=0,supported=0,reason_counts={},calm=0,noncalm=0))
        yr['months']+=1;yr['supported']+=int(ready);yr['calm' if e['calm'] else 'noncalm']+=1;yr['reason_counts'][status]=yr['reason_counts'].get(status,0)+1
        sources.append(dict(index=i,date=day,status=status,eligible=len(ids),source_valid=c['source_valid_stock_cells'],calm=e['calm'],matching_state_days=e['matching_state_days']))
    need(total==counts and audit['reason_counts']==dict(sorted(Counter(e['status'] for e in events).items())),'Recount all16 totals/reasons')
    need(counts['supported_months']+counts['fallback_months']==122,'All months classified')
    model_check(prepared['model'],counts,122);selection_check(prepared['selection'],122)
    for key in ('new_model_fits','predictive_model_fits','new_ridge_fits','new_tree_fits'):need(type(audit[key]) is int and audit[key]==0,'No new predictive models')
    for item in (prepared,completion,closure):need(item['original_array_count']==27 and item['model_counts']==counts,'27 original arrays/main counters')
    check_prepared_prefixes(prepared,completion,meta,events)
    return dict(counts=counts,status_counts=audit['reason_counts'],by_year=list(by_year.values()),supported_query_indices=supported,
        constant_supported_query_indices=constant,source_status_by_query=sources,
        supported_by_period={key:sum(lo<=i<hi for i in supported) for key,(lo,hi) in BOUNDS.items()},
        main_descriptive_drift_estimates=counts['drift_estimates'],main_ranked_observations=counts['ranked_observations'],
        main_repeated_market_price_observations=counts['market_price_observations'],new_predictive_models=0,
        independent_descriptive_estimates=prepared['model']['counts']['drift_estimates'],
        prefix_descriptive_estimates=sum(p['counts']['drift_estimates'] for p in prepared['model_prefix_evidence']),
        statistics_recomputed=False,npz_arrays_decoded=False,source_prefixes=list(FEATURE_PREFIXES),
        limitation='Saved raw drift values and cross-sectional ranks are bound to independent feature evidence, not recomputed. All original eligible membership and cached calm equality rely on independent preparation/source receipts without decoding cached arrays.')


def check_prepared_prefixes(prepared,completion,meta,events):
    need(prepared['prefixes']==123 and prepared['model_prefixes']==list(FEATURE_PREFIXES),'All123 order prefixes and five physical source prefixes')
    prefix=prepared['model_prefix_evidence'];need([p['prefix'] for p in prefix]==list(FEATURE_PREFIXES),'Five physical prefixes')
    for p in prefix:
        n=p['prefix'];counts={k:sum(e['counts'][k] for e in events[:n]) for k in COUNT_FIELDS}
        model_check(p['model'],counts,n);selection_check(p['selection'],n)
        need(p['counts']==counts and p['query_rows']==p['metadata_rows']==n and p['original_array_count']==27,'Physical prefix count/metadata evidence')
        need(type(p['price_rows']) is int and p['price_rows']>=0,'Physical prefix source rows')
        if n:
            need(p['query_date']==p['latest_price_date']==meta['dates'][n-1] and p['price_rows']>0,'Physical source ends exactly at current query')
        else:need(p['query_date'] is None and p['latest_price_date'] is None and p['price_rows']==0,'Empty source prefix')
        calm=p['calm'];need(calm['passed'] is True and calm['official_calendar'] is True and calm['exact_cached_calm'] is True and calm['query_rows']==n and len(calm['records'])==n,'Physical calm proof coverage')
        for i,row in enumerate(calm['records']):
            need(row['date']==row['price_end']==meta['dates'][i] and row['price_start']<row['price_end'] and row['sample_returns']==63 and type(row['calm']) is bool and row['calm']==events[i]['calm'],'Physical calm identity')
    for k in ('model_prefix_sources_truncated','model_prefix_metadata_truncated','model_prefix_original_arrays_truncated',
              'model_prefix_independently_verified','original_orders_exact','original_arrays_unchanged',
              'original_core_metadata_and_support_preserved','single_original_episode','no_new_score_budget',
              'full_eligible_replacement','constant_effect_not_fallback','calm_cache_independently_verified','selection_uses_zero_cost_paths_only'):
        need(prepared[k] is True,'Preparation invariant '+k)
    need(completion['new_returns_computed'] is False,'No preparation replay')
    for k in ('new_model_fits','predictive_model_fits','new_ridge_fits','new_tree_fits'):need(type(completion[k]) is int and completion[k]==0,'No preparation predictive fits')


def price_source_evidence(evidence,plan,context,sr,parity,meta,calendar,model_audit,prepared):
    need(registered_digest(plan,SOURCE_REVIEW)==SOURCE_REVIEW_SHA and sr['context_sha256']==CONTEXT_SHA and
        sr['proposal_sha256']==PROPOSAL_SHA and sr['clarifications_sha256']==CLARIFICATIONS_SHA,'Source final prospective pins')
    for file,h in ((PROPOSAL,PROPOSAL_SHA),(CLARIFICATIONS,CLARIFICATIONS_SHA)):
        evidence.load(file,h);need(registered_digest(plan,file)==h,'Prospective explicit source '+str(file))
    for k in ('passed','source_only','metadata_only','source_bytes_reverified','raw_source_map_matches_original','metadata_axes_verified','calendar_clock_verified','core_calm_source_definition_reviewed','single_episode_budget_source_reviewed','price_unit_source_fingerprints_bound'):need(sr[k] is True,'Source metadata proof '+k)
    for k in ('source_npz_decoded','npz_decoded','price_rows_read','source_models_imported','source_loader_executed','actual_support_computed','actual_features_computed','actual_labels_computed','actual_forecasts_computed','actual_returns_computed','actual_orders_computed','actual_new_parameters_computed','observers_modified','central_modified','frozen_files_modified','strict_pit'):need(sr[k] is False,'Source scope '+k)
    need(sr['actual_model_fits']==0 and sr['raw_source_hashes']==plan['raw_source_hashes'] and len(sr['raw_source_hashes'])==64,'Original64 sources with no fits')
    need(all(sr['evidence_sha256'].get(k)==v for k,v in context['evidence_sha256'].items()),'Context evidence subset')
    binding=plan['price_source_binding']
    need(path(binding['path'])==SOURCE_REVIEW and binding['sha256']==SOURCE_REVIEW_SHA and
        binding['no_new_training_labels'] is True and binding['no_new_historical_forecasts'] is True and binding['strict_pit'] is False,'No predictive target or history source needed for drift')
    for field in ('input_sha256','source_sha256','evidence_sha256','direct_feature_source_sha256','reviewed_code_sha256'):
        for filename,digest in sr[field].items():evidence.pin(filename,digest)
    for filename,digest in binding['input_sha256'].items():evidence.pin(filename,digest)
    need(parity['raw_source_hashes']==plan['raw_source_hashes'] and parity['price_source_binding']==binding,'Prepared original panel provenance')
    calm=parity['calm_parity'];need(calm['passed'] is True and calm['official_calendar'] is True and calm['exact_cached_calm'] is True and calm['query_rows']==122 and len(calm['records'])==122,'Cached calm exact parity')
    axis=calendar['sessions'];positions={d:i for i,d in enumerate(axis)}
    for i,row in enumerate(calm['records']):
        d=meta['dates'][i];t=positions[d]
        need(row['date']==row['price_end']==d and row['price_start']==axis[t-63] and row['sample_returns']==63,'Calm source official64 prices')
        need(finite(row['annualized_sample_volatility']) and row['annualized_sample_volatility']>=0 and type(row['calm']) is bool and row['calm']==(row['annualized_sample_volatility']<.10) and row['calm']==model_audit['event_history'][i]['calm'],'Saved calm threshold and feature branch')
    for prefix in prepared['model_prefix_evidence']:
        need(prefix['calm']['records']==calm['records'][:prefix['prefix']],'Every physical calm proof equals saved full prefix')
    for k in ('original_metadata_identity_verified','original_arrays_unchanged','original_orders_exact'):need(parity[k] is True,'Frozen source scope '+k)
    for k in ('new_training_labels','new_historical_forecasts','strict_pit'):need(parity[k] is False,'No new predictive source '+k)
    need(parity['calendar_sha256']==registered_digest(plan,CALENDAR),'Exact official calendar binding')
    fp=parity['original_array_fingerprints'];need(isinstance(fp,dict) and len(fp)==27,'All27 original byte fingerprints')
    for key,value in fp.items():
        need(set(value)=={'dtype','shape','sha256'} and isinstance(value['dtype'],str) and isinstance(value['shape'],list) and value['shape'][0]==122,'Original fingerprint metadata '+key)
        need(type(value['sha256']) is str and len(value['sha256'])==64 and all(c in '0123456789abcdef' for c in value['sha256']),'Original fingerprint digest '+key)
    for key in ('adjusted_levels_constant_scale','adjusted_opening_parity','missing_price_masks_exact','positive_price_masks_exact'):
        need(parity['execution_prices'][key] is True,'Execution unit parity '+key)
    return dict(raw_source_count=64,original_arrays=27,metadata_review_sha256=SOURCE_REVIEW_SHA,
        source_clock='Stock returns end t-42; prior-day SPY200 historical states, current t SPY200 query state; original current63 calm preserved.',
        price_units=sr['price_units'],calm_parity=calm,source_arrays_decoded=False,
        inherited_extra_forecast_sources_role='Immutable lineage only, not feature inputs',new_training_labels=False,new_forecasts=False,
        actual_feature_support_checked_by_registered_preparation=True,metadata_review_itself_computed_support=False,strict_pit=False)



def main(expected,test_count,expected_plan_sha,expected_code_review_sha):
    need(type(test_count) is int and test_count>0,'Explicit positive formal synthetic test count required')
    for value in (expected,expected_plan_sha,expected_code_review_sha):
        need(type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value),'Explicit terminal/PLAN/code-review SHA required')
    ACCOUNTING_RESIDUALS.clear()
    need(not (OUT/'RESEARCH_REVIEW.json').exists(),'Saved review is write-once')
    evidence=Evidence();closure=evidence.load(OUT/'COMPLETION.json',expected)
    need(closure['plan_sha256']==expected_plan_sha,'Explicit registered PLAN differs')
    plan=evidence.load(OUT/'PLAN.json',expected_plan_sha);plan_sha=expected_plan_sha
    need(closure.get('new_trials')==3 and closure.get('original_array_count')==27,'Exact three new trials and27 original arrays')
    need(closure.get('status')=='COMPLETED_INDEPENDENTLY_VERIFIED' and closure['plan_sha256']==plan_sha and closure['protected_files']==21444 and closure['observers_modified'] is False,'Terminal closure required')
    need(isinstance(plan.get('input_sha256'),dict) and plan['input_sha256'],'Registered input closure missing')
    for filename,digest in plan['input_sha256'].items():evidence.pin(filename,digest)
    context=evidence.load(CONTEXT,CONTEXT_SHA)
    need(registered_digest(plan,CONTEXT)==CONTEXT_SHA==plan['contract']['prospective_context_sha256'],'Context not registered exactly')
    need(plan['contract']['context']==context and context['arms']==list(ARMS) and plan['contract']['replace_single_episode_hook'] is True and plan['contract']['no_new_score_budget'] is True and plan['contract']['candidate']==ARMS[1] and plan['contract']['control_nonpromotable']==ARMS[2],'Contract/source theory differs')
    need(plan['contract']['selection_paths']==['same_close_zero','next_open_zero'] and plan['contract']['next_open_cost_diagnostic_only'] is True,'No fee-based selection guard in231')
    need(plan['contract']['full_eligible_reranking'] is True and plan['contract']['original_core_and_support_preserved'] is True,'Fixed231 architecture')
    for key in ('actual_new_feature_evaluations_before_registration','actual_new_model_fits_before_registration','actual_new_parameters_before_registration','actual_new_orders_before_registration','actual_new_returns_before_registration'):need(plan[key]==0,'No preregistration actual calculation')
    need(plan['protected_files']==21444 and len(evidence.load(PROTECTED,registered_digest(plan,PROTECTED)))==21444,'Registered protection manifest')
    before=evidence.load(OUT/'STATE_BEFORE.json',plan['prior_central_sha256'])
    need(len(before['completed'])==226 and sum(r['trials'] for r in before['completed'])==1195 and not before['registered_queue'],'Pre-registration state differs')
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
    need(registered_digest(plan,CODE_REVIEW)==expected_code_review_sha,'Explicit final code review differs')
    review=evidence.load(CODE_REVIEW,expected_code_review_sha)
    wanted_codes={str((ROOT/'work'/name).relative_to(ROOT)):registered_digest(plan,ROOT/'work'/name) for name in NUMERICAL_FILES}
    need(review['passed'] is True and review['unresolved_blockers']==[] and review['context_sha256']==CONTEXT_SHA and review['code_sha256']==wanted_codes,'Full six-file code review differs')
    for key in ('actual_features_computed','actual_model_fits_computed','actual_portfolio_returns_computed','production_files_modified','strategy_registered_by_reviewer'):
        need(review.get(key) is False,'Code-review scope differs: '+key)
    for filename,digest in review['input_sha256'].items():evidence.pin(filename,digest)
    need(plan['numerical_code_review']['sha256']==sha(CODE_REVIEW) and path(plan['numerical_code_review']['path'])==CODE_REVIEW,'Registered numerical review binding')
    frozen=evidence.load(OUT/'selected/FROZEN.json');training=evidence.load(OUT/'selected/TRAIN_RESULTS.json')
    need(frozen['plan_sha256']==plan_sha and frozen['training_sha256']==sha(OUT/'selected/TRAIN_RESULTS.json'),'Frozen first88 source')
    pc=evidence.phase(OUT/'prepared',frozen['prepared_sha256'],plan_sha,PREP)
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
    prepared=evidence.load(OUT/'prepared/INDEPENDENT_PREPARE.json');audit=evidence.load(OUT/'prepared/MODEL_AUDIT.json')
    calendar=evidence.load(CALENDAR,registered_digest(plan,CALENDAR))
    source=evidence.load(OUT/'prepared/SOURCE_PARITY.json')
    coverage=model_coverage(audit,prepared,pc,closure,meta,calendar,orders)
    need(pc['model_fit_accounting']==closure['model_fit_accounting']==plan['contract']['model_fit_accounting'],'Descriptive accounting description')
    for name,keys in (('FEATURES.npz',ARRAY_KEYS),('SCORES.npz',ARMS)):
        with zipfile.ZipFile(OUT/'prepared'/name) as archive:
            need(set(archive.namelist())=={k+'.npy' for k in keys} and len(archive.namelist())==len(keys),'NPZ member names without decoding: '+name)
    sr=evidence.load(SOURCE_REVIEW,SOURCE_REVIEW_SHA)
    source_evidence=price_source_evidence(evidence,plan,context,sr,source,meta,calendar,audit,prepared)
    results=evidence.load(OUT/'results/RESULTS.json');independent=evidence.load(OUT/'results/INDEPENDENT_OUTCOME.json')
    need(results['selection']==frozen['selection'] and results['strict_pit'] is False and results['reused_sample'] is True and results['observer_promotion'] is False,'Outcome interpretation or selection differs')
    for key in ('market_conditioned_drift','current_market_state','full_eligible_single_effect_replacement','zero_cost_selection','next_open_cost_diagnostic_only','same_support_both_arms'):need(results[key] is True,'Outcome mechanism differs '+key)
    need(independent['passed'] is True and independent['independent_accounts'] is True and independent['independent_of_production_model_and_engine'] is True and independent['checked_paths']==9 and independent['checked_slices']==45 and independent['terminal_liquidation'] is False,'Independent execution scope')
    need(independent['accounting_source_sha256']==registered_digest(plan,ROOT/'work/ptr169_results_independent.py'),'Independent accounting source')
    need(set(results['arms'])==set(independent['paths'])==set(training)==set(ARMS),'Outcome/training arms')
    metrics={};fee_evidence={};training_checks=0;zero_streams=0
    for arm in ARMS:
        need(set(results['arms'][arm])==set(independent['paths'][arm])==set(PATHS),'Three execution paths required')
        need(set(training[arm])=={'same_close_zero','next_open_zero'},'Training must use only the two zero-cost execution paths')
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
    need(results['arms'][ARMS[0]]['same_close_zero']['metrics']['cagr']==0.6766617372041686,'Exact frozen Local67 full CAGR differs')
    changes={a+'_vs_'+b:changed(orders,b,a) for a,b in ((ARMS[1],ARMS[0]),(ARMS[2],ARMS[0]),(ARMS[1],ARMS[2]))}
    selection=selection_from_saved(training,changes);need(selection==frozen['selection'],'Frozen first88 guards differ')
    expected_best=before['best_training_selected_candidate'];pick=selection['primary'];picked=results['arms'][pick]
    if selection['admitted'] and picked['same_close_zero']['metrics']['cagr']>expected_best['cagr']:
        m=picked['same_close_zero']['metrics'];expected_best=dict(directory=OUT.name,spec=dict(arm=pick),cagr=m['cagr'],sharpe=m['sharpe'],max_drawdown=m['max_drawdown'],next_open_5bp_cagr=picked['next_open_cost']['metrics']['cagr'],selected_on_first88=True,observer_promotion=False,limitations=plan['contract']['limitation'])
    need(closure['best_training_selected_candidate']==expected_best and closure['target_reached']==(expected_best['cagr']>=.7),'Best/target update differs from frozen admission')
    central_bytes=CENTRAL.read_bytes();state=json.loads(central_bytes)
    need(len(state['completed'])==227 and sum(x['trials'] for x in state['completed'])==1198 and state['registered_queue']==[],'Central postclosure counters')
    need(state['completed'][-1]['directory']==OUT.name and state['completed'][-1]['completion']['sha256']==expected and path(state['completed'][-1]['completion']['path'])==OUT/'COMPLETION.json','Central closure reference')
    need(state['best_training_selected_candidate']==expected_best and state['latest_training_selected_candidate']['selection']==selection,'Central selected/best mismatch')
    evidence.pin(Path(__file__))
    report=dict(passed=True,created_utc=datetime.now(timezone.utc).isoformat(),reviewer_role='Source231 metadata reviewer; this saved-evidence reader imports no model, oracle or execution engine.',scope='Saved JSON/hash/identity/count/selection/accounting only. NPZ directory member names inspected without decoding arrays. Only saved market-state/window/count identities and selection/accounting receipts are rechecked; no raw drift, ranks, orders or returns recomputed.',completion_sha256=expected,plan_sha256=plan_sha,prepared_completion_sha256=sha(OUT/'prepared/COMPLETION.json'),result_completion_sha256=sha(OUT/'results/COMPLETION.json'),input_sha256=evidence.pins,
        formal_synthetic_tests=test_count,reviewed_numerical_files=6,model_flags_checked=list(MODEL_FLAGS),selection_flags_checked=list(SELECTION_FLAGS),model_coverage=coverage,selection=selection,changed_holdings=changes,metrics=metrics,fee_evidence=fee_evidence,incumbent_versus230=incumbent,price_source_evidence=source_evidence,independent_recorded_metric_residuals=ACCOUNTING_RESIDUALS,
        independent_metric_tolerance=dict(absolute=1e-7,relative=2e-11,scope='Only production-vs-independent accounting nine-statistic checks use this established bound. Across rounds all incumbent metrics including cumulative fees are EXACT; only cost-path individual HOLDINGS.fee <=1e-10. Zero-cost fees exact0.'),
        independent_paths=9,independent_slices=45,independent_training_groups=18,zero_cost_streams=6,order_prefixes=123,source_prefixes=list(FEATURE_PREFIXES),original_array_count=27,protected_files_verified_by_registered_host_and_closer=21444,protected_archive_reread_in_this_review=False,limitations=context['limitations'],strict_pit=False,reused_sample=True,observers_modified=False,actual_new_features_computed=False,actual_new_parameters_computed=False,actual_new_orders_computed=False,actual_portfolio_returns_computed=False,source_npz_decoded=False,best=expected_best,target_reached=closure['target_reached'],completed_entries=227,trials=1198,central_snapshot_sha256=hashlib.sha256(central_bytes).hexdigest())
    need(CENTRAL.read_bytes()==central_bytes and sha(OUT/'COMPLETION.json')==expected and sha(OUT/'PLAN.json')==expected_plan_sha,'Central or immutable closure changed during review')
    with (OUT/'RESEARCH_REVIEW.json').open('x') as stream:stream.write(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(passed=True,review_sha256=sha(OUT/'RESEARCH_REVIEW.json'),selection=selection,model_counts=coverage['counts'],same_close={a:metrics[a]['same_close_zero']['full122'] for a in ARMS}),ensure_ascii=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--completion-sha256',required=True);parser.add_argument('--tests',type=int,required=True)
    parser.add_argument('--plan-sha256',required=True);parser.add_argument('--code-review-sha256',required=True)
    args=parser.parse_args();main(args.completion_sha256,args.tests,args.plan_sha256,args.code_review_sha256)
