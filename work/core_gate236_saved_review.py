"""Read terminal236 saved evidence; no producer, oracle, engine or array decoding.
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
import zipfile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/monthly_core_gate236'
PRIOR=ROOT/'outputs/monthly_relative_gate235'
CENTRAL=ROOT/'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
CONTEXT=ROOT/'work/monthly_core_gate236_context.json'
CONTEXT_SHA='83c74a2666b2f46221832020c316d782d4a56d9ed7902744805e5f4c55c86c98'
TESTS=ROOT/'work/monthly_core_gate236_TESTS.json'
CODE_REVIEW=ROOT/'work/core_gate236_CODE_REVIEW.json'
LOCAL=ROOT/'outputs/monthly_local_linear70'
PROTECTED=ROOT/'work/trade_reconcile_20261015_close/NEXT_EXPECTED_PROTECTED.json'
RUNTIME={'/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/bin/python3.9':'6b24be8ba00c4abc8b3e7e2620bd710e2cf7a4312d00c147f8f05fbb6f3d297a'}
ARMS=('incumbent_local67','core_horizon_gate')
PATHS={'same_close_zero':('close',0.),'next_open_zero':('open',0.),'next_open_cost':('open',.0005)}
BOUNDS={'full122':(0,122),'first88':(0,88),'early53':(0,53),'recent35':(53,88),'later34':(88,122)}
PREP={'FEATURES.npz','FEATURE_AUDIT.json','ORDERS.json','META.json','SOURCE_PARITY.json','INDEPENDENT_PREPARE.json'}
RESULT={'RESULTS.json','INDEPENDENT_OUTCOME.json'}|{a+'_'+p+'_'+k+'.json' for a in ARMS for p in PATHS for k in ('DAILY','TRADES','HOLDINGS','DAILY_POSITIONS')}
STAT_KEYS={'start','end','years','final_equity','cagr','sharpe','max_drawdown','trading_sessions','total_return'}
FEATURE_PREFIXES=(0,1,53,88,122)
NUMERICAL_FILES=('monthly_core_gate236_host.py','test_monthly_core_gate236_host.py','monthly_core_gate236_kernel.py','test_monthly_core_gate236_kernel.py','monthly_core_gate236_independent.py','test_monthly_core_gate236_independent.py')
PERIOD_FEE_ATOL=1e-10
ACCOUNTING_RESIDUALS=[]
# Formal test count must match the explicit CLI and immutable registered receipt.
# Explicit PLAN and terminal digests are mandatory. Optional code-review digest defaults to the registered immutable pin.


CALENDAR=ROOT/'outputs/monthly_paired_target143/CALENDAR.json'

COUNT_FIELDS=('events', 'symbols_per_event', 'checked_cells', 'original_eligible_cells', 'calm_events', 'active_events', 'required_cells', 'warm_cells', 'warm_observations', 'source_price_observations', 'added_cells', 'rejected_cells', 'candidate_eligible_cells', 'new_model_fits', 'predictive_model_fits')
FEATURE_FLAGS=('passed', 'independent_scalar_rebuild', 'original_eligibility_exact', 'original_calm_state_preserved', 'initial_partial_unchanged', 'official_endpoint_offsets_checked', 'official_panel295_axis_checked', 'warm295_notna_semantics_checked', 'current_positive_gate_checked', 'positive_nonboolean_endpoints_checked', 'required_source_only_checked', 'missing_required_source_fails', 'original210_eligibility_parity_checked', 'core252_sign_checked', 'complete_replacement_checked', 'no_eligibility_rerank', 'no_model_or_score_changes', 'audit_checked', 'descriptive_counts_checked', 'no_predictive_fits_checked', 'no_future_or_outcome_inputs')
SELECTION_FLAGS=('passed', 'independent_original_pre_checked', 'original_pre_unchanged', 'original_model_rank_scope_preserved', 'original_support_checked', 'original_budget_checked', 'original_effects_unchanged', 'final_eligibility_only_checked', 'complete_replacement_checked', 'one_episode_per_arm', 'initial_partial_unchanged', 'exact_scores_and_choices_checked')
MODEL_FLAGS=FEATURE_FLAGS
HOST_COUNT_FIELDS=COUNT_FIELDS
ARRAY_KEYS=('original_eligible', 'candidate_eligible', 'required', 'warm', 'momentum210', 'momentum252')
COVERAGE=ROOT/'work/after235_core_horizon_coverage.json'
COVERAGE_SHA='db9164b68e7a8469c8774c5921b7a85ee1b96cff7c85345fae96cbbc89a583fc'


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
                residuals.append(dict(signal_date=b['signal_date'],execution_date=b['execution_date'],previous235=a['fee'],current236=b['fee'],difference=difference))
        for rows,metric in ((oh,old['metrics']),(nh,new['metrics'])):
            total=0.
            for row in rows:total+=row['fee']
            need(total==metric['fees_paid'],'Saved chronological fees exactly equal cumulative fee metric')
        report[name]=dict(all_metrics_and_slices_exact=True,total_fees_exact=True,all_fills_exact=True,daily_equity_exact=True,daily_positions_exact=True,holding_nonfee_fields_exact=True,period_fee_differences=residuals,max_absolute_period_fee_difference=max([abs(x['difference']) for x in residuals],default=0.))
    return dict(paths=report,period_fee_tolerance=PERIOD_FEE_ATOL,tolerance_scope='ONLY individual next_open_cost HOLDINGS.fee <=1e-10; all metrics/slices including cumulative fees, turnover, trades, DAILY, positions and holding nonfee fields EXACT; zero-cost fee fields EXACT0.',prospective_policy='Inherited frozen232 comparison scope: no accumulated-fee exception and no result-driven widening.')


def model_check(check,counts,n):
    need(len(FEATURE_FLAGS)==21 and all(check.get(k) is True for k in FEATURE_FLAGS),'All21 independent source/gate flags')
    need(check.get('counts')==counts and all(type(check.get(k)) is int and check[k]==v for k,v in counts.items()),'Exact15 independent feature counters')
    need(counts['events']==n and check.get('numerical_tolerance')==0.,'Feature coverage/tolerance')
    need(check.get('producer_modules_imported') is False and check.get('computes_returns') is False,'Independent feature scope')


def holdings_counts(orders,n):
    return {arm:{str(k):sum(len(row['symbols'])==k for row in orders[arm][:n]) for k in (0,2,3)} for arm in ARMS}


def selection_check(check,n,orders):
    need(len(SELECTION_FLAGS)==12 and all(check.get(k) is True for k in SELECTION_FLAGS),'All12 original-pre/single-episode flags')
    need(check['ordered_choices_checked']==2*n and check['checked_events']==n and check['numerical_tolerance']==0. and check['original_source_parity_tolerance']==1e-12,'Exact independent selection coverage')
    need(check['producer_modules_imported'] is False and check['computes_returns'] is False and check['full_scores_supplied'] is True and check['full_pre_scores_supplied'] is True,'Independent pre/full-score scope')
    need(check['holdings_counts']==holdings_counts(orders,n),'Saved independent3/2/cash counts')


def aggregate_counts(per_event,n):
    result={k:sum(row[k] for row in per_event[:n]) for k in COUNT_FIELDS if k not in ('events','symbols_per_event')}
    result.update(events=n,symbols_per_event=49)
    return result


def model_coverage(audit,prepared,completion,closure,meta,calendar,orders):
    """Validate saved source clocks, warm witnesses and set accounting; no raw replay."""
    need(len(COUNT_FIELDS)==15 and audit.get('schema')=='monthly_core_gate236_features_v1','Source/gate audit schema')
    for key,value in dict(warm_price_count=295,old_start_offset=252,new_start_offset=294,end_offset=42,old_return_count=210,new_return_count=252).items():
        need(type(audit.get(key)) is int and audit[key]==value,'Source formation descriptor '+key)
    for key in ('rankings_recomputed','source_missing_filled','portfolio_returns_calculated'):
        need(audit.get(key) is False,'Feature scope '+key)
    axis=calendar['sessions'];need(axis==sorted(set(axis)),'Unique official calendar')
    positions={day:i for i,day in enumerate(axis)};monthends={day[:7]:day for day in axis}
    events=audit['event_history'];need(len(events)==122,'122 source/gate records')
    event_keys={'query_index','date','symbols','session_position','calm10','initial_partial','active','branch',
        'source_positions','source_dates','original_eligible_indices','candidate_eligible_indices',
        'required_indices','warm_indices','added_indices','rejected_indices'}
    per_event=[];branches=Counter();unchanged=[];status=[];source_windows=[]
    for i,event in enumerate(events):
        names=meta['symbols'][i];day=meta['dates'][i];t=positions[day]
        need(set(event)==event_keys and event['query_index']==i and event['date']==day and event['symbols']==names and event['session_position']==t,'Exact saved event identity/schema')
        partial=i==0 and day=='2016-08-17'
        need(partial or monthends[day[:7]]==day,'Official monthend/initialpartial')
        calm=event['calm10'];active=event['active']
        need(type(calm) is bool and type(active) is bool and type(event['initial_partial']) is bool and event['initial_partial']==partial and active==(not calm and not partial),'Recorded branch state')
        branch='initial_partial' if partial else 'core_horizon_gate' if active else 'calm_bypass'
        need(event['branch']==branch,'Exact branch precedence');branches[branch]+=1
        groups={key:event[key] for key in ('original_eligible_indices','candidate_eligible_indices','required_indices','warm_indices','added_indices','rejected_indices')}
        for key,ids in groups.items():
            need(isinstance(ids,list) and all(type(j) is int and 0<=j<49 for j in ids) and ids==sorted(set(ids)),'Sorted unique candidate-index witness '+key)
        old=groups['original_eligible_indices'];new=groups['candidate_eligible_indices'];required=groups['required_indices']
        warm=groups['warm_indices'];added=groups['added_indices'];rejected=groups['rejected_indices']
        need(set(added)==set(new)-set(old) and set(rejected)==set(old)-set(new),'Full replacement additions/removals accounting')
        need(len(new)==len(old)+len(added)-len(rejected),'Full replacement size identity')
        need(required==(list(range(49)) if active else []) and set(warm)<=set(required),'Every original name required on active rows; warm witness inside required')
        if active:
            points=[t-294,t-252,t-42,t]
            need(t>=294 and event['source_positions']==points and event['source_dates']==[axis[j] for j in points],'Exact official four endpoint offsets/dates')
            official=axis[t-294:t+1]
            need(len(official)==295 and official[0]==event['source_dates'][0] and official[-1]==day,'295 official-session clock witness')
            need(set(old)<=set(warm) and set(new)<=set(warm),'Both original210 and replacement252 qualification require warm')
            source_windows.append(dict(index=i,date=day,official_start=official[0],official_end=official[-1],official_sessions=295,
                source_positions=points,source_dates=event['source_dates'],required_names=49,warm_names=len(warm)))
        else:
            need(event['source_positions']==[] and event['source_dates']==[] and warm==[],'Bypass has no numeric source/warm evidence')
            need(new==old and added==rejected==[],'Bypass eligibility unchanged')
        for arm,ids in ((ARMS[0],old),(ARMS[1],new)):
            chosen=orders[arm][i]['symbols'];expected_size=3 if len(ids)>=3 else 2 if len(ids)==2 else 0
            need(len(chosen)==expected_size and set(chosen)<={names[j] for j in ids},'Saved3/2/cash portfolio uses admitted names')
        if new==old:
            unchanged.append(i);need(orders[ARMS[1]][i]==orders[ARMS[0]][i],'Unchanged eligibility preserves full original order/scores')
        counts=dict.fromkeys(COUNT_FIELDS,0)
        counts.update(events=1,symbols_per_event=49,checked_cells=49,original_eligible_cells=len(old),
            calm_events=int(calm),active_events=int(active),required_cells=len(required),warm_cells=len(warm),
            warm_observations=295*len(required),source_price_observations=4*len(warm),added_cells=len(added),
            rejected_cells=len(rejected),candidate_eligible_cells=len(new))
        per_event.append(counts)
        status.append(dict(index=i,date=day,branch=branch,original_eligible=len(old),candidate_eligible=len(new),
            required=len(required),warm=len(warm),added=len(added),rejected=len(rejected),
            added_symbols=[names[j] for j in added],rejected_symbols=[names[j] for j in rejected],
            incumbent_holdings=len(orders[ARMS[0]][i]['symbols']),candidate_holdings=len(orders[ARMS[1]][i]['symbols'])))
    counts=aggregate_counts(per_event,122)
    need(all(type(audit.get(k)) is int and audit[k]==v for k,v in counts.items()),'All15 saved descriptor counters')
    model_check(prepared['feature'],counts,122);selection_check(prepared['selection'],122,orders)
    need(prepared['prefixes']==123 and prepared['physical_cache_meta_order_prefixes'] is True and prepared['feature_prefixes']==list(FEATURE_PREFIXES),'123physical cache/meta/order and5price/calendar prefixes')
    prefixes=prepared['physical_prefix_evidence'];need([x['prefix'] for x in prefixes]==list(FEATURE_PREFIXES),'Exact5physical source prefixes')
    expected_prefix_keys={'prefix','source_rows','source_last_session','calendar_rows','calendar_last_session','feature','selection','counts'}
    for item in prefixes:
        n=item['prefix'];c=aggregate_counts(per_event,n)
        need(set(item)==expected_prefix_keys and item['counts']==c and set(item['source_rows'])=={'adj'},'Physical prefix schema/counts')
        model_check(item['feature'],c,n);selection_check(item['selection'],n,orders)
        rows=item['source_rows']['adj'];need(type(rows) is int and (rows>0 if n else rows==0),'Physical price source rows')
        need(item['source_last_session']==(meta['dates'][n-1] if n else None),'Physical price cutoff at query')
        size=positions[meta['dates'][n-1]]+2 if n else 0
        need(type(item['calendar_rows']) is int and item['calendar_rows']==size and item['calendar_last_session']==(axis[size-1] if n else None),'Physical calendar ends at next-session monthend witness')
    need(prefixes[-1]['feature']==prepared['feature'] and prefixes[-1]['selection']==prepared['selection'],'Full physical prefix certificates match main preparation')
    for key in ('original_orders_exact','original_arrays_unchanged','original_pre_preserved','original_core_preserved','original_E_for_pre','candidate_E_only_at_episode','single_original_episode','no_added_score_budget','feature_prefix_sources_truncated','calendar_prefixes_truncated','selection_uses_zero_cost_paths_only'):
        need(prepared[key] is True,'Preparation invariant '+key)
    need(completion['new_returns_computed'] is False,'Preparation computes no portfolio returns')
    for record in (prepared,completion,closure):
        need(record['original_array_count']==27 and all(type(record[k]) is int and record[k]==v for k,v in counts.items()),'Original27/flat15 counter receipt')
        need(type(record['new_predictive_model_fits']) is int and record['new_predictive_model_fits']==0,'No new predictive fits')
    for record in (prepared,completion):need(record['model_counts']==counts,'Nested preparation counters')
    return dict(counts=counts,branch_counts=dict(branches),eligibility_status_by_query=status,unchanged_eligibility_indices=unchanged,
        eligibility_by_period={label:{key:sum(e[key] for e in per_event[lo:hi]) for key in ('original_eligible_cells','calm_events','active_events','required_cells','warm_cells','warm_observations','source_price_observations','added_cells','rejected_cells','candidate_eligible_cells')} for label,(lo,hi) in BOUNDS.items()},
        holdings_counts_by_period={label:holdings_counts({arm:rows[lo:hi] for arm,rows in orders.items()},hi-lo) for label,(lo,hi) in BOUNDS.items()},
        official_window_witnesses=source_windows,independent_selection_certificates=6,physical_source_prefixes=list(FEATURE_PREFIXES),cache_meta_order_prefixes=123,
        source_momentum_or_eligibility_recomputed=False,npz_arrays_decoded=False,
        limitation='Saved official clocks, index-set, branch, warm/add/remove and count consistency only. Actual panel295-axis equality, notna warmup and numeric endpoints are certified by the registered independent preparation; this reader does not reread those source cells. It does not claim every295 interior price is finitepositive. Momentum, masks, pre-scores, episode scores and orders were not rebuilt; added names retain unequal original model support.')


def price_source_evidence(evidence,plan,context,parity,meta,calendar):
    coverage=evidence.load(COVERAGE,COVERAGE_SHA)
    need(registered_digest(plan,COVERAGE)==COVERAGE_SHA==plan['contract']['coverage_sha256'],'Bounded eligibility coverage binding')
    need(coverage.get('source_only') is True,'Static source coverage')
    for key in ('actual_features_computed','actual_support_computed','actual_orders_computed','actual_returns_computed','npz_decoded','observer_files_modified'):
        need(coverage.get(key) is False,'Prospective coverage scope '+key)
    for mapping in (context['evidence_sha256'],coverage['source_sha256']):
        for filename,digest in mapping.items():evidence.pin(filename,digest)
    binding=plan['price_source_binding'];need(binding==parity['price_source_binding'],'Prepared source binding')
    need(binding['events']==122 and binding['symbols_per_event']==49 and binding['original_array_count']==27 and binding['calendar_sha256']==registered_digest(plan,CALENDAR) and binding['context_sha256']==CONTEXT_SHA and binding['coverage_sha256']==COVERAGE_SHA,'Bound source metadata')
    for key in ('source_only','metadata_only'):need(binding[key] is True,'Prospective metadata source scope '+key)
    for key in ('source_npz_decoded','actual_support_computed','actual_returns_computed','strict_pit'):need(binding[key] is False,'Source admission qualification '+key)
    for filename,digest in binding['input_sha256'].items():
        evidence.pin(filename,digest);need(registered_digest(plan,filename)==digest,'Source input absent from PLAN')
    local_plan=evidence.load(LOCAL/'PLAN.json',registered_digest(plan,LOCAL/'PLAN.json'))
    need(plan['raw_source_hashes']==parity['raw_source_hashes']==binding['raw_source_hashes']==local_plan['raw_source_hashes'] and len(plan['raw_source_hashes'])==64,'Inherited64raw sources')
    need(local_plan['calendar']==calendar['provenance'] and parity['calendar_sha256']==registered_digest(plan,CALENDAR),'Original calendar provenance')
    for key in ('original_arrays_unchanged','original_orders_exact','original_pre_preserved','original_E_for_pre','candidate_E_only_at_episode'):
        need(parity.get(key) is True,'Source/cache parity '+key)
    need(parity.get('strict_pit') is False,'Source qualification')
    fps=parity['original_array_fingerprints'];need(len(fps)==27,'27 original byte fingerprints')
    for key,value in fps.items():
        need(set(value)=={'dtype','shape','sha256'} and isinstance(value['dtype'],str) and value['shape'][0]==122 and type(value['sha256']) is str and len(value['sha256'])==64,'Array fingerprint metadata '+key)
    for key in ('adjusted_levels_constant_scale','adjusted_opening_parity','missing_price_masks_exact','positive_price_masks_exact'):
        need(parity['execution_prices'][key] is True,'Execution-price source parity '+key)
    return dict(original_arrays=27,raw_source_count=64,coverage_sha256=COVERAGE_SHA,source_admission_metadata_only=True,
        source_clock='Registered noncalm source window has exactly295 official sessions t-294..t on the original panel axis; four numeric endpoints t-294,t-252,t-42,t only for warm cells. Original calm10 and old E remain frozen cache inputs.',
        strict_pit=False,source_publication_clock_certified=False,source_arrays_decoded_by_this_reader=False,
        original_array_fingerprints_recorded=27,original210_eligibility_parity_checked_by_registered_independent=True,official_panel295_axis_checked_by_registered_independent=True,
        limitation='Source pins and saved execution-price parity do not certify strict point-in-time vintages or pristine holdout performance.')


def main(expected,test_count,expected_plan_sha,expected_code_review_sha=None):
    need(type(test_count) is int and test_count>0,'Explicit positive formal synthetic test count required')
    for value in (expected,expected_plan_sha)+((expected_code_review_sha,) if expected_code_review_sha is not None else ()):
        need(type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value),'Explicit terminal/PLAN/code-review SHA required')
    ACCOUNTING_RESIDUALS.clear()
    need(not (OUT/'RESEARCH_REVIEW.json').exists(),'Saved review is write-once')
    evidence=Evidence();closure=evidence.load(OUT/'COMPLETION.json',expected)
    need(closure['plan_sha256']==expected_plan_sha,'Explicit registered PLAN differs')
    plan=evidence.load(OUT/'PLAN.json',expected_plan_sha);plan_sha=expected_plan_sha
    need(closure.get('new_trials')==2 and closure.get('original_array_count')==27,'Exact two new trials and27 original arrays')
    need(closure.get('status')=='COMPLETED_INDEPENDENTLY_VERIFIED' and closure['plan_sha256']==plan_sha and closure['protected_files']==21444 and closure['observers_modified'] is False,'Terminal closure required')
    need(isinstance(plan.get('input_sha256'),dict) and plan['input_sha256'],'Registered input closure missing')
    for filename,digest in plan['input_sha256'].items():evidence.pin(filename,digest)
    context=evidence.load(CONTEXT,CONTEXT_SHA)
    need(registered_digest(plan,CONTEXT)==CONTEXT_SHA==plan['contract']['prospective_context_sha256'],'Context not registered exactly')
    need(plan['contract']['context']==context and context['arms']==list(ARMS) and plan['contract']['additional_score_budget']==0. and plan['contract']['candidate']==ARMS[1] and plan['contract']['control']==ARMS[0],'Contract/source theory differs')
    need(plan['contract']['feature_prefixes']==list(FEATURE_PREFIXES) and plan['contract']['all123_physical_cache_meta_order_prefixes'] is True,'Five physical sources/all123 cache-meta-order prefixes registered')
    for key in ('original_pre_unchanged','original_eligibility_for_models','eligibility_injected_only_at_episode','original_episode_preserved','final_scores_may_change'):need(plan['contract'][key] is True,'Fixed final-selector architecture '+key)
    need(plan['contract']['source_window']==dict(warm_price_count=295,old_start_offset=252,new_start_offset=294,end_offset=42,old_return_count=210,new_return_count=252,numeric_source_observations_per_warm_cell=4),'Fixed source window')
    need(plan['contract']['selection_paths']==['same_close_zero','next_open_zero'] and plan['contract']['next_open_cost_diagnostic_only'] is True,'No fee-based selection guard in236')
    need(plan['contract']['original_pool_preserved'] is True and plan['contract']['no_parameter_search'] is True and plan['contract']['no_observer_writes'] is True and plan['contract']['strict_pit'] is False,'Fixed236 architecture')
    for key in ('actual_new_feature_evaluations_before_registration','actual_new_model_fits_before_registration','actual_new_parameters_before_registration','actual_new_orders_before_registration','actual_new_returns_before_registration'):need(plan[key]==0,'No preregistration actual calculation')
    need(plan['protected_files']==21444 and len(evidence.load(PROTECTED,registered_digest(plan,PROTECTED)))==21444,'Registered protection manifest')
    before=evidence.load(OUT/'STATE_BEFORE.json',plan['prior_central_sha256'])
    need(len(before['completed'])==231 and sum(r['trials'] for r in before['completed'])==1209 and not before['registered_queue'],'Pre-registration state differs')
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
    registered_review=registered_digest(plan,CODE_REVIEW)
    if expected_code_review_sha is None:expected_code_review_sha=registered_review
    need(registered_review==expected_code_review_sha,'Explicit final code review differs')
    review=evidence.load(CODE_REVIEW,expected_code_review_sha)
    wanted_codes={str((ROOT/'work'/name).relative_to(ROOT)):registered_digest(plan,ROOT/'work'/name) for name in NUMERICAL_FILES}
    need(type(review.get('tests')) is int and review['tests']==test_count,'Reviewed formal test count differs')
    need(registered_digest(review,TESTS)==registered_digest(plan,TESTS)==sha(TESTS),'Code-review/formal TESTS digest binding differs')
    need(review['passed'] is True and review['unresolved_blockers']==[] and review['context_sha256']==CONTEXT_SHA and review['code_sha256']==wanted_codes,'Full six-file code review differs')
    for key in ('actual_features_computed','actual_model_fits_computed','actual_portfolio_returns_computed','production_files_modified','strategy_registered_by_reviewer'):
        need(review.get(key) is False,'Code-review scope differs: '+key)
    for filename,digest in review['input_sha256'].items():evidence.pin(filename,digest)
    need(plan['numerical_code_review']['sha256']==sha(CODE_REVIEW) and path(plan['numerical_code_review']['path'])==CODE_REVIEW,'Registered numerical review binding')
    frozen=evidence.load(OUT/'selected/FROZEN.json');training=evidence.load(OUT/'selected/TRAIN_RESULTS.json')
    need(frozen['plan_sha256']==plan_sha and frozen['training_sha256']==sha(OUT/'selected/TRAIN_RESULTS.json'),'Frozen first88 source')
    pc=evidence.phase(OUT/'prepared',frozen['prepared_sha256'],plan_sha,PREP)
    rc=evidence.phase(OUT/'results',closure['result_completion_sha256'],plan_sha,RESULT)
    need(rc['paths']==6 and rc['slices']==30 and rc['frozen_sha256']==sha(OUT/'selected/FROZEN.json'),'Outcome/frozen lineage')
    need(frozen['orders_sha256']==sha(OUT/'prepared/ORDERS.json'),'Frozen prepared orders')
    meta=evidence.load(OUT/'prepared/META.json');orders=evidence.load(OUT/'prepared/ORDERS.json')
    need(meta==evidence.load(LOCAL/'META.json',registered_digest(plan,LOCAL/'META.json')),'Original monthly metadata differs')
    need(len(meta['dates'])==len(meta['ends'])==len(meta['symbols'])==122 and all(len(set(s))==49 for s in meta['symbols']),'Original122x49 axes')
    need(meta['ends'][:-1]==meta['dates'][1:] and meta['dates'][0]=='2016-08-17' and meta['ends'][-1]=='2026-09-24','Original period/calendar')
    need(meta['ends'][52]==meta['dates'][53]=='2020-12-31' and meta['ends'][87]==meta['dates'][88]=='2023-11-30','Frozen split identities')
    need(set(orders)==set(ARMS) and all(len(v)==122 for v in orders.values()),'Two complete order streams')
    need(orders[ARMS[0]]==evidence.load(LOCAL/'local_linear_ORDERS.json',registered_digest(plan,LOCAL/'local_linear_ORDERS.json')),'Incumbent orders differ')
    for rows in orders.values():
        for i,row in enumerate(rows):
            need(row['date']==row['signal_date']==meta['dates'][i] and row['end']==meta['ends'][i],'Order timestamps')
            need(len(row['symbols']) in (0,2,3) and len(set(row['symbols']))==len(row['symbols']) and set(row['symbols'])<=set(meta['symbols'][i]),'Order targets outside original universe')
    prepared=evidence.load(OUT/'prepared/INDEPENDENT_PREPARE.json');audit=evidence.load(OUT/'prepared/FEATURE_AUDIT.json')
    calendar=evidence.load(CALENDAR,registered_digest(plan,CALENDAR))
    source=evidence.load(OUT/'prepared/SOURCE_PARITY.json')
    coverage=model_coverage(audit,prepared,pc,closure,meta,calendar,orders)
    need(prepared['model_fit_accounting']==pc['model_fit_accounting']==closure['model_fit_accounting']==plan['contract']['model_fit_accounting'],'Descriptive accounting description')
    with zipfile.ZipFile(OUT/'prepared/FEATURES.npz') as archive:
        need(set(archive.namelist())=={k+'.npy' for k in ARRAY_KEYS} and len(archive.namelist())==6,'Six NPZ member names without decoding')
    source_evidence=price_source_evidence(evidence,plan,context,source,meta,calendar)
    results=evidence.load(OUT/'results/RESULTS.json');independent=evidence.load(OUT/'results/INDEPENDENT_OUTCOME.json')
    need(results['selection']==frozen['selection'] and results['strict_pit'] is False and results['reused_sample'] is True and results['observer_promotion'] is False,'Outcome interpretation or selection differs')
    for key in ('full_state_horizon_qualification','zero_cost_selection','next_open_cost_diagnostic_only','original_pre_unchanged','candidate_eligibility_only_at_episode'):need(results[key] is True,'Outcome mechanism differs '+key)
    need(independent['passed'] is True and independent['independent_accounts'] is True and independent['independent_of_production_model_and_engine'] is True and independent['checked_paths']==6 and independent['checked_slices']==30 and independent['terminal_liquidation'] is False,'Independent execution scope')
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
    need(training_checks==12 and zero_streams==4,'Incomplete training/zero-cost checks')
    incumbent=compare_incumbent(evidence,prior_done,results,prior_results)
    comparisons={arm:{name:{period:dict(
        cagr_difference=metrics[arm][name][period]['cagr']-metrics[ARMS[0]][name][period]['cagr'],
        sharpe_difference=metrics[arm][name][period]['sharpe']-metrics[ARMS[0]][name][period]['sharpe'],
        max_drawdown_difference=metrics[arm][name][period]['max_drawdown']-metrics[ARMS[0]][name][period]['max_drawdown'],
        cagr_relation=('equal' if metrics[arm][name][period]['cagr']==metrics[ARMS[0]][name][period]['cagr'] else
            'higher' if metrics[arm][name][period]['cagr']>metrics[ARMS[0]][name][period]['cagr'] else 'lower'))
        for period in ('full122','first88','later34')} for name in PATHS} for arm in ARMS[1:]}
    changes={a+'_vs_'+b:changed(orders,b,a) for a,b in ((ARMS[1],ARMS[0]),)}
    selection=selection_from_saved(training,changes);need(selection==frozen['selection'],'Frozen first88 guards differ')
    expected_best=before['best_training_selected_candidate'];pick=selection['primary'];picked=results['arms'][pick]
    if selection['admitted'] and picked['same_close_zero']['metrics']['cagr']>expected_best['cagr']:
        m=picked['same_close_zero']['metrics'];expected_best=dict(directory=OUT.name,spec=dict(arm=pick),cagr=m['cagr'],sharpe=m['sharpe'],max_drawdown=m['max_drawdown'],next_open_5bp_cagr=picked['next_open_cost']['metrics']['cagr'],selected_on_first88=True,observer_promotion=False,limitations=plan['contract']['limitation'])
    need(closure['best_training_selected_candidate']==expected_best and closure['target_reached']==(expected_best['cagr']>=.7),'Best/target update differs from frozen admission')
    central_bytes=CENTRAL.read_bytes();state=json.loads(central_bytes)
    need(len(state['completed'])==232 and sum(x['trials'] for x in state['completed'])==1211 and state['registered_queue']==[],'Central postclosure counters')
    need(state['completed'][-1]['directory']==OUT.name and state['completed'][-1]['completion']['sha256']==expected and path(state['completed'][-1]['completion']['path'])==OUT/'COMPLETION.json','Central closure reference')
    need(state['best_training_selected_candidate']==expected_best and state['latest_training_selected_candidate']['selection']==selection,'Central selected/best mismatch')
    evidence.pin(Path(__file__))
    report=dict(passed=True,created_utc=datetime.now(timezone.utc).isoformat(),reviewer_role='Round236 kernel author performing a saved-evidence audit; mathematical independence is supplied by the separately authored registered oracle. This reader imports no producer, oracle or engine.',scope='Saved JSON/hash/identity/count/selection/accounting only. NPZ directory member names inspected without decoding arrays. Only saved official-window clocks, warm/add/remove set/branch/count consistency and selection/accounting receipts are rechecked; no source momentum, eligibility masks, pre-scores, episode scores, orders or returns recomputed.',completion_sha256=expected,plan_sha256=plan_sha,prepared_completion_sha256=sha(OUT/'prepared/COMPLETION.json'),result_completion_sha256=sha(OUT/'results/COMPLETION.json'),input_sha256=evidence.pins,
        formal_synthetic_tests=test_count,reviewed_numerical_files=6,model_flags_checked=list(MODEL_FLAGS),selection_flags_checked=list(SELECTION_FLAGS),model_coverage=coverage,selection=selection,changed_holdings=changes,metrics=metrics,candidate_vs_incumbent=comparisons,fee_evidence=fee_evidence,incumbent_versus235=incumbent,price_source_evidence=source_evidence,independent_recorded_metric_residuals=ACCOUNTING_RESIDUALS,
        independent_metric_tolerance=dict(absolute=1e-7,relative=2e-11,scope='Only production-vs-independent accounting nine-statistic checks use this established bound. Across rounds all incumbent metrics including cumulative fees are EXACT; only cost-path individual HOLDINGS.fee <=1e-10. Zero-cost fees exact0.'),
        independent_paths=6,independent_slices=30,independent_training_groups=12,zero_cost_streams=4,order_prefixes=123,source_prefixes=list(FEATURE_PREFIXES),original_array_count=27,protected_files_verified_by_registered_host_and_closer=21444,protected_archive_reread_in_this_review=False,limitations=context['limitations'],strict_pit=False,reused_sample=True,same_close_zero_idealized=True,observers_modified=False,saved_outcomes_read=True,new_source_features_computed_by_this_reader=False,new_parameters_computed_by_this_reader=False,new_orders_computed_by_this_reader=False,portfolio_returns_recomputed_by_this_reader=False,source_npz_decoded_by_this_reader=False,best=expected_best,target_reached=closure['target_reached'],completed_entries=232,trials=1211,central_snapshot_sha256=hashlib.sha256(central_bytes).hexdigest())
    need(CENTRAL.read_bytes()==central_bytes and sha(OUT/'COMPLETION.json')==expected and sha(OUT/'PLAN.json')==expected_plan_sha,'Central or immutable closure changed during review')
    with (OUT/'RESEARCH_REVIEW.json').open('x') as stream:stream.write(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(passed=True,review_sha256=sha(OUT/'RESEARCH_REVIEW.json'),selection=selection,model_counts=coverage['counts'],same_close={a:metrics[a]['same_close_zero']['full122'] for a in ARMS}),ensure_ascii=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--completion-sha256',required=True);parser.add_argument('--tests',type=int,required=True)
    parser.add_argument('--plan-sha256',required=True);parser.add_argument('--code-review-sha256')
    args=parser.parse_args();main(args.completion_sha256,args.tests,args.plan_sha256,args.code_review_sha256)
