"""Write-once saved-evidence reader for closed round207.

The author also implemented the207 kernel; this reader is not an independent
mathematical verifier. Feature independence comes from the separately authored
scalar oracle and its hash-bound preparation receipt.

Reads saved evidence/results only; never imports a strategy, loads raw prices,
rebuilds signals/orders, replays returns, changes thresholds or writes central.
Run review only after root confirms closure with its explicit completion SHA.
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import copy
import hashlib
import json
import math
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/monthly_down_volume207'
PRIOR=ROOT/'outputs/monthly_identity_transfer206'
CENTRAL=ROOT/'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
CALENDAR=ROOT/'outputs/monthly_paired_target143/CALENDAR.json'
ARMS=('incumbent_local67','down_volume_concentration','all_volume_control')
PATHS=('same_close_zero','next_open_zero','next_open_cost')
BOUNDS={'full122':(0,122),'first88':(0,88),'early53':(0,53),'recent35':(53,88),'later34':(88,122)}
METRICS=('start','end','cagr','sharpe','max_drawdown')
COUNTS=('valid_price_observations','valid_volume_observations','down_comparisons',
        'dispersion_estimates','raw_score_observations','ranked_score_observations')
PREPARED_FILES={'FEATURES.npz','TRANSFORMS_DELTAS.npz','FEATURE_AUDIT.json','ORDERS.json','META.json','SOURCE_PARITY.json','INDEPENDENT_PREPARE.json'}
RESULT_FILES={'RESULTS.json','INDEPENDENT_OUTCOME.json'}|{
    a+'_'+p+'_'+k+'.json' for a in ARMS for p in PATHS for k in ('DAILY','TRADES','HOLDINGS','DAILY_POSITIONS')}
CONTEXT='work/monthly_down_volume207_context.json'
CONTEXT_SHA='223ac747a847b9d908cea27bf7596f544c783d0247ce56c255b24029e0712516'
TESTS='work/monthly_down_volume207_TESTS.json'
TESTS_SHA='2adf36c2795659163c416b63ea224f0ba68813c42d84a731b181a15599cf1d97'
CODE_REVIEW='work/downvolume207_CODE_REVIEW.json'
CODE_REVIEW_SHA='b39b69af99c8cd1310d61a34344ada709332e66aa55d3f6ff531a24a28c1f91d'
SOURCE='work/downvolume207_SOURCE_REVIEW.json'
SOURCE_SHA='77218c4d8e4d2b0d9d64a17adb6370ced64b833a90493c02020e74bbaa1e9d04'
NUMERICAL_FILES={
 'work/monthly_down_volume207_host.py','work/test_monthly_down_volume207_host.py',
 'work/monthly_down_volume207_kernel.py','work/test_monthly_down_volume207_kernel.py',
 'work/monthly_down_volume207_independent.py','work/test_monthly_down_volume207_independent.py'}
FROZEN_COMPONENTS={
 'work/monthly_down_volume207_host.py':'6c0e441f1485b2590285dd03dd32d62085b0e3b6b005ae786ad082e5535637b7',
 'work/test_monthly_down_volume207_host.py':'61c9dd40d060dadfc61dc1b384dbd75935e111b691c7e429daced87f706045b4',
 'work/monthly_down_volume207_kernel.py':'bdd78d2222afe702923fc40f26acd56139dee4ecf9e43a377477a9d63b288fe3',
 'work/test_monthly_down_volume207_kernel.py':'5c36d56d68f5a8eecbad13c0b262e646248adb2225f19e8c688b2a770bb32151',
 'work/monthly_down_volume207_independent.py':'558d7a8fa4dc032a14917cdf9c6d8c91050c1574ce96799f13d34092feda31af',
 'work/test_monthly_down_volume207_independent.py':'e724db8e0362856b5c8ae1b2ebcdfaf29b30a20ddd06bc25464f44d72ebbcdb5'}
FEATURE_FLAGS=('passed','independent_scalar_rebuild','strict_down_price_comparisons_checked',
 'current63_alignment_checked','population_cv_squared_checked','matched_all_volume_dispersion_checked',
 'common_minimum_down_support_checked','exact_constant_branches_checked','full64_price_support_checked',
 'full63_volume_support_checked','official_session_windows_checked','monthend_queries_checked',
 'exact_common_availability','audit_checked','no_future_or_outcome_inputs','centered_high_tie_ranks_checked')
DELTA_FLAGS=('passed','full_candidate_universe','initial_delta_zero','common_available_only',
 'no_eligibility_or_model_filter','centered_cross_sectional_ranking','direct_additive_pre_score','fixed_linear_mapping_checked')
PERIOD_FEE_ATOL=1e-10
ENGINE='outputs/monthly_factor_search_v2/validate.py'
ENGINE_SHA='b5c6bbbfa5f4db0821f6b0dacb33a43c974cceb8a150578803eef73132cd30e4'
REGISTERED_RUNTIME=Path('/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/bin/python3.9')
REGISTERED_RUNTIME_SHA='6b24be8ba00c4abc8b3e7e2620bd710e2cf7a4312d00c147f8f05fbb6f3d297a'


def require(ok,message):
    if not ok: raise ValueError(message)


def local_path(value):
    p=Path(value);p=(p if p.is_absolute() else ROOT/p).resolve()
    require(p.is_relative_to(ROOT),'Evidence must be local')
    return p


def input_path(value,expected=None):
    p=Path(value);p=(p if p.is_absolute() else ROOT/p).resolve()
    if p.is_relative_to(ROOT):return p
    require(p==REGISTERED_RUNTIME and expected==REGISTERED_RUNTIME_SHA,
        'Only the exact registered external runtime path and SHA are permitted')
    return p


class Evidence:
    def __init__(self): self.pins={}
    def verify(self,value,expected=None):
        p=input_path(value,expected);h=hashlib.sha256()
        with p.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
        digest=h.hexdigest();key=str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
        require(expected is None or digest==expected,'Changed evidence: '+key)
        require(key not in self.pins or self.pins[key]==digest,'Evidence changed during review: '+key)
        self.pins[key]=digest
        return digest
    def load(self,value,expected=None):
        self.verify(value,expected)
        return json.loads(local_path(value).read_text())
    def manifest(self,folder,expected,plan_sha,names):
        r=self.load(folder/'COMPLETION.json',expected)
        require(r['status']=='COMPLETE' and r['plan_sha256']==plan_sha,'Incomplete phase: '+str(folder))
        require(set(r['artifacts_sha256'])==names,'Phase artifact set differs')
        for p,h in r['artifacts_sha256'].items():
            require(Path(p).name==p,'Nonlocal phase artifact')
            self.verify(folder/p,h)
        return r


def metric_evidence(actual,expected,label):
    """Compare already recorded independent stats under frozen accounting tolerance."""
    for k,v in actual.items():
        require(k in expected,label+'/'+k+' missing')
        other=expected[k]
        if isinstance(v,str) or v is None or k=='trading_sessions':
            require(v==other,label+'/'+k)
        else:
            require(type(v) in (int,float) and type(other) in (int,float) and math.isfinite(v)
                and math.isfinite(other) and abs(v-other)<=1e-7+2e-11*abs(other),label+'/'+k)


def guard(candidate,reference,gain):
    return (candidate['training']['cagr']>=reference['training']['cagr']+gain-1e-12
        and candidate['training']['sharpe']>=reference['training']['sharpe']-.05
        and candidate['training']['max_drawdown']>=reference['training']['max_drawdown']-.03
        and all(candidate[p]['cagr']>=reference[p]['cagr']-.025 for p in ('early','recent')))


def selection_from_saved(metrics,orders):
    changed=[i for i in range(88) if set(orders[ARMS[1]][i]['symbols'])!=set(orders[ARMS[0]][i]['symbols'])]
    control=[i for i in range(88) if set(orders[ARMS[1]][i]['symbols'])!=set(orders[ARMS[2]][i]['symbols'])]
    guards=dict(close_vs_incumbent=guard(metrics[ARMS[1]]['same_close_zero'],metrics[ARMS[0]]['same_close_zero'],.005),
        open_vs_incumbent=guard(metrics[ARMS[1]]['next_open_cost'],metrics[ARMS[0]]['next_open_cost'],0.),
        close_vs_all_volume_control=guard(metrics[ARMS[1]]['same_close_zero'],metrics[ARMS[2]]['same_close_zero'],0.),
        open_vs_all_volume_control=guard(metrics[ARMS[1]]['next_open_cost'],metrics[ARMS[2]]['next_open_cost'],0.),
        changed_support=len(changed)>=3 and any(i<53 for i in changed) and any(i>=53 for i in changed),
        changed_support_vs_control=len(control)>=3 and any(i<53 for i in control) and any(i>=53 for i in control))
    admitted=all(guards.values())
    return dict(primary=ARMS[1] if admitted else ARMS[0],admitted=admitted,guards=guards,
        changed_months=changed,changed_months_vs_control=control,first88_only=True,later34_used=False)


def changes(orders,reference,candidate):
    rows=[]
    for i,(old,new) in enumerate(zip(orders[reference],orders[candidate])):
        require(old['date']==new['date'] and old['end']==new['end'],'Order axes differ')
        if set(old['symbols'])!=set(new['symbols']):
            rows.append(dict(index=i,date=old['date'],end=old['end'],reference=old['symbols'],candidate=new['symbols'],
                removed=sorted(set(old['symbols'])-set(new['symbols'])),added=sorted(set(new['symbols'])-set(old['symbols']))))
    return dict(counts={p:sum(lo<=r['index']<hi for r in rows) for p,(lo,hi) in BOUNDS.items()},months=rows)


def compare_holdings(old,new,path):
    require(len(old)==len(new)==122,'Incumbent holding period count differs')
    residuals=[]
    for a,b in zip(old,new):
        require(set(a)==set(b) and {k:v for k,v in a.items() if k!='fee'}==
            {k:v for k,v in b.items() if k!='fee'},'Nonfee incumbent holding field changed')
        require(type(a['fee']) in (int,float) and type(b['fee']) in (int,float)
            and math.isfinite(a['fee']) and math.isfinite(b['fee']),'Nonfinite period fee')
        if a['fee']!=b['fee']:
            difference=b['fee']-a['fee']
            require(path=='next_open_cost' and abs(difference)<=PERIOD_FEE_ATOL,'Period fee exceeds strict1e-10 bound')
            residuals.append(dict(signal_date=a['signal_date'],execution_date=a['execution_date'],
                previous206=a['fee'],current207=b['fee'],signed_difference=difference,
                absolute_difference=abs(difference),absolute_tolerance=PERIOD_FEE_ATOL))
    return residuals


def incumbent_comparison(evidence,prior_manifest,results,prior_results):
    evidence.verify(ENGINE,ENGINE_SHA)
    code=(ROOT/ENGINE).read_text()
    snippets=['symbols=set(old)|set(chosen)','def turnover(post):return sum(abs((post/n if s in chosen and n else 0)-old.get(s,0)) for s in symbols)',
        'post=(lo+hi)/2;fee=cost*turnover(post)','for s in sorted(symbols):','totalcost+=fee;trades+=tt','fees_paid=totalcost']
    require(all(x in code for x in snippets),'Frozen fee implementation differs')
    report={}
    for path in PATHS:
        current,prior=results['arms'][ARMS[0]][path],prior_results['arms'][ARMS[0]][path]
        require(current['metrics']==prior['metrics'] and current['slices']==prior['slices'],
            'Incumbent metrics/slices must be EXACT including total fees: '+path)
        exported={}
        for kind in ('TRADES','DAILY','DAILY_POSITIONS','HOLDINGS'):
            name=ARMS[0]+'_'+path+'_'+kind+'.json'
            a=evidence.load(PRIOR/'results'/name,prior_manifest['artifacts_sha256'][name])
            b=evidence.load(OUT/'results'/name)
            if kind!='HOLDINGS':require(a==b,'Incumbent exact '+kind+' differs: '+path)
            exported[kind]=(a,b)
        residuals=compare_holdings(*exported['HOLDINGS'],path)
        total_sums={}
        for label,holds,result in (('previous206',exported['HOLDINGS'][0],prior),('current207',exported['HOLDINGS'][1],current)):
            running=0.
            for row in holds:running+=row['fee']
            require(running==result['metrics']['fees_paid'],'Chronological period fees do not exactly match recorded total')
            total_sums[label]=running
        require(total_sums['previous206']==total_sums['current207'],'Incumbent total fees must be exact')
        report[path]=dict(metrics_and_all_slices_exact=True,fee_total_exact=True,fee_totals=total_sums,
            individual_fills_exact=True,fill_rows=len(exported['TRADES'][0]),fill_fee_values_exact=True,
            daily_equity_exact=True,daily_equity_rows=len(exported['DAILY'][0]),daily_positions_exact=True,
            holding_nonfee_fields_exact=True,period_fee_differences=residuals,
            max_absolute_period_fee_difference=max([x['absolute_difference'] for x in residuals],default=0.))
    return dict(paths=report,engine_sha256=ENGINE_SHA,
        tolerance_scope='Only individual next_open_cost HOLDINGS.fee differences may be<=1e-10; all metrics/slices including total fees, fills, daily equity, positions and nonfee holding fields are exact.',
        explanation='Pinned engine sums turnover over set(old)|set(chosen), then accumulates rebalance fees chronologically. Tiny per-period fee-only differences are consistent with floating summation order; process hash seeds are not reconstructed. No wider tolerance or portfolio recomputation is used.')




def number(value,label,nonnegative=False):
    require(type(value) in (int,float) and math.isfinite(value) and
        (not nonnegative or value>=0),label+' numeric descriptor')


def inspect_cell(cell):
    require(type(cell['source_available']) is bool and type(cell['available']) is bool,'Cell masks must be bool')
    for k in ('candidate_rank','control_rank'):
        number(cell[k],k);require(abs(cell[k])<=.5,'Centered rank bound')
    if not cell['source_available']:
        require(not cell['available'] and cell['dispersion_estimates']==0 and cell['down_count']==0 and
            cell['prices']==cell['volumes']==cell['down_flags']==cell['down_indices']==cell['down_volumes']==[],
            'Missing source was used')
    else:
        require(len(cell['prices'])==64 and len(cell['volumes'])==63 and len(cell['down_flags'])==63,
            'Source/flag window lengths')
        for k in ('prices','volumes'):
            require(all(type(x) in (int,float) and math.isfinite(x) and x>0 for x in cell[k]),'Invalid recorded source')
        require(all(type(x) is bool for x in cell['down_flags']),'Down flags are not bool')
        indices=[i for i,flag in enumerate(cell['down_flags']) if flag]
        require(cell['down_indices']==indices and cell['down_count']==len(indices) and
            cell['down_volumes']==[cell['volumes'][i] for i in indices],'Recorded down alignment differs')
        require(cell['available']==(len(indices)>=2),'Common minimum down support differs')
    if not cell['available']:
        require(cell['candidate'] is None and cell['control'] is None and
            cell['candidate_rank']==cell['control_rank']==0. and cell['dispersion_estimates']==0,
            'Missing raw/rank/count neutrality differs')
        if cell['source_available']:require(cell['reason']=='insufficient_down_days','Missing support reason')
        for arm in ('candidate','control'):
            require(cell[arm+'_scale'] is None and cell[arm+'_mean'] is None and
                cell[arm+'_squared_deviation_sum'] is None and cell[arm+'_scaled']==[] and
                cell[arm+'_constant'] is False,'Missing dispersion must not be estimated')
        return cell['reason']
    require(cell['reason']=='available' and cell['dispersion_estimates']==2,'Available status/count')
    for arm,n in (('candidate',cell['down_count']),('control',63)):
        number(cell[arm],arm,True);number(cell[arm+'_scale'],arm+' scale');number(cell[arm+'_mean'],arm+' mean')
        number(cell[arm+'_squared_deviation_sum'],arm+' centered sum',True)
        require(cell[arm+'_scale']>0 and cell[arm+'_mean']>0 and len(cell[arm+'_scaled'])==n and
            all(type(x) in (int,float) and math.isfinite(x) and 0<=x<=1 for x in cell[arm+'_scaled']),
            'Scaled positive-group descriptors')
        require(type(cell[arm+'_constant']) is bool,'Constant flag type')
        if cell[arm+'_constant']:
            require(cell[arm]==0. and cell[arm+'_squared_deviation_sum']==0. and
                cell[arm+'_mean']==1. and cell[arm+'_scaled']==[1.]*n,'Exact constant branch')
    return 'available'


def feature_history(feature,meta,sessions,source):
    require(feature['events']==122 and feature['symbols_per_event']==49 and feature['checked_cells']==5978 and
        feature['price_count']==64 and feature['volume_count']==63 and feature['minimum_down_days']==2,'Feature axes/window')
    require(feature['common_availability'] is True and feature['ranking_performed'] is True and
        feature['ranking_eligibility_filtered'] is False and feature['trading_eligibility_modified'] is False and
        feature['portfolio_returns_calculated'] is False,'Feature rank/scope interpretation')
    coverage=source['source_window_coverage'];events=feature['event_history']
    require(len(events)==len(coverage['by_query'])==122 and len(meta['symbols'])==len(meta['dates'])==122,
        'Feature/source query count')
    source_counts=[];available_counts=[];reasons={};down_counts=[]
    for i,(event,record) in enumerate(zip(events,coverage['by_query'])):
        day=meta['dates'][i];symbols=meta['symbols'][i];t=sessions.index(day)
        require(len(symbols)==len(set(symbols))==49 and event['date']==record['date']==day and
            event['symbols']==symbols and event['session_position']==t,'Feature query identities')
        require((i==0 and day=='2016-08-17') or (t+1<len(sessions) and sessions[t+1][:7]!=day[:7]),
            'Query is not an official month-end')
        pdays=sessions[t-63:t+1];vdays=pdays[1:]
        require(len(pdays)==64 and event['price_dates']==pdays and event['volume_dates']==vdays and
            record['price_count']==64 and record['volume_count']==63,'Official window dates')
        for key,wanted in dict(price_start=pdays[0],price_end=pdays[-1],volume_start=vdays[0],volume_end=vdays[-1]).items():
            require(event[key]==record[key]==wanted,'Source/feature window differs: '+key)
        for key in ('missing_official_price_sessions','missing_official_volume_sessions'):
            require(event[key]==record[key],'Missing session evidence differs')
        require(len(event['cells'])==len(record['cells'])==len(event['source_available'])==len(event['available'])==49,
            'Cell coverage differs')
        scount=acount=0
        for j,(cell,source_cell) in enumerate(zip(event['cells'],record['cells'])):
            require(cell['symbol']==source_cell['symbol']==symbols[j] and cell['date']==day,'Cell identity differs')
            reason=inspect_cell(cell);reasons[reason]=reasons.get(reason,0)+1
            require(type(source_cell['available']) is bool and cell['source_available']==source_cell['available']==
                event['source_available'][j] and cell['available']==event['available'][j],'Source/mask support differs')
            scount+=cell['source_available'];acount+=cell['available']
            if cell['source_available']:down_counts.append(cell['down_count'])
        require(event['source_available_count']==scount and event['available_count']==acount and
            record['common_available_cells']==scount,'Event counts differ')
        source_counts.append(scount);available_counts.append(acount)
    source_total,total=sum(source_counts),sum(available_counts)
    require(feature['source_available_by_event']==source_counts and feature['available_by_event']==available_counts and
        feature['source_available_cells']==source_total and feature['available_cells']==total and
        feature['unavailable_cells']==5978-total and feature['reason_counts']==reasons,'Top feature coverage')
    counts=dict(valid_price_observations=64*source_total,valid_volume_observations=63*source_total,
        down_comparisons=63*source_total,dispersion_estimates=2*total,raw_score_observations=2*total,
        ranked_score_observations=2*total)
    require(all(type(feature[k]) is int and feature[k]==v for k,v in counts.items()),'Feature totals differ')
    require(coverage['common_available_cells']==source_total and coverage['unavailable_cells']==5978-source_total and
        coverage['accepted_source_price_observations']==counts['valid_price_observations'] and
        coverage['accepted_source_volume_observations']==counts['valid_volume_observations'],'Source admission totals')
    return counts,dict(source_available_cells=source_total,available_cells=total,unavailable_cells=5978-total,
        source_available_by_event=source_counts,available_by_event=available_counts,reason_counts=reasons,
        recorded_down_count_range=[min(down_counts),max(down_counts)] if down_counts else None,
        complete_source_but_below_two_down=source_total-total,
        zero_predictive_fits=True,feature_math_recomputed_by_reader=False)


def saved_arrays(evidence,feature,validation):
    evidence.verify(OUT/'prepared/FEATURES.npz')
    with np.load(OUT/'prepared/FEATURES.npz',allow_pickle=False) as archive:
        require(set(archive.files)=={'candidate','control','available','candidate_rank','control_rank'},'Feature array keys')
        values={k:archive[k] for k in archive.files}
    mask=values['available']
    require(mask.shape==(122,49) and mask.dtype==np.dtype(bool),'Feature mask axes/types')
    for name in ('candidate','control'):
        raw,rank=values[name],values[name+'_rank']
        require(raw.shape==rank.shape==(122,49) and raw.dtype==rank.dtype==np.dtype(float) and
            not np.isinf(raw).any() and np.array_equal(np.isfinite(raw),mask) and np.all(raw[mask]>=0) and
            np.isfinite(rank).all() and np.all(np.abs(rank)<=.5) and np.all(rank[~mask]==0),'Raw/rank support/bounds')
    for i,row in enumerate(feature['event_history']):
        require(np.array_equal(mask[i],row['available']),'Saved cell availability')
        for j,cell in enumerate(row['cells']):
            for name in ('candidate','control'):
                require(cell[name]==float(values[name][i,j]) if mask[i,j] else cell[name] is None,'Saved raw vs audit')
                require(cell[name+'_rank']==float(values[name+'_rank'][i,j]),'Saved rank vs audit')
    evidence.verify(OUT/'prepared/TRANSFORMS_DELTAS.npz')
    with np.load(OUT/'prepared/TRANSFORMS_DELTAS.npz',allow_pickle=False) as archive:
        require(set(archive.files)=={'transform_'+a for a in ARMS[1:]}|{'delta_'+a for a in ARMS[1:]},'Transform/delta keys')
        data={k:archive[k] for k in archive.files}
    changed={}
    for arm,name in zip(ARMS[1:],('candidate','control')):
        transform,delta=data['transform_'+arm],data['delta_'+arm]
        require(transform.shape==delta.shape==(122,49) and transform.dtype==delta.dtype==np.dtype(float) and
            np.array_equal(transform,values[name+'_rank']) and np.isfinite(delta).all(),'Transform/rank identity')
        expected=.025*transform;expected[0]=0.
        require(np.array_equal(delta,expected) and np.all(np.abs(delta)<=.0125),'Direct fixed-dose delta/initial bound')
        changed[arm]=int(np.count_nonzero(delta))
    require(validation['delta']['changed_cells']==changed,'Delta changed-cell counts differ')
    return dict(shared_source_mask=True,all_original_candidates_ranked_before_eligibility=True,
        unchanged_raw_and_rank_audit=True,exact_saved_linear_delta=True,first_partial_delta_zero=True,
        changed_cells=changed,independent_numerical_scope='Separate scalar oracle verifies formulas and exact ranks; reader checks saved array/audit/delta identity only.')


def review(completion_sha,check_only=False):
    require(isinstance(completion_sha,str) and len(completion_sha)==64,'Explicit closed completion SHA required')
    require(not (OUT/'RESEARCH_REVIEW.json').exists(),'Final review is write-once')
    e=Evidence();closure=e.load(OUT/'COMPLETION.json',completion_sha)
    require(closure['status']=='COMPLETED_INDEPENDENTLY_VERIFIED','Round207 must already be closed')
    plan_sha=closure['plan_sha256'];plan=e.load(OUT/'PLAN.json',plan_sha)
    context=e.load(CONTEXT,CONTEXT_SHA)
    require(plan['schema']=='monthly_down_volume207_v1' and plan['contract']['arms']==list(ARMS) and
        plan['contract']['candidate']==ARMS[1] and plan['contract']['control_nonpromotable']==ARMS[2] and
        plan['contract']['context']==context and plan['contract']['prospective_context_sha256']==CONTEXT_SHA and
        plan['contract']['dose']==.025 and plan['contract']['max_absolute_delta']==.0125 and
        plan['contract']['max_absolute_rank']==.5,'Different207 prospective contract')
    require(len(plan['raw_source_hashes'])==64,'Original raw source lineage differs')
    for mapping in (plan['input_sha256'],plan['raw_source_hashes']):
        for path,h in mapping.items():
            require(input_path(path,h)!=CENTRAL,'Mutable central cannot be persistent PLAN input')
            require(plan['input_sha256'].get(str(input_path(path,h)))==h,'Source absent from persistent pins')
            e.verify(path,h)
    before=e.load(OUT/'STATE_BEFORE.json',plan['prior_central_sha256'])
    require(len(before['completed'])==204 and sum(x['trials'] for x in before['completed'])==1129,
        'Pre-registration counters differ')
    for key in ('actual_new_feature_evaluations_before_registration','actual_new_model_fits_before_registration',
        'actual_down_comparisons_before_registration','actual_dispersion_estimates_before_registration',
        'actual_ranks_before_registration','actual_new_returns_before_registration'):
        require(type(plan[key]) is int and plan[key]==0,'Pre-registration actual computation')
    rc=e.manifest(OUT/'results',closure['result_completion_sha256'],plan_sha,RESULT_FILES)
    require(rc['paths']==9 and rc['slices']==45,'9paths45slices required')
    frozen=e.load(OUT/'selected/FROZEN.json',rc['frozen_sha256'])
    require(frozen['plan_sha256']==plan_sha,'Selection plan differs')
    prepared=e.manifest(OUT/'prepared',frozen['prepared_sha256'],plan_sha,PREPARED_FILES)
    require(frozen['orders_sha256']==prepared['artifacts_sha256']['ORDERS.json'],'Frozen orders binding differs')
    training=e.load(OUT/'selected/TRAIN_RESULTS.json',frozen['training_sha256'])
    orders=e.load(OUT/'prepared/ORDERS.json',frozen['orders_sha256'])
    results=e.load(OUT/'results/RESULTS.json');meta=e.load(OUT/'prepared/META.json')
    feature=e.load(OUT/'prepared/FEATURE_AUDIT.json');validation=e.load(OUT/'prepared/INDEPENDENT_PREPARE.json')
    outcome=e.load(OUT/'results/INDEPENDENT_OUTCOME.json');parity=e.load(OUT/'prepared/SOURCE_PARITY.json')
    require(parity['raw_source_hashes']==plan['raw_source_hashes'] and
        parity['price_source_binding']==plan['price_source_binding'] and parity['source_support_exact'] is True,
        'Source binding differs')
    require(results['down_volume_dispersion'] is True and results['same_support_both_arms'] is True and
        results['observer_promotion'] is False and results['strict_pit'] is False and results['reused_sample'] is True,
        'Result interpretation differs')
    test_sha=plan['input_sha256'][str(ROOT/TESTS)];code_review_sha=plan['numerical_code_review']['sha256']
    source_sha=plan['price_source_binding']['sha256']
    require(test_sha==TESTS_SHA and code_review_sha==CODE_REVIEW_SHA,'Formal test/code review pins differ')
    require(local_path(plan['numerical_code_review']['path'])==ROOT/CODE_REVIEW and
        local_path(plan['price_source_binding']['path'])==ROOT/SOURCE,'Review reference paths differ')
    require(source_sha==SOURCE_SHA,'Source review differs from admitted receipt')
    code=e.load(CODE_REVIEW,code_review_sha);source=e.load(SOURCE,source_sha);tests=e.load(TESTS,test_sha)
    require(plan['input_sha256'][str(ROOT/CODE_REVIEW)]==code_review_sha and
        plan['input_sha256'][str(ROOT/SOURCE)]==source_sha,'Evidence binding differs')
    require(code['passed'] is True and code['unresolved_blockers']==[] and code['context_sha256']==CONTEXT_SHA and
        source['passed'] is True and source['unresolved_blockers']==[] and source['context_sha256']==CONTEXT_SHA and
        tests['passed'] is True and tests['synthetic_only'] is True and type(tests['tests']) is int and tests['tests']>0,
        'Incomplete reviews/tests')
    require(tests['runtime']==plan['runtime']==dict(python='3.9.6',numpy='2.0.2',pandas='2.3.3'),'Runtime differs')
    require(set(code['code_sha256'])==NUMERICAL_FILES,'Final six numerical source set differs')
    require(code['code_sha256']==FROZEN_COMPONENTS,'Final six source hashes changed')
    for path,h in code['code_sha256'].items():
        require(tests['code_sha256'][str(ROOT/path)]==h and plan['input_sha256'][str(ROOT/path)]==h,
            'Formal tested source/PLAN pin differs')
    for mapping in (code['code_sha256'],code['input_sha256'],tests['code_sha256']):
        for path,h in mapping.items():
            require(input_path(path,h)!=CENTRAL,'Mutable central in code/test closure')
            e.verify(path,h)
    for key in ('actual_features_computed','actual_model_fits_computed','actual_portfolio_returns_computed',
                'production_files_modified','strategy_registered_by_reviewer'):
        require(code[key] is False,'Pre-registration code review scope differs: '+key)
    require(source['source_only'] is True and source['actual_source_coverage_computed'] is True and
        source['strict_pit'] is False and source['strict_pit_vintage'] is False and
        source['strict_pit_membership'] is False,'Source admission interpretation')
    for key in ('actual_features_computed','actual_down_flags_computed','actual_dispersion_computed',
                'actual_ranks_computed','actual_deltas_computed','actual_orders_computed',
                'actual_returns_computed','actual_portfolio_returns_computed','production_files_modified',
                'strategy_registered_by_reviewer','observers_accessed'):
        require(source[key] is False,'Source review crossed feature boundary: '+key)
    require(source['raw_source_hashes']==plan['raw_source_hashes'] and source['new_model_fits']==0 and
        source['raw_source_parity']['exact_source_map_equal'] is True and source['raw_source_parity']['count']==64,
        'Original raw source binding differs')
    for path,h in source['source_sha256'].items():
        require(input_path(path,h)!=CENTRAL and plan['input_sha256'].get(str(input_path(path,h)))==h,
            'Source absent from PLAN/invalid persistent pin')
        e.verify(path,h)
    original_meta=ROOT/'outputs/monthly_local_linear70/META.json'
    require(meta==e.load(original_meta,plan['input_sha256'][str(original_meta)]),'Original query identity changed')
    original_orders=ROOT/'outputs/monthly_local_linear70/local_linear_ORDERS.json'
    require(orders[ARMS[0]]==e.load(original_orders,plan['input_sha256'][str(original_orders)]),'Original full orders differ')
    for key in FEATURE_FLAGS:require(validation['feature'].get(key) is True,'Missing feature proof: '+key)
    for key in DELTA_FLAGS:require(validation['delta'].get(key) is True,'Missing delta proof: '+key)
    check=validation['feature']
    require(check['canonical_feature_comparison']=='exact' and check['rank_comparison']=='exact' and
        check['pairwise_identity_atol']==1e-12 and check['computes_returns'] is False,
        'Independent exact math/tie and secondary identity tolerance differ')
    require(validation['delta']['dose']==.025 and validation['delta']['max_absolute_delta']==.0125 and
        validation['delta']['checked_events']==122,'Fixed delta scope differs')
    require(validation['prefixes']==123 and validation['feature_prefixes']==[0,1,53,88,122] and
        validation['feature_prefix_sources_truncated'] is True,'Causal prefix coverage incomplete')
    for key in ('source_support_exact','original_orders_exact','original_arrays_unchanged',
                'original_core_metadata_and_support_preserved','single_original_episode'):
        require(validation[key] is True,'Missing original invariant: '+key)
    require(validation['original_array_count']==prepared['original_array_count']==closure['original_array_count']==27,
        'Original27 accounting differs')
    require(prepared['new_returns_computed'] is False,'Preparation computed portfolio returns')
    for record in (validation,prepared,closure):
        for key in ('new_model_fits','predictive_model_fits','new_predictive_model_fits'):
            require(type(record[key]) is int and record[key]==0,'New predictive model fits reported')
        require(record['model_fit_accounting']==plan['contract']['model_fit_accounting'],'Fit accounting interpretation')
    require(check['new_model_fits']==check['predictive_model_fits']==feature['new_model_fits']==
        feature['predictive_model_fits']==0,'Independent/producer model fits')
    calendar=e.load(CALENDAR,plan['input_sha256'][str(CALENDAR)])
    counts,coverage=feature_history(feature,meta,calendar['sessions'],source)
    for key in ('events','symbols_per_event','checked_cells','source_available_cells','source_available_by_event',
                'available_cells','available_by_event','unavailable_cells'):
        require(check[key]==feature[key],'Independent feature coverage: '+key)
    for record in (check,validation,prepared,closure):
        require(all(type(record[k]) is int and record[k]==v for k,v in counts.items()),'Phase/independent counts differ')
    integration=saved_arrays(e,feature,validation)
    require(set(orders)==set(training)==set(results['arms'])==set(outcome['paths'])==set(ARMS),'Three arms required')
    require(len(meta['dates'])==len(meta['ends'])==122 and all(len(x)==122 for x in orders.values()),'122 monthly orders required')
    require(frozen['selection']==results['selection']==selection_from_saved(training,orders),'Exact first88 guards differ')
    selection=frozen['selection']
    require(outcome['passed'] is True and outcome['checked_paths']==9 and outcome['checked_slices']==45
        and outcome['independent_accounts'] is True and outcome['independent_of_production_model_and_engine'] is True
        and outcome['terminal_liquidation'] is False,'Independent outcome incomplete')
    e.verify('work/ptr169_results_independent.py',outcome['accounting_source_sha256'])
    require(closure['protected_files']==21444 and closure['observers_modified'] is False and closure['new_trials']==3,'Protection/trial closure differs')
    protected_path=ROOT/'work/trade_reconcile_20261015_close/NEXT_EXPECTED_PROTECTED.json'
    protected=e.load(protected_path,plan['input_sha256'][str(protected_path)])
    require(len(protected)==21444,'Protected manifest incomplete')
    predecessor=plan['predecessor_binding'];require(predecessor['directory']==PRIOR.name,'Different predecessor')
    old_closed=e.load(PRIOR/'COMPLETION.json',predecessor['input_sha256'][str(PRIOR/'COMPLETION.json')])
    old_manifest=e.load(PRIOR/'results/COMPLETION.json',old_closed['result_completion_sha256'])
    old_results=e.load(PRIOR/'results/RESULTS.json',old_manifest['artifacts_sha256']['RESULTS.json'])
    require(old_closed['status']=='COMPLETED_INDEPENDENTLY_VERIFIED' and old_manifest['status']=='COMPLETE','Prior closure incomplete')
    incumbent=incumbent_comparison(e,old_manifest,results,old_results)
    summaries={};checked_slices=0;checked_training=0
    for arm in ARMS:
        require(set(results['arms'][arm])==set(outcome['paths'][arm])==set(PATHS),'Missing execution path')
        summaries[arm]={}
        for path in PATHS:
            result,independent=results['arms'][arm][path],outcome['paths'][arm][path]
            require(independent['passed'] is True and independent['monthly_executions']==122
                and set(result['slices'])==set(independent['slices'])==set(BOUNDS),'Missing independent slice')
            require(result['metrics']==result['slices']['full122'],'Full metrics/slice mismatch')
            for period in BOUNDS:
                metric_evidence(independent['slices'][period],result['slices'][period],arm+'/'+path+'/'+period)
                checked_slices+=1
            summaries[arm][path]={period:{k:result['slices'][period][k] for k in METRICS} for period in BOUNDS}
        require(set(training[arm])=={'same_close_zero','next_open_cost'},'Selection paths differ')
        for path in training[arm]:
            require(set(training[arm][path])=={'training','early','recent'},'Selection eras differ')
            for short,full in (('training','first88'),('early','early53'),('recent','recent35')):
                metric_evidence(outcome['paths'][arm][path]['slices'][full],training[arm][path][short],'training/'+arm+'/'+path+'/'+short)
                checked_training+=1
    require(checked_slices==45 and checked_training==18,'Independent metrics coverage incomplete')
    changed=dict(candidate_vs_incumbent=changes(orders,ARMS[0],ARMS[1]),control_vs_incumbent=changes(orders,ARMS[0],ARMS[2]),
        candidate_vs_control=changes(orders,ARMS[2],ARMS[1]))
    require([x['index'] for x in changed['candidate_vs_incumbent']['months'] if x['index']<88]==selection['changed_months']
        and [x['index'] for x in changed['candidate_vs_control']['months'] if x['index']<88]==selection['changed_months_vs_control'],'First88 changed-set counts differ')
    central=e.load(CENTRAL)
    require(central['current']==OUT.name and central['current_status']=='completed_independently_verified'
        and central['registered_queue']==[] and len(central['completed'])==205
        and sum(r['trials'] for r in central['completed'])==1132,'Immediate closed207 central state differs')
    latest=central['completed'][-1];best=closure['best_training_selected_candidate'];oldbest=before['best_training_selected_candidate']
    require(latest['directory']==OUT.name and latest['completion']['sha256']==completion_sha
        and local_path(latest['completion']['path'])==OUT/'COMPLETION.json' and latest['primary']==selection['primary']
        and latest['new_admitted']==int(selection['admitted']) and latest['trials']==3
        and central['best_training_selected_candidate']==best,'Central receipt or selection differs')
    pick=selection['primary'];selected=results['arms'][pick]['same_close_zero']['metrics']
    should_update=selection['admitted'] and selected['cagr']>oldbest['cagr']
    if should_update:
        require(best['directory']==OUT.name and best['spec']==dict(arm=pick)
            and all(best[k]==selected[k] for k in ('cagr','sharpe','max_drawdown'))
            and best['next_open_5bp_cagr']==results['arms'][pick]['next_open_cost']['metrics']['cagr']
            and best['selected_on_first88'] is True and best['observer_promotion'] is False,'Best update does not match recorded selected result')
    else:require(best==oldbest,'Unqualified candidate changed historical best')
    latest_tracking=central['latest_training_selected_candidate']
    require(latest_tracking['directory']==OUT.name and latest_tracking['arm']==pick and latest_tracking['selection']==selection
        and latest_tracking['full_period_cagr']==selected['cagr'] and latest_tracking['full_period_sharpe']==selected['sharpe']
        and latest_tracking['selected_on_first88'] is True,'Latest central tracking differs')
    require(closure['target_reached']==(best['cagr']>=.7),'Target-reached label differs')

    e.verify(__file__)
    for p,h in list(e.pins.items()):e.verify(p,h)
    report=dict(schema='monthly_down_volume207_research_review_v1',created_utc=datetime.now(timezone.utc).isoformat(),passed=True,
        reviewer_was_production_kernel_author=True,independent_of_production_kernel=False,
        review_scope='Saved-evidence hashes, source/feature support, recorded volume-dispersion descriptors, counts, rank/delta identity, exact first88 guards and independently recorded outcomes. No new signals, source loads or portfolio replay.',
        independent_numerical_evidence_source='Separately authored work/monthly_down_volume207_independent.py and hash-bound INDEPENDENT_PREPARE.json/INDEPENDENT_OUTCOME.json',
        formal_test_receipt_sha256=test_sha,code_review_sha256=code_review_sha,source_review_sha256=source_sha,
        source_refs=e.pins,phase_sha256=dict(plan=plan_sha,prepared=frozen['prepared_sha256'],selected=rc['frozen_sha256'],
            results=closure['result_completion_sha256'],closure=completion_sha),
        result='ADMITTED' if selection['admitted'] else 'NOT_ADOPTED',selection=selection,
        failed_training_guards=[k for k,v in selection['guards'].items() if not v],
        best_training_selected_candidate=best,historical_best_unchanged=best==oldbest,
        best_update_matches_recorded_results=True,exact_incumbent_metrics_equal_predecessor=True,
        incumbent_export_diagnostic=incumbent,summary=summaries,holding_changes=changed,
        descriptive_counts=counts,feature_coverage=coverage,score_integration=integration,
        validation=dict(synthetic_tests=tests['tests'],independent_cells=5978,independent_execution_paths=9,
            independent_period_slices=45,training_metric_groups=18,order_prefixes=123,
            physical_source_prefixes=[0,1,53,88,122],original_arrays=27,protected_files=21444,
            protection_scope='Hash-bound manifest and completed registration/closure receipts; no repeated protected archive scan.',
            central_completed_entries=205,central_trials=1132,new_predictive_model_fits=0,
            exact_canonical_features_and_ranks=True,secondary_pairwise_identity_atol=1e-12),
        strict_pit=False,pristine_oos=False,reused_historical_sample=True,limitations=plan['contract']['limitation'],
        volume_source_provenance=source['volume_provenance'],
        observers_modified=False,central_modified=False,new_features_computed_in_this_review=False,
        new_orders_computed_in_this_review=False,new_portfolio_returns_computed_in_this_review=False,
        actual_production_modules_imported=False,saved_numerical_arrays_inspected=True,target_reached=closure['target_reached'])
    path=OUT/'RESEARCH_REVIEW.json'
    answer=dict(passed=True,check_only=check_only,path=str(path),result=report['result'],synthetic_tests=tests['tests'],
        historical_best_unchanged=report['historical_best_unchanged'],target_reached=report['target_reached'],
        failed_guards=report['failed_training_guards'],holding_change_counts={k:v['counts'] for k,v in changed.items()},
        descriptive_counts=counts,
        incumbent_period_fee_residuals={p:len(v['period_fee_differences']) for p,v in incumbent['paths'].items()},summary=summaries)
    if not check_only:
        with path.open('x') as stream:
            json.dump(report,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
        answer['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    print(json.dumps(answer,allow_nan=False))


def self_test():
    # Fabricated saved descriptors only; no source files or production imports.
    checks=0
    def reject(fn):
        nonlocal checks
        try:fn()
        except (ValueError,KeyError):checks+=1
        else:raise ValueError('Synthetic rejection expected')
    require(len(PREPARED_FILES)==7 and len(RESULT_FILES)==38,'Manifest cardinality');checks+=1
    holds=[dict(signal_date=str(i),execution_date=str(i),fee=1.,symbol='A') for i in range(122)]
    changed=copy.deepcopy(holds);changed[3]['fee']+=5e-11
    require(len(compare_holdings(holds,changed,'next_open_cost'))==1,'Fee disclosure');checks+=1
    reject(lambda:compare_holdings(holds,changed,'same_close_zero'))
    changed[3]['fee']+=2e-10;reject(lambda:compare_holdings(holds,changed,'next_open_cost'))
    changed=copy.deepcopy(holds);changed[1]['symbol']='B';reject(lambda:compare_holdings(holds,changed,'next_open_cost'))
    require(compare_holdings(holds,holds,'next_open_cost')==[],'No residual is valid');checks+=1
    metric=dict(training=dict(cagr=.5,sharpe=1.5,max_drawdown=-.2),early=dict(cagr=.5),recent=dict(cagr=.5))
    metrics={arm:{path:copy.deepcopy(metric) for path in ('same_close_zero','next_open_cost')} for arm in ARMS}
    orders={arm:[dict(symbols=['A','B','C']) for _ in range(122)] for arm in ARMS}
    require(not selection_from_saved(metrics,orders)['admitted'],'No false admission');checks+=1
    for path in metrics[ARMS[1]]:metrics[ARMS[1]][path]['training']['cagr']=.51
    for i in (1,2,53):orders[ARMS[1]][i]['symbols']=['A','B','D']
    selected=selection_from_saved(metrics,orders);require(selected['admitted'],'Synthetic admission');checks+=1
    for i in range(88,122):orders[ARMS[1]][i]['symbols']=['POISON']
    require(selection_from_saved(metrics,orders)==selected,'Later34 affects selection');checks+=1
    orders[ARMS[2]]=copy.deepcopy(orders[ARMS[1]])
    require(not selection_from_saved(metrics,orders)['guards']['changed_support_vs_control'],'Control guard');checks+=1
    reject(lambda:input_path('/private/tmp/unregistered-evidence',REGISTERED_RUNTIME_SHA))
    cell=dict(source_available=True,available=False,reason='insufficient_down_days',
        candidate=None,control=None,candidate_rank=0.,control_rank=0.,prices=[100.]*64,volumes=[1.]*63,
        down_flags=[False]*63,down_indices=[],down_count=0,down_volumes=[],candidate_scale=None,control_scale=None,
        candidate_scaled=[],control_scaled=[],candidate_mean=None,control_mean=None,
        candidate_squared_deviation_sum=None,control_squared_deviation_sum=None,
        candidate_constant=False,control_constant=False,dispersion_estimates=0)
    require(inspect_cell(cell)=='insufficient_down_days','Complete source but unavailable feature');checks+=1
    good=copy.deepcopy(cell);good.update(available=True,reason='available',candidate=0.,control=0.,
        down_indices=[0,62],down_count=2,down_volumes=[1.,1.],dispersion_estimates=2)
    good['down_flags'][0]=good['down_flags'][62]=True
    for arm,n in (('candidate',2),('control',63)):
        good.update({arm+'_scale':1.,arm+'_scaled':[1.]*n,arm+'_mean':1.,
            arm+'_squared_deviation_sum':0.,arm+'_constant':True})
    require(inspect_cell(good)=='available','Exact constant available branch');checks+=1
    bad=copy.deepcopy(good);bad['down_indices']=[0,61];reject(lambda:inspect_cell(bad))
    bad=copy.deepcopy(good);bad['control']=1.;reject(lambda:inspect_cell(bad))
    bad=copy.deepcopy(cell);bad['candidate']=0.;reject(lambda:inspect_cell(bad))
    dates=[str(d) for d in np.arange(np.datetime64('2000-01-01'),np.datetime64('2012-01-01'))
           if datetime.strptime(str(d),'%Y-%m-%d').weekday()<5]
    ends=[d for i,d in enumerate(dates[:-1]) if d[:7]!=dates[i+1][:7]][2:124]
    names=['S%02d'%j for j in range(49)];meta=dict(dates=ends,symbols=[names[:] for _ in ends])
    history=[];source_rows=[]
    for day in ends:
        t=dates.index(day);pd=dates[t-63:t+1];vd=pd[1:]
        fields=dict(date=day,price_start=pd[0],price_end=pd[-1],volume_start=vd[0],volume_end=vd[-1],
            missing_official_price_sessions=[],missing_official_volume_sessions=[])
        cells=[dict(copy.deepcopy(cell),symbol=s,date=day) for s in names]
        history.append(dict(fields,symbols=names[:],session_position=t,price_dates=pd,volume_dates=vd,
            source_available=[True]*49,available=[False]*49,source_available_count=49,available_count=0,cells=cells))
        source_rows.append(dict(fields,price_count=64,volume_count=63,common_available_cells=49,
            cells=[dict(symbol=s,available=True) for s in names]))
    counts=dict(valid_price_observations=64*5978,valid_volume_observations=63*5978,down_comparisons=63*5978,
        dispersion_estimates=0,raw_score_observations=0,ranked_score_observations=0)
    feature=dict(events=122,symbols_per_event=49,checked_cells=5978,price_count=64,volume_count=63,
        minimum_down_days=2,common_availability=True,ranking_performed=True,ranking_eligibility_filtered=False,
        trading_eligibility_modified=False,portfolio_returns_calculated=False,event_history=history,
        source_available_cells=5978,source_available_by_event=[49]*122,available_cells=0,
        available_by_event=[0]*122,unavailable_cells=5978,reason_counts={'insufficient_down_days':5978},**counts)
    source=dict(source_window_coverage=dict(common_available_cells=5978,unavailable_cells=0,
        accepted_source_price_observations=64*5978,accepted_source_volume_observations=63*5978,by_query=source_rows))
    checked,coverage=feature_history(feature,meta,dates,source)
    require(checked==counts and coverage['complete_source_but_below_two_down']==5978,'S/A counter distinction');checks+=1
    bad=copy.deepcopy(feature);bad['dispersion_estimates']=2;reject(lambda:feature_history(bad,meta,dates,source))
    bad=copy.deepcopy(feature);bad['event_history'][0]['volume_dates']=bad['event_history'][0]['price_dates'][:63]
    reject(lambda:feature_history(bad,meta,dates,source))
    print(json.dumps(dict(passed=True,synthetic_checks=checks,actual_artifacts_read=False,
        new_features=False,new_orders=False,new_returns=False)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--completion-sha256');parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--check-only',action='store_true');args=parser.parse_args()
    if args.self_test:
        require(args.completion_sha256 is None and not args.check_only,'Synthetic-only self-test')
        self_test()
    else:
        require(bool(args.completion_sha256),'Explicit closed completion SHA required')
        review(args.completion_sha256,args.check_only)

