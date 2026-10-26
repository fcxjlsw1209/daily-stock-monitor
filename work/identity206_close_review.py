"""Write-once saved-evidence reader for closed round206.

The author also implemented the206 model; this reader is not an independent
mathematical verifier. Model independence comes from the separately authored
augmented least-squares/scalar oracle and its hash-bound preparation receipt.

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
OUT=ROOT/'outputs/monthly_identity_transfer206'
PRIOR=ROOT/'outputs/monthly_adjacent_risk205'
CENTRAL=ROOT/'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
CALENDAR=ROOT/'outputs/monthly_paired_target143/CALENDAR.json'
ARMS=('incumbent_local67','peer_residual_transfer','own_residual_control')
PATHS=('same_close_zero','next_open_zero','next_open_cost')
BOUNDS={'full122':(0,122),'first88':(0,88),'early53':(0,53),'recent35':(53,88),'later34':(88,122)}
METRICS=('start','end','cagr','sharpe','max_drawdown')
COUNT_FIELDS=('source_target_fits','reference_pooled_log_refits','reference_local_evaluations','reference_local_solves','transfer_pooled_log_refits','own_constant_evaluations','peer_query_evaluations','peer_month_evaluations','peer_row_observations','actual_regression_solves')
RAW_KEYS=('source_log','local_linear','local_constant','candidate','control')
MODEL_PREFIXES=(0,1,53,88,122)
PREPARED_FILES={'FORECASTS.npz','RANKS_DELTAS.npz','MODEL_AUDIT.json','ORDERS.json','META.json','SOURCE_PARITY.json','INDEPENDENT_PREPARE.json'}
RESULT_FILES={'RESULTS.json','INDEPENDENT_OUTCOME.json'}|{
    a+'_'+p+'_'+k+'.json' for a in ARMS for p in PATHS for k in ('DAILY','TRADES','HOLDINGS','DAILY_POSITIONS')}
CONTEXT='work/monthly_identity_transfer206_context.json'
CONTEXT_SHA='29d7d503564aca5ac2ebe66a4180e519e93427e0497ba74ecebd15999ab7255b'
TESTS='work/monthly_identity_transfer206_TESTS.json'
CODE_REVIEW='work/identity206_CODE_REVIEW.json'
SOURCE='work/identity206_SOURCE_REVIEW.json'
SOURCE_SHA='cb4ac9494e06ffab11b718930471c4469646a3df6d2801dc4458b532a4bbf8ae'
MODEL_META='outputs/monthly_ptr_pool_source169/fixed_incumbent_rebuild/control/MONTHLY_META.json'
FINAL_CODE_SHA={
 'work/monthly_identity_transfer206_host.py':'00051399f9b00864990b8807fb9bf3306d491ce01042fceddd8ff85f8c1695c6',
 'work/test_monthly_identity_transfer206_host.py':'c912878b8eaedb769ace56ed2912c36660a44e5524d1a5bf3cea66de7c0ce4e8',
 'work/monthly_identity_transfer206_model.py':'8171b0bf0cd8d05a7ff53f914db5e3f1ff06d7e53ac29579a74554de1a166f33',
 'work/test_monthly_identity_transfer206_model.py':'0fad1530cf3772717f8d2a76c8a941a23d45da0f2f3bc11442d38d27e1ec2a15',
 'work/monthly_identity_transfer206_independent.py':'a321b2cd07eb2b19c08f7747842e18a47438d781bed9b474c8ef321ec9f5611e',
 'work/test_monthly_identity_transfer206_independent.py':'6d4f83fc4de39df277329161c1bf7746d25f294ff44d923064f6c340161cd931'}
MODEL_FLAGS=('passed','independent_augmented_least_squares','annual_prior_december_cutoffs_checked',
 'five_year_mature_cohort_checked','four_target_common_cohort_checked','original_source_and_local_parity',
 'availability_unchanged','own_history_threshold_checked','matched_month_identity_checked',
 'query_security_excluded','peer_kernel_checked','own_month_mass_preserved',
 'common_shrinkage_penalty_checked','no_future_or_outcome_inputs','audit_checked')
SLOT_FLAGS=('passed','full_original_universe','includes_eligible_etfs','no_local_history_filter',
 'replacement_not_additive_local_slot','paired_whole_row_readiness','initial_partial_unchanged','exact_rank_rebuild')
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
        close_vs_own_control=guard(metrics[ARMS[1]]['same_close_zero'],metrics[ARMS[2]]['same_close_zero'],0.),
        open_vs_own_control=guard(metrics[ARMS[1]]['next_open_cost'],metrics[ARMS[2]]['next_open_cost'],0.),
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
                previous205=a['fee'],current206=b['fee'],signed_difference=difference,
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
        for label,holds,result in (('previous205',exported['HOLDINGS'][0],prior),('current206',exported['HOLDINGS'][1],current)):
            running=0.
            for row in holds:running+=row['fee']
            require(running==result['metrics']['fees_paid'],'Chronological period fees do not exactly match recorded total')
            total_sums[label]=running
        require(total_sums['previous205']==total_sums['current206'],'Incumbent total fees must be exact')
        report[path]=dict(metrics_and_all_slices_exact=True,fee_total_exact=True,fee_totals=total_sums,
            individual_fills_exact=True,fill_rows=len(exported['TRADES'][0]),fill_fee_values_exact=True,
            daily_equity_exact=True,daily_equity_rows=len(exported['DAILY'][0]),daily_positions_exact=True,
            holding_nonfee_fields_exact=True,period_fee_differences=residuals,
            max_absolute_period_fee_difference=max([x['absolute_difference'] for x in residuals],default=0.))
    return dict(paths=report,engine_sha256=ENGINE_SHA,
        tolerance_scope='Only individual next_open_cost HOLDINGS.fee differences may be<=1e-10; all metrics/slices including total fees, fills, daily equity, positions and nonfee holding fields are exact.',
        explanation='Pinned engine sums turnover over set(old)|set(chosen), then accumulates rebalance fees chronologically. Tiny per-period fee-only differences are consistent with floating summation order; process hash seeds are not reconstructed. No wider tolerance or portfolio recomputation is used.')




def integer_counts(counts,label):
    require(isinstance(counts,dict) and set(counts)==set(COUNT_FIELDS) and
        all(type(v) is int and v>=0 for v in counts.values()),label+' count schema')
    require(counts['actual_regression_solves']==sum(counts[k] for k in
        ('source_target_fits','reference_pooled_log_refits','reference_local_solves','transfer_pooled_log_refits')),
        label+' regression accounting')
    require(counts['peer_query_evaluations']==counts['own_constant_evaluations']==
        counts['reference_local_evaluations'],label+' matched local evaluations')
    require(counts['reference_local_solves']<=counts['reference_local_evaluations'],label+' local solve bound')


def finite(value,label):
    require(type(value) in (int,float) and math.isfinite(value),label+' nonfinite')


def prefix_evidence(validation,meta,counts):
    require(validation['prefixes']==123 and validation['model_prefixes']==list(MODEL_PREFIXES) and
        validation['model_prefix_sources_truncated'] is True,'Causal prefix evidence incomplete')
    rows=validation['model_prefix_evidence']
    require([row['prefix'] for row in rows]==list(MODEL_PREFIXES),'Physical prefix coverage differs')
    for row in rows:
        n=row['prefix'];integer_counts(row['counts'],'prefix '+str(n))
        require(type(row['training_rows']) is int and row['training_rows']>=0 and row['query_rows']==49*n,
            'Prefix source shape differs')
        if n==0:
            require(row['training_rows']==0 and row['latest_label_end'] is None and
                row['query_date'] is None and row['cutoff'] is None and not any(row['counts'].values()),
                'Empty prefix contains source/model activity')
        else:
            day=meta['dates'][n-1];cutoff='%04d-12-31'%(int(day[:4])-1)
            require(row['query_date']==day and row['cutoff']==cutoff and
                (row['latest_label_end'] is None or row['latest_label_end']<=cutoff),'Physical prefix maturity differs')
    require(rows[-1]['counts']==counts,'Full physical prefix accounting differs')
    return rows


def saved_arrays(evidence,validation,plan):
    """Inspect saved forecasts/ranks/deltas only; no source NPZ or model execution."""
    evidence.verify(OUT/'prepared/FORECASTS.npz')
    with np.load(OUT/'prepared/FORECASTS.npz',allow_pickle=False) as archive:
        require(set(archive.files)=={'raw_'+k for k in RAW_KEYS}|{'available','local_available'},'Forecast artifact keys')
        values={k:archive[k] for k in archive.files}
    available=values['available'];local=values['local_available']
    require(available.shape==local.shape==(5978,) and available.dtype==local.dtype==np.dtype(bool),
        'Forecast availability axes/types')
    require(not np.any(local&~available),'Local availability exceeds pooled')
    for key in RAW_KEYS:
        raw=values['raw_'+key]
        require(raw.shape==(5978,) and raw.dtype==np.dtype(float) and not np.isinf(raw).any()
            and np.array_equal(np.isfinite(raw),available),'Raw forecast support differs: '+key)
    require(np.array_equal(values['raw_control'],values['raw_local_constant'],equal_nan=True),'Control exact-copy identity failed')
    fallback=available&~local
    for key in ('candidate','control'):
        require(np.array_equal(values['raw_'+key][fallback],values['raw_source_log'][fallback]),'Insufficient-own pooled fallback')
    evidence.verify(OUT/'prepared/RANKS_DELTAS.npz')
    with np.load(OUT/'prepared/RANKS_DELTAS.npz',allow_pickle=False) as archive:
        require(set(archive.files)=={'rank_'+a for a in ARMS[1:]}|{'delta_'+a for a in ARMS[1:]},'Rank/delta artifact keys')
        slots={k:archive[k] for k in archive.files}
    changed={}
    for arm in ARMS[1:]:
        rank,delta=slots['rank_'+arm],slots['delta_'+arm]
        require(rank.shape==delta.shape==(122,49) and rank.dtype==delta.dtype==np.dtype(float) and
            np.isfinite(rank).all() and np.isfinite(delta).all(),'Rank/delta axes/types/nonfinite')
        require(np.max(np.abs(rank))<=.5 and np.max(np.abs(delta))<=.0125+1e-12 and
            np.array_equal(delta[0],np.zeros(49)),'Original score slot/initial bound')
        changed[arm]=int(np.count_nonzero(delta))
    require(validation['slot']['changed_cells']==changed,'Recorded changed-cell counts')
    ready=validation['slot']['ready_rows']
    require(set(ready)==set(ARMS[1:]) and all(type(v) is int and 0<=v<=122 for v in ready.values())
        and len(set(ready.values()))==1,'Paired readiness differs')
    return values,dict(raw_keys=list(RAW_KEYS),available_queries=int(available.sum()),
        local_available_queries=int(local.sum()),exact_control_constant_copy=True,
        exact_insufficient_own_pooled_copy=True,rank_and_delta_bounds_checked=True,
        changed_cells=changed,paired_ready_rows=ready)


def model_history(audit,meta,training_meta,values,check):
    """Validate saved identities/clock/counts; do not recompute residuals or fits."""
    require(audit['arms']==list(ARMS[1:]) and audit['raw_keys']==list(RAW_KEYS) and audit['feature_count']==10
        and audit['target']=='unchanged cached centered log_excess','Model descriptor differs')
    for k,v in dict(alpha=10.,half_life_months=12.,min_months=18,min_own_months=18,
                    bandwidth=.5,intercept_penalty=12.).items():
        require(audit[k]==v,'Fixed parameter differs: '+k)
    counts=audit['counts'];integer_counts(counts,'main')
    require(check['counts']==counts,'Independent model counts differ')
    require(check['atol']==1e-12 and check['computes_returns'] is False and
        check['residuals_are_mature_in_sample'] is True and check['residuals_are_oof'] is False,
        'Independent model tolerance/scope differs')
    td=[];te=[];ts=[]
    for day,end,names in zip(training_meta['dates'],training_meta['ends'],training_meta['symbols']):
        td.extend([day]*len(names));te.extend([end]*len(names));ts.extend(names)
    require(len(td)==len(te)==len(ts)==8820,'Original training metadata coverage')
    query_dates=[d for d,names in zip(meta['dates'],meta['symbols']) for _ in names]
    query_symbols=[s for names in meta['symbols'] for s in names]
    require(len(query_dates)==len(query_symbols)==5978,'Query metadata coverage')
    annual=audit['annual'];by_year={r['year']:r for r in annual}
    years=sorted({int(d[:4]) for d in meta['dates']})
    require(sorted(by_year)==years and len(annual)==len(years)==check['annual_cohorts']==11,'Annual query coverage')
    available=values['available'];local=values['local_available'];predicted_fits=0;cohorts={}
    for row in annual:
        year=row['year'];cutoff='%04d-12-31'%(year-1);start='%04d-12-31'%(year-6)
        ids=row['training_row_ids']
        require(row['cutoff']==cutoff and row['lookback_exclusive_start']==start and
            isinstance(ids,list) and all(type(i) is int and 0<=i<len(td) for i in ids) and
            ids==sorted(set(ids)) and row['training_rows']==len(ids),'Annual source identity/clock')
        require(all(td[i]>start and te[i]<=cutoff for i in ids),'Immature/out-of-window source row')
        identities=[(td[i][:7],ts[i]) for i in ids]
        require(len(identities)==len(set(identities)),'Duplicate monthly security identity')
        by_month={};by_symbol={}
        for i in ids:
            by_month.setdefault(td[i][:7],[]).append(i)
            by_symbol.setdefault(ts[i],[]).append(i)
        cohorts[year]=(by_month,by_symbol);months=set(by_month)
        require(row['training_months']==len(months) and
            all(len(rows)>=2 for rows in by_month.values()),'Common mature month support')
        require(row['first_training_signal']==(min(td[i] for i in ids) if ids else None) and
            row['latest_label_end']==(max(te[i] for i in ids) if ids else None),'Annual source extrema')
        pred=sum(bool(available[j]) and int(day[:4])==year for j,day in enumerate(query_dates))
        require(row['prediction_rows']==pred and row['status']==('fit' if len(months)>=18 else 'cold_start'),'Annual status')
        if pred:
            predicted_fits+=1;finite(row['weight_sum'],'annual weight')
            require(row['status']=='fit' and row['weight_sum']>0,'Actual fit requires adequate support')
        else:require(row['weight_sum'] is None,'No-query year reports fit weight')
    require(counts['transfer_pooled_log_refits']==counts['reference_pooled_log_refits']==predicted_fits
        and counts['source_target_fits']==4*predicted_fits,'Pooled fit accounting')
    require(type(check['independent_pooled_fits']) is int and check['independent_pooled_fits']==predicted_fits,
        'Independent fit accounting')
    source=audit['source']
    require(source['targets']==['simple_excess','log_excess','vol_scaled','loss_utility'] and
        source['alpha']==10. and source['half_life_months']==12. and
        source['predictions_are_raw'] is True and source['label_ranks_used'] is False,'Original four-target source descriptor')
    require(len(source['annual'])==len(annual),'Source annual cohort coverage')
    for row in source['annual']:
        actual=by_year[row['year']]
        require(all(row[k]==actual[k] for k in ('cutoff','training_rows','training_months','status')) and
            (row['status']!='fit' or row['prediction_rows']==actual['prediction_rows']),
            'Source/new annual cohort mismatch')
    records=audit['local'];seen=set();fits=0;solves=0;peer_months=0;peer_rows=0
    for row in records:
        j=row['event_row'];require(type(j) is int and 0<=j<5978 and j not in seen and available[j],'Local query identity')
        seen.add(j);day=query_dates[j];symbol=query_symbols[j];year=int(day[:4]);yr=by_year[year]
        require(row['date']==day and row['symbol']==symbol and row['year']==year and row['cutoff']==yr['cutoff'],
            'Local query clock')
        by_month,by_symbol=cohorts[year];ids=by_symbol.get(symbol,[])
        own_dates=[td[i] for i in ids];own_ends=[te[i] for i in ids]
        nm=len({d[:7] for d in own_dates})
        require(row['own_row_ids']==ids and row['own_dates']==own_dates and row['own_label_ends']==own_ends
            and row['own_rows']==len(ids) and row['own_months']==nm and
            row['latest_label_end']==(max(own_ends) if ids else None),'Own cohort identity')
        sufficient=nm>=18
        require(row['status']==('fit' if sufficient else 'insufficient_own_history') and
            bool(local[j])==sufficient,'Own-history support threshold')
        for k in ('weight_mass','effective_samples','own_numerator','peer_numerator','constant_correction',
                  'linear_correction','peer_correction'):finite(row[k],'local '+k)
        if not sufficient:
            require(row['own_weights']==row['own_residuals']==row['peer_months']==[] and
                row['correction_denominator'] is None and
                all(row[k]==0. for k in ('weight_mass','effective_samples','own_numerator','peer_numerator',
                    'constant_correction','linear_correction','peer_correction')),'Insufficient-own row is not neutral')
            continue
        fits+=1;w=row['own_weights'];residual=row['own_residuals'];peers=row['peer_months']
        require(len(w)==len(residual)==len(peers)==len(ids),'Own/peer month lengths')
        for value in w:finite(value,'own weight');require(value>0,'Nonpositive own weight')
        for value in residual:finite(value,'own residual')
        require(row['weight_mass']>0 and row['correction_denominator']==row['weight_mass']+12.,'Shared mass/penalty')
        solves+=int(any(v!=0. for v in residual))
        for k,entry in enumerate(peers):
            expected=[i for i in by_month[own_dates[k][:7]] if ts[i]!=symbol]
            require(expected and entry['month']==own_dates[k][:7] and entry['own_row_id']==ids[k] and
                entry['own_weight']==w[k] and entry['own_residual']==residual[k] and
                entry['peer_row_ids']==expected and entry['peer_count']==len(expected),'Peer month/identity/mass changed')
            for name in ('peer_weight_mass','peer_weighted_residual','peer_mean'):finite(entry[name],'peer '+name)
            require(entry['peer_weight_mass']>0,'Empty or nonpositive peer mass')
            peer_months+=1;peer_rows+=len(expected)
    require(seen==set(np.flatnonzero(available).tolist()) and len(records)==check['local_queries'],
        'Local audit omits/duplicates available query')
    require(counts['own_constant_evaluations']==counts['peer_query_evaluations']==fits and
        counts['reference_local_solves']==solves and counts['peer_month_evaluations']==peer_months and
        counts['peer_row_observations']==peer_rows,'Local/peer recorded count mismatch')
    return dict(annual_cohorts=len(annual),available_queries=len(records),own_supported_queries=fits,
        insufficient_own_queries=len(records)-fits,peer_months=peer_months,peer_rows=peer_rows,
        counts=counts,mature_source_identity_checked=True,query_security_excluded=True,
        mathematical_audit_source='Separate independent augmented-LS/scalar preparation receipt; this reader only validates saved descriptors, identities and counts.')


def review(completion_sha,check_only=False):
    require(isinstance(completion_sha,str) and len(completion_sha)==64,'Explicit closed completion SHA required')
    require(not (OUT/'RESEARCH_REVIEW.json').exists(),'Final review is write-once')
    e=Evidence();closure=e.load(OUT/'COMPLETION.json',completion_sha)
    require(closure['status']=='COMPLETED_INDEPENDENTLY_VERIFIED','Round206 must already be closed')
    plan_sha=closure['plan_sha256'];plan=e.load(OUT/'PLAN.json',plan_sha)
    context=e.load(CONTEXT,CONTEXT_SHA)
    require(plan['schema']=='monthly_identity_transfer206_v1' and plan['contract']['arms']==list(ARMS) and
        plan['contract']['candidate']==ARMS[1] and plan['contract']['control_nonpromotable']==ARMS[2] and
        plan['contract']['context']==context and plan['contract']['prospective_context_sha256']==CONTEXT_SHA,
        'Different206 prospective contract')
    require(plan['contract']['training_months']==88 and plan['contract']['later_months']==34 and
        plan['contract']['model_prefixes']==list(MODEL_PREFIXES) and plan['contract']['all123_order_prefixes'] is True
        and plan['contract']['no_new_score_budget'] is True,'Query/rank/order scope differs')
    require(len(plan['raw_source_hashes'])==64,'Original raw source lineage differs')
    for mapping in (plan['input_sha256'],plan['raw_source_hashes']):
        for path,h in mapping.items():
            require(input_path(path,h)!=CENTRAL,'Mutable central cannot be persistent PLAN input')
            require(plan['input_sha256'].get(str(input_path(path,h)))==h,'Source absent from persistent pins')
            e.verify(path,h)
    before=e.load(OUT/'STATE_BEFORE.json',plan['prior_central_sha256'])
    require(len(before['completed'])==203 and sum(x['trials'] for x in before['completed'])==1126,
        'Pre-registration counters differ')
    for key in ('actual_new_feature_evaluations_before_registration','actual_new_model_fits_before_registration',
        'actual_new_peer_corrections_before_registration','actual_new_orders_before_registration',
        'actual_new_returns_before_registration'):
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
    audit=e.load(OUT/'prepared/MODEL_AUDIT.json');validation=e.load(OUT/'prepared/INDEPENDENT_PREPARE.json')
    outcome=e.load(OUT/'results/INDEPENDENT_OUTCOME.json');parity=e.load(OUT/'prepared/SOURCE_PARITY.json')
    require(parity['raw_source_hashes']==plan['raw_source_hashes'] and
        parity['model_source_binding']==plan['model_source_binding'],'Cached source binding differs')
    for key in ('original_training_cache_exact','original_identity_exact','original_control_constant_exact'):
        require(parity[key] is True,'Missing source parity: '+key)
    require(parity['new_labels'] is False and parity['new_raw_features'] is False,'Source scope changed')
    require(results['matched_month_identity_transfer'] is True and results['same_support_both_arms'] is True and
        results['observer_promotion'] is False and results['strict_pit'] is False and results['reused_sample'] is True,
        'Result interpretation differs')
    test_sha=plan['input_sha256'][str(ROOT/TESTS)];code_review_sha=plan['numerical_code_review']['sha256']
    require(local_path(plan['numerical_code_review']['path'])==ROOT/CODE_REVIEW and
        local_path(plan['model_source_binding']['path'])==ROOT/SOURCE,'Review reference paths differ')
    code=e.load(CODE_REVIEW,code_review_sha);source=e.load(SOURCE,SOURCE_SHA);tests=e.load(TESTS,test_sha)
    require(plan['input_sha256'][str(ROOT/CODE_REVIEW)]==code_review_sha and
        plan['model_source_binding']['sha256']==SOURCE_SHA and plan['input_sha256'][str(ROOT/SOURCE)]==SOURCE_SHA,
        'Pre-registration evidence binding differs')
    require(code['passed'] is True and code['unresolved_blockers']==[] and code['context_sha256']==CONTEXT_SHA and
        source['passed'] is True and source['context_sha256']==CONTEXT_SHA and tests['passed'] is True and
        tests['synthetic_only'] is True and type(tests['tests']) is int and tests['tests']>0,'Incomplete reviews/tests')
    require(tests['runtime']==plan['runtime']==dict(python='3.9.6',numpy='2.0.2',pandas='2.3.3'),'Runtime differs')
    require(code['code_sha256']==FINAL_CODE_SHA,'Final six numerical source hashes differ')
    for path,h in FINAL_CODE_SHA.items():
        require(tests['code_sha256'][str(ROOT/path)]==h and plan['input_sha256'][str(ROOT/path)]==h,
            'Formal tested source/PLAN pin differs')
    for mapping in (code['code_sha256'],code['input_sha256'],tests['code_sha256']):
        for path,h in mapping.items():
            require(input_path(path,h)!=CENTRAL,'Mutable central in code/test closure')
            e.verify(path,h)
    for key in ('actual_features_computed','actual_model_fits_computed','actual_portfolio_returns_computed',
                'production_files_modified','strategy_registered_by_reviewer'):
        require(code[key] is False,'Pre-registration code review scope differs: '+key)
    for key in ('actual_features_computed','actual_model_fits_computed','actual_orders_computed',
                'actual_returns_computed','actual_source_arrays_loaded','new_data_acquired'):
        require(source[key] is False,'Source review crossed boundary: '+key)
    require(source['immutable_source_files']==len(source['source_sha256']) and
        source['raw_source_hashes']==plan['raw_source_hashes'] and source['prior_saved_training_rows']==8820 and
        source['prior_saved_query_rows']==5978 and source['prior_saved_feature_count']==10 and
        source['prior_parity_receipt_reverified'] is True and source['strict_pit'] is False,'Original model sources differ')
    for path,h in source['source_sha256'].items():
        require(input_path(path,h)!=CENTRAL and plan['input_sha256'].get(str(input_path(path,h)))==h,
            'Source absent from PLAN/invalid persistent pin')
        e.verify(path,h)
    binding=plan['model_source_binding']
    require(binding['original_cached_training_and_queries'] is True and binding['new_labels'] is False and
        binding['new_raw_features'] is False and binding['strict_pit'] is False,'Model binding interpretation differs')
    original_meta=ROOT/'outputs/monthly_local_linear70/META.json'
    require(meta==e.load(original_meta,plan['input_sha256'][str(original_meta)]),'Original query identity changed')
    original_orders=ROOT/'outputs/monthly_local_linear70/local_linear_ORDERS.json'
    require(orders[ARMS[0]]==e.load(original_orders,plan['input_sha256'][str(original_orders)]),'Original full orders differ')
    cache_meta=e.load(MODEL_META,plan['input_sha256'][str(ROOT/MODEL_META)])
    require(cache_meta['monthly']==meta,'Model/cache query identity differs')
    for key in MODEL_FLAGS:require(validation['model'].get(key) is True,'Missing independent model proof: '+key)
    for key in SLOT_FLAGS:require(validation['slot'].get(key) is True,'Missing slot proof: '+key)
    for key in ('original_orders_exact','original_arrays_unchanged','original_core_metadata_and_support_preserved',
                'single_original_episode','no_new_score_budget'):
        require(validation[key] is True,'Missing original invariant: '+key)
    require(validation['original_array_count']==prepared['original_array_count']==closure['original_array_count']==27,
        'Original27 accounting differs')
    require(prepared['new_returns_computed'] is False,'Preparation computed portfolio returns')
    counts=audit['counts']
    require(validation['model_counts']==prepared['model_counts']==closure['model_counts']==counts,'Phase fit counts differ')
    require(prepared['model_fit_accounting']==closure['model_fit_accounting']==plan['contract']['model_fit_accounting'],
        'Fit-accounting interpretation changed')
    prefixes=prefix_evidence(validation,meta,counts)
    values,integration=saved_arrays(e,validation,plan)
    history=model_history(audit,meta,cache_meta['training'],values,validation['model'])
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
        and central['registered_queue']==[] and len(central['completed'])==204
        and sum(r['trials'] for r in central['completed'])==1129,'Immediate closed206 central state differs')
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
    report=dict(schema='monthly_identity_transfer206_research_review_v1',
        created_utc=datetime.now(timezone.utc).isoformat(),passed=True,
        reviewer_was_production_model_author=True,independent_of_production_model=False,
        review_scope='Saved artifact integrity, annual/own/peer identity and clock descriptors, actual fit accounting, exact first88 guards and existing outcome consistency. No new model fit, residual/forecast computation or strategy replay.',
        independent_numerical_evidence_source='Separately authored work/monthly_identity_transfer206_independent.py and hash-bound INDEPENDENT_PREPARE.json/INDEPENDENT_OUTCOME.json',
        exact_external_runtime_allowlist={str(REGISTERED_RUNTIME):REGISTERED_RUNTIME_SHA},
        source_admission=dict(source_review_sha256=SOURCE_SHA,source_pins=len(source['source_sha256']),
            raw_lineage_sources=64,prior_saved_training_rows=8820,prior_saved_query_rows=5978,
            original_cached_features=10,new_labels=False,new_raw_features=False,raw_prices_rebuilt_by_reader=False),
        formal_test_receipt_sha256=test_sha,code_review_sha256=code_review_sha,
        source_refs=e.pins,phase_sha256=dict(plan=plan_sha,prepared=frozen['prepared_sha256'],
            selected=rc['frozen_sha256'],results=closure['result_completion_sha256'],closure=completion_sha),
        result='ADMITTED' if selection['admitted'] else 'NOT_ADOPTED',selection=selection,
        failed_training_guards=[k for k,v in selection['guards'].items() if not v],
        best_training_selected_candidate=best,historical_best_unchanged=best==oldbest,
        best_update_matches_recorded_results=True,exact_incumbent_metrics_equal_predecessor=True,
        incumbent_export_diagnostic=incumbent,summary=summaries,holding_changes=changed,
        model_counts=counts,model_history=history,score_slot=integration,
        model_prefix_evidence=prefixes,
        validation=dict(synthetic_tests=tests['tests'],independent_queries=5978,
            independent_execution_paths=9,independent_period_slices=45,training_metric_groups=18,
            order_prefixes=123,physical_model_prefixes=list(MODEL_PREFIXES),original_arrays=27,
            protected_files=21444,
            protection_scope='Hash-bound manifest and completed registration/closure protection receipts; reader does not repeat the protected archive scan.',
            central_completed_entries=204,central_trials=1129,main_pass_model_counts=counts,
            independent_pooled_fits=validation['model']['independent_pooled_fits'],
            prefix_fit_counts_recorded_separately=True),
        strict_pit=False,pristine_oos=False,reused_historical_sample=True,
        residuals_are_mature_in_sample=True,residuals_are_oof=False,
        limitations=plan['contract']['limitation'],observers_modified=False,central_modified=False,
        new_features_computed_in_this_review=False,new_models_fitted_in_this_review=False,
        new_orders_computed_in_this_review=False,new_portfolio_returns_computed_in_this_review=False,
        actual_production_modules_imported=False,saved_numerical_arrays_inspected=True,
        target_reached=closure['target_reached'])
    path=OUT/'RESEARCH_REVIEW.json'
    summary=dict(passed=True,check_only=check_only,path=str(path),result=report['result'],
        synthetic_tests=tests['tests'],model_counts=counts,
        historical_best_unchanged=report['historical_best_unchanged'],target_reached=report['target_reached'],
        failed_guards=report['failed_training_guards'],holding_change_counts={k:v['counts'] for k,v in changed.items()},
        incumbent_period_fee_residuals={p:len(v['period_fee_differences']) for p,v in incumbent['paths'].items()},
        summary=summaries)
    if not check_only:
        with path.open('x') as stream:
            json.dump(report,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
        summary['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    print(json.dumps(summary,allow_nan=False))


def self_test():
    # Fabricated summary/model descriptors only. No strategy imports or real data.
    checks=0
    def reject(fn):
        nonlocal checks
        try:fn()
        except (ValueError,KeyError):checks+=1
        else:raise ValueError('Synthetic rejection expected')
    require(len(PREPARED_FILES)==7 and len(RESULT_FILES)==38,'Manifest cardinality');checks+=1
    a=[dict(signal_date=str(i),execution_date=str(i),fee=1.,symbol='A') for i in range(122)]
    b=copy.deepcopy(a);b[3]['fee']+=5e-11
    require(len(compare_holdings(a,b,'next_open_cost'))==1,'Fee residual disclosure');checks+=1
    reject(lambda:compare_holdings(a,b,'same_close_zero'))
    b[3]['fee']+=2e-10;reject(lambda:compare_holdings(a,b,'next_open_cost'))
    b=copy.deepcopy(a);b[1]['symbol']='B';reject(lambda:compare_holdings(a,b,'next_open_cost'))
    require(compare_holdings(a,a,'next_open_cost')==[],'Zero residual is valid');checks+=1
    metric=dict(training=dict(cagr=.5,sharpe=1.5,max_drawdown=-.2),early=dict(cagr=.5),recent=dict(cagr=.5))
    metrics={arm:{path:copy.deepcopy(metric) for path in ('same_close_zero','next_open_cost')} for arm in ARMS}
    orders={arm:[dict(symbols=['A','B','C']) for _ in range(122)] for arm in ARMS}
    require(not selection_from_saved(metrics,orders)['admitted'],'No false admission');checks+=1
    for path in metrics[ARMS[1]]:metrics[ARMS[1]][path]['training']['cagr']=.51
    for i in (1,2,53):orders[ARMS[1]][i]['symbols']=['A','B','D']
    selected=selection_from_saved(metrics,orders);require(selected['admitted'],'Expected synthetic admission');checks+=1
    for i in range(88,122):orders[ARMS[1]][i]['symbols']=['POISON']
    require(selection_from_saved(metrics,orders)==selected,'Later34 affects selection');checks+=1
    orders[ARMS[2]]=copy.deepcopy(orders[ARMS[1]])
    require(not selection_from_saved(metrics,orders)['guards']['changed_support_vs_control'],'Matched control guard');checks+=1
    require(input_path(REGISTERED_RUNTIME,REGISTERED_RUNTIME_SHA)==REGISTERED_RUNTIME,'Runtime allowlist');checks+=1
    reject(lambda:input_path(REGISTERED_RUNTIME,'0'*64))
    reject(lambda:input_path('/private/tmp/unregistered-evidence',REGISTERED_RUNTIME_SHA))
    zero={k:0 for k in COUNT_FIELDS};integer_counts(zero,'zero');checks+=1
    bad=copy.deepcopy(zero);bad['actual_regression_solves']=1;reject(lambda:integer_counts(bad,'bad'))
    bad=copy.deepcopy(zero);bad['peer_query_evaluations']=True;reject(lambda:integer_counts(bad,'bad bool'))
    names=['S%02d'%i for i in range(49)]
    date=lambda y,m:'%04d-%02d-28'%(y,m)
    td=[date(2001+i//12,1+i%12) for i in range(180)]
    te=[date(2001+(i+1)//12,1+(i+1)%12) for i in range(180)]
    tm=dict(dates=td,ends=te,symbols=[names[:] for _ in td])
    qd=[date(2016+(i+7)//12,1+(i+7)%12) for i in range(122)]
    meta=dict(dates=qd,symbols=[names[:] for _ in qd])
    available=np.zeros(5978,bool);available[0]=True
    values=dict(available=available,local_available=available.copy())
    ids=[49*m+j for m in range(156,174) for j in (0,1)]
    own=ids[::2];peer=ids[1::2];annual=[]
    for year in range(2016,2027):
        used=ids if year==2016 else []
        annual.append(dict(year=year,cutoff='%04d-12-31'%(year-1),
            lookback_exclusive_start='%04d-12-31'%(year-6),training_rows=len(used),
            training_months=18 if used else 0,prediction_rows=1 if used else 0,training_row_ids=used,
            first_training_signal=td[156] if used else None,latest_label_end=te[173] if used else None,
            status='fit' if used else 'cold_start',weight_sum=36. if used else None))
    counts=dict(source_target_fits=4,reference_pooled_log_refits=1,reference_local_evaluations=1,
        reference_local_solves=1,transfer_pooled_log_refits=1,own_constant_evaluations=1,
        peer_query_evaluations=1,peer_month_evaluations=18,peer_row_observations=18,actual_regression_solves=7)
    row=dict(event_row=0,date=qd[0],symbol=names[0],year=2016,cutoff='2015-12-31',
        own_rows=18,own_months=18,own_row_ids=own,own_dates=td[156:174],own_label_ends=te[156:174],
        latest_label_end=te[173],status='fit',weight_mass=18.,effective_samples=18.,
        own_weights=[1.]*18,own_residuals=[.1]*18,own_numerator=1.8,peer_numerator=3.6,
        correction_denominator=30.,constant_correction=.06,linear_correction=.07,peer_correction=.12,
        peer_months=[dict(month=td[m][:7],own_row_id=o,own_weight=1.,own_residual=.1,
            peer_row_ids=[p],peer_count=1,peer_weight_mass=1.,peer_weighted_residual=.2,peer_mean=.2)
            for m,o,p in zip(range(156,174),own,peer)])
    audit=dict(arms=list(ARMS[1:]),raw_keys=list(RAW_KEYS),feature_count=10,
        target='unchanged cached centered log_excess',alpha=10.,half_life_months=12.,min_months=18,
        min_own_months=18,bandwidth=.5,intercept_penalty=12.,counts=counts,annual=annual,local=[row],
        source=dict(targets=['simple_excess','log_excess','vol_scaled','loss_utility'],alpha=10.,
            half_life_months=12.,predictions_are_raw=True,label_ranks_used=False,annual=copy.deepcopy(annual)))
    check=dict(counts=counts,atol=1e-12,computes_returns=False,residuals_are_mature_in_sample=True,
        residuals_are_oof=False,annual_cohorts=11,independent_pooled_fits=1,local_queries=1)
    require(model_history(audit,meta,tm,values,check)['peer_rows']==18,'Recorded model lineage');checks+=1
    bad=copy.deepcopy(audit);bad['local'][0]['peer_months'][0]['peer_row_ids']=[own[0]]
    reject(lambda:model_history(bad,meta,tm,values,check))
    bad=copy.deepcopy(audit);bad['local'][0]['peer_months'][0]['own_weight']=2.
    reject(lambda:model_history(bad,meta,tm,values,check))
    bad=copy.deepcopy(audit);bad['annual'][0]['cutoff']='2016-12-31'
    reject(lambda:model_history(bad,meta,tm,values,check))
    prefixes=dict(prefixes=123,model_prefixes=list(MODEL_PREFIXES),model_prefix_sources_truncated=True,
        model_prefix_evidence=[dict(prefix=n,training_rows=0 if not n else 8820,query_rows=n*49,
            query_date=qd[n-1] if n else None,cutoff='%04d-12-31'%(int(qd[n-1][:4])-1) if n else None,
            latest_label_end='2015-01-28' if n else None,counts=counts if n else zero) for n in MODEL_PREFIXES])
    prefix_evidence(prefixes,meta,counts);checks+=1
    bad=copy.deepcopy(prefixes);bad['model_prefix_evidence'][1]['latest_label_end']='2026-12-31'
    reject(lambda:prefix_evidence(bad,meta,counts))
    bad=copy.deepcopy(prefixes);bad['model_prefix_evidence'][0]['counts']=counts
    reject(lambda:prefix_evidence(bad,meta,counts))
    print(json.dumps(dict(passed=True,synthetic_checks=checks,actual_artifacts_read=False,
        new_features=False,new_model_fits=False,new_orders=False,new_returns=False)))


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

