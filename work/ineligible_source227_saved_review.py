"""Review closed source227 saved JSON/hash evidence only.

Never imports models, decodes NPZ, rebuilds labels/features, computes portfolio
returns, writes central state, or executes a phase. Run only after root's explicit
close-complete authorization and with the terminal completion/PLAN SHA arguments.
"""
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import argparse
import hashlib
import json
import math
import platform

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/monthly_ineligible_source227'
PRIOR=ROOT/'outputs/monthly_joint_error226'
CENTRAL=ROOT/'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
CONTEXT=ROOT/'work/monthly_ineligible_source227_context.json'
CONTEXT_SHA='196773953dbaa5caccae64d3df2fd2b5e2fad149291d4db8c318a28eb022a9e7'
SOURCE=ROOT/'work/ineligible_source227_SOURCE_REVIEW.json'
SOURCE_SHA='4610aacb5941fe67b8496ce607cd2152c642835b761f48b295e74c33d5f370ba'
TESTS=ROOT/'work/monthly_ineligible_source227_TESTS.json'
TESTS_SHA='bdd3755d01efeaa8e65bd25a6eccd55f801d6da83397ed448ef5d85f72eba387'
PRIOR_SHA='80e1f20981a27348b4d2623d9103d3f3425d90d0ce083c4a76fb88937e68b414'
PRIOR_REVIEW_SHA='cea3cc1cece4aaa01b2cb2a55b8379ec4fb95591f0d1990fd27d5daf4170b09e'
PROTECTED=ROOT/'work/trade_reconcile_20261015_close/NEXT_EXPECTED_PROTECTED.json'
CALENDAR=ROOT/'outputs/monthly_paired_target143/CALENDAR.json'
NUMERIC_FILES=('monthly_ineligible_source227_host.py','test_monthly_ineligible_source227_host.py',
 'monthly_ineligible_source227_model.py','test_monthly_ineligible_source227_model.py',
 'monthly_ineligible_source227_independent.py','test_monthly_ineligible_source227_independent.py')
PREPARED={'FORECASTS.npz','LABELS.npz','META.json','SOURCE_AUDIT.json','INDEPENDENT.json'}
EXTERNAL={'/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/bin/python3.9':'6b24be8ba00c4abc8b3e7e2620bd710e2cf7a4312d00c147f8f05fbb6f3d297a'}
TARGETS=('simple_excess','log_excess','vol_scaled','loss_utility')
FORECAST_COUNTS=('query_rows','original_available_rows','expansion_requested_rows','added_available_rows','available_rows',
 'old_source_target_fits','new_source_target_fits','old_pooled_log_refits','new_pooled_log_refits',
 'old_local_evaluations','new_local_evaluations','old_local_solves','new_local_solves','actual_regression_solves')
LABEL_COUNTS=('training_rows','original_good_rows','added_rows','extended_good_rows','original_center_months','original_strict_exceptions','new_candidate_rows')
FORECAST_FLAGS=('passed','independent_annual_control_flow','shared_frozen_primitives_disclosed','old_forecasts_exact',
 'old_masks_exact','old_forecasts_copied','same_training_cohort','four_targets_unchanged','annual_prior_december_cutoffs',
 'five_year_mature_window','minimum_18_months','same_security_local_history','local_history_not_query_filter',
 'separate_expansion_replay','warmup_mask_enforced','no_query_rerank','fit_counts_checked','audit_checked')
LABEL_FLAGS=('passed','independent_label_rebuild','exact_original_identity','original_centering_cohort_unchanged',
 'fixed_four_clips_checked','old_targets_exact_copy','old_common_mask_preserved','original_reconstruction_checked',
 'only_previously_ineligible_added','finite_plain10_checked','full_295_history_checked','volatility63_checked',
 'full_official_future_path_checked','official_endpoints_checked','no_added_training_rows','audit_checked')
PREFIXES=(0,1,53,88,122)
CUTOFFS=('2019-12-31','2023-12-31','2025-12-31')


def need(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def resolved(value):
    p=Path(value);return (p if p.is_absolute() else ROOT/p).resolve()


def read_json(path):
    p=resolved(path);need(p.is_relative_to(ROOT),'JSON outside local research')
    def bad(v):raise ValueError('Nonfinite JSON '+v)
    return json.loads(p.read_text(),parse_constant=bad)


class Evidence:
    def __init__(self):self.pins={}
    def pin(self,path,digest=None):
        p=resolved(path);need(p!=CENTRAL.resolve(),'Do not persist mutable central as frozen input')
        need('CloudStorage' not in p.parts and (p.is_relative_to(ROOT) or EXTERNAL.get(str(p))==digest),'Input outside authorized root')
        current=sha(p);need(digest is None or current==digest,'Input SHA differs: '+str(p))
        key=str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
        need(key not in self.pins or self.pins[key]==current,'Input changed during review: '+key)
        self.pins[key]=current;return current
    def load(self,path,digest=None):self.pin(path,digest);return read_json(path)


def registered(plan,path):
    names=[h for p,h in plan['input_sha256'].items() if resolved(p)==resolved(path)]
    need(names and len(set(names))==1,'Unbound/conflicting registered input: '+str(path));return names[0]


def nonnegative_counts(counts,keys):
    need(set(counts)==set(keys),'Count schema')
    need(all(type(v) is int and v>=0 for v in counts.values()),'Counts must be nonnegative integers')


def finite(v):return type(v) in (int,float) and math.isfinite(v)


def query_axis(meta,n=None):
    n=len(meta['dates']) if n is None else n
    need(len(meta['dates'])==len(meta['ends'])==len(meta['symbols']),'Metadata query axes')
    rows=[]
    for d,e,ss in zip(meta['dates'][:n],meta['ends'][:n],meta['symbols'][:n]):
        need(len(ss)==49 and len(set(ss))==49 and all(type(s) is str and s for s in ss),'Original exact49 pool')
        rows.extend((d,s,e) for s in ss)
    need(len(rows)==len(set(rows)),'Unique query identities');return rows


def replay_audit(a,axis,scope):
    need(a['feature_count']==10 and a['alpha']==10. and a['half_life_months']==12. and a['min_months']==a['min_own_months']==18,'Original model parameters')
    need(a['bandwidth']==.5 and a['intercept_penalty']==12. and a['slope_penalty']==10.,'Original local parameters')
    need(a['same_local_rows_residuals_and_weights'] is True and a['local_weights_normalized'] is False,'Original local fit rows/weights')
    s=a['source'];need(s['alpha']==10. and s['half_life_months']==12. and s['min_months']==18 and s['lookback_years']==5 and s['targets']==list(TARGETS),'Original four-target source parameters')
    for k in ('shared_rows_and_weights','training_only_standardization','predictions_are_raw'):need(s[k] is True,'Original source '+k)
    need(s['label_ranks_used'] is False,'Targets must not be ranked')
    years=sorted(set(int(d[:4]) for d,_,_ in axis));annual=s['annual'];localannual=a['annual']
    need([r['year'] for r in annual]==years==[r['year'] for r in localannual],'Every query year separately audited')
    details=a['local'];seen=set()
    for r in details:
        j=r['event_row'];need(type(j) is int and 0<=j<len(axis) and j not in seen,'One local audit per available query');seen.add(j)
        d,symbol,_=axis[j];year=int(d[:4]);cut=f'{year-1}-12-31'
        need(r['date']==d and r['symbol']==symbol and r['year']==year and r['cutoff']==cut,'Exact query identity/cutoff')
        need(type(r['own_months']) is int and r['own_months']>=0 and r['own_rows']==r['own_months'],'One own row per month')
        need((r['latest_label_end'] is None)==(r['own_rows']==0),'Own latest maturity shape')
        if r['latest_label_end'] is not None:need(r['latest_label_end']<=cut,'No future own label')
        status=r['status'];need(status in ('fit','insufficient_own_history','zero_weight','zero_residual'),'Local status')
        if status=='insufficient_own_history':need(r['own_months']<18 and r['constant_correction']==r['linear_correction']==0.,'Own-history exact pooled fallback')
        else:need(r['own_months']>=18,'Local minimum18')
        for k in ('weight_mass','effective_samples','constant_correction','linear_correction'):need(finite(r[k]),'Finite saved local statistic')
    for r,l in zip(annual,localannual):
        y=r['year'];cut=f'{y-1}-12-31';lower=f'{y-6}-12-31'
        need(r['cutoff']==l['cutoff']==cut and r['lookback_exclusive_start']==l['lookback_exclusive_start']==lower,'Annual prior December/five-year clock')
        for key in ('training_rows','training_months','first_training_signal','latest_label_end','status'):need(r[key]==l[key],'Original source/local shared cohort '+key)
        need(r['status']==('fit' if r['training_months']>=18 else 'cold_start'),'Global minimum18')
        need(r['training_rows']>=2*r['training_months'],'At least2 original training rows per month')
        if r['training_rows']:
            need(lower<r['first_training_signal']<=r['latest_training_signal']<r['latest_label_end']<=cut,'Mature original five-year training window')
        else:need(all(r[k] is None for k in ('first_training_signal','latest_training_signal','latest_label_end')),'Empty training dates')
        actual=sum(int(row['year']==y) for row in details)
        need(l['prediction_rows']==actual,'Available local query count')
        need(actual==(r['prediction_rows'] if r['status']=='fit' else 0),'Shared source availability')
        if actual:need(finite(r['weight_sum']) and r['weight_sum']>0 and l['weight_sum']==r['weight_sum'],'Source/local shared weights')
    return dict(available_rows=len(details),source_target_fits=4*sum(r['status']=='fit' and r['prediction_rows']>0 for r in annual),
        pooled_log_refits=sum(r['prediction_rows']>0 for r in localannual),local_evaluations=len(details),
        local_solves=sum(r['status']=='fit' for r in details),event_rows=seen,annual=annual)


def forecast_audit(a,axis):
    need(a['method']=='same_query_ineligible_prediction_extension' and a['query_rows']==len(axis) and a['feature_count']==10,'Forecast method/shape')
    need(a['raw_keys']==['source_log','local_constant','local_linear'],'Three original raw forecasts')
    for k in ('old_parity_exact','original_forecasts_copied','new_queries_separate_replay','original_training_unchanged'):need(a[k] is True,'Forecast scope '+k)
    for k in ('features_recomputed','reranked'):need(a[k] is False,'No new feature/rank '+k)
    allowed,requested=a['expansion_allowed'],a['new_query_mask']
    need(len(allowed)==len(requested)==len(axis) and all(type(x) is bool for x in allowed+requested),'Exact query boolean audit')
    need(all(not x or w for x,w in zip(requested,allowed)),'New request respects signal-price allowance')
    old,new=replay_audit(a['old_replay'],axis,'old'),replay_audit(a['new_replay'],axis,'new')
    need(not (old['event_rows']&new['event_rows']),'Old and added forecast domains disjoint')
    need(all(requested[j] for j in new['event_rows']) and all(not requested[j] for j in old['event_rows']),'Forecast domains respect requested mask')
    cohort=('year','cutoff','lookback_exclusive_start','training_rows','training_months','latest_label_end','first_training_signal','latest_training_signal','status')
    for x,y in zip(old['annual'],new['annual']):need(all(x[k]==y[k] for k in cohort),'Expanded query mask cannot change training cohort')
    c=dict(query_rows=len(axis),original_available_rows=old['available_rows'],expansion_requested_rows=sum(requested),added_available_rows=new['available_rows'],available_rows=old['available_rows']+new['available_rows'])
    for name,r in (('old',old),('new',new)):
        for k in ('source_target_fits','pooled_log_refits','local_evaluations','local_solves'):c[name+'_'+k]=r[k]
    c['actual_regression_solves']=sum(c[k] for k in ('old_source_target_fits','new_source_target_fits','old_pooled_log_refits','new_pooled_log_refits','old_local_solves','new_local_solves'))
    nonnegative_counts(a['counts'],FORECAST_COUNTS);need(a['counts']==c,'Recounted forecast fits/domain counts')
    return c


def label_audit(a,meta,calendar):
    need(a['method']=='same_query_ineligible_label_extension' and a['label_parity_atol']==1e-12 and a['label_parity_rtol']==0.,'Original label reconstruction contract')
    for k in ('original_targets_copied','original_common_good_preserved','original_centers_only','added_rows_never_train','full_official_paths_required_for_added','original_strict_exceptions_retained'):need(a[k] is True,'Label scope '+k)
    triples=query_axis(meta);rows=a['rows'];months=a['months'];need(len(rows)==len(triples) and len(months)==len(meta['dates']),'Every source training identity audited')
    official=set(calendar);monthends={d[:7]:d for d in calendar};reasons=Counter();exceptions=[];added_by_year=Counter()
    c=dict(training_rows=len(rows),original_good_rows=0,added_rows=0,extended_good_rows=0,original_center_months=0,original_strict_exceptions=0,new_candidate_rows=0)
    for i,(d,s,e) in enumerate(triples):
        r=rows[i];need((r['row'],r['date'],r['symbol'],r['end'])==(i,d,s,e),'Exact training row identity')
        need(d in official and e in official and monthends[d[:7]]==d and monthends[e[:7]]==e,'Scheduled complete month endpoints')
        for k in ('original_eligible','original_good','added','original_strict_exception'):need(type(r[k]) is bool,'Exact boolean row field')
        strict=r['strict_source_valid'];need(strict is None or type(strict) is bool,'Strict source audit')
        raw=r['raw_targets'];need(raw is None or (set(raw)==set(TARGETS) and all(finite(v) for v in raw.values())),'Four finite saved raw transforms')
        need(r['vol63'] is None or finite(r['vol63']),'Saved volatility')
        if strict:need(raw is not None and r['vol63']>0,'Strict source success requires raw targets and positive volatility')
        if r['original_good']:
            need(r['original_eligible'] and not r['added'] and type(strict) is bool,'Retain original good domain')
            need(r['original_strict_exception'] is (not strict),'Original stricter-path exception marker')
            need(r['reason']==('original_retained' if strict else 'original_retained_exception:'+r['reason'].split(':',1)[-1]),'Original retained reason')
            if not strict:exceptions.append(dict(row=i,date=d,symbol=s,reason=r['reason']))
        elif r['original_eligible']:
            need(r['reason']=='original_eligible_not_admitted' and strict is None and raw is None and not r['added'],'Do not resurrect previously rejected eligible labels')
        else:
            c['new_candidate_rows']+=1;need(not r['original_strict_exception'],'Only old labels have retained exceptions')
            if r['added']:
                need(strict is True and r['reason']=='added','Only complete added labels');added_by_year[d[:4]]+=1
            else:need(r['reason']!='added','Unadded reason')
        c['original_good_rows']+=int(r['original_good']);c['added_rows']+=int(r['added']);reasons[r['reason']]+=1
    for i,m in enumerate(months):
        ids=list(range(i*49,(i+1)*49));orig=[j for j in ids if rows[j]['original_good']]
        need(m['date']==meta['dates'][i] and m['end']==meta['ends'][i] and m['rows']==ids and m['original_center_row_ids']==orig,'Frozen original month-center identities')
        need(m['original_good_rows']==len(orig) and m['added_rows']==sum(rows[j]['added'] for j in ids),'Per-month label counts')
        centers=m['centers'];need(set(centers)==set(TARGETS),'Four original centers')
        if orig:
            need(len(orig)>=2 and all(finite(v) for v in centers.values()),'Original minimum2 center')
            c['original_center_months']+=1
        else:need(all(v is None for v in centers.values()) and m['added_rows']==0,'Added rows may not create a center')
    c['extended_good_rows']=c['original_good_rows']+c['added_rows'];c['original_strict_exceptions']=len(exceptions)
    nonnegative_counts(a['counts'],LABEL_COUNTS);need(c==a['counts'] and dict(reasons)==a['reason_counts'],'Recounted label counts/reasons')
    return dict(counts=c,reasons=dict(reasons),original_strict_exceptions=exceptions,added_rows_by_signal_year=dict(added_by_year))


def main(completion_sha,plan_sha):
    output=OUT/'RESEARCH_REVIEW.json';need(not output.exists(),'Source review output is write-once')
    evidence=Evidence();evidence.pin(Path(__file__))
    closure=evidence.load(OUT/'COMPLETION.json',completion_sha);plan=evidence.load(OUT/'PLAN.json',plan_sha)
    need(closure['status']=='SOURCE_COMPLETED_INDEPENDENTLY_VERIFIED' and closure['plan_sha256']==plan_sha,'Terminal source closure/PLAN')
    need(closure['source_only'] is True and closure['observers_modified'] is False and closure['original_strategy_unchanged'] is True and closure['new_portfolio_returns'] is False and closure['strategy_trials']==0,'Source-only terminal scope')
    expected_runtime=dict(python=platform.python_version(),numpy=version('numpy'),pandas=version('pandas'))
    need(expected_runtime==dict(python='3.9.6',numpy='2.0.2',pandas='2.3.3')==plan['runtime'],'Pinned offline runtime')
    for p,h in plan['input_sha256'].items():evidence.pin(p,h)
    context=evidence.load(CONTEXT,CONTEXT_SHA);need(context==plan['contract'],'Frozen source contract')
    source=evidence.load(SOURCE,SOURCE_SHA);tests=evidence.load(TESTS,TESTS_SHA)
    for p in (CONTEXT,SOURCE,TESTS):need(registered(plan,p)==sha(p),'Registered source/context/test bytes')
    need(source['passed'] is True and source['metadata_only'] is True and source['context_sha256']==CONTEXT_SHA and source['strict_pit'] is False,'Source admission metadata scope')
    for k in ('source_npz_decoded','training_npz_decoded','npz_decoded','source_code_imported','actual_new_features','actual_new_forecasts','actual_new_labels','actual_new_support_counts','actual_new_model_fits','actual_new_orders','actual_new_returns','actual_source_coverage_recomputed'):need(source[k] is False,'Prerequisite source-only flag '+k)
    for p,h in source['input_sha256'].items():need(registered(plan,p)==h,'Source closure registered');evidence.pin(p,h)
    need(source['raw_source_hashes']==plan['raw_source_hashes'],'Registered raw source map')
    need(tests['passed'] is True and tests['synthetic_only'] is True and tests['tests']==65 and tests['runtime']==plan['runtime'],'Formal65 synthetic tests')
    code_expected={str(ROOT/'work'/p) for p in NUMERIC_FILES};need(set(tests['code_sha256'])==code_expected,'Exact six numerical files')
    for p,h in tests['code_sha256'].items():need(registered(plan,p)==h,'Formal code pin');evidence.pin(p,h)
    for k in ('actual_new_queries_or_labels_before_registration','actual_new_fits_before_registration','actual_new_portfolio_returns_before_registration'):need(plan[k]==0,'Prospective registration scope '+k)
    need(context['source_only'] is True and context['strategy_trials']==0 and context['portfolio_returns'] is False and context['observer_writes'] is False and context['original_strategy_changed'] is False,'Registered source-only0trials')
    before=evidence.load(OUT/'STATE_BEFORE.json',plan['prior_central_sha256'])
    need(len(before['completed'])==224 and sum(r['trials'] for r in before['completed'])==1189 and not before['registered_queue'],'Before strategy history')
    prior=evidence.load(PRIOR/'COMPLETION.json',PRIOR_SHA);need(registered(plan,PRIOR/'COMPLETION.json')==PRIOR_SHA,'Predecessor binding')
    manifest=evidence.load(PROTECTED,registered(plan,PROTECTED));need(len(manifest)==plan['protected_files']==closure['protected_files']==21444,'Full protected21444 closure evidence')
    prepared=evidence.load(OUT/'prepared/COMPLETION.json',closure['prepared_sha256'])
    need(prepared['status']=='COMPLETE' and prepared['plan_sha256']==plan_sha and prepared['source_only'] is True and prepared['portfolio_returns'] is False and prepared['strategy_trials']==0 and set(prepared['artifacts_sha256'])==PREPARED,'Exact source-only prepared5 manifest')
    for name,h in prepared['artifacts_sha256'].items():evidence.pin(OUT/'prepared'/name,h)
    meta=evidence.load(OUT/'prepared/META.json');audit=evidence.load(OUT/'prepared/SOURCE_AUDIT.json');checks=evidence.load(OUT/'prepared/INDEPENDENT.json')
    need(len(meta['monthly']['dates'])==122 and len(meta['training']['dates'])==180,'Original122/180 metadata')
    axis=query_axis(meta['monthly']);need(len(axis)==5978 and len(query_axis(meta['training']))==8820,'Original5978/8820 identities')
    for k in ('original_array_bytes_unchanged','no_new_orders','no_new_portfolio_returns'):need(audit[k] is True,'Source verification '+k)
    need(audit['strict_PIT'] is False,'Conditional research source only')
    need(closure['forecasts']==audit['forecasts'] and closure['labels']==audit['labels'],'Closure carries exact prepared audit')
    calendar=evidence.load(CALENDAR,registered(plan,CALENDAR))['sessions'];need(calendar==sorted(set(calendar)),'Original calendar order')
    fc=forecast_audit(audit['forecasts'],axis);lc=label_audit(audit['labels'],meta['training'],calendar)
    need(set(checks)=={'forecasts','labels'},'Both independent audits required')
    for name,flags,counts in (('forecasts',FORECAST_FLAGS,fc),('labels',LABEL_FLAGS,lc['counts'])):
        r=checks[name]
        for k in flags:need(r.get(k) is True,'Independent flag absent: '+name+'/'+k)
        need(r['counts']==counts and r['numerical_tolerance']==0. and r['producer_modules_imported'] is False and r['computes_returns'] is False,'Independent evidence scope/counts')
    need(checks['forecasts']['shared_numerical_primitives'] is True and checks['labels']['original_reconstruction_atol']==1e-12 and checks['labels']['computes_labels'] is True,'Honest independent numerical scope')
    fp=audit['forecast_prefixes'];need([r['prefix'] for r in fp]==list(PREFIXES),'Five physical query prefixes')
    fp_counts=[]
    for r in fp:
        n=r['prefix'];q=n*49;need(r['query_rows']==q,'Prefix query count')
        cutoff=f"{int(meta['monthly']['dates'][n-1][:4])-1}-12-31" if n else None
        ends=[e for e in meta['training']['ends'] if cutoff is not None and e<=cutoff]
        need(r['training_rows']==49*len(ends) and r['latest_label_end']==max(ends,default=None),'Physical training maturity prefix')
        c=forecast_audit(r['audit'],axis[:q]);fp_counts.append(dict(prefix=n,counts=c))
        for domain in ('old_replay','new_replay'):
            need(r['audit'][domain]['local']==[z for z in audit['forecasts'][domain]['local'] if z['event_row']<q],'Prefix local numerical audit exact')
        need(r['audit']['expansion_allowed']==audit['forecasts']['expansion_allowed'][:q] and r['audit']['new_query_mask']==audit['forecasts']['new_query_mask'][:q],'Prefix query masks exact')
    lp=audit['label_prefixes'];need([r['cutoff'] for r in lp]==list(CUTOFFS),'Three physical label cutoffs')
    lp_counts=[]
    for r in lp:
        selected=[i for i,e in enumerate(meta['training']['ends']) if e<=r['cutoff']];n=len(selected);count=n*49
        need(selected==list(range(n)) and r['rows']==count,'Physical full-month label prefix')
        tm={k:meta['training'][k][:n] for k in ('dates','ends','symbols')}
        proof=label_audit(r['audit'],tm,calendar);lp_counts.append(dict(cutoff=r['cutoff'],counts=proof['counts']))
        need(r['audit']['rows']==audit['labels']['rows'][:count] and r['audit']['months']==audit['labels']['months'][:n],'Label audit exact prefix')
    central=read_json(CENTRAL);central_sha=sha(CENTRAL)
    need(len(central['completed'])==224 and sum(r['trials'] for r in central['completed'])==1189 and not central['registered_queue'],'Source-only close keeps strategy count224/1189 and empty queue')
    need(central['completed']==before['completed'] and central['best_training_selected_candidate']==before['best_training_selected_candidate'] and central['target_full_period_cagr']==before['target_full_period_cagr']==.7,'Source work must not alter strategy history/best')
    matches=[r for r in central.get('source_experiments',[]) if r['directory']==OUT.name]
    need(len(matches)==1 and matches[0]['strategy_trials']==0 and matches[0]['completion']['sha256']==completion_sha and resolved(matches[0]['completion']['path'])==OUT/'COMPLETION.json','Central source-only completion reference')
    prior_review=evidence.load(PRIOR/'RESEARCH_REVIEW.json',PRIOR_REVIEW_SHA)
    reference=prior_review['metrics']['incumbent_local67']['same_close_zero']['full122']
    need(reference['cagr']==.6766617372041686 and reference['cagr']<central['target_full_period_cagr'],'Prior incumbent reference or target changed')
    fitting=dict(main=fc,independent=checks['forecasts']['counts'],physical_prefix_replays=fp_counts,
        physical_prefix_regression_solves=sum(x['counts']['actual_regression_solves'] for x in fp_counts))
    fitting['total_registered_prepare_regression_solves']=fc['actual_regression_solves']+checks['forecasts']['counts']['actual_regression_solves']+fitting['physical_prefix_regression_solves']
    report=dict(schema='source227_saved_result_review_v1',created_utc=datetime.now(timezone.utc).isoformat(),passed=True,
        plan_sha256=plan_sha,completion_sha256=completion_sha,prepared_sha256=closure['prepared_sha256'],source_only=True,strategy_trials=0,
        original_strategy_unchanged=True,observers_modified=False,portfolio_returns_computed=False,source_npz_decoded=False,models_imported=False,
        new_labels_or_features_computed=False,new_model_fits_executed_by_reader=0,central_modified=False,strict_PIT=False,
        runtime=expected_runtime,formal_tests=65,input_sha256=evidence.pins,independent_flags=dict(forecasts=list(FORECAST_FLAGS),labels=list(LABEL_FLAGS)),
        forecasts=fc,labels=lc,fit_work_accounting=fitting,label_physical_prefixes=lp_counts,forecast_physical_prefixes=list(PREFIXES),
        protection=dict(members=21444,manifest_sha256=sha(PROTECTED),registered_and_closed_full_verification_receipts=True,all_members_rehashed_by_reader=False),
        central_snapshot=dict(sha256=central_sha,completed_strategies=224,trials=1189,queue_empty=True,source_only_completion_registered=True),
        prior_incumbent_reference=dict(source='outputs/monthly_joint_error226/RESEARCH_REVIEW.json',path='same_close_zero',transaction_cost_per_side=0.,
            metrics=reference,new_source227_performance_result=False,target70_reached=False),
        limitations=context['limitations']+['此审查只重数保存的身份、审计与拟合次数；预测/标签数值一致性依赖已绑定的独立准备回执，未重新解码NPZ。',
          '独立预测审核共享原冻结数值求解原语，独立重建年度样本和逐查询控制流；不是第二套独立线性代数求解器。',
          '新增标签只按原月份中心计算，未进入原训练；新增数据数量不代表策略收益改善。'])
    # Recheck mutable snapshot and immutable closing receipts immediately before
    # the sole new write. Existing evidence is never modified by this reader.
    need(sha(CENTRAL)==central_sha and sha(OUT/'COMPLETION.json')==completion_sha and sha(OUT/'PLAN.json')==plan_sha,'Evidence changed during saved review')
    with output.open('x') as f:f.write(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(passed=True,review_sha256=sha(output),forecast_counts=fc,label_counts=lc['counts'],actual_prepare_regression_solves=fitting['total_registered_prepare_regression_solves'],source_only=True),ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--completion-sha256',required=True);parser.add_argument('--plan-sha256',required=True)
    args=parser.parse_args();main(args.completion_sha256,args.plan_sha256)
