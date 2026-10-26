"""Read a closed source229's saved JSON and immutable hashes only.

This reviewer never imports numerical/model modules or decodes NPZ members.
Run only after explicit close authorization; output is one new write-once JSON.
"""
from collections import Counter
from datetime import datetime, timezone, date
from importlib.metadata import version
from pathlib import Path
import argparse
import hashlib
import json
import math
import platform

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/monthly_early_history229'
PRIOR=ROOT/'outputs/monthly_expanded_error228'
SOURCE227=ROOT/'outputs/monthly_ineligible_source227'
CENTRAL=ROOT/'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
CONTEXT=ROOT/'work/monthly_early_history229_context.json'
CONTEXT_SHA='bdcddb7e70b12ede9c42464a3ed0d7be230ec7b467720b0262c3358b2126670f'
SOURCE=ROOT/'work/early_history229_SOURCE_REVIEW.json'
SOURCE_SHA='5fa5643c4e390f4b5faab87d2363a5ed73a2a75f8d6eca9abb8cfc8ad72298d9'
TESTS=ROOT/'work/monthly_early_history229_TESTS.json'
CODE_REVIEW=ROOT/'work/early_history229_CODE_REVIEW.json'
PRIOR_SHA='8d65e817a0c34e79950b957f3263d651b792fbaf6c128681a98f1be1ed59df7f'
PRIOR_REVIEW_SHA='0c04840c9199d4e60419872e548bf62ce7f8505b373d5b922322952c5ac21952'
PROTECTED=ROOT/'work/trade_reconcile_20261015_close/NEXT_EXPECTED_PROTECTED.json'
CALENDAR=ROOT/'outputs/monthly_paired_target143/CALENDAR.json'
NUMERIC_FILES=('monthly_early_history229_host.py','test_monthly_early_history229_host.py',
 'monthly_early_history229_model.py','test_monthly_early_history229_model.py',
 'monthly_early_history229_independent.py','test_monthly_early_history229_independent.py')
PREPARED={'FORECASTS.npz','LABELS.npz','META.json','SOURCE_AUDIT.json','INDEPENDENT.json'}
EXTERNAL={'/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/bin/python3.9':'6b24be8ba00c4abc8b3e7e2620bd710e2cf7a4312d00c147f8f05fbb6f3d297a'}
TARGETS=('simple_excess','log_excess','vol_scaled','loss_utility')
FORECAST_COUNTS=('query_rows','original_query_rows','expansion_requested_rows','union_requested_rows',
 'original_available_rows','added_available_rows','available_rows','local_available_rows',
 'source_target_fits','pooled_log_refits','local_evaluations','local_solves','actual_regression_solves')
FORECAST_FLAGS=('passed','independent_annual_control_flow','independent_augmented_least_squares',
 'same_training_cohort','four_targets_unchanged','annual_prior_december_cutoffs',
 'five_year_mature_window','minimum_18_months','same_security_local_history',
 'local_history_not_query_filter','early_query_date_bound','exact_query_identity',
 'original_eligibility_preserved','expansion_warmup_mask_enforced','union_query_mask_checked',
 'no_future_label_query_filter','no_query_rerank','fit_counts_checked','audit_checked',
 'input_bytes_unchanged_by_checker')
PREFIXES=(0,12,24,48,67)
INDEPENDENT_PRIMITIVE='work/monthly_identity_transfer206_independent.py'
INDEPENDENT_PRIMITIVE_SHA='a321b2cd07eb2b19c08f7747842e18a47438d781bed9b474c8ef321ec9f5611e'


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
    need(a['method']=='early_original_query_history_extension' and a['query_rows']==len(axis) and a['feature_count']==10,'Forecast method/shape')
    need(a['raw_keys']==['source_log','local_constant','local_linear'] and a['query_date_exclusive_upper_bound']=='2016-08-17','Three raw early-query forecasts')
    need(all(d<'2016-08-17' for d,_,_ in axis),'No current trading queries in early source')
    need(a['query_qualification']=='finite cached plain10 AND (original train_eligible OR caller-verified 295-price warmup)','Query eligibility contract')
    for k in ('query_future_labels_used','training_targets_changed','training_common_good_changed','new_features_computed'):
        need(a[k] is False,'Unchanged source scope '+k)
    need(a['annual_forecasts_calls']==1,'Single union-query replay')
    masks={k:a[k] for k in ('original_query_mask','expansion_query_mask','union_query_mask','expansion_allowed')}
    for k,v in masks.items():need(len(v)==len(axis) and all(type(x) is bool for x in v),'Exact query mask '+k)
    old,added,union,allowed=[masks[k] for k in ('original_query_mask','expansion_query_mask','union_query_mask','expansion_allowed')]
    need(all(not(o and n) and u==(o or n) and (not n or w) for o,n,u,w in zip(old,added,union,allowed)),'Disjoint signal-time query union')
    r=replay_audit(a['replay_audit'],axis,'union')
    need(all(union[j] for j in r['event_rows']),'Only union-qualified queries forecast')
    for ar in r['annual']:
        need(ar['prediction_rows']==sum(u and int(d[:4])==ar['year'] for u,(d,_,_) in zip(union,axis)),'Year request count')
    c=dict(query_rows=len(axis),original_query_rows=sum(old),expansion_requested_rows=sum(added),union_requested_rows=sum(union),
        original_available_rows=sum(old[j] for j in r['event_rows']),added_available_rows=sum(added[j] for j in r['event_rows']),
        available_rows=r['available_rows'],local_available_rows=sum(x['weight_mass']>0. for x in a['replay_audit']['local']),
        source_target_fits=r['source_target_fits'],pooled_log_refits=r['pooled_log_refits'],
        local_evaluations=r['local_evaluations'],local_solves=r['local_solves'])
    c['actual_regression_solves']=c['source_target_fits']+c['pooled_log_refits']+c['local_solves']
    nonnegative_counts(a['counts'],FORECAST_COUNTS);need(c==a['counts'],'Recounted early-query and fit counts')
    return c


def exact_early_metadata(history,original,calendar):
    need(len(history['dates'])==len(history['ends'])==len(history['symbols'])==67,'Early67 metadata')
    expected={k:(v[:67] if isinstance(v,list) and len(v)==len(original['dates']) else v) for k,v in original.items()}
    need(history==expected,'Exact original early metadata prefix')
    axis=query_axis(history);need(len(axis)==3283,'Early67x49 identities')
    need(history['dates'][0]=='2011-01-31' and history['dates'][-1]=='2016-07-29' and history['ends'][-1]=='2016-08-31','Original early endpoints')
    need(history['dates']==sorted(set(history['dates'])),'Unique early signals')
    official=set(calendar);monthends={d[:7]:d for d in calendar}
    for d,_,e in axis:
        need(date.fromisoformat(d).isoformat()==d and date.fromisoformat(e).isoformat()==e,'Canonical ISO metadata')
        need(d in official and e in official and monthends[d[:7]]==d and monthends[e[:7]]==e,'Official signal/label monthends')
        y,m=map(int,d[:7].split('-'));need(e[:7]==('%04d-%02d'%(y+(m==12),m%12+1)),'Next natural month labels')
    return axis


def main(completion_sha,plan_sha,test_count):
    output=OUT/'RESEARCH_REVIEW.json';need(not output.exists(),'Source review output is write-once')
    need(type(test_count) is int and test_count>0,'Explicit positive formal test count')
    evidence=Evidence();evidence.pin(Path(__file__))
    closure=evidence.load(OUT/'COMPLETION.json',completion_sha);plan=evidence.load(OUT/'PLAN.json',plan_sha)
    need(closure['status']=='SOURCE_COMPLETED_INDEPENDENTLY_VERIFIED' and closure['plan_sha256']==plan_sha,'Terminal source closure/PLAN')
    need(closure['source_only'] is True and closure['observers_modified'] is False and closure['original_strategy_unchanged'] is True and closure['new_portfolio_returns'] is False and closure['strategy_trials']==0,'Source-only terminal scope')
    runtime=dict(python=platform.python_version(),numpy=version('numpy'),pandas=version('pandas'))
    need(runtime==dict(python='3.9.6',numpy='2.0.2',pandas='2.3.3')==plan['runtime'],'Pinned offline runtime')
    for p,h in plan['input_sha256'].items():evidence.pin(p,h)
    context=evidence.load(CONTEXT,CONTEXT_SHA);source=evidence.load(SOURCE,SOURCE_SHA)
    need(context==plan['contract'],'Registered immutable contract')
    for p in (CONTEXT,SOURCE,TESTS,CODE_REVIEW):need(registered(plan,p)==sha(p),'Registered review/test/context bytes')
    need(source['passed'] is True and source['metadata_only'] is True and source['source_only'] is True and source['context_sha256']==CONTEXT_SHA and source['strict_PIT'] is False,'Metadata-only conditional source evidence')
    for k in ('source_npz_decoded','training_npz_decoded','numerical_source_code_imported','actual_new_forecasts','actual_new_features','actual_new_labels','actual_new_support_counts','actual_new_model_fits','actual_new_orders','actual_new_returns','frozen_files_modified','central_modified','observers_modified'):
        need(source[k] is False,'Source metadata scope '+k)
    need(source['source_bytes_reverified'] is True and source['future_label_validity_must_not_gate_queries'] is True and source['original_training_common_good_and_four_targets_unchanged'] is True,'Source identity/training/clock proof')
    need(source['raw_source_hashes']==plan['raw_source_hashes'] and len(source['raw_source_hashes'])==64,'Original64 raw source map')
    for p,h in source['input_sha256'].items():need(registered(plan,p)==h,'Source input bound');evidence.pin(p,h)
    for p,h in plan['raw_source_hashes'].items():evidence.pin(p,h)
    tests=evidence.load(TESTS,registered(plan,TESTS));review=evidence.load(CODE_REVIEW,registered(plan,CODE_REVIEW))
    need(tests['passed'] is True and tests['synthetic_only'] is True and tests['tests']==test_count and tests['runtime']==runtime,'Formal synthetic tests/runtime')
    expected_files={str(ROOT/'work'/p) for p in NUMERIC_FILES}
    need(set(tests['code_sha256'])==expected_files and review['passed'] is True and review['unresolved_blockers']==[] and review['code_sha256']==tests['code_sha256'] and review['context_sha256']==CONTEXT_SHA,'Exact six-code review/test closure')
    for p,h in tests['code_sha256'].items():need(registered(plan,p)==h,'Formal code registered');evidence.pin(p,h)
    need(registered(plan,ROOT/INDEPENDENT_PRIMITIVE)==INDEPENDENT_PRIMITIVE_SHA,'Independent augmented-LS primitive registered')
    evidence.pin(ROOT/INDEPENDENT_PRIMITIVE,INDEPENDENT_PRIMITIVE_SHA)
    for k in ('actual_new_queries_forecast_before_registration','actual_new_fits_before_registration','actual_portfolio_returns_before_registration'):
        need(type(plan[k]) is int and plan[k]==0,'Prospective source registration '+k)
    need(context['source_only'] is True and context['strategy_trials']==0 and context['new_portfolio_returns'] is False and context['new_ranks'] is False and context['new_labels'] is False and context['strict_PIT'] is False and context['observer_writes'] is False and context['original_strategy_changed'] is False and context['parameter_search'] is False,'Fixed source-only no strategy/label/return scope')
    need(context['prefixes']==list(PREFIXES),'Registered five prefixes')
    before=evidence.load(OUT/'STATE_BEFORE.json',plan['prior_central_sha256'])
    need(len(before['completed'])==225 and sum(r['trials'] for r in before['completed'])==1192 and before['completed'][-1]['directory']==PRIOR.name and not before['registered_queue'],'Source registration predecessor225/1192')
    evidence.load(PRIOR/'COMPLETION.json',PRIOR_SHA)
    need(registered(plan,PRIOR/'COMPLETION.json')==PRIOR_SHA,'Predecessor close binding')
    manifest=evidence.load(PROTECTED,registered(plan,PROTECTED))
    need(len(manifest)==plan['protected_files']==closure['protected_files']==21444,'Full21444 registered/close protection')
    source227=source['source227'];prepared227=evidence.load(SOURCE227/'prepared/COMPLETION.json',source227['prepared_completion_sha256'])
    need(prepared227['artifacts_sha256']==source227['artifacts_sha256'] and set(prepared227['artifacts_sha256'])==PREPARED,'Source227 exact five source artifacts')
    for name,h in prepared227['artifacts_sha256'].items():evidence.pin(SOURCE227/'prepared'/name,h)
    original_meta=evidence.load(SOURCE227/'prepared/META.json',source227['artifacts_sha256']['META.json'])
    need(len(original_meta['training']['dates'])==180 and len(original_meta['monthly']['dates'])==122,'Original180training/122trading metadata')
    prepared=evidence.load(OUT/'prepared/COMPLETION.json',closure['prepared_sha256'])
    need(prepared['status']=='COMPLETE' and prepared['plan_sha256']==plan_sha and prepared['source_only'] is True and prepared['portfolio_returns'] is False and prepared['strategy_trials']==0 and set(prepared['artifacts_sha256'])==PREPARED,'Exact source-only prepared5 manifest')
    for name,h in prepared['artifacts_sha256'].items():evidence.pin(OUT/'prepared'/name,h)
    meta=evidence.load(OUT/'prepared/META.json');audit=evidence.load(OUT/'prepared/SOURCE_AUDIT.json');check=evidence.load(OUT/'prepared/INDEPENDENT.json')
    need(set(meta)=={'history','current_query_axis_unchanged','training_reference'} and meta['current_query_axis_unchanged'] is True,'Separate historical query axis')
    reference=evidence.load(meta['training_reference'],registered(plan,meta['training_reference']))
    need(reference==original_meta,'Exact original TRAIN metadata reference')
    calendar=evidence.load(CALENDAR,registered(plan,CALENDAR))['sessions'];need(calendar==sorted(set(calendar)),'Sorted original sessions')
    axis=exact_early_metadata(meta['history'],original_meta['training'],calendar)
    for k in ('original_arrays_unchanged','copied_label_exact','no_new_orders','no_portfolio_returns'):need(audit[k] is True,'Prepared source scope '+k)
    need(audit['copied_label_rows']==3283 and audit['source227_label_sha256']==source227['artifacts_sha256']['LABELS.npz'] and audit['original_trading_queries']==122 and audit['new_history_months']==67 and audit['strict_PIT'] is False,'Only exact early227 risk-outcome copy')
    need(closure['forecasts']==audit['forecasts'],'Closure prepared forecast audit exact')
    fc=forecast_audit(audit['forecasts'],axis)
    for flag in FORECAST_FLAGS:need(check.get(flag) is True,'Independent flag missing: '+flag)
    need(check['counts']==fc and check['query_rows']==3283 and check['numerical_tolerance']==1e-12 and check['producer_modules_imported'] is False and check['computes_returns'] is False and check['shared_numerical_primitives'] is False,'Independent numerical scope/counts')
    need(check['independent_primitive_path']==INDEPENDENT_PRIMITIVE and check['independent_primitive_sha256']==INDEPENDENT_PRIMITIVE_SHA,'Independent solver primitive binding')
    prefix_counts=[];fp=audit['forecast_prefixes']
    need([r['prefix'] for r in fp]==list(PREFIXES),'Five exact physical prefixes')
    for r in fp:
        n=r['prefix'];q=n*49;need(r['query_rows']==q,'Prefix67x49 row count')
        cut=f"{int(meta['history']['dates'][n-1][:4])-1}-12-31" if n else None
        ends=[e for e in original_meta['training']['ends'] if cut is not None and e<=cut]
        need(r['training_rows']==49*len(ends) and r['latest_label_end']==max(ends,default=None),'Physical label-maturity truncation')
        counts=forecast_audit(r['audit'],axis[:q]);prefix_counts.append(dict(prefix=n,counts=counts))
        for key in ('original_query_mask','expansion_query_mask','union_query_mask','expansion_allowed'):
            need(r['audit'][key]==audit['forecasts'][key][:q],'Exact prefix mask '+key)
        need(r['audit']['replay_audit']['local']==[z for z in audit['forecasts']['replay_audit']['local'] if z['event_row']<q],'Exact saved local audit prefix')
        for k in ('source_target_fits','pooled_log_refits','local_solves','actual_regression_solves'):
            need(counts[k]<=fc[k],'Prefix does not create later-fit counts '+k)
    main_annual=audit['forecasts']['replay_audit']['source']['annual']
    need([r['year'] for r in main_annual]==list(range(2011,2017)) and all(r['status']=='cold_start' for r in main_annual[:2]),'Early2011/2012 source cold starts retained')
    central=read_json(CENTRAL);central_sha=sha(CENTRAL)
    need(len(central['completed'])==225 and sum(r['trials'] for r in central['completed'])==1192 and not central['registered_queue'],'Source-only central225/1192 no strategy trials')
    need(central['completed']==before['completed'] and central['best_training_selected_candidate']==before['best_training_selected_candidate'] and central['best_training_selected_candidate']['directory']=='monthly_local_linear70' and central['target_full_period_cagr']==before['target_full_period_cagr']==.7,'Original incumbent/history/target unchanged')
    matches=[r for r in central.get('source_experiments',[]) if r['directory']==OUT.name]
    need(len(matches)==1 and matches[0]['strategy_trials']==0 and matches[0]['completion']['sha256']==completion_sha and resolved(matches[0]['completion']['path'])==OUT/'COMPLETION.json','Central source229 closure')
    need(central.get('source_experiments',[])==before.get('source_experiments',[])+matches,'Only one new source experiment')
    prior_review=evidence.load(PRIOR/'RESEARCH_REVIEW.json',PRIOR_REVIEW_SHA)
    incumbent=prior_review['metrics']['incumbent_local67']['same_close_zero']['full122']
    need(incumbent['cagr']==.6766617372041686 and incumbent['cagr']<.7,'Unchanged zero-cost incumbent reference')
    annual_support=[dict(year=r['year'],training_months=r['training_months'],requested=r['prediction_rows'],
        available=sum(z['year']==r['year'] for z in audit['forecasts']['replay_audit']['local']),status=r['status']) for r in main_annual]
    fitting=dict(main=fc,independent=check['counts'],physical_prefix_replays=prefix_counts,
        physical_prefix_regression_solves=sum(x['counts']['actual_regression_solves'] for x in prefix_counts))
    fitting['total_registered_prepare_regression_solves']=fc['actual_regression_solves']+check['counts']['actual_regression_solves']+fitting['physical_prefix_regression_solves']
    report=dict(schema='source229_saved_result_review_v1',created_utc=datetime.now(timezone.utc).isoformat(),passed=True,
        plan_sha256=plan_sha,completion_sha256=completion_sha,prepared_sha256=closure['prepared_sha256'],source_only=True,strategy_trials=0,
        original_strategy_unchanged=True,observers_modified=False,portfolio_returns_computed=False,source_npz_decoded=False,models_imported=False,
        new_labels_or_features_computed=False,new_model_fits_executed_by_reader=0,central_modified=False,strict_PIT=False,
        runtime=runtime,formal_tests=test_count,input_sha256=evidence.pins,independent_flags=list(FORECAST_FLAGS),
        forecasts=fc,annual_support=annual_support,fit_work_accounting=fitting,forecast_physical_prefixes=list(PREFIXES),
        labels=dict(copied_original_early_rows=3283,source227_sha256=audit['source227_label_sha256'],
            exact_copy_verified_by_registered_host=True,labels_recomputed_by_reader=False,used_in_source_training=False),
        protection=dict(members=21444,manifest_sha256=sha(PROTECTED),registered_and_closed_full_verification_receipts=True,all_members_rehashed_by_reader=False),
        central_snapshot=dict(sha256=central_sha,completed_strategies=225,trials=1192,queue_empty=True,source_only_completion_registered=True),
        prior_incumbent_reference=dict(source='outputs/monthly_expanded_error228/RESEARCH_REVIEW.json',path='same_close_zero',transaction_cost_per_side=0.,
            metrics=incumbent,new_source229_performance_result=False,target70_reached=False),
        limitations=context['limitations']+['只重数已保存身份、年度审计和拟合次数，预测数值依赖绑定的独立增广最小二乘准备回执，未解码NPZ。',
          'source227标签只复制为后续风险误差的结果；没有新建标签，也没有进入本轮原模型训练。',
          '新增历史查询数量不证明可用共同误差样本增多或策略收益改善，后续使用需单独预注册。'])
    need(sha(CENTRAL)==central_sha and sha(OUT/'COMPLETION.json')==completion_sha and sha(OUT/'PLAN.json')==plan_sha,'Evidence changed during review')
    with output.open('x') as f:f.write(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(passed=True,review_sha256=sha(output),forecast_counts=fc,source_only=True,
        actual_prepare_regression_solves=fitting['total_registered_prepare_regression_solves']),ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--completion-sha256',required=True)
    parser.add_argument('--plan-sha256',required=True)
    parser.add_argument('--tests',type=int,required=True)
    args=parser.parse_args();main(args.completion_sha256,args.plan_sha256,args.tests)
