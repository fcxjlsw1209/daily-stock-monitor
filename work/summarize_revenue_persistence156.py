"""Publish only verified round156 results and advance local resume notes."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import tempfile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/monthly_revenue_persistence156'
NAMES=('incumbent_local67','matched_latest_revenue_growth','matched_annualized_total_revenue_growth','weakest_annual_revenue_growth')
LABELS=('原策略','最近一年收入增长','三年年化收入增长','三年最弱增长')
MODES=('same_close','next_open_zero','next_open_cost')


def read(p):return json.loads(p.read_bytes())
def encode(v):return (json.dumps(v,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n').encode()
def ref(p):return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def require(ok,message):
    if not ok:raise ValueError(message)


def validate_result_contract(done,state,results,orders,audit,frozen,meta,plan_sha,completion_ref):
    """Validate the published selected result; never select the best diagnostic arm."""
    require(done.get('round')==156 and done.get('directory')=='monthly_revenue_persistence156'
        and done.get('plan_sha256')==plan_sha and done.get('status')=='completed_independently_verified',
        'Exact completed round156 required')
    for key,value in (('independent_execution_paths',12),('protected_files',21444),('trials',4),('new_model_fits',0)):
        require(type(done.get(key)) is int and done[key]==value,'Completion scope mismatch: '+key)
    completed=state.get('completed',[])
    require(len(completed)==156 and all(type(row.get('trials')) is int for row in completed)
        and sum(row['trials'] for row in completed)==982 and state.get('registered_queue')==[],
        'Final156/982 with empty queue required')
    require(state.get('current')=='monthly_revenue_persistence156'
        and state.get('current_status')=='completed_independently_verified'
        and completed[-1].get('directory')=='monthly_revenue_persistence156'
        and completed[-1].get('completion')==completion_ref and state.get('latest_round_receipt')==completion_ref,
        'Continuation does not bind this completion')
    require(set(results)==set(NAMES) and results==done.get('all_arm_results')
        and all(set(results[name])==set(MODES) for name in NAMES),'Exact four-arm/twelve-path results required')
    require(audit.get('passed') is True and audit.get('stage')=='evaluate' and audit.get('plan_sha256')==plan_sha
        and audit.get('producer_modules_imported') is False,'Bound independent outcome required')
    for key,value in (('execution_paths_verified',12),('total_monthly_orders_verified',488),('models_fit',0)):
        require(type(audit.get(key)) is int and audit[key]==value,'Independent scope mismatch: '+key)
    primary=done.get('primary');selection=done.get('selection',{})
    require(primary in (NAMES[0],NAMES[3]) and primary==selection.get('primary')==audit.get('selected_name')
        and frozen.get('selection')==selection and type(selection.get('admitted')) is bool
        and selection['admitted']==(primary==NAMES[3]),'Frozen first88 primary mismatch')
    metric=results[primary]['same_close']['metrics']
    for key in ('cagr','sharpe','max_drawdown'):
        require(type(metric.get(key)) in (int,float) and math.isfinite(metric[key]),'Finite selected metrics required')
    require(done.get('primary_cagr')==metric['cagr'] and done.get('primary_sharpe')==metric['sharpe']
        and type(audit.get('primary_cagr')) in (int,float) and math.isfinite(audit['primary_cagr'])
        and abs(audit['primary_cagr']-metric['cagr'])<1e-12,'Selected metrics disagree')
    target=metric['cagr']>=.7
    require(type(done.get('target_reached')) is bool and done['target_reached']==target
        and type(audit.get('primary_target_reached')) is bool and audit['primary_target_reached']==target
        and completed[-1].get('primary')==primary and completed[-1].get('target_reached') is target,
        'Target flag must describe only the training-selected primary')
    require(set(orders)==set(NAMES) and len(meta['dates'])==len(meta['ends'])==122
        and meta['dates'][0]=='2016-08-17' and meta['ends'][-1]=='2026-09-24','Original order horizon required')
    for name in NAMES:
        require(len(orders[name])==122,'Truncated order comparison: '+name)
        for i,row in enumerate(orders[name]):
            require(row['date']==row['signal_date']==meta['dates'][i] and row['end']==meta['ends'][i],
                'Misaligned order comparison')
    return target


def main(plan_sha):
    seen={}
    def load(path):
        raw=path.read_bytes();seen[path]=hashlib.sha256(raw).hexdigest();return json.loads(raw)
    def unchanged():
        require(all(hashlib.sha256(path.read_bytes()).hexdigest()==digest for path,digest in seen.items()),
            'Input changed while summarizing')
    plan=load(OUT/'PLAN.json');require(seen[OUT/'PLAN.json']==plan_sha,'Exact PLAN digest required')
    done=load(OUT/'COMPLETION.json');state=load(ROOT/'outputs/monthly_target70_research/CONTINUATION_STATUS.json')
    results=load(OUT/'RESULTS.json');orders=load(OUT/'PRETRAIN_ORDERS.json');audit=load(OUT/'INDEPENDENT_OUTCOME.json')
    frozen=load(OUT/'FROZEN.json');meta=load(OUT/'META.json');protected=load(OUT/'PROTECTED_AFTER.json')
    target=validate_result_contract(done,state,results,orders,audit,frozen,meta,plan_sha,ref(OUT/'COMPLETION.json'))
    require(seen[OUT/'INDEPENDENT_OUTCOME.json']==done['independent_outcome_sha256']
        and seen[OUT/'PROTECTED_AFTER.json']==done['protected_sha256'],'Completion evidence digest mismatch')
    require(audit.get('frozen_sha256')==seen[OUT/'FROZEN.json']
        and audit.get('orders_sha256')==seen[OUT/'PRETRAIN_ORDERS.json'],'Independent frozen/order binding mismatch')
    for name in ('RESULTS.json','PRETRAIN_ORDERS.json','FROZEN.json','META.json'):
        require(audit.get('artifact_hashes',{}).get(str(OUT/name))==seen[OUT/name],'Unaudited summary input: '+name)
    require(protected.get('passed') is True and protected.get('checked')==protected.get('verified')==21444
        and protected.get('changed')==protected.get('missing')==protected.get('read_or_identity_errors')==[],
        'Full protected-file check required')
    source_ref=plan['inputs']['source_manifest'];source_path=Path(source_ref['path'])
    if not source_path.is_absolute():source_path=ROOT/source_path
    manifest=load(source_path);require(seen[source_path]==source_ref['sha256'],'Registered source manifest changed')
    source_coverage=load(source_path.parent/'COVERAGE_GATE.json')
    require(seen[source_path.parent/'COVERAGE_GATE.json']==manifest['outputs']['COVERAGE_GATE.json']['sha256']
        and source_coverage.get('source_coverage_floor_passed') is True,'Registered source coverage changed')
    changes={name:[dict(date=b['date'],incumbent=b['symbols'],candidate=c['symbols'],
        constituent_set_changed=set(b['symbols'])!=set(c['symbols']))
        for b,c in zip(orders[NAMES[0]],orders[name]) if b['symbols']!=c['symbols']] for name in NAMES[1:]}
    summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),round=156,total_trials=982,
        period=['2016-08-17','2026-09-24'],training_months=88,later_diagnostic_months=34,
        primary=done['primary'],selection=done['selection'],target_reached=target,
        results=results,changed_order_months=changes,
        source_coverage=source_coverage,
        plan=ref(OUT/'PLAN.json'),completion=ref(OUT/'COMPLETION.json'),independent=ref(OUT/'INDEPENDENT_OUTCOME.json'),
        protected=ref(OUT/'PROTECTED_AFTER.json'),observer_changes=0,
        limitations=['Sparse fixed-source pilot; source availability remains issuer-specific.',
                     'Source coverage is reported separately by time segment and is not evidence of factor stability.',
                     'Repeated development history; not pristine out-of-sample evidence.',
                     'Same-close zero-cost is an idealized research convention; next-open costs are reported separately.',
                     'Nominal reported revenue and later-retrieved original releases retain stated accounting/vintage limitations.'])
    lines=['# 收入持续性四组对照：第156轮','',
        '2016-08-17至2026-09-24；前88个月选取，后34个月诊断。月末三只、月内固定持仓。','',
        '|组别|同收盘零成本年化|Sharpe|最大回撤|次日开盘每边5bps年化|成分改变月份|',
        '|---|---:|---:|---:|---:|---:|']
    for name,label in zip(NAMES,LABELS):
        met=results[name]['same_close']['metrics'];op=results[name]['next_open_cost']['metrics']
        n=sum(x['constituent_set_changed'] for x in changes.get(name,[]))
        lines.append(f"|{label}|{met['cagr']:.2%}|{met['sharpe']:.3f}|{met['max_drawdown']:.2%}|{op['cagr']:.2%}|{n}|")
    coverage=source_coverage['segments']['all122']['issuers']
    early=sum(row['usable_rows'] for row in source_coverage['segments']['early53']['issuers'].values())
    joint=len(source_coverage['qualifying_joint_cutoffs'])
    lines+=['',f"预声明选择结果：{done['primary']}；该选择结果达到70%目标：{'是' if target else '否'}。",'',
        f"微软{coverage['MSFT']['usable_rows']}个月、艾伯维{coverage['ABBV']['usable_rows']}个月、苹果{coverage['AAPL']['usable_rows']}个月有可用原始收入链；训练段合格公司共同覆盖{joint}个月。早期53个月共有{early}个可用公司月份。缺失不改变候选资格，而是回到原策略评分。",
        '', '已完成独立准备核对、12条执行路径与488组月度持仓核对，以及21,444个历史保护文件的完整检查。未修改任何冻结观察账户。',
        '', '这些是反复使用历史样本的研究结果，不是未来收益承诺，也不代表已复现作者策略。']
    path=ROOT/'migration/RESEARCH_RESUME.json';raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest();resume=json.loads(raw)
    backup=ROOT/('migration/round156_resume_'+digest+'.json')
    require(not (OUT/'SUMMARY.json').exists() and not (OUT/'SUMMARY.md').exists(),'Summary already published; preserve original')
    unchanged()
    with backup.open('xb') as f:f.write(raw)
    with (OUT/'SUMMARY.json').open('xb') as f:f.write(encode(summary))
    with (OUT/'SUMMARY.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    artifacts={name:ref(OUT/name) for name in ('PLAN.json','INDEPENDENT_PREPARE.json','INDEPENDENT_OUTCOME.json','PROTECTED_AFTER.json','COMPLETION.json','SUMMARY.json')}
    resume.update(round=156,completed_research_rounds=156,strategy_evaluations=982,current_experiment='monthly_revenue_persistence156',
        current_experiment_artifacts=artifacts,formal_round156_completed=True,formal_round156_completion=artifacts['COMPLETION.json'],
        plan_sha256=plan_sha,frozen_primary=done['primary'],latest_protected_check=artifacts['PROTECTED_AFTER.json'],
        full_protection_check_passed=True,remaining_formal_archive_check='None for156;21444 protected files verified',
        target_70_reached=target,next_mechanism_registered=False,
        next_research_status='round156_complete_no_next_mechanism_registered',
        next_source_preparation=dict(status='sparse_msft_abbv_pilot_completed_aapl_pending',checkpoint=artifacts['SUMMARY.json']),
        next_step='Review this fixed four-arm result before declaring a distinct next mechanism. Do not tune sign, dose or source subset to these observed returns. AAPL chain admission and early-period source coverage remain separate source opportunities.',
        last_goal_turn_classification='progress: completed156 including independent twelve-path verification',
        resume_update_checkpoint=str(backup.relative_to(ROOT)),updated_utc=summary['created_utc'])
    with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as f:temp=Path(f.name);f.write(encode(resume))
    unchanged()
    require(hashlib.sha256(path.read_bytes()).hexdigest()==digest,'Concurrent resume update')
    os.replace(temp,path)
    print(json.dumps(dict(primary=done['primary'],cagr=done['primary_cagr'],target_reached=done['target_reached'],summary=ref(OUT/'SUMMARY.json'))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan-sha',required=True);main(p.parse_args().plan_sha)
