"""Publish a human-readable summary only from completed, verified round158."""
from datetime import datetime, timezone
from pathlib import Path
import json

import annual_lag158_repair as repair

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / repair.OUT
AMENDMENT_SHA = 'a8ba1ea3047f20d70448e00fae6f967473c70dd6fe73a880aef87c5b396064c1'


def main():
    repair.verify_amendment(ROOT, repair.PLAN_SHA, AMENDMENT_SHA)
    done = repair.read(OUT/'COMPLETION.json')
    closing = repair.read(OUT/'REPAIR_COMPLETION_01.json')
    proof = repair.read(OUT/'PROTECTED_AFTER.json')
    result = repair.read(OUT/'RESULTS.json')
    feature = repair.read(OUT/'FEATURE_AUDIT.json')
    outcome = repair.read(OUT/'INDEPENDENT_OUTCOME.json')
    repair.verify_repair_report(ROOT, outcome, repair.PLAN_SHA, AMENDMENT_SHA)
    repair.require(done['status'] == 'completed_independently_verified' and done['round'] == 158,
                   'Completed round required')
    for key, name in (('original_completion','COMPLETION.json'),('protected','PROTECTED_AFTER.json'),
                      ('independent_outcome','INDEPENDENT_OUTCOME.json')):
        repair.require(closing[key] == repair.reference(ROOT, repair.OUT+'/'+name), 'Closure mismatch')
    repair.require(proof['passed'] is True and proof['verified'] == 21444 and not proof['changed']
                   and not proof['missing'] and not proof['read_or_identity_errors'], 'Full protection required')
    repair.require(outcome['passed'] is True and outcome['execution_paths_verified'] == 9 and
                   outcome['total_monthly_orders_verified'] == 366, 'Full independent outcome required')
    incumbent = result['incumbent_local67']
    repair.require(all(value == incumbent for value in result.values()), 'This null summary requires identical paths')
    repair.require(all(value == 0 for value in feature['ordered_holdings_changed_months'].values()),
                   'This null summary requires unchanged ordered holdings')
    lines = ['# 第158轮：年报披露时滞修正', '',
             '2016-08-17至2026-09-24；前88个月选择、后34个月诊断。月末三只、月内固定持仓。', '',
             '原策略、披露时滞修正、时滞变化修正的122个月持仓及排序完全一致。'
             '评分分别在72个月和41个月发生变化，但没有形成组合层面的增量证据，保留原策略。', '',
             '| 执行口径 | 全期年化 | Sharpe | 最大回撤 |',
             '|---|---:|---:|---:|']
    for key, label in (('same_close','同收盘、零成本'),('next_open_zero','次日开盘、零成本'),
                       ('next_open_cost','次日开盘、每边0.05%')):
        m = incumbent[key]['metrics']
        lines.append(f"|{label}|{m['cagr']:.2%}|{m['sharpe']:.3f}|{m['max_drawdown']:.2%}|")
    lines += ['', '| 同收盘口径分段 | 年化 | Sharpe |', '|---|---:|---:|']
    for key, label in (('early2016_2020','2016–2020早期段'),('training2016_2023','前88个月'),('later34','后34个月')):
        m = incumbent['same_close']['slices'][key]
        lines.append(f"|{label}|{m['cagr']:.2%}|{m['sharpe']:.3f}|")
    lines += ['', '70%全期年化目标尚未达到。后34个月仍是被反复使用的历史诊断段，不能称独立样本外。',
              '', '完成独立因子与123个历史截断核验、9条收益路径及366组月度决策核验；'
              '21,444个受保护历史文件无变更。原计划、失败日志与代码修复修订记录全部保留。'
              '冻结观察账户未修改。本轮固定两发行人试验停止，不据此调剂量或追选股票。', '',
              '证据：[正式完成记录](COMPLETION.json)、[修复完成记录](REPAIR_COMPLETION_01.json)、'
              '[独立收益审计](INDEPENDENT_OUTCOME.json)、[逐月比较](MONTH_COMPARISON.json)。', '']
    repair.publish_bytes(OUT/'SUMMARY.md', '\n'.join(lines).encode())
    source_paths = {'source_plan':'work/after158_ba_source_plan.json',
                    'original_statement_review':'work/after158_ba_original_statement_review.json',
                    'independent_review':'work/after158_ba_independent_review.json',
                    'comparison':'work/after158_ba_original_statement_review/COMPARISON.json'}
    comparison = repair.read(ROOT/source_paths['comparison'])
    repair.require(comparison['passed'] is True and not comparison['substantive_contradictions'] and
                   comparison['source_admission_changed'] is False and comparison['new_strategy_trials'] == 0,
                   'Bounded source comparison required')
    checkpoint = dict(recorded_utc=datetime.now(timezone.utc).isoformat(),
        goal_turn_classification='progress', round158=repair.reference(ROOT,repair.OUT+'/REPAIR_COMPLETION_01.json'),
        source_refs={k:repair.reference(ROOT,p) for k,p in source_paths.items()},
        source_stage='BA FY2014–2018 earnings-release statement review complete; not admitted',
        next_required=['Verify or explicitly contract issuer reporting-currency basis.',
                       'Declare conservative release-date availability and check every original signal cutoff.',
                       'Preserve the FY2018 basis break and latest-report no-fallback rule.',
                       'Only a distinct registered experiment may evaluate an expanded source scope.'],
        source_strategy_trials=0, target_reached=False, observer_changes=False)
    repair.write_once(OUT/'NEXT_SOURCE_STATUS.json',checkpoint)
    print(json.dumps({'summary':str(OUT/'SUMMARY.md'),'checkpoint':str(OUT/'NEXT_SOURCE_STATUS.json'),
                      'cagr':incumbent['same_close']['metrics']['cagr'],'target_reached':False}))


if __name__ == '__main__':
    main()
