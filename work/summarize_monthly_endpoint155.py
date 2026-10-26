"""Export a completion-bound report, without models, prices or replay imports.

Run only after formal completion, with externally verified PLAN/completion SHAs.
This reporting script is intentionally outside the registered execution closure.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re


NAMES = ('incumbent_local67', 'matched_mean_return_ridge', 'matched_loss_indicator_ridge')
LABELS = dict(zip(NAMES, ('原策略', '匹配样本：平均收益 Ridge', '匹配样本：亏损指示 Ridge')))
MODES = ('same_close', 'next_open_zero', 'next_open_cost')
DIRECTORY = 'outputs/monthly_endpoint155'
PROTECTED_COUNT = 21444
PREPARE_CHECKS = (
    'original27_arrays_byte_identical', 'training_identity_and_plain10_verified',
    'raw_prices_independently_verified', 'calendar_overlay_independently_verified',
    'endpoint_labels_independently_reconstructed', 'shared_maturity_rows_weights_scalers_verified',
    'annual_forecasts_independently_reconstructed', 'shared_query_mask_and_centered_ranks_verified',
    'all_three_score_matrices_and_ordered_choices_independently_verified',
    'all_signal_prefixes_verified', 'initial_partial_and_missing_exact_fallback_verified',
    'one_episode_and_top2_lock_verified',
)
OUTCOME_CHECKS = (
    'daily_nav_trades_holdings_metrics_slices', 'first88_and_three_fold_selection_verified',
    'annual_partial_flags_verified', 'monthly_difference_and_attribution_verified',
    'first88_causal_replay_verified', 'model_fit_counts_verified', 'all_changed_months_reported',
    'diagnostic_start_marker_verified', 'baseline_local67_full_orders_unchanged',
)
METRICS = ('cagr', 'sharpe', 'max_drawdown')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def extract_metrics(row):
    require(all(finite(row.get(k)) for k in METRICS), 'Finite return/risk metrics required')
    result = {key: row[key] for key in METRICS}
    for key in ('start', 'end', 'years', 'total_return'):
        if key in row:
            result[key] = row[key]
    return result


def normalize(mapping, root):
    require(isinstance(mapping, dict) and mapping, 'Nonempty hash mapping required')
    result = {}
    for raw, digest in mapping.items():
        require(isinstance(raw, str) and isinstance(digest, str)
                and re.fullmatch(r'[0-9a-f]{64}', digest), 'Invalid source reference')
        candidate = Path(raw)
        path = candidate if candidate.is_absolute() else root/candidate
        require(not path.is_symlink() and not any('OneDrive' in x or x == 'CloudStorage' for x in path.parts),
                'Reporting must use local independent files')
        key = str(path.resolve())
        require(key not in result or result[key] == digest, 'Conflicting normalized hash')
        result[key] = digest
    return result


def build(root, completion_sha, plan_sha):
    root = Path(root).resolve()
    out = root/DIRECTORY
    observed = {}

    def read(name, expected=None):
        path = out/name
        require(not path.is_symlink(), 'Source symlink forbidden')
        digest = sha(path)
        require(expected is None or digest == expected, 'Changed artifact: '+name)
        observed[str(path)] = digest
        with path.open() as stream:
            return json.load(stream)

    def verify_files(mapping):
        for path, digest in mapping.items():
            require(sha(path) == digest, 'Changed source file: '+path)
            observed[path] = digest

    require(all(isinstance(x,str) and re.fullmatch(r'[0-9a-f]{64}',x)
                for x in (completion_sha,plan_sha)), 'Exact completion and PLAN SHAs required')
    done = read('COMPLETION.json', completion_sha)
    require(done.get('round') == 155 and done.get('directory') == 'monthly_endpoint155'
        and done.get('status') == 'completed_independently_verified', 'Formal completion required')
    require(done.get('plan_sha256') == plan_sha, 'Completion/PLAN mismatch')
    plan = read('PLAN.json', plan_sha)
    require(plan.get('round') == 155 and plan.get('status') == 'REGISTERED'
        and plan.get('arms') == list(NAMES) and plan.get('primary_challenger') == NAMES[2],
        'Wrong registered experiment')
    plan_pins = normalize(plan['hashes'], root)
    require(str(Path(__file__).resolve()) not in plan_pins,
            'Post-completion reporter must not become an execution dependency')
    source_code = {}
    reports = {}
    for kind, file, stage, checks in (
        ('prepare','INDEPENDENT_PREPARE.json','prepare',PREPARE_CHECKS),
        ('outcome','INDEPENDENT_OUTCOME.json','evaluate',OUTCOME_CHECKS),
    ):
        report = read(file, done['independent_outcome_sha256'] if kind == 'outcome' else None)
        require(report.get('passed') is True and report.get('stage') == stage
            and report.get('producer_modules_imported') is False
            and report.get('plan_sha256') == plan_sha
            and all(report.get(k) is True for k in checks), 'Incomplete independent '+kind+' proof')
        if kind == 'prepare':
            require(report.get('returns_or_nav_computed') is False, 'Prepare phase isolation')
        expected = normalize(plan['independent_'+kind+'_source_hashes'], root)
        actual = normalize(report['source_hashes'], root)
        require(actual == expected and all(plan_pins.get(p) == h for p,h in actual.items()),
                'Independent reviewer source closure differs')
        verify_files(actual); source_code.update(actual)
        reports[kind] = report
    audited = normalize(reports['outcome']['artifact_hashes'], root)
    artifacts = {}
    for file in ('RESULTS.json','TRAIN_TRIALS.json','TRAIN_OPEN_TRIALS.json','FROZEN.json',
                 'MODEL_AUDIT.json','PRETRAIN_ORDERS.json','META.json','MONTH_COMPARISON.json',
                 'ATTRIBUTION.json','VALIDATION.json','CACHE.json','RANK_AUDIT.json','INDEPENDENT_PREPARE.json'):
        expected = audited.get(str(out/file))
        require(expected is not None, 'Missing independently bound summary artifact: '+file)
        artifacts[file] = read(file, expected)
    require(all(reports[k].get('cache_sha256') == observed[str(out/'CACHE.json')]
        and reports[k].get('orders_sha256') == observed[str(out/'PRETRAIN_ORDERS.json')]
        for k in reports), 'Receipt cache/order mismatch')
    require(reports['outcome'].get('frozen_sha256') == observed[str(out/'FROZEN.json')],
            'Outcome/frozen mismatch')
    protection = read('PROTECTED_AFTER.json', done['protected_sha256'])
    require(protection.get('passed') is True and protection.get('checked') == PROTECTED_COUNT
        and protection.get('verified') == PROTECTED_COUNT
        and all(protection.get(k) == [] for k in ('changed','missing','read_or_identity_errors'))
        and protection.get('manifest_sha256') == plan['inputs']['protected_manifest']['sha256'],
        'Complete immutable archive proof required')
    require(done.get('trials') == 3 and done.get('independent_execution_paths') == 9
            and done.get('protected_files') == PROTECTED_COUNT, 'Wrong completion scope')
    results = artifacts['RESULTS.json']
    require(results == done['all_arm_results'] and set(results) == set(NAMES)
        and all(set(results[n]) == set(MODES) for n in NAMES), 'Completion/result mismatch')
    selection = artifacts['FROZEN.json']['selection']
    require(selection == done['selection'] and selection == artifacts['VALIDATION.json']['selection']
        and selection.get('primary') == done['primary'] == reports['outcome']['selected_name']
        and type(selection.get('admitted')) is bool, 'Frozen selected model mismatch')
    require(artifacts['FROZEN.json'].get('pristine_oos') is False
        and artifacts['FROZEN.json'].get('all_diagnostic_orders_frozen') is True
        and artifacts['VALIDATION.json'].get('no_observer_writes') is True
        and artifacts['VALIDATION.json'].get('no_intramonth_trades') is True, 'Research/execution invariants')
    target = results[done['primary']]['same_close']['metrics']['cagr'] >= .7
    require(type(done.get('target_reached')) is bool and done['target_reached'] == target
        and reports['outcome'].get('primary_target_reached') == target, 'Selected target inconsistency')
    fits = 0; fit_years = []; seen_years = set()
    for annual in artifacts['MODEL_AUDIT.json']['annual']:
        year,status = annual['year'],annual['status']
        require(type(year) is int and year not in seen_years
            and status in ('fit','cold_start','no_queries'), 'Unique annual model status')
        seen_years.add(year)
        count = 2 if status == 'fit' else 0
        require(type(annual.get('model_fits')) is int and annual['model_fits'] == count,
                'Two paired fits per fitted year')
        fits += count
        if count: fit_years.append(year)
    require(all(type(x) is int and x == fits for x in (
        done['new_model_fits'], artifacts['MODEL_AUDIT.json']['new_pooled_model_fits'],
        artifacts['VALIDATION.json']['new_model_fits'],reports['prepare']['new_model_fits'],
        reports['outcome']['producer_pooled_model_fits'])), 'Actual producer fit count mismatch')
    require(artifacts['MODEL_AUDIT.json'].get('local_model_fits') == 0
        and reports['outcome'].get('models_fit') == 0, 'No local/outcome model fits')
    meta = artifacts['META.json']; orders = artifacts['PRETRAIN_ORDERS.json']
    require(len(meta['dates']) == len(meta['ends']) == 122 and meta['dates'][0] == '2016-08-17'
        and meta['ends'][-1] == '2026-09-24' and set(orders) == set(NAMES)
        and all(len(orders[n]) == 122 for n in NAMES), 'Exact 122-period horizon')
    train = artifacts['TRAIN_TRIALS.json']; opened = artifacts['TRAIN_OPEN_TRIALS.json']
    require([x['name'] for x in train] == [x['name'] for x in opened] == list(NAMES), 'Fixed training rows')
    rows = []
    for n, t, o in zip(NAMES, train, opened):
        paths = results[n]
        for source, computed in ((t,paths['same_close']), (o,paths['next_open_cost'])):
            require(abs(source['training']['cagr']-computed['slices']['training2016_2023']['cagr']) < 1e-10,
                    'Training/report CAGR mismatch')
        rows.append(dict(name=n,label=LABELS[n],
            full={mode:extract_metrics(paths[mode]['metrics']) for mode in MODES},
            same_close_slices={key:extract_metrics(paths['same_close']['slices'][key])
                for key in ('training2016_2023','early2016_2020','later34')},
            next_open_5bp_slices={key:extract_metrics(paths['next_open_cost']['slices'][key])
                for key in ('training2016_2023','early2016_2020','later34')},
            first88_training={key:extract_metrics(t[key]) for key in ('training','early','recent')},
            first88_next_open_5bp={key:extract_metrics(o[key]) for key in ('training','early','recent')}))
    comparison = artifacts['ATTRIBUTION.json']['comparisons']; changes = {}
    for label, name, control in (
        ('return_vs_incumbent',NAMES[1],NAMES[0]), ('loss_vs_incumbent',NAMES[2],NAMES[0]),
        ('loss_vs_return',NAMES[2],NAMES[1]),
    ):
        set_changes = []; ordered = 0
        for i in range(122):
            old,new = orders[control][i]['symbols'],orders[name][i]['symbols']
            ordered += int(old != new)
            if set(old) != set(new):
                set_changes.append(dict(index=i,date=meta['dates'][i],old=old,new=new,
                                        split='first88' if i < 88 else 'later34'))
        require(comparison[label]['changed_months'] == len(set_changes)
            and comparison[label]['ordered_changed_months'] == ordered, 'Holdings-change count mismatch')
        changes[label] = dict(changed_months=len(set_changes),ordered_changed_months=ordered,
            first88=sum(x['index'] < 88 for x in set_changes),
            later34=sum(x['index'] >= 88 for x in set_changes),months=set_changes)
    source_code[str(Path(__file__).resolve())] = sha(__file__)
    summary = dict(schema_version=1,round=155,status='completed_summary',
        completed_utc=done['completed_utc'],period=[meta['dates'][0],meta['ends'][-1]],
        completion_sha256=completion_sha,plan_sha256=plan_sha,
        selection=selection,primary=done['primary'],admitted=selection['admitted'],
        target_cagr=.7,target_reached=target,arms=rows,holdings_changes=changes,
        producer_model_fits=fits,paired_fitted_years=fit_years,independent_outcome_model_fits=0,
        registered_trials=3,execution_paths=9,monthly_orders=366,
        protection=dict(verified=PROTECTED_COUNT,receipt_sha256=done['protected_sha256'],rescan_performed_here=False),
        additional_training_label_or_return_calculations=0,
        limitations=[
            '固定1%排名修正的双目标重测；未搜索权重或更换股票池。只有两种新目标间的差异隔离了标签选择。',
            '前88期用于选择；后34期及全期为诊断。历史样本已被多轮重复研究，不能称纯样本外；早期轮次是否重复同一假设尚未完全排除。',
            '亏损指示 Ridge 是未经概率校准的线性排名分，可能超出[0,1]；没有0.5交易阈值或亏损概率保证。',
            '两个目标保留旧common_good对完整路径、波动率和SPY有效性的筛选，不是纯端点样本。',
            '沿用本地历史32股加17 ETF股票池、估算及资料版本；其历史成员、数据版本与现实可交易性限制仍在。',
            '同收盘零成本属于理想化研究口径；另列下一开盘每边5bp。开盘成本路径与收盘标签预测对象不同。',
            '2016-08-17首个非月末信号严格回退原策略；2026-09-24末期仅按已有部分区间估值，不缩短训练标签端点。',
            '原冻结观察账户及其模拟账本未改动，没有自动推广本次候选。',
        ],artifact_hashes=dict(observed),reporting_source_hashes=source_code)
    verify_files(dict(observed))
    return summary


def markdown(summary):
    pct = lambda x: f'{100*x:.2f}%'
    number = lambda x: f'{x:.3f}'
    selected = LABELS[summary['primary']]
    lines = ['# 第155轮：固定1%端点目标重测', '',
        f"区间：{summary['period'][0]} 至 {summary['period'][1]}。正式完成、独立准备与九条执行路径审核均已通过。", '',
        f"前88期冻结选择：**{selected}**；亏损指示候选{'通过' if summary['admitted'] else '未通过'}准入。"
        f"已选策略全期年化70%目标：**{'达到' if summary['target_reached'] else '未达到'}**。", '',
        '| 策略 | 全期同收盘年化 | Sharpe | 最大回撤 | 次日开盘每边5bp年化 | Sharpe | 最大回撤 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in summary['arms']:
        a,b = row['full']['same_close'],row['full']['next_open_cost']
        lines.append(f"| {row['label']} | {pct(a['cagr'])} | {number(a['sharpe'])} | {pct(a['max_drawdown'])} | {pct(b['cagr'])} | {number(b['sharpe'])} | {pct(b['max_drawdown'])} |")
    lines += ['', '分段同收盘指标；每格依次为年化 / Sharpe / 最大回撤。', '',
        '| 策略 | 前88期（2016–2023） | 早期2016–2020 | 后34期 |', '|---|---|---|---|']
    for row in summary['arms']:
        cells=[]
        for key in ('training2016_2023','early2016_2020','later34'):
            a=row['same_close_slices'][key]
            cells.append(f"{pct(a['cagr'])} / {number(a['sharpe'])} / {pct(a['max_drawdown'])}")
        lines.append('| '+row['label']+' | '+' | '.join(cells)+' |')
    lines += ['', '持仓集合变化与名单顺序变化分别统计：', '',
        '| 比较 | 集合变化月份 | 其中前88期 | 其中后34期 | 含顺序变化月份 |',
        '|---|---:|---:|---:|---:|']
    labels={'return_vs_incumbent':'平均收益 vs 原策略','loss_vs_incumbent':'亏损指示 vs 原策略','loss_vs_return':'亏损指示 vs 平均收益'}
    for key,row in summary['holdings_changes'].items():
        lines.append(f"| {labels[key]} | {row['changed_months']} | {row['first88']} | {row['later34']} | {row['ordered_changed_months']} |")
    lines += ['', f"实际生产模型拟合 **{summary['producer_model_fits']}次**，由年度审计逐项汇总；每个有拟合的年份两个目标各一次。"
        '独立收益审核不拟合模型；本汇总也没有新增训练或收益计算。', '',
        f"保护档案：{summary['protection']['verified']:,}个文件已由正式关闭流程核验；此处验证其绑定回执，不再次扫描整个档案。", '',
        *['- '+text for text in summary['limitations']], '',
        '完整分段（含开盘5bp）、逐月持仓差异及输入哈希见 [SUMMARY.json](SUMMARY.json)。', '',
        f"PLAN SHA-256：`{summary['plan_sha256']}`", '',
        f"COMPLETION SHA-256：`{summary['completion_sha256']}`", '']
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--completion-sha',required=True)
    parser.add_argument('--plan-sha',required=True)
    args=parser.parse_args()
    out=args.root.resolve()/DIRECTORY
    require(not any((out/name).exists() for name in ('SUMMARY.json','SUMMARY.md')),
            'Preserve existing summary outputs')
    summary=build(args.root,args.completion_sha,args.plan_sha)
    document=markdown(summary)
    encoded=json.dumps(summary,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+'\n'
    with (out/'SUMMARY.json').open('x') as stream:stream.write(encoded)
    with (out/'SUMMARY.md').open('x') as stream:stream.write(document)
    print(json.dumps({name:sha(out/name) for name in ('SUMMARY.json','SUMMARY.md')}))


if __name__ == '__main__':
    main()
