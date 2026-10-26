"""Read-only attribution of bound round188 exports; no feature/replay imports."""
import argparse
from datetime import date, datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/monthly_state_vol188'
TEST_FILE=ROOT / 'work/test_monthly_state_vol188_attribution.py'
DEST = OUT / 'attribution'
ARMS = ('incumbent_local67', 'state_conditioned_vol_cdf', 'unconditional_vol_cdf')
PATHS = ('same_close_zero', 'next_open_zero', 'next_open_cost')
SHOW_PATHS = ('same_close_zero', 'next_open_cost')
SLICES = ('full122', 'first88', 'early53', 'recent35', 'later34')
COMPARISONS = {'conditional_vs_incumbent': (ARMS[1], ARMS[0]),
               'unconditional_vs_incumbent': (ARMS[2], ARMS[0]),
               'conditional_vs_unconditional': (ARMS[1], ARMS[2])}


def require(value, message):
    if not value:
        raise ValueError(message)


def near(actual, expected, message):
    require(math.isfinite(actual) and math.isfinite(expected) and
            math.isclose(actual, expected, rel_tol=2e-10, abs_tol=2e-10),
            message + ': ' + repr((actual, expected)))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def verify(pins):
    for name, expected in pins.items():
        path = Path(name).resolve()
        require('CloudStorage' not in path.parts and
                (path.is_relative_to(ROOT) or str(path).startswith('/Library/Developer/CommandLineTools/')),
                'Unapproved input path: ' + name)
        require(path.is_file() and sha(path) == expected, 'Input hash mismatch: ' + name)


def metrics(rows, initial):
    days = [r['date'] for r in rows]
    values = [r['equity'] for r in rows]
    require(len(values) > 1 and all(math.isfinite(v) and v > 0 for v in values), 'Invalid NAV slice')
    years = (date.fromisoformat(days[-1]) - date.fromisoformat(days[0])).days / 365.2425
    returns = [b / a - 1. for a, b in zip(values, values[1:])]
    deviation = statistics.stdev(returns)
    require(years > 0 and deviation > 0, 'Degenerate metric slice')
    peak, drawdown = initial, 0.
    for value in values:
        peak = max(peak, value)
        drawdown = min(drawdown, value / peak - 1.)
    return dict(start=days[0], end=days[-1], years=years,
                cagr=(values[-1] / initial) ** (1. / years) - 1.,
                sharpe=statistics.mean(returns) / deviation * math.sqrt(252.),
                max_drawdown=drawdown, observations=len(values),
                wealth_multiple=values[-1] / initial)


def decompose(factors, reference_cagr, years):
    require(all(math.isfinite(f) and f > 0 for f in factors) and years > 0, 'Invalid factors')
    logs = [math.log(f) for f in factors]
    total = math.fsum(logs)
    difference = (1. + reference_cagr) * math.expm1(total / years)
    slope = difference / total if total else (1. + reference_cagr) / years
    contributions = [slope * value for value in logs]
    near(math.fsum(contributions), difference, 'CAGR contribution sum')
    return dict(terminal_wealth_ratio=math.exp(total),
                annual_growth_multiplier=math.exp(total / years),
                cagr_difference=difference,
                factors=factors, log_factors=logs,
                annual_growth_factors=[math.exp(v / years) for v in logs],
                cagr_contributions=contributions)


def guard(candidate, reference, gain):
    return dict(cagr=candidate['training']['cagr'] >= reference['training']['cagr'] + gain - 1e-12,
                sharpe=candidate['training']['sharpe'] >= reference['training']['sharpe'] - .05,
                drawdown=candidate['training']['max_drawdown'] >= reference['training']['max_drawdown'] - .03,
                early=candidate['early']['cagr'] >= reference['early']['cagr'] - .025,
                recent=candidate['recent']['cagr'] >= reference['recent']['cagr'] - .025)


def describe(values):
    values=sorted(values)
    return dict(count=len(values),minimum=min(values) if values else None,
                median=statistics.median(values) if values else None,
                maximum=max(values) if values else None)


def support_diagnostics(feature,values,transforms):
    """Inspect saved features only, without recalculating prices or choosing trades."""
    history=feature['event_history'];n=len(history);width=feature['symbols_per_event']
    require(set(values)=={'candidate','control','available'},'Saved feature names')
    mask=np.asarray(values['available']);require(mask.shape==(n,width) and mask.dtype==bool,'Saved support shape')
    raw=dict(zip(ARMS[1:],('candidate','control')))
    expected_keys={'transform_'+a for a in raw}|{'delta_'+a for a in raw}
    require(set(transforms)==expected_keys,'Saved transform/delta names')
    maximum={};changed={}
    for arm,key in raw.items():
        v=np.asarray(values[key]);require(v.shape==mask.shape and np.array_equal(np.isfinite(v),mask) and not np.isinf(v).any(),'Saved raw common support')
        require(np.all(np.abs(v[mask])<=.5),'Saved CDF bounds')
        wanted=np.where(mask,v,0.);delta=.01*wanted
        if n:delta[0]=0.
        require(np.array_equal(transforms['transform_'+arm],wanted),'Unexpected saved secondary transform')
        require(np.array_equal(transforms['delta_'+arm],delta),'Saved delta formula differs')
        maximum[arm]=float(np.max(np.abs(delta),initial=0.))
        require(maximum[arm]<=.005,'Saved delta over budget')
        changed[arm]=int(np.count_nonzero(delta))
    counts={'calm':dict(cells=0,available=0),'other':dict(cells=0,available=0),'unknown':dict(cells=0,available=0)}
    refs=[];matched=[];available_refs=[];available_matched=[];reasons={};rows=[]
    for i,event in enumerate(history):
        require(len(event['cells'])==width and len(event['symbols'])==width,'Saved audit row width')
        state=event['current_calm'];require(state is None or type(state) is bool,'Unknown state must remain missing')
        group='unknown' if state is None else 'calm' if state else 'other'
        row_refs=[];row_matches=[]
        for j,cell in enumerate(event['cells']):
            require(cell['symbol']==event['symbols'][j] and cell['date']==event['date'],'Saved cell identity')
            require(cell['current_calm'] is state and cell['available']==bool(mask[i,j]),'Saved state/support audit differs')
            r,m=cell['reference_count'],cell['matched_count']
            require(type(r) is int and type(m) is int and 0<=m<=r<=756,'Saved reference counts')
            if state is None:require(m==0 and not mask[i,j],'Missing state cannot count noncalm')
            if mask[i,j]:
                require(r>=252 and m>=63 and cell['reason']=='available','Saved common minimum support')
                available_refs.append(r);available_matched.append(m)
            for key in ('candidate','control'):
                require((cell[key] is None)==(not mask[i,j]),'Saved unavailable raw must be null')
                if mask[i,j]:require(cell[key]==float(values[key][i,j]),'Saved cell raw differs')
            refs.append(r);matched.append(m);row_refs.append(r);row_matches.append(m)
            reasons[cell['reason']]=reasons.get(cell['reason'],0)+1
            counts[group]['cells']+=1;counts[group]['available']+=int(mask[i,j])
        rows.append(dict(date=event['date'],current_calm=state,available_cells=int(mask[i].sum()),
                         reference_count=describe(row_refs),matched_count=describe(row_matches)))
    require(sum(refs)==feature['reference_observations'] and sum(matched)==feature['matched_observations'],'Saved descriptive counts differ')
    require(reasons==feature['reason_counts'] and int(mask.sum())==feature['available_cells'] and int((~mask).sum())==feature['unavailable_cells'],'Saved support totals differ')
    require(feature['own_return_count']==126 and feature['market_return_count']==63 and feature['history_slots']==756,'Saved risk/history definition')
    return dict(common_availability=True,cells=n*width,available_cells=int(mask.sum()),unavailable_cells=int((~mask).sum()),
                reference_observations=sum(refs),matched_observations=sum(matched),reason_counts=reasons,
                by_market_state=counts,all_reference_counts=describe(refs),all_matched_counts=describe(matched),
                available_reference_counts=describe(available_refs),available_matched_counts=describe(available_matched),
                max_absolute_delta=maximum,delta_bound=.005,changed_delta_cells=changed,event_support=rows,
                references_are_overlapping_not_independent_samples=True)


def concentration(logs,dates):
    """Descriptive absolute-log attribution, without selecting or changing a strategy."""
    require(len(logs)==len(dates) and all(math.isfinite(x) for x in logs),'Invalid log concentration inputs')
    mass=math.fsum(abs(x) for x in logs)
    weights=[abs(x)/mass for x in logs] if mass else [0.]*len(logs)
    ranked=sorted(range(len(logs)),key=lambda i:(-abs(logs[i]),i))
    hhi=math.fsum(w*w for w in weights)
    def items(indices):return [dict(date=dates[i],log_wealth_factor=logs[i],absolute_log_share=weights[i]) for i in indices]
    return dict(total_log_wealth_ratio=math.fsum(logs),positive_log_sum=math.fsum(x for x in logs if x>0),
                negative_log_sum=math.fsum(x for x in logs if x<0),absolute_log_sum=mass,
                nonzero_months=sum(x!=0. for x in logs),absolute_log_hhi=hhi,
                effective_month_count=1/hhi if hhi else None,
                top_absolute_log_share={str(k):math.fsum(weights[i] for i in ranked[:k]) for k in (1,3,5)},
                largest_absolute_months=items(ranked[:5]),
                best_months=items(sorted([i for i,x in enumerate(logs) if x>0],key=lambda i:(-logs[i],i))[:5]),
                worst_months=items(sorted([i for i,x in enumerate(logs) if x<0],key=lambda i:(logs[i],i))[:5]),
                diagnostic_only=True)


def display(symbol):
    return {'research_equity_FB': 'META'}.get(symbol, symbol)


def markdown(report):
    names=dict(zip(ARMS,('Local67','同市况历史波动CDF','无条件历史波动CDF')))
    segments=dict(zip(SLICES,('全期122','训练88','早期53','近期35','后段34')))
    admitted=report['selection']['admitted']
    decision='候选通过冻结的训练门槛。' if admitted else '候选未通过冻结的训练门槛，保留 Local67。'
    lines=['# 第188轮结果归因','',decision+'后段34期只作事后诊断。','',
        '新修正比较自身126日波动率在此前756交易日中的位置。市况由SPY63日年化波动是否低于10%定义；缺失市况保持未知。候选使用相同市况历史，对照使用全部共同有效历史，均要求252条共同参考及63条同市况参考，排除当日。两臂只以0.01乘中心化CDF加到原pre-score，最大±0.005；原长期动量、模型、支持分和一次episode保持。','',
        '以下为 CAGR／Sharpe／最大回撤。回撤保留负号；同收盘为零成本研究口径，次开盘每边5bp。','']
    for path,title in [('same_close_zero','同收盘零成本'),('next_open_cost','次开盘每边5bp')]:
        lines+=['## '+title,'','|区间|Local67|同市况CDF|无条件CDF|','|---|---:|---:|---:|']
        for segment in SLICES:
            vals=[report['metrics'][arm][path][segment] for arm in ARMS]
            lines.append('|'+segments[segment]+'|'+'|'.join(f"{x['cagr']:.4%} / {x['sharpe']:.4f} / {x['max_drawdown']:.4%}" for x in vals)+'|')
        lines.append('')
    lines+=['## 训练门槛','']
    for key,details in report['guard_details'].items():
        failed=[k for k,v in details.items() if type(v) is bool and not v]
        lines.append('- '+key+'：'+('未通过 '+', '.join(failed) if failed else '通过'))
    d=report['support_diagnostics']
    lines+=['','## 共同支持与修正幅度','',f"有效 {d['available_cells']}/{d['cells']} 个标的月份；缺失原因 {d['reason_counts']}。共同参考合计 {d['reference_observations']}，同市况参考合计 {d['matched_observations']}。这些滚动窗口重复使用历史，计数不代表独立样本。",'',
        '|组别|参考数最小／中位／最大|','|---|---:|']
    for key,label in [('available_reference_counts','有效标的共同参考'),('available_matched_counts','有效标的同市况参考')]:
        z=d[key];lines.append(f"|{label}|{z['minimum']} / {z['median']} / {z['maximum']}|")
    lines+=['', '两臂实际最大绝对修正：'+', '.join(names[a]+' '+str(d['max_absolute_delta'][a]) for a in ARMS[1:])+'；上限0.005。','',
        '## 持仓集合变化','', '|信号日 → 期末|Local67|同市况CDF|无条件CDF|','|---|---|---|---|']
    for row in report['changed_months']:
        lines.append('|'+row['signal_date']+' → '+row['end']+'|'+'|'.join(', '.join(display(s) for s in row['holdings'][arm]) or '现金' for arm in ARMS)+'|')
    for path,title in [('same_close_zero','同收盘'),('next_open_cost','次开盘5bp')]:
        lines+=['','## '+title+'月度差异','',
            '按信号日收盘到下一信号日/末期收盘划分。次开盘包含旧仓隔夜、次日调仓及费用，可影响下一段，即使该段持仓集合相同。','',
            '|信号日|候选−Local67 log收益差|候选−无条件 log收益差|候选−Local67 CAGR贡献pp|候选−无条件 CAGR贡献pp|','|---|---:|---:|---:|---:|']
        for row in report['all122_periods']:
            pair=row['paths'][path]['comparisons'];x=pair['conditional_vs_incumbent'];y=pair['conditional_vs_unconditional']
            if x['log_wealth_factor']!=0 or y['log_wealth_factor']!=0:
                lines.append(f"|{row['signal_date']}|{x['log_wealth_factor']:+.8f}|{y['log_wealth_factor']:+.8f}|{x['full_cagr_contribution_pp']:+.5f}|{y['full_cagr_contribution_pp']:+.5f}|")
        lines+=['','|比较|总log收益差|绝对log差Top1／3／5占比|HHI|','|---|---:|---:|---:|']
        for key in ('conditional_vs_incumbent','conditional_vs_unconditional'):
            z=report['log_concentration'][path][key]['full122'];shares=z['top_absolute_log_share']
            lines.append(f"|{key}|{z['total_log_wealth_ratio']:+.8f}|{shares['1']:.2%} / {shares['3']:.2%} / {shares['5']:.2%}|{z['absolute_log_hhi']:.5f}|")
    lines+=['', '每期财富倍数为(1+目标收益)/(1+参照收益)，122期乘积等于终值比。CAGR贡献按各期log倍数线性分配并精确加总；这是一种会计分解，不是唯一因果解释。所有三条执行路径、五切片和122期明细保存在JSON。', '',
        '45个切片及18个训练指标均由保存净值独立复核。PLAN、根完成回执、结果完成回执及全部读取产物绑定SHA并在运行前后检查。归因不改变候选、门槛、中央状态或观察账户。历史代理股票池、回溯数据版本和反复使用样本的限制仍在；不能当独立样本外。','']
    return '\n'.join(lines)


def main(plan_sha, result_sha, root_sha):
    for label, digest in [('PLAN', plan_sha), ('result completion', result_sha), ('root completion',root_sha)]:
        require(isinstance(digest, str) and len(digest) == 64 and all(c in '0123456789abcdef' for c in digest),
                'An explicit valid SHA-256 is required for ' + label)
    require(not DEST.exists(), 'Attribution output is write-once')
    require(sha(OUT / 'PLAN.json') == plan_sha, 'Unexpected PLAN')
    require(sha(OUT / 'results/COMPLETION.json') == result_sha, 'Result completion not ready or changed')
    require(sha(OUT / 'COMPLETION.json') == root_sha, 'Root completion not ready or changed')
    root=read(OUT / 'COMPLETION.json')
    require(root['status']=='COMPLETED_INDEPENDENTLY_VERIFIED' and root['plan_sha256']==plan_sha and root['result_completion_sha256']==result_sha and root['observers_modified'] is False, 'Incomplete root closure')
    plan = read(OUT / 'PLAN.json')
    prepared = read(OUT / 'prepared/COMPLETION.json')
    completed = read(OUT / 'results/COMPLETION.json')
    frozen = read(OUT / 'selected/FROZEN.json')
    pins = dict(plan['input_sha256'])
    for path in (OUT / 'COMPLETION.json', Path(__file__).resolve(), TEST_FILE, OUT / 'PLAN.json', OUT / 'prepared/COMPLETION.json',
                 OUT / 'results/COMPLETION.json', OUT / 'selected/FROZEN.json'):
        pins[str(path)] = sha(path)
    for directory, receipt in [('prepared', prepared), ('results', completed)]:
        require(receipt['status'] == 'COMPLETE' and receipt['plan_sha256'] == plan_sha, 'Incomplete receipt')
        for name, digest in receipt['artifacts_sha256'].items():
            require(Path(name).name == name, 'Unexpected artifact path')
            pins[str(OUT / directory / name)] = digest
    require(set(prepared['artifacts_sha256']) == {'FEATURES.npz', 'TRANSFORMS_DELTAS.npz', 'FEATURE_AUDIT.json', 'ORDERS.json', 'META.json', 'SOURCE_PARITY.json', 'INDEPENDENT_PREPARE.json'} and set(completed['artifacts_sha256']) == ({a+'_'+p+'_'+k+'.json' for a in ARMS for p in PATHS for k in ('DAILY','TRADES','HOLDINGS','DAILY_POSITIONS')}|{'RESULTS.json','INDEPENDENT_OUTCOME.json'}),
            'Unexpected artifact coverage')
    for key, name in [('prepared_sha256', 'prepared/COMPLETION.json'),
                      ('orders_sha256', 'prepared/ORDERS.json'),
                      ('training_sha256', 'selected/TRAIN_RESULTS.json')]:
        require(sha(OUT / name) == frozen[key], 'Frozen binding changed: ' + name)
        pins[str(OUT / name)] = frozen[key]
    require(frozen['plan_sha256'] == plan_sha and completed['frozen_sha256'] == sha(OUT / 'selected/FROZEN.json'), 'Frozen selection changed')
    verify(pins)  # No full-period numerical analysis precedes this check.
    results = read(OUT / 'results/RESULTS.json')
    independent = read(OUT / 'results/INDEPENDENT_OUTCOME.json')
    training = read(OUT / 'selected/TRAIN_RESULTS.json')
    meta, orders = read(OUT / 'prepared/META.json'), read(OUT / 'prepared/ORDERS.json')
    require(frozen['selection'] == results['selection'], 'Result selection differs')
    require(frozen['selection']['first88_only'] is True and frozen['selection']['later34_used'] is False, 'Selection boundary changed')
    require(prepared['created_utc'] < frozen['created_utc'] < completed['created_utc'] <= root['created_utc'], 'Receipt sequence changed')
    require(independent['passed'] is True and independent['checked_paths'] == 9 and independent['checked_slices'] == 45
            and independent['independent_accounts'] is True and independent['independent_of_production_model_and_engine'] is True
            and independent['terminal_liquidation'] is False, 'Independent execution receipt incomplete')
    require(meta['dates'][0] == '2016-08-17' and meta['ends'][-1] == '2026-09-24'
            and len(meta['dates']) == len(meta['ends']) == 122 and set(orders) == set(ARMS), 'Unexpected period or arms')
    require(meta['ends'][:-1] == meta['dates'][1:], 'Attribution periods are not contiguous')
    for arm in ARMS:
        require(len(orders[arm]) == 122, 'Missing orders')
        for i, order in enumerate(orders[arm]):
            require(order['date'] == meta['dates'][i] and order['end'] == meta['ends'][i], 'Order/period mismatch')
    bounds = dict(full122=(meta['dates'][0], meta['ends'][-1]),
                  first88=(meta['dates'][0], meta['ends'][87]), early53=(meta['dates'][0], meta['ends'][52]),
                  recent35=(meta['dates'][53], meta['ends'][87]), later34=(meta['dates'][88], meta['ends'][-1]))
    calculated, nav = {}, {}
    for arm in ARMS:
        calculated[arm] = {}
        for path in PATHS:
            rows = read(OUT / 'results' / (arm + '_' + path + '_DAILY.json'))
            nav[arm, path] = {r['date']: r['equity'] for r in rows}
            require(len(nav[arm, path]) == len(rows) and list(nav[arm, path]) == sorted(nav[arm, path]), 'NAV order/duplicates')
            calculated[arm][path] = {}
            for segment, (start, end) in bounds.items():
                part = [r for r in rows if start <= r['date'] <= end]
                require(part[0]['date'] == start and part[-1]['date'] == end, 'Missing NAV endpoints')
                value = metrics(part, 100000. if segment == 'full122' else part[0]['equity'])
                calculated[arm][path][segment] = value
                for key in ('cagr', 'sharpe', 'max_drawdown'):
                    near(value[key], results['arms'][arm][path]['slices'][segment][key], 'Production metric ' + key)
                    near(value[key], independent['paths'][arm][path]['slices'][segment][key], 'Independent metric ' + key)
            if path in SHOW_PATHS:
                for short, full in [('training', 'first88'), ('early', 'early53'), ('recent', 'recent35')]:
                    for key in ('cagr', 'sharpe', 'max_drawdown'):
                        near(calculated[arm][path][full][key], training[arm][path][short][key], 'Frozen training ' + key)
    changes = {arm: [i for i in range(122) if set(orders[arm][i]['symbols']) != set(orders[ARMS[0]][i]['symbols'])]
               for arm in ARMS[1:]}
    require([i for i in changes[ARMS[1]] if i < 88] == frozen['selection']['changed_months'], 'Training changed-month mismatch')
    guard_details = {}
    for key, path, reference, gain in [('close_vs_incumbent', 'same_close_zero', ARMS[0], .005),
                                     ('open_vs_incumbent', 'next_open_cost', ARMS[0], 0.),
                                     ('close_vs_unconditional', 'same_close_zero', ARMS[2], 0.),
                                     ('open_vs_unconditional', 'next_open_cost', ARMS[2], 0.)]:
        details = guard(training[ARMS[1]][path], training[reference][path], gain)
        require(all(details.values()) == frozen['selection']['guards'][key], 'Selection guard mismatch')
        guard_details[key] = details
    changed_training = [i for i in changes[ARMS[1]] if i < 88]
    support = len(changed_training) >= 3 and any(i < 53 for i in changed_training) and any(i >= 53 for i in changed_training)
    require(support == frozen['selection']['guards']['changed_support'], 'Changed support guard differs')
    changed_control = [i for i in range(88) if set(orders[ARMS[1]][i]['symbols']) != set(orders[ARMS[2]][i]['symbols'])]
    control_support = len(changed_control) >= 3 and any(i < 53 for i in changed_control) and any(i >= 53 for i in changed_control)
    require(changed_control == frozen['selection']['changed_months_vs_control'], 'Control changed months differ')
    require(control_support == frozen['selection']['guards']['changed_support_vs_control'], 'Control support guard differs')
    admitted = all(all(details.values()) for details in guard_details.values()) and support and control_support
    require(admitted == frozen['selection']['admitted'] and
            frozen['selection']['primary'] == (ARMS[1] if admitted else ARMS[0]), 'Frozen primary differs')
    guard_details['changed_support'] = dict(passed=support, changed_training=changed_training)
    guard_details['changed_support_vs_control'] = dict(passed=control_support, changed_training=changed_control)
    changed_indices = set().union(*[set(v) for v in changes.values()])
    periods = []
    for i, (start, end) in enumerate(zip(meta['dates'], meta['ends'])):
        row = dict(index=i, signal_date=start, end=end, stage='first88' if i < 88 else 'later34',
                   holdings={a: orders[a][i]['symbols'] for a in ARMS},
                   changed_vs_incumbent={a: i in changes[a] for a in ARMS[1:]}, paths={})
        for path in PATHS:
            growth = {a: nav[a, path][end] / nav[a, path][start] for a in ARMS}
            row['paths'][path] = dict(returns={a: growth[a] - 1. for a in ARMS}, comparisons={})
            for key, (target, reference) in COMPARISONS.items():
                row['paths'][path]['comparisons'][key] = dict(
                    wealth_factor=growth[target] / growth[reference],
                    return_difference_pp=100. * (growth[target] - growth[reference]))
        periods.append(row)
    decomposition = {}
    for path in PATHS:
        decomposition[path] = {}
        for key, (target, reference) in COMPARISONS.items():
            near(nav[target, path][meta['dates'][0]], nav[reference, path][meta['dates'][0]], 'Initial NAV differs')
            base = calculated[reference][path]['full122']
            factors = [r['paths'][path]['comparisons'][key]['wealth_factor'] for r in periods]
            dec = decompose(factors, base['cagr'], base['years'])
            near(dec['terminal_wealth_ratio'], nav[target, path][meta['ends'][-1]] / nav[reference, path][meta['ends'][-1]], 'Terminal factorization')
            near(dec['cagr_difference'], calculated[target][path]['full122']['cagr'] - base['cagr'], 'CAGR factorization')
            for i, row in enumerate(periods):
                row['paths'][path]['comparisons'][key].update(
                    log_wealth_factor=dec['log_factors'][i], annual_growth_factor=dec['annual_growth_factors'][i],
                    full_cagr_contribution_pp=100. * dec['cagr_contributions'][i])
            decomposition[path][key] = {k: v for k, v in dec.items() if not isinstance(v, list)}
    log_concentration={}
    ranges=dict(full122=(0,122),first88=(0,88),early53=(0,53),recent35=(53,88),later34=(88,122))
    for path in PATHS:
        log_concentration[path]={}
        for key in COMPARISONS:
            log_concentration[path][key]={}
            for segment,(lo,hi) in ranges.items():
                part=periods[lo:hi]
                log_concentration[path][key][segment]=concentration([r['paths'][path]['comparisons'][key]['log_wealth_factor'] for r in part],[r['signal_date'] for r in part])
    feature=read(OUT / 'prepared/FEATURE_AUDIT.json')
    with np.load(OUT / 'prepared/FEATURES.npz',allow_pickle=False) as z:values={k:z[k].copy() for k in z.files}
    with np.load(OUT / 'prepared/TRANSFORMS_DELTAS.npz',allow_pickle=False) as z:transforms={k:z[k].copy() for k in z.files}
    support=support_diagnostics(feature,values,transforms)
    evidence=read(OUT / 'prepared/INDEPENDENT_PREPARE.json')
    require(evidence['feature']['passed'] is True and evidence['delta']['passed'] is True, 'Preparation not independently verified')
    for key in ('reference_observations','matched_observations'):
        require(support[key]==evidence['feature'][key]==prepared[key]==root[key], 'Support receipt chain differs')
    extras = []
    for row in periods:
        if row['index'] in changed_indices:
            continue
        for path in SHOW_PATHS:
            c = row['paths'][path]['comparisons']
            if any(abs(c[k]['return_difference_pp']) > 1e-8 for k in ('conditional_vs_incumbent', 'unconditional_vs_incumbent')):
                extras.append(dict(index=row['index'], signal_date=row['signal_date'], path=path,
                    conditional_difference_pp=c['conditional_vs_incumbent']['return_difference_pp'],
                    unconditional_difference_pp=c['unconditional_vs_incumbent']['return_difference_pp']))
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), period=['2016-08-17', '2026-09-24'],
        verification=dict(plan_sha256=plan_sha, result_completion_sha256=result_sha, root_completion_sha256=root_sha,
            input_hash_count=len(pins), prepared_artifacts=7, result_artifacts=38,
            metric_slices_independently_recomputed=45, frozen_training_slices_checked=18,
            all122_period_factors_checked=True, no_strategy_or_feature_imports=True,
            no_new_strategy_runs=True, no_parameter_search=True, inputs_unchanged=True),
        selection=frozen['selection'], guard_details=guard_details, changed_indices=changes,
        support_diagnostics=support,
        metrics=calculated,
        decompositions=decomposition, log_concentration=log_concentration, changed_months=[r for r in periods if r['index'] in changed_indices],
        unchanged_holdings_return_effects=extras, all122_periods=periods,
        attribution_method=dict(monthly_return='Saved NAV(end)/NAV(signal close)-1; open execution includes old-position overnight and rebalancing fees',
            wealth_factor='(1+target_period_return)/(1+reference_period_return)',
            annual_growth_factor='wealth_factor**(1/full_period_years)',
            additive_cagr_contribution='total_CAGR_difference*log(period_wealth_factor)/sum(log(period_wealth_factor)); continuous limit when total log ratio is zero',
            interpretation='Exact accounting decomposition, not a unique causal effect or a counterfactual strategy replay'),
        limitations=['Proxy historical Top32+theme17 universe, retrospectively retrieved source prices, repeatedly reused sample; not pristine OOS/PIT',
            'Same-close zero-cost execution is a research assumption; next-open comparison charges 5bp per side',
            'Original long core, models and support remain unchanged; new direct CDF correction has maximum magnitude .005 and one original episode',
            'Own126 volatility and SPY63 market state have overlapping historical windows; conditional references are a subset of common support, not independent samples; missing state is unknown',
            'Frozen candidate selection used only first88; later34 diagnostics cannot change selection'],
        input_sha256=pins, script_sha256=sha(Path(__file__)))
    content = markdown(report)
    verify(pins)
    DEST.mkdir()
    with (DEST / 'REPORT.json').open('x') as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False); stream.write('\n')
    with (DEST / 'REPORT.md').open('x') as stream:
        stream.write(content)
    verify(pins)
    receipt = dict(status='COMPLETE', created_utc=datetime.now(timezone.utc).isoformat(),
        plan_sha256=plan_sha, result_completion_sha256=result_sha, root_completion_sha256=root_sha,
        artifacts_sha256={n: sha(DEST / n) for n in ('REPORT.json', 'REPORT.md')},
        script_sha256=sha(Path(__file__)), verified_input_count=len(pins), inputs_unchanged=True)
    with (DEST / 'COMPLETION.json').open('x') as stream:
        json.dump(receipt, stream, indent=2); stream.write('\n')
    print(json.dumps(dict(receipt=receipt, changed_indices=changes, decompositions=decomposition,
        full_metrics={a: {p: calculated[a][p]['full122'] for p in SHOW_PATHS} for a in ARMS}), indent=2))



if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan-sha256',required=True)
    parser.add_argument('--result-sha256',required=True)
    parser.add_argument('--root-sha256',required=True)
    args=parser.parse_args()
    main(args.plan_sha256,args.result_sha256,args.root_sha256)
