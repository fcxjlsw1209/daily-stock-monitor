"""Format the completed 44-period universe substitution experiment."""
from pathlib import Path
import json,hashlib
from datetime import datetime,timezone

P=Path(__file__).resolve().parent
def read(name):return json.loads((P/name).read_text())
def dump(name,x):(P/name).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')

def main():
    r=read('RESULTS.json');rows=read('MONTHLY_COMPARISON.json');audit=read('POOL_AUDIT.json')
    labels={'author__supplied_ledger':'作者给定持仓回放','latest_multilayer__published':'原复现模型＋新股票池',
            'return_leader__published':'此前收益优先候选＋新股票池','simple_leader__published':'此前简洁候选＋新股票池',
            'holdings_leader__published':'此前简洁匹配候选＋新股票池','pure_mom252_skip42__published':'纯动量252日跳过42日',
            'retrospective44_match_leader__published':'本轮44期事后匹配搜索第一名',
            'calibration23_match_leader__published':'仅前23期选定参数的规则'}
    lines=['新历史月末Top32：44期选股与作者持仓对照','',
      '结论：尚未完整复现作者组合。固定原模型、更换股票池后，仍然15/44期三只全中，90/132个持仓位置相同；在作者三只都在池内的40期中，全中15期、重合84/120个位置。',
      '时间：2023-01-31收盘—2026-09-24收盘；44次月度选择，最后一段不足整月。全部以相同区间、零成本、月末等权买入、月内固定份额回放。',
      '股票池使用已下载的全球日期排名，筛选当时美国交易所上市发行人，合并同公司股类。无指数成分限制。ETF沿用各模型原core17/theme17假设；作者完整ETF名单仍未知。','',
      '|方案|三只全中|持仓重合|年化收益|每日Sharpe|最大日终回撤|','|---|---:|---:|---:|---:|---:|']
    for key,label in labels.items():
        v=r['runs'][key];m=v['matches'];s=v['performance']
        match='基准记录' if key.startswith('author') else f"{m['exact']}/44"
        shared='基准记录' if key.startswith('author') else f"{m['shared']}/132"
        lines.append(f"|{label}|{match}|{shared}|{s['cagr']:.2%}|{s['sharpe']:.3f}|{s['max_drawdown']:.2%}|")
    s=r['qqq'];lines.append(f"|QQQ买入持有|—|—|{s['cagr']:.2%}|{s['sharpe']:.3f}|{s['max_drawdown']:.2%}|")
    lines+=['','四期硬性股票池差异（没有按作者持仓放宽Top32）：',
      '2024-03-28：NFLX第33名；2025-06-30：BABA第35名；2026-03-31：BABA第34名；2026-05-29：AMAT第33名。',
      '因此这个股票池最多允许40/44期三只全中、128/132个位置相同。其余25个池内但未全中的月份，不能用这四个边界差异解释。','',
      '固定规则对照：五套已有参数都先在旧股票池重算，逐期持仓与原已保存结果44/44一致，再更换股票池。原多层复现规则只改变了两个月的组合：',
      '2024-08-30：NVDA、LLY、AVGO → NVDA、LLY、TSM；',
      '2025-05-30：PLTR、NFLX、TSLA → PLTR、BABA、NFLX。',
      '年度Ridge沿用已保存参数，只重算新候选的特征与排名；训练仍使用原股票池，未重训。用此设计隔离更换股票池的影响。','',
      f"补充搜索：{r['search_trials']:,}组有限规则，核心是189/252/294日动量、跳过最近42日，辅以波动、量价、行业相对强弱等；核心权重至少50%。选择目标是三只全中数，再比较持仓重合数，没有按收益挑选。",
      '44期事后匹配搜索最好12/44期，未超过已有多层模型15/44；该规则为60%动量252跳42＋30%波动63＋10%下行波动63，使用池内百分位。',
      '只用前23期选择的规则为60%动量294跳42＋30%下行波动63＋10%距MA200，前23期全中7次，后21期全中4次。全部历史已被研究过，不称独立样本外。','',
      '数据修正与检查：',
      '1. 2023-02-28榜单及旧上市注册表错误地保留LTM。官方披露其2020年退市、2024-07-25重新上市，因此剔除该日LTM，并从原第33名顺延补位。证据见LTM_SOURCE.json。原下载原文未改。',
      '2. BRK-B映射到现有BRK.B价格；补取SHEL、SKHY历史行情。SPCX、SKHY上市时间太短，不符合原295日预热要求，不能被选择。',
      '3. SKHY的上市注册表有同日退市冲突。另跑了完全排除SKHY并补入第33名的敏感性对照；原多层复现规则持仓和收益未变。',
      '4. 所有组合逐日净值通过另一种月内固定份额计算独立核对；4项测试通过；332个原观察账户和研究保护文件未变。',
      '5. Sharpe为每日算术收益均值/样本标准差×√252，无风险利率0；CAGR按实际日历年数。作者回放的全部44个月收益也核对到原账本舍入精度。','',
      '限制：这是历史描述性组合回放和规则匹配研究；同收盘信号/同收盘成交只是与作者对齐的比较口径，不证明可按该价实盘执行。公开历史市值榜不保证等于作者Yahoo快照估算口径，也未证实原始发布日期版本。','',
      '逐月组合见MONTHLY_DETAILS.txt；机器可读为MONTHLY_COMPARISON.json；每日净值为DAILY_EQUITY.json；每套方案的_HOLDINGS.json和_TRADES.json保留了假设交易。未改变任何实盘、模拟账本或定时观察策略。']
    (P/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    selected=['latest_multilayer','return_leader','simple_leader','retrospective44_match_leader','calibration23_match_leader']
    by={(x['date'],x['model']):x for x in rows if x['universe']=='published'}
    detail=['44期逐月选股对照；零交易成本；日期为月末选择日期。','']
    for a in audit:
        detail += [a['date'], '作者：'+', '.join(a['author'])]
        for model in selected:
            x=by[a['date'],model]
            detail.append(model+': '+', '.join(x['selected'])+f" | 匹配{x['shared']}/3")
        if a['missing_from_pool']:detail.append('作者不在Top32的持仓：'+', '.join(a['missing_from_pool']))
        detail.append('')
    (P/'MONTHLY_DETAILS.txt').write_text('\n'.join(detail))
    originals=json.loads((P.parent/'monthly_supplied_ledger_audit/PERIODS.json').read_text())
    periods={a['start']:a for a in originals}
    trades=read('author__supplied_ledger_HOLDINGS.json')
    eq=read('DAILY_EQUITY.json');nav={x['date']:x['author__supplied_ledger'] for x in eq}
    errors=[abs((nav[periods[x['signal_date']]['end']]/nav[x['signal_date']]-1-periods[x['signal_date']]['reported_return'])*10000) for x in trades]
    assert max(errors)<=.5+1e-8
    validation=read('VALIDATION.json');validation.update(author_months_rounding_matches=len(errors),max_author_return_rounding_error_bp=max(errors),unit_tests_passed=4)
    dump('VALIDATION.json',validation)
    sources=read('SOURCE_HASHES.json')
    for f in [P/'LTM_SOURCE.json',P/'test_replay.py',Path(__file__)]:sources[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
    dump('SOURCE_HASHES.json',sources)
    dump('STATUS.json',dict(status='COMPLETED_44_PERIOD_DIAGNOSTIC_NOT_FULLY_REPLICATED',updated_utc=datetime.now(timezone.utc).isoformat(),
        fixed_rule_best_exact=15,periods=44,author_covered_months=40,frozen_observers_changed=False,
        full_historical_pool_certified=False,new_rules_tested=r['search_trials']))
    print('REPORT READY; author return rounding checks',len(errors),'max bp',max(errors))


if __name__=='__main__':main()
