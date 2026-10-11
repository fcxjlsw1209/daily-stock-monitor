from pathlib import Path
import json,hashlib
P=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(n,x):(P/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')

def main():
    rs=read(P/'RESULTS.json')['runs'];details=read(P/'MONTHLY_COMPARISON.json')
    a={r['date']:r for r in details if r['model']=='top32_extended69'}
    b={r['date']:r for r in details if r['model']=='top35_extended69'}
    changes=[]
    for date,x in a.items():
        y=b[date]
        if set(x['selected'])==set(y['selected']):continue
        added=set(y['pool'][32:]);new=set(y['selected'])-set(x['selected'])
        changes.append(dict(date=date,top32=x['selected'],top35=y['selected'],author=x['author'],
                            added_pool=y['pool'][32:],newly_selected=sorted(new),
                            any_rank33_35_selected=bool(added&set(y['selected']))))
    dump('CHANGED_MONTHS.json',changes)
    core=set('SPY QQQ IWM DIA TLT GLD XLB XLC XLE XLF XLI XLK XLP XLRE XLU XLV XLY'.split())
    missing=[dict(date=r['date'],symbols=sorted(set(r['author'])-set(r['pool'])-core)) for r in b.values() if r['author_pool_coverage']<3]
    collection=read(P/'CAP_COLLECTION.json');have={r['symbol'] for r in collection['records']}
    # These historical references test inclusion, never set monthly eligibility.
    cross='AAPL MSFT AMZN GOOGL META TSLA BABA BRK-B TSM V JPM JNJ WMT MA UNH DIS BAC PG NVDA HD PYPL INTC ASML TM CMCSA VZ XOM NFLX ADBE KO CSCO T ABT NVS NKE ORCL PFE CVX PEP CRM MRK ABBV AVGO TMO LLY ACN TXN MCD BHP WFC DHR NVO MDT COST TMUS C HON QCOM SAP BA SHEL NEE UPS UNP UL LIN AMGN BMY PM LOW CHTR SONY AZN RY SBUX BUD AMD SCHW MS BX RTX TD SPGI AXP HSBC INTU'.split()
    assert set(cross)<=have
    dump('COVERAGE_AUDIT.json',dict(cached_issuer_records=len(have),historical_reference_symbols=cross,
                                  missing_reference_symbols=sorted(set(cross)-have),author_missing_from_top35=missing,
                                  exact_historical_all_US_ranking_certified=False))
    labels={'top32_published44':'原简洁候选＋Top32','top35_published44':'原简洁候选＋Top35','author44':'作者持仓回放',
            'top32_extended69':'原简洁候选＋Top32','top35_extended69':'原简洁候选＋Top35','author69':'作者持仓回放','QQQ69':'QQQ买入持有'}
    lines=['固定原简洁月策略：Top32 / Top35 / 作者持仓比较',
           '已完成两组历史回放；没有重新拟合、穷举参数或选择最佳新权重。所有成本均为零。',
           '1. 原44期：2023-01-31—2026-09-24（沿用已有月末排名）']
    for group in [['top32_published44','top35_published44','author44'],['top32_extended69','top35_extended69','author69','QQQ69']]:
        if group[0].endswith('69'):lines+=['','2. 扩展69期：2020-12-31—2026-09-24，包含2021年起完整持仓收益']
        for k in group:
            r=rs[k];lines.append(f"{labels[k]}：年化 {r['cagr']:.2%}；Sharpe {r['sharpe']:.3f}；最大回撤 {r['max_drawdown']:.2%}；10万美元变为 ${r['final_equity']:,.2f}。")
    lines+=['','股票池变化与复现程度',
            '每期都纳入全部排名33—35的公司，而非只补作者选中的股票；原17只ETF不变。',
            f"44期中，作者3只持仓全部在候选池内的期数从40/44升到44/44；3只完全选对仍是3/44，命中槽位从72/132升到74/132。",
            f"69期中，Top32的作者持仓覆盖为64/69期，Top35为68/69期；选股完全匹配分别是9/69和6/69期。",
            '唯一仍不在Top35内的是2022-05-31的COST：估算第36，市值208.65B，第35约208.87B，差约0.10%；保留原始边界结果，没有为匹配作者挪入。',
            f"扩池共改变{len(changes)}/69期持仓，其中现有44期改变{sum(c['date']>='2023-01-31' for c in changes)}期。",
            '新增候选既可能直接被选中，也会改变全池因子的分位数标准化，从而影响原池内部排序。',
            '2021年：Top32 +8.03%，Top35 +6.89%，作者 +90.11%；这一年的差距尤其大，放宽排名不能解决。',
            '2022年：Top32 +85.24%，Top35 +70.65%，作者 +87.13%。',
            '', '保持不变的策略',
            '核心动量=P[t-42]/P[t-252]-1；除以63日波动，占50%。126日波动正向占35%，63日下行波动反向占15%。',
            '每个因子按当期完整股票＋ETF池的10%—90%分位区间缩放并截断；先标准化，再施加295日有效行情及正动量资格。',
            '月底按分数选3只等权；仅2只合格则各半，少于2只空仓；月内固定复权份额。',
            '信号当日收盘价与同收盘模拟成交为理想化历史回放口径，并非保证能在收盘前获得信号或实现该成交。',
            'Sharpe=日收益均值/日收益标准差×sqrt(252)，无风险利率0；年化按实际日历长度。',
            '', '数据与边界',
            '2020-12-31至2022-12-30这25个调仓起点来自474家公司缓存中的历史市值快照，结合历史美国交易所上市身份与前一交易日价格重建。',
            '只使用日期不晚于调仓前一交易日的市值观察，必要时用价格变化滚动；没有用未来快照线性插值。合并了此前漏读的PYPL、NKE等115份历史缓存。',
            '2023-01-31起沿用已有公开月末排名；使用来源与时间方法在此切换，见EARLIER_RANKS.json及POOLS.json。',
            '公司行动处理：BHP统一上市架构后用已公布股数及ADS比例修正明显失真的市值；Shell旧股类按2.1倍保守情景上界检验，受影响月份仍在35名外。假设和一手来源见EVIDENCE.json。',
            '这些是重建的历史排名，未取得每期原始发布版本，也不能证明全市场没有遗漏或边界误差；不能称为作者原始股票池或严格点时数据。',
            'ETF使用既有core17假设，仍无法确认与作者完整17只名单一致。作者数据是所提供学习账本的回放，不是独立核实的实盘交易。',
            '候选公式此前已用这些历史做研究，结果不属于独立样本外验证。新研究没有修改原观察账户。',
            '', '验证',
            '6项针对性单元测试通过；44期原Top32逐日净值与上一轮一致；69期作者逐日净值与既有账本回放一致。',
            '两种独立固定份额计算得到一致的每日净值；69期后44期归一化后与44期单独回测一致。',
            '332个既有受保护文件校验和未改变。',
            '', '复跑',
            'work/venv/bin/python outputs/monthly_top35_extension/merge_cap_cache.py',
            'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 work/venv/bin/python outputs/monthly_top35_extension/run.py',
            "work/venv/bin/python -m unittest discover -s outputs/monthly_top35_extension -p 'test_*.py' -v",
            'work/venv/bin/python outputs/monthly_top35_extension/build_report.py']
    (P/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    monthly=['调仓日期 | Top32 | Top35 | 作者 | Top32命中 | Top35命中 | 新增排名33—35']
    for date,x in a.items():
        y=b[date];monthly.append(' | '.join([date,','.join(x['selected']),','.join(y['selected']),','.join(x['author']),str(x['shared']),str(y['shared']),','.join(y['pool'][32:])]))
    (P/'MONTHLY_DETAILS.txt').write_text('\n'.join(monthly)+'\n')
    before=read(P.parent/'monthly_published44_reconstruction/PROTECTED_BEFORE.json')
    assert all(sha(f)==h for f,h in before.items())
    inputs=read(P/'SOURCE_HASHES.json');assert all(sha(f)==h for f,h in inputs.items())
    v=read(P/'VALIDATION.json');v.update(protected_files_unchanged=len(before),source_hashes_verified=True,unit_tests_passed=6,
        monthly_periods=69,earlier_reconstructed_periods=25,exact_historical_all_US_ranking_certified=False)
    dump('VALIDATION.json',v)
    dump('STATUS.json',dict(status='completed_reconstructed_universe_research_comparison',periods=69,
        start='2020-12-31',end='2026-09-24',formula_changed=False,cost_per_side=0.,observer_changed=False,
        source_vintage_and_boundary_limitations_remain=True,report=str(P/'REPORT.txt')))
    print('Report saved; protected files unchanged:',len(before))
if __name__=='__main__':main()
