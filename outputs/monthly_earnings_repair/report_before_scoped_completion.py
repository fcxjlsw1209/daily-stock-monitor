"""Render an honest coverage report from the current completed local build."""
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

P = Path(__file__).resolve().parent


def main():
    status = json.loads((P/'STATUS.json').read_text())
    coverage = json.loads((P/'COVERAGE.json').read_text())
    previous = json.loads((P/'round3_snapshot/STATUS.json').read_text())
    panel = json.loads((P/'MONTHLY_FEATURES.json').read_text())['rows']
    actual_counts = {str(h): dict(Counter(r['reactions'][str(h)]['status']
        for r in panel if str(h) in r['reactions'])) for h in (3, 10, 21)}
    if actual_counts != status['monthly_feature_status_counts']:
        raise ValueError('Monthly feature artifact disagrees with STATUS')
    if sum(r['verified_events'] for r in coverage) != status['verified_events']:
        raise ValueError('Coverage and status event counts disagree')
    comparison = dict(verified_events_before=previous['verified_events'],
        verified_events_after=status['verified_events'],
        added_events=status['verified_events']-previous['verified_events'],
        monthly_availability_before=previous['monthly_availability_counts'],
        monthly_availability_after=status['monthly_availability_counts'])
    tests = {}
    for f in [P/name for name in ('TESTS.txt', 'apple/FINAL_TESTS.txt', 'amazon/FINAL_TESTS.txt',
            'amazon/supplement/TESTS.txt', 'alphabet/FINAL_TESTS.txt', 'meta/FINAL_TESTS.txt', 'jpmorgan/TESTS.txt',
            'berkshire/TESTS.txt', 'walmart/tests.log', 'visa/repair_tests.log', 'pg/TESTS.txt', 'jnj/TESTS.txt',
            'warmup/TESTS.txt')]:
        text = f.read_text()
        count = re.search(r'Ran (\d+) tests?', text)
        if not count or not text.rstrip().endswith('OK'):
            raise ValueError('Missing successful test result: ' + str(f))
        tests[str(f)] = int(count[1])
    missing = [r['symbol'] for r in coverage if r['verified_events'] == 0]
    lines = [
        '财报事件数据修复与覆盖报告',
        '生成时间（UTC）：' + status['generated_at'],
        '',
        '结论：财政期与发布时间的数据处理已修复；当前整池财报因子仍未达到完整回测准入。',
        '这是数据准备和事件计算验证，没有训练新模型、优化收益、改动观察账户或新增交易。',
        '',
        '本轮完成',
        '已验证事件 %d → %d，新增 %d 条。' % (comparison['verified_events_before'], comparison['verified_events_after'], comparison['added_events']),
        '1. 供应商 fiscalDateEnding 原标签与公司实际 fiscal period end 分开保存；以公司身份和经官方核对的发布日期连接。',
        '   Apple供应商核对按日期匹配；实际财政期与供应商标签差异保留于PROVIDER_RECONCILIATION.json，不覆盖原标签。',
        '2. 日期级公告保留 publication_at=null；采用明确的公司所在地日历映射，保守到公告日期之后首个美股交易日收盘才可用。',
        '   地点映射是研究政策，不宣称观测到网站CMS时区。电话会时间、页面修改时间、SEC受理时间均不冒充发布时刻。',
        '3. 经核对的精确新闻线时刻单独保留证据，盘后信息不能进入当天收盘信号。',
        '4. 月度特征从上述可用时点之后的收盘起，观察3/10/21个完整交易日，与SPY相同端点比较。',
        '   这是保守的公布后走势/漂移指标，不包括最初财报跳空，也不是EPS surprise；未成熟或缺数据不补零。',
        '5. 季度身份、原文证据哈希、重复日期、时区/DST、假日/提前收盘、未来数据隔离、源限流停止均有校验。',
        '',
        '官方事件覆盖（包含已取得的研究前预热历史）',
    ]
    for row in coverage:
        if row['collected_records']:
            lines.append('%s：已验证%d条，%s 至 %s；%s。' % (row['symbol'], row['verified_events'],
                row['earliest_release'], row['latest_release'], row['status']))
    lines += [
        '合计已验证 %d 条；精度分布 %s。' % (status['verified_events'], status['precision_counts']),
        '此前已补齐Amazon现有历史区间的缺季：17份官方原稿PDF、5条SEC原稿的公开索引正文观察。',
        '后者保留原文摘录、URL和SHA，不宣称下载了完整SEC HTML；详细来源类型见原始事件表。',
        'Amazon迁移网页的页眉日期存在错误，已采用原稿正文日期；无法核实的保留pending。',
        'Alphabet区分原稿发布主体Google/Alphabet；Meta保留历史FB身份并使用META价格序列，避免接入目前重用FB代码的其他证券。',
        '本轮继续扩展全部未覆盖发行人，另补原有发行人的2011–2014预热；采用多来源官方原稿与明确标记的公开索引观察。',
        'Berkshire早期2010Q4/2011Q4已找到公司8-K正文确认实际发布年报日期，并以年报Q4表核实财季；未采用预告、申报日或股东信日期。',
        '台积电已补齐2010Q4至2026Q2连续63季；微软等早期原稿及雪佛龙历史原稿也已补入。部分SHA对应观察JSON，不冒充完整PDF/HTML字节。',
        'Walmart区分公司财季与13周可比销售统计期；Visa区分公司季度报表与payments-volume统计期；PG财年截至6月。',
        'JNJ采用52/53周实际财季，以官方季度报表核实，原稿与公开索引证据分别标记。',
        '',
        '逐月可用特征',
        '按%d个实际月度信号日期逐项计算，共%d个候选月份，不改股票池。' % (status['monthly_signal_dates'], len(panel)),
        '最新已知事件的观察窗口未完成时保留缺值，不使用未来价格，也不回退至较旧已成熟事件。',
        '月度可用性原分布：' + json.dumps(previous['monthly_availability_counts'], ensure_ascii=False),
        '月度可用性现分布：' + json.dumps(status['monthly_availability_counts'], ensure_ascii=False),
        '3/10/21日反应状态：' + json.dumps(actual_counts, ensure_ascii=False),
        '',
        '离线继续补全所需的线索已整理',
        '旧Yahoo缓存：62个股票身份、1,391条有实际EPS的历史日期线索；仍属未验证提示，不采用其默认盘前/盘后钟点。',
        'SEC缓存：43家发行人、1,953份8-K Item2.02候选；需要核对原始公告/附件，不能直接用申报日期代替财报日期。',
        '这些SEC记录还包括可能的更新、初步业绩、修订，均未自动作为季度财报入库。',
        '',
        '仍缺什么',
        '历史研究池共73个股票身份和17只ETF；ETF的单公司财报指标明确为不适用。',
        '尚无本轮官方事件表的 %d 个股票身份：%s' % (len(missing), ', '.join(missing)),
        '若2016年开始使用此前五年的滚动训练，需逐发行人检查2011年起预热；新上市证券、非季报发行人和更名身份必须单独核验，不能机械伪造季度。',
        '不能把尚未收集财报当作“没有财报”，也不能只保留易补数据的赢家后宣称整池因子有效。',
        '新小因子的收益与Sharpe尚未回测；当前历史股票池和价格本身的研究近似仍然存在。',
        '',
        '访问状态',
        'Alpha Vantage沿用当日限流停止；Yahoo首次探测429后停止；TSMC、Amazon IR、SEC及部分公司IR主站直接访问受限。',
        'Berkshire、JNJ和Visa IR首次403后停止直接请求；可用的独立官方静态CDN与公开索引另行记录来源。',
        '原始失败响应与成功证据均保留，继续工作应复用缓存、遵守Retry-After。',
        '',
        '验证结果',
        '%d份成功日志，累计%d次测试执行；%d条实际事件反应计算预览成功。' % (len(tests), sum(tests.values()), status['preview_status_counts'].get('ready', 0)),
        '%d个受保护文件相对继续研究基线逐一核对，本次研究改动%d。' % (status['protected_files_checked'], status['protected_files_changed']),
        '继续基线PROTECTED_CONTINUATION保留并说明此前已授权heartbeat的40个运行产物变化；未覆盖原始保护清单。',
        '',
        '主要产物',
        'EVENTS.json：统一事件和证据。PROVIDER_RECONCILIATION.json：财政期标签核对。',
        'MONTHLY_AVAILABILITY.json：每个月度候选的已知/缺失/N/A状态，不改变原池。',
        'MONTHLY_FEATURES.json：3/10/21日财报后走势在各月末的实际可用值；MONTHLY_PANEL_AUDIT.json：逐行时间边界核验。',
        'REACTION_PREVIEW.json：所有已验证事件的财报后走势技术预览，非策略收益。',
        'COVERAGE.json：逐股票覆盖。SEC_REVIEW_QUEUE.json、VENDOR_HINTS.json：后续核对清单。',
        '测试日志：' + '、'.join(str(Path(f).relative_to(P)) for f in tests),
        '',
        '离线重建（在工作区根目录）',
        'PYTHONDONTWRITEBYTECODE=1 work/venv/bin/python outputs/monthly_earnings_repair/build.py',
        'PYTHONDONTWRITEBYTECODE=1 work/venv/bin/python outputs/monthly_earnings_repair/audit_panel.py',
        'PYTHONDONTWRITEBYTECODE=1 work/venv/bin/python outputs/monthly_earnings_repair/report.py',
    ]
    (P/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    code = [p for p in P.rglob('*.py') if '__pycache__' not in str(p)]
    validation = dict(unit_and_integration_test_executions=sum(tests.values()), test_logs=tests,
        round3_comparison=comparison, monthly_feature_status_counts=actual_counts,
        test_log_sha256={str(f): hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in tests},
        data_build_verified_events=status['verified_events'],
        protected_files_checked=status['protected_files_checked'], protected_files_changed=status['protected_files_changed'],
        protected_manifest=status['protected_manifest'],
        full_universe_event_factor_ready=status['full_universe_event_factor_ready'], backtest_run=status['factor_backtested'],
        code_sha256={str(p.relative_to(P)): hashlib.sha256(p.read_bytes()).hexdigest() for p in code},
        output_sha256={n: hashlib.sha256((P/n).read_bytes()).hexdigest() for n in
            ['EVENTS.json', 'COVERAGE.json', 'STATUS.json', 'PROVIDER_RECONCILIATION.json', 'MONTHLY_AVAILABILITY.json', 'MONTHLY_FEATURES.json', 'MONTHLY_PANEL_AUDIT.json', 'REACTION_PREVIEW.json', 'REPORT.txt']})
    (P/'VALIDATION.json').write_text(json.dumps(validation, indent=2, ensure_ascii=False)+'\n')
    print('Report saved; %d tests passed.' % sum(tests.values()))


if __name__ == '__main__':
    main()
