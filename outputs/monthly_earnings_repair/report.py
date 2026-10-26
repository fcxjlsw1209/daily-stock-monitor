"""Render the source build and independently checked candidate-scope data gate."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
P=Path(__file__).resolve().parent

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    read=lambda n:json.loads((P/n).read_text())
    status,coverage,scope=read('STATUS.json'),read('COVERAGE.json'),read('SCOPED_READINESS.json')
    if not {'audit_scope.py','audit_panel.py','event_schema.py','research_calendar.py'} <= set(scope['inputs']):
        raise ValueError('Legacy scope audit lacks code-version provenance')
    for name,expected in scope['inputs'].items():
        if sha(P/name)!=expected: raise ValueError('Stale scope audit: '+name)
    if sum(r['verified_events'] for r in coverage)!=status['verified_events']:raise ValueError('Coverage mismatch')
    log=(P/'TESTS.txt').read_text();tests=re.search(r'Ran (\d+) tests?',log)
    if not tests or not log.rstrip().endswith('OK'):raise ValueError('Regression checks did not pass')
    events=read('EVENTS.json');sources=dict(Counter(e.get('source_kind','unspecified') for e in events))
    ready=scope['scoped_post_known_drift_data_ready']
    lines=['财报数据补全与逐候选验证','生成时间（UTC）：'+status['generated_at'],'',
        '结论：现有研究池的月度与训练候选日期，已通过常规财报公布后走势数据准入。' if ready else '结论：数据补全仍在进行，逐候选准入尚未通过。',
        '没有训练新模型或产生新收益回测；没有改动观察账户、冻结交易、权重或候选股票池。','',
        '实际范围','共%d个股票身份：月度池%d个，训练池另有%d个；另有%d只ETF。'%(status['all_scope_equities'],status['universe_equities'],len(status['training_only_equities']),status['etfs_not_applicable']),
        '训练专有身份：'+', '.join(status['training_only_equities']),
        '已核验常规财报事件：%d条。'%status['verified_events'],
        '证据包括发行人原始PDF/HTML、明确标注的官方原稿公开索引，以及单独说明来源的发行人报告镜像；索引不冒充原始文件字节。',
        '来源类型：'+json.dumps(sources,ensure_ascii=False),'','逐候选可用性']
    for name,label in [('monthly','月度'),('training','训练')]:
        panel=read(name.upper()+'_FEATURES.json')['rows']
        counts={str(h):dict(Counter(r['reactions'][str(h)]['status'] for r in panel if str(h) in r['reactions'])) for h in (3,10,21)}
        if counts!=status[name+'_feature_status_counts']:raise ValueError('Feature counts mismatch: '+name)
        lines.extend([label+'候选状态：'+json.dumps(scope['counts'][name],ensure_ascii=False),label+'3/10/21交易日反应状态：'+json.dumps(counts,ensure_ascii=False)])
    lines.extend(['','缺口与边界核验',
        '共检查%d个需复核项目，未解决%d个；候选价格/成交量/SPY窗口错误%d个。'%(scope['issues_reviewed'],scope['unresolved_count'],len(scope['invalid_candidate_price_windows'])),
        '最近120天有公告不足以证明完整：同时检查中间漏季、档案端点及供应商未匹配日期。',
        '缺值不补零、不改变原池。ETF不适用；首份已核实上市后常规公告之前保持缺值；最新事件未完成3/10/21日观察时继续等待。',
        '尚未发布的下一季公告仅作为审计边界，不把后来的信息用于过去选股。',
        '详细证据：SCOPED_READINESS.json、SCOPE_RESOLUTIONS.json、BOUNDARY_OBSERVATIONS.json。'])
    if scope['unresolved']:lines.append('未解决分布：'+json.dumps(dict(Counter(r['symbol']+':'+r['kind'] for r in scope['unresolved'])),ensure_ascii=False))
    lines.extend(['','财政期、时点与日历',
        '保留供应商财政期原标签，另存发行人实际期末；52/53周财年不强行改为自然季度。',
        '日期级公告不臆造时分秒，保守至公告日之后首个XNYS交易日收盘才可用。',
        '电话会、网页修改、SEC受理时间均不替代实际发布时刻。修订归档保留版本限制，只使用独立核实的事件日期和财季元数据。',
        '按官方停市通告修正2025-01-09国丧休市；未修改全局日历或依赖版本。',
        '反应值为已知时点后的3/10/21个完整交易日收盘走势及相同端点SPY比较，不包括原始财报跳空。','',
        '准入的准确含义',
        '仅限此冻结候选面板的常规财报公布后走势特征，不是全球所有公告、初步业绩或修订的穷尽归档。',
        '不声称精确复现历史EPS surprise、原作者财报因子或首发时分秒。官方现存归档和索引也不是当时冻结的新闻源。',
        '历史股票池与缓存价格本身的研究近似仍保留。新因子是否提高收益须另行固定研究方案再验证。','',
        '验证','中央回归测试：%s项通过；各补表另有原稿、季度连续性和SHA测试日志。'%tests[1],
        '两面板均通过重复候选、身份、最新已知事件、未来价格和缺值不补零核验。',
        '受保护文件%d个，研究改动%d个。'%(status['protected_files_checked'],status['protected_files_changed']),
        '来源访问失败及停止记录保留，当天403/429的来源不重复请求。','',
        '已核验档案覆盖（候选区间之外未收季度不冒充已补齐）'])
    for r in coverage:lines.append('%s：%d条，%s 至 %s。'%(r['symbol'],r['verified_events'],r['earliest_release'],r['latest_release']))
    lines.extend(['','工作区根目录离线重建顺序：']+['work/venv/bin/python outputs/monthly_earnings_repair/'+n for n in ['build.py','audit_panel.py','audit_training_coverage.py','assemble_scope_resolutions.py','audit_scope.py','report.py']])
    (P/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    names=['EVENTS.json','COVERAGE.json','STATUS.json','SCOPED_READINESS.json','REPORT.txt','MONTHLY_FEATURES.json','TRAINING_FEATURES.json','MONTHLY_PANEL_AUDIT.json','TRAINING_PANEL_AUDIT.json']
    v=dict(central_tests_passed=int(tests[1]),central_test_log_sha256=sha(P/'TESTS.txt'),verified_events=status['verified_events'],scoped_post_known_drift_data_ready=ready,
        full_global_earnings_archive_complete=False,backtest_run=False,unresolved_count=scope['unresolved_count'],protected_files_checked=status['protected_files_checked'],protected_files_changed=status['protected_files_changed'],
        source_type_counts=sources,code_sha256={str(p.relative_to(P)):sha(p) for p in P.glob('*.py')},output_sha256={n:sha(P/n) for n in names})
    (P/'VALIDATION.json').write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(dict(events=status['verified_events'],scoped_ready=ready,unresolved=scope['unresolved_count'])))

if __name__=='__main__':main()
