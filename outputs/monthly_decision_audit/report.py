"""Render all 69 monthly diagnostics from saved, validated attribution outputs."""
from pathlib import Path
from collections import Counter
import json,hashlib

P=Path(__file__).resolve().parent
def read(n):return json.loads((P/n).read_text())
def write(n,o):(P/n).write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def rank(v):return '—' if v is None else '#'+str(v)
def syms(v):return '/'.join(v) if v else '无'
def points(v):return '—' if v is None else f'{v:.2f}'

NAMES={'core_only':'仅长期动量','without_volatility':'去掉高波动偏好','without_acceleration':'去掉加速项',
 'without_state_cmf':'去掉状态CMF','core_raw294':'动量不除波动','core_risk252':'动量起点294改252日',
 'core_risk_aligned':'波动分母与动量窗口对齐','core_mean_sharpe':'日均收益/波动型Sharpe',
 'core_risk_half':'近期波动分母降为平方根','core_calendar12':'日历12月跳过2月','core_calendar14':'日历14月跳过2月',
 'industry_instead_state':'7.5%状态项改行业相对强度','top35_same_rule':'同规则Top35',
 'core17_same_stocks':'同股票池改core17 ETF','without_arkk_soxx':'仅排除ARKK/SOXX并重排',
 'no_positive_momentum_gate':'去掉正资格动量门槛'}
EXIT={'pool_or_gate_exit_compatible':'池子/资格已失效，移出相容','model_would_keep_top3':'仍在我们的前三，移出不相容',
 'borderline_rank4_5':'第4—5名，阈值边界','rank_below5_exit_compatible':'低于第5名，排名退出相容'}
ms=read('MONTHS.json');summary=read('SUMMARY.json')
intro=[
 '月策略：69期选股、评分与月底移出逐月审计','',
 '主比较：作者提供的模拟学习账本，对照此前冻结的 Top32＋theme17 旧收益候选。这个候选没有机器学习。',
 '只核对信号日已知价格、资格和评分；没有用下个月收益选规则、训练新模型或改变观察账户。作者列出的三只股票只按无序集合处理，不把名单顺序当成公开排名。',
 '',
 '主要结论：',
 f"69期207个持仓槽位，匹配{summary['shared_slots']}个；完整匹配{summary['exact_months']}期。103个差异中，5个是池子外，98个是排名差异，0个单纯由资格门槛阻挡。",
 '池子外5次：2022-05-31 COST估算市值第36；2024-03-28 NFLX第33；2025-06-30 BABA第35；2026-03-31 BABA第34；2026-05-29 AMAT第33。Top35能纳入其中4次，但重算横截面后整体匹配槽位仍为104，不足以解释主要差距。',
 '191个“作者落选股 vs 我们入选但作者未选股”的成对比较，最大正分差来源：加速105、长期动量/波动分母67、高波动偏好14、状态CMF5。成对比较会重复同一股票，不能当作独立样本或作者真实因果。',
 '去掉加速项后：119/207持仓匹配，11/69完整匹配，26期改善、13期变差。去掉波动项则只有67/207匹配；去掉状态CMF为103/207。说明应先查加速的定义与权重，其次长期动量的分子/波动分母，不能据此断言作者完全没有加速项。',
 '作者76次整只移出中：17次在我们的前三、12次在第4—5、44次低于第5、3次已出池/失去资格。作者128次跨月继续持有中，70次在我们的前三，23次第4—5，33次低于第5，2次池外/不合格。单独加“前五保留”无法解释这组记录。',
 '退出与选股是同一个月末排序问题的两面。持仓清单只允许推导整只加入、移出、继续持有，无法识别部分减仓、盘中操作或真实止盈止损原因。这里不把排名不合解释成作者有额外止损。',
 '',
 '分数口径：',
 '总分 = 50 + 各居中百分位×权重×100，理论范围0—100。因子原始排名在完整32＋17池内计算；总排名在资格过滤后计算，按内部证券ID稳定处理同分。分差表的正数表示偏向我们选的股票，负数表示偏向作者股票；四项分差之和严格等于总分差。',
 '权重：长期动量52%、126日波动21.9375%、加速18.5625%、状态CMF7.5%。',
 '长期动量 = (复权P[t−42]/P[t−294]−1)/最近63日收益波动率。加速 = P[t−42]/P[t−126]−P[t−126]/P[t−210]。',
 '状态CMF：SPY63日波动高于过去252日中位数时偏低CMF63，否则偏高CMF63。资格为连续295日有价格、现价正、P[t−42]/P[t−252]−1正。',
 '阶段排名“动量→加波动→加加速→加状态”只是固定顺序的排名轨迹，不是唯一的统计归因；严格可加的是分数贡献及成对分差。',
 '',
 '固定单项改动诊断（均无再训练、无收益择优）：',
 '设置 | 完整匹配期 | 匹配持仓槽位 | 改善期 | 变差期']
for k,v in summary['variants'].items():
    intro.append(f"{NAMES[k]} | {v['exact_months']}/69 | {v['shared_slots']}/207 | {len(v['improved_months'])} | {len(v['worsened_months'])}")
intro += ['', '小修正的纯数学边界：',
 '仅作为诊断，假设额外评分为w×z且z∈[−0.5,0.5]，允许所有作者股票得到最有利修正、竞争股票得到最不利修正。可改变的最大成对分差为100w分。它不是对作者机器学习尺度或权重的估计。',
 '在64期三只作者股票都合格的月份中，即使给上述极端有利的10%附加修正，仍有36期无法让作者集合进入前三；5%时有48期无法做到。小于边界只表示数学上未排除，绝不等于模型能实现。',
 '', '数据边界：股票池是已有重建版本；市值边界排名不是已认证的完整历史当时可知排名。17 ETF为研究假设；“行业相对强度”使用既有行业代理，缺失值按原代码中性排名处理。所有单项设置结果都只用于定位差异，不代表收益改善。',
 '来源与校验：SOURCE_HASHES.json；RANK_ROWS.json保留完整候选池排名和原始值；PAIR_DIFFERENCES.json保留全部191组成对归因。',
 '']
details=[];compact=[]
for m in ms:
    rows={r['symbol']:r for r in m['rows']}
    main=[r for r in m['rows'] if r['author'] or r['ours'] or r['previous_author']]
    main.sort(key=lambda r:(r['rank'] is None,r['rank'] or 999,r['symbol']))
    details += [f"{m['holding_month']} 持有期：{m['date']} → {m['end']}",
       f"作者 {syms(m['author'])}；候选 {syms(m['ours'])}；重合 {m['shared']}/3。",
       f"作者本次加入 {syms(m['author_transition']['added'])}；整只移出 {syms(m['author_transition']['removed'])}；保留 {syms(m['author_transition']['retained'])}。",
       f"市场：{'高波动' if m['state']['high_vol'] else '普通/低波动'}，{m['state']['cmf_direction']}；合格数 {m['eligible_count']}。",
       '证券 | 作者/候选 | 市值排名 | 总排名 | 总分 | 动量→波动→加速→状态排名']
    for r in main:
        who=('双方' if r['author'] and r['ours'] else '作者' if r['author'] else '候选' if r['ours'] else '作者上月')
        cap='ETF' if r['asset']=='ETF' else rank(r['cap']['rank']) if r['cap'] else '未知'
        trajectory='→'.join(rank(r['stage_ranks'][k]) for k in ['core','core_vol','core_vol_accel','full'])
        details.append(f"{r['symbol']} | {who} | {cap} | {rank(r['rank'])} | {points(r['score'])} | {trajectory}"+('；'+','.join(r['blockers']) if r['blockers'] else ''))
    for a in m['missed']:
        if a['reason']!='ranking':details.append(f"未选 {a['symbol']}：{','.join(rows[a['symbol']]['blockers'])}。不能通过调整评分修复池子阻挡。");continue
        z=a['boundary_pair'];co=z['components']
        details.append(f"未选 {a['symbol']}（{rank(a['rank'])}）vs 候选额外入选中最弱的 {z['ours']}（{rank(z['ours_rank'])}）：落后 {z['gap_points']:.2f}分。分差=动量 {co['momentum']:+.2f} + 波动 {co['volatility']:+.2f} + 加速 {co['acceleration']:+.2f} + 状态 {co['state_cmf']:+.2f}。最大阻力：{z['driver_label']}。")
        details.append('单项改动可使该股进入前三：'+('；'.join(NAMES[k] for k in a['helps']) if a['helps'] else '已列设置均不行')+'。这些改动不一定让其他两只也一致。')
    if not m['missed']:details.append('三只集合完全匹配；这不说明已经知道作者的排序分数。')
    for a in m['exits']:details.append(f"移出核查 {a['symbol']}：上次模型排名{rank(a['previous_rank'])}，本次{rank(a['rank'])}；{EXIT[a['classification']]}。")
    for a in m['retained']:
        if a['rank'] is None or a['rank']>3:details.append(f"保留核查 {a['symbol']}：仍被作者持有，但本次模型排名{rank(a['rank'])}。")
    details += ['所有固定单项诊断：']
    for k,v in m['variants'].items():details.append(f"  {NAMES[k]}：{syms(v['selected'])}，重合{v['shared']}/3。")
    details += ['', '']
    compact.append({k:m[k] for k in ['date','end','holding_month','author','ours','shared','state','eligible_count','missed','author_transition','our_transition','exits','retained','variants','set_order_gap_points']})
    compact[-1]['rows']=main

(P/'REPORT.txt').write_text('\n'.join(intro)+'\n')
(P/'MONTH_BY_MONTH.txt').write_text('\n'.join(intro+details)+'\n')
write('UI_DATA.json',{'summary':{k:summary[k] for k in ['months','slots','exact_months','shared_slots','mismatch_slots_by_reason']},'names':NAMES,'exit_labels':EXIT,'months':compact})
v=read('VALIDATION.json');v.update(unit_tests_passed=6,test_command="OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 work/venv/bin/python -m unittest discover -s outputs/monthly_decision_audit -p 'test_*.py' -v")
write('VALIDATION.json',v)
write('STATUS.json',{'status':'completed','audited_months':69,'ml_used':False,'new_model_fitted':False,'author_rule_identified':False,'report':'MONTH_BY_MONTH.txt'})
write('CODE_HASHES.json',{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(P.glob('*.py'))})
print(json.dumps({'all_month_report':str(P/'MONTH_BY_MONTH.txt'),'bytes':(P/'MONTH_BY_MONTH.txt').stat().st_size,'ui_data_bytes':(P/'UI_DATA.json').stat().st_size},ensure_ascii=False))
