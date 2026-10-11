"""Record useful and unsuccessful hypotheses without promoting fitted winners."""
from engine import *


def read(name):return json.loads((P/name).read_text())
def names(s):return '/'.join(t.replace('research_equity_','') for t in s)


def run():
    r=read('ALL_RESULTS.json');orders=read('ALL_ORDERS.json');v=read('VALIDATION.json');s=read('FINAL_SELECTION.json')
    baseline='theme17__all_retrospective__stock_only__lag0_cost0';portfolio='theme17__all_retrospective__portfolio__lag0_cost0';positive='EXPANDED_theme17__all_retrospective__positive_only__lag0_cost0';encoding='ENCODING_theme17__all_retrospective__robust_scale__lag0_cost0';core='CORE_WEIGHT_theme17__all_retrospective__core40__lag0_cost0'
    compared=[('前轮参考规则，联合求解对照',baseline),('按组合波动/相关性选三只',portfolio),('取消动量前半门槛，保留正动量',positive),('数值稳健标准化，替代百分位排名',encoding),('核心权重30%/40%中本轮最高',core)]
    lines=['月策略继续复现：联合选择、资格条件、因子刻度和核心权重','2026-10-09','',
        '结果：完成24个拟合模型、72组实际三只持仓回放。没有超过前轮63.46%年化、1.704 Sharpe，作者80.64%仍未复现。没有将软组合训练时的混合持仓收益当作可交易策略业绩。',
        '唯一值得单独保留的局部改善：提前一个交易日生成信号、月底收盘成交并计双边0.10%成本时，主题ETF池组合选择版本年化49.25%、Sharpe1.415；前轮固定个股排序为45.20%、1.334。该组合版本最大回撤也扩大，且同收盘零成本口径更差，不能据此宣布全面改进。','',
        '一、同口径对照',
        '实际回放2016-08-31—2026-09-24；作者图2016-08-17开始。月末收盘信号/收盘价代理成交、零成本，之后一个月固定复权价格单位，只做多，无杠杆ETF。',
        '方法 | 年化 | Sharpe | 最大回撤 | 严格候选命中月份',
        '作者图中月策略标注 | 80.64% | 1.867 | -34.46% | 未知']
    for label,key in compared:
        z=r[key];lines.append(f"{label} | {z['cagr_252']:.2%} | {z['sharpe']:.3f} | {z['max_dd']:.2%} | {z['strict_hits']}/121")
    lines+=['严格命中指选出的三只属于该假设下累计前95%拟合权重的备选组合；不是命中真实交易，也不是统计置信水平。放宽资格后备选集合变化，命中数不能直接当成跨假设的胜率。','',
        '二、执行对照：将信号提前一天',
        '方法 | 年化 | Sharpe | 最大回撤']
    for label,key in [('前轮固定个股排序',baseline),('本轮组合选择',portfolio)]:
        z=r[key.replace('lag0_cost0','lag1_cost10')];lines.append(f"{label} | {z['cagr_252']:.2%} | {z['sharpe']:.3f} | {z['max_dd']:.2%}")
    lines+=['两者都在月底换仓，持有接下来的月份，没有把月底排序用于该月月初。提前一天是对执行敏感性的测试，不是对作者使用未来月份数据的假设。',
        '这一局部改善来自反复看过的历史图像和价格，仍然是事后研究。未加入原观察账户。','',
        '三、这轮为什么测试这些变化',
        '诊断1：2026年5月候选中的INTC，入场前长期动量百分位约0.959，满足资格，却被前轮风险/量价修正压到第12名。2025年9月候选中的ORCL、PLTR、TSLA也并未被排除，而是排名不够。这首先是排序解释的问题。',
        '组合选择：用三只组合的过去63日协方差，计算分散程度=1−组合波动/个股平均波动，以及三只波动率最大/最小值的对数差。两项各在当时所有合格三只组合中做排名；保留50%动量核心，个股次级项绝对权重合计≤40%，组合项≤10%。没有按未来收益选择组合。',
        '组合项的系数可正可负，由图像拟合决定。原ETF池的全样本模型甚至偏向相关性较高的组合；主题池学出的组合项接近零。因此数据并不支持“作者一定做了分散化”这一推断。',
        '资格门槛：作者月策略文字没有说必须处于长期动量前半。之前无分位门槛的图像候选路径中，原ETF池80/363、主题ETF池81/363个股票-月份落在前半之外，分别涉及61和62个月。那些仍是事后候选，不是已确认作者交易。本轮完整重建正动量候选，取消分位门槛后，最优规则实际收益没有提高。',
        '因子刻度：百分位排名可能压缩异常强势股与普通强势股的差距。因此对同样原始因子，按当日股票池中位数/MAD标准化，MAD=0回退标准差，截到±3后除6。没有新增未来信息或证券身份记忆。最高年化50.48%，低于参考结果。',
        '动量核心权重：用户最初要求最多50%，并非固定50%。对核心30%和40%分别测试，次级因子L1上限为1−核心权重；其余交易和资格条件相同。该方向本轮最高年化52.17%，没有改善目标。部分次级因子也有动量含义，显式核心比例不等于总经济风险暴露。','',
        '四、月份组合的实际差异',
        '以下仍是图像假设，不是作者已确认的买卖。',
        '月份 | 原图前三组备选 | 前轮参考规则 | 组合选择 | 数值标准化 | 核心40%']
    meta={q['month']:q for q in json.loads((OLD/'BAGS_theme17.json').read_text())};details=[]
    for month in ['2020-08','2025-09','2026-04','2026-05','2026-06']:
        m=meta[month];selected={}
        for key in [baseline,portfolio,encoding,core]:selected[key]=next(o['holdings'] for o in orders[key] if o['date']==m['signal_date'])
        candidates=[q['holdings'] for q in m['candidate_triples'][:3]]
        lines.append(f"{month} | {'；'.join(names(q) for q in candidates)} | {' | '.join(names(selected[k]) for k in [baseline,portfolio,encoding,core])}")
        details.append(dict(month=month,candidate_triples=candidates,actual_rule_choices=selected,verified_author_holdings=False))
    lines+=['高年化的参考规则仍不能解释关键月份的全部持仓；本轮的新假设也没有补上这一证据链。没有从所有候选中挑出一条80%事后路径后，冒充可以事前运行的选股规则。','',
        '五、模型学习和验证',
        '新求解器先把三只组合的选择做成可微的软分布，联合反推连续复利净值与回撤；拟合结束后每个月必须按同一组系数只选一个三只组合。报告只使用这些实际组合的回放。',
        '损失函数延续前轮：全局log净值RMSE＋4×月内形状logRMSE＋0.5×月度最差回撤RMSE。优化器不接收CAGR或Sharpe目标；但全样本图像本身含事后信息，所以仍不能当独立样本外业绩。',
        '两套ETF名单、2016—2021校准和全样本校准、原门槛/正动量门槛、排名/数值标准化、核心权重30/40/50分别保存结果。第一阶段用两次初始化、逐步收紧软选择温度，再做固定次数硬组合求解；无按月单独调权重。',
        f"全部{v['replays']}个回放共{v['monthly_events']}次月度事件，固定股数和交易费独立重算；净值最大相对误差{v['independent_nav_max_relative_error']:.3g}，年化/Sharpe/回撤最大数值差{v['metric_max_difference']:.3g}。",
        f"保存{v['soft_optimization_states_saved']:,}个软优化状态、{v['hard_refinement_trials_saved']:,}个硬选择细化候选。梯度（含连续复利和回撤）与有限差分核对通过；组合唯一选择、无效候选屏蔽、权重上限也通过单元测试。",
        f"截掉2020-07-01以后的数据后，旧/新因子和资格结果最大差异{v['feature_prefix_max_difference']:.1f}，组合相关性特征差异{v['portfolio_correlation_prefix_max_difference']:.1f}。这验证输入的时间顺序，不证明全样本拟合系数当时已经可得。",
        f"484个新候选文件校验和通过；94个原账户保护文件、38个数据来源、54个上轮研究文件均未变。",'',
        '六、当前状态和可保留结论',
        '目标未达成：不能声称已复现作者80.64%。目前63.46%仍仅是全样本反推参考。',
        '本轮排查削弱了三种简单解释：只放宽动量资格、只换标准化刻度、只改核心占比，都未提高完整样本结果。它们没有被数学上证明无效，只是在这组已记录的假设和优化预算下未成功。',
        '组合选择的提前一天对照略好，可留作后续研究分支，但原观察账户和冻结规则不变。',
        '下一步仍应围绕能解释多个关键月份的统一排序关系，而不是将不确定的备选持仓直接当真值。原作者ETF名单、全美股Top32与我们指数并集Top32代理的差异、具体模型训练目标仍未确定。',
        '文件：ALL_RESULTS.json、ALL_ORDERS.json、ALL_CURVES.csv.gz、FINAL_SELECTION.json、CLUE_AUDIT.json、VALIDATION.json、TESTS.txt；各阶段配置、模型、候选和拟合轨迹独立保存。',
        '复跑顺序：research.py → expanded.py → encoding.py → core_weight.py → verify.py → report.py。使用PYTHONPATH=work/research-plot-deps OPENBLAS_NUM_THREADS=1 work/venv/bin/python。']
    (P/'REPORT.txt').write_text('\n'.join(lines)+'\n');dump('CLUE_AUDIT.json',details)
    dump('CODE_HASHES.json',{str(f):base.sha(f) for f in P.glob('*.py')})
    dump('DELIVERY.json',dict(target_reproduced=False,best_fullsample_reference=r[baseline],portfolio_prior_session_case=r[portfolio.replace('lag0_cost0','lag1_cost10')],
        tested_models=v['models'],replays=v['replays'],original_accounts_unchanged=True))
    print('REPORT_COMPLETE',flush=True)


if __name__=='__main__':run()
