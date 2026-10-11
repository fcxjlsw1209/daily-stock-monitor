"""Readable short-window reconstruction evidence and a standalone research plot."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parent;O=P.parent
sys.path.insert(0,str(O/'monthly_path_constraints'))
from study import load,read,sha,I

def main():
    s=read(P/'SELECTED_RULE.json');st=read(P/'RULE_STABILITY.json');a=read(P/'MONTHLY_AMBIGUITY.json')
    confidence=read(P/'ANCHOR_STRESS.json');con=read(P/'DECISION_CONTRIBUTIONS.json')
    lines=['从较高确定性的短段反推月策略','',
        '选段：2022-07-29收盘至2022-10-31收盘，对应8、9、10月持仓。先按既有9条反推路径的持仓一致性选段，没有按收益或规则易拟合程度选段。',
        '本轮只复盘这一短段，并将固定规则延伸到紧接的11月一次；没有运行全期回测。',
        '这里的确定性是现有价格、代理股票池、图片精度与起始净值假设下的相对识别稳定性，不是持仓真实概率。','',
        '一、哪些交易线索比较可信',
        '10月：CVX、XLE、XOM是最强线索。9条既有完整路径全部一致；重新穷举当时21只合格标的的1330个三只组合，横轴与统一起始净值各做3种扰动，共9种情况下最优组合仍相同。8种情况下近似最优集合仅这一组，另1种有3组，XOM在全部近似集合中。',
        '8月：XLE、XOM较稳定，第三只存在歧义。固定原起始净值、横轴微调后的共同近似候选为AAPL/XLE/XOM、AZN/XLE/XOM、MRK/XLE/XOM；增加起始净值不确定性后歧义扩大，不能把第三只锁死。',
        '9月：医疗/防御方向较稳定，具体三只较弱。保留10个在3种横轴微调下都接近最优的候选，不把旧记录中的ABBV/JNJ/LLY当唯一标签。','',
        '二、一套能解释三个月的固定规则候选',
        '在上述3×10×1=30条短段持仓路径中，检查动量纯线性、动量至少50%加量价/波动/行业的同一套系数。',
        '180个固定规则可行性检查中，2个找到严格共同排序并且连续短段图片未被既有宽松筛查排除；它们给出同一条持仓路径。没有按盈利最大化挑选。',
        '持仓月份 | 月末决策日期 | 候选持仓 | 当月收益（零成本）']
    for o in s['orders']:
        lines.append(f"{pd.Timestamp(o['end']).strftime('%Y-%m')} | {o['start']} | {' / '.join(o['holdings'])} | {o['period_return']:+.2%}")
    lines.extend(['',
        '交易衔接：8月设为起始持仓MRK/XLE/XOM；8月底保留MRK、卖出XLE/XOM、买入LLY/UNH；9月底卖出医疗三只，买入CVX/XLE/XOM。每次回到三只等权，月内固定股数。',
        f"三个月累计{s['check']['window_return']:+.2%}。自窗口起点算最大回撤{s['check']['drawdown_from_window_start']:.2%}；考虑此前账户高点，该段最深回撤{s['check']['global_peak_aware_drawdown_in_window']:.2%}。这些是本候选的回放结果，不是作者确认的月度收益。",'',
        '规则的含义：中期趋势仍为主，年度动量和风险调整年度动量共同占主要权重；再以波动形态、资金流、量比和均线位置调节排序。全部动量跳过最近42个交易日，未加入日策略的回撤硬门槛。',
        '当前具体参数（只供研究，不是已恢复的作者参数）：'])
    for n,w in zip(s['feature_names'],s['fit']['weights']):
        if abs(w)>1e-7:lines.append(f'  {n}: {w:+.6f}')
    lines.extend(['采用各调仓日候选池的10%/90%分位缩放并截断，居中到[-0.5,0.5]。动量权重和为50%，侧因子绝对权重和为50%。','',
        '三、逐月决策证据'])
    for d in con:
        lines.append(d['date']+'：前三 '+' / '.join(d['picked'])+'；随后三名 '+' / '.join(d['runners_up'])+'。')
        for e in d['entries']:
            strong=sorted(e['contributions'].items(),key=lambda z:-abs(z[1]))[:3]
            lines.append(f"  {e['symbol']}超过{e['nearest_unchosen']}分差{e['margin']:.6f}，主要分差项："+'；'.join(f'{n} {v:+.4f}' for n,v in strong))
    lines.extend(['',
        '四、短段曲线和延伸核对',
        '8—10月：净值26个可见列、相对QQQ26列、回撤13列，现有误差容许范围下均未出现不相交列。图片线宽和横向压缩使这只能说明暂未排除，不能说明精确相等。',
        '连续净值从同一起点向前累计，使用此前真实保存的峰值，不做逐月调平或峰值重置。',
        f"不改权重延伸11月，选{' / '.join(s['next_decision']['order']['holdings'])}，当月{s['next_decision']['order']['period_return']:+.2%}。净值与回撤面板未被排除，但相对QQQ面板7列中1列轻微超出容差，按原规则仍判为未通过，不放宽标准。", 
        '11月检查只是未用其标签拟合的近邻稳定性检查；历史图片和数据此前已被看过，不称独立样本外。3个月的表现也不能验证原图十年的Sharpe。','',
        '五、为什么目前仍只是候选解释',
        f"最小选中与落选分差只有{s['fit']['margin']:.6f}。权重保留到小数点后三位，10月XLE会被UNH替代；保留两位，9月MRK会被XOM替代。",
        '500次系数相对±1%扰动后，只有37.8%仍保留全部三个月的9个选中位置；±5%扰动则没有一组保持全部一致。这是参数敏感性，不是作者交易概率。',
        '因此没有把这一规则推广到观察账户，也没有宣称复现80%年化。当前有用成果是找到了更可靠的10月能源锚点、8/9月候选集合，以及一个能在短段同时解释选择与曲线的可检验例子。',
        '下一步应围绕这些锚点压缩规则复杂度、扩大排序安全分差，并用相邻月份检验；不重新锁死有歧义的第三只股票。','',
        f"验证：48项截至决策日重新计算的因子检查通过；{st['original_sources_unchanged']}个来源及原研究/观察文件校验未变。原账户未修改。",
        '文件：WINDOW_SELECTION.json、MONTHLY_AMBIGUITY.json、ANCHOR_STRESS.json、RULE_TRIALS.json、SELECTED_RULE.json、RULE_STABILITY.json、SHORT_WINDOW.png。'])
    (P/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    # A static, exportable research plot; the source screenshot is untouched.
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    px,_,_=load();cal=read(I/'CALIBRATION.json');obs=pd.read_csv(I/'DIGITIZED_WEALTH.csv')
    nav=pd.Series({pd.Timestamp(r['date']):r['value'] for r in s['nav']});ext=pd.Series({pd.Timestamp(r['date']):r['value'] for r in s['extension_nav']})
    total=pd.concat([nav,ext]);total=total[~total.index.duplicated(keep='last')]
    base=pd.read_csv(O/'monthly_path_constraints/theme17__prior__draw_v2__blocks_NAV.csv',index_col=0,parse_dates=True).iloc[:,0]
    base=base.loc[nav.index[0]:ext.index[-1]];initial=s['initial_nav'];peak=s['prior_peak']
    fig,axes=plt.subplots(2,1,figsize=(12,7.5),sharex=True,gridspec_kw={'height_ratios':[1.5,1]})
    z=obs[obs.panel=='wealth'];dt=pd.to_datetime(z.date);mask=(dt>=nav.index[0])&(dt<=ext.index[-1]);z=z[mask];dt=dt[mask]
    sc=cal['wealth']['log_scale']
    lo=np.exp(z.log_wealth-(z.ymax-z.y+2)/sc)/initial;hi=np.exp(z.log_wealth+(z.y-z.ymin+2)/sc)/initial
    axes[0].fill_between(dt,lo,hi,color='#9c6aaf',alpha=.20,label='Digitized reference + existing pixel allowance')
    axes[0].scatter(dt,np.exp(z.log_wealth)/initial,color='#9560a5',s=12,zorder=3)
    axes[0].plot(base.index,base/initial,color='#8a8a8a',lw=1,label='Previous inferred path')
    axes[0].plot(nav.index,nav/initial,color='#007c91',lw=2,label='One fixed rule: Aug-Oct fit')
    axes[0].plot(ext.index,ext/initial,color='#007c91',ls='--',lw=2,label='Same coefficients: November check')
    axes[0].set_ylabel('Wealth / July 29 value');axes[0].legend(loc='upper left',fontsize=8)
    dd=total/np.maximum(peak,total.cummax())-1
    axes[1].plot(dd.loc[nav.index].index,100*dd.loc[nav.index],color='#007c91',lw=1.7)
    axes[1].plot(dd.loc[ext.index].index,100*dd.loc[ext.index],color='#007c91',ls='--',lw=1.7)
    source_dd=pd.read_csv(I/'purple_dd_pixels.csv');start,end=pd.to_datetime(cal['dates'])
    dates=start+pd.to_timedelta((source_dd.x-cal['wealth']['x0'])/cal['wealth']['width']*(end-start).days,unit='D')
    m=(dates>=nav.index[0])&(dates<=ext.index[-1]);z=source_dd[m];dt=dates[m]
    axes[1].vlines(dt,100*(cal['dd']['y0']-z.ymax-4)/cal['dd']['scale'],100*(cal['dd']['y0']-z.ymin+4)/cal['dd']['scale'],color='#9c6aaf',alpha=.35,lw=5)
    axes[1].scatter(dt,100*(cal['dd']['y0']-z.y)/cal['dd']['scale'],color='#9560a5',s=12)
    axes[1].set_ylabel('Drawdown from carried peak (%)')
    for ax in axes:
        ax.axvline(nav.index[-1],color='#888',ls=':',lw=1)
        ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
    axes[1].xaxis.set_major_locator(mdates.MonthLocator());axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    fig.suptitle('Monthly reconstruction: the most stable short-window clue',fontsize=15,y=.98)
    fig.text(.5,.02,'Jul 29-Nov 30, 2022 | Long only, equal thirds at month end, fixed units, zero costs | Retrospective explanation; weights remain fragile',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.045,1,.96]);fig.savefig(P/'SHORT_WINDOW.png',dpi=150);plt.close(fig)
    (P/'CODE_HASHES.json').write_text(json.dumps({str(f):sha(f) for f in P.glob('*.py')},indent=2)+'\n')

if __name__=='__main__':main()
