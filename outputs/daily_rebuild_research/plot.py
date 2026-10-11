from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd,numpy as np
P=Path(__file__).resolve().parent
curves=[('Old daily candidate','diagnostics/runs/old_daily_cap30_cost10.json','#1769aa','-'),('Monthly (price-scaled cap pool)','diagnostics/runs/monthly_scaled_prior_close_cost10.json','#8e44ad','-'),('New equal-factor rules','runs/rules10_next_open_cost10.json','#787f8b','--'),('New joint Ridge','joint/runs/joint_rank10_next_open_cost10.json','#009e8f','-')]
fig,axs=plt.subplots(2,1,figsize=(13,9),sharex=True,gridspec_kw={'height_ratios':[1.5,1]});fig.patch.set_facecolor('#f8fafc')
for ax in axs:
 ax.set_facecolor('#f8fafc');ax.spines[['top','right']].set_visible(False);ax.grid(True,axis='y',alpha=.18)
for label,path,color,style in curves:
 x=pd.DataFrame(json.loads((P/path).read_text())['daily']);dates=pd.to_datetime(x.date);eq=x.equity.to_numpy()/100000;cagr=eq[-1]**(252/2539)-1
 axs[0].plot(dates,eq,color=color,ls=style,lw=1.6,label=f'{label} | CAGR {cagr:.2%}')
 dd=eq/np.maximum.accumulate(np.r_[1,eq])[1:]-1;axs[1].plot(dates,dd,color=color,ls=style,lw=1.2,label=label)
q=pd.read_csv(P.parent/'index_union/sources/yahoo_prices/QQQ.csv',index_col=0,parse_dates=True).loc['2016-08-17':'2026-09-24','Adj Close'];eq=q/q.iloc[0]
axs[0].plot(q.index,eq,color='#c28b16',lw=1.8,label='QQQ buy & hold | CAGR 20.92%')
axs[1].plot(q.index,eq/eq.cummax()-1,color='#c28b16',lw=1.3)
axs[0].set_yscale('log');axs[0].set_yticks([1,2,4,8,16,32]);axs[0].yaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:g}x'));axs[0].set_ylim(.65,32)
axs[0].legend(loc='upper left',ncol=2,fontsize=9,frameon=False);axs[0].set_ylabel('Wealth / initial capital (log scale)')
axs[1].yaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:.0%}'));axs[1].set_ylabel('Drawdown from own peak');axs[1].set_ylim(-.55,.03)
fig.suptitle('Fresh reconstruction: the reported 119% CAGR was not reproduced',x=.08,ha='left',y=.97,fontsize=17,fontweight='bold',color='#17324d')
fig.text(.08,.925,'2016-08-17 to 2026-09-24 | Daily: next-open fills | Monthly: previous-close signals, month-end-close fills',fontsize=10,color='#526170')
fig.text(.08,.035,'Strategies: 0.10% round-trip costs; QQQ: zero costs. CAGR uses 252 trading days/year. No leverage.\nReused history and provisional universe/cap data; research replay, not independent out-of-sample performance.',fontsize=9,color='#526170')
fig.subplots_adjust(top=.88,bottom=.12,left=.08,right=.97,hspace=.16);fig.savefig(P/'COMPARISON.png',dpi=160);fig.savefig(P/'COMPARISON.svg');print('plot saved')
