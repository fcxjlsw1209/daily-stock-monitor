"""Read-only extraction of recent fills from the fixed monthly research run."""
from pathlib import Path
import json,hashlib
import pandas as pd
P=Path(__file__).resolve().parent
SOURCE=P/'cost10.json'
START='2024-01-01'
ROOT=P.parents[2]

def display(symbol):
 return 'META' if symbol=='research_equity_FB' else symbol

def main():
 blob=SOURCE.read_bytes();r=json.loads(blob);assert r['status']=='complete'
 orders=[o for o in r['orders'] if o['date']>=START]
 trades=[t for t in r['trades'] if t['exit_date']>=START]
 sells={(t['exit_date'],t['symbol']):t for t in trades};assert len(sells)==len(trades)
 assert len(trades)==sum(o['side']=='sell' for o in orders)
 for o in orders:
  if o['side']=='sell':
   t=sells[(o['date'],o['symbol'])]
   assert abs(t['quantity']-o['quantity'])<1e-7
   assert abs(t['exit_price']-o['price'])<1e-7
   assert abs(t['exit_fee']-o['fee'])<1e-7
  assert abs(o['fee']-o['quantity']*o['price']*.0005)<1e-6
 # Use the identical source prices from the manifest for open inventory valuation.
 union=P.parent.parent/'index_union'
 manifest=json.loads((union/'alias_resolved_inputs/MANIFEST.json').read_text())
 end=r['daily'][-1]['date'];symbols=list(r['state']['positions']);panels={}
 for field in ['close','raw_per_signal']:
  item=next(v for v in manifest['fields'] if v['field']==field);path=Path(item['file']);path=path if path.is_absolute() else union/path
  assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
  panels[field]=pd.read_csv(path,index_col=0,parse_dates=True,low_memory=False).loc[end,symbols]
 positions=[]
 for symbol,p in r['state']['positions'].items():
  mark=float(panels['close'][symbol]*panels['raw_per_signal'][symbol]);value=mark*p['qty'];pnl=value+p['dividends']-p['cost']
  positions.append(dict(symbol=symbol,display_symbol=display(symbol),entry_date=p['entry_date'],entry_price=p['entry_price'],quantity=p['qty'],entry_cost_including_fee=p['cost'],entry_fee=p['entry_fee'],dividends=p['dividends'],as_of=end,mark=mark,market_value=value,unrealized_pnl=pnl,unrealized_return=pnl/p['cost']))
 assert abs(sum(p['market_value'] for p in positions)+r['state']['cash']-r['state']['equity'])<1e-6
 rows=[]
 for day in sorted({o['date'] for o in orders}):
  closed=[t for t in trades if t['exit_date']==day];bought=[o for o in orders if o['date']==day and o['side']=='buy']
  rows.append(dict(date=day,closed=[dict(symbol=display(t['symbol']),entry_date=t['entry_date'],net_return=t['net_return'],net_pnl=t['net_pnl']) for t in closed],buys=[display(o['symbol']) for o in bought]))
 out=dict(strategy='m_base_positive',run_start='2010-01-04',initial_capital=100000,report_start=START,report_end=end,source_sha256=hashlib.sha256(blob).hexdigest(),source=str(SOURCE),research_only=True,strict_historical_admission=False,roundtrip_cost_bps=10,order_count=len(orders),closed_trade_count=len(trades),open_trade_count=len(positions),recent_closed_win_rate=sum(t['net_pnl']>0 for t in trades)/len(trades),orders=orders,closed_trades=trades,open_positions=positions,monthly_rows=rows)
 (P/'RECENT_TRADES_2024_2026.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
 lines=['开发期选出的纯月策略：2024年至2026年10月6日完整交易明细',
 '账户：2010-01-04以100000美元开始，连续复利；2024年未重置本金。',
 '性质：有条件历史研究，不是实盘；严格历史数据准入仍为false。',
 '信号：月末前一交易日收盘；成交：月末收盘价代理；每次买或卖各0.05%手续费。',
 '所有美元金额均为USD，显示数量六位小数、价格四位、金额两位；JSON保留完整精度。',
 'META对应内部连续证券ID research_equity_FB；普通买卖表价格为当时原始价格。',
 '完整往返表中的买入价与股数如跨拆股，统一到卖出时单位；原始买入值见普通买卖表。',
 '净收益率按该笔入场总支出计算，含分红、买卖手续费，并非占初始10万美元的收益率。',
 '同股票月底卖出再买入是等权调仓；不表示策略在当日转为空头。',
 f'共{len(rows)}次调仓、{len(orders)}条成交、{len(trades)}笔平仓、{len(positions)}笔未平仓；近期平仓净胜率{out["recent_closed_win_rate"]:.2%}。',
 '','期初继承仓位（不重复计作2024年买入）：']
 for t in trades:
  if t['entry_date']<START:
   lines.append(f"{t['entry_date']} {display(t['symbol']):6} 数量 {t['quantity']:.6f} 买价 {t['entry_price']:.4f} 含费成本 {t['entry_outlay']:,.2f}")
 lines+=['','一、逐笔普通买卖（2024年起全部成交）','日期        买卖 ticker        数量             成交价         成交金额           手续费          现金变化         平仓净盈亏']
 for o in orders:
  gross=o['price']*o['quantity'];cash=-(gross+o['fee']) if o['side']=='buy' else gross-o['fee'];pnl=sells[(o['date'],o['symbol'])]['net_pnl'] if o['side']=='sell' else None
  lines.append(f"{o['date']} {o['side']:4} {display(o['symbol']):6} {o['quantity']:15.6f} {o['price']:13.4f} {gross:17,.2f} {o['fee']:13,.2f} {cash:17,.2f} {pnl:17,.2f}" if pnl is not None else f"{o['date']} {o['side']:4} {display(o['symbol']):6} {o['quantity']:15.6f} {o['price']:13.4f} {gross:17,.2f} {o['fee']:13,.2f} {cash:17,.2f}                 —")
 lines+=['','二、已平仓往返交易（按卖出日，包含2023年末建仓）','买入日期    卖出日期    ticker        股数        买入价(同单位)   卖出价       入场含费支出       分红     买入费      卖出费         净盈亏       净收益率']
 for t in trades:
  lines.append(f"{t['entry_date']} {t['exit_date']} {display(t['symbol']):6} {t['quantity']:13.6f} {t['entry_price']:13.4f} {t['exit_price']:11.4f} {t['entry_outlay']:17,.2f} {t['dividends']:10,.2f} {t['entry_fee']:10,.2f} {t['exit_fee']:10,.2f} {t['net_pnl']:15,.2f} {t['net_return']:+10.2%}")
 lines+=['','三、区间内公司行动']
 for a in r['actions']:
  if a.get('session','')>=START:lines.append(json.dumps(a,ensure_ascii=False))
 lines+=['','四、期末未平仓（未扣假设卖出费）']
 for x in positions:
  lines.append(f"{x['display_symbol']} 买入{x['entry_date']}，数量{x['quantity']:.6f}，入场价{x['entry_price']:.4f}，含费成本{x['entry_cost_including_fee']:,.2f}，买入费{x['entry_fee']:,.2f}；{end}收盘{x['mark']:.4f}，市值{x['market_value']:,.2f}，未实现净盈亏{x['unrealized_pnl']:+,.2f}（{x['unrealized_return']:+.2%}）。")
 lines+=['',f"期末现金 {r['state']['cash']:,.2f}；期末总净值 {r['state']['equity']:,.2f}。",f"原始回测SHA256：{out['source_sha256']}"]
 (P/'RECENT_TRADES_2024_2026.txt').write_text('\n'.join(lines)+'\n')
 assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==out['source_sha256']
 print(json.dumps({k:v for k,v in out.items() if k not in ['orders','closed_trades','open_positions','monthly_rows','source']},ensure_ascii=False));print('open',json.dumps(positions));print('report_lines',len(lines))
if __name__=='__main__':main()
