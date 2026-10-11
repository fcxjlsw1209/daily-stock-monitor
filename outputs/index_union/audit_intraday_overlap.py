"""Compare intraday proxy closes with independently archived daily prices.

Scale-normalized comparisons are diagnostics, not corporate-action certificates.
"""
from pathlib import Path
import json,hashlib
import pandas as pd
P=Path(__file__).resolve().parent

def main():
 results=[]
 for symbol in ['ALXN','MXIM','CA','ESRX','CTXS','FOX','MON','YHOO','TWX','LMCK','QVCA','KRFT','SNDK','WFM']:
  a=P/'sources/union_intraday'/f'{symbol}.csv'
  b=P/'sources/pystock/extracted'/f'{symbol}.csv'
  if not a.exists() or not b.exists():continue
  x=pd.read_csv(a).set_index('date');y=pd.read_csv(b).set_index('date')
  z=x[['close','volume']].join(y[['adj_close','close','volume']],lsuffix='_intra',rsuffix='_daily',how='inner').sort_index()
  z=z[(z.adj_close>0)&(z.close_intra>0)]
  scale=(z.close_intra/z.adj_close).median()
  diff=(z.close_intra.pct_change()-z.adj_close.pct_change()).abs().dropna()
  results.append(dict(symbol=symbol,overlap=len(z),median_adjusted_price_scale=scale,max_daily_return_difference=diff.max(),returns_differ_over_10bp=int((diff>.001).sum()),returns_differ_over_100bp=int((diff>.01).sum()),worst_dates=diff.nlargest(5).to_dict(),sources={str(a.relative_to(P)):hashlib.sha256(a.read_bytes()).hexdigest(),str(b.relative_to(P)):hashlib.sha256(b.read_bytes()).hexdigest()},certified=False))
 out=dict(results=results,warning='Last intraday trade is an opening/closing proxy. Different adjustment vintages and corporate actions must be reconciled; this is not admission.')
 (P/'INTRADAY_OVERLAP_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n')
 for r in results:print(r['symbol'],r['overlap'],r['returns_differ_over_100bp'],r['max_daily_return_difference'])
if __name__=='__main__':main()
