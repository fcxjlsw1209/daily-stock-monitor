"""Diagnostic overlap audit. Ratios are not automatic adjustment factors."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parent;D=P/'recovered_prices';rows=[]
for newer in sorted(D.glob('priority_yahoo_*.csv.gz')):
 symbol=newer.name.split('priority_yahoo_')[1].split('.csv')[0]
 older=D/f'priority_kaggle2020_{symbol}.csv.gz'
 if not older.exists() or symbol=='SI':continue
 a=pd.read_csv(older);b=pd.read_csv(newer)
 a=a[~a.quarantine.astype(str).str.lower().eq('true')];b=b[~b.quarantine.astype(str).str.lower().eq('true')]
 m=a[['date','close']].merge(b[['date','close']],on='date',suffixes=('_archive','_yahoo'));m=m[m.date>='2010-01-01']
 if m.empty:continue
 ratio=m.close_yahoo/m.close_archive
 rows.append(dict(symbol=symbol,overlap_sessions=len(m),first=m.date.min(),last=m.date.max(),median_yahoo_to_archive_close=float(ratio.median()),min_ratio=float(ratio.min()),max_ratio=float(ratio.max()),days_difference_over_1pct=int((abs(ratio-1)>.01).sum()),interpretation='price-basis/identity review required' if (abs(ratio-1)>.01).any() else 'close agreement within 1%; does not certify OHLC/actions/identity',adjustment_verified=False))
(P/'PRICE_OVERLAP_AUDIT.json').write_text(json.dumps(rows,indent=2));print(pd.DataFrame(rows)[['symbol','overlap_sessions','median_yahoo_to_archive_close','days_difference_over_1pct']].to_string(index=False))
