"""Use independently corroborated issuer links solely for class/capacity checks."""
from collections import defaultdict
import numpy as np
import pandas as pd

def select_panel(data,issuer_links):
 issuers=dict(data['issuers']);groups=defaultdict(list)
 for symbol,issuer in issuer_links.items():
  if symbol in data['close']:issuers[symbol]=issuer
 for s,issuer in issuers.items():groups[issuer].append(s)
 eligible=data['eligible'].copy();audit=[]
 # Split adjustment cancels between price and share volume. Validate both
 # prices and volume before comparing; never inspect the next session.
 dv=data['close']*data['volume'];means=dv.rolling(63,min_periods=20).mean()
 observed=data['close'].gt(0)&data['volume'].ge(0)&data['adj'].gt(0)&data['opening'].gt(0)&data['high'].gt(0)&data['low'].gt(0)
 for issuer,cols in groups.items():
  if len(cols)<2:continue
  cols=sorted(cols);avail=eligible[cols]&observed[cols];many=avail.sum(axis=1)>1
  scores=means[cols].where(avail)
  missing=(scores.isna()&avail).any(axis=1)&many
  # No class is yet selectable while the mandated history comparison lacks data.
  scores.loc[missing]=np.nan
  choice=scores.fillna(-np.inf).idxmax(axis=1)
  for s in cols:eligible[s]=avail[s]&(~many|choice.eq(s))&~missing
  audit.append(dict(issuer_id=issuer,classes=cols,multiple_available_sessions=int(many.sum()),unresolved_lookback_sessions=int(missing.sum())))
 result=dict(data,eligible=eligible,issuers=issuers)
 return result,audit
