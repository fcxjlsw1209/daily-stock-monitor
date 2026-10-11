"""Test the dated old News -> 21CF archive alias, never the new News equity.

This certifies neither a return splice nor full-universe coverage. Raw closing
price agreement supports source identity, while adjustment/volume differences
remain explicit. All source hashes are pinned in the output receipt.
"""
from pathlib import Path
import hashlib,json
import pandas as pd
P=Path(__file__).resolve().parent

def main():
 old=P/'sources/qstk/NWSA.csv'; wiki=P/'sources/wiki_union/FOXA.csv'
 primary=P/'sources/nasdaq_fox2013.html'
 a=pd.read_csv(old).rename(columns={'Date':'date','Close':'close','Volume':'volume','Adj Close':'adj_close'}).set_index('date').sort_index()
 b=pd.read_csv(wiki).set_index('date').sort_index()
 z=a[['close','volume','adj_close']].join(b[['close','volume','adj_close']],lsuffix='_old',rsuffix='_wiki',how='inner').loc['2008-01-01':'2013-06-30']
 if z.empty: raise ValueError('No overlap')
 rdiff=(z.adj_close_old.pct_change()-z.adj_close_wiki.pct_change()).abs().dropna()
 result=dict(provider='wiki_union_archive',provider_symbol='FOXA',historical_symbol='NWSA',
  applicable_from='2008-01-02',applicable_to_exclusive='2013-07-01',
  security_episode='old_news_class_a_continues_as_21cf_class_a',
  overlap_sessions=len(z),raw_close_exact_matches=int(z.close_old.eq(z.close_wiki).sum()),
  raw_volume_exact_matches=int(z.volume_old.eq(z.volume_wiki).sum()),
  max_adjusted_daily_return_difference=float(rdiff.max()),
  warning='QSTK rounded adjusted close and volume differ. Identity support does not certify raw volume or total-return splice. New News NWSA excluded.',
  source_files={str(f.relative_to(P)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [old,wiki,primary]},
  candidate_date_alias_supported=bool(len(z)>1000 and z.close_old.eq(z.close_wiki).all()),
  adjustment_certified=False,admission=False)
 (P/'NEWS_LINEAGE_OVERLAP_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
