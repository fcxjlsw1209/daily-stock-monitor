from pathlib import Path
import pandas as pd,json,sqlite3
from audit_price_archives import audit,summary
P=Path(__file__).resolve().parent;A=P/'additional_sources';D=P/'recovered_prices'
for f in sorted((A/'kaggle2020').glob('*.csv')):
 d=pd.read_csv(f);d['symbol']=f.stem;audit(d,'kaggle2020_'+f.stem)
d=pd.read_csv(A/'a1ip_atvi.txt');d=d[['Date','Open','High','Low','Close','Adj Close','Volume']];d['symbol']='ATVI';audit(d,'a1ip_atvi')
pd.DataFrame(summary).to_csv(D/'kaggle2020_coverage.csv',index=False)
allcov=pd.concat([pd.read_csv(D/f) for f in ['coverage.csv','stooq_coverage.csv','kaggle2020_coverage.csv']],ignore_index=True)
allcov.to_csv(D/'all_source_coverage.csv',index=False)
hist=pd.read_csv(P/'historical_candidate_inventory.csv');ident=pd.read_csv(P/'expanded_sec/candidate_identity_evidence.csv',dtype={'cik':str});ident.symbol=ident.symbol.str.replace('.','-',regex=False)
shares=pd.read_csv(P/'expanded_sec/disclosure_coverage.csv',dtype={'cik':str})
with sqlite3.connect('file:'+str(P.parent/'universe_expansion/research_prices.sqlite')+'?mode=ro',uri=True) as db:
 existing=pd.read_sql_query('select symbol,min(Date) first,max(Date) last,count(*) rows from daily_prices group by symbol',db)
rows=[]
for _,r in hist.iterrows():
 s=r.symbol;ciks=sorted(set(ident.loc[ident.symbol==s,'cik']));g=allcov[allcov.symbol.str.replace('.','-',regex=False)==s];sh=shares[shares.cik.isin(ciks)];ex=existing[existing.symbol==s]
 rows.append(dict(symbol=s,current_sp500=bool(r.current_sp500),candidate_ciks=';'.join(ciks),ambiguous_cik=len(ciks)>1,recovered_sources=len(g),recovered_rows=int(g.rows.sum()),quarantined=int(g.quarantined.sum()),existing_rows=int(ex.rows.sum()),earliest_recovered=g.first_valid.min() if len(g) else None,latest_recovered=g.last_valid.max() if len(g) else None,share_events=int(sh.events.sum()),first_share_filed=sh.first_filed.min() if len(sh) else None,rank_ready=False,remaining='historical identity/listing coverage; splits/classes/ADR units; daily gap audit and total return/terminal events; historical sector'))
gaps=pd.DataFrame(rows);gaps.to_csv(P/'CANDIDATE_GAPS.csv',index=False)
with sqlite3.connect(P/'free_evidence.sqlite') as db:
 allcov.to_sql('price_source_coverage',db,if_exists='replace',index=False);gaps.to_sql('candidate_gaps',db,if_exists='replace',index=False)
report=dict(recovered_rows=int(allcov.rows.sum()),quarantined_rows=int(allcov.quarantined.sum()),unique_symbols=int(allcov.symbol.nunique()),historical_candidates=len(gaps),historical_candidates_with_recovered_prices=int((gaps.recovered_rows>0).sum()),historical_candidates_with_share_events=int((gaps.share_events>0).sum()),retired_candidates_with_recovered_prices=int(((~gaps.current_sp500)&(gaps.recovered_rows>0)).sum()),rank_ready=0,counts_include_overlapping_providers=True)
(P/'recovered_prices/summary.json').write_text(json.dumps(report,indent=2));print(report)
