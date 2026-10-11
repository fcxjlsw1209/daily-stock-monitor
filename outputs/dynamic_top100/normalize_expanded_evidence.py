"""Normalize filing-dated share evidence; never certify units or coverage."""
from pathlib import Path
import json,sqlite3
import pandas as pd
P=Path(__file__).resolve().parent
TAGS={'EntityCommonStockSharesOutstanding','CommonStockSharesOutstanding','NumberOfSharesOutstanding'}
def normalize(raw):
 d=raw.copy()
 d['cik']=d.cik.astype(str).str.zfill(10)
 good=(pd.to_datetime(d.end)<=pd.to_datetime(d.filed))&(pd.to_numeric(d.val,errors='coerce')>0)&d.concept.isin(TAGS)
 d=d[good];out=[]
 for (cik,tax,concept,filed,end),g in d.groupby(['cik','taxonomy','concept','filed','end']):
  values=pd.to_numeric(g.val).unique();conflict=len(values)!=1
  available=(pd.Timestamp(filed)+pd.Timedelta(days=1)).tz_localize('America/New_York').tz_convert('UTC')
  out.append(dict(cik=cik,taxonomy=tax,concept=concept,reported_as_of=end,filed=filed,available_no_earlier_than=available.isoformat(),shares=None if conflict else float(values[0]),conflict=conflict,accessions=';'.join(sorted(set(g.accn))),source_file=';'.join(sorted(set(g.source_file))),units_classes_and_splits_verified=False,usable_for_rank=False))
 return pd.DataFrame(out)
def main():
 frames=[];files=[]
 for folder in ['expanded_sec','foreign_sec','priority_sec']:
  mapping={}
  if folder!='expanded_sec':mapping={r['symbol']:str(r['cik']).zfill(10) for r in json.loads((P/folder/'coverage.json').read_text())}
  for f in sorted((P/folder).glob('*_shares.csv')):
   d=pd.read_csv(f,dtype={'cik':str})
   if d.empty:continue
   key=f.name.removesuffix('_shares.csv');d['cik']=key if key.isdigit() else mapping[key]
   d['source_file']=str(f.relative_to(P));frames.append(d);files.append(str(f.relative_to(P)))
 raw=pd.concat(frames,ignore_index=True);d=normalize(raw)
 coverage=d.groupby('cik').agg(events=('cik','size'),conflicts=('conflict','sum'),first_filed=('filed','min'),last_filed=('filed','max')).reset_index()
 d.to_csv(P/'expanded_sec/disclosure_events_unverified.csv.gz',index=False,compression='gzip');coverage.to_csv(P/'expanded_sec/disclosure_coverage.csv',index=False)
 with sqlite3.connect(P/'free_evidence.sqlite') as db:
  d.to_sql('share_disclosures_unverified',db,if_exists='replace',index=False)
  db.execute('CREATE INDEX IF NOT EXISTS idx_disclosure_cik_filed ON share_disclosures_unverified(cik,filed)')
  coverage.to_sql('share_coverage',db,if_exists='replace',index=False)
 summary=dict(raw_rows=len(raw),source_files=len(files),events=len(d),issuers=int(d.cik.nunique()),conflicts=int(d.conflict.sum()),issuers_with_disclosure_before_2010=int(d.loc[d.filed<'2010-01-01','cik'].nunique()),issuers_with_disclosure_before_2019=int(d.loc[d.filed<'2019-01-01','cik'].nunique()),usable_for_rank=0,priority_sources_included=True,across_source_conflicts_checked=True)
 (P/'expanded_sec/normalization_summary.json').write_text(json.dumps(summary,indent=2));print(summary)
if __name__=='__main__':main()
