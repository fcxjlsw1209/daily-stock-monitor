"""Separate alias-resolved diagnostic bundle; no admission or paper mutations."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
import pandas as pd
from consolidate_security_aliases import consolidate
from consolidate_issuer_classes import select_panel
from research_scope import identity_links,apply_scope
from assemble_input_diagnostics import sector_candidates,ETFS
from sector_history import benchmark_returns
P=Path(__file__).resolve().parent
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
def main():
 source=P/'input_diagnostics/MANIFEST.json';m=json.loads(source.read_text());data={}
 for r in m['fields']:
  f=P/r['file']
  if sha(f)!=r['sha256']:raise ValueError('Input changed')
  v=pd.read_csv(f,index_col=0,parse_dates=True,low_memory=False);data[r['field']]=v.iloc[:,0] if r['field'].startswith('market_') else v
 data['issuers']={s:'unresolved_'+s for s in data['close']}
 _,obs,_=sector_candidates(data['close'].index,data['close'].columns)
 rows=json.loads((P/'DATED_SYMBOL_IDENTITIES.json').read_text())['rows']
 data,decisions=consolidate(data,rows,obs)
 etf=pd.DataFrame(index=data['close'].index)
 for s in ETFS.values():etf[s]=pd.read_csv(P/'sources/yahoo_prices'/f'{s}.csv',index_col='Date',parse_dates=True)['Adj Close'].reindex(etf.index)
 data['sector60']=benchmark_returns(data['sectors'],etf,ETFS)
 current=json.loads((P/'SEC_IDENTITY_REVIEW.json').read_text())
 links,identity_proofs=identity_links(m,current)
 data,scope_exclusions=apply_scope(data,links)
 data,class_audit=select_panel(data,links)
 out=P/'alias_resolved_inputs';out.mkdir(exist_ok=True);fields=[]
 for key,v in data.items():
  if key=='issuers':continue
  f=out/(key+'.csv.gz');v.to_csv(f,compression={'method':'gzip','mtime':0});fields.append(dict(field=key,file=str(f.relative_to(P)),sha256=sha(f)))
 manifest=dict(generated_at=datetime.now(timezone.utc).isoformat(),start=m['start'],end=m['end'],status='alias_resolution_diagnostic_not_admitted',fields=fields,issuers=data['issuers'],alias_decisions=decisions,class_selection=class_audit,identity_proofs=identity_proofs,scope_exclusions=scope_exclusions,sources=[dict(file=str(source.relative_to(P)),sha256=sha(source)),dict(file='DATED_SYMBOL_IDENTITIES.json',sha256=sha(P/'DATED_SYMBOL_IDENTITIES.json'))],performance_publishable=False)
 (out/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(decisions,indent=2))
if __name__=='__main__':main()
