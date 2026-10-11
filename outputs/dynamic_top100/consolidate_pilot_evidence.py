"""Build auditable pilot-only identity hints and normalized SEC observations."""
import ast,json
from pathlib import Path
import pandas as pd
from normalize_expanded_evidence import normalize
P=Path(__file__).resolve().parent;folder=P/'pilot_identity'
tree=ast.parse((P/'collect_pilot_identity.py').read_text());targets={}
for node in tree.body:
 if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='targets' for t in node.targets):targets.update(ast.literal_eval(node.value))
 if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and isinstance(node.value.func.value,ast.Name) and node.value.func.value.id=='targets':targets.update(ast.literal_eval(node.value.args[0]))
raw=[];links=[]
for name,cik in targets.items():
 sub=folder/f'{cik:010}_submissions.json';facts=folder/f'{cik:010}_facts.json'
 if not sub.exists():continue
 j=json.loads(sub.read_text());assert int(j['cik'])==cik
 links.append({'reference_name':name,'cik':f'{cik:010}','sec_name':j['name'],'former_names':j.get('formerNames',[]),'source':f'https://data.sec.gov/submissions/CIK{cik:010}.json','scope':'2010-03 discovery mapping; not a complete exchange eligibility interval'})
 if not facts.exists():continue
 j=json.loads(facts.read_text())
 for tax,tags in j.get('facts',{}).items():
  for tag,v in tags.items():
   if tag not in ['EntityCommonStockSharesOutstanding','CommonStockSharesOutstanding','NumberOfSharesOutstanding']:continue
   for row in v.get('units',{}).get('shares',[]):raw.append(dict(row,cik=f'{cik:010}',taxonomy=tax,concept=tag,source_file=str(facts.relative_to(P))))
# Keep manually extracted totals in a separate taxonomy: XBRL may report one class.
# These observations establish evidence presence, never a verified capitalization.
for row in json.loads((folder/'VERIFIED_CLASS_COUNTS.json').read_text()):
 raw.append(dict(cik=row['cik'],taxonomy='manual_cover_page',concept='CommonStockSharesOutstanding',end=row['as_of'],filed=row['filed'],val=row['total_common_share_count'],accn=row['accession'],source_file=row['source']))
d=normalize(pd.DataFrame(raw));d.to_csv(folder/'disclosure_events_unverified.csv.gz',index=False,compression='gzip')
(folder/'month_aliases.json').write_text(json.dumps({'session_from':'2010-03-01','session_to':'2010-03-31','aliases':{x['reference_name']:x['cik'] for x in links},'evidence':links,'scope':'Issuer hints only; not certification of daily listing eligibility or cap basis'},indent=2))
print('issuers',len(links),'events',len(d),'with pre-March disclosure',d.loc[d.filed<'2010-03-01','cik'].nunique())
