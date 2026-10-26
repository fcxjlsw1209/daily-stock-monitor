"""Metadata-only identity inventory; no prices or array decoding."""
from pathlib import Path
import json,hashlib
R=Path(__file__).resolve().parents[1]
def read(p):return json.loads((R/p).read_text())
m=read('outputs/monthly_local_linear70/META.json')
x=read('outputs/index_union/alias_resolved_inputs/MANIFEST.json')
l=read('work/nested199_SOURCE_REVIEW.json')['source_lineage']
u=sorted({s for row in m['symbols'] for s in row})
print(json.dumps(dict(symbols=u,etfs=m['etfs'],records=[dict(symbol=s,issuer=x['issuers'].get(s),operations=[{k:v for k,v in z.items() if k in ('operation','path','provider_symbol','upstream_source')} for z in l[s]],identity=[p for p in x['identity_proofs'] if p.get('symbol')==s]) for s in u if s not in m['etfs']]),indent=2))
