"""Research-local PFE distribution evidence; never rewrite original source panels."""
from pathlib import Path
import json,hashlib
import pandas as pd
P=Path(__file__).resolve().parent

def patch(d):
 evidence=P/'sources/pfizer_web_retrieval.json';raw=P.parent/'universe_expansion/raw/VTRS.csv.gz'
 assert '0.124079' in evidence.read_text() and 'November 17, 2020' in evidence.read_text()
 sha=hashlib.sha256(raw.read_bytes()).hexdigest();assert sha=='6f11fcc52bffd3fc4a391e7e48351184da569d54e704af6a8cb21e3b8219863e'
 x=pd.read_csv(raw,index_col=0,parse_dates=True);price=float(x.at[pd.Timestamp('2020-11-17'),'Open'])
 e=dict(event_id='PFE_VTRS_2020-11-17',security_id='PFE',child_security_id='VTRS',execution_session='2020-11-17',known_at='2020-11-16T23:59:59-05:00',ratio=.124079,provider_adjustment=1.054,terms_verified=True,source_file=str(evidence),source_sha256=hashlib.sha256(evidence.read_bytes()).hexdigest(),source_url='https://www.pfizer.com/news/press-release/press-release-detail/pfizer-completes-transaction-combine-its-upjohn-business',child_open=dict(session='2020-11-17',kind='regular_way_open',price=price,source_file=str(raw),source_sha256=sha),policy='Same existing research spin protocol: sell distributed child at first regular-way open, credit net proceeds, do not create parent shares.',price_evidence='Cached Yahoo OHLCV; first regular-way date verified by Pfizer; quote not independently certified exchange auction.')
 assert abs(d['splits'].at[pd.Timestamp('2020-11-17'),'PFE']-1.054)<1e-8
 d=dict(d);d['spin_distributions']=[x for x in d.get('spin_distributions',[]) if x['event_id']!=e['event_id']]+[e]
 (P/'PFE_REPAIR.json').write_text(json.dumps(e,indent=2)+'\n');return d
