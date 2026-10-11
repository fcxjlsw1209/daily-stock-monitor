"""Recover targeted large-cap evidence from an already-downloaded archive."""
from pathlib import Path
import json,zipfile,hashlib
import pandas as pd
from audit_price_archives import audit,summary
P=Path(__file__).resolve().parent
report=json.loads((P/'historical_largecap_priority_audit.json').read_text())
with zipfile.ZipFile(P/'additional_sources/kaggle_stooq_2017.zip') as z:
    names=set(z.namelist())
    for row in report['rows']:
        name='Stocks/'+row['symbol'].lower()+'.us.txt'
        if name not in names or not z.getinfo(name).file_size:
            row['archive_recovery']='not found';continue
        raw=z.read(name);data=pd.read_csv(z.open(name));data['symbol']=row['symbol']
        audit(data,'priority_stooq2017_'+row['symbol'])
        row['archive_recovery']=dict(summary[-1],member=name,sha256=hashlib.sha256(raw).hexdigest())
report['recovery_summary']={'recovered_symbols':len(summary),'rows':sum(r['rows'] for r in summary),'quarantined':sum(r['quarantined'] for r in summary),'rank_ready':0,'note':'Prices are not yet reconciled for identity, units, corporate actions or full history.'}
(P/'historical_largecap_priority_audit.json').write_text(json.dumps(report,indent=2))
pd.DataFrame(summary).to_csv(P/'recovered_prices/priority_largecap_coverage.csv',index=False)
print(json.dumps(report['recovery_summary'],indent=2))
print('not found:',[r['symbol'] for r in report['rows'] if r['archive_recovery']=='not found'])
