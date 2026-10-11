"""Read only prices, point-in-time proxy membership, and supplied ledger.
No original chart, digitization or inferred holdings are loaded.
"""
from pathlib import Path
import hashlib,json
import pandas as pd
O=Path(__file__).resolve().parent.parent
ALIAS={'META':'research_equity_FB'}
def actual(s):return ALIAS.get(s,s)
def display(s):return 'META' if s=='research_equity_FB' else s
def read(f):return json.loads(Path(f).read_text())
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def load_inputs():
    sources={}
    def frame(f):
        sources[str(f)]=sha(f)
        return pd.read_csv(f,index_col=0,parse_dates=True)
    px=frame(O/'index_union/alias_resolved_inputs/adj.csv.gz')
    raw={k:frame(O/'index_union/alias_resolved_inputs'/f'{k}.csv.gz') for k in ['close','high','low','volume','sector60']}
    manifests=[]
    fs=[O/'four_factor_rebound/ETF_SOURCES.json',O/'monthly_reconstruction_v2/ETF_MANIFEST.json',O/'monthly_reconstruction_v2/foreign/ETF_MANIFEST.json']
    for f in fs:
        sources[str(f)]=sha(f);manifests+=read(f)
    for e in manifests:
        assert sha(e['file'])==e['sha256'];z=frame(Path(e['file']))
        px[e['symbol']]=z['Adj Close'].reindex(px.index)
        for k,col in [('close','Close'),('high','High'),('low','Low'),('volume','Volume')]:raw[k][e['symbol']]=z[col].reindex(px.index)
        b={'TSM':'XLK','BABA':'XLY','NVO':'XLV','SAP':'XLK','TM':'XLY'}.get(e['symbol'],'SPY')
        raw['sector60'][e['symbol']]=px[b].pct_change(60,fill_method=None)
    raw={k:z.reindex(index=px.index,columns=px.columns) for k,z in raw.items()}
    member=frame(O/'monthly_reconstruction_v2/foreign/membership.csv.gz').reindex(index=px.index,columns=px.columns,fill_value=False).fillna(False).astype(bool)
    core=[e['symbol'] for e in read(fs[0])]
    theme=[s for s in core if s not in {'DIA','XLB','XLC','XLRE','XLU','XLP'}]+['SMH','SOXX','IGV','VGT','MTUM','ARKK']
    members={}
    for name,etfs in [('core17',core),('theme17',theme)]:
        m=member.copy();m[etfs]=px[etfs].notna();members[name]=(m,set(etfs))
    f=O/'monthly_supplied_ledger_audit/PERIODS.json';sources[str(f)]=sha(f)
    return px,raw,members,read(f),sources

def market_states(px):
    vol=px.SPY.pct_change(fill_method=None).rolling(63).std()
    return pd.DataFrame(dict(spy_trend=px.SPY>px.SPY.rolling(200).mean(),
                            qqq_trend=px.QQQ>px.QQQ.rolling(200).mean(),
                            tech_relative=px.QQQ.pct_change(63,fill_method=None)>px.SPY.pct_change(63,fill_method=None),
                            high_market_vol=vol>vol.rolling(252).median())).astype(float)
