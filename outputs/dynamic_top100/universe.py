"""Strict provider-neutral PIT membership contract. Never infer missing history."""
from pathlib import Path
import hashlib,json,math
import pandas as pd

POLICY='daily_us_listed_company_top100_v1'
SCHEMA={'issuer_id','security_id','symbol','market_cap_usd','price_date','available_at','sector','exchange','instrument','cap_basis'}
EXCHANGES={'NYSE','NASDAQ','NYSE_AMERICAN'}

def utc(value):
    t=pd.Timestamp(value)
    if t.tzinfo is None:raise ValueError('Knowledge timestamps must include timezone')
    return t.tz_convert('UTC')

def select(rows,coverage,session,decision_at,opening,n=100):
    """rows: effective-dated normalized issuer rows, not a guessed vendor schema."""
    day=str(pd.Timestamp(session).date());decision=utc(decision_at)
    if decision>=utc(opening)-pd.Timedelta(minutes=1):raise ValueError('Plan deadline missed')
    if decision<=utc(coverage['session_close']):raise ValueError('Signal session not complete')
    if coverage.get('session')!=day:raise ValueError('Coverage date mismatch')
    if coverage.get('policy')!=POLICY:raise ValueError('Coverage scope mismatch')
    for key in ['complete_active_roster','foreign_issuers_included','company_cap_basis_verified','security_identity_verified','sector_history_verified']:
        if coverage.get(key) is not True:raise ValueError('Unverified coverage: '+key)
    if not coverage.get('source') or not coverage.get('source_sha256'):raise ValueError('Missing source provenance')
    if coverage.get('knowledge_basis') not in ['provider_as_reported','contemporaneous_capture']:raise ValueError('Unknown knowledge basis')
    if coverage['knowledge_basis']=='contemporaneous_capture' and utc(coverage['captured_at'])>decision:raise ValueError('Capture occurred after decision')
    if not SCHEMA.issubset(rows.columns):raise ValueError('Missing normalized fields: '+str(SCHEMA-set(rows.columns)))
    r=rows.copy();r['price_date']=pd.to_datetime(r.price_date).dt.strftime('%Y-%m-%d')
    r['available_at']=[utc(x) for x in r.available_at]
    r=r[(r.price_date==day)&(r.available_at<=decision)]
    if r.duplicated(['issuer_id','available_at']).any():raise ValueError('Ambiguous issuer revision/share-class aggregation')
    r=r.sort_values('available_at').drop_duplicates('issuer_id',keep='last')
    expected=set(coverage['active_issuer_ids'])
    if set(r.issuer_id)!=expected:raise ValueError('Incomplete contemporaneous issuer universe')
    if len(r)<n:raise ValueError('Fewer than requested issuer count')
    if r[list(SCHEMA)].isna().any().any():raise ValueError('Null PIT fields')
    if r.security_id.duplicated().any() or r.symbol.duplicated().any():raise ValueError('Security/symbol collision at same date')
    if not r.exchange.isin(EXCHANGES).all():raise ValueError('Out-of-scope listing')
    if not r.instrument.isin(['common_stock','ADR']).all():raise ValueError('Out-of-scope instrument')
    if not r.cap_basis.eq('total_company_equity_usd').all():raise ValueError('ADR/share-class cap cannot replace issuer cap')
    if not all(math.isfinite(float(x)) and float(x)>0 for x in r.market_cap_usd):raise ValueError('Invalid market capitalization')
    # Numeric rather than lexical ordering, stable permanent-ID tie break.
    r['market_cap_usd']=r.market_cap_usd.astype(float)
    r=r.sort_values(['market_cap_usd','issuer_id'],ascending=[False,True],kind='stable').head(n).copy()
    r.insert(0,'rank',range(1,n+1))
    payload={'policy':POLICY,'signal_session':day,'decision_at':decision.isoformat(),'target_open':utc(opening).isoformat(),'universe_count':len(expected),'coverage':coverage,'members':r.assign(available_at=r.available_at.astype(str)).to_dict('records')}
    encoded=json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    return {**payload,'membership_sha256':hashlib.sha256(encoded).hexdigest()}

def freeze(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    encoded=json.dumps(payload,sort_keys=True,indent=2,allow_nan=False)+'\n'
    try:
        with path.open('x') as f:f.write(encoded)
    except FileExistsError:
        if json.loads(path.read_text())!=payload:raise ValueError('Refusing to overwrite immutable membership')
    return path

def membership_matrix(snapshots,dates,securities):
    """Use stable security IDs; warmup-only dates may be empty, trading dates may not."""
    matrix=pd.DataFrame(False,index=dates,columns=securities)
    for snap in snapshots:
        day=pd.Timestamp(snap['signal_session'])
        if day not in matrix.index:raise ValueError('Membership session absent from price calendar')
        ids=[x['security_id'] for x in snap['members']]
        if not set(ids).issubset(matrix.columns):raise ValueError('Missing security history for Top100 members')
        matrix.loc[day,ids]=True
    return matrix

def rank_factor(values,membership,valid):
    if not values.index.equals(membership.index) or not values.columns.equals(membership.columns):raise ValueError('Membership/value alignment mismatch')
    return values.where(membership & valid).rank(axis=1,pct=True)

def require_history(snapshots,signal_sessions,coverage):
    wanted={str(pd.Timestamp(x).date()) for x in signal_sessions}
    found=[x['signal_session'] for x in snapshots]
    if len(found)!=len(set(found)) or set(found)!=wanted:raise ValueError('Daily historical membership incomplete or duplicated')
    for key in ['delisted_prices_verified','terminal_returns_verified','corporate_actions_verified','historical_listing_intervals_verified']:
        if coverage.get(key) is not True:raise ValueError('Historical backtest blocked: '+key)
