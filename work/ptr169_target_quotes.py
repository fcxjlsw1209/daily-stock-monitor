"""Preregistered last-actual-session quote selection; never synthesize bars."""
import hashlib
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from ptr169_quote_audit import normalize_chart, EXPECTED

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/monthly_ptr_pool_source169'
HOLIDAYS={
 '601857.SS.json': {'2017-01-30':'2017-01-26','2017-05-30':'2017-05-26','2020-01-30':'2020-01-23'},
 '0857.HK.json': {'2017-01-30':'2017-01-27','2017-05-30':'2017-05-29'}}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,x):
    with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def register():
    review=read(OUT/'missing_quote/INDEPENDENT_REVIEW.json')
    assert review['verdict']=='CORROBORATED_FOR_CONDITIONAL_HISTORICAL_REPLAY_NOT_PIT_VINTAGE'
    files=['QUOTE_AUDIT.json','601857.SS.json','0857.HK.json','calendar/DECISIONS.json',
           'calendar/VERIFICATION.json','missing_quote/RESOLUTION.json','missing_quote/INDEPENDENT_REVIEW.json']
    for name in ['601857.SS.json','0857.HK.json']:files.append(name+'.receipt.json')
    pins={str((OUT/f).relative_to(ROOT)):sha(OUT/f) for f in files}
    for f in ['outputs/monthly_2016_extension/EARLY_RANKS.json','work/ptr169_quote_audit.py']:pins[f]=sha(ROOT/f)
    write(OUT/'TARGET_QUOTES_PLAN.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        code_sha256=sha(Path(__file__)),input_sha256=pins,holiday_observation_dates=HOLIDAYS,
        policy='Exact asof Yahoo quote normally. On the five officially closed cells, select the specified most recent actual quote, keep original observation date and disclose age. No synthetic bar, no price forward-fill. Open-market missing A2019-04-29 uses separately corroborated Tencent OHLC. No other substitutions.',
        timing='Local A/H close is assumed completed before that same calendar-date US close;2026 retrieved historical quote version, not original PIT data feed.',
        scope='53 historical signal/asof pairs; source prices only. No capitalization or strategy calculation.'))
def run():
    plan=read(OUT/'TARGET_QUOTES_PLAN.json');assert plan['code_sha256']==sha(Path(__file__))
    for p,h in plan['input_sha256'].items():assert sha(ROOT/p)==h,p
    cal=read(OUT/'calendar/DECISIONS.json')
    closed={(r['symbol'],r['date']):r for r in cal['rows'] if r['classification']=='OFFICIAL_EXCHANGE_CLOSED'}
    expected={(s,d) for s,entries in HOLIDAYS.items() for d in entries}
    assert set(closed)==expected
    resolution=read(OUT/'missing_quote/RESOLUTION.json')
    review=read(OUT/'missing_quote/INDEPENDENT_REVIEW.json')
    for p,h in review['input_sha256'].items():assert sha(ROOT/p)==h,p
    assert resolution['date']=='2019-04-29' and resolution['ohlc']['close']==7.4
    quotes={s:normalize_chart(read(OUT/s),EXPECTED[s])[1] for s in HOLIDAYS}
    targets=read(ROOT/'outputs/monthly_2016_extension/EARLY_RANKS.json')
    rows=[]
    for target in targets:
        row=dict(signal=target['date'],asof=target['asof'],quotes={})
        asof=target['asof']
        for symbol,book in quotes.items():
            source=symbol;observed=asof;kind='exact_asof_actual_quote'
            if asof in HOLIDAYS[symbol]:
                assert asof not in book
                observed=HOLIDAYS[symbol][asof]
                assert observed==max(d for d in book if d<asof), (symbol,asof,observed)
                day=date.fromisoformat(observed)+timedelta(days=1)
                evidence=closed[(symbol,asof)]
                start=evidence.get('announced_closure_start',evidence.get('original_announced_closure_start',asof))
                end=evidence.get('announced_closure_end',evidence.get('original_announced_closure_end',asof))
                while day<=date.fromisoformat(asof):
                    assert day.weekday()>=5 or start<=day.isoformat()<=end,(symbol,day)
                    day+=timedelta(days=1)
                kind='last_actual_quote_during_documented_exchange_closure'
            if symbol=='601857.SS.json' and asof=='2019-04-29':
                assert asof not in book
                value=resolution['ohlc'];source='missing_quote/RESOLUTION.json';kind='exact_asof_corroborated_tencent_quote'
            else:
                value=book[observed];assert value['ohlc_range_consistent'] is True
            op,cl,hi,lo=(value[k] for k in ['open','close','high','low'])
            assert lo<=min(op,cl)<=max(op,cl)<=hi
            row['quotes'][symbol]=dict(source=source,observation_date=observed,selection=kind,
                age_calendar_days=(date.fromisoformat(asof)-date.fromisoformat(observed)).days,
                currency=EXPECTED[symbol][1],ohlc={k:value[k] for k in ['open','high','low','close']})
        rows.append(row)
    assert len(rows)==53
    counts={kind:sum(x['selection']==kind for r in rows for x in r['quotes'].values())
            for kind in ('exact_asof_actual_quote','exact_asof_corroborated_tencent_quote',
                         'last_actual_quote_during_documented_exchange_closure')}
    assert list(counts.values())==[100,1,5],counts
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),plan_sha256=sha(OUT/'TARGET_QUOTES_PLAN.json'),
        status='COMPLETE_CONDITIONAL_HISTORICAL_QUOTE_SELECTION',rows=rows,summary=dict(targets=53,
            exact_yahoo_cells=counts['exact_asof_actual_quote'],
            exact_tencent_cells=counts['exact_asof_corroborated_tencent_quote'],
            documented_last_actual_quote_cells=counts['last_actual_quote_during_documented_exchange_closure']),
        synthetic_bars=0,original_quote_files_modified=False,capitalization_computed=False,returns_computed=False,
        limits=[plan['timing'],'Currency/raw-price unit supported by issuer identity and provider metadata; not complete split/corporate-action certification.'])
    write(OUT/'TARGET_QUOTES.json',result);print(json.dumps(result['summary']))
if __name__=='__main__':{'register':register,'run':run}[sys.argv[1]]()
