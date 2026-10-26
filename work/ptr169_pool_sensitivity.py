"""One registered PTR A/H valuation sensitivity; no portfolio calculations."""
import hashlib
import json
import math
import platform
import sys
import unittest
from datetime import date, datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'outputs/monthly_ptr_pool_source169'
OUT=SRC/'conditional_pool'
A_COUNT=161922077818
H_COUNT=21098900000
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def now():return datetime.now(timezone.utc).isoformat()
def write(p,x):
    with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def valuation(a,h,cny,hkd):
    vals=[a,h,cny,hkd]
    if not all(type(x) in (float,int) and math.isfinite(x) and x>0 for x in vals):
        raise ValueError('Invalid quote or FX')
    return math.fsum([A_COUNT*a/cny,H_COUNT*h/hkd])

def insert_ptr(top40, cap):
    if len(top40)!=40 or len({x['symbol'] for x in top40})!=40:
        raise ValueError('Expected 40 unique archived issuers')
    if any(x['symbol']=='PTR' for x in top40):
        raise ValueError('Expected original PTR outside archived top40')
    if not math.isfinite(cap) or cap<=0:
        raise ValueError('Invalid value')
    ordered=sorted(top40,key=lambda x:(-x['market_cap_usd'],x['symbol']))
    if ordered!=top40:raise ValueError('Original ranks not sorted')
    return sorted(top40+[dict(symbol='PTR',market_cap_usd=cap)],key=lambda x:(-x['market_cap_usd'],x['symbol']))

def register():
    OUT.mkdir(exist_ok=True)
    required=['TARGET_QUOTES_PLAN.json','TARGET_QUOTES.json','annual/INDEPENDENT_REVIEW.json',
              'missing_quote/INDEPENDENT_REVIEW.json','fed/bulk/RECONCILED_RATES.json']
    for f in required:assert (SRC/f).is_file(),f
    assert read(SRC/'annual/INDEPENDENT_REVIEW.json')['verdict']=='PASS_TRANSCRIPTION_AND_CONDITIONAL_CONSTANT_ONLY_NOT_STRICT_PIT'
    assert read(SRC/'missing_quote/INDEPENDENT_REVIEW.json')['verdict']=='CORROBORATED_FOR_CONDITIONAL_HISTORICAL_REPLAY_NOT_PIT_VINTAGE'
    paths=[p for p in SRC.rglob('*') if p.is_file() and OUT not in p.parents and p.suffix!='.pyc']
    paths += [ROOT/p for p in ['work/ptr169_target_quotes.py','outputs/monthly_2016_extension/EARLY_RANKS.json',
        'outputs/monthly_2016_extension/POOLS.json','outputs/monthly_2016_extension/extend.py',
        'outputs/dynamic_top100/additional_sources/tiingo/PTRCY_prices.json']]
    write(OUT/'PLAN.json',dict(schema='ptr169_conditional_fixed_count_valuation_v1',registered_utc=now(),
        script_sha256=sha(Path(__file__)),runtime=platform.python_version(),
        input_sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(paths))},
        scope='53 frozen early signal/asof pairs. Data sensitivity only; no model fitting, selection or return calculation.',
        shares=dict(A=A_COUNT,H=H_COUNT,total=A_COUNT+H_COUNT),
        formula='A_count*A_local_close/CNY_per_USD + H_count*H_local_close/HKD_per_USD',
        quotes='Use exactly preselected TARGET_QUOTES; distinct actual observation dates retained; no new or filled price rows.',
        fx='Use exactly reconciled Fed H10 selections with max(index date,page publication date)<asof; latest same-day nonmissing CNY/HKD observations in that prior release. Explicitly lagged FX proxy.',
        ranking='Keep every other issuer value and legacy Shell convention unchanged. Insert PTR into original top40, sort(-cap,symbol), first32. Original PTR must be outside40 and old ADR-valued cap <= rank40. Preserve all69 later pools.',
        assumptions=[
            'Fixed2015 year-end share-class counts throughout2016-2020; annual2016-2019 consistency does not prove every intrayear action or2020. No later annual data enters the formula.',
            '2015 report and corresponding SEC filing index support approximate pre-Aug2016 availability, not direct certification of original field publication timestamps.',
            'A/H local raw quotes retrieved in2026 treated as same-date completed local closes before US close; exact original data vintages unavailable.',
            'Last actual local quote during independently documented holidays is a marked, possibly stale valuation observation, never a fabricated current-day price.',
            'Sum-of-class local market values differs from valuing all shares at the H-share/ADS price. Neither convention is chosen by strategy returns.',
            'Other companies keep original retrospective cap and listing assumptions; this does not certify a full-market Top32 census.'],
        hard_fail=['source/code drift','invalid units','missing selected quote/FX','FX unavailable at cutoff','original PTR not outside40','original top32 mismatch'],
        next_gate='Independent data audit. Any changed pool requires fresh current-candidate factor ranks and relevant training reconstruction before a separate preregistered strategy sensitivity.',
        strategy_trials_added=0,observers_modified=False))
    print(json.dumps(dict(status='REGISTERED_NO_VALUATIONS_YET',sha256=sha(OUT/'PLAN.json'))))

def run():
    plan=read(OUT/'PLAN.json');assert plan['script_sha256']==sha(Path(__file__))
    assert plan['runtime']==platform.python_version()
    for p,h in plan['input_sha256'].items():assert sha(ROOT/p)==h,p
    quote_rows=read(SRC/'TARGET_QUOTES.json')['rows']
    quotes={r['signal']:r for r in quote_rows}
    fxdoc=read(SRC/'fed/bulk/RECONCILED_RATES.json')
    rates={r['asof']:r for r in fxdoc['selected_rates']}
    archived=read(ROOT/'outputs/monthly_2016_extension/EARLY_RANKS.json')
    assert len(quotes)==len(quote_rows)==len(rates)==len(fxdoc['selected_rates'])==len(archived)==53
    assert set(quotes)=={r['date'] for r in archived}
    assert set(rates)=={r['asof'] for r in archived}
    oldpools=read(ROOT/'outputs/monthly_2016_extension/POOLS.json');pools=dict(oldpools)
    oldpx={r['date'][:10]:r['close'] for r in read(ROOT/'outputs/dynamic_top100/additional_sources/tiingo/PTRCY_prices.json')}
    rows=[]
    for event in archived:
        signal,asof=event['date'],event['asof']
        qr=quotes[signal];fr=rates[asof]
        assert qr['asof']==asof and fr['signal_date']==signal and fr['known_date']<asof
        assert fr['known_date']==max(fr['indexed_release_date'],fr['page_release_date'])
        assert fr['observation_date']<=fr['known_date']
        for symbol,currency in [('601857.SS.json','CNY'),('0857.HK.json','HKD')]:
            q=qr['quotes'][symbol]
            assert q['currency']==currency and q['observation_date']<=asof
            assert q['age_calendar_days']==(date.fromisoformat(asof)-date.fromisoformat(q['observation_date'])).days
        a=qr['quotes']['601857.SS.json']['ohlc']['close'];h=qr['quotes']['0857.HK.json']['ohlc']['close']
        cny=float(fr['cny_per_usd']);hkd=float(fr['hkd_per_usd'])
        cap=valuation(a,h,cny,hkd)
        previous=oldpx[asof]*(A_COUNT+H_COUNT)/100
        assert previous<=event['top40'][-1]['market_cap_usd']
        before=[r['symbol'] for r in event['top40'][:32]]
        assert before==oldpools[signal]
        ranking=insert_ptr(event['top40'],cap);after=[r['symbol'] for r in ranking[:32]]
        pools[signal]=after
        rows.append(dict(signal=signal,asof=asof,quotes=qr['quotes'],fx=fr,
            original_ADR_proxy=previous,conditional_AH_proxy=cap,new_to_old_proxy=cap/previous,
            original_rank='outside archived40',conditional_candidate_rank=1+[r['symbol'] for r in ranking].index('PTR'),
            non_PTR_boundary=event['top40'][31],old_top32=before,new_top32=after,
            added=sorted(set(after)-set(before)),removed=sorted(set(before)-set(after))))
    early=set(quotes)
    assert len(rows)==53 and all(pools[k]==v for k,v in oldpools.items() if k not in early)
    changed=[dict(signal=r['signal'],added=r['added'],removed=r['removed']) for r in rows if r['added'] or r['removed']]
    result=dict(status='CONDITIONAL_PROXY_SENSITIVITY_NOT_PIT_CERTIFICATION',created_utc=now(),
        plan_sha256=sha(OUT/'PLAN.json'),rows=rows,summary=dict(events=53,later_unchanged=len(pools)-53,
            membership_changes=len(changed),changed_events=changed,
            proxy_ratio_min=min(r['new_to_old_proxy'] for r in rows),proxy_ratio_max=max(r['new_to_old_proxy'] for r in rows)),
        strategy_trials_added=0,returns_computed=False,original_files_modified=False,assumptions=plan['assumptions'])
    write(OUT/'DATA_COMPARISON.json',result);write(OUT/'CONDITIONAL_POOLS.json',pools)
    print(json.dumps(result['summary'],ensure_ascii=False))

class Tests(unittest.TestCase):
    def original(self):return [dict(symbol='T%02d'%i,market_cap_usd=100-i) for i in range(40)]
    def test_fx_direction(self):self.assertAlmostEqual(valuation(7,8,7,8),A_COUNT+H_COUNT)
    def test_invalid_fx(self):
        with self.assertRaises(ValueError):valuation(7,8,0,8)
    def test_outside_stays_outside(self):self.assertNotIn('PTR',[r['symbol'] for r in insert_ptr(self.original(),1)[:32]])
    def test_insert_replaces_boundary(self):
        out=insert_ptr(self.original(),200);self.assertEqual(out[0]['symbol'],'PTR');self.assertEqual(out[31]['symbol'],'T30')
    def test_duplicate_original_ptr_rejected(self):
        rows=self.original();rows[0]['symbol']='PTR'
        with self.assertRaises(ValueError):insert_ptr(rows,200)

if __name__=='__main__':
    if sys.argv[1]=='test':
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
        sys.exit(0 if result.wasSuccessful() else 1)
    {'register':register,'run':run}[sys.argv[1]]()
