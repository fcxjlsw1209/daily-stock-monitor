"""Independent PTR169 source-to-pool audit; offline and no strategy returns.

Uses raw Yahoo/Tencent JSON, original Fed HTML, Decimal valuation, and the
archived top40. Does not import production or previous audit helpers.
"""
import argparse
import hashlib
import html
import json
import math
import platform
import re
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, localcontext
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT/'outputs/monthly_ptr_pool_source169'
OUT = SRC/'conditional_pool'
PINS = {}
F = ('open','high','low','close')
HOLIDAYS = {
    ('601857.SS.json','2017-01-30'): '2017-01-26',
    ('601857.SS.json','2017-05-30'): '2017-05-26',
    ('601857.SS.json','2020-01-30'): '2020-01-23',
    ('0857.HK.json','2017-01-30'): '2017-01-27',
    ('0857.HK.json','2017-05-30'): '2017-05-29',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def track(path, expected=None):
    path = Path(path)
    actual = sha(path)
    assert expected is None or actual == expected, str(path)
    PINS[str(path.relative_to(ROOT))] = actual
    return path


def read(path, expected=None):
    return json.loads(track(path, expected).read_text())


def bind(mapping):
    for name, expected in mapping.items():
        track(ROOT/name, expected)


def clean(s):
    return ' '.join(html.unescape(re.sub(r'<[^>]+>', ' ', s)).split())


def fed_page(path):
    raw = track(path).read_text(encoding='utf-8-sig')
    text = clean(raw)
    matches = re.findall(r'Release Date:\s*([A-Za-z]+ \d{1,2}, \d{4})', text)
    assert len(matches) == 1, (path, matches)
    release = datetime.strptime(matches[0], '%B %d, %Y').date()
    assert 'Rates in currency units per U.S. dollar except as noted' in text
    rows = [[clean(cell) for cell in re.findall(r'<t[dh]\b[^>]*>(.*?)</t[dh]>', row, re.I|re.S)]
            for row in re.findall(r'<tr\b[^>]*>(.*?)</tr>', raw, re.I|re.S)]
    header, = [r for r in rows if r[:2] == ['COUNTRY','CURRENCY']]
    china, = [r for r in rows if r[:2] == ['CHINA, P.R.','YUAN']]
    hongkong, = [r for r in rows if r[:2] == ['HONG KONG','DOLLAR']]
    assert len(header) == len(china) == len(hongkong) == 7
    observations = []
    for label, cny, hkd in zip(header[2:], china[2:], hongkong[2:]):
        month, day = re.fullmatch(r'([A-Za-z]+)\.?\s+(\d+)', label).groups()
        m = datetime.strptime(month[:3], '%b').month
        obs = date(release.year - (m > release.month), m, int(day))
        assert 0 < (release-obs).days <= 14
        values = []
        for value in (cny,hkd):
            if value in ('ND','ND.'):
                values.append(None)
            else:
                assert re.fullmatch(r'\d+\.\d+', value) and Decimal(value) > 0
                values.append(value)
        observations.append(dict(date=obs.isoformat(), cny=values[0], hkd=values[1]))
    assert [r['date'] for r in observations] == sorted({r['date'] for r in observations})
    return release.isoformat(), observations


def yahoo(name, symbol, currency, tz):
    raw = read(SRC/name)['chart']
    assert raw['error'] is None and len(raw['result']) == 1
    body = raw['result'][0]
    assert body['meta']['symbol'] == symbol and body['meta']['currency'] == currency
    assert body['meta']['exchangeTimezoneName'] == tz and body['meta']['dataGranularity'] == '1d'
    q, = body['indicators']['quote']
    assert all(len(q[k]) == len(body['timestamp']) for k in F)
    rows = {}
    for i, stamp in enumerate(body['timestamp']):
        day = datetime.fromtimestamp(stamp, ZoneInfo(tz)).date().isoformat()
        assert day not in rows
        rows[day] = {k:q[k][i] for k in F}
    return rows


def valid_ohlc(row):
    assert all(type(row[k]) in (float,int) and math.isfinite(row[k]) and row[k] > 0 for k in F)
    assert row['low'] <= min(row['open'],row['close']) <= max(row['open'],row['close']) <= row['high']


def close_enough(a,b):
    assert math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=0.0001), (a,b)


def main(write=False):
    plan = read(OUT/'PLAN.json', '14fa6e900df478414b2f6fccac2ee36180bb09ae63f3425b0028bac375203b1f')
    bind(plan['input_sha256'])
    track(ROOT/'work/ptr169_pool_sensitivity.py', plan['script_sha256'])
    assert plan['runtime'] == platform.python_version()
    qplan = read(SRC/'TARGET_QUOTES_PLAN.json', '7f3f6b6afce4436ee2b37350df6bcd252d79e452be37a886f2216a5adb78a7ea')
    bind(qplan['input_sha256'])
    track(ROOT/'work/ptr169_target_quotes.py', qplan['code_sha256'])
    target = read(SRC/'TARGET_QUOTES.json')
    assert target['plan_sha256'] == sha(SRC/'TARGET_QUOTES_PLAN.json')
    result = read(OUT/'DATA_COMPARISON.json', '361904b034a24984a79dfda33748b39312c2033f973f63e4b73468c88d86964f')
    output_pools = read(OUT/'CONDITIONAL_POOLS.json', '9d6b38754e2732d3aaf31a1fd1f1c3671af4a9b6ecbc2673a44d8339b81fd289')
    assert result['plan_sha256'] == sha(OUT/'PLAN.json')
    assert target['created_utc'] < plan['registered_utc'] < result['created_utc']
    annual = read(SRC/'annual/INDEPENDENT_REVIEW.json')
    bind(annual['source_and_input_sha256'])
    first = annual['rows'][0]
    assert first['year'] == 2015 and first['period_end'] == '2015-12-31'
    a_count = first['independent_counts']['A']['post']
    h_count = first['independent_counts']['H']['post']
    assert plan['shares'] == dict(A=a_count,H=h_count,total=a_count+h_count)
    assert (a_count,h_count) == (161922077818,21098900000)

    calendar = read(SRC/'calendar/DECISIONS.json')
    closed = {(r['symbol'],r['date']):r for r in calendar['rows'] if r['classification']=='OFFICIAL_EXCHANGE_CLOSED'}
    assert set(closed) == set(HOLIDAYS)
    qbooks = {'601857.SS.json':yahoo('601857.SS.json','601857.SS','CNY','Asia/Shanghai'),
              '0857.HK.json':yahoo('0857.HK.json','0857.HK','HKD','Asia/Hong_Kong')}
    tencent = read(SRC/'missing_quote/tencent_201904.json')
    assert tencent['code'] == 0 and set(tencent['data']) == {'sh601857'}
    bar, = [r for r in tencent['data']['sh601857']['day'] if r[0] == '2019-04-29']
    replacement = dict(zip(('open','close','high','low'), map(float,bar[1:5])))
    valid_ohlc(replacement)
    quote_review = read(SRC/'missing_quote/INDEPENDENT_REVIEW.json')
    bind(quote_review['input_sha256'])
    assert quote_review['quote']['ohlc'] == replacement

    rates_doc = read(SRC/'fed/bulk/RECONCILED_RATES.json')
    bind(rates_doc['inputs'])
    index = read(SRC/'fed/releaseDates.json')
    index_dates = sorted({date(int(x[:4]),int(x[4:6]),int(x[6:])).isoformat()
                          for y in index for m in y['Months'] for x in m['Dates']})
    fed = {}
    receipt_paths = [SRC/'fed/sample_20160815.html.receipt.json'] + sorted((SRC/'fed/bulk').glob('*.html.receipt.json'))
    for receipt_path in receipt_paths:
        receipt = read(receipt_path)
        match = re.fullmatch(r'https://www\.federalreserve\.gov/releases/h10/(\d{8})/',receipt['url'])
        assert match and receipt['status'] == 200
        ds = match[1]
        indexed = date(int(ds[:4]),int(ds[4:6]),int(ds[6:])).isoformat()
        assert indexed in index_dates and indexed not in fed
        path = receipt_path.with_name(receipt_path.name.removesuffix('.receipt.json'))
        track(path, receipt['body_sha256'])
        assert path.stat().st_size == receipt['bytes']
        release, observations = fed_page(path)
        fed[indexed] = dict(release=release,known=max(indexed,release),observations=observations,
                            url=receipt['url'],body_sha256=receipt['body_sha256'])
    assert len(fed) == 54

    archived = read(ROOT/'outputs/monthly_2016_extension/EARLY_RANKS.json')
    oldpools = read(ROOT/'outputs/monthly_2016_extension/POOLS.json')
    expected_pools = dict(oldpools)
    assert len(archived) == len(target['rows']) == len(result['rows']) == len(rates_doc['selected_rates']) == 53
    def keyed(rows,key):
        out = {r[key]:r for r in rows}
        assert len(out) == len(rows)
        return out
    quotes = keyed(target['rows'],'signal')
    products = keyed(result['rows'],'signal')
    rates = keyed(rates_doc['selected_rates'],'asof')
    assert set(quotes) == set(products) == {r['date'] for r in archived}
    assert set(rates) == {r['asof'] for r in archived}
    tiingo = read(ROOT/'outputs/dynamic_top100/additional_sources/tiingo/PTRCY_prices.json')
    oldpx = {r['date'][:10]:r['close'] for r in tiingo}
    assert len(oldpx) == len(tiingo)
    changes, checked, used_holidays = [], [], []
    exact, extra, last = 0,0,0
    ratios = []
    for event in archived:
        signal, asof = event['date'], event['asof']
        qr, output, fr = quotes[signal],products[signal],rates[asof]
        assert qr['asof'] == output['asof'] == asof < signal
        verified_quotes = {}
        for symbol, currency in [('601857.SS.json','CNY'),('0857.HK.json','HKD')]:
            q, book = qr['quotes'][symbol],qbooks[symbol]
            key = (symbol,asof)
            if key in HOLIDAYS:
                obs = HOLIDAYS[key]
                assert asof not in book and obs == max(d for d in book if d < asof)
                declaration = closed[key]
                start = declaration.get('announced_closure_start',declaration.get('original_announced_closure_start',asof))
                end = declaration.get('announced_closure_end',declaration.get('original_announced_closure_end',asof))
                d = date.fromisoformat(obs)+timedelta(days=1)
                while d <= date.fromisoformat(asof):
                    assert d.weekday() >= 5 or start <= d.isoformat() <= end
                    d += timedelta(days=1)
                price, source, kind = book[obs],symbol,'last_actual_quote_during_documented_exchange_closure'
                used_holidays.append(dict(symbol=symbol,asof=asof,observed=obs)); last += 1
            elif key == ('601857.SS.json','2019-04-29'):
                assert asof not in book
                obs, price, source, kind = asof,replacement,'missing_quote/RESOLUTION.json','exact_asof_corroborated_tencent_quote'
                extra += 1
            else:
                obs, price, source, kind = asof,book[asof],symbol,'exact_asof_actual_quote'; exact += 1
            valid_ohlc(price)
            expected = dict(source=source,observation_date=obs,selection=kind,
                            age_calendar_days=(date.fromisoformat(asof)-date.fromisoformat(obs)).days,
                            currency=currency,ohlc=price)
            assert q == expected and q == output['quotes'][symbol]
            verified_quotes[symbol] = price

        original_release = max(d for d in index_dates if d < asof)
        assert fr['originally_selected_index_date'] == original_release
        # The first eligible page must be present; an uncollected intervening page
        # is never skipped based on availability of a more convenient cached rate.
        for indexed in reversed([d for d in index_dates if d <= original_release]):
            assert indexed in fed, ('Missing prior release required for audit',indexed,asof)
            page = fed[indexed]
            common = [r for r in page['observations'] if r['cny'] is not None and r['hkd'] is not None]
            if page['known'] < asof and common:
                chosen = max(common,key=lambda r:r['date'])
                break
        assert chosen['date'] < page['known'] < asof
        assert fr['indexed_release_date'] == indexed and fr['page_release_date'] == page['release']
        assert fr['known_date'] == page['known'] and fr['observation_date'] == chosen['date']
        assert fr['url'] == page['url'] and fr['body_sha256'] == page['body_sha256']
        assert fr['cny_per_usd'] == chosen['cny'] and fr['hkd_per_usd'] == chosen['hkd']
        assert not fr['cross_release_fill'] and not fr['cross_currency_date_mix']
        assert fr['signal_date'] == signal and output['fx'] == fr
        assert fr['quote_age_calendar_days'] == (date.fromisoformat(asof)-date.fromisoformat(chosen['date'])).days
        with localcontext() as ctx:
            ctx.prec = 45
            a = Decimal(str(verified_quotes['601857.SS.json']['close']))
            h = Decimal(str(verified_quotes['0857.HK.json']['close']))
            value = Decimal(a_count)*a/Decimal(chosen['cny'])+Decimal(h_count)*h/Decimal(chosen['hkd'])
            previous = Decimal(str(oldpx[asof]))*Decimal(a_count+h_count)/Decimal(100)
            ratio = value/previous
        close_enough(value,output['conditional_AH_proxy'])
        close_enough(previous,output['original_ADR_proxy'])
        close_enough(ratio,output['new_to_old_proxy']); ratios.append(float(ratio))
        original = event['top40']
        symbols = [r['symbol'] for r in original]
        assert len(original) == len(set(symbols)) == 40 and 'PTR' not in symbols
        base_keys = [(-Decimal(str(r['market_cap_usd'])),r['symbol']) for r in original]
        assert base_keys == sorted(base_keys) and previous <= Decimal(str(original[-1]['market_cap_usd']))
        assert symbols[:32] == oldpools[signal] == output['old_top32']
        assert output['non_PTR_boundary'] == original[31]
        ranked = sorted(base_keys+[(-value,'PTR')])
        after = [s for _,s in ranked[:32]]
        rank = [s for _,s in ranked].index('PTR')+1
        assert after == output['new_top32'] == output_pools[signal]
        assert rank == output['conditional_candidate_rank']
        added, removed = sorted(set(after)-set(symbols[:32])),sorted(set(symbols[:32])-set(after))
        assert added == output['added'] and removed == output['removed']
        if added or removed:
            assert added == ['PTR'] and removed == [symbols[31]]
            changes.append(dict(signal=signal,added=added,removed=removed))
        expected_pools[signal] = after
        checked.append(dict(signal=signal,asof=asof,formula_match=True,rank_match=True,quote_match=True,
                            fx_match=True,conditional_candidate_rank=rank,membership_changed=bool(added)))
    assert expected_pools == output_pools
    early = set(quotes)
    later = [k for k in oldpools if k not in early]
    assert len(oldpools) == len(output_pools) == 122 and len(later) == 69
    assert all(oldpools[k] == output_pools[k] for k in later)
    assert (exact,extra,last) == (100,1,5)
    assert len(changes) == 33 == result['summary']['membership_changes']
    assert changes == result['summary']['changed_events']
    close_enough(min(ratios),result['summary']['proxy_ratio_min'])
    close_enough(max(ratios),result['summary']['proxy_ratio_max'])
    report = dict(schema='ptr169_conditional_pool_independent_v1',created_utc=datetime.now(timezone.utc).isoformat(),
        verdict='PASS_CONDITIONAL_DATA_SENSITIVITY_NOT_PIT_CERTIFICATION',code_sha256=sha(__file__),input_sha256=PINS,
        method='Raw Yahoo/Tencent JSON and all54 original Fed HTML releases independently parsed; Decimal recomputation and tuple sort of frozen top40; no production helper imports.',
        checks=dict(events=53,source_quote_cells=106,exact_yahoo=exact,exact_tencent=extra,documented_holiday_last_quotes=last,
                    fed_pages_parsed=54,formula_rows=53,old_PTR_outside40=53,new_ranking_rows=53,
                    membership_changes=len(changes),unchanged_later_pools=len(later),total_pools=122,
                    explicit_plan_before_computation=True,all_source_pins_unchanged=True),
        fixed_shares=dict(A=a_count,H=h_count,source='2015 year-end; annual independent review and original source pins rechecked'),
        used_holidays=used_holidays,changed_events=changes,rows=checked,
        limits=[
            'This is one fixed2015 A/H share-count conditional sensitivity, not verified daily historical outstanding shares. Subsequent annual evidence is only retrospective consistency support.',
            '2015 pre-Aug2016 filing index support is indirect; the original filed A/H field timestamp bridge is not independently certified.',
            'Raw equity prices were retrieved in2026; exact original data vintages, delivery times and all corporate actions remain uncertified.',
            'FX is deliberately lagged: max(index date,page date)<asof, latest common non-ND observation. It is a Fed noon rate, not same-date stock-market closing FX.',
            'Five closed-market cells retain older actual quotation dates; one open-market gap uses the separately corroborated Tencent bar. No cache is repaired or source value replaced.',
            'The candidate rank is within the frozen original40 plus PTR. Other issuer values and the legacy Shell proxy remain unchanged; no complete full-market census is claimed.',
            'A changed pool is not an unchanged model input. Recompute affected ranks/features and relevant model training under a separate registration before assessing strategy returns.'
        ],new_network_requests=0,price_cache_modified=False,existing_files_modified=False,strategy_returns_computed=False)
    for name, expected in PINS.items():
        assert sha(ROOT/name) == expected,name
    if write:
        with (OUT/'INDEPENDENT_REVIEW.json').open('x') as stream:
            json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(verdict=report['verdict'],checks=report['checks'],input_files=len(PINS),written=write)))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--write',action='store_true')
    main(parser.parse_args().write)
