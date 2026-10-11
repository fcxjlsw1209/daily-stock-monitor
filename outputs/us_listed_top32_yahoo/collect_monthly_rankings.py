"""Collect dated published rankings for monthly research, without fitting a model.

This is a retrospective vendor archive, not proof of an unrevised PIT vintage.
Keep the global list, US listing evidence, exclusions, and original HTML.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import time
import requests
from lxml import html

P = Path(__file__).resolve().parent
D = P / 'monthly_rank_sources'
D.mkdir(exist_ok=True)
FIRST = '2023-01-04'
EXCHANGES = {'NYSE', 'NASDAQ', 'NYSE ARCA', 'NYSE MKT', 'AMEX', 'BATS'}


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_ranking(body, requested_date):
    doc = html.fromstring(body)
    title = ''.join(doc.xpath('//h1//text()')).strip()
    prefix = 'Largest companies by marketcap on '
    if not title.startswith(prefix):
        raise ValueError('No dated ranking heading')
    actual = datetime.strptime(title[len(prefix):], '%A, %B %d, %Y').date().isoformat()
    if actual != requested_date:
        raise ValueError('Date mismatch: requested %s, returned %s' % (requested_date, actual))
    rows = []
    for tr in doc.xpath('//tbody/tr'):
        cells = tr.xpath('./td')
        code = tr.xpath('.//*[contains(@class,"company-code")]')
        name = tr.xpath('.//*[contains(@class,"company-name")]')
        link = tr.xpath('.//div[@class="name-div"]/a/@href')
        if not code or len(cells) < 4:
            continue
        rows.append(dict(global_rank=int(cells[1].get('data-sort')),
            vendor_symbol=code[0].text_content().strip(), name=name[0].text_content().strip(),
            market_cap_usd=float(cells[3].get('data-sort')),
            issuer_url='https://companiesmarketcap.com' + link[0]))
    if len(rows) < 100 or len({r['issuer_url'] for r in rows}) != len(rows):
        raise ValueError('Incomplete or duplicate published ranking')
    if any(a['market_cap_usd'] < b['market_cap_usd'] for a,b in zip(rows,rows[1:])):
        raise ValueError('Ranking is not descending by market cap')
    return rows


def listing_status(symbol, date, evidence):
    records = [r for r in evidence if r['symbol'] == symbol and r['exchange'] in EXCHANGES]
    active = []
    for r in records:
        if r.get('ipoDate', '9999') > date:
            continue
        end = r.get('delistingDate')
        if end and end != 'null' and end <= date:
            continue
        # Historical active snapshots alone do not establish continued listing.
        active_dates = [f[:10] for f in r.get('evidence_files', []) if f.endswith('_active.csv')]
        if any(d >= date for d in active_dates):
            active.append(r)
    contradictory = any(r.get('delistingDate') not in (None, 'null') and
                        r['delistingDate'] <= date for r in records) and bool(active)
    return active, contradictory


def filter_us(rows, date, evidence, catalog):
    by_url = {v['url']:s for s,v in catalog.items()}
    selected, excluded = [], []
    for row in rows:
        symbol = by_url.get(row['issuer_url'], row['vendor_symbol'])
        symbol = {'GOOG':'GOOGL', 'BRK.A':'BRK-B', 'BRK.B':'BRK-B'}.get(symbol,symbol)
        active, conflict = listing_status(symbol, date, evidence)
        item = dict(row, symbol=symbol)
        if active:
            item.update(us_rank=len(selected)+1, listing_evidence=active,
                        listing_conflict=conflict)
            selected.append(item)
        else:
            item['reason'] = 'No active US exchange listing established in cached registry on date'
            excluded.append(item)
    if len(selected) < 32:
        raise ValueError('Fewer than 32 supported US listed issuers')
    return dict(top32=selected[:32], boundary33_40=selected[32:40],
                excluded=excluded, us_issuers_in_published_ranking=len(selected),
                selection_status='research_candidate_pending_listing_and_boundary_validation')


def main():
    periods = json.loads((P.parent/'monthly_supplied_ledger_audit/PERIODS.json').read_text())
    dates = sorted({r['start'] for r in periods})
    evidence = json.loads((P/'LISTING_EVIDENCE.json').read_text())
    catalog = json.loads((P/'broad_caps/CATALOG.json').read_text())
    result = dict(source='CompaniesMarketCap Time Machine', source_first_date=FIRST,
        requested_dates=dates, periods=[], unavailable_before_source=[d for d in dates if d<FIRST],
        failures=[], historical_pool_ready=False,
        policy='Published global dated ranking filtered by historical US exchange listing evidence; no index gate. No author holdings used to select members.',
        vintage='Retrospective archive; original publication timestamp/revisions not certified',
        model_search_paused=True)
    limit = D/'RATE_LIMIT.json'
    today = datetime.now(timezone.utc).date().isoformat()
    throttled = limit.exists() and json.loads(limit.read_text())['utc_date'] == today
    with requests.Session() as session:
        session.headers.update({'User-Agent':'Mozilla/5.0'})
        for date in dates:
            if date < FIRST:
                continue
            url = 'https://companiesmarketcap.com/time-machine/' + date + '/'
            raw = D/('cmc_' + date.replace('-','_') + '.html')
            metadata = raw.with_suffix('.source.json')
            try:
                if not raw.exists():
                    if throttled:
                        result['failures'].append(dict(date=date, reason='source_rate_limited'))
                        continue
                    r = session.get(url,timeout=25)
                    if r.status_code == 429:
                        dump(limit, dict(utc_date=today, retry_after=r.headers.get('Retry-After')))
                        throttled = True
                        result['failures'].append(dict(date=date, reason='HTTP429'))
                        continue
                    r.raise_for_status()
                    # Validate before caching under requested date.
                    parse_ranking(r.content, date)
                    raw.write_bytes(r.content)
                    dump(metadata, dict(requested_url=url, response_url=r.url,
                        retrieved_utc=datetime.now(timezone.utc).isoformat(), sha256=digest(raw)))
                    time.sleep(.25)
                rows = parse_ranking(raw.read_bytes(), date)
                chosen = filter_us(rows, date, evidence, catalog)
                result['periods'].append(dict(date=date, source_url=url, raw_file=str(raw),
                    raw_sha256=digest(raw), global_ranking=rows, **chosen))
                print(date, 'published ranking',len(rows), 'US candidates',chosen['us_issuers_in_published_ranking'],flush=True)
            except (requests.RequestException, ValueError, KeyError) as e:
                result['failures'].append(dict(date=date, reason=str(e)[:240]))
            dump(P/'PUBLISHED_MONTHLY_RANKINGS.json', result)
    dump(P/'PUBLISHED_MONTHLY_RANKINGS.json', result)
    lines=['Monthly US-listed Top32 candidates from published historical rankings',
           'Research candidates only; listing exclusions and boundary values still require validation.',
           'No factor filtering. ETF list is separate. No model or paper ledger changed.', '']
    for entry in result['periods']:
        lines += [entry['date'], ', '.join('%d %s' % (r['us_rank'],r['symbol']) for r in entry['top32']),
                  'Boundary: ' + ', '.join('%d %s' % (r['us_rank'],r['symbol']) for r in entry['boundary33_40']), '']
    lines += ['Earlier missing dates: '+', '.join(result['unavailable_before_source']),
              'Failed dates: '+str(result['failures'])]
    (P/'PUBLISHED_MONTHLY_RANKINGS.txt').write_text('\n'.join(lines)+'\n')
    print('DONE',len(result['periods']),'dates; earlier missing',len(result['unavailable_before_source']),
          'failed',len(result['failures']),flush=True)


if __name__ == '__main__':
    main()
