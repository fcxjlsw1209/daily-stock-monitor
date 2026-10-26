"""Source-only Microsoft periodic-release identity and clock verification.

This validates original dated quarterly pages, not completeness of every news
announcement. It grants no calendar/factor/strategy admission. No prices,
financial amounts, model inputs, orders or return arrays are loaded.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/monthly_msft_event_evidence156'
BASE = 'outputs/monthly_earnings_repair/microsoft/'
FIXED = {
    'outputs/monthly_earnings_repair/EVENTS.json': '1e027fc8b933af9753241c15a2e0dcc5a32daa13205ba4d74f90b64db6e2e103',
    'outputs/monthly_paired_target143/CALENDAR.json': '8ee4da69fba84db1edee37a7e0f5af5d3ab16c6257e5d37cddbd531f01404622',
    'outputs/monthly_earnings_repair/event_schema.py': 'cdac20a9dadd88dc7b7dffc7937651d0a8b7473b0255957a572b1ea561ccedff',
}
ZERO = dict(price_values_read=0, financial_values_extracted=0, factor_calculations=0,
            model_fits=0, orders=0, returns=0, new_strategy_trials=0)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def local(value):
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    require(path.is_relative_to(ROOT) and path.resolve() == path, 'Nonlocal/symlink input')
    return path


def sha(value):
    return hashlib.sha256(local(value).read_bytes()).hexdigest()


def read(value):
    return json.loads(local(value).read_text())


def write(name, value):
    OUT.mkdir(exist_ok=True, parents=True)
    with (OUT / name).open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, local(path))
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def sources():
    result = list(FIXED) + [BASE+'MANIFEST.json', BASE+'collect.py',
        'outputs/monthly_earnings_repair/research_calendar.py', 'work/after155_msft_event_evidence.py',
        'outputs/monthly_event_scope156/MANIFEST.json',
        'outputs/monthly_event_scope156/INTERVALS.json',
        'outputs/monthly_event_scope156/SLOTS.json']
    for year in range(2015, 2027):
        for quarter in range(1, 5):
            stem = BASE+'raw/FY%dQ%d' % (year, quarter)
            result.extend([stem+'.html', stem+'.meta.json'])
    pins = {name:sha(name) for name in result}
    for path, expected in read('outputs/monthly_paired_target143/CALENDAR.json')['provenance']['source_hashes'].items():
        name = str(local(path).relative_to(ROOT))
        require(sha(name) == expected, 'Calendar dependency mismatch '+name)
        pins[name] = expected
    return pins


def validate_pins(pins):
    for name, digest in pins.items():
        require(sha(name) == digest, 'Source changed: '+name)
    require(all(pins.get(name) == digest for name, digest in FIXED.items()), 'Fixed source mismatch')


def tests(parser):
    text = '<h1>Earnings Release FY20 Q2</h1><p>REDMOND, Wash. — January 29, 2020 — Microsoft today announced results for the quarter ended December 31, 2019.</p>'
    def parse(body=text, year=2020, q=2):
        return parser.parse(body, year, q, 'https://www.microsoft.com/example', Path('synthetic.html'), 'synthetic')
    require(parse()['release_date'] == '2020-01-29', 'Actual dateline')
    require(parse()['fiscal_period_end_actual'] == '2019-12-31', 'Actual period')
    require(parse()['publication_timestamp'] is None, 'Do not invent clock')
    require(parse(text+'<p>Call starts at 2:30 p.m. Pacific time</p>')['publication_timestamp'] is None, 'Call is not publication')
    for args in [(text,2020,3), (text.replace('REDMOND, Wash.', 'Elsewhere'),2020,2),
                 (text.replace('December 31, 2019', 'December 31, 2021'),2020,2),
                 (text.replace('December 31, 2019', 'September 30, 2019'),2020,2)]:
        try:
            parse(*args)
        except ValueError:
            pass
        else:
            raise ValueError('Synthetic invalid identity/dateline/period accepted')
    return 8


def products(expected):
    require(sha(OUT/'SPEC.json') == expected, 'Exact frozen spec required')
    spec = read(OUT/'SPEC.json')
    require(spec['runtime_python'] == sys.version, 'Runtime mismatch')
    validate_pins(spec['hashes'])
    parser = module('msft_source_parser156', BASE+'collect.py')
    calendars = module('research_calendar', 'outputs/monthly_earnings_repair/research_calendar.py')
    schema = module('msft_clock_schema156', 'outputs/monthly_earnings_repair/event_schema.py')
    frozen_calendar = read('outputs/monthly_paired_target143/CALENDAR.json')
    calendar = calendars.market_calendar(frozen_calendar['start'], frozen_calendar['end'])
    require([str(d.date()) for d in calendar.sessions] == frozen_calendar['sessions'], 'Exact session axis')
    checks = tests(parser)
    raw = read('outputs/monthly_earnings_repair/EVENTS.json')
    old = {e['event_id']:e for e in raw if e['symbol']=='MSFT'}
    scope = [r for r in read('outputs/monthly_event_scope156/INTERVALS.json') if r['symbol']=='MSFT']
    require(len(scope)==1 and len(scope[0]['existing_event_ids'])==44, 'Frozen MSFT consumed scope')
    rows=[]
    for year in range(2015,2027):
        for quarter in range(1,5):
            stem=BASE+'raw/FY%dQ%d' % (year,quarter)
            meta=read(stem+'.meta.json'); path=local(stem+'.html')
            require(meta['http_status']==200 and sha(path)==meta['sha256'], 'Source receipt/hash')
            expected_url=parser.TEMPLATE.format(year=year,quarter=quarter)
            require(meta['url']==expected_url, 'Source identity URL')
            parsed=parser.parse(path.read_text(),year,quarter,expected_url,path,sha(path))
            normalized=schema.normalize_event(dict(symbol='MSFT',asset_type='equity',official=dict(
                verified=True,publication_kind='release',actual_period_end=parsed['fiscal_period_end_actual'],
                publication_date=parsed['release_date'],publication_timezone='America/Los_Angeles',source_url=expected_url)),calendar)
            require(normalized['status']=='ready', 'Clock normalizer rejected source')
            identifier='MSFT_'+parsed['release_date']
            prior=old.get(identifier)
            keys=('publication_date','publication_timezone','precision','available_session','known_by_at','date_only_upper_bound_at','official_actual_period_end')
            if prior:
                require(all(normalized[k]==prior[k] for k in keys), 'Parent clock mismatch '+identifier)
                require((prior['fiscal_year'],prior['fiscal_quarter'])==(year,quarter), 'Parent fiscal identity mismatch')
            rows.append(dict(event_id=identifier,fiscal_year=year,fiscal_quarter=quarter,
                source=dict(path=stem+'.html',sha256=sha(path),receipt=stem+'.meta.json',url=expected_url),
                identity_evidence=dict(title='Earnings Release FY%02d Q%d'%(year%100,quarter),
                    dateline_literal=parsed['date_literal'],period_end_literal=parsed['period_literal']),
                clock={k:normalized[k] for k in keys},consumed=identifier in scope[0]['existing_event_ids'],
                parent_event_present=prior is not None, first_public_time_verified=False))
    ids=[r['event_id'] for r in rows]
    require(len(ids)==len(set(ids))==48, 'One unique source per observed period')
    consumed=[r for r in rows if r['consumed']]
    require({r['event_id'] for r in consumed}==set(scope[0]['existing_event_ids']), 'All consumed source pages verified')
    require(all(4*b['fiscal_year']+b['fiscal_quarter']-(4*a['fiscal_year']+a['fiscal_quarter'])==1
                for a,b in zip(consumed,consumed[1:])), 'Consecutive consumed fiscal quarters')
    summary=dict(stage='MSFT_ORIGINAL_PERIODIC_RELEASE_IDENTITY_AND_CLOCK_VERIFIED',spec_sha256=expected,
        source_pages_verified=48,consumed_events_verified=44,original_slots_covered=len(scope[0]['source_slot_ids']),
        required_anchor_window=scope[0]['required_anchor_window'],synthetic_checks=checks,
        rows_have_consecutive_fiscal_quarters=True,source_hash_and_receipt_mismatches=0,parent_clock_mismatches=0,
        source_scope='Original full quarterly results at issuer FY/quarter canonical URLs; date-only conservative timing.',
        original_raw_and_catalog_changed=False,calendar_completeness_verified=False,factor_admitted=False,
        remaining_requirements=['Independent source/clock review.',
            'Source-supported interval enumeration and treatment of any distinct correction/preliminary entries; one page per fiscal quarter is not by itself all-news completeness.',
            'Reconcile vendor date leads and freeze regular-release taxonomy before any formation-window factor.'],
        limitations=['Current historical issuer pages, not proven contemporaneously captured bytes.',
            'No extraction or admission of financial amounts from these pages.'],**ZERO)
    validate_pins(spec['hashes'])
    return rows,summary


def main():
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['freeze','run','verify']);ap.add_argument('--spec-sha')
    args=ap.parse_args()
    if args.phase=='freeze':
        pins=sources();validate_pins(pins)
        write('SPEC.json',dict(stage='SOURCE_ONLY_FROZEN_SCOPE',created_utc=datetime.now(timezone.utc).isoformat(),
            runtime_python=sys.version,hashes=pins,issuer='MSFT',fiscal_years=[2015,2026],quarters=[1,2,3,4],
            calendar_completeness_verified=False,factor_admitted=False,**ZERO))
        print(json.dumps({'phase':'freeze','spec_sha256':sha(OUT/'SPEC.json')}));return
    rows,summary=products(args.spec_sha)
    if args.phase=='run':
        write('EVENT_EVIDENCE.json',rows);write('SUMMARY.json',summary)
        write('MANIFEST.json',dict(spec_sha256=args.spec_sha,outputs={n:sha(OUT/n) for n in ['EVENT_EVIDENCE.json','SUMMARY.json']}))
    else:
        require(read(OUT/'EVENT_EVIDENCE.json')==rows and read(OUT/'SUMMARY.json')==summary,'Output reproduction mismatch')
        manifest=read(OUT/'MANIFEST.json')
        require(manifest['spec_sha256']==args.spec_sha and all(sha(OUT/n)==h for n,h in manifest['outputs'].items()),'Manifest mismatch')
    print(json.dumps({'phase':args.phase,**summary},ensure_ascii=False))


if __name__=='__main__':
    main()
