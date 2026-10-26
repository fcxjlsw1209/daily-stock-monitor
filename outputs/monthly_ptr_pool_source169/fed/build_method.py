"""Offline H.10 release-index selection and sample extraction only."""
from bisect import bisect_left
from datetime import datetime
from decimal import Decimal
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'work'))
import ptr169_fed_sources as source

OUT=Path(__file__).resolve().parent


def previous_release(releases, asof):
    pos=bisect_left(releases, asof)-1
    assert pos >= 0, ('No earlier official release',asof)
    return releases[pos]


def dates_from_index(index):
    result=[]
    for year in index:
        for month in year['Months']:
            assert month['MonthValue'].startswith(year['yearValue'])
            for text in month['Dates']:
                assert text.startswith(month['MonthValue'])
                result.append(datetime.strptime(text,'%Y%m%d').date().isoformat())
    assert len(result)==len(set(result))
    return sorted(result)


def extract_sample(html):
    parser=source.Inspector(source.BASE+'/releases/h10/20160815/')
    parser.feed(html)
    text=' '.join(' '.join(parser.text).split())
    matched=re.search(r'Release Date:\s*([A-Za-z]+ \d{1,2}, \d{4})',text)
    assert matched
    release=datetime.strptime(matched[1],'%B %d, %Y').date()
    assert release.isoformat()=='2016-08-15'
    assert 'Rates in currency units per U.S. dollar except as noted' in text
    tables=[t for t in parser.tables if t and t[0][:2]==['COUNTRY','CURRENCY']]
    assert len(tables)==1
    table=tables[0]
    days=[datetime.strptime(x.replace('.','')+' '+str(release.year),'%b %d %Y').date() for x in table[0][2:]]
    assert all(day<release for day in days) and days==sorted(set(days))
    assert [d.isoformat() for d in days]==['2016-08-08','2016-08-09','2016-08-10','2016-08-11','2016-08-12']
    values={}
    for country,currency,key in [('CHINA, P.R.','YUAN','CNY'),('HONG KONG','DOLLAR','HKD')]:
        rows=[r for r in table if r[0]==country]
        assert len(rows)==1 and rows[0][1]==currency and len(rows[0])==len(days)+2
        assert not rows[0][0].startswith('*')
        values[key]=[]
        for day,value in zip(days,rows[0][2:]):
            if value=='ND':
                parsed=None
            else:
                assert Decimal(value).is_finite() and Decimal(value)>0
                parsed=value
            values[key].append(dict(observation_date=day.isoformat(),displayed_local_currency_per_usd=parsed,raw_cell=value))
    return dict(release_date=release.isoformat(),rate_unit='local currency per USD',rates=values,
                header_cells=table[0],rate_timestamp_is_not_release_timestamp=True)


def main():
    # Small synthetic check: same-day releases are excluded, including Tuesdays.
    fake=['2016-01-11','2016-01-19','2016-01-25']
    assert previous_release(fake,'2016-01-19')=='2016-01-11'
    assert previous_release(fake,'2016-01-20')=='2016-01-19'
    discovery=source.load(OUT/'DISCOVERY_FINAL.json')
    for name,digest in discovery['source_pins'].items():
        assert source.sha((OUT/name).read_bytes())==digest,name
    plan=source.verify_plan(OUT/'SOURCE_PLAN.json')
    target=source.load(ROOT/plan['target_file'])
    index_html=(OUT/'index.html').read_text()
    about=(OUT/'about.html').read_text()
    assert '$.getJSON("releaseDates.json"' in index_html
    assert '''href="/releases/h10/' +''' in index_html
    assert 'On Mondays at 4:15 p.m.' in index_html
    assert 'noon buying rates in New York' in about
    assert 'past releases are not revised and may not reflect subsequent revisions to the data' in about
    releases=dates_from_index(source.load(OUT/'releaseDates.json'))
    sample=extract_sample((OUT/'sample_20160815.html').read_text())
    assert sample['release_date'] in releases
    selected=[]
    for pair in target:
        release=previous_release(releases,pair['asof'])
        selected.append(dict(signal_date=pair['date'],asof=pair['asof'],release_date=release,
            release_url=source.BASE+'/releases/h10/'+release.replace('-','')+'/',
            same_day_release_excluded=pair['asof'] in releases,
            release_calendar_days_before_asof=(datetime.fromisoformat(pair['asof'])-datetime.fromisoformat(release)).days,
            source_page_verified=release==sample['release_date'],
            rates_extracted=release==sample['release_date']))
    period=[d for d in releases if '2016-01-01'<=d<='2020-12-31']
    distinct=sorted({r['release_url'] for r in selected})
    result=dict(
        schema='ptr169_fed_release_method_v1',created_utc=source.now(),
        source_pins=discovery['source_pins'],target_file=plan['target_file'],target_sha256=plan['target_sha256'],
        source_code_sha256=source.sha(Path(source.__file__).read_bytes()),method_code_sha256=source.sha(Path(__file__).read_bytes()),
        official_index=dict(url=source.BASE+'/releases/h10/releaseDates.json',
            observed_reference='index.html:1393 getJSON; 1427-1430 /releases/h10/ plus Dates entry',
            structure='yearValue -> Months[] -> MonthValue, Dates[]',total_dates=len(releases),
            first=releases[0],last=releases[-1],dates_2016_2020=period,
            non_monday_dates_2016_2020=[d for d in period if datetime.fromisoformat(d).weekday()!=0]),
        source_policy=dict(
            publication='Monday 4:15 p.m.; next business day if Monday is a Federal holiday. Use actual indexed release dates, not a computed Monday schedule.',
            publication_source='index.html:1365',
            time_zone_explicit_in_saved_schedule_sentence=False,
            measurement='Noon buying rates in New York for cable transfers payable in listed currencies; FRBNY certified for customs purposes.',
            measurement_source='about.html:1403-1404',
            historical_archive_policy='Past releases are not revised and may not reflect subsequent revisions to the data.',
            historical_archive_policy_source='about.html:1406',
            latest_country_data_policy='Country Data is updated through the latest H.10 release and is not the frozen release archive.',
            country_data_source='about.html:1407'),
        selection_rule='Select max indexed release_date strictly less than the frozen asof calendar date. Same-day releases are excluded completely; do not use observation date as release date.',
        selection_scope='Source URL eligibility only. This stage does not choose or carry any exchange-rate value for missing observations.',
        target_release_candidates=selected,unique_target_release_urls=distinct,
        bulk_manifest=dict(targets=len(selected),unique_urls=len(distinct),
            cached_verified_urls=[source.BASE+'/releases/h10/20160815/'],
            remaining_unfetched_urls=[u for u in distinct if u!=source.BASE+'/releases/h10/20160815/']),
        sample=sample,
        extraction_contract=[
            'Confirm HTTP200, hash, official route, page Release Date, and membership in the saved official releaseDates index.',
            'Find the unique table beginning COUNTRY/CURRENCY, then exact rows CHINA, P.R. + YUAN and HONG KONG + DOLLAR; do not use a fixed table index.',
            'Confirm local-currency-per-USD table unit and that neither currency row has the inverse-quotation star.',
            'Parse observation dates from table headers using release date context; around New Year inspect both possible years, require chronology and observations preceding release, fail on ambiguity.',
            'Preserve displayed decimal values as strings and ND as null. Do not fill a missing observation or replace it with a newer release silently.',
            'Parse all observations in each selected release. Choosing the last available rate or an older rate requires separately stated downstream policy.',
        ],
        limitations=[
            'Official policy supports unrevised historical-release values. Files were retrieved in 2026, not preserved locally at the historical publication time.',
            'Saved pages do not provide per-release actual publication timestamps or delivery/outage history. The strict prior-calendar-date rule is conservative source selection, not forensic proof of each historical delivery time.',
            'Do not use the latest Country Data series as a substitute for these dated archived releases.',
            'Only one historical release has been downloaded and parsed; the other candidate pages still require collection and validation.',
            'These are weekly published lagged noon reference rates, not simultaneous exchange closing FX quotes.',
        ],
        source_discovery_complete=True,full53_rate_extraction_complete=False,
        bulk_release_downloads_executed=False,cap_computed=False,pool_computed=False,returns_computed=False)
    source.write('RELEASE_METHOD.json',result)
    print(json.dumps(dict(output='RELEASE_METHOD.json',sha256=source.sha((OUT/'RELEASE_METHOD.json').read_bytes()),
        targets=len(selected),unique_urls=len(distinct),remaining=len(result['bulk_manifest']['remaining_unfetched_urls']),
        same_day_releases_excluded=sum(r['same_day_release_excluded'] for r in selected))))


if __name__=='__main__': main()
