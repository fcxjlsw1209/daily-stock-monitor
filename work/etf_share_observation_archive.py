"""Build a source-only observation archive; no actual feature or strategy math."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import etf_flow_source_quality as source

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/monthly_etf_primary_flow_sources'
DEST = BASE / 'reported_observations_v1'
EXPECTED_SUMMARY = '67fe2492cb9561dec5700566310fae39f1ca04de46c8e2f92e47e708d0da39ab'
EXPECTED_CALENDAR = '8ee4da69fba84db1edee37a7e0f5af5d3ab16c6257e5d37cddbd531f01404622'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path):
    return json.loads(path.read_text())

def verify(path, digest):
    if sha(path) != digest:
        raise ValueError('Source digest mismatch: ' + str(path))

def dump(path, data):
    with path.open('x') as stream:
        json.dump(data, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')

def select_rows(rows, sessions):
    mapping = {}
    for row in rows:
        if row['date'] in mapping:
            raise ValueError('Duplicate source date')
        mapping[row['date']] = row
    selected = []
    for day in sessions:
        if day not in mapping:
            raise ValueError('Missing source date: ' + day)
        row = mapping[day]
        if source.positive_decimal(row['shares_outstanding']) is None:
            raise ValueError('Invalid source shares: ' + day)
        selected.append(dict(report_date=day, shares_as_reported=row['shares_outstanding'],
                             source_row=row['source_row'], first_public_at_utc=None))
    return selected

def main():
    if DEST.exists():
        raise RuntimeError('Write-once destination already exists')
    summary_path = BASE/'SOURCE_FEASIBILITY_SUMMARY.json'
    verify(summary_path, EXPECTED_SUMMARY)
    summary = read(summary_path)
    for name, digest in summary['evidence_sha256'].items():
        # Central state was only a snapshot; its unchanged digest is checked
        # here but is deliberately not made a permanent archive dependency.
        verify(ROOT/name, digest)
    calendar_path=ROOT/'outputs/monthly_paired_target143/CALENDAR.json'
    verify(calendar_path, EXPECTED_CALENDAR)
    calendar_payload=read(calendar_path)
    if set(calendar_payload) != {'start','end','sessions','provenance'}:
        raise ValueError('Unexpected original calendar structure')
    calendar=calendar_payload['sessions']
    if not isinstance(calendar,list):
        raise ValueError('Expected original calendar sessions list')
    sessions=[d for d in calendar if '2016-05-11' <= d <= '2026-09-24']
    if len(sessions)!=2608 or sessions!=sorted(set(sessions)):
        raise ValueError('Required date axis differs')
    pins={str(summary_path.relative_to(ROOT)):sha(summary_path),
          str(calendar_path.relative_to(ROOT)):sha(calendar_path),
          'work/etf_flow_source_quality.py':sha(ROOT/'work/etf_flow_source_quality.py'),
          'work/etf_share_observation_archive.py':sha(Path(__file__))}
    pending={}
    for kind in ('ssga','ishares'):
        feasibility_path=BASE/kind/'FEASIBILITY.json'
        pins[str(feasibility_path.relative_to(ROOT))]=sha(feasibility_path)
        data=read(feasibility_path)
        for item in data['sources' if kind=='ssga' else 'funds']:
            ticker=item['symbol' if kind=='ssga' else 'ticker']
            raw=ROOT/item['source_file'] if kind=='ssga' else BASE/kind/item['source_file']
            verify(raw,item['source_sha256'])
            pins[str(raw.relative_to(ROOT))]=sha(raw)
            receipt_path=raw.with_name(raw.stem+'_RETRIEVAL.json')
            receipt=read(receipt_path)
            if receipt['status']!=200 or receipt['sha256']!=sha(raw) or receipt['bytes']!=raw.stat().st_size:
                raise ValueError('Raw receipt mismatch')
            pins[str(receipt_path.relative_to(ROOT))]=sha(receipt_path)
            if kind=='ssga':
                extracted=ROOT/item['extraction_file']
                verify(extracted,item['extraction_sha256'])
                pins[str(extracted.relative_to(ROOT))]=sha(extracted)
                rows=read(extracted)['rows']
                sheet='navhist'; column='C'; repairs=0
            else:
                rows,repairs=source.parse_ishares(raw.read_bytes())
                sheet='Historical'; column='D'
            selected=select_rows(rows,sessions)
            pending[ticker]={
                'schema':'reported_share_observations_v1', 'symbol':ticker,
                'fund_name':item['fund_name'], 'source_provider':kind,
                'source_file':str(raw.relative_to(ROOT)), 'source_sha256':sha(raw),
                'source_url':receipt['requested_url'],
                'retrieved_at_utc':receipt['retrieved_utc'],
                'receipt_file':str(receipt_path.relative_to(ROOT)), 'receipt_sha256':sha(receipt_path),
                'source_coordinates':{'sheet':sheet,'shares_column':column},
                'in_memory_xml_ampersand_repairs':repairs,
                'source_date_or_share_values_shifted':False,
                'source_date_semantics':'report row label; economic/accounting date not inferred',
                'share_unit_semantics':'not admitted by this archive; requires separate versioned action evidence',
                'publication_clock':'UNKNOWN; retrieval time is not first-public time',
                'strict_PIT':False, 'factor_admission':False, 'observations':selected,
                'full_raw_rows_retained_in_source':len(rows),
                'excluded_from_session_extract':'Off-session/outside-window rows remain in immutable original source',
                'no_fill_no_split_adjustment_no_features':True}
    if set(pending)!=set(summary['complete_raw_nav_and_share_date_coverage']['symbols']):
        raise ValueError('Unexpected source subset')
    # All inputs/rows are checked before opening the write-once output directory.
    DEST.mkdir()
    artifacts={}
    for ticker, payload in sorted(pending.items()):
        path=DEST/(ticker+'.json');dump(path,payload);artifacts[path.name]=sha(path)
    manifest={
        'schema':'source_only_observation_archive_v1',
        'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'status':'COMPLETE_SOURCE_EXTRACTION_NOT_FACTOR_ADMISSION',
        'source_axis':['2016-05-11','2026-09-24'], 'sessions_per_fund':len(sessions),
        'funds':sorted(pending),'observations':sum(len(d['observations']) for d in pending.values()),
        'input_sha256':pins,'artifacts_sha256':artifacts,
        'new_factors_models_ranks_orders_returns':0,'source_values_adjusted':False,
        'frozen_or_observer_modified':False,'no_publication_timestamps_invented':True}
    dump(DEST/'MANIFEST.json',manifest)
    print(json.dumps({'output':str(DEST.relative_to(ROOT)),'manifest_sha256':sha(DEST/'MANIFEST.json'),
                      'funds':len(pending),'observations':manifest['observations'],'factor_admission':False}))

if __name__=='__main__':main()
