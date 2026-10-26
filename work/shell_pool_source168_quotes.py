"""Registered, offline transcription and source comparison of Shell A/B quotes."""
import argparse
import csv
import hashlib
import io
import json
import math
import statistics
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/monthly_shell_pool_source168'
FILES = {
    'xls': 'outputs/monthly_shell_pool_source168/external_followup/euroland_browser_quotes.xls',
    'receipt': 'outputs/monthly_shell_pool_source168/external_followup/euroland_browser_quotes.receipt.json',
    'wheel': 'work/shell168_xls_reader_wheels/xlrd-2.0.1-py2.py3-none-any.whl',
    'spy': 'outputs/index_union/sources/yahoo_prices/SPY.csv',
    'yahoo_shel': 'outputs/monthly_published44_reconstruction/prices/SHEL.csv',
    'stooq': 'outputs/dynamic_top100/additional_sources/kaggle_stooq_2017.zip',
    'signal_dates': 'outputs/monthly_2016_extension/EARLY_RANKS.json',
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_once(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def now():
    return datetime.now(timezone.utc).isoformat()


def register():
    plan = {
        'schema': 'shell168_quotes_plan_v1', 'registered_utc': now(),
        'scope': 'Transcribe source quote cells, validate dates/numbers, compare against independent local quote files. No cap, pool, model, trade or return computation.',
        'source_files': {key: {'path': path, 'sha256': digest(ROOT / path)} for key, path in FILES.items()},
        'script_sha256': digest(Path(__file__)),
        'period': ['2016-01-01', '2020-11-30'],
        'mapping': {'sheet': 'Share Data', 'header_row_1based': 4, 'date_column': 'A', 'class_A': {'close': 'B', 'volume': 'C'}, 'class_B': {'close': 'D', 'volume': 'E'}},
        'required': ['Exact class ADR NYSE and USD headers', 'Unique ascending dates', 'Finite positive closing quotes and nonnegative integral volumes', 'No interpolation, imputation or filling'],
        'calendar_check': 'Compare date set against local SPY dates, not a claim of independent exchange-calendar certification',
        'overlap_check': {'close_tolerance_usd': 0.0051, 'stooq_members': ['Stocks/rds-a.us.txt', 'Stocks/rds-b.us.txt'], 'meaning': 'Describe matches and differences; do not infer unadjusted basis solely from a file name or numerical coincidence'},
        'admission': 'Preserve adjustment-basis uncertainty. This phase does not certify net outstanding shares or historical market cap.'
    }
    write_once(OUT / 'QUOTES_PLAN.json', plan)
    print(json.dumps({'plan_sha256': digest(OUT / 'QUOTES_PLAN.json'), 'source_files': len(FILES)}))


def read_csv(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def compare(rows, other, which, column):
    external = {r['Date'][:10]: r for r in other}
    results = []
    for row in rows:
        item = external.get(row['date'])
        if item is None:
            continue
        try:
            value = float(item[column])
        except (ValueError, TypeError, KeyError):
            continue
        if not math.isfinite(value) or value <= 0:
            continue
        actual = row[which]['close']
        results.append({'date': row['date'], 'source_close': actual, 'comparison_close': value, 'absolute_difference': abs(actual-value), 'source_to_comparison_ratio': actual/value})
    return {
        'overlapping_days': len(results),
        'within_half_cent_days': sum(r['absolute_difference'] <= 0.0051 for r in results),
        'maximum_absolute_difference': max((r['absolute_difference'] for r in results), default=None),
        'ratio_min': min((r['source_to_comparison_ratio'] for r in results), default=None),
        'ratio_max': max((r['source_to_comparison_ratio'] for r in results), default=None),
        'ratio_median': statistics.median(r['source_to_comparison_ratio'] for r in results) if results else None,
        'first_last': results[:2]+results[-2:],
        'largest_differences': sorted(results, key=lambda r:r['absolute_difference'], reverse=True)[:5],
    }


def extract():
    plan = json.loads((OUT / 'QUOTES_PLAN.json').read_text())
    assert plan['script_sha256'] == digest(Path(__file__))
    for value in plan['source_files'].values():
        assert digest(ROOT / value['path']) == value['sha256'], value['path']
    assert plan['source_files']['xls']['sha256'] == 'b8b74d3b8384afb7abd7d1e3886c2d85c8ef63cea2b0cd77bc8f351fde9ca374'
    assert plan['source_files']['wheel']['sha256'] == '6a33ee89877bd9abc1158129f6e94be74e2679636b8a205b43b85206c3f0bbdd'
    sys.path.insert(0, str(ROOT / FILES['wheel']))
    import xlrd
    book = xlrd.open_workbook(str(ROOT / FILES['xls']))
    assert book.sheet_names() == ['Share Data'] and book.datemode == 0
    sheet = book.sheet_by_index(0)
    assert sheet.ncols == 5
    assert sheet.row_values(1) == ['Share','Royal Dutch Shell A ADR (NYSE)','','Royal Dutch Shell B ADR (NYSE)','']
    assert sheet.row_values(2) == ['Currency','USD','','USD','']
    assert sheet.row_values(3) == ['Date','Close','Volume','Close','Volume']
    rows = []
    for n in range(4, sheet.nrows):
        values = sheet.row_values(n)
        assert all(isinstance(value, (int,float)) and math.isfinite(value) for value in values)
        date = xlrd.xldate_as_datetime(values[0], book.datemode)
        assert date.hour == date.minute == date.second == 0
        date = date.date().isoformat()
        assert plan['period'][0] <= date <= plan['period'][1]
        assert values[1] > 0 and values[3] > 0
        assert values[2] >= 0 and int(values[2]) == values[2]
        assert values[4] >= 0 and int(values[4]) == values[4]
        rows.append({'date': date, 'xls_row': n+1, 'class_A': {'close': values[1], 'volume': int(values[2])}, 'class_B': {'close': values[3], 'volume': int(values[4])}})
    dates = [r['date'] for r in rows]
    assert dates == sorted(set(dates))
    spy = read_csv(ROOT / FILES['spy'])
    spy_dates = {r['Date'][:10] for r in spy if plan['period'][0] <= r['Date'][:10] <= plan['period'][1]}
    signal_records = json.loads((ROOT / FILES['signal_dates']).read_text())
    requested = {r['asof'][:10] for r in signal_records}
    yh = read_csv(ROOT / FILES['yahoo_shel'])
    comparisons = {f'yahoo_SHEL_{column}_{which}': compare(rows, yh, which, column) for column in ['Close','Adj Close'] for which in ['class_A','class_B']}
    with zipfile.ZipFile(ROOT / FILES['stooq']) as archive:
        for which, member in zip(['class_A','class_B'], plan['overlap_check']['stooq_members']):
            data = list(csv.DictReader(io.StringIO(archive.read(member).decode('utf-8-sig'))))
            comparisons['stooq_'+which] = compare(rows, data, which, 'Close')
    result = {
        'schema':'shell168_quotes_v1', 'created_utc': now(), 'plan_sha256': digest(OUT/'QUOTES_PLAN.json'),
        'status':'QUOTES_TRANSCRIBED_ADJUSTMENT_BASIS_UNCONFIRMED',
        'sheet':'Share Data', 'header_rows':[sheet.row_values(i) for i in range(4)],
        'currency':'USD', 'security_basis':'NYSE ADR A/B as labelled by issuer-embedded source',
        'rows':rows, 'row_count':len(rows), 'first_date':dates[0], 'last_date':dates[-1],
        'missing_vs_spy':sorted(spy_dates-set(dates)), 'extra_vs_spy':sorted(set(dates)-spy_dates),
        'early_signal_asof_dates':len(requested), 'missing_early_signal_asof_dates':sorted(requested-set(dates)),
        'comparisons':comparisons,
        'limits':['No explicit raw/split/dividend adjustment declaration found in source workbook', 'Numerical overlap is source evidence, not historical publication-vintage proof', 'No net class share counts or market cap derived'],
        'market_cap_computed':False, 'strategy_trials_added':0, 'frozen_outputs_modified':False,
    }
    write_once(OUT/'QUOTES.json',result)
    print(json.dumps({k:result[k] for k in ['status','row_count','first_date','last_date','missing_vs_spy','extra_vs_spy','early_signal_asof_dates','missing_early_signal_asof_dates','comparisons']},indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=['register','extract'])
    args=parser.parse_args()
    register() if args.phase=='register' else extract()
