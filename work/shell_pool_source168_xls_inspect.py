"""Read-only structural inspection of the immutable public legacy XLS export."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'outputs/monthly_shell_pool_source168/external_followup/euroland_browser_quotes.xls'
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == 'b8b74d3b8384afb7abd7d1e3886c2d85c8ef63cea2b0cd77bc8f351fde9ca374'
WHEEL = ROOT / 'work/shell168_xls_reader_wheels/xlrd-2.0.1-py2.py3-none-any.whl'
sys.path.insert(0, str(WHEEL))
import xlrd

book = xlrd.open_workbook(str(SOURCE))
print(json.dumps({'xlrd_version': xlrd.__version__, 'wheel_sha256': hashlib.sha256(WHEEL.read_bytes()).hexdigest(), 'date_mode': book.datemode}))
for sheet in book.sheets():
    print(json.dumps({'sheet': sheet.name, 'rows': sheet.nrows, 'cols': sheet.ncols, 'first_rows': [sheet.row_values(i) for i in range(min(8, sheet.nrows))], 'last_rows': [sheet.row_values(i) for i in range(max(0,sheet.nrows-2),sheet.nrows)]}, ensure_ascii=False))
