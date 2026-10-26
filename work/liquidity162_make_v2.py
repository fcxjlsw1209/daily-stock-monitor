"""Preserve frozen v1 and create a narrowly scoped dotted-month parser revision."""
from pathlib import Path
import json
import liquidity162_audit as a

a.bound()
source = Path(a.__file__).read_text()
edits = {
    "OUT = ROOT / 'outputs/monthly_liquidity162_sources'": "OUT = ROOT / 'outputs/monthly_liquidity162_sources/v2'\nSOURCE_OUT = ROOT / 'outputs/monthly_liquidity162_sources'",
    "CODE = ('work/liquidity162_audit.py',": "CODE = ('work/liquidity162_audit_v2.py', 'work/test_liquidity162_audit_v2.py',\n        'work/liquidity162_make_v2.py',\n        'outputs/monthly_liquidity162_sources/EXTRACTION_CODE.json',\n        'outputs/monthly_liquidity162_sources/IDENTITY_FAILURE.json',\n        'work/liquidity162_audit.py',",
    "pinned(OUT / 'SOURCE_PLAN.json', PLAN_SHA)": "pinned(SOURCE_OUT / 'SOURCE_PLAN.json', PLAN_SHA)",
    "norm(text).replace('Sept ', 'Sep ')": "re.sub(r'\\b([A-Za-z]{3,4})\\.', r'\\1', norm(text)).replace('Sept ', 'Sep ')",
    "date_pattern = r'([A-Za-z]+\\s+\\d{1,2},?\\s+\\d{4})'": "date_pattern = r'([A-Za-z]+\\.?\\s+\\d{1,2},?\\s+\\d{4})'",
    "names = ('test_liquidity162_audit',": "names = ('test_liquidity162_audit_v2', 'test_liquidity162_audit',",
    "'financial_amount_extraction_started': False,": "'financial_amount_extraction_started': False,\n        'revision': 'v2: support dotted abbreviated month literals; v1 failure and all original artifacts retained',\n        'supersedes_failed_v1_sha256': digest(SOURCE_OUT/'EXTRACTION_CODE.json'),",
}
for old, new in edits.items():
    if source.count(old) != 1:
        raise ValueError('Expected exactly one patch site: ' + old)
    source = source.replace(old, new)
out = a.ROOT/'work/liquidity162_audit_v2.py'
with out.open('x') as f:
    f.write(source)
(a.OUT/'v2').mkdir()
print(json.dumps({'revision_code':str(out),'sha256':a.digest(out),
                  'frozen_v1_retained':True,'changes':'dotted month recognition only; separate outputs'}))
