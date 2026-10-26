"""Seal source-parser synthetic tests without opening any original reports."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, importlib, io, json, unittest
R=Path(__file__).resolve().parent.parent
mods=['after141_aapl_source_extract_tests','after141_amd_source_extract_tests','after141_profitability_source_extract_tests']
suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromModule(importlib.import_module(name)) for name in mods)
stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
paths=['work/'+name+'.py' for name in mods]+['work/after141_aapl_source_extract.py','work/after141_amd_source_extract.py','work/after141_profitability_source_extract.py','work/after141_source_test_bundle.py']
receipt=dict(passed=result.wasSuccessful(),synthetic_only=True,tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),created_utc=datetime.now(timezone.utc).isoformat(),source_hashes={p:hashlib.sha256((R/p).read_bytes()).hexdigest()for p in paths},log=stream.getvalue(),factor_rank_order_return_computations=0)
if not result.wasSuccessful():raise SystemExit(stream.getvalue())
with (R/'outputs/monthly_profitability_extension_sources_v1/TESTS.json').open('x')as f:json.dump(receipt,f,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in receipt.items()if k not in ('log','source_hashes')}))
