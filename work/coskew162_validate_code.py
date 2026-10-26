"""Bind isolated synthetic suites to the final coskew162 implementation bytes."""
import argparse
import hashlib
import importlib
import io
import json
from pathlib import Path
import unittest
import coskew162_independent as independent

ROOT=Path(__file__).resolve().parents[1]
SUITES=('kernel','runner','host','independent')
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def code_hashes():
    paths=independent.source_files(ROOT)+[ROOT/'work'/name for name in (
        'coskew162_host.py','coskew162_kernel.py','coskew162_runner.py',
        'test_coskew162_kernel.py','test_coskew162_runner.py','test_coskew162_host.py',
        'coskew162_validate_code.py')]
    return {str(p):digest(p) for p in sorted(set(paths))}
def save(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n')
def run_suite(name,directory):
    before=code_hashes()
    module=importlib.import_module('test_coskew162_'+name)
    if name=='independent':independent.prepare.independent_imports()
    stream=io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    log=directory/(name+'.log')
    with log.open('x') as f:f.write(stream.getvalue())
    after=code_hashes()
    passed=result.wasSuccessful() and before==after and result.testsRun>0
    receipt=dict(suite=name,passed=passed,synthetic_only=True,tests_run=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
        implementation_hashes=before,code_unchanged=before==after,log=dict(path=str(log),sha256=digest(log)))
    save(directory/(name+'.json'),receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k!='implementation_hashes'}))
    if not passed:
        print(stream.getvalue());raise SystemExit(1)
def aggregate(directory):
    pins=code_hashes();suites=[]
    for name in SUITES:
        path=directory/(name+'.json');item=json.loads(path.read_text())
        assert item['suite']==name and item['passed'] is True
        assert item['synthetic_only'] is True and item['code_unchanged'] is True
        assert item['implementation_hashes']==pins
        assert item['failures']==item['errors']==item['skipped']==0
        assert digest(item['log']['path'])==item['log']['sha256']
        suites.append(dict(suite=name,tests_run=item['tests_run'],receipt=dict(path=str(path),sha256=digest(path)),log=item['log']))
    receipt=dict(passed=True,synthetic_only=True,implementation_hashes=pins,suites=suites,
        tests_run=sum(x['tests_run'] for x in suites),real_features_orders_or_returns_computed=False)
    p=ROOT/'work/coskew162_registration_tests.json';save(p,receipt)
    print(json.dumps(dict(passed=True,tests_run=receipt['tests_run'],receipt_sha256=digest(p))))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('suite',choices=SUITES+('aggregate',));parser.add_argument('--directory',required=True,type=Path)
    args=parser.parse_args();args.directory.mkdir(parents=True,exist_ok=True)
    if args.suite=='aggregate':aggregate(args.directory)
    else:run_suite(args.suite,args.directory)
