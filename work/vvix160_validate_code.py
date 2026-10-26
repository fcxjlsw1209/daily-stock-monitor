"""Run isolated synthetic suites and bind their receipts to the exact code bytes."""
import argparse
import hashlib
import importlib
import io
import json
from pathlib import Path
import sys
import unittest
import vvix160_independent as independent

ROOT = Path(__file__).resolve().parents[1]
SUITES = ('kernel', 'runner', 'host', 'independent')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_hashes():
    paths = independent.source_files(ROOT) + [
        ROOT/'work'/name for name in (
            'vvix160_host.py', 'vvix160_kernel.py', 'vvix160_runner.py',
            'test_vvix160_kernel.py', 'test_vvix160_runner.py',
            'test_vvix160_host.py', 'vvix160_validate_code.py')]
    return {str(p): digest(p) for p in sorted(set(paths))}


def save(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write('\n')


def run_suite(name, directory):
    before = code_hashes()
    module = importlib.import_module('test_vvix160_' + name)
    if name == 'independent':
        independent.prepare.independent_imports()
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromModule(module))
    log = directory/(name+'.log')
    with log.open('x') as handle:
        handle.write(stream.getvalue())
    after = code_hashes()
    passed = result.wasSuccessful() and before == after and result.testsRun > 0
    receipt = dict(suite=name, passed=passed, synthetic_only=True,
                   tests_run=result.testsRun, failures=len(result.failures),
                   errors=len(result.errors), skipped=len(result.skipped),
                   implementation_hashes=before, code_unchanged=before == after,
                   log=dict(path=str(log), sha256=digest(log)))
    save(directory/(name+'.json'), receipt)
    print(json.dumps({key: value for key, value in receipt.items()
                      if key != 'implementation_hashes'}))
    if not passed:
        print(stream.getvalue())
        raise SystemExit(1)


def aggregate(directory):
    pins = code_hashes()
    suites = []
    for name in SUITES:
        path = directory/(name+'.json')
        item = json.loads(path.read_text())
        assert item['suite'] == name and item['passed'] is True
        assert item['synthetic_only'] is True and item['code_unchanged'] is True
        assert item['implementation_hashes'] == pins
        assert item['failures'] == item['errors'] == item['skipped'] == 0
        assert digest(item['log']['path']) == item['log']['sha256']
        suites.append(dict(suite=name, tests_run=item['tests_run'],
                           receipt=dict(path=str(path), sha256=digest(path)),
                           log=item['log']))
    receipt = dict(passed=True, synthetic_only=True,
                   implementation_hashes=pins, suites=suites,
                   tests_run=sum(item['tests_run'] for item in suites),
                   real_features_orders_or_returns_computed=False)
    save(ROOT/'work/vvix160_registration_tests.json', receipt)
    print(json.dumps(dict(passed=True, tests_run=receipt['tests_run'],
                          receipt_sha256=digest(ROOT/'work/vvix160_registration_tests.json'))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('suite', choices=SUITES+('aggregate',))
    parser.add_argument('--directory', required=True, type=Path)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    if args.suite == 'aggregate':
        aggregate(args.directory)
    else:
        run_suite(args.suite, args.directory)
