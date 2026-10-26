"""Collect six separately launched synthetic suites, never launch tests or research.

After all code is stable, run --begin. For each name in the saved suite map,
launch exactly one independent ``./run_local.sh work/<test>.py`` process with
stdout and stderr redirected to ``TEST_<NAME>.log`` in the round157 directory.
Immediately pass its actual shell exit status to --record NAME --exit-code N.
Finally --collect validates all receipts and writes the new TESTS.json.

The offline wrapper forbids child processes. This collector therefore does not
spawn subprocesses or import test suites into a common interpreter. Exit codes
are caller observations; log content alone cannot establish a process exit.
All artifacts are written once; a failed collection is never silently repaired.
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import importlib
import json
import os
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/monthly_filing_association157'
SNAPSHOT = OUT/'TEST_SOURCE_SNAPSHOT.json'
FINAL = OUT/'TESTS.json'
SELF = Path(__file__).resolve()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1 << 20), b''):
            value.update(data)
    return value.hexdigest()


def reference(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=digest(path))


def write_new(path, document):
    encoded = json.dumps(document, indent=2, sort_keys=True, allow_nan=False)+'\n'
    with Path(path).open('x') as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def read(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    def bad(value):
        raise ValueError('Nonfinite JSON token: '+value)
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=bad)


def dependencies():
    """Import definitions only. Never call a producer phase or a test suite."""
    lifecycle = importlib.import_module('monthly_filing_association157_lifecycle')
    runner = importlib.import_module('monthly_filing_association157_runner')
    prepare = importlib.import_module('monthly_filing_association157_prepare_review')
    outcome = importlib.import_module('monthly_filing_association157_outcome_review')
    suites = dict(lifecycle.SUITES)
    require(set(suites) == {'factor', 'runner', 'prepare_review', 'outcome_review', 'lifecycle', 'proposal'},
            'Exactly six declared synthetic suites required')
    paths = set(map(Path, runner.source_files(ROOT)))
    paths.update(prepare.source_files(ROOT))
    paths.update(outcome.source_files(ROOT))
    paths.update(ROOT/'work'/name for name in suites.values())
    paths.update([SELF, Path(sys.executable), Path(lifecycle.__file__), Path(lifecycle.old.__file__),
                  ROOT/'work/prepare_filing_association157_proposal.py',
                  ROOT/'work/monthly_filing_association157_register_review.py'])
    paths.update(ROOT/name for name in ('AGENTS.md', 'LOCAL_SETUP.md', 'run_local.sh',
                                        'local_run.py', 'local.sandbox.sb'))
    hashes = {str(path.resolve()): digest(path) for path in sorted(paths)}
    runtime = dict(python=sys.version, numpy=runner.np.__version__, pandas=runner.pd.__version__,
                   executable=str(Path(sys.executable).resolve()))
    return lifecycle, suites, hashes, runtime


def guard_no_research():
    require(not any((OUT/name).exists() for name in (
        'PLAN.json', 'CACHE.json', 'DATA.npz', 'SCORES.npz', 'FEATURE_AUDIT.json',
        'PRETRAIN_ORDERS.json', 'TRAIN_TRIALS.json', 'FROZEN.json', 'RESULTS.json')),
        'Synthetic collection must precede registration and actual preparation')
    require(not FINAL.exists(), 'TESTS.json is write-once')


def log_path(name):
    return OUT/('TEST_'+name.upper()+'.log')


def receipt_path(name):
    return OUT/('TEST_'+name.upper()+'_RECEIPT.json')


def log_count(content):
    matches = re.findall(r'^Ran (\d+) tests? in [^\n]+$', content, re.MULTILINE)
    require(len(matches) == 1 and int(matches[0]) > 0, 'Exactly one positive Ran count required')
    require(content.rstrip().splitlines()[-1:] == ['OK'], 'Unqualified final unittest OK required')
    require(not re.search(r'^FAILED(?: |\(|$)|^ERROR:|^FAIL:', content, re.MULTILINE),
            'Failure diagnostics cannot precede success')
    return int(matches[0])


def checked_snapshot():
    guard_no_research()
    lifecycle, suites, hashes, runtime = dependencies()
    snapshot = read(SNAPSHOT)
    require(snapshot.get('round') == 157 and snapshot.get('synthetic_only') is True, 'Synthetic source snapshot')
    require(snapshot.get('suites') == suites and snapshot.get('runtime') == runtime, 'Suite/runtime changed')
    require(snapshot.get('source_hashes') == hashes, 'Source changed since pre-test snapshot')
    require(snapshot.get('collector') == reference(SELF), 'Collector changed')
    return lifecycle, snapshot


def begin():
    guard_no_research()
    _, suites, hashes, runtime = dependencies()
    require(not SNAPSHOT.exists(), 'Pre-test source snapshot is write-once')
    require(not any(log_path(name).exists() or receipt_path(name).exists() for name in suites),
            'Preexisting logs cannot establish a fresh test run')
    OUT.mkdir(parents=True, exist_ok=True)
    write_new(SNAPSHOT, dict(schema_version=1, round=157, synthetic_only=True,
        created_utc=now(), suites=suites, runtime=runtime, source_hashes=hashes, collector=reference(SELF),
        launch_rule='One fresh ./run_local.sh process per suite; merge stdout/stderr; record actual exit code.'))
    return dict(snapshot=reference(SNAPSHOT), suites={name: dict(
        command=['./run_local.sh', 'work/'+source], log=str(log_path(name))) for name, source in suites.items()})


def record(name, exit_code):
    _, snapshot = checked_snapshot()
    require(name in snapshot['suites'], 'Unknown suite')
    require(type(exit_code) is int, 'Actual shell exit code required')
    path = log_path(name)
    require(path.is_file() and path.stat().st_mtime_ns >= SNAPSHOT.stat().st_mtime_ns,
            'Fresh suite log after the source snapshot required')
    count = log_count(path.read_text()) if exit_code == 0 else None
    passed = exit_code == 0 and count is not None
    document = dict(schema_version=1, name=name, count=count, passed=passed, exit_code=exit_code,
        recorded_utc=now(), command=['./run_local.sh', 'work/'+snapshot['suites'][name]],
        launch_protocol='independent_wrapper_process_with_caller_observed_exit_code',
        source=reference(ROOT/'work'/snapshot['suites'][name]), log=reference(path),
        source_snapshot=reference(SNAPSHOT), source_hashes_unchanged=True)
    write_new(receipt_path(name), document)
    require(passed, 'Suite failed; receipt preserved and collection cannot pass')
    return dict(receipt=reference(receipt_path(name)), passed=True, count=count)


def collect():
    lifecycle, snapshot = checked_snapshot()
    rows = []
    for name, source in snapshot['suites'].items():
        path = receipt_path(name)
        row = read(path)
        require(row.get('name') == name and row.get('passed') is True and
                type(row.get('exit_code')) is int and row['exit_code'] == 0, 'Successful process receipt required')
        require(row.get('command') == ['./run_local.sh', 'work/'+source] and row.get('launch_protocol') ==
                'independent_wrapper_process_with_caller_observed_exit_code', 'Independent launch receipt required')
        require(row.get('source_snapshot') == reference(SNAPSHOT) and row.get('source_hashes_unchanged') is True,
                'Common pre-test source snapshot required')
        require(row.get('source') == reference(ROOT/'work'/source) and row.get('log') == reference(log_path(name)),
                'Changed source or suite log')
        require(type(row.get('count')) is int and row['count'] == log_count(log_path(name).read_text()),
                'Receipt count must match actual unittest output')
        rows.append(dict(row, receipt=reference(path)))
    _, _, after, runtime = dependencies()
    require(after == snapshot['source_hashes'] and runtime == snapshot['runtime'], 'Source changed during collection')
    document = dict(schema_version=1, round=157, passed=True, synthetic_only=True,
        test_count=sum(row['count'] for row in rows), suites=rows, created_at=now(),
        collector=reference(SELF), source_snapshot=reference(SNAPSHOT), source_hashes=after,
        source_hashes_unchanged_before_after=True, runtime=runtime,
        independent_processes_required=6,
        evidence_limit='Exit codes and standalone process launches are caller observations; log content alone is insufficient.',
        actual_factors_calculated=0, actual_returns_calculated=0, actual_orders_calculated=0,
        actual_predictive_model_fits=0, actual_nuisance_estimates=0)
    lifecycle.validate_tests(document, lifecycle.old.Reader(ROOT))
    write_new(FINAL, document)
    return dict(passed=True, test_count=document['test_count'], manifest=reference(FINAL))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--begin', action='store_true')
    mode.add_argument('--record', metavar='SUITE')
    mode.add_argument('--collect', action='store_true')
    parser.add_argument('--exit-code', type=int)
    args = parser.parse_args()
    require((args.record is not None) == (args.exit_code is not None), '--exit-code is only required with --record')
    result = begin() if args.begin else record(args.record, args.exit_code) if args.record is not None else collect()
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
