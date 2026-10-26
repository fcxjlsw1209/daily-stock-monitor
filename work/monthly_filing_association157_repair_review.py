"""Independent synthetic repair review and unchanged numerical audit wrappers.

Actual audit modes never import the repaired producer.  Only the separately
invoked synthetic prereview compares it with the frozen independent oracle.
The original registered numerical reviewers and their source scopes stay intact.
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import importlib
import json
import sys
import unittest
from io import StringIO

ROOT = Path(__file__).resolve().parents[1]
OUT = 'outputs/monthly_filing_association157'
SELF = 'work/monthly_filing_association157_repair_review.py'
TEST = 'work/test_monthly_filing_association157_repair_review.py'
PLAN_SHA = '43adc171b3ee82cd952544b75db63fffaf5b87bd25b96a03e709fd2ff22356db'
AMENDMENT = OUT + '/AMENDMENT_01.json'
PREREVIEW = OUT + '/REPAIR_PREREVIEW_01.json'
FAILURE = OUT + '/PREPARE_INTERFACE_FAILURE_01.json'
SUITES = {'runner': 'test_monthly_filing_association157_repaired_runner',
          'lifecycle': 'test_monthly_filing_association157_repair',
          'independent': 'test_monthly_filing_association157_repair_review'}
FORBIDDEN = ('monthly_filing_association157_runner',
             'monthly_filing_association157_repaired_runner',
             'monthly_filing_association157_lifecycle',
             'after156_association_measure')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def independent_imports():
    require(not any(name in sys.modules for name in FORBIDDEN),
            'Actual repair audit must not import producer modules')


def helper():
    return importlib.import_module('monthly_filing_association157_repair')


def local(root, value):
    root = Path(root).resolve()
    path = Path(value)
    path = (path if path.is_absolute() else root/path).resolve()
    require(path == root or root in path.parents, 'Repair evidence must remain in the local research root')
    return path


def verify_repair_evidence(root, amendment, amendment_sha):
    """Bind additional provenance without changing the original reviewer scope."""
    root = Path(root).resolve()
    require(amendment.get('original_plan_sha256') == PLAN_SHA,
            'Amendment refers to the fixed original registration')
    implementation = amendment['implementation_hashes']
    require(implementation == helper().implementation_hashes(root), 'Exact six-file repair implementation closure')
    artifacts = {}
    path = local(root, AMENDMENT)
    require(sha(path) == amendment_sha, 'Repair amendment changed')
    artifacts[str(path)] = amendment_sha
    for label in ('failure_record', 'independent_prereview'):
        ref = amendment[label]
        path = local(root, ref['path'])
        require(sha(path) == ref['sha256'], 'Repair evidence changed: ' + label)
        artifacts[str(path)] = ref['sha256']
    for name, expected in implementation.items():
        path = local(root, name)
        require(sha(path) == expected, 'Repair source changed: ' + name)
        artifacts[str(path)] = expected
    return artifacts


def augment(report, amendment, amendment_sha, artifacts):
    """Keep all old fields, especially source_hashes, and append repair evidence."""
    require(report.get('passed') is True and report.get('producer_modules_imported') is False,
            'Completed independent original audit required')
    keys = ('amendment_sha256', 'original_plan_sha256', 'repair_source_hashes', 'repair_review_sha256')
    require(not any(key in report for key in keys), 'Original audit unexpectedly contains repair fields')
    result = dict(report)
    merged = dict(report['artifact_hashes'])
    for path, digest in artifacts.items():
        require(path not in merged or merged[path] == digest, 'Conflicting repair artifact binding')
        merged[path] = digest
    result.update(amendment_sha256=amendment_sha, original_plan_sha256=PLAN_SHA,
        repair_source_hashes=dict(amendment['implementation_hashes']),
        repair_review_sha256=amendment['implementation_hashes'][SELF], artifact_hashes=merged)
    return result


def audit(stage, root, plan_sha, amendment_sha):
    require(stage in ('prepare', 'outcome'), 'Known independent audit stage required')
    require(plan_sha == PLAN_SHA, 'Exact original registered PLAN SHA required')
    independent_imports()
    root = Path(root).resolve()
    target = root/OUT/('INDEPENDENT_PREPARE.json' if stage == 'prepare' else 'INDEPENDENT_OUTCOME.json')
    require(not target.exists(), 'Independent audit already published')
    module = helper()
    amendment = module.verify_amendment(root, plan_sha, amendment_sha)
    before = verify_repair_evidence(root, amendment, amendment_sha)
    independent_imports()
    reviewer = importlib.import_module('monthly_filing_association157_' + stage + '_review')
    report = (reviewer.audit(root, root/OUT/'PLAN.json', plan_sha) if stage == 'prepare'
              else reviewer.audit(root, plan_sha))
    independent_imports()
    after = module.verify_amendment(root, plan_sha, amendment_sha)
    require(after == amendment and verify_repair_evidence(root, after, amendment_sha) == before,
            'Repair evidence changed during the independent audit')
    return augment(report, amendment, amendment_sha, before)


def prereview(root=ROOT, failure=FAILURE):
    root = Path(root).resolve()
    require(sha(root/OUT/'PLAN.json') == PLAN_SHA, 'Exact original PLAN required')
    module = helper()
    before = module.implementation_hashes(root)
    failure_path = local(root, failure)
    failure_sha = sha(failure_path)
    tests = {}
    for label, name in SUITES.items():
        source = local(root, 'work/' + name + '.py')
        stream = StringIO()
        suite = unittest.defaultTestLoader.loadTestsFromModule(importlib.import_module(name))
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
        require(result.wasSuccessful() and result.testsRun > 0 and not result.skipped,
                'Repair synthetic suite failed: ' + name + '\n' + stream.getvalue())
        tests[label] = dict(count=result.testsRun, passed=True, source_sha256=sha(source), log=stream.getvalue())
    require(before == module.implementation_hashes(root) and failure_sha == sha(failure_path)
            and sha(root/OUT/'PLAN.json') == PLAN_SHA, 'Repair prereview inputs changed')
    return dict(schema_version=1, round=157, stage='independent_synthetic_repair_review',
        passed=True, synthetic_only=True, reviewed_utc=datetime.now(timezone.utc).isoformat(),
        original_plan_sha256=PLAN_SHA, semantics_preserved=True, tests=tests,
        implementation_hashes=before,
        failure_record=dict(path=str(failure_path.relative_to(root)), sha256=failure_sha),
        producer_modules_imported_for_synthetic_comparison=True,
        actual_factors_calculated=0, actual_orders_calculated=0, actual_returns_calculated=0,
        actual_predictive_model_fits=0, actual_nuisance_estimates=0,
        prior_failed_activity='Recorded separately in failure_record; these zero counts describe only this synthetic review.',
        scope='Zero-budget dummy hand only; unchanged .01H, original episode, source/price windows, gates and numerical reviewers.')


def publish(path, report):
    with Path(path).open('x') as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', nargs='?', choices=('prepare', 'outcome'))
    parser.add_argument('--prereview', action='store_true')
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--plan-sha')
    parser.add_argument('--amendment-sha')
    parser.add_argument('--failure', default=FAILURE)
    args = parser.parse_args()
    if args.prereview:
        if args.stage or args.plan_sha or args.amendment_sha:
            parser.error('--prereview cannot be combined with actual audit arguments')
        report = prereview(args.root, args.failure)
        target = args.root/PREREVIEW
    else:
        if not args.stage or not args.plan_sha or not args.amendment_sha:
            parser.error('Actual audit requires stage, --plan-sha and --amendment-sha')
        report = audit(args.stage, args.root, args.plan_sha, args.amendment_sha)
        target = args.root/OUT/('INDEPENDENT_PREPARE.json' if args.stage == 'prepare' else 'INDEPENDENT_OUTCOME.json')
    publish(target, report)
    print(json.dumps(dict(passed=True, path=str(target), sha256=sha(target))))


if __name__ == '__main__':
    main()
