"""Write the independent static/synthetic prereview for round177's shape repair.

This script verifies receipts and source digests; it does not call strategy
preparation, feature calculation, training, or any portfolio replay.
"""
import json
from pathlib import Path

import monthly_breakpoint177_repaired_runner as r


EXPECTED = {
    'work/monthly_breakpoint177_repaired_runner.py': 'f22ed72211c3f7d8db87b586c6953f90d05952286bc6a4a2d25b5e3580c29230',
    'work/test_monthly_breakpoint177_repaired_runner.py': '7fa03b123bef391fd63f6219858bf65e9d335d1980f5e00fa656c2d4bdc487b8',
    'work/breakpoint177_weight_repair.py': '59fcf9e48004c3d3bf4635fb81a642f71a311783dccff25144e835242024c885',
    'work/test_breakpoint177_weight_repair.py': '04ee294766a9a2696fca4391ffd5cb21ad71d6eeac39bef013301479d7ee3b8f',
}
FAILURE_SHA = '8b67baa13115baa6aa8755c4f6a8e904f913f225f0fe50f29b67155339e4243e'
TESTS_SHA = 'c1cf7388ee19d19269380d712817d34dce1c259a4f5fa6d0b590d7349c86df3f'


def main():
    r.require(r.implementation_hashes() == EXPECTED, 'Reviewed repair source bytes changed')
    r.require(r.sha(r.FAILURE) == FAILURE_SHA and r.sha(r.TESTS) == TESTS_SHA,
              'Exact parent-delivered source receipts required')
    plan = r.check_original(r.PLAN_SHA)
    r.unprepared()
    failure, tests = r.read(r.FAILURE), r.read(r.TESTS)
    r.validate_failure(failure)
    r.validate_tests(tests)
    r.require({k: v['count'] for k, v in tests['suites'].items()} == {
        'test_monthly_breakpoint177_repaired_runner.py': 24,
        'test_breakpoint177_weight_repair.py': 13}, 'Unexpected synthetic test coverage')
    frozen = {}
    for base in ('host', 'kernel', 'independent'):
        for prefix in ('', 'test_'):
            path = r.ROOT / 'work' / (prefix + 'monthly_breakpoint177_' + base + '.py')
            r.require(plan['input_sha256'][str(path)] == r.sha(path), 'Original file changed')
            frozen[str(path.relative_to(r.ROOT))] = r.sha(path)
    doc = dict(
        passed=True, synthetic_only=True, semantics_preserved=True, independent_review=True,
        original_frozen_files_unchanged=True, original_plan_sha256=r.PLAN_SHA,
        implementation_hashes=EXPECTED, actual_returns_calculated=0,
        actual_strategy_phases_executed_by_reviewer=0, failure_record=r.reference(r.FAILURE),
        tests=r.reference(r.TESTS), reviewed_utc=r.now(),
        reviewer='ptr169_rebuild_map', reviewer_script=r.reference(Path(__file__)),
        original_plan_all_input_hashes_verified=True, original_six_files_sha256=frozen,
        synthetic_test_count=37, findings=[],
        verified_contracts=[
            'Weight representation alone is changed in a shallow verifier input copy.',
            'All cells of n, n-by-1, and n-by-width shapes must exactly match Boolean calm state: .50/.52.',
            'The normalized vector is separately copied; other arrays retain identity and original bytes are checked on success and failure.',
            'The unchanged frozen verifier still checks the complete factor mask, original core rank and every delta.',
            'Only prepare temporarily wraps verify_deltas; nested/foreign replacement is rejected and all function identities restored.',
            'Every phase verifies original PLAN and new amendment pins; saved delta proof carries explicit amendment and code references.',
            'Failure provenance explicitly identifies a retained tool-output excerpt, not an unavailable complete raw log.',
            'Pre-registration receipt checks require exact source hashes, all synthetic suites and chronological immutable records.',
            'Registration preserves original queue, trial counts and best candidate and retains the previous amendment reference.',
            'Completion checks original manifests, frozen selection binding, saved repair provenance and uses compare-and-swap central update.',
        ],
        limitations=[
            'This is static code and synthetic-test review; actual repaired preparation and portfolio outcomes remain unexecuted.',
            'No conclusion about investment performance follows from accepting this audit-interface repair.',
        ])
    r.validate_prereview(doc)
    r.require(r.implementation_hashes() == EXPECTED, 'Repair source changed during review')
    r.write_once(r.PREREVIEW, doc)
    print(json.dumps(dict(passed=True, review=r.reference(r.PREREVIEW), synthetic_tests=37)))


if __name__ == '__main__':
    main()
