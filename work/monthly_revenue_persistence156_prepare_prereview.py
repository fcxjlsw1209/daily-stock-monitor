"""Cross-review receipt for the independent prepare adapter; synthetic tests only.

Does not call the actual audit or load real source amounts, prices or returns.
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import importlib
import io
import json
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'work/monthly_revenue_persistence156_prepare_prereview.json'
PROPOSAL = ROOT / 'work/monthly_revenue_persistence156_proposal.json'
PROPOSAL_SHA = '18101719da500f23aff007a6249b2c224a99b28829b562f5db2a78484bc58d4b'
IMPLEMENTATIONS = ('after155_revenue_pilot_factor.py', 'monthly_revenue_persistence156_runner.py',
                   'monthly_revenue_persistence156_lifecycle.py', 'prepare_revenue_persistence156_proposal.py')
INPUTS = ('work/monthly_revenue_persistence156_proposal.json',
          'work/after155_revenue_experiment_path.json', 'work/after155_revenue_pilot_source_plan.json',
          'outputs/monthly_revenue_pilot_source156/MANIFEST.json')
FORBIDDEN_IMPORTS = ('monthly_revenue_persistence156_runner', 'after155_revenue_pilot_factor',
                     'after155_revenue_pilot_prepare', 'after155_multi_issuer_chain')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-review', action='store_true')
    args = parser.parse_args()
    if sha(PROPOSAL) != PROPOSAL_SHA:
        raise ValueError('Final prospective proposal changed')
    adapter = importlib.import_module('monthly_revenue_persistence156_prepare_review')
    tests = importlib.import_module('test_monthly_revenue_persistence156_prepare_review')
    sources = {str(path): sha(path) for path in adapter.source_files(ROOT)}
    implementations = {str(ROOT / 'work' / name): sha(ROOT / 'work' / name) for name in IMPLEMENTATIONS}
    inputs = {str(ROOT / path): sha(ROOT / path) for path in INPUTS}
    bound = dict(sources, **implementations, **inputs)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromModule(tests))
    print(stream.getvalue())
    if any(name in sys.modules for name in FORBIDDEN_IMPORTS):
        raise ValueError('Synthetic independent review imported a new producer')
    if any(sha(path) != expected for path, expected in bound.items()):
        raise ValueError('Reviewed bytes changed during synthetic tests')
    if not result.wasSuccessful():
        raise SystemExit(1)
    report = dict(schema_version=1, round=156, passed=True, synthetic_only=True,
        stage='preregistration_prepare_review', created_utc=datetime.now(timezone.utc).isoformat(),
        proposal_sha256=PROPOSAL_SHA, source_hashes=sources,
        implementation_hashes=implementations, input_hashes=inputs,
        actual_factors_calculated=0, actual_returns_calculated=0,
        actual_orders_calculated=0, actual_model_fits=0,
        producer_modules_imported=False, actual_prepare_audit_executed=False,
        bindings_unchanged_before_after=True,
        cross_reviewer_role='Reviewer authored the new producer runner but did not author the initial independent preparation adapter. This is a disclosed cross-review of that adapter, not an independent approval of the producer runner.',
        review_method='Inspected adapter/source binding and independent arithmetic and scoring dependencies; added adversarial synthetic regressions and ran the complete synthetic adapter suite through the unchanged offline wrapper.',
        resolved_findings=[
            'Require explicit independently reviewed original axis, original catalog, calendar closes, source cutoff availability and complete manifest closure before inheriting source admission.',
            'Bind the original META, source plan and calendar to canonical paths and fixed digests; require positive source coverage and driver source verification.',
            'Reject tied latest release clocks, duplicate original fiscal years, invalid common scales, or timestamp/decision-day mismatch.',
            'Require exact feature and score shapes and dtypes before numeric comparisons to prevent broadcasting from hiding a malformed matrix.',
            'Require canonical original policy/orders/prices, pinned episode policy, precise preparation tolerances, and no already-started selection or diagnostics.',
            'Bind source-manifest and source-review digests in the emitted FEATURE_AUDIT.',
        ],
        checks=dict(independent_integer_and_newton_arithmetic=True,
            shared_three_feature_mask=True, original_episode_once_per_arm=True,
            fixed_point01_bounded_adjustment=True, source_holds_and_etf_exclusion=True,
            all123_score_prefixes_tested=True, first88_early53_recent35_contract=True,
            raw64_loader_and_price_parity_path_reviewed=True,
            calendar_verification_inherited_from_exact_independent_actual_grid=True,
            complete_prepared_artifacts_and_consumed_hash_closure=True),
        test_execution=dict(command='./run_local.sh work/monthly_revenue_persistence156_prepare_prereview.py --write-review',
            tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
            output=stream.getvalue(), offline_wrapper_unchanged=True),
        reviewer_code=dict(path=str(Path(__file__).resolve()), sha256=sha(__file__)),
        remaining_requirements=[
            'A separately registered PLAN must bind this exact adapter source closure before actual preparation auditing.',
            'The actual independent preparation audit must run only after producer preparation and before any returns or first88 selection.',
            'Independent producer-runner and outcome review remain distinct requirements.',
        ])
    if args.write_review:
        with OUT.open('x') as handle:
            json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write('\n')
    print(json.dumps(dict(passed=True, stage=report['stage'], tests=result.testsRun,
                         output=str(OUT) if args.write_review else None)))


if __name__ == '__main__':
    main()
