"""Prospective runner review: fixed contract and fabricated runner tests only.

This entry point never calls a real Runner phase or opens a price array. Source
admission has its separate independent package receipt; performance is pending.
"""
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
import argparse
import unittest
import monthly_filing_association157_runner as runner
import monthly_filing_association157_lifecycle as lifecycle
import test_monthly_filing_association157_runner as synthetic


def review(root):
    reader = lifecycle.old.Reader(root)
    proposal = reader.read(lifecycle.PROPOSAL)
    runner.validate_proposal(proposal)
    expected = dict(total_months=122, training_months=88, early_months=53,
        recent_months=35, later_months=34, walkforward=[[36,60],[60,76],[76,88]],
        execution_paths=[dict(execution='close',cost=0.,key='same_close'),
                         dict(execution='open',cost=0.,key='next_open_zero'),
                         dict(execution='open',cost=.0005,key='next_open_cost')],
        same_close_cagr='strict_greater_than_all_three_controls',
        next_open_cost_cagr='noninferior_incumbent_strict_greater_than_two_controls',
        strict_cagr_epsilon=1e-12, max_sharpe_degradation=.05,
        max_drawdown_degradation=.03, max_early_cagr_degradation=.025,
        max_recent_cagr_degradation=.025, new_predictive_model_fits=0,
        nuisance_estimates='preformation_beta_attempts_and_completions_counted_separately',
        intramonth_trades=False, later34_for_selection=False)
    lifecycle.require(proposal['selection_contract']==expected, 'Independent selection specification differs')
    measurement=proposal['measurement']
    for key,value in dict(beta_prices=[-504,-252],formation_prices=[-252,-42],aligned_prices=463,
        minimum_full19_bundles=2,minimum_E=38,minimum_N=80,original_candidate_axis=[122,49],
        unchanged_original_arrays=27,original_eligibility_unchanged=True,no_parameter_search=True,
        new_predictive_model_fits=0,etf_features=False,monthly_equal_weight_positions=3,
        intramonth_trades=False,exact_acceptance_certified=False).items():
        lifecycle.require(measurement.get(key)==value, 'Fixed measurement changed: '+key)
    log=StringIO()
    tests=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(synthetic))
    lifecycle.require(tests.wasSuccessful() and tests.testsRun>0 and not tests.skipped,
                      'Runner synthetic failure: '+log.getvalue())
    implementations=['work/'+name for name in ('after156_association_measure.py',
        'monthly_filing_association157_runner.py','monthly_filing_association157_lifecycle.py',
        'prepare_filing_association157_proposal.py')]
    sources=implementations+['work/monthly_filing_association157_register_review.py',
                            'work/test_monthly_filing_association157_runner.py']
    inputs=(lifecycle.PROPOSAL,lifecycle.DESIGN,lifecycle.SOURCE_MANIFEST,lifecycle.SOURCE_REVIEW)
    result=dict(round=157,stage='preregistration_runner_review',passed=True,synthetic_only=True,
        reviewed_utc=datetime.now(timezone.utc).isoformat(),proposal_sha256=reader.hash(lifecycle.PROPOSAL),
        actual_factors_calculated=0,actual_returns_calculated=0,actual_orders_calculated=0,
        actual_predictive_model_fits=0,actual_nuisance_estimates=0,tests=tests.testsRun,test_log=log.getvalue(),
        implementation_hashes={str(reader.path(p)):reader.hash(p) for p in implementations},
        source_hashes={str(reader.path(p)):reader.hash(p) for p in sources},
        input_hashes={str(reader.path(p)):reader.hash(p) for p in inputs},
        scope='Fixed selection and measurement specification plus fabricated-source/price runner regressions; actual preparation and accounting remain pending.')
    reader.verify(dict(reader.seen))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=lifecycle.ROOT)
    args=parser.parse_args()
    result=review(args.root)
    path=args.root/lifecycle.PREREVIEWS['runner']
    lifecycle.write(path,result)
    print(lifecycle.old.encoded(dict(passed=True,path=str(path),tests=result['tests'],sha256=lifecycle.sha(path))).decode())
