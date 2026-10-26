"""Build the separate preregistration specification; never read price values."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import monthly_filing_association157_lifecycle as lifecycle


def build_proposal(design, references, created_utc, measurement, selection_contract):
    c = lifecycle
    c.require(design.get('predecessor', {}).get('completed_round') == 156 and
              design['predecessor'].get('completed_trials') == 982, 'Exact design predecessor required')
    c.require([row['name'] for row in design.get('arms', [])] == list(c.NAMES), 'Four exact design arms required')
    c.require(design.get('clock', {}).get('id') == 'second_XNYS_session_strictly_after_filingDate_v1',
              'Reconstructed date convention required')
    c.require(design.get('fixed_transform', {}).get('score_dose') == .01, 'Fixed dose required')
    expected = dict(source_manifest=c.SOURCE_MANIFEST, source_grid_review=c.SOURCE_REVIEW,
                    source_plan=c.DESIGN, source_calendar=c.SOURCE_CALENDAR)
    c.require(set(references) == set(expected), 'Exact source references required')
    for key, path in expected.items():
        ref = references[key]
        c.require(set(ref) == {'path', 'sha256'} and ref['path'] == path and c.old.digest(ref['sha256']),
                  'Invalid source reference ' + key)
    c.require(references['source_plan']['sha256'] == c.DESIGN_SHA and
              references['source_calendar']['sha256'] == c.SOURCE_CALENDAR_SHA, 'Design/calendar changed')
    return dict(round=157, status='FINAL_PROSPECTIVE_SPECIFICATION_BEFORE_REGISTRATION',
        created_utc=created_utc, arms=list(c.NAMES), primary_challenger=c.NAMES[3], dose=.01,
        hypothesis=deepcopy(design['hypothesis']), design=dict(path=c.DESIGN, sha256=c.DESIGN_SHA),
        previous_completion=dict(path=c.PREVIOUS + 'COMPLETION.json', sha256=c.PREVIOUS_DONE_SHA),
        source_inputs=deepcopy(references), measurement=deepcopy(measurement),
        selection_contract=deepcopy(selection_contract),
        fixed=dict(period=['2016-08-17', '2026-09-24'], monthly_signals=122, width=49,
            first_training_months=88, early_months=53, recent_months=35, later_diagnostic_months=34,
            executions=list(c.MODES), execution_paths=12, monthly_orders=488, original_arrays=27,
            new_predictive_model_fits=0, nuisance_estimates='one252-return preformation beta per usable stock-slot',
            no_parameter_search=True, universe='Unchanged original candidates and original eligibility.',
            holdings='Unchanged skip42 long momentum; month-end EW3/two/cash; fixed units intramonth.'),
        interpretation=deepcopy(design['explicit_reconstruction_assumptions']) + [
            'All three new hands share the exact source-and-price availability mask.',
            'Missing prices never compress, forward-fill or shorten the fixed formation or beta window.',
            'The sole contrast challenger must pass first88 guards against the incumbent and both controls.',
            'Later34 and full122 diagnostics cannot override the previously frozen first88 selection.',
            'The history has been repeatedly explored; this is not pristine out-of-sample evidence.',
            'No observation account, live ledger or previously frozen artifact is modified.'],
        actual_new_factors=0, actual_new_orders=0, actual_new_returns=0,
        actual_new_predictive_model_fits=0, actual_new_nuisance_estimates=0,
        target='Training-selected, independently verified full-period same-close CAGR at least70%; attainment not guaranteed.')


def build(root=lifecycle.ROOT):
    c = lifecycle; root = Path(root).resolve(); reader = c.old.Reader(root); r = c.runner()
    state, _ = c.state_snapshot(root/c.STATE); c.predecessor(state)
    c.require(state.get('registered_queue') == [], 'Empty registration queue required')
    c.require(reader.hash(c.PREVIOUS + 'COMPLETION.json') == c.PREVIOUS_DONE_SHA, 'Predecessor changed')
    c.require(reader.hash(c.DESIGN) == c.DESIGN_SHA, 'Separate frozen design changed')
    refs = {key: dict(path=path, sha256=reader.hash(path)) for key, path in (
        ('source_manifest', c.SOURCE_MANIFEST), ('source_grid_review', c.SOURCE_REVIEW),
        ('source_plan', c.DESIGN), ('source_calendar', c.SOURCE_CALENDAR))}
    c.source_gate(reader, refs['source_grid_review'])
    proposal = build_proposal(reader.read(c.DESIGN), refs, datetime.now(timezone.utc).isoformat(),
                              r.measurement_contract(), r.selection_contract())
    r.validate_proposal(proposal); reader.verify(dict(reader.seen))
    return proposal


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=lifecycle.ROOT)
    args = parser.parse_args()
    result = build(args.root); path = args.root/lifecycle.PROPOSAL
    lifecycle.write(path, result)
    print(lifecycle.old.encoded(dict(path=lifecycle.PROPOSAL, sha256=lifecycle.sha(path))).decode())
