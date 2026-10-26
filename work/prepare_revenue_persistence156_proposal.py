"""Write a prospective revenue-persistence proposal after source-only admission.

No numerical financial factors, scores, portfolios or returns are calculated.
This command does not register an experiment or change continuation state.
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
from copy import deepcopy

import monthly_revenue_persistence156_lifecycle as lifecycle

ROOT = Path(__file__).resolve().parents[1]


def build_proposal(design, references, created_utc, measurement, selection_contract):
    """Pure construction with no source reads; useful for adversarial tests."""
    require = lifecycle.require
    mechanism = design.get('proposed_mechanism', {})
    require([a['name'] for a in mechanism.get('arms', [])] == list(lifecycle.NAMES) and
            mechanism.get('dose') == .01 and mechanism['arms'][3].get('primary') is True,
            'Frozen four-arm design required')
    require(design.get('parent_completed_round') == 155 and design.get('parent_total_trials') == 978,
            'Wrong frozen design predecessor')
    require(set(references) == {'source_manifest', 'source_grid_review', 'source_plan', 'source_calendar'},
            'Exact source-only input references required')
    expected = dict(source_manifest=lifecycle.SOURCE_MANIFEST, source_grid_review=lifecycle.SOURCE_REVIEW,
                    source_plan=lifecycle.SOURCE_PLAN, source_calendar=lifecycle.SOURCE_CALENDAR)
    for key, ref in references.items():
        require(set(ref) == {'path', 'sha256'} and ref['path'] == expected[key] and
                lifecycle.old.digest(ref['sha256']), 'Canonical source reference required: '+key)
    require(references['source_plan']['sha256'] == lifecycle.SOURCE_PLAN_SHA and
            references['source_calendar']['sha256'] == lifecycle.SOURCE_CALENDAR_SHA, 'Frozen source contract changed')
    return dict(round=156, status='FINAL_PROSPECTIVE_SPECIFICATION_BEFORE_REGISTRATION',
        created_utc=created_utc, arms=list(lifecycle.NAMES), primary_challenger=lifecycle.NAMES[3], dose=.01,
        hypothesis='Persistent annual reported revenue growth may improve long-momentum selection beyond latest-year and annualized-total growth on the identical reviewed four-level mask.',
        design=dict(path=lifecycle.DESIGN, sha256=lifecycle.DESIGN_SHA),
        previous_completion=dict(path=lifecycle.PREVIOUS_DONE, sha256=lifecycle.PREVIOUS_DONE_SHA),
        source_inputs=deepcopy(references), measurement=deepcopy(measurement),
        selection_contract=deepcopy(selection_contract),
        fixed=dict(period=['2016-08-17', '2026-09-24'], monthly_signals=122, width=49,
            first_training_months=88, early_months=53, recent_months=35, later_diagnostic_months=34,
            executions=list(lifecycle.MODES), execution_paths=12, monthly_orders=488,
            original_arrays=27, model_fits=0, no_parameter_search=True,
            universe='Unchanged original candidates and original eligibility; all missing-source rows retained.',
            holdings='Unchanged skip42 long momentum, monthly EW3/two/cash fallback; fixed units intramonth.'),
        interpretation=[
            'All three financial arms require the identical independently reviewed four-original chain mask.',
            'Unavailable features remain NaN with explicit reasons; zero score correction yields exact incumbent fallback.',
            'Reported nominal issuer revenue is not organic growth, constant-currency growth, or a homogeneous acquisition perimeter.',
            'ABBV uses an explicit reviewed within-issuer reported-dollar assumption; bare dollar signs never certify ISO USD.',
            'AAPL accounting chains remain unavailable until independently admitted; no source-year or issuer-subset search.',
            'Currently retrieved original-dated releases are not certified first-public bytes or exhaustive amendment histories.',
            'The predeclared first88 source floor establishes sparse-pilot feasibility, not statistical power.',
            'First88 selects the primary before later34 diagnostics; historical reuse is not pristine out-of-sample evidence.',
            'No sign, dose, parameter, threshold, source-subset, or model search is authorized.',
            'No observation accounts, live ledgers, or historical frozen artifacts change.'],
        actual_new_factors=0, actual_new_orders=0, actual_new_returns=0, actual_new_model_fits=0,
        target='Training-selected and independently verified full-period same-close CAGR at least70%; no guaranteed attainment.')


def build(root=ROOT):
    root = Path(root).resolve(); reader = lifecycle.old.Reader(root); r = lifecycle.runner()
    state, _ = lifecycle.state_snapshot(root/lifecycle.STATE)
    lifecycle.predecessor(state)
    lifecycle.require(state.get('registered_queue') == [], 'Empty registration queue required')
    lifecycle.require(reader.hash(lifecycle.PREVIOUS_DONE) == lifecycle.PREVIOUS_DONE_SHA,
                      'Finalized155 completion changed')
    lifecycle.require(reader.hash(lifecycle.DESIGN) == lifecycle.DESIGN_SHA, 'Frozen revenue design changed')
    design = reader.read(lifecycle.DESIGN)
    references = {key: dict(path=path, sha256=reader.hash(path)) for key, path in (
        ('source_manifest', lifecycle.SOURCE_MANIFEST), ('source_grid_review', lifecycle.SOURCE_REVIEW),
        ('source_plan', lifecycle.SOURCE_PLAN), ('source_calendar', lifecycle.SOURCE_CALENDAR))}
    lifecycle.source_gate(reader, references['source_grid_review'])
    proposal = build_proposal(design, references, datetime.now(timezone.utc).isoformat(),
                              r.measurement_contract(), r.selection_contract())
    r.validate_proposal(proposal)
    reader.verify(dict(reader.seen))
    return proposal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args(); root = args.root.resolve()
    proposal = build(root)
    path = root/lifecycle.PROPOSAL
    lifecycle.write(path, proposal)
    print(lifecycle.old.encoded(dict(path=lifecycle.PROPOSAL, sha256=lifecycle.sha(path))).decode())


if __name__ == '__main__':
    main()
