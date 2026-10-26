"""Write only the prospective endpoint155 specification and source references."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import monthly_endpoint155_runner as experiment

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT/'work/monthly_endpoint155_proposal.json'


def ref(path, expected=None):
    full = Path(path)
    full = full if full.is_absolute() else ROOT/full
    digest = hashlib.sha256(full.read_bytes()).hexdigest()
    experiment.require(expected is None or digest == expected, 'Changed source: '+str(path))
    return dict(path=str(full.relative_to(ROOT)), sha256=digest)


def build():
    state = json.loads((ROOT/'outputs/monthly_target70_research/CONTINUATION_STATUS.json').read_bytes())
    experiment.require(len(state['completed']) == 154 and not state['registered_queue']
        and sum(x['trials'] for x in state['completed']) == 975,
        'Final154 /975 with empty queue required')
    experiment.require(state['completed'][-1]['directory'] == 'monthly_financial_coverage154'
        and state['completed'][-1]['completion']['sha256'] == experiment.PREVIOUS_COMPLETION_SHA,
        'Exact finalized154 predecessor required')
    previous = ref(experiment.PREVIOUS_COMPLETION, experiment.PREVIOUS_COMPLETION_SHA)
    source_plan = ref(experiment.SOURCE_PLAN, experiment.SOURCE_PLAN_SHA)
    plan = json.loads((ROOT/source_plan['path']).read_bytes())
    design = ref(experiment.DESIGN_PATH, experiment.DESIGN_SHA)
    spec = json.loads((ROOT/design['path']).read_bytes())
    experiment.require(spec['candidate_contract']['arms'] == list(experiment.NAMES), 'Design arms changed')
    inputs = {key: ref(value['path'], value['sha256']) for key, value in plan['inputs'].items()
              if key != 'proposal'}
    inputs.update(design=design, calendar=ref('outputs/monthly_paired_target143/CALENDAR.json', experiment.CALENDAR_SHA))
    experiment.require(set(inputs) == set(experiment.INPUTS)-{'proposal'}, 'Source input schema')
    return dict(round=155, status='FINAL_PROSPECTIVE_SPECIFICATION_BEFORE_REGISTRATION',
        created_utc=datetime.now(timezone.utc).isoformat(),
        hypothesis=spec['hypothesis'], arms=list(experiment.NAMES), primary_challenger=experiment.NAMES[2],
        design=design, previous_completion=previous, source_plan=source_plan, source_inputs=inputs,
        measurement=experiment.measurement_contract(), selection_contract=experiment.selection_contract(),
        label_contract=spec['candidate_contract']['labels'],
        fit_contract=spec['candidate_contract']['training'],
        score_contract=spec['candidate_contract']['query_and_ranking'],
        price_and_calendar_contract=spec['price_and_calendar_contract'],
        fixed=dict(period=['2016-08-17', '2026-09-24'], monthly_signals=122, first_training_months=88,
            later_diagnostic_months=34, width=49, original_arrays=27,
            universe='Unchanged original monthly32 equities plus17 ETFs; no new membership or eligibility filters.',
            core='Unchanged dominant50%/52% long momentum skipping latest42 sessions.',
            holdings='Original EW3/two/cash fallback; fixed adjusted units within each month.',
            executions=['same_close', 'next_open_zero', 'next_open_cost'], cost_per_side=.0005,
            ordered_decisions=366, score_cells=17934, execution_paths=9, prefix_checks=123,
            observer_changes=False, downloads=0),
        interpretation=[
            'Loss-minus-matched-return isolates the target; either new arm minus incumbent also includes an added pooled-model correction.',
            'Binary Ridge output may lie outside zero to one; it is an uncalibrated ranking score, not a probability or trading threshold.',
            'Both arms preserve the older common_good restrictions including full-path, volatility and SPY validity; this is not a newly endpoint-only sample.',
            'Natural month endpoints are determined before prices. Missing endpoints stay missing, including September30 beyond the final September24 evaluation mark.',
            'The first August17,2016 partial query receives exact incumbent fallback in both arms.',
            'First88 selection is frozen before later34/full diagnostics. A diagnostic-only winner cannot be promoted or retuned.',
            'Historical repetition was not excluded in rounds32/116. The reused history is not pristine out-of-sample evidence.',
            'No financial-coverage admission or the five financial source batches is part of this experiment.',
        ], novelty=spec['novelty'],
        actual_new_labels=0, actual_new_model_fits=0, actual_new_orders=0, actual_new_returns=0,
        target='Training-selected, independently verified full-period same-close CAGR at least70%; no guaranteed attainment.')


if __name__ == '__main__':
    proposal = build()
    experiment.validate_proposal(proposal)
    with DEST.open('x') as stream:
        json.dump(proposal, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps(ref(DEST)))
