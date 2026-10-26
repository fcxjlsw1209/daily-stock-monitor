"""Record verified round157 results and unadmitted source work; no replay."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import monthly_filing_association157_lifecycle as lifecycle

ROOT = Path(__file__).resolve().parents[1]
OUT = 'outputs/monthly_filing_association157'
CHECKPOINT = 'migration/after157_research_checkpoint.json'
PLAN_SHA = '43adc171b3ee82cd952544b75db63fffaf5b87bd25b96a03e709fd2ff22356db'
AMENDMENT_SHA = '1e17a0eaccd4669c9706c128e817d5605334efba42e1539d483dc385e43ed73e'
PINS = {
    'work/after157_annual_pair_source_feasibility_plan.json': '5750f95e17d8891f7781962efed7bd841408385cce1abfeee1a345827d3fb57d',
    'work/after157_annual_pair_source_feasibility.json': '4fdeaad7de09625991a2b3603d33799bf1c62dd0b99f560b900ac8928abc9242',
    'work/after157_annual_original_body_inventory.json': '43c83311b2328e535303faed748ab7e385f962e61c7fb340216cea4d29056f44',
    'work/after157_annual_pair_independent_review.json': 'b5cc384ec45660c512f974e4849dfe1900363e19aba5bd313e2f63a442c5835f',
}


def read(name):
    return json.loads((ROOT / name).read_text())


def reference(name):
    return dict(path=name, sha256=hashlib.sha256((ROOT / name).read_bytes()).hexdigest())


def main():
    for name, expected in PINS.items():
        lifecycle.require(reference(name)['sha256'] == expected, 'Changed source record: ' + name)
    done = read(OUT + '/COMPLETION.json')
    repair = read(OUT + '/REPAIR_COMPLETION_01.json')
    audit = read(OUT + '/INDEPENDENT_OUTCOME.json')
    amendment = read(OUT + '/AMENDMENT_01.json')
    lifecycle.require(done['status'] == 'completed_independently_verified'
        and done['round'] == 157 and done['independent_execution_paths'] == 12
        and done['protected_files'] == 21444 and done['trials'] == 4
        and done['target_reached'] is False and audit['passed'] is True,
        'Formal round157 completion required')
    lifecycle.require(repair['original_completion'] == reference(OUT + '/COMPLETION.json'),
                      'Repair completion binding changed')
    lifecycle.require(done['independent_outcome_sha256'] == reference(OUT + '/INDEPENDENT_OUTCOME.json')['sha256']
        and repair['independent_outcome'] == reference(OUT + '/INDEPENDENT_OUTCOME.json')
        and done['plan_sha256'] == repair['preserved_original_plan_sha256'] == audit['plan_sha256'] == PLAN_SHA
        and reference(OUT + '/PLAN.json')['sha256'] == PLAN_SHA
        and repair['amendment'] == reference(OUT + '/AMENDMENT_01.json')
        and audit['amendment_sha256'] == reference(OUT + '/AMENDMENT_01.json')['sha256'] == AMENDMENT_SHA
        and amendment['original_plan_sha256'] == PLAN_SHA
        and amendment['same_experiment'] is True and amendment['additional_strategy_trials'] == 0,
        'Completion, repair, outcome and amendment must share exact bindings')
    lifecycle.require(done['primary'] == audit['selected_name'] == 'incumbent_local67'
        and done['selection']['admitted'] is False and audit['primary_target_reached'] is False
        and abs(done['primary_cagr'] - 0.6766617372041686) < 1e-12
        and abs(done['primary_cagr'] - audit['primary_cagr']) < 1e-12,
        'Incumbent-retained conclusion must match verified selection and metrics')
    source = read('work/after157_annual_pair_source_feasibility.json')
    bodies = read('work/after157_annual_original_body_inventory.json')
    lifecycle.require(source['summary']['all122']['tier3']['slots'] == 0
        and bodies['counts']['normal_consecutive_pairs_admitted'] == 0,
        'Source status changed; reassess rather than overwrite')
    next_step = (
        '157 completed rounds / 986 trials. Round157 filing-date association did not '
        'improve the incumbent; full-period selected CAGR remains 67.66617372041686%, '
        'below the 70% target. Twelve execution paths were independently verified. '
        'The unregistered annual filing-lag-change idea has source-only inventories: '
        '2379/3904 stock slots pass conditional metadata screening, but zero ordinary '
        'consecutive annual pairs are admitted. Of 324 exact accessions in 266 pairs, '
        'the declared local catalogs contain zero exact-source body candidates and '
        'one explicitly associated PDF with unresolved publisher chain. Do not run '
        'lag features or silently relax those gates. Next work needs genuine original '
        'annual-period evidence or a separately reviewed distinct hypothesis. Preserve '
        'the SEC source stop for 2026-10-20; do not retry or bypass it. All future '
        'experiments require prospective registration; no frozen observer changes.'
    )
    with lifecycle.state_lock(ROOT):
        state, before = lifecycle.state_snapshot(ROOT / lifecycle.STATE)
        lifecycle.require(len(state['completed']) == 157
            and sum(x['trials'] for x in state['completed']) == 986
            and state['registered_queue'] == []
            and state['latest_round_receipt'] == reference(OUT + '/COMPLETION.json')
            and state['latest_execution_repair_completion'] == reference(OUT + '/REPAIR_COMPLETION_01.json'),
            'Unexpected central research state')
        lifecycle.require(abs(state['best_training_selected_candidate']['cagr'] - done['primary_cagr']) < 1e-12,
                          'Central incumbent metrics changed')
        path = ROOT / CHECKPOINT
        checkpoint = dict(
            created_utc=datetime.now(timezone.utc).isoformat(),
            prior_goal_turn_classification='progress',
            current_goal_turn_classification='progress',
            round_completion=reference(OUT + '/COMPLETION.json'),
            repair_completion=reference(OUT + '/REPAIR_COMPLETION_01.json'),
            independent_outcome=reference(OUT + '/INDEPENDENT_OUTCOME.json'),
            source_records=[reference(name) for name in PINS],
            completed_rounds=157, strategy_trials=986,
            target_reached=False, incumbent=state['best_training_selected_candidate'],
            source_only_followup=dict(strategy_trials=0,
                first88_conditional_stock_slots=source['summary']['first88']['tier2']['slots'],
                full_conditional_stock_slots=source['summary']['all122']['tier2']['slots'],
                original_body_counts=bodies['counts'],
                normal_consecutive_pairs_admitted=0),
            next_step=next_step,
            evidence_limits=[
                'Historical samples have been repeatedly explored; neither segment is pristine out of sample.',
                'Filing-date association uses stated reconstruction and issuer-continuity assumptions, not exact news timestamps.',
                'Body absence is limited to the declared local source catalogs, not all possible public sources.',
            ])
        if path.exists():
            prior = read(CHECKPOINT)
            lifecycle.require({k:v for k,v in prior.items() if k != 'created_utc'} ==
                {k:v for k,v in checkpoint.items() if k != 'created_utc'}, 'Conflicting checkpoint')
            checkpoint = prior
        else:
            lifecycle.write(path, checkpoint)
        if state.get('after157_research_checkpoint') == reference(CHECKPOINT):
            print(json.dumps(dict(checkpoint=reference(CHECKPOINT), target_reached=False, already_recorded=True)))
            return
        state.update(after157_research_checkpoint=reference(CHECKPOINT), next_step=next_step,
                     last_goal_turn_classification='progress', updated_utc=datetime.now(timezone.utc).isoformat())
        lifecycle.update_state(ROOT / lifecycle.STATE, before, state)
    print(json.dumps(dict(checkpoint=reference(CHECKPOINT), target_reached=False)))


if __name__ == '__main__':
    main()
