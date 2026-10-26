"""Archive round138 after independent verification, without account writes."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import argparse
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
P = ROOT/'outputs/monthly_sector_asymmetry138'
S = ROOT/'outputs/monthly_target70_research/CONTINUATION_STATUS.json'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def ref(path):
    return dict(path=str(Path(path).relative_to(ROOT)), sha256=sha(path))


def save(path, obj, exclusive=False):
    with Path(path).open('x' if exclusive else 'w') as f:
        json.dump(obj, f, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-sha', required=True)
    args = parser.parse_args()
    assert sha(P/'PLAN.json') == args.plan_sha
    plan = read(P/'PLAN.json')
    pre, audit, protected = [read(P/f) for f in
        ('INDEPENDENT_PREPARE.json', 'INDEPENDENT_OUTCOME.json', 'PROTECTED_AFTER.json')]
    assert pre['passed'] and audit['passed'] and protected['passed']
    assert pre['plan_sha256'] == audit['plan_sha256'] == args.plan_sha
    assert protected['checked'] == 21444 and not protected['changed'] and not protected['missing']
    assert protected['manifest_sha256'] == plan['inputs']['protected_manifest']['sha256']
    assert audit['execution_paths_verified'] == 9 and pre['orders_checked'] == 366
    for receipt in (pre, audit):
        for section in ('source_hashes', 'artifact_hashes'):
            for path, digest in receipt[section].items():
                assert sha(ROOT/path) == digest
    for path, digest in plan['hashes'].items():
        assert sha(ROOT/path) == digest
    frozen, result, state, orders, features = [read(path) for path in
        (P/'FROZEN.json', P/'RESULTS.json', S, P/'PRETRAIN_ORDERS.json',
         P/'FEATURE_AUDIT.json')]
    assert len(state['completed']) == 137 and sum(x.get('trials', 0) for x in state['completed']) == 910
    assert any(x.get('round') == 138 for x in state['registered_queue'])
    assert features['new_model_fits'] == 0
    primary = frozen['selection']['primary']
    metrics = result[primary]['same_close']['metrics']
    target = metrics['cagr'] >= .7
    assert primary == audit['selected_name'] and target == audit['primary_target_reached']
    base = orders['incumbent_local67']
    changes = {name: dict(ordered=sum(a['symbols'] != b['symbols'] for a, b in zip(rows, base)),
        sets=sum(set(a['symbols']) != set(b['symbols']) for a, b in zip(rows, base)))
        for name, rows in orders.items()}
    tests = read(P/'TESTS.json')
    assert tests['passed'] is True and len(tests['suites']) == 4
    for suite in tests['suites']:
        assert suite['passed'] is True and suite['count'] > 0
        assert sha(ROOT/suite['log']['path']) == suite['log']['sha256']
    now = datetime.now(timezone.utc).isoformat()
    conclusion = ('Sector-correlation-asymmetry candidate passed the fixed first88 gate.' if frozen['selection']['admitted']
                  else 'Sector-correlation-asymmetry candidate failed the fixed first88 gate; retain local67.')
    completion = dict(schema_version=1, round=138, experiment='monthly_sector_asymmetry138',
        status='completed_independently_verified', completed_utc=now, strategy_evaluations=3,
        new_challengers=1, new_admitted=int(frozen['selection']['admitted']), new_model_fits=0,
        primary=primary, selection=frozen['selection'], primary_metrics=metrics,
        primary_next_open_cost=result[primary]['next_open_cost']['metrics'], all_arm_results=result,
        holding_changes=changes, target_reached=target, finding=conclusion,
        feature_audit=ref(P/'FEATURE_AUDIT.json'),
        verification=dict(independent_execution_paths=9, independent_orders=366,
            original_arrays_unchanged=pre['original27_arrays_byte_identical'], protected_files=21444,
            prepare_report=ref(P/'INDEPENDENT_PREPARE.json'),
            max_independent_nav_absolute_error=audit['max_nav_absolute_difference'],
            synthetic_tests={x['name']: x['count'] for x in tests['suites']}),
        limitations=read(Path(plan['inputs']['proposal']['path']))['limitations'] + [
            'Same-close zero-cost primary is idealized; identical frozen orders next-open zero and5bp per side are separately reported.',
            'No frozen observer, plan, ledger or real order changed.'],
        artifacts={p.name: ref(p) for p in sorted(P.iterdir()) if p.is_file()})
    train = read(P/'TRAIN_TRIALS.json')
    completion['diagnostic_summary'] = {name: dict(same_close=result[name]['same_close']['metrics'],
        next_open_cost=result[name]['next_open_cost']['metrics'],
        training=next(row['training'] for row in train if row['name'] == name),
        later34=result[name]['same_close']['slices']['later34']) for name in result}
    save(P/'COMPLETION.json', completion, True)
    state['completed'].append(dict(completed_utc=now, completion=ref(P/'COMPLETION.json'),
        directory='monthly_sector_asymmetry138', independent_execution_paths=9,
        new_admitted=int(frozen['selection']['admitted']), primary=primary,
        primary_cagr=metrics['cagr'], primary_sharpe=metrics['sharpe'], protected_files=21444,
        target_reached=target, trials=3))
    state['registered_queue'] = [x for x in state['registered_queue'] if x.get('round') != 138]
    state.update(current='monthly_sector_asymmetry138', current_status='completed_independently_verified',
        updated_utc=now, latest_round_receipt=ref(P/'COMPLETION.json'),
        last_result_review=ref(P/'INDEPENDENT_OUTCOME.json'), last_protected_check=ref(P/'PROTECTED_AFTER.json'))
    if metrics['cagr'] > state['best_training_selected_candidate']['cagr'] and frozen['selection']['admitted']:
        state['best_training_selected_candidate'] = dict(directory='monthly_sector_asymmetry138',
            spec=primary, cagr=metrics['cagr'], sharpe=metrics['sharpe'], max_drawdown=metrics['max_drawdown'],
            next_open_5bp_cagr=result[primary]['next_open_cost']['metrics']['cagr'], observer_promotion=False)
    state['next_step'] = ('Keep the admitted best. Further research requires a distinct economic hypothesis and prospective registration; '
                         'do not reverse the sector-asymmetry sign or select its control using later34 results.')
    state['last_goal_turn_classification'] = ('progress: completed independent sector-asymmetry and nine-path validation; target '
        + ('reached in repeated historical research' if target else 'unmet; no external blocker') + '.')
    tmp = S.with_suffix('.json.tmp')
    save(tmp, state)
    tmp.replace(S)
    print(json.dumps(dict(completion=ref(P/'COMPLETION.json'), completed_rounds=len(state['completed']),
        strategy_evaluations=sum(x.get('trials', 0) for x in state['completed']), primary=primary,
        cagr=metrics['cagr'], target_reached=target)))


if __name__ == '__main__':
    main()
