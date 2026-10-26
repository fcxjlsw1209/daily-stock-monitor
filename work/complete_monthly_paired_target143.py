"""Close round143 from verified metadata and bytes; never computes market returns.

Root remains the sole continuation-state writer. A completed report is published
exclusively, before the checked atomic state update. Resume only by its exact SHA.
"""
from pathlib import Path
from datetime import datetime, timezone
from copy import deepcopy
import argparse
import hashlib
import json
import math
import os
import re
import stat
import tempfile

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = 'monthly_paired_target143'
STATE = 'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
NAMES = ('incumbent_local67', 'paired_close_target', 'paired_open_target')
MODES = ('same_close', 'next_open_zero', 'next_open_cost')
PROPOSAL_SHA = 'd1b93c8ae18e11998e011a3b91a76af761d79be0ada8f3435258b3365cfe5256'
PROTECTED_SHA = '89e82ac5cbec3b998f1d8aab93f1f8fd7c5e19522321b2dd1da2f88dfff922cb'
PROTECTED_COUNT = 21444
CACHE = ('DATA.npz', 'SCORES.npz', 'META.json', 'PRETRAIN_ORDERS.json', 'PRICE_PARITY.json',
         'LABELS.npz', 'LABEL_AUDIT.json', 'MODEL_AUDIT.json', 'IDENTITY_AUDIT.json', 'CALENDAR.json', 'RANK_AUDIT.json')
FROZEN = ('PLAN.json', 'CACHE.json', 'PRETRAIN_ORDERS.json', 'TRAIN_TRIALS.json',
          'TRAIN_OPEN_TRIALS.json', 'WALKFORWARD.json', 'INDEPENDENT_PREPARE.json')
PREPARE_CHECKS = ('original27_arrays_byte_identical', 'training_identity_and_plain10_verified',
    'raw_prices_independently_verified', 'calendar_overlay_independently_verified',
    'paired_labels_independently_reconstructed', 'shared_maturity_and_cohort_verified',
    'annual_forecasts_independently_reconstructed', 'all_three_score_matrices_and_ordered_choices_independently_verified',
    'all_signal_prefixes_verified', 'whole_month_incumbent_fallback_verified', 'one_episode_and_top2_lock_verified')
OUTCOME_CHECKS = ('daily_nav_trades_holdings_metrics_slices', 'first88_and_three_fold_selection_verified',
    'annual_partial_flags_verified', 'monthly_difference_and_attribution_verified',
    'first88_causal_replay_verified', 'model_fit_counts_verified', 'all_changed_months_reported',
    'diagnostic_start_marker_verified', 'baseline_local67_full_orders_unchanged')
SUITE_SOURCES = {'labels':'after142_paired_target_labels_tests.py',
    'model':'test_monthly_paired_target143_model.py', 'runner':'test_monthly_paired_target143_runner.py',
    'prepare_review':'test_monthly_paired_target143_prepare_review.py',
    'outcome_review':'test_monthly_paired_target143_outcome_review.py',
    'completion':'test_complete_monthly_paired_target143.py'}
SUITE_NAMES = tuple(SUITE_SOURCES)



def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(part)
    return value.hexdigest()


def decode(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    def bad(value):
        raise ValueError('Nonfinite JSON token: ' + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=bad)


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)+'\n').encode()


class Reader:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.seen = {}

    def path(self, value):
        path = Path(value)
        require(not any(x in str(path) for x in ('CloudStorage', 'OneDrive')), 'Cloud path forbidden')
        absolute = path if path.is_absolute() else self.root/path
        resolved = absolute.resolve()
        require(not any(x in str(resolved) for x in ('CloudStorage', 'OneDrive')), 'Cloud target forbidden')
        require(path.is_absolute() or resolved.is_relative_to(self.root), 'Relative path escapes root')
        return resolved

    def note(self, path, value):
        key = str(path)
        require(key not in self.seen or self.seen[key] == value, 'Input changed during closure: '+key)
        self.seen[key] = value
        return value

    def hash(self, value):
        path = self.path(value)
        return self.note(path, sha(path))

    def read(self, value):
        path = self.path(value)
        raw = path.read_bytes()
        self.note(path, hashlib.sha256(raw).hexdigest())
        return decode(raw)

    def verify(self, mapping, base=None):
        require(isinstance(mapping, dict) and bool(mapping), 'Nonempty hash map required')
        result = {}
        for value, expected in mapping.items():
            require(digest(expected), 'Invalid SHA256')
            if base is not None:
                require(Path(value).name == value and value not in ('', '.', '..'), 'Artifact basename required')
                value = Path(base)/value
            key = str(self.path(value))
            require(key not in result or result[key] == expected, 'Conflicting canonical hash')
            require(self.hash(key) == expected, 'Hash mismatch: '+key)
            result[key] = expected
        return result


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def selection(rows, open_rows):
    def check(group):
        require(isinstance(group, list) and len(group) == 3 and
                all(isinstance(row, dict) and row.get('name') == name for row, name in zip(group, NAMES)),
                'Three ordered training arms required')
    check(rows); check(open_rows)
    fields = (('training', 'cagr'), ('training', 'sharpe'), ('training', 'max_drawdown'),
              ('early', 'cagr'), ('recent', 'cagr'))
    def compare(candidate, reference, strict):
        values = [row[a].get(b) if isinstance(row.get(a), dict) else None for row in (candidate,reference) for a,b in fields]
        if not all(number(x) for x in values):return False
        candidate,reference = values[:5],values[5:]
        first = candidate[0] > reference[0]+1e-12 if strict else candidate[0] >= reference[0]-1e-12
        return bool(first and all(a >= b-t for a,b,t in zip(candidate[1:],reference[1:],(.05,.03,.025,.025))))
    guards = {name:compare(rows[2],rows[i],True) for i,name in enumerate(NAMES[:2])}
    open_guards = {name:compare(open_rows[2],open_rows[i],i==1) for i,name in enumerate(NAMES[:2])}
    admitted = all(guards.values()) and all(open_guards.values())
    return dict(primary=NAMES[2] if admitted else NAMES[0], admitted=admitted, guards=admitted,
                guards_against=guards, next_open_guards_against=open_guards, mechanism_confirmed=all(open_guards.values()))


def model_fits(audit):
    require(isinstance(audit,dict) and isinstance(audit.get('annual'),list), 'Annual model audit required')
    count = audit.get('new_pooled_model_fits')
    require(type(count) is int and count >= 0, 'Nonnegative pooled model fit count required')
    years = set(); derived=0
    for row in audit['annual']:
        require(isinstance(row,dict) and type(row.get('year')) is int and row['year'] not in years,
                'Unique integer model audit year required')
        years.add(row['year'])
        require(row.get('status') in ('fit','cold_start','no_queries'), 'Unknown annual model status')
        if row['status']=='fit':derived+=2
    require(count==derived, 'Pooled model fits must equal two times fitted annual cohorts')
    return count


def exact_count(obj, key, count):
    require(type(obj.get(key)) is int and obj[key] == count, 'Wrong count: '+key)


def subset(mapping, allowed, label):
    require(all(allowed.get(key) == value for key, value in mapping.items()), 'Unbound '+label)


def utc(value):
    require(isinstance(value, str) and value.endswith(('Z', '+00:00')), 'UTC timestamp required')
    result = datetime.fromisoformat(value[:-1]+'+00:00' if value.endswith('Z') else value)
    require(result.utcoffset().total_seconds() == 0, 'UTC timestamp required')
    return result


def protected_scan(reader, plan, proof):
    ref = plan['inputs']['protected_manifest']
    require(ref['sha256'] == PROTECTED_SHA and reader.hash(ref['path']) == PROTECTED_SHA,
            'Original protected manifest binding')
    require(proof.get('passed') is True and not proof.get('changed') and not proof.get('missing')
            and proof.get('read_or_identity_errors') == []
            and 'changed' in proof and 'missing' in proof, 'Protected receipt incomplete')
    exact_count(proof, 'checked', PROTECTED_COUNT)
    exact_count(proof, 'verified', PROTECTED_COUNT)
    require(proof.get('manifest_sha256') == PROTECTED_SHA, 'Protected receipt manifest mismatch')
    manifest = reader.read(ref['path'])
    require(isinstance(manifest, dict) and len(manifest) == PROTECTED_COUNT, 'Full protected inventory required')
    resolved = set()
    for name, expected in manifest.items():
        raw = Path(name)
        path = reader.path(name)
        require(raw.is_absolute() and raw == path and path.is_relative_to(reader.root),
                'Protected member must be an exact local path without symlinks')
        require(path not in resolved and stat.S_ISREG(path.stat().st_mode), 'Unique regular protected member required')
        resolved.add(path)
        require(digest(expected) and reader.hash(path) == expected, 'Protected member changed: '+name)
    return len(resolved)


def validate(root, plan_sha):
    reader = Reader(root)
    out = reader.root/'outputs'/EXPERIMENT
    require(digest(plan_sha) and reader.hash(out/'PLAN.json') == plan_sha, 'Exact PLAN SHA required')
    plan = reader.read(out/'PLAN.json')
    require(plan.get('round') == 143 and plan.get('status') == 'REGISTERED'
            and plan.get('arms') == list(NAMES) and plan.get('primary_challenger') == NAMES[2]
            and reader.path(plan['output_directory']) == out, 'Registered143 three-arm PLAN required')
    pins = reader.verify(plan['hashes'])
    require(str(reader.root/STATE) not in pins, 'Mutable continuation state cannot be an immutable PLAN pin')
    for filename in ('complete_monthly_paired_target143.py', 'test_complete_monthly_paired_target143.py'):
        path = reader.root/'work'/filename
        require(pins.get(str(path)) == reader.hash(path), 'Closer source/test must be registered')
    require(reader.hash(reader.root/'work/complete_monthly_paired_target143.py') == sha(__file__),
            'Running closer differs from bound source')
    for name in ('raw_source_hashes', 'independent_prepare_source_hashes',
                 'independent_outcome_source_hashes'):
        subset(reader.verify(plan[name]), pins, name)
    require(len(plan['raw_source_hashes']) == 64, 'Original raw64 scope')
    for name in ('actual_model_fits_before_registration', 'actual_feature_calculations_before_registration',
                 'actual_returns_before_registration'):
        exact_count(plan, name, 0)
    require(set(plan['inputs']) == {'proposal', 'meta', 'policy', 'original_orders', 'prices', 'train_inputs',
                                   'training_meta', 'protected_manifest'}, 'Exact named inputs')
    for ref in plan['inputs'].values():
        key = str(reader.path(ref['path']))
        require(reader.hash(key) == ref['sha256'] == pins.get(key), 'Unbound named input')
    require(plan['inputs']['proposal']['sha256'] == PROPOSAL_SHA, 'Frozen proposal binding')
    proposal = reader.read(plan['inputs']['proposal']['path'])
    cache = reader.read(out/'CACHE.json')
    require(set(cache) == set(CACHE), 'All paired-target cache artifacts required')
    cached = reader.verify(cache, out)
    prep = reader.read(out/'INDEPENDENT_PREPARE.json')
    outcome = reader.read(out/'INDEPENDENT_OUTCOME.json')
    for report, stage, checks in ((prep, 'prepare', PREPARE_CHECKS), (outcome, 'evaluate', OUTCOME_CHECKS)):
        require(report.get('passed') is True and report.get('stage') == stage
                and all(report.get(key) is True for key in checks), 'Incomplete independent '+stage+' proof')
        require(report.get('producer_modules_imported') is False, 'Independent producer isolation')
        for key, file in (('plan_sha256', 'PLAN.json'), ('cache_sha256', 'CACHE.json'),
                          ('orders_sha256', 'PRETRAIN_ORDERS.json')):
            require(report.get(key) == reader.hash(out/file), 'Independent receipt binding: '+key)
    require(prep.get('returns_or_nav_computed') is False,
            'Prepare scope mismatch')
    fits = model_fits(reader.read(out/'MODEL_AUDIT.json'))
    for key, count in (('orders_checked',366),('candidate_cells_checked',5978),('months_checked',122),
                       ('training_rows_checked',8820),('forecast_arrays_verified',6),('prefixes_checked',123),
                       ('new_model_fits',fits)):
        exact_count(prep,key,count)
    for key, count in (('execution_paths_verified',9),('total_monthly_orders_verified',366),
                       ('paired_forecast_arrays_verified',6),('models_fit',0),('producer_pooled_model_fits',fits)):
        exact_count(outcome,key,count)
    prepared_required = dict(cached, **{str(out/file):reader.hash(out/file) for file in ('PLAN.json', 'CACHE.json')})
    outcome_files = (*CACHE, *FROZEN, 'CACHE.json', 'FROZEN.json', 'DIAGNOSE_STARTED.json',
                     'RESULTS.json', 'MONTH_COMPARISON.json', 'ATTRIBUTION.json', 'VALIDATION.json')
    outcome_files += tuple(name+'_'+mode+'_'+kind+'.json' for name in NAMES for mode in MODES
                           for kind in ('DAILY', 'TRADES', 'HOLDINGS'))
    outcome_required = {str(out/file):reader.hash(out/file) for file in outcome_files}
    for report, source_key, required in ((prep, 'independent_prepare_source_hashes', prepared_required),
                                       (outcome, 'independent_outcome_source_hashes', outcome_required)):
        sources = reader.verify(report['source_hashes'])
        require(sources == reader.verify(plan[source_key]), 'Independent source set mismatch')
        artifacts = reader.verify(report['artifact_hashes'])
        subset(required, artifacts, 'required independent artifacts')
        subset(artifacts, dict(pins, **required), 'independent artifacts')
    frozen = reader.read(out/'FROZEN.json')
    require(set(frozen['hashes']) == set(FROZEN), 'Frozen preselection artifacts')
    reader.verify(frozen['hashes'], out)
    require(frozen.get('pristine_oos') is False and frozen.get('all_diagnostic_orders_frozen') is True,
            'Frozen diagnostic scope')
    train = reader.read(out/'TRAIN_TRIALS.json')
    selected = selection(train, reader.read(out/'TRAIN_OPEN_TRIALS.json'))
    require(frozen['selection'] == selected and outcome.get('selected_name') == selected['primary'],
            'Frozen first88 selection mismatch')
    require(outcome.get('frozen_sha256') == reader.hash(out/'FROZEN.json'), 'Outcome frozen binding')
    marker = reader.read(out/'DIAGNOSE_STARTED.json')
    require(set(marker) == {'plan_sha256', 'frozen_sha256', 'started_utc'} and
            marker['plan_sha256'] == plan_sha and marker['frozen_sha256'] == reader.hash(out/'FROZEN.json')
            and outcome.get('diagnose_started_sha256') == reader.hash(out/'DIAGNOSE_STARTED.json'),
            'Diagnostic start binding')
    require(utc(marker['started_utc']) < utc(outcome['audit_started_utc']) <= datetime.now(timezone.utc),
            'Diagnostic start/audit chronology')
    result = reader.read(out/'RESULTS.json')
    require(set(result) == set(NAMES) and all(set(result[name]) == set(MODES) for name in NAMES),
            'Three arms and nine execution paths required')
    primary = selected['primary']
    metrics = result[primary]['same_close']['metrics']
    require(all(number(metrics.get(key)) for key in ('cagr', 'sharpe', 'max_drawdown')), 'Finite selected metrics')
    require(number(outcome.get('primary_cagr')) and abs(outcome['primary_cagr']-metrics['cagr']) <= 1e-10,
            'Independent selected CAGR mismatch')
    target = metrics['cagr'] >= .7
    require(type(outcome.get('primary_target_reached')) is bool and outcome['primary_target_reached'] == target,
            'Target must use selected same-close CAGR')
    orders = reader.read(out/'PRETRAIN_ORDERS.json')
    require(set(orders) == set(NAMES) and all(isinstance(rows, list) and len(rows) == 122 for rows in orders.values()),
            '366 original monthly orders required')
    require(orders[NAMES[0]] == reader.read(plan['inputs']['original_orders']['path']), 'Incumbent orders changed')
    ranks = reader.read(out/'RANK_AUDIT.json')
    require(isinstance(ranks.get('months'),list) and len(ranks['months'])==122, 'All paired rank months required')
    for i,month in enumerate(ranks['months']):
        require(isinstance(month,dict) and type(month.get('signal_index')) is int and month['signal_index']==i
                and month.get('date')==orders[NAMES[0]][i]['date'], 'Paired monthly identity mismatch')
        require(type(month.get('paired_ready')) is bool and type(month.get('fallback_to_local_incumbent')) is bool
                and month['fallback_to_local_incumbent'] is not month['paired_ready'], 'Paired fallback flags required')
        require(type(month.get('original_population')) is int and 0<=month['original_population']<=49,
                'Valid original query population required')
        if not month['paired_ready']:
            require(all(orders[name][i]==orders[NAMES[0]][i] for name in NAMES[1:]),
                    'Unavailable paired month must exactly preserve incumbent order')
    tests = reader.read(out/'TESTS.json')
    require(pins.get(str(out/'TESTS.json')) == reader.hash(out/'TESTS.json'), 'Exact test manifest must be registered')
    require(tests.get('passed') is True and isinstance(tests.get('suites'), list) ,
            'Synthetic suites required')
    found = {}
    for suite in tests['suites']:
        name = suite['name']
        require(isinstance(name, str) and name not in found and suite.get('passed') is True, 'Unique passed synthetic suite')
        require(type(suite.get('count')) is int and suite['count'] > 0, 'Positive synthetic test count required')
        for field in ('source', 'log'):
            ref = suite[field]; key = str(reader.path(ref['path']))
            require(reader.hash(key) == ref['sha256'] == pins.get(key), 'Unbound synthetic source/log')
        if name in SUITE_SOURCES:
            require(Path(suite['source']['path']).name == SUITE_SOURCES[name], 'Wrong suite source identity')
        ref = suite['log']
        text = reader.path(ref['path']).read_text()
        require(re.search(r'Ran '+str(suite['count'])+r' tests? in ', text) and text.rstrip().endswith('OK'),
                'Synthetic log does not report success')
        found[name] = suite['count']
    require(set(SUITE_NAMES) <= set(found), 'All new synthetic suites required')
    exact_count(tests, 'test_count', sum(found.values()))
    protected = protected_scan(reader, plan, reader.read(out/'PROTECTED_AFTER.json'))
    return reader, plan, prep, outcome, frozen, result, orders, train, proposal, protected, found


def expected_queue(plan, plan_sha):
    return [dict(round=143, directory=EXPERIMENT,
                 plan=dict(path='outputs/'+EXPERIMENT+'/PLAN.json', sha256=plan_sha),
                 registered_utc=plan['registered_utc'])]


def predecessor(state, plan, plan_sha):
    require(isinstance(state.get('completed'), list) and len(state['completed']) == 142
            and all(isinstance(x, dict) and type(x.get('trials')) is int and x['trials'] >= 0 for x in state['completed'])
            and sum(x['trials'] for x in state['completed']) == 926, 'Expected142 completed rounds and926 trials')
    require(state['completed'][-1].get('directory') == 'monthly_formation142', 'Round142 must be the immediate predecessor')
    require(not any(x.get('directory') == EXPERIMENT for x in state['completed']), 'Round143 already in history')
    require(state.get('registered_queue') == expected_queue(plan, plan_sha), 'Exact registered143 queue required')


def proposal_limitations(proposal):
    """Use the frozen proposal's actual nested schema, never an invented field."""
    require(isinstance(proposal, dict), 'Proposal must be an object')
    derived = []
    for section, key in (('novelty_limits', 'registry_only_previous_definition'), ('novelty_limits','override_rationale'),
                         ('evaluation', 'later'), ('evaluation','mechanism'), ('fixed', 'original_pool')):
        obj = proposal.get(section)
        require(isinstance(obj, dict) and isinstance(obj.get(key), str) and bool(obj[key].strip()),
                'Missing proposal limitation source: '+section+'.'+key)
        derived.append(obj[key])
    for key in ('contrast_interpretation','budget_implication'):
        require(isinstance(proposal.get(key),str) and bool(proposal[key].strip()),
                'Missing proposal limitation source: '+key)
        derived.append(proposal[key])
    return derived + [
        'new_model_fits counts pooled annual fits only; it excludes query-local residual corrections '
        'and independent preparation reconstruction fits. The independent outcome reviewer fits no models.',
        'This experiment inherits the original monthly-universe construction, market-cap proxies, '
        'and retrieved-price vintages. Unchanged source hashes do not independently certify '
        'publication-time availability or complete historical universe coverage.',
        'Same-close zero-cost primary is idealized; next-open zero and5bp per side use identical frozen orders.',
        'No observer, ledger, automation or real order was changed.']


def build(root, plan_sha):
    root = Path(root).resolve(); out = root/'outputs'/EXPERIMENT
    require(not (out/'COMPLETION.json').exists(), 'Completion exists; use exact --resume-completion-sha')
    reader, plan, prep, outcome, frozen, result, orders, train, proposal, protected, suites = validate(root, plan_sha)
    state_raw = (root/STATE).read_bytes(); state = decode(state_raw)
    predecessor(state, plan, plan_sha)
    primary = frozen['selection']['primary']; metrics = result[primary]['same_close']['metrics']
    base = orders[NAMES[0]]
    changes = {name:dict(ordered=sum(a['symbols'] != b['symbols'] for a,b in zip(rows,base)),
                         sets=sum(set(a['symbols']) != set(b['symbols']) for a,b in zip(rows,base)))
               for name, rows in orders.items()}
    def ref(path):return dict(path=str(path.relative_to(root)), sha256=reader.hash(path))
    artifacts = {name:ref(out/name) for name in sorted({Path(p).name for p in reader.seen if Path(p).parent == out})}
    now = datetime.now(timezone.utc).isoformat()
    record = dict(schema_version=1, round=143, experiment=EXPERIMENT,
        status='completed_independently_verified', completed_utc=now, plan_sha256=plan_sha,
        strategy_evaluations=3, new_challengers=1, new_admitted=int(frozen['selection']['admitted']),
        new_model_fits=model_fits(reader.read(out/'MODEL_AUDIT.json')),
        independent_outcome_models_fit=0, primary=primary, selection=frozen['selection'], primary_metrics=metrics,
        primary_next_open_cost=result[primary]['next_open_cost']['metrics'], all_arm_results=result,
        target_reached=metrics['cagr'] >= .7, holding_changes=changes,
        finding='Paired-open target passed both fixed first88 execution gates.' if frozen['selection']['admitted'] else
                'Paired-open target failed a fixed first88 gate; retain local67.',
        model_audit=ref(out/'MODEL_AUDIT.json'), rank_audit=ref(out/'RANK_AUDIT.json'), label_audit=ref(out/'LABEL_AUDIT.json'),
        verification=dict(independent_execution_paths=9, independent_orders=366, paired_forecast_arrays=6,
            original_arrays_unchanged=True, protected_files=protected, protected_members_rehashed=True,
            live_cloud_account_state_certified=False,
            prepare_report=ref(out/'INDEPENDENT_PREPARE.json'), outcome_report=ref(out/'INDEPENDENT_OUTCOME.json'),
            protected_report=ref(out/'PROTECTED_AFTER.json'), synthetic_tests=suites,
            max_independent_nav_absolute_error=outcome['max_nav_absolute_difference']),
        limitations=proposal_limitations(proposal),
        artifacts=artifacts, input_hashes=dict(reader.seen),
        state_precondition=dict(path=STATE, sha256=hashlib.sha256(state_raw).hexdigest()),
        continuation_before=state)
    record['diagnostic_summary'] = {name:dict(same_close=result[name]['same_close']['metrics'],
        next_open_cost=result[name]['next_open_cost']['metrics'], training=train[i]['training'],
        later34=result[name]['same_close']['slices']['later34']) for i,name in enumerate(NAMES)}
    return record


def sync_directory(path):
    fd = os.open(str(path), os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def publish(path, value):
    """Publish complete fsynced bytes with an exclusive link, never overwrite."""
    path = Path(path); temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.COMPLETION.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name); stream.write(encoded(value)); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, path)
        sync_directory(path.parent)
    finally:
        if temporary is not None:temporary.unlink(missing_ok=True)


def state_after(completion, completion_ref):
    state = deepcopy(completion['continuation_before'])
    metrics = completion['primary_metrics']; primary = completion['primary']; now = completion['completed_utc']
    state['completed'].append(dict(completed_utc=now, completion=completion_ref, directory=EXPERIMENT,
        independent_execution_paths=9, new_admitted=completion['new_admitted'], primary=primary,
        primary_cagr=metrics['cagr'], primary_sharpe=metrics['sharpe'], protected_files=PROTECTED_COUNT, actual_model_fits=completion['new_model_fits'],
        target_reached=completion['target_reached'], trials=3))
    state['registered_queue'] = []
    state.update(current=EXPERIMENT, current_status='completed_independently_verified', updated_utc=now,
        latest_round_receipt=completion_ref, last_result_review=completion['verification']['outcome_report'],
        last_protected_check=completion['verification']['protected_report'])
    best = state['best_training_selected_candidate']
    require(number(best['cagr']), 'Existing best CAGR invalid')
    if completion['new_admitted'] == 1 and metrics['cagr'] > best['cagr']:
        state['best_training_selected_candidate'] = dict(directory=EXPERIMENT, spec=primary,
            cagr=metrics['cagr'], sharpe=metrics['sharpe'], max_drawdown=metrics['max_drawdown'],
            next_open_5bp_cagr=completion['primary_next_open_cost']['cagr'], observer_promotion=False)
    state['next_step'] = ('Keep the fixed first88-selected rule. Further research requires a distinct economic hypothesis '
        'and prospective registration; do not promote a diagnostic control or rescue this candidate with later34.')
    state['last_goal_turn_classification'] = ('progress: completed independent paired-target and nine-path validation; target '
        + ('reached in repeated historical research' if completion['target_reached'] else 'unmet')+'.')
    return state


def commit(root, completion_sha):
    root = Path(root).resolve(); reader = Reader(root); path = root/'outputs'/EXPERIMENT/'COMPLETION.json'
    require(digest(completion_sha) and reader.hash(path) == completion_sha, 'Exact completion SHA required')
    completion = reader.read(path)
    require(completion.get('round') == 143 and completion.get('experiment') == EXPERIMENT
            and completion.get('status') == 'completed_independently_verified', 'Wrong completion')
    require(completion['state_precondition']['path'] == STATE, 'Wrong state precondition')
    exact_count(completion, 'strategy_evaluations', 3)
    exact_count(completion, 'new_challengers', 1)
    exact_count(completion, 'independent_outcome_models_fit', 0)
    exact_count(completion, 'new_model_fits', model_fits(reader.read(root/'outputs'/EXPERIMENT/'MODEL_AUDIT.json')))
    plan = reader.read(root/'outputs'/EXPERIMENT/'PLAN.json')
    require(reader.hash(root/'outputs'/EXPERIMENT/'PLAN.json') == completion['plan_sha256'], 'Completion PLAN changed')
    recorded = completion['input_hashes']
    require(isinstance(recorded, dict), 'Completion input hash closure required')
    for key, expected in plan['hashes'].items():
        require(recorded.get(str(reader.path(key))) == expected, 'Completion omitted a PLAN dependency')
    manifest = reader.read(plan['inputs']['protected_manifest']['path'])
    require(len(manifest) == PROTECTED_COUNT, 'Full protected inventory required on resume')
    for name, expected in manifest.items():
        member_path = reader.path(name)
        require(Path(name).is_absolute() and Path(name) == member_path and member_path.is_relative_to(root)
                and stat.S_ISREG(member_path.stat().st_mode), 'Protected resume member path changed')
        require(recorded.get(str(member_path)) == expected, 'Completion omitted a protected member')
    reader.verify(recorded)
    chosen = selection(reader.read(root/'outputs'/EXPERIMENT/'TRAIN_TRIALS.json'),
                       reader.read(root/'outputs'/EXPERIMENT/'TRAIN_OPEN_TRIALS.json'))
    require(completion['selection'] == chosen and completion['primary'] == chosen['primary']
            and type(completion['new_admitted']) is int
            and completion['new_admitted'] == int(chosen['admitted']), 'Completion selected arm changed')
    results = reader.read(root/'outputs'/EXPERIMENT/'RESULTS.json')
    metrics = results[chosen['primary']]['same_close']['metrics']
    require(completion['primary_metrics'] == metrics and number(metrics['cagr'])
            and type(completion['target_reached']) is bool and completion['target_reached'] == (metrics['cagr'] >= .7),
            'Completion selected target changed')
    predecessor(completion['continuation_before'], plan, completion['plan_sha256'])
    ref = dict(path=str(path.relative_to(root)), sha256=completion_sha)
    wanted = state_after(completion, ref)
    destination = root/STATE; before = destination.read_bytes(); current = decode(before)
    if current == wanted:return False
    expected = completion['state_precondition']['sha256']
    require(hashlib.sha256(before).hexdigest() == expected and current == completion['continuation_before'],
            'Continuation changed; refusing to overwrite history')
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix='.CONTINUATION_STATUS.', suffix='.tmp', delete=False) as stream:
            temporary=Path(stream.name);stream.write(encoded(wanted));stream.flush();os.fsync(stream.fileno())
        require(sha(destination) == expected, 'Continuation changed before commit')
        require(sha(path) == completion_sha, 'Completion changed before commit')
        os.replace(temporary, destination);sync_directory(destination.parent)
    finally:
        if temporary is not None:temporary.unlink(missing_ok=True)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--plan-sha')
    mode.add_argument('--resume-completion-sha')
    args = parser.parse_args(); root=args.root.resolve()
    path=root/'outputs'/EXPERIMENT/'COMPLETION.json'
    if args.plan_sha:
        record=build(root,args.plan_sha);publish(path,record);completion_sha=sha(path)
    else:completion_sha=args.resume_completion_sha
    changed=commit(root,completion_sha)
    record=decode(path.read_bytes())
    print(json.dumps(dict(completion=dict(path=str(path.relative_to(root)),sha256=completion_sha),
        state_committed=changed,completed_rounds=143,strategy_evaluations=929,
        primary=record['primary'],cagr=record['primary_metrics']['cagr'],target_reached=record['target_reached'])))


if __name__ == '__main__':main()
