"""Close round145 from verified metadata and bytes; never computes market returns.

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
EXPERIMENT = 'monthly_same_report145'
STATE = 'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
NAMES = ('incumbent_local67', 'expanded_same_report_gross_growth', 'expanded_same_report_profitability')
MODES = ('same_close', 'next_open_zero', 'next_open_cost')
PROPOSAL_SHA = '0e6fa70e1406234b4957a5b996f6a4551e28c6a4cf988f10d823a171231061b2'
PROTECTED_SHA = '89e82ac5cbec3b998f1d8aab93f1f8fd7c5e19522321b2dd1da2f88dfff922cb'
PROTECTED_COUNT = 21444
CACHE = ('DATA.npz', 'SCORES.npz', 'META.json', 'FEATURE_AUDIT.json',
         'PRETRAIN_ORDERS.json', 'PRICE_PARITY.json')
FROZEN = ('PLAN.json', 'CACHE.json', 'PRETRAIN_ORDERS.json', 'TRAIN_TRIALS.json', 'TRAIN_OPEN_TRIALS.json', 'TRAIN_REFERENCE_OPEN.json',
          'WALKFORWARD.json', 'INDEPENDENT_PREPARE.json')
PREPARE_CHECKS = ('original27_arrays_byte_identical', 'all_features_independently_reconstructed',
    'all_three_score_matrices_and_ordered_choices_independently_verified',
    'one_episode_and_prefix_isolation_verified', 'zero_and_missing_exact_fallback',
    'raw_prices_independently_verified', 'calendar_overlay_independently_verified',
    'all_signal_prefixes_verified', 'source_admission_bound', 'common_availability_verified',
    'same_report_comparatives_independently_verified', 'admitted_report_pairs_independently_reconstructed', 'exact_fraction_signs', 'source_union_preservation_verified', 'fixed144_reference_verified')
OUTCOME_CHECKS = ('daily_nav_trades_holdings_metrics_slices', 'first88_and_three_fold_selection_verified',
    'annual_partial_flags_verified', 'monthly_difference_and_attribution_verified',
    'first88_causal_replay_verified', 'zero_model_fit_count_verified', 'all_changed_months_reported',
    'diagnostic_start_marker_verified', 'baseline_local67_full_orders_unchanged',
    'source_admission_bound', 'common_availability_verified', 'first88_dual_execution_selection_verified', 'fixed144_reference_selection_verified')
SUITE_NAMES = ('factor', 'runner', 'prepare_review', 'outcome_review', 'registration', 'completion')
SUITE_SOURCES = {name: ('test_register_monthly_same_report145.py' if name == 'registration' else
    'test_complete_monthly_same_report145.py' if name == 'completion' else
    'test_monthly_same_report145_'+name+'.py') for name in SUITE_NAMES}
SUITE_SOURCES['factor'] = 'test_monthly_same_report144_factor.py'


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


def selection(rows, open_rows, reference_open):
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
    require(isinstance(reference_open,dict) and reference_open.get('name')=='same_report_profitability','Fixed144 reference training required')
    reference_ok=compare(open_rows[2],reference_open,False)
    admitted = all(guards.values()) and all(open_guards.values()) and reference_ok
    return dict(primary=NAMES[2] if admitted else NAMES[0], admitted=admitted, guards=admitted,
                guards_against=guards, next_open_guards_against=open_guards, mechanism_confirmed=all(open_guards.values()), reference144_next_open_noninferior=reference_ok)


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


ADMISSION_FILES = ('PLAN.json','POLICY.json','DATA.json','AUDIT.json','MANIFEST.json','TESTS.json')
UNION_FILES = ('PLAN.json','DATA.json','AUDIT.json','MANIFEST.json','TESTS.json')
METHOD = 'same_current_release_annual_comparison'
CERTIFICATIONS = ('first_public_version_verified','complete_correction_history_verified','certified_daily_security_master')
COUNTERS = ('financial_calculations','model_fits','ranks_computed','returns_computed','network_requests')
PACKAGE_DIRS = {'aapl_amd':'outputs/monthly_same_report144_admission',
                'msft_nvda':'outputs/monthly_msft_nvda_same_report_admission_v1'}
PACKAGE_SYMBOLS = {'aapl_amd':{'AAPL','AMD'},'msft_nvda':{'MSFT','NVDA'}}
PACKAGE_COUNTS = {'aapl_amd':20,'msft_nvda':24}
PACKAGE_CHECKS = {
 'aapl_amd': ('all_eighty_values_and_scales_preserved','all_twenty_events_preserved','causal_identity_windows','four_rejection_sentinels_retained','latest_invalid_blocks_fallback','missing_AMD2019_not_fabricated','old_measurement_definition_not_merged','same_current_release_four_cells','strict_next_session_availability','two_annual_periods_checked'),
 'msft_nvda': ('all24_events_preserved','all96_cells_independently_verified','both_FY2015_rejections_retained','accounting_flags_resolved_only_by_evidence','same_current_release_four_cells','strict_next_session_availability','causal_identity_windows','latest_invalid_blocks_fallback')}
UNION_CHECKS = ('complete_packages_preserved','unique_event_ids','disjoint_issuer_sets','rejected_events_preserved','availability_unmodified','original_package_hashes_verified')
UNION_REVIEW_FILES=('work/after144_union145_independent_review.json','work/after144_union145_independent_review_check.py')
REFERENCE_FILES = {'reference144_plan':'PLAN.json','reference144_orders':'PRETRAIN_ORDERS.json',
                   'reference144_training_open':'TRAIN_OPEN_TRIALS.json'}


def merge_maps(*groups):
    answer={}
    for group in groups:
        for key,value in group.items():
            require(key not in answer or answer[key]==value,'Conflicting source closure')
            answer[key]=value
    return answer


def source_only(doc, counters=False, dataset=False):
    require(all(doc.get(k) is False for k in CERTIFICATIONS),'Source certification limitations must remain explicit')
    if counters:
        for k in COUNTERS:exact_count(doc,k,0)
    if dataset:
        require(doc.get('model_admitted') is False and doc.get('strategy_admitted') is False,'Source-only dataset')
        for k in ('ratios_computed','signs_computed','ranks_computed','returns_computed'):exact_count(doc,k,0)


def check_source_tests(reader, tests, code, package):
    require(tests.get('passed') is True and tests.get('synthetic_only') is True,'Source synthetic tests required')
    for k in ('failures','errors'):exact_count(tests,k,0)
    key='actual_financial_calculations' if package=='aapl_amd' else 'real_source_mappings'
    exact_count(tests,key,0)
    require(type(tests.get('tests_run')) is int and tests['tests_run']>0 and
            reader.verify(tests['source_code_hashes'])==code,'Source tests/code binding')
    log=tests.get('log','')
    require(isinstance(log,str) and re.search(r'Ran '+str(tests['tests_run'])+r' tests? in ',log)
            and log.rstrip().endswith('OK'),'Source test log success')


def source_closure(reader, audit_path, dataset_path):
    """Inspect exact metadata/source dictionaries only; never financial arithmetic."""
    base=reader.root/'outputs/monthly_same_report145_union'
    require(reader.path(audit_path)==base/'AUDIT.json' and reader.path(dataset_path)==base/'DATA.json','Exact source admission paths')
    files={n:reader.read(base/n) for n in UNION_FILES}
    audit=files['AUDIT.json'];plan=files['PLAN.json'];data=files['DATA.json']
    require(audit.get('stage')=='conditional_union_source_admission' and audit.get('passed') is True
            and audit.get('method')==METHOD,'Conditional union source admission required')
    require(set(audit.get('checks',{}))==set(UNION_CHECKS) and all(audit['checks'][k] is True for k in UNION_CHECKS),'Complete union checks')
    source_only(audit, counters=True);source_only(data,dataset=True)
    require(plan.get('stage')=='source_union_predeclared' and plan.get('method')==METHOD and data.get('method')==METHOD,'Source-only union plan')
    for k in COUNTERS:exact_count(plan,k,0)
    exact_count(plan,'expected_events',44);exact_count(plan,'expected_raw_values',176)
    for k,n in (('data_sha256','DATA.json'),('plan_sha256','PLAN.json')):
        require(audit.get(k)==reader.hash(base/n),'Union input binding')
    require(files['MANIFEST.json']==dict(plan_sha256=reader.hash(base/'PLAN.json'),data_sha256=reader.hash(base/'DATA.json'),audit_sha256=reader.hash(base/'AUDIT.json')),'Source manifest bindings')
    sources=reader.verify(audit['source_hashes']);code=reader.verify(audit['source_code_hashes'])
    require(sources==merge_maps(reader.verify(plan['source_hashes']),{str(base/'PLAN.json'):reader.hash(base/'PLAN.json')}),'Union PLAN/AUDIT closure mismatch')
    require(set(code)=={str(reader.root/'work'/n) for n in ('same_report145_union.py','test_same_report145_union.py')},'Exact union code closure')
    subset(code,sources,'union code')
    tests=files['TESTS.json']
    require(tests.get('passed') is True and type(tests.get('tests')) is int and tests['tests']>0,'Union synthetic tests')
    require(tests.get('source_sha256')==code[str(reader.root/'work/same_report145_union.py')] and
            tests.get('test_sha256')==code[str(reader.root/'work/test_same_report145_union.py')],'Union tests/code binding')
    log=tests.get('log',{});require(type(log) is dict and set(log)=={'path','sha256'},'Union test log binding')
    logpath=reader.path(log['path']);require(reader.hash(logpath)==log['sha256']==sources.get(str(logpath)),'Union test log hash')
    text=logpath.read_text();require(re.search(r'Ran '+str(tests['tests'])+r' tests? in ',text) and text.rstrip().endswith('OK'),'Union test log success')
    require(set(plan.get('packages',{}))==set(PACKAGE_DIRS) and plan['packages']==audit.get('packages'),'Exact two source packages')
    events={};package_counts={};assumptions={};missing={}
    for name,relative in PACKAGE_DIRS.items():
        folder=reader.root/relative;docs={n:reader.read(folder/n) for n in ADMISSION_FILES}
        receipt=docs['AUDIT.json'];sp=docs['PLAN.json'];policy=docs['POLICY.json'];pd=docs['DATA.json'];ref=audit['packages'][name]
        require(set(ref)=={'data_path','data_sha256','audit_path','audit_sha256'} and
                reader.path(ref['data_path'])==folder/'DATA.json' and reader.path(ref['audit_path'])==folder/'AUDIT.json','Exact package paths')
        require(ref['data_sha256']==reader.hash(folder/'DATA.json') and ref['audit_sha256']==reader.hash(folder/'AUDIT.json'),'Package SHA binding')
        require(receipt.get('stage')=='conditional_source_admission' and receipt.get('passed') is True,'Conditional package source admission')
        require(set(receipt.get('checks',{}))==set(PACKAGE_CHECKS[name]) and all(receipt['checks'][k] is True for k in PACKAGE_CHECKS[name]),'Complete source admission checks')
        for key,n in (('data_sha256','DATA.json'),('plan_sha256','PLAN.json'),('policy_sha256','POLICY.json')):
            require(receipt.get(key)==reader.hash(folder/n),'Package policy/plan binding')
        source_only(receipt,counters=True);source_only(policy);source_only(pd,dataset=True)
        require(sp.get('stage')=='source_admission_plan' and all(x.get('method')==METHOD for x in (sp,policy,pd)),'Frozen source-only plan')
        financial_key='financial_computations_permitted' if name=='aapl_amd' else 'financial_calculations_permitted'
        require(sp.get(financial_key) is False and sp.get('network_requests_permitted') is False,'Frozen source-only plan permissions')
        ps=reader.verify(receipt['source_hashes']);pc=reader.verify(receipt['source_code_hashes'])
        subset(reader.verify(sp['source_hashes']),ps,'source PLAN');require(reader.verify(sp['source_code_hashes'])==pc,'Source PLAN/AUDIT code mismatch')
        subset(merge_maps(ps,pc),sources,'original package dependencies')
        pm=docs['MANIFEST.json'];require(pm.get('stage')=='conditional_source_admission_complete' and pm.get('plan_sha256')==reader.hash(folder/'PLAN.json') and pm.get('output_hashes')=={n:reader.hash(folder/n) for n in ('DATA.json','AUDIT.json')},'Package manifest binding')
        if name=='msft_nvda':
            bp=folder/'REVIEW_BINDING.json';binding=reader.read(bp)
            rp=reader.root/'work/after144_msft_nvda_source_review.json';review=reader.read(rp)
            require(binding.get('stage')=='independent_review_binding' and binding.get('plan_sha256')==reader.hash(folder/'PLAN.json') and
                    pm.get('review_binding_sha256')==receipt.get('review_binding_sha256')==reader.hash(bp),'Independent source review binding')
            require(reader.path(binding.get('review_path',''))==rp and binding.get('review_sha256')==receipt.get('independent_review_sha256')==reader.hash(rp),'Independent final source review binding')
            require(review.get('stage')=='independent_same_report_source_review' and review.get('passed') is True and
                    review.get('source_only') is True and review.get('method')==METHOD,'Passed independent source review required')
            expected_binding=merge_maps(reader.verify(sp['source_hashes']),pc,reader.verify(review['source_hashes']),{str(rp):reader.hash(rp),str(folder/'PLAN.json'):reader.hash(folder/'PLAN.json')})
            require(reader.verify(binding['source_hashes'])==expected_binding and ps==merge_maps(expected_binding,{str(bp):reader.hash(bp)}),'Independent source review exact closure')
        else:require(ps==reader.verify(sp['source_hashes']),'Source PLAN/AUDIT closure mismatch')
        check_source_tests(reader,docs['TESTS.json'],pc,name)
        require(bool(policy.get('assumptions')) and receipt.get('limitations')==policy['assumptions']==pd.get('assumptions'),'Source assumptions must be carried unchanged')
        assumptions[name]=pd['assumptions'];missing[name]=pd.get('missing_publications',[])
        rows=pd.get('events');require(isinstance(rows,list) and len(rows)==PACKAGE_COUNTS[name],'Complete source event census')
        require({x.get('symbol') for x in rows}==PACKAGE_SYMBOLS[name],'Exact package symbols')
        for row in rows:
            key=row.get('event_id');require(isinstance(key,str) and key and key not in events,'Unique source event IDs')
            require(row.get('method')==METHOD and type(row.get('admitted')) is bool and isinstance(row.get('rejection_codes'),list) and row['admitted']==(not row['rejection_codes']),'Source rejection states')
            events[key]=row
        counts=dict(events=len(rows),conditionally_admitted=sum(x['admitted'] for x in rows),rejected=sum(not x['admitted'] for x in rows),raw_values=4*len(rows))
        require(pd.get('counts')==receipt.get('counts')==counts,'Package counts disagree with events');package_counts[name]=counts
        subset({str(folder/n):reader.hash(folder/n) for n in ADMISSION_FILES},sources,'original package files')
    rows=data.get('events');require(isinstance(rows,list) and len(rows)==44,'Complete union event census')
    require(rows==sorted(events.values(),key=lambda x:(x['symbol'],x['available_from'],x['event_id'])),'Exact original package events preserved')
    counts=dict(events=44,conditionally_admitted=sum(x['admitted'] for x in rows),rejected=sum(not x['admitted'] for x in rows),raw_values=176)
    require(data.get('counts')==audit.get('counts')==counts and data.get('package_counts')==package_counts,'Union counts disagree with events')
    require(data.get('assumptions')==audit.get('limitations')==assumptions and data.get('missing_publications')==missing,'Union source assumptions preserved')
    review_path=reader.root/UNION_REVIEW_FILES[0];review=reader.read(review_path)
    checker=reader.root/UNION_REVIEW_FILES[1]
    require(review.get('stage')=='independent_source_union_review' and review.get('current_artifact_passed') is True and review.get('source_only') is True,'Independent current union review required')
    for k in ('financial_calculations','model_fits','ranks_computed','returns_computed'):exact_count(review,k,0)
    require(type(review.get('checks')) is dict and bool(review['checks']) and all(v is True for v in review['checks'].values()),'Independent union checks')
    union_artifacts={str(base/n):reader.hash(base/n) for n in UNION_FILES}
    require(reader.verify(review['input_hashes'])==union_artifacts and reader.verify(review['implementation_hashes'])==code and review.get('counts')==counts,'Independent union input binding')
    return merge_maps(sources,code,union_artifacts,{str(review_path):reader.hash(review_path),str(checker):reader.hash(checker)})


def verify_admission(reader,plan,pins):
    refs=plan['inputs']
    closure=source_closure(reader,refs['source_admission']['path'],refs['admitted_dataset']['path'])
    require(closure==reader.verify(plan['fundamental_source_hashes']),'Exact admitted source closure')
    subset(closure,pins,'admitted sources')


def verify_contract(reader,plan,proposal):
    contract=proposal.get('sources',{});base=reader.root/'outputs/monthly_same_report145_union'
    require(contract.get('directory')=='outputs/monthly_same_report145_union' and
            contract.get('dataset_sha256')==plan['inputs']['admitted_dataset']['sha256'] and
            contract.get('audit_sha256')==plan['inputs']['source_admission']['sha256'],'Final proposal source contract binding')
    require(contract.get('hashes')=={n:reader.hash(base/n) for n in ('DATA.json','AUDIT.json','PLAN.json','MANIFEST.json')},'Final proposal complete union binding')
    references=proposal.get('reference144',{}).get('hashes')
    require(isinstance(references,dict) and set(references)==set(REFERENCE_FILES.values()) and references==plan.get('reference144_hashes'),'Fixed144 proposal reference binding')
    for key,n in REFERENCE_FILES.items():
        ref=plan['inputs'][key];path=reader.root/'outputs/monthly_same_report144'/n
        require(reader.path(ref['path'])==path and ref['sha256']==references[n]==reader.hash(path),'Fixed144 reference binding')
    old=reader.read(plan['inputs']['reference144_plan']['path'])
    require(old.get('round')==144 and old.get('status')=='REGISTERED','Fixed144 registered reference required')

def verified_reference(reader,plan,out):
    reference=reader.read(out/'TRAIN_REFERENCE_OPEN.json')
    original=reader.read(plan['inputs']['reference144_training_open']['path'])
    rows=[r for r in original if r.get('name')=='same_report_profitability']
    require(len(rows)==1 and reference.get('name')=='same_report_profitability','Fixed144 reference training required')
    for group,field in (('training','cagr'),('training','sharpe'),('training','max_drawdown'),('early','cagr'),('recent','cagr')):
        actual=reference.get(group,{}).get(field);expected=rows[0].get(group,{}).get(field)
        require(number(actual) and number(expected) and abs(actual-expected)<1e-10,'Replayed fixed144 reference training mismatch')
    return reference


def verify_reused_factor(reader,suite,pins=None):
    require(suite.get('reused_from_round')==144 and suite.get('count')==33,'Original 33-test factor suite must be explicitly reused')
    ref=suite.get('original_receipt',{});path=reader.root/'outputs/monthly_same_report144/TESTS.json'
    require(reader.path(ref.get('path',''))==path and reader.hash(path)==ref.get('sha256'),'Original factor test receipt binding')
    receipt=reader.read(path);require(receipt.get('passed') is True,'Original factor tests must have passed')
    original=[x for x in receipt.get('suites',[]) if x.get('name')=='factor']
    require(len(original)==1 and original[0].get('passed') is True and original[0].get('count')==33,'Original factor test count')
    for key in ('source','log'):
        old=original[0][key];new=suite[key]
        require(reader.path(old['path'])==reader.path(new['path']) and old['sha256']==new['sha256']==reader.hash(new['path']),'Original factor test evidence changed')
    require(reader.path(suite['source']['path'])==reader.root/'work/test_monthly_same_report144_factor.py' and
            reader.path(suite['log']['path'])==reader.root/'outputs/monthly_same_report144/FACTOR_TESTS.log','Exact original factor evidence paths')
    if pins is not None:require(pins.get(str(path))==ref['sha256'],'Original factor receipt absent from PLAN')


def validate(root, plan_sha):
    reader = Reader(root)
    out = reader.root/'outputs'/EXPERIMENT
    require(digest(plan_sha) and reader.hash(out/'PLAN.json') == plan_sha, 'Exact PLAN SHA required')
    plan = reader.read(out/'PLAN.json')
    require(plan.get('round') == 145 and plan.get('status') == 'REGISTERED'
            and plan.get('arms') == list(NAMES) and plan.get('primary_challenger') == NAMES[2]
            and reader.path(plan['output_directory']) == out, 'Registered145 three-arm PLAN required')
    pins = reader.verify(plan['hashes'])
    require(str(reader.root/STATE) not in pins, 'Mutable continuation state cannot be an immutable PLAN pin')
    for filename in ('complete_monthly_same_report145.py', 'test_complete_monthly_same_report145.py'):
        path = reader.root/'work'/filename
        require(pins.get(str(path)) == reader.hash(path), 'Closer source/test must be registered')
    require(reader.hash(reader.root/'work/complete_monthly_same_report145.py') == sha(__file__),
            'Running closer differs from bound source')
    for name in ('raw_source_hashes', 'independent_prepare_source_hashes',
                 'independent_outcome_source_hashes', 'fundamental_source_hashes'):
        subset(reader.verify(plan[name]), pins, name)
    require(len(plan['raw_source_hashes']) == 64, 'Original raw64 scope')
    for name in ('actual_model_fits_before_registration', 'actual_feature_calculations_before_registration',
                 'actual_returns_before_registration'):
        exact_count(plan, name, 0)
    require(set(plan['inputs']) == {'proposal', 'meta', 'policy', 'original_orders', 'prices', 'sessions',
                                   'protected_manifest', 'admitted_dataset', 'source_admission', *REFERENCE_FILES}, 'Exact named inputs')
    for ref in plan['inputs'].values():
        key = str(reader.path(ref['path']))
        require(reader.hash(key) == ref['sha256'] == pins.get(key), 'Unbound named input')
    require(plan['inputs']['proposal']['sha256'] == PROPOSAL_SHA, 'Frozen proposal binding')
    verify_admission(reader, plan, pins)
    proposal = reader.read(plan['inputs']['proposal']['path'])
    verify_contract(reader,plan,proposal)
    cache = reader.read(out/'CACHE.json')
    require(set(cache) == set(CACHE), 'Six cache artifacts required')
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
    for key, count in (('orders_checked', 366), ('source_cells_checked', 5978), ('stock_cells_checked', 3904),
                       ('monthly_label_models_fit', 0), ('new_model_fits', 0)):
        exact_count(prep, key, count)
    for key, count in (('execution_paths_verified', 9), ('total_monthly_orders_verified', 366),
                       ('profitability_feature_matrices_verified', 3), ('models_fit', 0), ('producer_model_fits', 0)):
        exact_count(outcome, key, count)
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
    selected = selection(train, reader.read(out/'TRAIN_OPEN_TRIALS.json'), verified_reference(reader,plan,out))
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
    feature = reader.read(out/'FEATURE_AUDIT.json')
    exact_count(feature, 'new_model_fits', 0)
    require(feature.get('latest_event_before_validity_filter') is True and
            feature.get('same_current_release_for_both_periods') is True and
            feature.get('sign_method') == 'exact_decimal_cross_products_after_declared_scale_normalization' and
            feature.get('original_auxiliary_states_and_core_coefficients_preserved') is True and
            feature.get('fixed144_orders_reproduced_without_new_sources') is True and
            feature.get('original_arrays_unchanged') is True and feature.get('returns_or_nav_computed') is False,
            'Profitability feature contract')
    exact_count(feature, 'original_months', 122)
    exact_count(feature, 'original_width', 49)
    cells=feature.get('cells')
    require(isinstance(cells,list) and len(cells)==5978 and
            all(isinstance(cell,dict) and type(cell.get('available')) is bool for cell in cells),
            'All profitability source cells required')
    available=sum(cell['available'] for cell in cells)
    exact_count(feature, 'common_available_cells', available)
    exact_count(outcome, 'available_feature_cells', available)
    exact_count(outcome, 'unavailable_feature_cells', 5978-available)
    for field,key in (('source_admission_sha256','source_admission'),('admitted_dataset_sha256','admitted_dataset')):
        expected=plan['inputs'][key]['sha256']
        require(feature.get(field)==expected and outcome.get(field)==expected,'Feature/outcome admission binding: '+field)
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
        if name=='factor':verify_reused_factor(reader,suite,pins)
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
    return [dict(round=145, directory=EXPERIMENT,
                 plan=dict(path='outputs/'+EXPERIMENT+'/PLAN.json', sha256=plan_sha),
                 registered_utc=plan['registered_utc'])]


def predecessor(state, plan, plan_sha):
    require(isinstance(state.get('completed'), list) and len(state['completed']) == 144
            and all(type(x.get('trials')) is int and x['trials'] >= 0 for x in state['completed'])
            and sum(x['trials'] for x in state['completed']) == 932, 'Expected144 completed rounds and932 trials')
    require(state['completed'][-1].get('directory') == 'monthly_same_report144', 'Round144 must be the immediate predecessor')
    require(not any(x.get('directory') == EXPERIMENT for x in state['completed']), 'Round145 already in history')
    require(state.get('registered_queue') == expected_queue(plan, plan_sha), 'Exact registered145 queue required')


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
    record = dict(schema_version=1, round=145, experiment=EXPERIMENT,
        status='completed_independently_verified', completed_utc=now, plan_sha256=plan_sha,
        strategy_evaluations=3, new_challengers=1, new_admitted=int(frozen['selection']['admitted']),
        new_model_fits=0, primary=primary, selection=frozen['selection'], primary_metrics=metrics,
        primary_next_open_cost=result[primary]['next_open_cost']['metrics'], all_arm_results=result,
        target_reached=metrics['cagr'] >= .7, holding_changes=changes,
        finding='Same-report profitability-improvement candidate passed the fixed first88 dual-execution gate.' if frozen['selection']['admitted'] else
                'Same-report profitability-improvement candidate failed the fixed first88 dual-execution gate; retain local67.',
        feature_audit=ref(out/'FEATURE_AUDIT.json'),
        source_admission=dict(plan['inputs']['source_admission']), admitted_dataset=dict(plan['inputs']['admitted_dataset']),
        source_admission_is_conditional=True, first_public_version_verified=False,
        complete_correction_history_verified=False, certified_daily_security_master=False,
        source_assumptions=reader.read(plan['inputs']['source_admission']['path'])['limitations'],
        verification=dict(independent_execution_paths=9, independent_orders=366, profitability_feature_matrices=3,
            source_admission_bound=True, common_availability_verified=True, first88_dual_execution_selection_verified=True,
            original_arrays_unchanged=True, protected_files=protected, protected_members_rehashed=True,
            live_cloud_account_state_certified=False,
            prepare_report=ref(out/'INDEPENDENT_PREPARE.json'), outcome_report=ref(out/'INDEPENDENT_OUTCOME.json'),
            protected_report=ref(out/'PROTECTED_AFTER.json'), synthetic_tests=suites,
            newly_executed_test_count=sum(suites.values())-33, reused_test_count=33,
            reused_factor_validation=next(x for x in reader.read(out/'TESTS.json')['suites'] if x['name']=='factor'),
            max_independent_nav_absolute_error=outcome['max_nav_absolute_difference']),
        limitations=proposal.get('interpretation',proposal.get('interpretation_limits',proposal.get('limitations',[])))+['Same-close zero-cost primary is idealized; next-open zero and5bp per side use identical frozen orders.',
            'No observer, ledger, automation or real order was changed.'],
        artifacts=artifacts, input_hashes=dict(reader.seen),
        state_precondition=dict(path=STATE, sha256=hashlib.sha256(state_raw).hexdigest()),
        continuation_before=state)
    record['first88_next_open_training'] = reader.read(out/'TRAIN_OPEN_TRIALS.json')
    record['first88_fixed144_reference_training'] = reader.read(out/'TRAIN_REFERENCE_OPEN.json')
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
        primary_cagr=metrics['cagr'], primary_sharpe=metrics['sharpe'], protected_files=PROTECTED_COUNT,
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
    state['next_step'] = ('Keep the training-selected incumbent. Further research requires a distinct economic hypothesis '
        'and prospective registration; do not promote a diagnostic control or rescue this candidate with later34.')
    state['last_goal_turn_classification'] = ('progress: completed independent profitability-improvement and nine-path validation; target '
        + ('reached in repeated historical research' if completion['target_reached'] else 'unmet')+'.')
    return state


def commit(root, completion_sha):
    root = Path(root).resolve(); reader = Reader(root); path = root/'outputs'/EXPERIMENT/'COMPLETION.json'
    require(digest(completion_sha) and reader.hash(path) == completion_sha, 'Exact completion SHA required')
    completion = reader.read(path)
    require(completion.get('round') == 145 and completion.get('experiment') == EXPERIMENT
            and completion.get('status') == 'completed_independently_verified', 'Wrong completion')
    require(completion['state_precondition']['path'] == STATE, 'Wrong state precondition')
    exact_count(completion, 'strategy_evaluations', 3)
    exact_count(completion, 'new_challengers', 1)
    exact_count(completion, 'new_model_fits', 0)
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
    verify_contract(reader,plan,reader.read(plan['inputs']['proposal']['path']))
    chosen = selection(reader.read(root/'outputs'/EXPERIMENT/'TRAIN_TRIALS.json'),
                       reader.read(root/'outputs'/EXPERIMENT/'TRAIN_OPEN_TRIALS.json'),
                       verified_reference(reader,plan,root/'outputs'/EXPERIMENT))
    require(completion['selection'] == chosen and completion['primary'] == chosen['primary']
            and type(completion['new_admitted']) is int
            and completion['new_admitted'] == int(chosen['admitted']), 'Completion selected arm changed')
    suites=reader.read(root/'outputs'/EXPERIMENT/'TESTS.json')['suites']
    factor=next(x for x in suites if x['name']=='factor')
    require(completion['verification'].get('reused_factor_validation')==factor and
            completion['verification'].get('reused_test_count')==33 and
            completion['verification'].get('newly_executed_test_count')==sum(x['count'] for x in suites)-33,
            'Completion changed synthetic test reuse provenance')
    source_receipt = reader.read(plan['inputs']['source_admission']['path'])
    require(completion.get('source_admission_is_conditional') is True and
            all(completion.get(key) is False for key in CERTIFICATIONS) and
            completion.get('source_assumptions') == source_receipt.get('limitations'),
            'Completion changed conditional source assumptions')
    require(completion.get('source_admission') == plan['inputs']['source_admission'] and
            completion.get('admitted_dataset') == plan['inputs']['admitted_dataset'],
            'Completion source references changed')
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
        state_committed=changed,completed_rounds=145,strategy_evaluations=935,
        primary=record['primary'],cagr=record['primary_metrics']['cagr'],target_reached=record['target_reached'])))


if __name__ == '__main__':main()
