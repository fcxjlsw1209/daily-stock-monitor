"""Prospective own-sector source audit: mappings and endpoint validity only.

The pure inspector never computes a price ratio, score, eligibility or return.
Actual sources are loaded only by audit() after write-once source registration.
"""
from pathlib import Path
from datetime import date, datetime, timezone
from numbers import Real
import argparse
import ast
import hashlib
import importlib.util
import io
import json
import math
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/monthly_ownsector238_source_audit'
CONTEXT = ROOT / 'work/ownsector238_source_context.json'
CONTEXT_SHA = '9ef5ffd55d3b7c615a519b6f44ffdc3405330b3a17ca421cd40bab7291bd50d6'
AMENDMENT = ROOT / 'work/ownsector238_source_context_amendment.json'
AMENDMENT_SHA = '08e132f6edfaf9da7034d533ffde7508f5ec0d60211af1e993c8ef15caf9f5d3'
COVERAGE = ROOT / 'work/after237_ownsector_gate_coverage.json'
COVERAGE_SHA = 'be0f6bc8bd9f111ed91c7a1106d61c734d09bd3b57df39ccfec9a50ddb19330d'
TESTS = ROOT / 'work/ownsector238_source_TESTS.json'
CODE_REVIEW = ROOT / 'work/ownsector238_source_CODE_REVIEW.json'
PRIOR = ROOT / 'outputs/monthly_allstate_gate237'
LOCAL = ROOT / 'outputs/monthly_local_linear70'
CALENDAR = ROOT / 'outputs/monthly_paired_target143/CALENDAR.json'
PROTECTED = ROOT / 'work/trade_reconcile_20261015_close/NEXT_EXPECTED_PROTECTED.json'
CENTRAL = ROOT / 'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
MAPPING = ROOT / 'outputs/monthly_strict_relative/strict_study.py'
MAPPING_SHA = 'ddb11c0149e920de2952daba515ce28625a56df1f5c1039d4b9920c6d58d02c6'
PREFIXES = (0, 1, 53, 88, 122)
SECTOR = {'Basic Materials':'XLB', 'Communication Services':'XLC', 'Energy':'XLE',
    'Financial Services':'XLF', 'Industrials':'XLI', 'Technology':'XLK',
    'Consumer Defensive':'XLP', 'Real Estate':'XLRE', 'Utilities':'XLU',
    'Healthcare':'XLV', 'Consumer Cyclical':'XLY'}
FOREIGN = {'ASML':'XLK', 'ARM':'XLK', 'TSM':'XLK', 'BABA':'XLY', 'NVO':'XLV',
    'SAP':'XLK', 'TM':'XLY', 'SHEL':'XLE', 'SKHY':'XLK', 'BHP':'XLB',
    'HSBC':'XLF', 'SMFG':'XLF', 'NVS':'XLV', 'SNDK':'XLK'}
COUNT_FIELDS = ('events','symbols_per_event','checked_cells','initial_partial_events',
    'regular_events','original_eligible_cells','required_cells','stock_required_cells',
    'etf_required_cells','mapping_available_cells','mapping_missing_cells',
    'security_benchmark_endpoint_checks','market_endpoint_checks','valid_endpoints',
    'missing_endpoints','invalid_endpoints','admitted_cells','gap_cells','market_gap_events')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, obj):
    with Path(path).open('x') as stream:
        json.dump(obj, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def now():
    return datetime.now(timezone.utc).isoformat()


def _iso(value):
    require(isinstance(value, str), 'Canonical ISO date required')
    d = date.fromisoformat(value)
    require(d.isoformat() == value, 'Canonical ISO date required')
    return pd.Timestamp(d)


def _frames(panel):
    require(isinstance(panel, dict) and set(panel) == {'adj', 'sectors'}, 'Exactly adj/sectors sources required')
    for label, frame in panel.items():
        require(isinstance(frame, pd.DataFrame) and isinstance(frame.index, pd.DatetimeIndex), label+' DataFrame required')
        idx = frame.index
        require(not idx.hasnans and idx.is_unique and idx.is_monotonic_increasing and idx.tz is None
            and idx.equals(idx.normalize()), label+' invalid date axis')
        require(frame.columns.is_unique and all(isinstance(s,str) and s and s.strip()==s for s in frame.columns), label+' invalid identity axis')


def _inputs(panel, meta, calendar, eligible, limit):
    _frames(panel)
    e = np.asarray(eligible)
    require(e.ndim == 2 and e.shape[1] > 0, 'Eligibility event/security axes required')
    total, width = e.shape
    require(limit is None or type(limit) is int and 0 <= limit <= total, 'Invalid event prefix')
    n = total if limit is None else limit
    require(all(isinstance(v,(bool,np.bool_)) for v in e[:n].flat), 'Consumed eligibility must be exact boolean')
    require(isinstance(meta,dict) and all(isinstance(meta.get(k),list) and len(meta[k])==total for k in ('dates','symbols')), 'Metadata event axes differ')
    etfs = meta.get('etfs')
    require(isinstance(etfs,(tuple,list)) and all(isinstance(s,str) and s and s.strip()==s for s in etfs)
        and len(set(etfs))==len(etfs), 'Explicit unique ETF identities required')
    require(isinstance(calendar,(tuple,list)) and (len(calendar)>0 or n==0), 'Official session sequence required')
    sessions = [_iso(s) for s in calendar]
    require(all(s.weekday()<5 for s in sessions) and all(a<b for a,b in zip(sessions,sessions[1:])), 'Official sessions must increase uniquely')
    positions = {d:i for i,d in enumerate(sessions)}
    queries = []
    for i in range(n):
        day = _iso(meta['dates'][i]); names = meta['symbols'][i]
        require(day in positions and (not queries or day>queries[-1][0]), 'Queries must increase on official sessions')
        require(isinstance(names,list) and len(names)==width and all(isinstance(s,str) and s and s.strip()==s for s in names)
            and len(set(names))==width, 'Query identities must be unique and full-width')
        pos = positions[day]; partial = meta['dates'][i]=='2016-08-17'
        require(not partial or i==0, 'Named partial must be first')
        if not partial:
            require(pos>=252 and pos+1<len(sessions) and sessions[pos+1].to_period('M')!=day.to_period('M'), 'Official endpoint history or month-end witness missing')
        queries.append((day, list(names), pos, partial))
    return e[:n].astype(bool,copy=True), queries, sessions, list(etfs), width


def _mapped(day, symbol, sectors, etfs):
    if symbol in etfs:
        return None
    if symbol in FOREIGN:
        return FOREIGN[symbol]
    if day not in sectors.index or symbol not in sectors.columns:
        return None
    try:
        return SECTOR.get(sectors.at[day,symbol])
    except TypeError as exc:
        raise ValueError('Malformed sector classification') from exc


def endpoint(frame, role, symbol, day):
    """Classify a single source scalar, without a ratio or stored price output."""
    status, reason = 'valid', None
    if symbol is None:
        status, reason = 'missing', 'missing_benchmark_mapping'
    elif day not in frame.index:
        status, reason = 'missing', 'missing_row'
    elif symbol not in frame.columns:
        status, reason = 'missing', 'missing_column'
    else:
        value = frame.at[day,symbol]
        if value is None or value is pd.NA:
            status, reason = 'missing', 'missing_value'
        elif isinstance(value,(bool,np.bool_)):
            status, reason = 'invalid', 'invalid_boolean'
        elif not isinstance(value,Real):
            status, reason = 'invalid', 'invalid_nonreal'
        else:
            try:
                val = float(value)
            except (ValueError,TypeError,OverflowError):
                status, reason = 'invalid', 'invalid_numeric_conversion'
            else:
                if math.isnan(val):
                    status, reason = 'missing', 'missing_value'
                elif not math.isfinite(val):
                    status, reason = 'invalid', 'invalid_nonfinite'
                elif val <= 0:
                    status, reason = 'invalid', 'invalid_nonpositive'
    return dict(role=role,symbol=symbol,date=str(day.date()),status=status,reason=reason)


def inspect_sources(panel, meta, calendar, eligible, benchmark_map, limit=None):
    e, queries, sessions, etfs, width = _inputs(panel,meta,calendar,eligible,limit)
    require(callable(benchmark_map), 'Frozen mapping callable required')
    records, events, gaps = [], [], []
    for i,(day,names,pos,partial) in enumerate(queries):
        d = str(day.date()); indices = [j for j,v in enumerate(e[i]) if v]
        required = [] if partial else indices.copy()
        positions = [] if partial else [pos-252,pos-42]
        dates = [sessions[k] for k in positions]
        event = dict(query_index=i,date=d,initial_partial=partial,active=not partial,
            original_eligible_indices=indices,required_indices=required,source_positions=positions,
            source_dates=[str(x.date()) for x in dates],market_endpoints=[],required_count=len(required),admitted_count=0,gap_count=0)
        if not partial:
            # Detached, as-of sector input prevents future information or callback mutation.
            mapping = benchmark_map(d,names.copy(),panel['sectors'].loc[:day].copy(deep=True),etfs.copy())
            require(isinstance(mapping,dict) and set(mapping)==set(names), 'Mapping must contain exact query identities')
            require(all(v is None or isinstance(v,str) and v and v.strip()==v for v in mapping.values()), 'Malformed benchmark identity')
            for j in required:
                require(mapping[names[j]] == _mapped(day,names[j],panel['sectors'],etfs), 'Mapping differs from frozen precedence')
            market = [endpoint(panel['adj'],role,'SPY',dd) for role,dd in zip(('market_start','market_end'),dates)]
            event['market_endpoints'] = market
            missing = list(dict.fromkeys(x['reason'] for x in market if x['reason'] is not None))
            if missing:
                gaps.append(dict(scope='market',query_index=i,column_index=None,date=d,symbol='SPY',benchmark='SPY',reasons=missing))
            for j in required:
                symbol = names[j]; etf = symbol in etfs
                benchmark = 'SPY' if etf else mapping[symbol]
                origin = 'original_META.etfs_SPY' if etf else 'frozen_FOREIGN_override' if symbol in FOREIGN else 'query_date_sector_panel'
                slots = [endpoint(panel['adj'],role,sym,dd) for role,sym,dd in (
                    ('security_start',symbol,dates[0]),('security_end',symbol,dates[1]),
                    ('benchmark_start',benchmark,dates[0]),('benchmark_end',benchmark,dates[1]))]
                reasons = [] if benchmark is not None else ['missing_benchmark_mapping']
                for x in slots:
                    if x['reason'] is not None and x['reason'] not in reasons:
                        reasons.append(x['reason'])
                record = dict(query_index=i,column_index=j,date=d,symbol=symbol,is_etf=etf,benchmark=benchmark,
                    mapping_source=origin,mapping_status='available' if benchmark is not None else 'missing',
                    endpoints=slots,admitted=not reasons,gap_reasons=reasons)
                records.append(record)
                if reasons:
                    gaps.append(dict(scope='cell',query_index=i,column_index=j,date=d,symbol=symbol,benchmark=benchmark,reasons=reasons))
                    event['gap_count'] += 1
                else:
                    event['admitted_count'] += 1
        events.append(event)
    points = [x for event in events for x in event['market_endpoints']] + [x for cell in records for x in cell['endpoints']]
    counts = dict(events=len(events),symbols_per_event=width,checked_cells=len(events)*width,
        initial_partial_events=sum(x['initial_partial'] for x in events),regular_events=sum(x['active'] for x in events),
        original_eligible_cells=sum(len(x['original_eligible_indices']) for x in events),required_cells=len(records),
        stock_required_cells=sum(not x['is_etf'] for x in records),etf_required_cells=sum(x['is_etf'] for x in records),
        mapping_available_cells=sum(x['mapping_status']=='available' for x in records),mapping_missing_cells=sum(x['mapping_status']=='missing' for x in records),
        security_benchmark_endpoint_checks=4*len(records),market_endpoint_checks=2*sum(x['active'] for x in events),
        valid_endpoints=sum(x['status']=='valid' for x in points),missing_endpoints=sum(x['status']=='missing' for x in points),
        invalid_endpoints=sum(x['status']=='invalid' for x in points),admitted_cells=sum(x['admitted'] for x in records),
        gap_cells=sum(not x['admitted'] for x in records),market_gap_events=sum(x['scope']=='market' for x in gaps))
    require(tuple(counts)==COUNT_FIELDS, 'Source audit counter schema')
    return dict(schema='ownsector238_source_inspection_v1',status='NOT_ADMITTED' if gaps else 'ADMITTED',admitted=not gaps,
        counts=counts,events=events,records=records,gaps=gaps,source_only=True,momentum_computed=False,
        eligibility_changed=False,rankings_computed=False,orders_computed=False,returns_computed=False,model_fits=0,strict_pit=False)


def physical_prefix(panel, meta, calendar, eligible, n):
    require(type(n) is int and 0<=n<=len(meta['dates']), 'Invalid physical prefix')
    mm = dict(meta, dates=list(meta['dates'][:n]), symbols=[list(x) for x in meta['symbols'][:n]], etfs=list(meta['etfs']))
    if n:
        cutoff = _iso(mm['dates'][-1]); i = calendar.index(str(cutoff.date()))
        require(i+1<len(calendar), 'Calendar witness missing')
        cc = list(calendar[:i+2])
        pp = {k:v.loc[:cutoff].copy(deep=True) for k,v in panel.items()}
    else:
        cc = []; pp = {k:v.iloc[:0].copy(deep=True) for k,v in panel.items()}
    return pp, mm, cc, np.asarray(eligible)[:n].copy()


def mapping_constants():
    require(sha(MAPPING)==MAPPING_SHA, 'Frozen mapper SHA differs')
    tables = {}
    for node in ast.parse(MAPPING.read_text()).body:
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name) and target.id in ('SECTOR','FOREIGN'):
                    require(target.id not in tables, 'Duplicate frozen table')
                    tables[target.id] = ast.literal_eval(node.value)
    require(tables==dict(SECTOR=SECTOR,FOREIGN=FOREIGN), 'Frozen mapper table differs')


def previous():
    import monthly_allstate_gate237_host as obj
    return obj


def code_files():
    import ownsector238_source_independent as independent
    first = [ROOT/'work'/n for n in ('ownsector238_source_audit.py','test_ownsector238_source_audit.py',
        'ownsector238_source_independent.py','test_ownsector238_source_independent.py')]
    other = [ROOT/'work/monthly_allstate_gate237_host.py',ROOT/'work/ptr169_rebuild_evaluate.py',
        CONTEXT,AMENDMENT,COVERAGE,MAPPING]+list(independent.dependency_files())
    return first + sorted(set(other)-set(first))


def test():
    require(not TESTS.exists(), 'Formal synthetic receipt is write-once')
    old = previous(); modules = [old.module('source238_tests_'+str(i),p) for i,p in enumerate(code_files()[1:4:2])]
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromModule(m) for m in modules)
    stream = io.StringIO(); result = unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    print(stream.getvalue(),flush=True)
    require(result.wasSuccessful(), 'Synthetic source audit tests failed')
    dump(TESTS,dict(passed=True,synthetic_only=True,tests=result.testsRun,runtime=old.ev.runtime(),
        context_sha256=sha(CONTEXT),amendment_sha256=sha(AMENDMENT),code_sha256={str(p):sha(p) for p in code_files()}))


def source_binding():
    old = previous()
    require(sha(CONTEXT)==CONTEXT_SHA and sha(AMENDMENT)==AMENDMENT_SHA and sha(COVERAGE)==COVERAGE_SHA, 'Source context/amendment/coverage changed')
    context = read(CONTEXT); pins = dict(read(PRIOR/'PLAN.json')['input_sha256'])
    for name,digest in context['evidence_sha256'].items():
        old.bind(pins,ROOT/name,digest)
    mapping_constants()
    prior_plan = read(PRIOR/'PLAN.json'); closed = read(PRIOR/'COMPLETION.json')
    require(closed['status']=='COMPLETED_INDEPENDENTLY_VERIFIED' and closed['plan_sha256']==sha(PRIOR/'PLAN.json')
        and closed['protected_files']==21444 and closed['observers_modified'] is False, 'Prior237 must be closed')
    raw = read(LOCAL/'PLAN.json')['raw_source_hashes']
    require(raw==prior_plan['raw_source_hashes'] and len(raw)==64, 'Original64 raw map differs')
    provenance = old.price_source_binding(raw)
    for name,digest in provenance['input_sha256'].items():old.bind(pins,Path(name),digest)
    old.verify(pins); old.verify(raw)
    return pins,raw,provenance


def register():
    require(not OUT.exists(), 'New source-only output directory required')
    old = previous(); before = CENTRAL.read_bytes()
    t = read(TESTS)
    require(t.get('passed') is True and t.get('synthetic_only') is True and t.get('runtime')==old.ev.runtime()
        and t.get('context_sha256')==CONTEXT_SHA and t.get('amendment_sha256')==AMENDMENT_SHA, 'Formal tests/runtime/context mismatch')
    require(t.get('code_sha256')=={str(p):sha(p) for p in code_files()}, 'Code changed after synthetic tests')
    review = read(CODE_REVIEW)
    require(review.get('passed') is True and review.get('unresolved_blockers')==[]
        and review.get('context_sha256')==CONTEXT_SHA and review.get('amendment_sha256')==AMENDMENT_SHA and review.get('tests_sha256')==sha(TESTS), 'Independent code review incomplete')
    require(review.get('code_sha256')=={str(p.relative_to(ROOT)):sha(p) for p in code_files()[:4]}, 'Code review pins differ')
    require(review.get('actual_source_audit_executed') is False and review.get('actual_strategy_math_computed') is False, 'Code review scope differs')
    pins,raw,provenance = source_binding()
    for name,digest in review.get('input_sha256',{}).items():old.bind(pins,Path(name) if Path(name).is_absolute() else ROOT/name,digest)
    for p in code_files()+[TESTS,CODE_REVIEW,PROTECTED]:old.bind(pins,p)
    old.verify(pins); protected = read(PROTECTED)
    require(len(protected)==21444, 'Complete protected manifest required')
    old.verify(protected)
    require(CENTRAL.read_bytes()==before, 'Central changed during source registration')
    OUT.mkdir()
    dump(OUT/'SOURCE_PLAN.json',dict(schema='monthly_ownsector238_source_admission_v1',status='REGISTERED_SOURCE_ONLY',created_utc=now(),
        runtime=old.ev.runtime(),context_path=str(CONTEXT),context_sha256=CONTEXT_SHA,
        amendment_path=str(AMENDMENT),amendment_sha256=AMENDMENT_SHA,input_sha256=pins,
        raw_source_hashes=raw,calendar_sha256=sha(CALENDAR),mapping_source_sha256=MAPPING_SHA,
        protected_manifest=dict(path=str(PROTECTED),sha256=sha(PROTECTED),count=21444),
        prior_plan_sha256=sha(PRIOR/'PLAN.json'),prior_completion_sha256=sha(PRIOR/'COMPLETION.json'),
        source_binding=provenance,physical_prefixes=list(PREFIXES),source_only=True,strategy_trials=0,
        cache_access=dict(expected_members=27,decoded_members=['eligible'],
            protection='whole DATA.npz SHA; other26 members never decoded; no per-array fingerprint claim'),
        actual_source_audit_before_registration=False,actual_strategy_math_computed=False,
        central_modified=False,observers_modified=False,strict_pit=False))
    require(CENTRAL.read_bytes()==before, 'Central changed during source registration')
    print(json.dumps(dict(status='REGISTERED_SOURCE_ONLY',plan_sha256=sha(OUT/'SOURCE_PLAN.json'))),flush=True)


def gate(expected):
    require(isinstance(expected,str) and len(expected)==64 and sha(OUT/'SOURCE_PLAN.json')==expected, 'Explicit SOURCE_PLAN SHA mismatch')
    plan = read(OUT/'SOURCE_PLAN.json'); old = previous()
    require(plan.get('schema')=='monthly_ownsector238_source_admission_v1' and plan.get('status')=='REGISTERED_SOURCE_ONLY'
        and plan.get('context_path')==str(CONTEXT) and plan.get('context_sha256')==CONTEXT_SHA==sha(CONTEXT)
        and plan.get('amendment_path')==str(AMENDMENT) and plan.get('amendment_sha256')==AMENDMENT_SHA==sha(AMENDMENT), 'Source plan/context mismatch')
    require(plan.get('runtime')==old.ev.runtime() and plan.get('source_only') is True and plan.get('strategy_trials')==0
        and plan.get('physical_prefixes')==list(PREFIXES), 'Source-only runtime/scope mismatch')
    old.verify(plan['input_sha256']); old.verify(plan['raw_source_hashes'])
    require(plan['raw_source_hashes']==read(LOCAL/'PLAN.json')['raw_source_hashes'] and len(plan['raw_source_hashes'])==64, 'Raw map mismatch')
    require(plan['calendar_sha256']==sha(CALENDAR) and plan['mapping_source_sha256']==MAPPING_SHA, 'Calendar/mapper pin mismatch')
    require(plan['prior_plan_sha256']==sha(PRIOR/'PLAN.json') and plan['prior_completion_sha256']==sha(PRIOR/'COMPLETION.json'), 'Prior closure changed')
    require(plan['protected_manifest']==dict(path=str(PROTECTED),sha256=sha(PROTECTED),count=21444), 'Protected manifest pin mismatch')
    return plan


def require_check(check, report):
    import ownsector238_source_independent as independent
    require(all(check.get(k) is True for k in independent.FLAGS), 'Independent source audit flags incomplete')
    require(check.get('counts')==report['counts'] and check.get('status')==report['status'] and check.get('admitted') is report['admitted'], 'Independent source conclusions differ')
    require(check.get('checked_events')==len(report['events']) and check.get('checked_required_records')==len(report['records'])
        and check.get('checked_endpoint_records')==report['counts']['security_benchmark_endpoint_checks']+report['counts']['market_endpoint_checks'], 'Independent scope differs')
    require(check.get('mapping_source_sha256')==MAPPING_SHA and check.get('numerical_tolerance')==0
        and check.get('producer_modules_imported') is False, 'Independent mapping/precision differs')
    for k in ('momentum_computed','eligibility_changed','rankings_computed','orders_computed','returns_computed'):
        require(check.get(k) is False, 'Strategy math forbidden: '+k)
    require(type(check.get('model_fits')) is int and check['model_fits']==0, 'No model fits permitted')


def audit(expected):
    plan = gate(expected)
    require(not (OUT/'REPORT.json').exists() and not (OUT/'COMPLETION.json').exists(), 'Actual source audit is write-once')
    old = previous(); before = CENTRAL.read_bytes(); protected = read(PROTECTED)
    require(len(protected)==21444, 'Full archive required'); old.verify(protected)
    import ownsector238_source_independent as independent
    meta = read(LOCAL/'META.json'); cache_sha = sha(LOCAL/'DATA.npz')
    with np.load(LOCAL/'DATA.npz',allow_pickle=False) as cache:
        members = list(cache.files)
        require(len(members)==len(set(members))==27 and 'eligible' in members, 'Original27 cache members required')
        eligible = cache['eligible'].copy()
    require(eligible.dtype==bool and eligible.shape==(122,49) and len(meta['dates'])==122
        and all(len(ss)==49 for ss in meta['symbols']), 'Original eligibility/meta axes differ')
    eligible_bytes = eligible.tobytes()
    # Import only defines frozen functions; never call study.load or scope.load.
    study = old.local()
    source,info = study.arch.old.panel_load(meta['etfs']); raw = study.raw_source_hashes(info)
    require(raw==plan['raw_source_hashes'], 'Loaded raw map differs'); old.verify(raw)
    panel = {'adj':source['adj'],'sectors':source['sectors']}
    mapper = study.ex.t.strict.benchmark_map
    require(sha(Path(study.ex.t.strict.__file__))==MAPPING_SHA, 'Loaded mapper differs')
    calendar = read(CALENDAR)['sessions']
    result = inspect_sources(panel,meta,calendar,eligible,mapper)
    check = independent.check_sources(panel,meta,calendar,eligible,mapper,result); require_check(check,result)
    prefix_checks = []
    for n in PREFIXES:
        pp,mm,cc,ee = physical_prefix(panel,meta,calendar,eligible,n)
        rr = inspect_sources(pp,mm,cc,ee,mapper)
        ic = independent.check_sources(pp,mm,cc,ee,mapper,rr); require_check(ic,rr)
        require(rr['events']==result['events'][:n]
            and rr['records']==[x for x in result['records'] if x['query_index']<n]
            and rr['gaps']==[x for x in result['gaps'] if x['query_index']<n], 'Physical source prefix differs')
        logical = inspect_sources(panel,meta,calendar,eligible,mapper,limit=n)
        require(logical==rr, 'Physical/logical source prefix counts differ')
        prefix_checks.append(dict(prefix=n,source_rows={k:len(v) for k,v in pp.items()},
            source_cutoff=mm['dates'][-1] if n else None,calendar_rows=len(cc),calendar_last_session=cc[-1] if n else None,
            counts=rr['counts'],status=rr['status'],independent=ic))
    require(eligible.tobytes()==eligible_bytes and sha(LOCAL/'DATA.npz')==cache_sha, 'Original cache archive/eligible changed')
    gate(expected); old.verify(protected)
    require(CENTRAL.read_bytes()==before, 'Central changed during source audit')
    final = dict(result,created_utc=now(),plan_sha256=expected,context_sha256=CONTEXT_SHA,
        input_sha256=plan['input_sha256'],raw_source_hashes=raw,independent=check,
        physical_prefixes=prefix_checks,original_cache_archive_unchanged=True,original_array_count=27,
        cached_member_names=members,decoded_cache_members=['eligible'],original_cache_sha256=cache_sha,
        other_cached_arrays_decoded=False,per_array_fingerprints_computed=False,protected_files=21444,
        central_modified=False,observers_modified=False,limitations=read(CONTEXT)['limitations'])
    dump(OUT/'REPORT.json',final)
    dump(OUT/'COMPLETION.json',dict(status='COMPLETED_SOURCE_AUDIT_INDEPENDENTLY_VERIFIED',created_utc=now(),
        plan_sha256=expected,context_sha256=CONTEXT_SHA,amendment_sha256=AMENDMENT_SHA,artifacts_sha256={'REPORT.json':sha(OUT/'REPORT.json')},
        admitted=result['admitted'],admission_status=result['status'],counts=result['counts'],source_only=True,
        strategy_trials=0,actual_strategy_math_computed=False,protected_files=21444,
        central_modified=False,observers_modified=False,strict_pit=False,physical_prefixes=list(PREFIXES)))
    print(json.dumps(dict(status='COMPLETED_SOURCE_AUDIT_INDEPENDENTLY_VERIFIED',admitted=result['admitted'],
        counts=result['counts'],report_sha256=sha(OUT/'REPORT.json'),completion_sha256=sha(OUT/'COMPLETION.json'))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('test','register','audit'))
    parser.add_argument('--plan-sha256')
    args=parser.parse_args()
    if args.phase=='test':test()
    elif args.phase=='register':register()
    else:
        require(args.plan_sha256 is not None, 'audit requires --plan-sha256')
        audit(args.plan_sha256)
