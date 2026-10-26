"""Predeclared source-availability feasibility, with no financial ratios or returns."""
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
import copy
import io
import json
import sys
import unittest
import numpy as np
import pandas as pd

import liquidity162_audit_v2 as src

OUT = src.SOURCE_OUT/'feasibility'
TICKERS = ('MSFT','NVDA')
SPLITS = dict(early=(0,53),recent=(53,88),train=(0,88),later=(88,122),full=(0,122))
CODE = ('work/liquidity162_coverage.py','work/liquidity162_coverage_independent.py')
SOURCE_FILES = ('SOURCE_PLAN.json','v2/EXTRACTION_CODE.json','v2/EXTRACTED.json',
                'v2/INDEPENDENT_REVIEW.json','v2/COMPLETION.json')
BASE_PINS = {
    'outputs/monthly_local_linear70/DATA.npz':'be846073cb55a775c1382d6e584caa34dc1eaa3b46e7f7815a5c2f63f5e4de3f',
    'outputs/monthly_local_linear70/META.json':'3b46968f486e88fa170ac01f120dd5ef74ebd5a56243c6991c5a4fea506076f4',
}

def clock(value):
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        raise ValueError('Timezone-aware source clock required')
    return int(stamp.value)

def calculate(records,meta,eligible,closes):
    dates,names = meta['dates'],meta['symbols']
    if np.asarray(eligible).dtype != np.dtype(bool) or len(dates)!=len(names) or len(names)!=len(eligible):
        raise ValueError('Invalid original eligibility axes')
    rows=[]
    for i,day in enumerate(dates):
        if len(names[i])!=len(eligible[i]): raise ValueError('Column alignment')
        cutoff=int(closes[day])
        for ticker in TICKERS:
            indices=[j for j,name in enumerate(names[i]) if name==ticker]
            if len(indices)>1: raise ValueError('Duplicate issuer in original pool')
            j=indices[0] if indices else None
            known=[r for r in records if r['ticker']==ticker and clock(r['availability']['known_by_at'])<=cutoff]
            latest=None
            if known:
                newest=max(clock(r['availability']['known_by_at']) for r in known)
                top=[r for r in known if clock(r['availability']['known_by_at'])==newest]
                if len(top)!=1: raise ValueError('Ambiguous latest report')
                latest=top[0]
            row=dict(index=i,date=day,cutoff_ns=cutoff,ticker=ticker,pool_index=j,present=j is not None,
                eligible=bool(eligible[i][j]) if j is not None else False,selected_source=None,
                known_by_at=None,period_end=None,age_days=None,status=None,source_usable=False,
                actionable=False,reason='no_known_source')
            if latest is not None:
                availability=latest['availability']
                if availability['status']!='ready' or availability['known_by_is_actual_publication'] is not False:
                    raise ValueError('Unverified availability record')
                age=(date.fromisoformat(day)-date.fromisoformat(latest['fiscal_period_end'])).days
                reason=('latest_rejected' if latest['status']!='candidate' else
                        'fiscal_period_future' if age<0 else 'stale' if age>400 else 'usable')
                row.update(selected_source=f"{ticker}:{latest['fiscal_year']}:{latest['source_binding']['raw_sha256']}",
                    known_by_at=availability['known_by_at'],period_end=latest['fiscal_period_end'],
                    age_days=age,status=latest['status'],reason=reason,source_usable=reason=='usable',
                    actionable=i>0 and reason=='usable' and row['eligible'])
            rows.append(row)
    return rows,summarize(rows)

def summarize(rows):
    result={}
    for name,(start,end) in SPLITS.items():
        subset=[r for r in rows if start<=r['index']<end]
        per={}
        for ticker in TICKERS:
            r=[x for x in subset if x['ticker']==ticker]
            per[ticker]=dict(source_usable=sum(x['source_usable'] for x in r),
                pool_usable=sum(x['source_usable'] and x['present'] for x in r),
                actionable=sum(x['actionable'] for x in r),
                reports=sorted({x['selected_source'] for x in r if x['actionable']}))
        pairs=[]
        for i in sorted({r['index'] for r in subset}):
            pair=[r for r in subset if r['index']==i]
            if len(pair)!=2: raise ValueError('Incomplete issuer pair')
            pairs.append(pair)
        result[name]=dict(issuers=per,
            both_source_usable=sum(all(x['source_usable'] for x in p) for p in pairs),
            both_pool_usable=sum(all(x['source_usable'] and x['present'] for x in p) for p in pairs),
            both_actionable=sum(all(x['actionable'] for x in p) for p in pairs),
            overlap_reports={t:sorted({x['selected_source'] for p in pairs if all(z['actionable'] for z in p)
                                      for x in p if x['ticker']==t}) for t in TICKERS},
            reason_counts=dict(Counter(x['reason'] for x in subset)))
    return result

def gate(summary):
    checks={'train_overlap_24':summary['train']['both_actionable']>=24,
            'early_overlap_8':summary['early']['both_actionable']>=8,
            'recent_overlap_8':summary['recent']['both_actionable']>=8}
    for era in ('early','recent'):
        for ticker in TICKERS:
            checks[f'{era}_{ticker}_two_distinct_reports']=len(summary[era]['overlap_reports'][ticker])>=2
    return dict(passed=all(checks.values()),checks=checks,
                failed_checks=[k for k,v in checks.items() if not v])

def write(name,value):
    with (OUT/name).open('x') as f:
        json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')

def source_closure():
    source_plan=src.bound()
    c=json.loads((src.OUT/'COMPLETION.json').read_text())
    if c['status']!='source_only_audit_complete' or c['source_plan_sha256']!=src.PLAN_SHA:
        raise ValueError('Source audit incomplete')
    pins=dict(source_plan['source_hashes'])
    pins[str(src.SOURCE_OUT/'SOURCE_PLAN.json')]=src.PLAN_SHA
    for name,h in c['evidence_hashes'].items():
        p=src.OUT/name
        src.pinned(p,h)
        pins[str(p)]=h
    code=json.loads(src.pinned(src.OUT/'EXTRACTION_CODE.json',c['evidence_hashes']['EXTRACTION_CODE.json']))
    for p,h in code['code_hashes'].items():
        absolute=str(src.ROOT/p)
        if absolute in pins and pins[absolute]!=h:raise ValueError('Conflicting dependency')
        pins[absolute]=h
    review=json.loads(src.pinned(src.OUT/'INDEPENDENT_REVIEW.json',c['evidence_hashes']['INDEPENDENT_REVIEW.json']))
    if (review['passed'] is not True or len(review['records'])!=24
            or not all(r['verified'] for r in review['records'])
            or review['extraction_sha256']!=c['evidence_hashes']['EXTRACTED.json']
            or review['code_binding_sha256']!=c['evidence_hashes']['EXTRACTION_CODE.json']):
        raise ValueError('Source completion/review bindings disagree')
    return pins

def register():
    pins=source_closure()
    for p,h in BASE_PINS.items():
        src.pinned(src.ROOT/p,h)
        pins[str(src.ROOT/p)]=h
    import liquidity162_coverage_independent as independent
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(Synthetic),
                            unittest.defaultTestLoader.loadTestsFromModule(independent)])
    before={p:src.digest(src.ROOT/p) for p in CODE}
    stream=io.StringIO();test=unittest.TextTestRunner(stream=stream).run(suite)
    print(stream.getvalue())
    if not test.wasSuccessful() or before!={p:src.digest(src.ROOT/p) for p in CODE}:
        raise ValueError('Synthetic validation failed or code changed')
    paths=[src.SOURCE_OUT/p for p in SOURCE_FILES]+[src.ROOT/p for p in CODE]+[
        src.ROOT/'outputs/monthly_local_linear70/META.json',
        src.ROOT/'outputs/monthly_local_linear70/DATA.npz',
        src.ROOT/'outputs/monthly_paired_target143/CALENDAR.json']
    OUT.mkdir(exist_ok=True)
    for p in paths:
        h=src.digest(p)
        if str(p) in pins and pins[str(p)]!=h:raise ValueError('Changed previously bound dependency')
        pins[str(p)]=h
    write('PLAN.json',dict(created_utc=src.now(),schema='liquidity162_coverage_v1',
        hashes=pins,runtime=src.runtime(),
        source_records=24,scope='Source availability only; prospective decision on two-issuer rank feasibility.',
        cutoff='Actual XNYS decision-date close; no financial t-minus42 shift. Momentum rules unmodified.',
        selection='Latest known report across all records, rejected included; no older fallback.',
        freshness='0 through400 calendar days from actual fiscal period end, inclusive.',
        original_universe_and_eligibility_only=True,initial_partial_excluded_from_actionable_support=True,
        gate='Training actionable pair overlap>=24; early53>=8; recent35>=8; each issuer>=2 distinct usable annual releases in each training-era overlap.',
        later34_diagnostic_only=True,tests_run=test.testsRun,test_log=stream.getvalue(),
        actual_coverage_maps_before_registration=0,financial_ratios_computed=0,returns_computed=0,
        failure_action='Do not admit this two-issuer rank design or relax these thresholds after observing coverage.',
        limitation='A small feasibility gate is not statistical power or general financial-factor validation.'))
    print(json.dumps({'plan_sha256':src.digest(OUT/'PLAN.json'),'tests':test.testsRun}))

def run(expected):
    if src.digest(OUT/'PLAN.json')!=expected: raise ValueError('Exact plan digest required')
    plan=json.loads((OUT/'PLAN.json').read_text())
    if plan['runtime']!=src.runtime(): raise ValueError('Runtime changed')
    for p,h in plan['hashes'].items():src.pinned(p,h)
    closure=source_closure()
    if any(plan['hashes'].get(p)!=h for p,h in closure.items()):raise ValueError('Incomplete source dependency closure')
    def read_bound(path):return src.pinned(path,plan['hashes'][str(path)])
    extracted=json.loads(read_bound(src.OUT/'EXTRACTED.json'))['records']
    meta=json.loads(read_bound(src.ROOT/'outputs/monthly_local_linear70/META.json'))
    if len(meta['dates'])!=122 or meta['early_months']!=53 or meta['train_months']!=88:
        raise ValueError('Original monthly axes changed')
    with np.load(io.BytesIO(read_bound(src.ROOT/'outputs/monthly_local_linear70/DATA.npz')),allow_pickle=False) as arrays:
        eligible=arrays['eligible']
    if eligible.shape!=(122,49): raise ValueError('Original eligibility shape')
    calendar_record=json.loads(read_bound(src.ROOT/'outputs/monthly_paired_target143/CALENDAR.json'))
    sys.path.insert(0,str(src.ROOT/'outputs/monthly_earnings_repair'))
    from research_calendar import market_calendar
    calendar=market_calendar(calendar_record['start'],calendar_record['end'])
    if [str(s.date()) for s in calendar.schedule.index]!=calendar_record['sessions']:
        raise ValueError('Registered exchange calendar changed')
    closes={str(s.date()):int(pd.Timestamp(close).value) for s,close in calendar.schedule['close'].items()}
    rows,summary=calculate(extracted,meta,eligible,closes)
    import liquidity162_coverage_independent as independent
    alt_rows,alt_summary=independent.calculate(extracted,meta,eligible,closes)
    if json.dumps([rows,summary],sort_keys=True)!=json.dumps([alt_rows,alt_summary],sort_keys=True):
        raise ValueError('Independent coverage mismatch')
    # Prefixes recompute latest selection, rather than truncate a full-result map.
    for n in range(123):
        submeta=dict(meta,dates=meta['dates'][:n],symbols=meta['symbols'][:n])
        subrows,_=calculate(extracted,submeta,eligible[:n],closes)
        if subrows!=rows[:2*n]:raise ValueError('Coverage prefix failure')
    outcome=gate(summary)
    write('RESULTS.json',dict(created_utc=src.now(),plan_sha256=expected,rows=rows,summary=summary,
        gate=outcome,independent_exact_match=True,recomputed_prefixes=123,
        strategy_trials_added=0,financial_ratios_computed=0,orders_computed=0,returns_computed=0,
        decision='eligible_for_separate_preregistration' if outcome['passed'] else 'do_not_admit_two_issuer_rank_design'))
    print(json.dumps({'rows':len(rows),'gate':outcome,
        'overlap':{k:v['both_actionable'] for k,v in summary.items()}}))

class Synthetic(unittest.TestCase):
    def setup_data(self):
        def r(year,status='candidate',known='2020-02-01T21:00:00+00:00',end='2019-12-31'):
            return dict(ticker='MSFT',fiscal_year=year,status=status,fiscal_period_end=end,
                source_binding={'raw_sha256':str(year)},availability=dict(known_by_at=known,
                    status='ready',known_by_is_actual_publication=False))
        self.r=r
        return [r(2019)],dict(dates=['2020-02-28','2020-03-31'],symbols=[['MSFT','NVDA']]*2),np.ones((2,2),dtype=bool),{
            '2020-02-28':clock('2020-02-28T21:00Z'),'2020-03-31':clock('2020-03-31T20:00Z')}
    def test_initial_and_future(self):
        records,meta,eligible,closes=self.setup_data()
        rows,_=calculate(records,meta,eligible,closes)
        self.assertFalse(rows[0]['actionable']);self.assertTrue(rows[2]['actionable'])
        future=self.r(2020,known='2020-04-01T20:00Z')
        self.assertEqual(rows,calculate(records+[future],meta,eligible,closes)[0])
    def test_latest_rejected_blocks_old(self):
        records,meta,eligible,closes=self.setup_data()
        records.append(self.r(2020,'rejected','2020-03-01T20:00Z'))
        rows,_=calculate(records,meta,eligible,closes)
        self.assertEqual(rows[2]['reason'],'latest_rejected');self.assertFalse(rows[2]['actionable'])
    def test_stale_and_exact_clock(self):
        records,meta,eligible,closes=self.setup_data()
        records[0]['fiscal_period_end']='2018-01-01'
        self.assertEqual(calculate(records,meta,eligible,closes)[0][0]['reason'],'stale')
        records[0]['fiscal_period_end']='2019-12-31'
        records[0]['availability']['known_by_at']='2020-02-28T21:00:00+00:00'
        self.assertTrue(calculate(records,meta,eligible,closes)[0][0]['source_usable'])
    def test_duplicate_and_naive_rejected(self):
        records,meta,eligible,closes=self.setup_data()
        with self.assertRaises(ValueError):calculate(records+records,meta,eligible,closes)
        records[0]['availability']['known_by_at']='2020-02-01'
        with self.assertRaises(ValueError):calculate(records,meta,eligible,closes)
    def test_future_eligible_does_not_change_prefix(self):
        records,meta,eligible,closes=self.setup_data()
        before=calculate(records,meta,eligible,closes)[0][:2]
        eligible[1,:]=False
        self.assertEqual(before,calculate(records,meta,eligible,closes)[0][:2])

if __name__=='__main__':
    if sys.argv[1]=='register':register()
    elif sys.argv[1]=='run':run(sys.argv[2])
    else:raise ValueError('register or run required')
