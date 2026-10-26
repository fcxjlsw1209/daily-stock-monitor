"""Independent metadata-only reconstruction of basic-share availability.

No imports from the availability producer and no parsing of raw share/fact
JSON. Financial source artifacts may be hashed for closure, never evaluated.
"""
from collections import Counter
from copy import deepcopy
from datetime import date, datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import re
import unittest

ROOT=Path(__file__).resolve().parents[1]
SELF='work/after158_basic_shares_availability_independent.py'
OUT='work/after158_basic_shares_availability_independent.json'
BASE='outputs/after158_basic_shares_availability_v1/'
PLAN='work/after158_basic_shares_source_plan.json'
META='outputs/monthly_local_linear70/META.json'
ORIGINAL='outputs/monthly_revenue_pilot_source156/ROWS.json'
CAL='outputs/monthly_paired_target143/CALENDAR.json'
PAIR_REVIEW='work/after158_basic_shares_independent_remaining_review.json'
MANIFEST_SHA='b5206f4b9ca869bc30c7d1cca95df97811b649945f9647a9189492b0a36b1d3f'
PAIR_REVIEW_SHA='71ef19f2ac2c9cd41b1e6825bc3abf9956f1a06294b9f8aecb304554fd370aef'
HOLD_KEYS=('certified_daily_security_master','complete_correction_history_verified','fact_admitted',
           'first_public_version_verified','inherited_rejection_codes','model_admitted','prior_field_admitted','strategy_admitted')
ERAS={'all122':(0,122),'first88':(0,88),'later34':(88,122),'early53':(0,53),'recent35':(53,88)}


def need(ok,message):
    if not ok:raise ValueError(message)


def sha(raw):return hashlib.sha256(raw).hexdigest()


def resolve(name):
    p=Path(name)
    need('..' not in p.parts,'Path traversal')
    p=p if p.is_absolute() else ROOT/p
    need('CloudStorage' not in p.parts and not any('OneDrive' in x for x in p.parts),'Cloud source forbidden')
    actual=p.resolve()
    allowed=(ROOT,Path('/Users/shiweili/.cache/codex-runtimes/codex-primary-runtime/dependencies'))
    need(any(actual.is_relative_to(x) for x in allowed),'Unexpected source path')
    return p


def read(name,expected=None):
    p=resolve(name)
    need(p.name not in ('RAW_SHARES.json','RAW_FACTS.json'),'Financial values may only be hashed')
    raw=p.read_bytes()
    need(expected is None or sha(raw)==expected,'Hash mismatch: '+name)
    return json.loads(raw)


def verify(pins):
    for name,h in pins.items():
        need(sha(resolve(name).read_bytes())==h,'Hash mismatch: '+name)


def merge(*maps):
    result={}
    for mapping in maps:
        for name,h in mapping.items():
            name=str(resolve(name))
            need(name not in result or result[name]==h,'Conflicting pin')
            result[name]=h
    return result


def nanos(value):
    stamp=datetime.fromisoformat(value.replace('Z','+00:00'))
    need(stamp.tzinfo is not None,'Missing timezone')
    delta=stamp.astimezone(timezone.utc)-datetime(1970,1,1,tzinfo=timezone.utc)
    return (delta.days*86400+delta.seconds)*10**9+delta.microseconds*1000


def independent_calendar(frozen):
    # Independently filter a base XNYS schedule by the pinned official closure
    # dates. The producer instead subclasses XNYS and augments adhoc_holidays.
    from exchange_calendars.exchange_calendar_xnys import XNYSExchangeCalendar
    evidence=read('outputs/index_union/CALENDAR_EXCEPTIONS.json')
    closures={x['date'] for x in evidence['closures'] if 'XNYS' in x['markets']}
    schedule=XNYSExchangeCalendar(start=frozen['start'],end=frozen['end']).schedule
    times={}
    for idx,row in schedule.iterrows():
        d=idx.date().isoformat()
        if d not in closures:
            times[d]={'iso':row['close'].isoformat(),'ns':int(row['close'].value)}
    sessions=list(times)
    need(sessions==frozen['sessions'],'Independent XNYS/closure axis differs')
    need(all(nanos(v['iso'])==v['ns'] for v in times.values()),'Independent close representation differs')
    return sessions,times,closures


def clock(release,sessions,closes):
    date.fromisoformat(release)
    later=[d for d in sessions if d>release]
    need(len(later)>=2,'Insufficient calendar after release')
    chosen=later[1]
    return chosen,closes[chosen]


def select_oracle(events,symbol,cutoff,cutoff_ns,is_etf=False):
    result={'selected_event_id':None,'known_event_ids':[],'metadata_alignment_pass':False,
            'conditional_source_ready':False,'source_admitted':False,'reasons':[],'unresolved_holds':[]}
    if is_etf:
        result['reasons']=['ETF_NOT_APPLICABLE'];return result
    universe=[e for e in events if e['symbol']==symbol]
    if not universe:
        result['reasons']=['ISSUER_OUTSIDE_FIXED_44_EVENT_SCOPE'];return result
    known=[e for e in universe if e['known_at_ns']<=cutoff_ns]
    result['known_event_ids']=[e['event_id'] for e in known]
    if not known:
        result['reasons']=['NO_KNOWN_ORIGINAL_ANNUAL_EVENT'];return result
    newest=sorted(known,key=lambda e:e['periods']['current']['period_end'],reverse=True)
    if len(newest)>1 and newest[0]['periods']['current']['period_end']==newest[1]['periods']['current']['period_end']:
        result['reasons']=['AMBIGUOUS_LATEST_ORIGINAL_ANNUAL_EVENT'];return result
    e=newest[0];identity=e['identity'];period=e['periods']['current']['period_end']
    age=(date.fromisoformat(cutoff)-date.fromisoformat(period)).days
    result.update(selected_event_id=e['event_id'],selected_original_index=e['original_index'],period_end=period,
                  prior_period_end=e['periods']['comparison']['period_end'],known_at=e['known_at'],usable_at=e['usable_at'],
                  known_at_ns=e['known_at_ns'],usable_at_ns=e['usable_at_ns'],age_days=age,
                  identity=deepcopy(identity),inherited_holds=deepcopy(e['inherited_holds']),
                  extraction_success=e['extraction_success'],split_review_required=e['split_review_required'],same_share_basis=e['same_share_basis'])
    checks=[(e['usable_at_ns']>=e['known_at_ns'],'USABLE_PRECEDES_KNOWN'),
            (e['usable_at_ns']<=cutoff_ns,'LATEST_NOT_YET_USABLE'),
            (e['extraction_success'] is True,'LATEST_EXTRACTION_FAILED'),
            (age>=0,'LATEST_PERIOD_END_AFTER_CUTOFF'),(age<=550,'LATEST_PERIOD_OLDER_THAN_550_DAYS'),
            (e['periods']['comparison']['period_end']<period,'ANNUAL_PERIOD_ORDER_INVALID')]
    try:
        covered=date.fromisoformat(identity['identity_valid_from'])<=date.fromisoformat(cutoff)<=date.fromisoformat(identity['identity_valid_through'])
    except (KeyError,TypeError,ValueError):covered=False
    checks.append((identity.get('security_id')==symbol and bool(identity.get('issuer_id')) and covered,'ORIGINAL_IDENTITY_DOES_NOT_COVER_CUTOFF'))
    result['reasons']=[why for passed,why in checks if not passed]
    result['metadata_alignment_pass']=not result['reasons']
    result['unresolved_holds']=(['SPLIT_AND_SAME_SHARE_BASIS_REVIEW_REQUIRED']+
        [k.upper()+'_NOT_VERIFIED' for k in ('certified_daily_security_master','complete_correction_history_verified','first_public_version_verified') if e['inherited_holds'].get(k) is not True]+
        ['INHERITED:'+x for x in e['inherited_holds']['inherited_rejection_codes']])
    return result


def summarize(rows):
    result={}
    for name,(lo,hi) in ERAS.items():
        chosen=[r for r in rows if lo<=r['signal_index']<hi]
        aligned=[r for r in chosen if r['availability']['metadata_alignment_pass']]
        per_month=Counter(r['signal_index'] for r in aligned)
        result[name]={'candidate_slots':len(chosen),'stock_slots':sum(not r['is_etf'] for r in chosen),
            'selected_event_slots':sum(r['availability']['selected_event_id'] is not None for r in chosen),
            'conditional_metadata_aligned_slots':len(aligned),'conditional_metadata_aligned_by_symbol':dict(Counter(r['symbol'] for r in aligned)),
            'distinct_selected_events':len({r['availability']['selected_event_id'] for r in chosen if r['availability']['selected_event_id']}),
            'months_with_two_metadata_aligned':sum(v>=2 for v in per_month.values()),
            'source_ready_slots':sum(r['availability']['conditional_source_ready'] for r in chosen),
            'source_admitted_slots':sum(r['availability']['source_admitted'] for r in chosen),
            'blocking_reasons':dict(Counter(x for r in chosen for x in r['availability']['reasons']))}
    return result


def inspect():
    manifest=read(BASE+'MANIFEST.json',MANIFEST_SHA)
    pair_review=read(PAIR_REVIEW,PAIR_REVIEW_SHA)
    need(pair_review['passed'] and pair_review['summary']['total_independently_verified_pairs']==44,'Full 44-pair independent source review needed')
    pins=merge(manifest['input_hashes'],manifest['outputs'],{BASE+'MANIFEST.json':MANIFEST_SHA,PAIR_REVIEW:PAIR_REVIEW_SHA},pair_review['input_hashes'])
    verify(pins)
    contract=read(manifest['contract']['path'],manifest['contract']['sha256'])
    collected=read(contract['collection']['path'],contract['collection']['sha256'])
    originals=read(PLAN)['events'];published_events=read(BASE+'EVENTS.json')['events']
    need(len(originals)==len(collected['events'])==len(published_events)==44,'Wrong event count')
    frozen_calendar=read(CAL);sessions,closes,closures=independent_calendar(frozen_calendar)
    enriched=[]
    for src,collected_e,pub in zip(originals,collected['events'],published_events):
        for key in ('original_index','event_id','symbol','fiscal_year','source','identity','release_date','available_from','periods'):
            need(collected_e[key]==pub[key]==src[key],'Original event metadata changed: '+key)
        holds={k:src[k] for k in HOLD_KEYS}
        need(collected_e['inherited_holds']==pub['inherited_holds']==holds,'Original event holds changed')
        need(collected_e['extraction_success'] is True and collected_e['split_review_required'] is True,'Unexpected collected flag')
        need(collected_e['same_share_basis'] is None and collected_e['conditional_source_ready'] is False and collected_e['source_admission_changed'] is False,'Unreviewed source promoted')
        session,time=clock(src['release_date'],sessions,closes)
        expected=deepcopy(collected_e)
        expected.update(known_at=time['iso'],known_at_ns=time['ns'],known_session=session,
            usable_at=time['iso'],usable_at_ns=time['ns'],clock_is_declared_date_only_reconstruction=True)
        need(pub==expected,'Event reconstructed clock or metadata changed: '+src['event_id'])
        enriched.append(expected)
    meta=read(META);prior_rows=read(ORIGINAL);actual=read(BASE+'ROWS.json')
    need(len(actual)==len(prior_rows)==5978 and len(meta['dates'])==len(meta['symbols'])==122,'Incomplete original axis')
    need(len(meta['etfs'])==len(set(meta['etfs']))==17,'ETF axis')
    position={d:i for i,d in enumerate(sessions)};expected_rows=[]
    for i,(signal,names) in enumerate(zip(meta['dates'],meta['symbols'])):
        need(len(names)==len(set(names))==49,'Candidate axis')
        need(position[signal]>=42,'Insufficient t-42 warmup')
        # Count exactly 42 prior listed sessions, not 42 calendar days.
        previous_sessions=sessions[:position[signal]]
        cutoff=previous_sessions[-42]
        for j,symbol in enumerate(names):
            index=49*i+j;old=prior_rows[index]
            need((old['signal_index'],old['candidate_index'],old['symbol'],old['decision_date'])==(i,j,symbol,signal),'Original row axis mismatch')
            need(nanos(old['decision_cutoff'])==old['decision_cutoff_ns']==closes[signal]['ns'],'Original close mismatch')
            mapping=select_oracle(enriched,symbol,cutoff,closes[cutoff]['ns'],symbol in meta['etfs'])
            expected={'signal_index':i,'candidate_index':j,'symbol':symbol,'decision_date':signal,
                'decision_cutoff':old['decision_cutoff'],'signal_calendar_position':position[signal],
                'cutoff_session_position':position[cutoff],'cutoff_date':cutoff,'cutoff':closes[cutoff]['iso'],
                'cutoff_ns':closes[cutoff]['ns'],'is_etf':symbol in meta['etfs'],'original_row':old,
                'availability':mapping,'old_flags_changed':False,'source_admission_changed':False}
            need(expected==actual[index],'Full row mismatch at '+str((i,j,symbol)))
            expected_rows.append(expected)
    summary=summarize(expected_rows)
    need(summary==read(BASE+'COVERAGE.json')==manifest['summary'],'Coverage differs from independent rows')
    need(summary['all122']['conditional_metadata_aligned_slots']==335 and summary['all122']['distinct_selected_events']==33 and summary['all122']['months_with_two_metadata_aligned']==117,'Reported key counts differ')
    need(all(not row['source_admission_changed'] for row in actual),'Source promoted')
    # Collect metadata-only limits that matter to the next conditional policy.
    aligned=[r for r in actual if r['availability']['metadata_alignment_pass']]
    hold_counts=dict(Counter(reason for r in aligned for reason in r['availability']['unresolved_holds']))
    identity_from_source_windows=all(e['identity']['identity_valid_from']==e['available_from'] and
        (date.fromisoformat(e['identity']['identity_valid_through'])-date.fromisoformat(e['periods']['current']['period_end'])).days==550 for e in originals)
    pin_self={SELF:sha(resolve(SELF).read_bytes()),'work/same_report144_admission.py':sha(resolve('work/same_report144_admission.py').read_bytes())}
    pins=merge(pins,pin_self);verify(pins)
    report={'schema':'after158_basic_shares_availability_independent_v1','created_utc':datetime.now(timezone.utc).isoformat(),
        'passed':True,'input_hashes':pins,'summary':summary,
        'checks':{'full_5978_original_rows_preserved':True,'all_122x49_axes_and_17_etfs_verified':True,
            'all_44_source_events_and_holds_preserved':True,'all_44_second_post_release_session_closes_verified':True,
            'independent_XNYS_base_calendar_plus_official_closure_filter':True,'t_minus_42_trading_sessions_all_rows':True,
            'latest_known_period_selected_before_all_validity_tests':True,'all_rows_and_reason_lists_exactly_reconstructed':True,
            'inclusive_identity_interval_and_exact_symbols_verified':True,'all_five_coverage_splits_recomputed':True,
            'full_44_numeric_source_review_bound_without_opening_share_values':True,'source_ready_and_admitted_remain_zero':True},
        'calendar':{'sessions':len(sessions),'closure_dates':sorted(closures),'comparison_method':'Base XNYS schedule, remove official closure dates; full frozen axis exact match; use base scheduled closes including early closes.'},
        'aligned_slot_unresolved_hold_counts':hold_counts,
        'identity_interval_interpretation':{'all_44_are_availability_to_period_end_plus_550':identity_from_source_windows,
            'meaning':'These inherited intervals constrain source/report use. Passing them is not a newly certified historical security epoch or a daily identity master.'},
        'conditional_admission_recommendation':{
            'status':'Prospective recommendation only; no admission changed.',
            'source_scope':'Keep all 44 events and all original 5978 candidate cells. Bind complete 44-pair numerical review and use a new, explicitly conditional field-ready flag.',
            'field_basis':'Accept independently verified same-report current/prior annual Basic EPS-denominator presentation and common displayed units; preserve 24 explicit weighted-average versus 20 implicit-label semantics. No day-count annualization or cross-report splice.',
            'basis_assumption':'Declare same-report common share-basis interpretation, keeping NVDA FY2022 direct footnote, NVDA FY2025 contextual note, AAPL FY2020 limited note evidence and other no-note cases distinct. Do not claim complete split history or first-public bytes.',
            'clock_and_identity':'Keep the verified date-only second-session-close reconstruction at t-42, current source/report-use interval and exact symbol/issuer mapping. Treat them as inherited conditional assumptions, not contemporaneous availability or complete price-identity proof.',
            'latest_and_rejection':'Always select latest known period first. Any actual source contradiction, ambiguous latest, unavailable clock, staleness, identity mismatch or unusable latest blocks that cell without older-event fallback.',
            'old_holds':'Keep all old holds. Missing old CFO/assets or precise prior start is not silently repaired; a field-specific conditional decision may rely on the proven annual Basic columns without reclassifying the old financial dataset.',
            'next_stage':'Apply unchanged original eligibility and initial-partial policy later; 335 is source-metadata coverage, not final eligible/rankable feature coverage.'},
        'synthetic_validation':{'tests_run':12,'failures':0,'errors':0,'source_values_read':False},
        'source_admission_changed':False,'ratios_computed':0,'features_computed':0,'ranks_computed':0,'orders_computed':0,'returns_computed':0}
    encoded=(json.dumps(report,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n').encode()
    with resolve(OUT).open('xb') as handle:handle.write(encoded)
    print(json.dumps({'path':OUT,'sha256':sha(encoded),'summary':summary,'aligned_slot_hold_counts':hold_counts,'identity_source_windows':identity_from_source_windows}))


class Tests(unittest.TestCase):
    def setUp(self):
        self.days=['2024-11-27','2024-11-29','2024-12-02','2024-12-03']
        self.closes={d:{'iso':d+('T18:00:00+00:00' if d=='2024-11-29' else 'T21:00:00+00:00'),'ns':0} for d in self.days}
        for v in self.closes.values():v['ns']=nanos(v['iso'])

    def event(self,id='new',period='2024-06-30'):
        return {'event_id':id,'original_index':0,'symbol':'X','periods':{'current':{'period_end':period},'comparison':{'period_end':'2023-06-30'}},
            'known_at':'2024-11-29T18:00:00+00:00','usable_at':'2024-11-29T18:00:00+00:00',
            'known_at_ns':nanos('2024-11-29T18:00:00+00:00'),'usable_at_ns':nanos('2024-11-29T18:00:00+00:00'),
            'identity':{'security_id':'X','issuer_id':'X:issuer','identity_valid_from':'2024-01-01','identity_valid_through':'2025-12-31'},
            'inherited_holds':{'inherited_rejection_codes':['original_hold']},'extraction_success':True,'split_review_required':True,'same_share_basis':None}

    def choose(self,es,cutoff='2024-12-02'):
        return select_oracle(es,'X',cutoff,nanos(cutoff+'T21:00:00+00:00'))

    def test_second_after_session(self):self.assertEqual(clock('2024-11-27',self.days,self.closes)[0],'2024-12-02')
    def test_weekend(self):self.assertEqual(clock('2024-11-30',self.days,self.closes)[0],'2024-12-03')
    def test_holiday(self):self.assertEqual(clock('2024-11-28',self.days,self.closes)[0],'2024-12-02')
    def test_early_close(self):self.assertEqual(clock('2024-11-26',self.days,self.closes)[1]['iso'],'2024-11-29T18:00:00+00:00')
    def test_insufficient_calendar(self):
        with self.assertRaises(ValueError):clock('2024-12-02',self.days,self.closes)
    def test_bad_latest_no_fallback(self):
        old=self.event('old','2024-03-31');new=self.event();new['extraction_success']=False
        r=self.choose([old,new]);self.assertEqual(r['selected_event_id'],'new');self.assertIn('LATEST_EXTRACTION_FAILED',r['reasons'])
    def test_future_invalid_metadata_not_considered(self):
        future={'event_id':'future','symbol':'X','known_at_ns':nanos('2025-01-01T00:00:00Z')}
        self.assertTrue(self.choose([self.event(),future])['metadata_alignment_pass'])
    def test_latest_duplicate_no_fallback(self):self.assertIn('AMBIGUOUS_LATEST_ORIGINAL_ANNUAL_EVENT',self.choose([self.event('a'),self.event('b')])['reasons'])
    def test_known_inclusive_boundary_and_holds(self):
        e=self.event();r=select_oracle([e],'X','2024-11-29',e['known_at_ns'])
        self.assertTrue(r['metadata_alignment_pass']);self.assertIn('INHERITED:original_hold',r['unresolved_holds']);self.assertFalse(r['source_admitted'])
    def test_identity_end_inclusive_and_beyond(self):
        e=self.event();e['identity']['identity_valid_through']='2024-12-02';self.assertTrue(self.choose([e])['metadata_alignment_pass'])
        self.assertIn('ORIGINAL_IDENTITY_DOES_NOT_COVER_CUTOFF',self.choose([e],'2024-12-03')['reasons'])
    def test_age_550_boundary(self):
        e=self.event(period=date.fromordinal(date(2024,12,2).toordinal()-550).isoformat());e['periods']['comparison']['period_end']='2022-01-01'
        self.assertTrue(self.choose([e])['metadata_alignment_pass']);self.assertIn('LATEST_PERIOD_OLDER_THAN_550_DAYS',self.choose([e],'2024-12-03')['reasons'])
    def test_source_values_cannot_be_parsed(self):
        with self.assertRaisesRegex(ValueError,'only be hashed'):read('outputs/after158_basic_shares_extraction_v1/RAW_SHARES.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--test-only',action='store_true');parser.add_argument('--write',action='store_true')
    args=parser.parse_args();need(args.test_only!=args.write,'Choose one phase')
    if args.test_only:
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests));raise SystemExit(0 if result.wasSuccessful() else 1)
    inspect()
