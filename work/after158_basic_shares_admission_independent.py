"""Independent source-only audit; no producer imports or strategy calculations.

Reuses the earlier independent availability/calendar oracle, never its producer.
The only numerical work compares literal source counts and rounding intervals.
"""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import date, datetime, timezone
from decimal import Decimal
from fractions import Fraction
import json
import re
import unittest
import after158_basic_shares_availability_independent as prior

SELF='work/after158_basic_shares_admission_independent.py'
OUT='work/after158_basic_shares_admission_independent.json'
BASE='outputs/after158_basic_shares_admission_v2/'
BR='work/after158_basic_shares_gaap_bridge_v1/'
RAW='outputs/after158_basic_shares_extraction_v1/RAW_SHARES.json'
POLICY='work/after158_basic_shares_admission_policy_v2.json'
POLICY_SHA='1cca7a5459902ad11b1a329a6a610cf01907ff02c6850557692ff37e120597e1'
PLAN_SHA='11c0849d2d0226a3cf58237a1d8b6b62d579d87bf0f33ab77c6bfbc7d442e528'
FAILURE='outputs/after158_basic_shares_admission_v1/FAILURE.json'
FAILURE_SHA='417990e785e6a4dd2dd7b6607d0428f1eb6311766bf1149156d28f9d7836c27d'
MANUAL='work/after158_basic_shares_independent_manual.json'
REMAINING='work/after158_basic_shares_independent_remaining_evidence.json'
PRIOR='work/after158_basic_shares_availability_independent.json'
PRIOR_SHA='192e666d11b8f7b2452cabddfd7ad7237e63c6d1d9fef3ab26fcd1a132b206eb'
CONCEPT='WeightedAverageNumberOfSharesOutstandingBasic'
need=prior.need

def load(name,expected=None):
    data=prior.resolve(name).read_bytes()
    need(expected is None or prior.sha(data)==expected,'Digest mismatch: '+name)
    return json.loads(data,parse_float=Decimal)

def strict_date(value):
    result=date.fromisoformat(value)
    need(result.isoformat()==value,'Noncanonical date')
    return result

def candidates(event,values,side):
    period=event['periods'][side];result=[]
    for i,row in enumerate(values):
        if (row.get('form'),row.get('fp'),row.get('fy'),row.get('end'))!=('10-K','FY',event['fiscal_year'],period['period_end']):continue
        try:
            start,end,filed=map(strict_date,(row['start'],row['end'],row['filed']))
        except (ValueError,TypeError,KeyError):continue
        if not 330<=(end-start).days+1<=400 or filed<end:continue
        if period['period_start'] is not None and row['start']!=period['period_start']:continue
        if not isinstance(row.get('accn'),str) or not row['accn']:continue
        entry={k:row[k] for k in ('accn','filed','form','fp','fy','start','end')}
        entry['json_pointer']='/facts/us-gaap/'+CONCEPT+'/units/shares/'+str(i)
        result.append(entry)
    return result

def contexts(event,values):
    current=candidates(event,values,'current')
    need(current,'Missing current annual candidate')
    first=min(x['filed'] for x in current)
    earliest=[x for x in current if x['filed']==first]
    need(len({x['accn'] for x in earliest})==1,'Ambiguous earliest accession')
    need(len(earliest)==1,'Ambiguous current context')
    chosen=earliest[0]
    previous=[x for x in candidates(event,values,'comparison') if (x['accn'],x['filed'])==(chosen['accn'],first)]
    need(len(previous)==1,'Missing or ambiguous prior in earliest accession')
    return current,previous,chosen,previous[0]

def check_cell(raw,sec,comparison,unit_independently_verified=False):
    literal=raw['displayed_value']
    need(re.fullmatch(r'(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?',literal) is not None,'Bad source literal')
    source=Fraction(literal.replace(',',''));scale=Fraction(str(raw['unit']['scale']))
    need(scale in (1000,1000000,1000000000),'Bad share scale')
    need(raw['unit']['share_specific'] is True or unit_independently_verified is True,'Unverified general statement unit')
    need(raw['semantic_basis'] in ('basic_eps_denominator_weighting_implicit','explicit_basic_weighted_average_shares'),'Not Basic denominator')
    value=Fraction(str(sec['val']))
    need(source>0 and value>0,'Nonpositive source count')
    normalized=source*scale
    need(Fraction(raw['value_decimal'])==source and Fraction(raw['value_in_shares'])==normalized,'Original normalization differs')
    precision=len(literal.split('.')[1]) if '.' in literal else 0
    step=scale/Fraction(10**precision);lower,upper=normalized-step/2,normalized+step/2
    equal=normalized==value;compatible=lower<=value<=upper
    need(comparison['equal'] is equal and comparison['native_rounding_interval_compatible'] is compatible,'False equality/interval flag')
    expected={'original_value_in_shares':normalized,'companyfacts_value_in_shares':value,
        'original_share_scale':scale,'published_last_place_in_shares':step,
        'native_rounding_interval_lower_inclusive':lower,'native_rounding_interval_upper_inclusive':upper}
    for key,val in expected.items():need(Fraction(comparison[key])==val,'Wrong comparison field '+key)
    need(comparison['original_displayed_value']==literal,'Changed original display')
    return equal,compatible

def normal_pair(periods):
    try:
        endpoints=[]
        for side in ('comparison','current'):
            p=periods[side];a,b=map(strict_date,(p['period_start'],p['period_end']))
            days=b.toordinal()-a.toordinal()+1
            if not 330<=days<=400 or p['days_inclusive']!=days:return False
            endpoints.append((a,b))
        return endpoints[1][0].toordinal()-endpoints[0][1].toordinal()==1
    except (KeyError,TypeError,ValueError):return False

def verified_unit_cells(events):
    """Bind existing independent original-table observations, not bridge values."""
    first=load(MANUAL)['manual_source_pairs'];remaining=load(REMAINING)['records']
    observations={}
    for row in first:
        observations[row['event_id']]={s:(row[s],row['share_scale']) for s in ('current','comparison')}
    for row in remaining:
        need(row['failure'] is None and row['formal_title'] and row['unit_literals'],'Original table evidence missing')
        need(row['event_id'] not in observations,'Duplicate independent source observations')
        observations[row['event_id']]={s:(row['cells'][s]['displayed_value'],row['cells'][s]['share_scale']) for s in ('current','comparison')}
    need(len(observations)==len(events)==44,'Independent 17+27 unit scope')
    generic=0
    for event in events:
        for side in ('current','comparison'):
            cell=event['cells'][side];literal,scale=observations[event['event_id']][side]
            need(Fraction(literal.replace(',',''))==Fraction(cell['value_decimal']) and scale==cell['unit']['scale'],'Independent original unit/cell differs')
            if cell['unit']['share_specific'] is False:
                generic+=1;need(event['symbol'] in ('AMD','MSFT','NVDA'),'Unexpected generic-unit issuer')
                evidence=cell['unit']['evidence'];need(evidence,'Generic unit evidence missing')
                for entry in evidence:
                    need(entry['locator'] and entry['text'],'Generic unit locator missing')
                    words={x.lower() for x in re.findall(r'\b(thousands|millions|billions)\b',entry['text'],re.I)}
                    need(words=={cell['unit']['unit']},'Ambiguous generic unit literal')
                    need(not re.search(r'except[^)]*(number\s+of\s+shares|share\s+counts)',entry['text'],re.I),'Unparsed share-count exception')
            else:need(cell['unit']['share_specific'] is True,'Nonboolean original unit flag')
    need(generic==68,'Exact 34-event generic unit scope')
    return observations

def event_expected(raw,available,metadata,bridge,clock):
    need(raw['event_id']==available['event_id']==metadata['event_id']==bridge['event_id'],'Wrong event')
    need(raw['symbol']==available['symbol']==metadata['symbol']==bridge['symbol'],'Wrong symbol')
    need(raw['periods']==available['periods']==metadata['periods'],'Period metadata differs')
    need(raw['identity']==available['identity'] and raw['inherited_holds']==available['inherited_holds'],'Original flags/identity differs')
    reasons=['INHERITED:'+x for x in raw['inherited_holds']['inherited_rejection_codes']]
    normal=normal_pair(raw['periods'])
    if not normal:reasons.append('ORIGINAL_NORMAL_CONSECUTIVE_PERIODS_NOT_ESTABLISHED')
    unique=metadata['status']==bridge['metadata_status']=='unique_same_accession_annual_pair'
    if not unique:reasons.append('UNIQUE_EARLIEST_CACHED_GAAP_PAIR_UNAVAILABLE')
    compatible=unique and all(bridge['cells'][s]['equal'] or bridge['cells'][s]['native_rounding_interval_compatible'] for s in ('current','comparison'))
    if not compatible:reasons.append('ORIGINAL_GAAP_DENOMINATOR_NOT_CORROBORATED')
    known=clock(raw['release_date']);usable=clock(metadata['selected_filed']) if metadata['selected_filed'] else known
    if prior.nanos(usable)<prior.nanos(known):usable=known
    need(known==available['known_at'],'Release clock differs')
    return dict(event_id=raw['event_id'],symbol=raw['symbol'],fiscal_year=raw['fiscal_year'],
        period_end=raw['periods']['current']['period_end'],prior_period_end=raw['periods']['comparison']['period_end'],
        known_at=known,usable_at=usable,current_shares=raw['cells']['current']['value_in_shares'],
        prior_shares=raw['cells']['comparison']['value_in_shares'],identity=deepcopy(raw['identity']),
        conditional_source_ready=not reasons,same_report_annual_pair=normal,same_share_basis=compatible,
        basis_is_declared_presentation_assumption=True,unresolved_reasons=reasons,
        inherited_holds=deepcopy(raw['inherited_holds']),original_event=deepcopy(raw),gaap_bridge=deepcopy(bridge),
        selected_gaap_contexts=deepcopy(metadata),strict_source_admitted=False,old_admission_flags_changed=False)

def row_expected(row,events):
    known=[e for e in events if e['symbol']==row['symbol'] and prior.nanos(e['known_at'])<=row['cutoff_ns']]
    if known:
        largest=max(e['period_end'] for e in known);latest=[e for e in known if e['period_end']==largest]
        need(len(latest)==1,'Ambiguous latest original');chosen=latest[0]
    else:chosen=None
    selected=chosen['event_id'] if chosen else None
    need(selected==row['availability']['selected_event_id'],'Unexpected fallback or changed selection')
    reasons=list(row['availability']['reasons'])
    if chosen:
        reasons+=chosen['unresolved_reasons']
        if prior.nanos(chosen['usable_at'])>row['cutoff_ns']:reasons.append('LATEST_GAAP_CORROBORATION_NOT_YET_USABLE')
    if row['signal_index']==0:reasons.append('INITIAL_PARTIAL_EXACT_INCUMBENT')
    return dict(signal_index=row['signal_index'],candidate_index=row['candidate_index'],symbol=row['symbol'],
        cutoff=row['cutoff'],selected_event_id=selected,conditional_source_ready=chosen is not None and not reasons,
        reasons=list(dict.fromkeys(reasons)),original_availability_row=deepcopy(row),
        strict_source_admitted=False,old_admission_flags_changed=False)

def audit(manifest_sha):
    plan=load(BASE+'PLAN.json',PLAN_SHA);policy=load(POLICY,POLICY_SHA)
    manifest=load(BASE+'MANIFEST.json',manifest_sha);oldreview=load(PRIOR,PRIOR_SHA)
    need(plan['policy']==policy and manifest['plan_sha256']==PLAN_SHA,'Plan/policy binding')
    need(oldreview['passed'] is True,'Independent availability prerequisite failed')
    need(manifest['strict_source_admitted'] is False and manifest['old_admission_flags_changed'] is False and manifest['numerical_experiment_registered'] is False,'Unauthorized admission/strategy claim')
    need(manifest['activity']==dict(ratios=0,ranks=0,scores=0,orders=0,returns=0),'Numerical activity')
    bridgeplan=load(BR+'PLAN.json');metadata=load(BR+'METADATA.json');comparison=load(BR+'COMPARISON.json')
    expected_pins=prior.merge(plan['input_hashes'],plan['implementation_hashes'],{BASE+'PLAN.json':PLAN_SHA,BASE+'TESTS.json':plan['tests_sha256']})
    need(prior.merge(manifest['input_hashes'])==expected_pins,'Manifest closure differs from frozen PLAN')
    pins=prior.merge(expected_pins,manifest['outputs'],oldreview['input_hashes'],bridgeplan['input_hashes'],
        {BASE+'MANIFEST.json':manifest_sha,PRIOR:PRIOR_SHA,SELF:prior.sha(prior.resolve(SELF).read_bytes()),BASE+'TESTS.json':plan['tests_sha256']})
    prior.verify(pins)
    failure=load(FAILURE,FAILURE_SHA)
    need(failure['exit_code']==1 and failure['actual_ratio_score_return_calculations']==0,'Old failed run receipt differs')
    need(not any(prior.resolve('outputs/after158_basic_shares_admission_v1/'+n).exists() for n in ('EVENTS.json','ROWS.json','MANIFEST.json')),'Old failed run unexpectedly produced artifacts')
    old_policy=load('work/after158_basic_shares_admission_policy.json')
    need({k:v for k,v in policy.items() if k not in ('schema','counts')}=={k:v for k,v in old_policy.items() if k not in ('schema','counts')},'V2 altered policy beyond unit guard')
    need(metadata['plan_sha256']==comparison['plan_sha256']==prior.sha(prior.resolve(BR+'PLAN.json').read_bytes()),'Bridge plan binding')
    need(comparison['metadata_sha256']==prior.sha(prior.resolve(BR+'METADATA.json').read_bytes()) and metadata['values_used_for_selection'] is False,'Bridge pre-value metadata binding')
    raw=load(RAW)['events'];available=load(prior.BASE+'EVENTS.json')['events'];actual_events=load(BASE+'EVENTS.json')
    observations=verified_unit_cells(raw)
    need(str(prior.resolve(MANUAL)) in pins and str(prior.resolve(REMAINING)) in pins,'Independent unit evidence not pinned')
    need(actual_events['strict_source_admitted'] is False,'Promoted actual event wrapper')
    need(len(raw)==len(available)==len(metadata['records'])==len(comparison['records'])==len(actual_events['events'])==44,'44 event axis')
    caches={}
    for symbol,source in bridgeplan['caches'].items():
        facts=load(source['path']);need(facts['cik']==source['cik'],'Wrong companyfacts CIK');caches[symbol]=facts
        if source['provenance']:
            provenance=load(source['provenance'])
            need(provenance['sha256']==prior.sha(prior.resolve(source['path']).read_bytes()),'Wrong sidecar content hash')
            need(provenance['source']=='https://data.sec.gov/api/xbrl/companyfacts/CIK%010d.json'%source['cik'],'Wrong sidecar origin')
    sessions,closes,closures=prior.independent_calendar(load(prior.CAL))
    clock=lambda d:prior.clock(d,sessions,closes)[1]['iso']
    events=[];exact=interval=0;bridge_evidence=[]
    for e,a,m,b,published,planned in zip(raw,available,metadata['records'],comparison['records'],actual_events['events'],bridgeplan['events']):
        need(e['event_id']==planned['event_id'] and e['periods']==planned['periods'] and e['source']==planned['source'],'Bridge original scope differs')
        values=caches[e['symbol']]['facts']['us-gaap'][CONCEPT]['units']['shares']
        all_current,all_prior,current,previous=contexts(e,values)
        need(m['current_candidates']==all_current and m['prior_candidates']==all_prior and m['current_selected']==current and m['prior_selected']==previous,'Independent context selection differs')
        need(m['selected_accession']==b['selected_accession']==current['accn'] and m['selected_filed']==b['selected_filed']==current['filed'],'Bridge accession/date differs')
        need(m['status']==b['metadata_status']=='unique_same_accession_annual_pair','Bridge status differs')
        for side,selected in (('current',current),('comparison',previous)):
            cell=e['cells'][side];bc=b['cells'][side];sec=values[int(selected['json_pointer'].split('/')[-1])]
            need(bc['json_pointer']==selected['json_pointer'] and bc['source_period_end']==cell['period_end']==e['periods'][side]['period_end'],'Cell locator/period differs')
            need(cell['source_sha256']==bc['original_source_sha256']==e['source']['sha256'],'Wrong original bytes')
            eq,within=check_cell(cell,sec,bc,unit_independently_verified=e['event_id'] in observations)
            if cell['unit']['share_specific'] is False:need(eq is True,'Generic unit requires exact GAAP corroboration')
            exact+=int(eq);interval+=int(within)
        expected=event_expected(e,a,m,b,clock);need(expected==published,'Event differs: '+e['event_id']);events.append(expected)
        bridge_evidence.append(dict(event_id=e['event_id'],selected_accession=current['accn'],selected_filed=current['filed'],
            current_pointer=current['json_pointer'],prior_pointer=previous['json_pointer'],known_at=expected['known_at'],
            usable_at=expected['usable_at'],conditional_source_ready=expected['conditional_source_ready'],reasons=expected['unresolved_reasons']))
    meta=load(prior.META);original=load(prior.ORIGINAL);arows=load(prior.BASE+'ROWS.json');published_rows=load(BASE+'ROWS.json')
    need(len(arows)==len(original)==len(published_rows)==5978 and len(meta['dates'])==len(meta['symbols'])==122,'Full grid missing')
    rows=[];positions={s:i for i,s in enumerate(sessions)}
    for i,(signal,names) in enumerate(zip(meta['dates'],meta['symbols'])):
        need(len(names)==len(set(names))==49,'49 unique candidates');cutoff=sessions[:positions[signal]][-42]
        for j,symbol in enumerate(names):
            k=i*49+j;r=arows[k];old=original[k]
            need((r['signal_index'],r['candidate_index'],r['symbol'],r['decision_date'])==(i,j,symbol,signal),'Wrong original axes')
            need(r['original_row']==old and r['cutoff_date']==cutoff and r['cutoff_ns']==closes[cutoff]['ns'] and r['cutoff']==closes[cutoff]['iso'],'Original row or t-42 differs')
            need(r['availability']==prior.select_oracle(available,symbol,cutoff,closes[cutoff]['ns'],symbol in meta['etfs']),'Original availability/holds differs')
            expected=row_expected(r,events);need(expected==published_rows[k],'Row differs '+str((i,j,symbol)));rows.append(expected)
    summary=dict(events=len(events),conditionally_ready_events=sum(e['conditional_source_ready'] for e in events),
        event_blockers=dict(Counter(x for e in events for x in e['unresolved_reasons'])))
    for name,(lo,hi) in prior.ERAS.items():
        good=[r for r in rows if lo<=r['signal_index']<hi and r['conditional_source_ready']];months=Counter(r['signal_index'] for r in good)
        summary[name]=dict(slots=len(good),by_symbol=dict(Counter(r['symbol'] for r in good)),distinct_events=len({r['selected_event_id'] for r in good}),months_with_at_least_two=sum(n>=2 for n in months.values()))
    need(summary==manifest['summary']==load(BASE+'COVERAGE.json'),'Independent coverage differs')
    need(exact==interval==88,'Actual fixed bridge expected 88 exact corroborated cells')
    rejected=[e['event_id'] for e in events if e['inherited_holds']['inherited_rejection_codes']]
    need(len(rejected)==6 and all(not e['conditional_source_ready'] for e in events if e['event_id'] in rejected),'Original six holds changed')
    tests=load(BASE+'TESTS.json');need(tests['passed'] is True and tests['implementation_hashes']==plan['implementation_hashes'],'Source tests not bound')
    prior.verify(pins)
    return dict(schema='after158_basic_shares_admission_independent_v2',created_utc=datetime.now(timezone.utc).isoformat(),passed=True,
        source_manifest_sha256=manifest_sha,plan_sha256=PLAN_SHA,
        input_hashes=pins,summary=summary,bridge_event_evidence=bridge_evidence,exact_source_cells_verified=exact,
        checks={k:True for k in ('all_input_output_and_implementation_hashes_verified','independent_earliest_cached_accession_selection',
            'all_88_original_counts_and_native_intervals_verified','all_44_original_periods_and_holds_verified','max_of_two_second_session_close_clocks',
            'v1_failure_preserved_and_only_v2_unit_guard_changed','all_34_general_units_have_independent_original_table_evidence',
            'all_5978_grid_rows_and_original_nested_records_preserved','latest_known_before_validity_no_fallback','original_t_minus_42_and_identity_constraints',
            'six_original_rejections_not_lifted','initial_partial_unchanged','all_five_coverage_splits_recomputed','strict_admission_flags_remain_false')},
        delayed_corroboration_slots=sum('LATEST_GAAP_CORROBORATION_NOT_YET_USABLE' in r['reasons'] for r in rows),
        rejected_original_events=rejected,
        conditional_only=True,source_admission_changed=False,strict_source_admitted=False,
        limitations=policy['source_limitations'],identity_interval_scope='Inherited report-use intervals, not certified historical security epochs.',
        activity=dict(ratios=0,factors=0,ranks=0,scores=0,orders=0,returns=0,new_strategy_trials=0))

class Tests(unittest.TestCase):
    def event(self):return dict(fiscal_year=2020,periods=dict(current=dict(period_start='2020-01-01',period_end='2020-12-31',days_inclusive=366),comparison=dict(period_start='2019-01-01',period_end='2019-12-31',days_inclusive=365)))
    def context(self,**kw):
        r=dict(form='10-K',fp='FY',fy=2020,accn='a',filed='2021-02-01',start='2020-01-01',end='2020-12-31',val=1);r.update(kw);return r
    def test_metadata_ignores_values(self):
        r=[self.context(val=-99),self.context(start='2019-01-01',end='2019-12-31',val=None)]
        self.assertEqual(contexts(self.event(),r)[2]['accn'],'a')
    def test_no_later_accession_fallback(self):
        with self.assertRaises(ValueError):contexts(self.event(),[self.context(),self.context(accn='b',filed='2021-03-01',start='2019-01-01',end='2019-12-31')])
    def test_duplicate_context_rejected(self):
        with self.assertRaises(ValueError):contexts(self.event(),[self.context(),self.context()])
    def test_interim_and_amendment_excluded(self):
        self.assertEqual(candidates(self.event(),[self.context(form='10-K/A'),self.context(start='2020-10-01')],'current'),[])
    def test_periods_and_gap(self):
        p=self.event()['periods'];self.assertTrue(normal_pair(p));p['current']['period_start']='2020-01-02';p['current']['days_inclusive']=365;self.assertFalse(normal_pair(p))
    def test_unknown_start_not_imputed(self):
        p=self.event()['periods'];p['comparison']['period_start']=None;self.assertFalse(normal_pair(p))
    def cell(self):return dict(displayed_value='1.20',value_decimal='1.20',value_in_shares='1200',unit=dict(scale=1000,share_specific=True),semantic_basis='explicit_basic_weighted_average_shares')
    def cmp(self,val='1205',**kw):
        r=dict(equal=False,native_rounding_interval_compatible=True,original_displayed_value='1.20',original_value_in_shares='1200',companyfacts_value_in_shares=val,original_share_scale='1000',published_last_place_in_shares='10',native_rounding_interval_lower_inclusive='1195',native_rounding_interval_upper_inclusive='1205');r.update(kw);return r
    def test_interval_boundary(self):self.assertEqual(check_cell(self.cell(),dict(val=1205),self.cmp()),(False,True))
    def test_false_interval_rejected(self):
        with self.assertRaises(ValueError):check_cell(self.cell(),dict(val=1206),self.cmp('1206'))
    def test_false_exact_rejected(self):
        with self.assertRaises(ValueError):check_cell(self.cell(),dict(val=1205),self.cmp(equal=True))
    def test_raw_not_replaced(self):
        r=self.cell();r['value_in_shares']='1205'
        with self.assertRaises(ValueError):check_cell(r,dict(val=1205),self.cmp())
    def test_general_unit_requires_independent_evidence(self):
        r=self.cell();r['unit']['share_specific']=False
        with self.assertRaises(ValueError):check_cell(r,dict(val=1205),self.cmp())
        self.assertEqual(check_cell(r,dict(val=1205),self.cmp(),True),(False,True))
    def row(self):return dict(signal_index=1,candidate_index=0,symbol='X',cutoff='2021-02-02T21:00:00+00:00',cutoff_ns=prior.nanos('2021-02-02T21:00:00+00:00'),availability=dict(selected_event_id='new',reasons=[]))
    def chosen(self):return dict(symbol='X',event_id='new',period_end='2020-12-31',known_at='2021-01-20T21:00:00+00:00',usable_at='2021-02-03T21:00:00+00:00',unresolved_reasons=[])
    def test_filing_delay_blocks(self):self.assertFalse(row_expected(self.row(),[self.chosen()])['conditional_source_ready'])
    def test_old_fallback_rejected(self):
        r=self.row();r['availability']['selected_event_id']='old'
        with self.assertRaises(ValueError):row_expected(r,[self.chosen()])
    def test_initial_partial_and_old_reason(self):
        r=self.row();r['signal_index']=0;r['availability']['reasons']=['old'];out=row_expected(r,[self.chosen()]);self.assertIn('old',out['reasons']);self.assertIn('INITIAL_PARTIAL_EXACT_INCUMBENT',out['reasons'])
    def test_exact_usable_boundary(self):
        e=self.chosen();e['usable_at']=self.row()['cutoff'];self.assertTrue(row_expected(self.row(),[e])['conditional_source_ready'])

def main():
    p=argparse.ArgumentParser();p.add_argument('--test-only',action='store_true');p.add_argument('--write',action='store_true');p.add_argument('--manifest-sha');a=p.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    need(result.wasSuccessful(),'Independent synthetic tests failed')
    if a.test_only:return
    need(a.write and a.manifest_sha,'Explicit write and manifest SHA required')
    report=audit(a.manifest_sha);report['synthetic_validation']=dict(tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),tests_embedded_in=SELF)
    report['required_checks']=sorted(report['checks'])
    report['implementation_hashes']={SELF:prior.sha(prior.resolve(SELF).read_bytes()),prior.SELF:prior.sha(prior.resolve(prior.SELF).read_bytes())}
    encoded=(json.dumps(report,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n').encode()
    with prior.resolve(OUT).open('xb') as handle:handle.write(encoded)
    print(json.dumps(dict(path=OUT,sha256=prior.sha(encoded),summary=report['summary'],delayed_slots=report['delayed_corroboration_slots'])))

if __name__=='__main__':main()
