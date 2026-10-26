"""Pure synthetic source-admission tests; no real statement amounts consumed."""
from copy import deepcopy
from datetime import date,timedelta
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import same_report144_admission as m


def fixture():
    ref=dict(path='synthetic/source',sha256='a'*64)
    review='synthetic/review.json';bindings={ref['path']:ref['sha256'],review:'b'*64}
    rows=[];sessions=[]
    for symbol,years in m.YEARS.items():
        for year in years:
            release=date(year+1,2,1);available=release+timedelta(days=1)
            sessions.extend([release.isoformat(),available.isoformat()])
            unknown=m.REJECTIONS.get((symbol,year))==['unknown_comparison_duration']
            def period(y,missing=False):
                begin,end=date(y,1,1),date(y,12,31)
                return dict(start=None if missing else begin.isoformat(),end=end.isoformat(),
                    days_inclusive=None if missing else (end-begin).days+1,
                    derived=True,start_is_direct_quote=False,derivation='synthetic prior-year boundary',evidence=[deepcopy(ref)])
            current,prior=period(year),period(year-1,unknown)
            def amount(value,period):
                return dict(amount_decimal=value,available_from=available.isoformat(),currency_iso=None,
                    currency_literal='$',date=period['end'],displayed_amount=value,original_cell={},
                    scale=1000000,source=deepcopy(ref))
            rows.append(dict(event_id=f'{symbol}:FY{year}:{release}',symbol=symbol,fiscal_year=year,
                release_date=release.isoformat(),available_from=available.isoformat(),current_period=current,comparison_period=prior,
                eligible_under_listed_research_assumptions=(symbol,year)not in m.REJECTIONS,
                rejection_codes=deepcopy(m.REJECTIONS.get((symbol,year),[])),
                fact_admitted=False,model_admitted=False,strategy_admitted=False,first_public_version_verified=False,source_only=True,
                identity=dict(current_issuer_name=m.ISSUERS[symbol],certified_daily_security_master=False,
                    valid_from=available.isoformat(),valid_through=(date(year,12,31)+timedelta(days=550)).isoformat(),
                    anchors=[deepcopy(ref)],kind='synthetic own-time anchor'),
                financials={name:dict(gross_profit=amount('10',period),total_assets=amount('100',period))for name,period in [('current',current),('comparison',prior)]},
                source=deepcopy(ref),reviews=[review],original_record={},provenance={}))
    original=dict(schema_version=1,status='synthetic source preflight',assumptions={'status':'not admitted','vintage':'unverified'},
        counts=dict(events=20),events=rows,missing_publications=[dict(symbol='AMD',fiscal_year=2019,event_created=False,reason='Synthetic missing original')],
        financial_computations=0,network_requests=0)
    policy=m.make_policy(dict(assumptions=deepcopy(original['assumptions'])))
    return original,sorted(set(sessions)),bindings,policy


class MappingTests(unittest.TestCase):
    def setUp(self):self.source,self.sessions,self.bindings,self.policy=fixture()
    def build(self):return m.build_dataset(self.source,self.sessions,self.bindings,self.policy)
    def event(self):return self.source['events'][1]
    def test_all_events_and_eighty_cells_preserved(self):
        before=deepcopy(self.source);data=self.build()
        self.assertEqual(data['counts'],dict(events=20,conditionally_admitted=16,rejected=4,raw_values=80))
        self.assertEqual(self.source,before)
        for src,dst in zip(self.source['events'],data['events']):
            for period in('current','comparison'):
                for left,right in [('gross_profit','gross_profit'),('total_assets','assets')]:
                    a=src['financials'][period][left];b=dst['same_report'][period][right]
                    self.assertEqual(b,dict(value=a['amount_decimal'],scale=str(a['scale']),currency_literal=a['currency_literal']))
    def test_no_model_strategy_or_first_version_certification(self):
        data=self.build()
        for key in('model_admitted','strategy_admitted','first_public_version_verified','complete_correction_history_verified','certified_daily_security_master'):self.assertIs(data[key],False)
        for key in('ratios_computed','signs_computed','ranks_computed','returns_computed'):self.assertEqual(data[key],0)
    def test_exclusions_are_not_silently_removed(self):
        data=self.build();rejected={(r['symbol'],r['fiscal_year']):r['rejection_codes']for r in data['events']if not r['admitted']}
        self.assertEqual(rejected,m.REJECTIONS)
    def test_same_report_four_values_current_availability(self):
        e=self.event();e['financials']['comparison']['gross_profit']['available_from']='2016-01-01'
        with self.assertRaisesRegex(ValueError,'availability'):self.build()
    def test_different_source_cannot_be_joined(self):
        self.event()['financials']['comparison']['total_assets']['source']['sha256']='c'*64
        with self.assertRaisesRegex(ValueError,'share current source'):self.build()
    def test_missing_or_unknown_event_field(self):
        for key in('new_assumption',None):
            original=deepcopy(self.source)
            if key:self.event()[key]=True
            else:del self.event()['current_period']
            with self.assertRaisesRegex(ValueError,'event fields'):self.build()
            self.source=original
    def test_unknown_amount_and_period_fields(self):
        e=self.event();e['financials']['current']['gross_profit']['new']=1
        with self.assertRaisesRegex(ValueError,'amount'):self.build()
        del e['financials']['current']['gross_profit']['new'];e['current_period']['new']=1
        with self.assertRaisesRegex(ValueError,'period fields'):self.build()
    def test_nonfinite_bool_float_and_exponent_values_rejected(self):
        for value in('NaN','Infinity','1e2',True,1.0,'+1'):
            self.event()['financials']['current']['gross_profit']['amount_decimal']=value
            with self.subTest(value=value),self.assertRaises(ValueError):self.build()
    def test_zero_negative_gross_profit_kept(self):
        for value in('0','-20'):
            self.event()['financials']['current']['gross_profit']['amount_decimal']=value
            self.assertEqual(self.build()['events'][1]['same_report']['current']['gross_profit']['value'],value)
    def test_assets_must_be_positive(self):
        for value in('0','-20'):
            self.event()['financials']['current']['total_assets']['amount_decimal']=value
            with self.assertRaisesRegex(ValueError,'Nonpositive'):self.build()
    def test_scale_must_be_positive_integer(self):
        for value in(0,-1,True,1.5,'1000000'):
            self.event()['financials']['current']['gross_profit']['scale']=value
            with self.assertRaisesRegex(ValueError,'scale'):self.build()
    def test_currency_not_falsely_promoted_to_USD(self):
        for key,value in [('currency_literal','USD'),('currency_iso','USD')]:
            e=self.event();e['financials']['current']['gross_profit'][key]=value
            with self.assertRaisesRegex(ValueError,'currency'):self.build()
            e['financials']['current']['gross_profit'][key]='$'if key=='currency_literal'else None
    def test_no_dropped_or_phantom_year(self):
        self.source['events'].pop()
        with self.assertRaisesRegex(ValueError,'twenty'):self.build()
    def test_missing_AMD2019_not_invented(self):
        self.source['missing_publications'][0]['event_created']=True
        with self.assertRaisesRegex(ValueError,'AMD FY2019'):self.build()
    def test_no_prior_duration_fabrication(self):
        self.source['events'][0]['comparison_period'].update(start='2015-01-01',days_inclusive=365)
        # Even though numerically annual, an unknown-duration sentinel must stay unknown.
        with self.assertRaises(ValueError):self.build()
    def test_eligible_unknown_duration_rejected(self):
        self.event()['comparison_period'].update(start=None,days_inclusive=None)
        with self.assertRaisesRegex(ValueError,'Unknown duration'):self.build()
    def test_wrong_year_or_noncontiguous_period(self):
        self.event()['comparison_period']['end']='2015-12-31'
        with self.assertRaisesRegex(ValueError,'year'):self.build()
    def test_annual_duration_count_checked(self):
        self.event()['current_period']['days_inclusive']=True
        with self.assertRaisesRegex(ValueError,'duration'):self.build()
    def test_pinned_calendar_strict_next_session(self):
        e=self.event();e['available_from']=e['release_date']
        with self.assertRaisesRegex(ValueError,'next-session'):self.build()
    def test_calendar_incomplete_or_duplicate(self):
        for sessions in([self.sessions[-1]],self.sessions+[self.sessions[-1]]):
            with self.assertRaises(ValueError):m.build_dataset(self.source,sessions,self.bindings,self.policy)
    def test_identity_window_cannot_use_future_event(self):
        e=self.event();e['identity']['valid_through']='2099-12-31'
        with self.assertRaisesRegex(ValueError,'window'):self.build()
    def test_identity_begin_matches_current_availability(self):
        self.event()['identity']['valid_from']='2010-01-01'
        with self.assertRaisesRegex(ValueError,'own release'):self.build()
    def test_late_publication_beyond_550_is_not_eligible(self):
        e=self.event();release=date(e['fiscal_year']+2,10,1)
        e.update(release_date=release.isoformat(),event_id=f'{e["symbol"]}:FY{e["fiscal_year"]}:{release}',available_from=(release+timedelta(days=1)).isoformat())
        e['identity']['valid_from']=e['available_from'];self.sessions=sorted(set(self.sessions+[e['release_date'],e['available_from']]))
        with self.assertRaisesRegex(ValueError,'late publication'):self.build()
    def test_source_reference_must_be_bound(self):
        self.bindings['synthetic/source']='c'*64
        with self.assertRaisesRegex(ValueError,'Unbound event'):self.build()
    def test_review_reference_must_be_bound(self):
        del self.bindings['synthetic/review.json']
        with self.assertRaisesRegex(ValueError,'review binding'):self.build()
    def test_inherited_assumption_cannot_change(self):
        self.source['assumptions']['vintage']='certified'
        with self.assertRaisesRegex(ValueError,'assumptions'):self.build()
    def test_exact_current_source_event_identity_preserved(self):
        e=self.build()['events'][1]
        self.assertEqual(e['currency_key'],'same-report-dollar:'+e['event_id'])
        self.assertEqual(e['method'],m.METHOD)
        self.assertEqual(e['evidence']['preflight_event_sha256'],m.digest(m.encoded(self.event())))


class SelectionMetadataTests(unittest.TestCase):
    def setUp(self):
        src,sessions,bindings,policy=fixture();self.rows=m.build_dataset(src,sessions,bindings,policy)['events']
    def test_invalid_latest_blocks_previously_valid(self):
        chosen=next(r for r in self.rows if r['symbol']=='AMD'and r['fiscal_year']==2018)
        result=m.latest_metadata(self.rows,'AMD',chosen['available_from'])
        self.assertEqual(result,dict(event_id=chosen['event_id'],usable=False,reason='latest_event_rejected'))
    def test_expired_latest_does_not_fall_back(self):
        chosen=next(r for r in self.rows if r['symbol']=='AAPL'and r['fiscal_year']==2025)
        self.assertEqual(m.latest_metadata(self.rows,'AAPL','2030-01-01')['reason'],'identity_expired')
    def test_future_mutation_does_not_change_earlier_selection(self):
        asof='2018-03-01';old=m.latest_metadata(self.rows,'AAPL',asof)
        self.rows[-1]['admitted']=True;self.rows[-1]['same_report']['current']['gross_profit']['value']='999'
        self.assertEqual(m.latest_metadata(self.rows,'AAPL',asof),old)
    def test_same_date_ambiguity_stops(self):
        row=deepcopy(self.rows[1]);row['event_id']='conflicting';self.rows.append(row)
        with self.assertRaisesRegex(ValueError,'same-date'):m.latest_metadata(self.rows,'AAPL',row['available_from'])
    def test_unknown_symbol_is_neutral(self):
        self.assertEqual(m.latest_metadata(self.rows,'MSFT','2020-01-01'),dict(event_id=None,usable=False,reason='no_known_event'))


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='synthetic-admission144-');self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name).resolve()
    def test_hash_tamper_detected(self):
        (self.root/'source').write_bytes(b'original');pins={'source':m.hash_file(self.root,'source')}
        m.verify_hashes(self.root,pins);(self.root/'source').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'SHA mismatch'):m.verify_hashes(self.root,pins)
    def test_exclusive_publication_preserves_bytes(self):
        m.publish(self.root,'result.json',{'value':1});before=(self.root/'result.json').read_bytes()
        with self.assertRaises(FileExistsError):m.publish(self.root,'result.json',{'value':2})
        self.assertEqual((self.root/'result.json').read_bytes(),before)
    def test_local_path_escape_cloud_symlink_rejected(self):
        (self.root/'target').write_text('x');(self.root/'link').symlink_to(self.root/'target')
        for path in('../x','/tmp/x','CloudStorage/x','link'):
            with self.subTest(path=path),self.assertRaises(ValueError):m.local(self.root,path)
    def test_duplicate_nonfinite_json_rejected(self):
        for raw in('{"a":1,"a":2}','{"a":NaN}','{"a":Infinity}'):
            with self.assertRaises(ValueError):m.decode(raw)
    def test_conflicting_nested_source_refs(self):
        with self.assertRaisesRegex(ValueError,'Conflicting'):m.source_refs([dict(path='x',sha256='a'*64),dict(path='x',sha256='b'*64)])


def write_report():
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
    if not result.wasSuccessful():raise SystemExit(stream.getvalue())
    report=dict(passed=True,synthetic_only=True,tests_run=result.testsRun,failures=0,errors=0,
        actual_financial_calculations=0,log=stream.getvalue(),
        source_code_hashes={p:m.hash_file(m.ROOT,p)for p in(m.SELF,m.TEST)})
    m.publish(m.ROOT,m.OUT+'/TESTS.json',report)
    print(json.dumps(dict(passed=True,tests_run=result.testsRun,sha256=m.hash_file(m.ROOT,m.OUT+'/TESTS.json'))))
if __name__=='__main__':
    if sys.argv[1:]==['--write-report']:write_report()
    else:unittest.main(verbosity=2)
