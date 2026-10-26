"""Generated-source tests only; no actual documents or model data are loaded."""
from copy import deepcopy
from datetime import date, timedelta
import json
from pathlib import Path
import tempfile
import unittest

import after149_batch1_admission as m


def sessions():
    start, end = date(2015, 1, 1), date(2027, 12, 31)
    return [start + timedelta(days=i) for i in range((end-start).days+1)
            if (start+timedelta(days=i)).weekday() < 5]


def event(symbol='AMZN', year=2020, missing=None):
    current = date(year, 12, 31)
    release = date(year+1, 2, 1).isoformat()
    available = m.first_session(release, sessions()) or date(year+1, 2, 2).isoformat()
    source = dict(path='source.html', sha256='a'*64)
    period = dict(period_start=date(year,1,1).isoformat(), period_end=current.isoformat(),
                  days_inclusive=(current-date(year,1,1)).days+1)
    cells = {}
    for name in m.FIELDS:
        cells[name] = dict(value_decimal='-10' if name == 'annual_net_income' else '100',
                           scale='1000000', currency_literal='$', currency_iso=None,
                           displayed_amount='(10)' if name == 'annual_net_income' else '100',
                           locator={'row': 1}, source=deepcopy(source), available_from=available,
                           fact_admitted=False, as_of=(date(year-1,12,31) if name == 'prior_total_assets' else current).isoformat())
        if name == 'annual_net_income':
            cells[name]['period'] = deepcopy(period)
    if missing:
        cells[missing] = None
    item = dict(event_id='%s_%s'%(symbol,release), symbol=symbol, fiscal_year=year,
                release_date=release, available_from=available, source=source, fields=cells,
                fact_admitted=False, model_admitted=False, strategy_admitted=False,
                gaps=['synthetic_missing'] if missing else [], unrelated_metadata={'keep': [1,2,3]})
    context = dict(event_id=item['event_id'], fiscal_year=year, release_date=release,
                   publication_date=release, available_session=available)
    if symbol == 'AMZN':
        context.update(source)
        context.update(current_end=current.isoformat(), comparison_end=date(year-1,12,31).isoformat())
    else:
        context.update(source=source, periods={'current': period, 'comparison': {'period_end':date(year-1,12,31).isoformat(),
                      'period_start':None,'days_inclusive':None}})
    return item, (symbol, '0001018724' if symbol=='AMZN' else '0000858877', context)


def rowreview(item):
    return dict(event_id=item['event_id'], passed=True, metadata_verified=True,
                fields={name: dict(verified=value is not None, is_null=value is None,
                                  null_reason_verified=True if value is None else None,
                                  source_value_present=value is not None)
                        for name,value in item['fields'].items()})


def catalogrow(item):
    return dict(event_id=item['event_id'], symbol=item['symbol'], fiscal_year=item['fiscal_year'],
                fiscal_quarter=4, publication_date=item['release_date'],
                available_session=item['available_from'], unmodified_catalog_metadata='keep')


def assessed(symbol='AMZN',year=2020):
    item, context = event(symbol,year)
    return m.assess_event(item,context,rowreview(item),sessions())


def readwrite(root,name,value):
    p=Path(root)/name
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,sort_keys=True)+'\n')
    return m.digest(p)


def fixture(root):
    root=Path(root)
    root.joinpath('source.html').write_text('synthetic only')
    root.joinpath('receipt.json').write_text('{}')
    source_sha=m.digest(root/'source.html')
    docs={'AMZN':[], 'CSCO':[]}
    items=[]
    records=[]
    cat=[]
    for symbol in ('AMZN','CSCO'):
        for year in range(2010,2026):
            item,ct=event(symbol,year)
            cat.append(catalogrow(item))
            if symbol=='AMZN' and year>=2020:
                continue
            item['source']['sha256']=source_sha
            for cell in item['fields'].values():
                cell['source']['sha256']=source_sha
            ct[2]['sha256']=source_sha if symbol=='AMZN' else ct[2].get('sha256')
            if symbol=='CSCO':
                ct[2]['source']['sha256']=source_sha
            docs[symbol].append(ct[2])
            items.append(item)
            records.append(dict(event_id=item['event_id'],symbol=symbol,fiscal_year=year,
                context_path=m.CONTEXTS[symbol],source_path='source.html',source_sha256=source_sha,receipt_path='receipt.json'))
    policy=dict(schema_version=1,stage='prospective_source_policy_not_strategy_registration',
        required_fields=list(m.FIELDS),source_scope={'symbols':['AMZN','CSCO'],'no_automatic_admission':True},
        known_time={'maximum_age_days':550},preservation={name:False for name in
        ('first_public_version_verified','complete_correction_history_verified','certified_daily_security_master')},
        accounting={'AMZN_FY2019':'Hold pending lease evidence','CSCO_FY2019':'Hold pending ASC606 evidence'})
    policy_sha=readwrite(root,m.POLICY,policy)
    readwrite(root,m.CONTEXTS['AMZN'],dict(symbol='AMZN',identity={'cik':'0001018724'},documents=docs['AMZN']))
    readwrite(root,m.CONTEXTS['CSCO'],dict(symbol='CSCO',identity={'cik':'0000858877'},events=docs['CSCO']))
    names=['AMZN','CSCO']+['STOCK%02d'%i for i in range(71)]
    etfs=['ETF%02d'%i for i in range(17)]
    dates=['%04d-%02d-01'%(2016+(7+i)//12, (7+i)%12+1) for i in range(122)]
    panels=[[names[(i+j)%73] for j in range(32)]+etfs for i in range(122)]
    slots=[dict(session=d,symbol=s) for d,ss in zip(dates,panels) for s in ss if s not in etfs]
    readwrite(root,m.CENSUS,dict(stocks=[{'symbol':s} for s in names],annual_documents=cat,monthly_candidate_coverage=sorted(slots,key=lambda row:(row['symbol'],row['session']))))
    readwrite(root,m.META,dict(dates=dates,symbols=panels,etfs=etfs))
    readwrite(root,m.SESSIONS,dict(sessions=[s.isoformat() for s in sessions()]))
    old=dict(events=[dict(event_id='old%02d'%i,symbol=['AAPL','AMD','MSFT','NVDA'][i%4],
        stock_compensation_admission={'admitted':i<31}) for i in range(44)])
    old_sha=readwrite(root,m.OLD_DATA,old)
    readwrite(root,m.OLD_ADMISSION,dict(passed=True,data_sha256=old_sha,events=44,admitted_events=31,held_events=13))
    readwrite(root,'code.py',{'synthetic':'stub'})
    for name in ('work/after149_batch1_admission.py','work/test_after149_batch1_admission.py'):
        readwrite(root,name,{'synthetic':'stub'})
    input_names=[m.POLICY,m.CENSUS,m.META,m.SESSIONS,m.OLD_DATA,m.OLD_ADMISSION,*m.CONTEXTS.values(),'source.html','receipt.json']
    pins={name:m.digest(root/name) for name in input_names}
    plan=dict(inputs=pins,code={'code.py':m.digest(root/'code.py')},records=records)
    plan_sha=readwrite(root,m.BATCH+'/PLAN.json',plan)
    extraction=dict(plan_sha256=plan_sha,status='source_values_extracted_pending_independent_review',events=items,
        new_financial_admissions=0,new_strategy_trials=0,ratios_computed=0,returns_computed=0)
    extracted_sha=readwrite(root,m.BATCH+'/EXTRACTIONS.json',extraction)
    review=dict(passed=True,plan_sha256=plan_sha,extractions_sha256=extracted_sha,
        event_reviews=[rowreview(e) for e in items],input_hashes={**pins,
        m.BATCH+'/PLAN.json':plan_sha,m.BATCH+'/EXTRACTIONS.json':extracted_sha})
    review_sha=readwrite(root,m.REVIEW,review)
    return plan_sha,review_sha,policy_sha


class SourceTests(unittest.TestCase):
    def test_signed_income_zero_or_loss_admitted(self):
        for value in ['0','-10','1.25']:
            e,c=event();e['fields']['annual_net_income']['value_decimal']=value
            self.assertTrue(m.assess_event(e,c,rowreview(e),sessions())['ni_assets_admission']['admitted'])

    def test_per_field_scales_not_forced_equal_or_rescaled(self):
        e,c=event();e['fields']['annual_net_income']['scale']='1000'
        out=m.assess_event(e,c,rowreview(e),sessions())
        self.assertTrue(out['ni_assets_admission']['admitted'])
        self.assertEqual(out['fields'],e['fields'])

    def test_currency_iso_null_allowed(self):
        self.assertTrue(assessed()['ni_assets_admission']['admitted'])

    def test_currency_conflict_held(self):
        e,c=event();e['fields']['annual_net_income']['currency_iso']='USD'
        self.assertIn('inconsistent_currency_iso',m.assess_event(e,c,rowreview(e),sessions())['ni_assets_admission']['reasons'])

    def test_asset_nonpositive_held(self):
        for value in ['0','-1','NaN','Infinity']:
            e,c=event();e['fields']['current_total_assets']['value_decimal']=value
            self.assertFalse(m.assess_event(e,c,rowreview(e),sessions())['ni_assets_admission']['admitted'])

    def test_negative_scale_held(self):
        e,c=event();e['fields']['annual_net_income']['scale']='-1'
        self.assertFalse(m.assess_event(e,c,rowreview(e),sessions())['ni_assets_admission']['admitted'])

    def test_missing_field_retained_and_held(self):
        e,c=event(missing='prior_total_assets')
        out=m.assess_event(e,c,rowreview(e),sessions())
        self.assertIsNone(out['fields']['prior_total_assets'])
        self.assertIn('missing_prior_total_assets',out['ni_assets_admission']['reasons'])

    def test_early_calendar_held(self):
        self.assertIn('unverified_exchange_calendar',assessed(year=2011)['ni_assets_admission']['reasons'])

    def test_wrong_first_session_held(self):
        e,c=event();e['available_from']='2021-02-03';c[2]['available_session']=e['available_from']
        for f in e['fields'].values():f['available_from']=e['available_from']
        self.assertIn('availability_not_first_pinned_session',m.assess_event(e,c,rowreview(e),sessions())['ni_assets_admission']['reasons'])

    def test_exact_next_session_not_release_day(self):
        ss=sessions()
        self.assertEqual(m.first_session('2020-02-07',ss),'2020-02-10')
        self.assertIsNone(m.first_session('2014-12-31',ss))

    def test_two_2019_accounting_holds(self):
        for symbol in m.SYMBOLS:
            self.assertFalse(assessed(symbol,2019)['ni_assets_admission']['admitted'])
            self.assertIn('transition_unresolved',assessed(symbol,2019)['ni_assets_admission']['reasons'][0])

    def test_cashflow_only_note_not_ni_asset_rejection(self):
        e,c=event();e['accounting_context']=['cashflow excess tax benefit reclassification']
        self.assertTrue(m.assess_event(e,c,rowreview(e),sessions())['ni_assets_admission']['admitted'])

    def test_wrong_prior_endpoint_held(self):
        e,c=event();e['fields']['prior_total_assets']['as_of']='2020-01-01'
        self.assertFalse(m.assess_event(e,c,rowreview(e),sessions())['ni_assets_admission']['admitted'])

    def test_quarter_income_held(self):
        e,c=event();e['fields']['annual_net_income']['period']['period_start']='2020-10-01'
        e['fields']['annual_net_income']['period']['days_inclusive']=92
        self.assertFalse(m.assess_event(e,c,rowreview(e),sessions())['ni_assets_admission']['admitted'])

    def test_wrong_source_or_availability_cell_held(self):
        for key,value in [('source',{'path':'other','sha256':'b'*64}),('available_from','2020-01-01')]:
            e,c=event();e['fields']['annual_net_income'][key]=value
            self.assertFalse(m.assess_event(e,c,rowreview(e),sessions())['ni_assets_admission']['admitted'])

    def test_identity_is_report_use_window_not_certification(self):
        out=assessed()
        self.assertEqual(out['identity']['issuer_id'],'0001018724')
        self.assertEqual(out['identity']['identity_valid_from'],out['available_from'])
        self.assertEqual(out['identity']['identity_valid_through'],(date(2020,12,31)+timedelta(days=550)).isoformat())
        self.assertFalse(out['identity']['certified_daily_security_master'])

    def test_unrelated_metadata_preserved_no_mutation(self):
        e,c=event();before=deepcopy(e)
        out=m.assess_event(e,c,rowreview(e),sessions())
        self.assertEqual(e,before)
        self.assertEqual({k:out[k] for k in e},e)

    def test_previous_comparison_income_start_not_required(self):
        self.assertTrue(assessed('CSCO')['ni_assets_admission']['admitted'])

    def test_unknown_cik_rejected(self):
        with self.assertRaises(ValueError):m.context_map({'AMZN':{'symbol':'AMZN','identity':{'cik':None},'documents':[]},'CSCO':{}})

    def test_review_missing_metadata_rejected(self):
        e,_=event();rr=rowreview(e);rr['metadata_verified']=False
        with self.assertRaises(ValueError):m.verify_independent_review(dict(passed=True,plan_sha256='p',extractions_sha256='e',event_reviews=[rr]),[e],'p','e')

    def test_review_missing_value_verification_rejected(self):
        e,_=event();rr=rowreview(e);rr['fields']['annual_net_income']['verified']=False
        with self.assertRaises(ValueError):m.verify_independent_review(dict(passed=True,plan_sha256='p',extractions_sha256='e',event_reviews=[rr]),[e],'p','e')

    def test_null_review_can_confirm_parser_gap_not_source_absence(self):
        e,_=event(missing='prior_total_assets');rr=rowreview(e)
        rr['fields']['prior_total_assets']['source_value_present']=True
        m.verify_independent_review(dict(passed=True,plan_sha256='p',extractions_sha256='e',event_reviews=[rr]),[e],'p','e')

    def test_duplicate_event_review_rejected(self):
        e,_=event();rr=rowreview(e)
        with self.assertRaises(ValueError):m.verify_independent_review(dict(passed=True,plan_sha256='p',extractions_sha256='e',event_reviews=[rr,rr]),[e],'p','e')

    def test_wrong_extraction_digest_rejected(self):
        e,_=event();rr=rowreview(e)
        with self.assertRaises(ValueError):m.verify_independent_review(dict(passed=True,plan_sha256='p',extractions_sha256='other',event_reviews=[rr]),[e],'p','e')

    def test_all_asset_field_scales_retained(self):
        e,c=event();e['fields']['prior_total_assets']['scale']='1000'
        before=deepcopy(e)
        out=m.assess_event(e,c,rowreview(e),sessions())
        self.assertTrue(out['ni_assets_admission']['admitted'])
        self.assertEqual(out['fields'],before['fields'])

    def test_null_reason_must_be_verified(self):
        e,_=event(missing='prior_total_assets');rr=rowreview(e);rr['fields']['prior_total_assets']['null_reason_verified']=False
        with self.assertRaises(ValueError):m.verify_independent_review(dict(passed=True,plan_sha256='p',extractions_sha256='e',event_reviews=[rr]),[e],'p','e')


class CatalogTests(unittest.TestCase):
    def setup_catalog(self,years=(2020,2021)):
        events=[assessed(year=y) for y in years]
        cats=m.annual_catalog({'annual_documents':[catalogrow(e) for e in events]},
                              {e['event_id']:1 for e in events},sessions())
        return cats,{e['event_id']:e for e in events}

    def test_latest_invalid_blocks_old(self):
        cat,ev=self.setup_catalog();last=ev[cat[-1]['event_id']]
        last['ni_assets_admission']['admitted']=False;last['ni_assets_admission']['reasons']=['bad']
        self.assertEqual(m.select_latest(cat,ev,'AMZN','2022-03-01')['reason'],'latest_annual_event_held')

    def test_latest_unprocessed_blocks_old(self):
        cat,ev=self.setup_catalog();del ev[cat[-1]['event_id']]
        self.assertEqual(m.select_latest(cat,ev,'AMZN','2022-03-01')['reason'],'latest_annual_event_unprocessed')

    def test_future_not_selected(self):
        cat,ev=self.setup_catalog()
        self.assertEqual(m.select_latest(cat,ev,'AMZN','2021-03-01')['event_id'],cat[0]['event_id'])

    def test_no_release_day_eligibility(self):
        cat,ev=self.setup_catalog((2020,))
        self.assertEqual(m.select_latest(cat,ev,'AMZN','2021-02-01')['reason'],'no_known_annual_event')

    def test_550_inclusive_551_stale(self):
        cat,ev=self.setup_catalog((2020,));end=date(2020,12,31)
        self.assertTrue(m.select_latest(cat,ev,'AMZN',(end+timedelta(days=550)).isoformat())['available'])
        self.assertEqual(m.select_latest(cat,ev,'AMZN',(end+timedelta(days=551)).isoformat())['reason'],'latest_annual_event_stale')

    def test_unverified_clock_is_blocker_not_omitted(self):
        cat,ev=self.setup_catalog();cat[-1]['clock_review']['verified']=False
        self.assertEqual(m.select_latest(cat,ev,'AMZN','2022-03-01')['reason'],'latest_annual_clock_unverified')

    def test_same_publication_tie_is_ambiguous(self):
        cat,ev=self.setup_catalog((2020,));other=deepcopy(cat[0]);other['event_id']='different';cat.append(other)
        self.assertEqual(m.select_latest(cat,ev,'AMZN','2021-03-01')['reason'],'ambiguous_latest_annual_event')

    def test_no_tie_break_by_availability(self):
        cat,ev=self.setup_catalog((2020,));other=deepcopy(cat[0]);other['event_id']='different';other['available_session']='2021-02-03';cat.append(other)
        self.assertEqual(m.select_latest(cat,ev,'AMZN','2021-03-01')['reason'],'ambiguous_latest_annual_event')

    def test_catalog_metadata_preserved(self):
        e=assessed();row=catalogrow(e)
        out=m.annual_catalog({'annual_documents':[row]}, {e['event_id']:1},sessions())[0]
        self.assertEqual({k:out[k] for k in row},row)

    def test_duplicate_catalog_rejected(self):
        e=assessed();row=catalogrow(e)
        with self.assertRaises(ValueError):m.annual_catalog({'annual_documents':[row,row]}, {},sessions())

    def test_unknown_stock_remains_in_coverage(self):
        meta=dict(dates=['2021-03-01'],symbols=[['UNKNOWN','ETF']],etfs=['ETF'])
        census=dict(stocks=[{'symbol':'UNKNOWN'}],monthly_candidate_coverage=[{'session':'2021-03-01','symbol':'UNKNOWN'}])
        out=m.coverage(meta,census,[],[],strict=False)
        self.assertEqual(out['counts']['stock_slots'],1)
        self.assertEqual(out['stocks'],['UNKNOWN'])

    def test_symbol_major_census_reordered_to_original_meta_axis(self):
        meta=dict(dates=['2021-03-01'],symbols=[['A','B','ETF']],etfs=['ETF'])
        census=dict(stocks=[{'symbol':'A'},{'symbol':'B'}],monthly_candidate_coverage=[{'session':'2021-03-01','symbol':'B'},{'session':'2021-03-01','symbol':'A'}])
        result=m.coverage(meta,census,[],[],strict=False)
        self.assertEqual([row['symbol'] for row in result['monthly_candidate_coverage']],['A','B'])

    def test_missing_duplicate_or_extra_slot_identity_rejected(self):
        meta=dict(dates=['2021-03-01'],symbols=[['A','B','ETF']],etfs=['ETF'])
        rows=[{'session':'2021-03-01','symbol':'A'},{'session':'2021-03-01','symbol':'B'}]
        for changed in (rows[:1],rows+[rows[0]],rows+[{'session':'2021-03-01','symbol':'C'}],
                        [rows[0],{'session':'2021-04-01','symbol':'B'}]):
            census=dict(stocks=[{'symbol':'A'},{'symbol':'B'}],monthly_candidate_coverage=changed)
            with self.assertRaises(ValueError):m.coverage(meta,census,[],[],strict=False)


class PackageTests(unittest.TestCase):
    def test_full_generated_package_26_32_73_3904_and_original_unchanged(self):
        with tempfile.TemporaryDirectory() as root:
            pins=fixture(root);before=Path(root,m.OLD_DATA).read_bytes()
            out,evidence=m.build(root,*pins)
            self.assertEqual(len(out['DATASET.json']['events']),26)
            self.assertEqual(len(out['DATASET.json']['latest_event_catalog']),32)
            self.assertEqual(out['COVERAGE.json']['counts']['stock_slots'],3904)
            self.assertEqual(out['COVERAGE.json']['counts']['stocks'],73)
            self.assertEqual(Path(root,m.OLD_DATA).read_bytes(),before)
            self.assertEqual(out['ADMISSION.json']['actual_factors_calculated'],0)
            m.publish(root,out,evidence)
            with self.assertRaises(ValueError):m.publish(root,out,evidence)

    def test_source_hash_tamper_fails_before_output(self):
        with tempfile.TemporaryDirectory() as root:
            pins=fixture(root);Path(root,'source.html').write_text('changed')
            with self.assertRaises(ValueError):m.build(root,*pins)
            self.assertFalse(Path(root,m.OUT).exists())

    def test_review_hash_tamper_fails(self):
        with tempfile.TemporaryDirectory() as root:
            pins=fixture(root);Path(root,m.REVIEW).write_text('{}')
            with self.assertRaises(ValueError):m.build(root,*pins)

    def test_independent_review_must_bind_contexts(self):
        with tempfile.TemporaryDirectory() as root:
            a,b,c=fixture(root)
            rr=json.loads(Path(root,m.REVIEW).read_text())
            del rr['input_hashes'][m.CONTEXTS['AMZN']]
            b=readwrite(root,m.REVIEW,rr)
            with self.assertRaisesRegex(ValueError,'context/receipt binding missing'):m.build(root,a,b,c)

    def test_independent_review_must_bind_receipts(self):
        with tempfile.TemporaryDirectory() as root:
            a,b,c=fixture(root)
            rr=json.loads(Path(root,m.REVIEW).read_text())
            del rr['input_hashes']['receipt.json']
            b=readwrite(root,m.REVIEW,rr)
            with self.assertRaisesRegex(ValueError,'context/receipt binding missing'):m.build(root,a,b,c)

    def test_independent_review_must_bind_raw_sources(self):
        with tempfile.TemporaryDirectory() as root:
            a,b,c=fixture(root)
            rr=json.loads(Path(root,m.REVIEW).read_text())
            del rr['input_hashes']['source.html']
            b=readwrite(root,m.REVIEW,rr)
            with self.assertRaisesRegex(ValueError,'raw-source binding missing'):m.build(root,a,b,c)

    def test_wrong_explicit_policy_sha_fails(self):
        with tempfile.TemporaryDirectory() as root:
            a,b,_=fixture(root)
            with self.assertRaises(ValueError):m.build(root,a,b,'f'*64)

    def test_original_population_gate_not_relaxed(self):
        old=dict(events=[dict(event_id=str(i),symbol=['AAPL','AMD','MSFT','NVDA'][i%4],stock_compensation_admission={'admitted':i<32}) for i in range(44)])
        with self.assertRaises(ValueError):m.old_population(old)

    def test_evidence_detects_changed_after_initial_read(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root,'x');path.write_text('a');e=m.Evidence(root);e.pin('x');path.write_text('b')
            with self.assertRaises(ValueError):e.stable()

    def test_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ValueError):m.Evidence(root).path('../escape')

    def test_policy_cannot_remove_adoption_hold(self):
        with tempfile.TemporaryDirectory() as root:
            fixture(root);p=json.loads(Path(root,m.POLICY).read_text());p['accounting']['AMZN_FY2019']='Admit'
            with self.assertRaises(ValueError):m.policy_contract(p)


if __name__=='__main__':
    unittest.main(verbosity=2)
