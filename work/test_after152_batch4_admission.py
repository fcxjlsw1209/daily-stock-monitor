"""Synthetic-only batch4 source admission tests; no real source reads."""
from copy import deepcopy
from datetime import date,timedelta
import json
from pathlib import Path
import tempfile
import unittest

import after152_batch4_admission as m
import test_after151_batch3_admission as prior


def sessions():
    start,end=date(2015,1,1),date(2028,1,1)
    return [start+timedelta(days=i) for i in range((end-start).days) if (start+timedelta(days=i)).weekday()<5]


def example(symbol='MU',year=2016,missing=False):
    end=date(year,8,28) if symbol=='MU' else date(year,12,24)
    prior_end=end-timedelta(days=364)
    start=prior_end+timedelta(days=1)
    release=(end+timedelta(days=35)).isoformat()
    available=m.first_session(release,sessions()) or (m.day(release)+timedelta(days=1)).isoformat()
    source=dict(path='source.html',sha256='a'*64,url='https://example.test/source',receipt_path='receipt.json',receipt_sha256='b'*64)
    holds=['monetary_cells_and_sign_geometry_not_yet_independently_verified',
           'currency_ISO_and_first_public_version_not_certified',
           'accounting_notes_require_field_specific_independent_review',
           'proposed_annual_start_boundary_assumption_not_yet_approved']
    clock_ok=m.first_session(release,sessions()) == available
    if not clock_ok:holds.append('publication_outside_pinned_calendar_range')
    period=dict(period_start=str(start),period_end=str(end),days_inclusive=364)
    fields={f:dict(displayed_amount='(10)' if f=='annual_net_income' else '100',
                   value_decimal='-10' if f=='annual_net_income' else '100',scale='1000000',
                   currency_literal='$',currency_iso=None,source=deepcopy(source),available_from=available,
                   locator={'row':3},as_of=str(prior_end if f=='prior_total_assets' else end),
                   period=deepcopy(period) if f=='annual_net_income' else None,fact_admitted=False) for f in m.FIELDS}
    if missing:fields=dict.fromkeys(m.FIELDS)
    c=dict(symbol=symbol,event_id=symbol+'_'+release,fiscal_year=year,publication_date=release,available_session=available,
           available_session_verified=clock_ok,source_kind='HTML_original',original_report_bytes_verified=True,
           official_actual_period_end=str(end),source=source,evidence_path=source['path'],evidence_sha256=source['sha256'],
           source_url=source['url'],context_status=m.STATUS,context_holds=holds,accounting_context_numbers_redacted=['keep note'],
           accounting_terms=['synthetic accounting note'],identity_anchors=[m.ISSUERS[symbol]],
           fields={f:{'scale':'1000000'} for f in m.FIELDS},
           periods=dict(current_end=str(end),prior_end=str(prior_end),proposed_current_start=str(start)))
    e=dict(symbol=symbol,event_id=c['event_id'],fiscal_year=year,publication_date=release,available_from=available,
           source=deepcopy(source),context_holds=deepcopy(holds),accounting_context_numbers_redacted=['keep note'],fields=fields,
           accounting_review_flags=deepcopy(c['accounting_terms']),
           periods=dict(current=period,comparison=dict(period_end=str(prior_end)),
                        start_basis='derived_same_report_prior_annual_end_plus_one_not_directly_quoted',
                        start_assumption_certified=False,assumption_id=m.ASSUMPTIONS[symbol]),
           identity={'issuer':m.ISSUERS[symbol],'symbol':symbol,'source_anchors':deepcopy(c['identity_anchors']),
                     'certified_daily_security_master':False},
           fact_admitted=False,model_admitted=False,strategy_admitted=False,first_public_version_verified=False,
           first_public_time_verified=False,complete_correction_history_verified=False,certified_daily_security_master=False,
           financial_calculations=0,gaps=[dict(field=f,reason='synthetic missing cell') for f in m.FIELDS] if missing else [],
           complete_required_fields=not missing)
    return e,c


def accounting(e,resolved=True):
    return dict(event_id=e['event_id'],source=deepcopy(e['source']),status='resolved' if resolved else 'unresolved',
                scope=m.ACCOUNTING_SCOPE,reasons=[],field_checks={f:True for f in m.FIELDS},current_prior_common_basis=True,
                basis='Synthetic same-report basis',evidence=[dict(path=e['source']['path'],sha256=e['source']['sha256'],
                locator=dict(kind='document_text',start=0,end=9),quote='synthetic')])


def review_row(e):
    r=dict(event_id=e['event_id'],symbol=e['symbol'],fiscal_year=e['fiscal_year'],passed=True,metadata_verified=True,
           identity_verified=True,issuer_id=m.ISSUERS[e['symbol']],security_id=e['symbol'],periods_verified=True,currency_verified=True,
           retained_context_holds=deepcopy(e['context_holds']),accounting_context_numbers_redacted=deepcopy(e['accounting_context_numbers_redacted']),
           source=deepcopy(e['source']),available_from=e['available_from'],release_date=e['publication_date'],
           clock_verified=m.first_session(e['publication_date'],sessions())==e['available_from'],
           boundary_assumption_verified=all(e['fields'].values()),assumption_id=m.ASSUMPTIONS[e['symbol']],
           start_is_source_quoted=False,derived_start=e['periods']['current']['period_start'],
           boundary_evidence={f:dict(table=1,date=cell['as_of'],headers=[{'text':'synthetic date'}]) for f,cell in e['fields'].items() if cell},
           accounting_review=accounting(e),fields={})
    for f,cell in e['fields'].items():
        r['fields'][f]=(dict(is_null=True,null_reason_verified=True,null_classification='synthetic_missing_cell') if cell is None else
                        dict(is_null=False,verified=True,source_value_present=True,value_and_sign_verified=True,
                             units_verified=True,period_verified=True,source_locator_verified=True))
    return r


def catalogrow(e):
    kind = ('issuer_original_release_full_bytes' if e['symbol']=='MU' else
            'issuer_distributed_original_prnewswire_release')
    if e['symbol']=='PEP' and e['fiscal_year']>=2019:
        kind = ('public_indexed_issuer_release_excerpt' if e['fiscal_year']>=2024
                else 'public_index_observation_not_original_bytes')
    return dict(event_id=e['event_id'],symbol=e['symbol'],fiscal_year=e['fiscal_year'],fiscal_quarter=4,
                publication_date=e['publication_date'],available_session=e['available_from'],
                official_actual_period_end=e['periods']['current']['period_end'],evidence_path=e['source']['path'],
                evidence_sha256=e['source']['sha256'],source_url=e['source']['url'],source_kind=kind)


def write(root,path,value):
    p=Path(root)/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,sort_keys=True)+'\n');return m.digest(p)


def prior_packages(root):
    prior.prior_packages(root)
    original,batch1,batch2=m.previous.preserved_references(m.Evidence(root))
    data=dict(method=m.METHOD,events=[dict(event_id='b3_'+str(i),symbol='KO' if i<6 else 'LRCX',
               ni_assets_admission={'admitted':i<5}) for i in range(10)],
               latest_event_catalog=[{'id':i} for i in range(32)],original_four_issuer_reference=original,
               prior_batch1_reference=batch1,prior_batch2_reference=batch2)
    inputs={m.BATCH2+'/'+name:m.digest(Path(root,m.BATCH2,name)) for name in ('DATASET.json','ADMISSION.json','MANIFEST.json')}
    outs={'DATASET.json':write(root,m.BATCH3+'/DATASET.json',data),
          'ADMISSION.json':write(root,m.BATCH3+'/ADMISSION.json',dict(passed=True,counts=dict(events=10,admitted=5,held=5),input_hashes=inputs)),
          'COVERAGE.json':write(root,m.BATCH3+'/COVERAGE.json',dict(counts={'stock_slots':3904}))}
    write(root,m.BATCH3+'/MANIFEST.json',dict(input_hashes=inputs,outputs=outs))


def fixture(root):
    root=Path(root);prior_packages(root)
    raw_sha=write(root,'source.html',{'synthetic':True});receipt_sha=write(root,'receipt.json',{})
    docs,events,catalog=[],[],[]
    for symbol,years in (('MU',range(2019,2026)),('PEP',range(2010,2026))):
        for year in years:
            e,c=example(symbol,year)
            c['source']['sha256']=raw_sha;c['source']['receipt_sha256']=receipt_sha
            c['evidence_sha256']=raw_sha;e['source']=deepcopy(c['source'])
            for cell in e['fields'].values():cell['source']=deepcopy(c['source'])
            selected=(symbol=='MU' and 2024<=year<=2025) or (symbol=='PEP' and 2010<=year<=2016)
            if not selected:c['context_status']='unprocessed_source'
            if symbol=='PEP' and year>=2019:c.update(source_kind='index_observation_not_original_report',original_report_bytes_verified=False)
            row=catalogrow(e);c.update({k:v for k,v in row.items() if k!='source_kind'});catalog.append(row);docs.append(c)
            if selected:events.append(e)
    csha=write(root,m.CONTEXT,dict(events=docs));cref=dict(path=m.CONTEXT,sha256=csha)
    policy=dict(schema_version=1,policy_id='mu_pep_annual_ni_same_report_assets_batch4_v1',
        stage='prospective_source_policy_not_strategy_registration',definition=m.DEFINITION,required_fields=list(m.FIELDS),
        source_scope=dict(symbols=['MU','PEP'],catalog_events=23,html_context_reviewed_events=9,no_automatic_admission=True,source_context=cref),
        known_time={'maximum_age_days':550},periods={'boundary_assumptions':{s:dict(enabled=True,not_a_quoted_fact=True,assumption_id=m.ASSUMPTIONS[s],days_inclusive=[364,371]) for s in m.SYMBOLS}},
        preservation=dict(all_context_holds=True,first_public_version_verified=False,first_public_time_verified=False,complete_correction_history_verified=False,certified_daily_security_master=False),
        input_hashes={m.CONTEXT:csha})
    policy['fixed_catalog_metadata']=m.catalog_projection(dict(events=docs))
    psha=write(root,m.POLICY,policy)
    names=['MU','PEP']+['STOCK%02d'%i for i in range(71)];etfs=['ETF%02d'%i for i in range(17)]
    dates=['%04d-%02d-01'%(2016+(7+i)//12,(7+i)%12+1) for i in range(122)]
    panels=[[names[(i+j)%73] for j in range(32)]+etfs for i in range(122)]
    slots=[dict(session=d,symbol=s) for d,p in zip(dates,panels) for s in p if s not in etfs]
    write(root,m.CENSUS,dict(annual_documents=catalog,stocks=[dict(symbol=s) for s in names],monthly_candidate_coverage=list(reversed(slots))))
    write(root,m.META,dict(dates=dates,symbols=panels,etfs=etfs));write(root,m.SESSIONS,dict(sessions=[str(s) for s in sessions()]))
    for path in m.source_files()+('code.py','work/after152_batch4_independent_review.py','work/test_after152_batch4_independent_review.py'):
        write(root,path,{'synthetic_code':True})
    inputs={p:m.digest(root/p) for p in (m.POLICY,m.CONTEXT,m.CENSUS,m.META,m.SESSIONS,'source.html','receipt.json',m.BATCH3+'/DATASET.json',m.BATCH3+'/ADMISSION.json',m.BATCH3+'/MANIFEST.json')}
    plan=dict(inputs=inputs,code={'code.py':m.digest(root/'code.py')},catalog_ref=cref,context=m.CONTEXT,policy=m.POLICY,
              records=[dict(event_id=e['event_id']) for e in events],catalog_ids=[e['event_id'] for e in docs],catalog_events=23,catalog_metadata=m.catalog_projection(dict(events=docs)))
    plan_sha=write(root,m.BATCH+'/PLAN.json',plan)
    extraction=dict(plan_sha256=plan_sha,catalog_ref=cref,catalog_ids=plan['catalog_ids'],catalog_metadata=deepcopy(plan['catalog_metadata']),events=events,
                    status='source_values_extracted_pending_independent_review',new_financial_admissions=0,new_strategy_trials=0,ratios_computed=0,returns_computed=0)
    esha=write(root,m.BATCH+'/EXTRACTIONS.json',extraction)
    ar=dict(schema_version=1,stage='independent_field_specific_accounting_review',passed=True,plan_sha256=plan_sha,
            extractions_sha256=esha,input_hashes={'source.html':raw_sha},events=[accounting(e) for e in events])
    asha=write(root,m.ACCOUNTING,ar)
    review=dict(passed=True,stage='independent_source_value_and_null_classification_review',plan_sha256=plan_sha,
                extractions_sha256=esha,event_reviews=[review_row(e) for e in events],
                financial_calculations=0,factor_calculations=0,returns_computed=0,new_admissions=0,
                input_hashes=dict(inputs,**{m.BATCH+'/PLAN.json':plan_sha,m.BATCH+'/EXTRACTIONS.json':esha,m.ACCOUNTING:asha}),
                source_hashes={p:m.digest(root/p) for p in ('work/after152_batch4_independent_review.py','work/test_after152_batch4_independent_review.py')})
    review['input_hashes'].update(review['source_hashes']);rsha=write(root,m.REVIEW,review)
    return plan_sha,esha,rsha,psha


class AdmissionTests(unittest.TestCase):
    def assess(self,symbol='MU',year=2016):
        e,c=example(symbol,year);return m.assess_event(e,c,review_row(e),sessions())

    def test_two_issuer_happy_paths_and_signed_ni(self):
        for symbol in ('MU','PEP'):
            for value in ('0','-1','1.5'):
                e,c=example(symbol);e['fields']['annual_net_income']['value_decimal']=value
                self.assertTrue(m.assess_event(e,c,review_row(e),sessions())['ni_assets_admission']['admitted'])

    def test_original_holds_identity_and_source_retained(self):
        e,c=example();before=deepcopy(e);out=m.assess_event(e,c,review_row(e),sessions())
        self.assertEqual(e,before);self.assertEqual(out['context_holds'],e['context_holds'])
        self.assertEqual(out['identity']['issuer'],e['identity']['issuer']);self.assertEqual(out['source'],e['source'])
        self.assertFalse(out['fact_admitted'])

    def test_missing_accounting_is_hold(self):
        e,c=example();r=review_row(e);del r['accounting_review']
        self.assertIn('accounting_comparability_unresolved',m.assess_event(e,c,r,sessions())['ni_assets_admission']['reasons'])

    def test_unresolved_accounting_is_hold(self):
        e,c=example();r=review_row(e);r['accounting_review']['status']='unresolved'
        self.assertFalse(m.assess_event(e,c,r,sessions())['ni_assets_admission']['admitted'])

    def test_forged_resolved_accounting_rejected(self):
        for key,value in [('scope','other'),('field_checks',{}),('current_prior_common_basis',False),('evidence',[]),('basis','')]:
            e,c=example();r=review_row(e);r['accounting_review'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):m.assess_event(e,c,r,sessions())

    def test_accounting_evidence_source_mismatch(self):
        e,c=example();r=review_row(e);r['accounting_review']['evidence'][0]['sha256']='f'*64
        with self.assertRaises(ValueError):m.assess_event(e,c,r,sessions())

    def test_unknown_hold_never_cleared(self):
        e,c=example();e['context_holds'].append('new_undecided_issue');c['context_holds'].append('new_undecided_issue')
        self.assertIn('retained_context_hold:new_undecided_issue',m.assess_event(e,c,review_row(e),sessions())['ni_assets_admission']['reasons'])

    def test_early_calendar_stays_held(self):
        self.assertIn('unverified_exchange_calendar',self.assess('PEP',2011)['ni_assets_admission']['reasons'])

    def test_false_clock_claim_rejected(self):
        e,c=example('PEP',2011);r=review_row(e);r['clock_verified']=True
        with self.assertRaises(ValueError):m.assess_event(e,c,r,sessions())

    def test_boundary_not_verified_is_hold(self):
        for symbol in ('MU','PEP'):
            e,c=example(symbol);r=review_row(e);r['boundary_assumption_verified']=False
            self.assertFalse(m.assess_event(e,c,r,sessions())['ni_assets_admission']['admitted'])

    def test_wrong_week_duration_is_hold(self):
        e,c=example('MU');e['fields']['annual_net_income']['period']['days_inclusive']=365
        self.assertFalse(m.assess_event(e,c,review_row(e),sessions())['ni_assets_admission']['admitted'])

    def test_certified_boundary_claim_is_hold(self):
        e,c=example();e['periods']['start_assumption_certified']=True
        self.assertFalse(m.assess_event(e,c,review_row(e),sessions())['ni_assets_admission']['admitted'])

    def test_missing_fields_retained_and_held(self):
        e,c=example(missing=True);out=m.assess_event(e,c,review_row(e),sessions())
        self.assertFalse(out['ni_assets_admission']['admitted']);self.assertTrue(all(v is None for v in out['fields'].values()))

    def test_verified_boundary_with_frozen_null_values_still_held(self):
        e,c=example(missing=True);r=review_row(e)
        e['gaps']=[dict(field=f,reason='non_gaap_statement') for f in m.FIELDS]
        r['boundary_assumption_verified']=True
        r['boundary_evidence']={f:dict(table=1,headers=[{'text':'annual date'}],
            date=e['periods']['comparison' if f=='prior_total_assets' else 'current']['period_end']) for f in m.FIELDS}
        r['periods_verified']=False;r['currency_verified']=False
        doc=dict(passed=True,plan_sha256='a',extractions_sha256='b',event_reviews=[r])
        verified=m.verify_independent_review(doc,[e],'a','b')
        result=m.assess_event(e,c,verified[e['event_id']],sessions())
        self.assertFalse(result['ni_assets_admission']['admitted'])
        self.assertEqual(result['fields'],dict.fromkeys(m.FIELDS))
        self.assertEqual(result['gaps'],e['gaps'])
        self.assertEqual(result['context_holds'],e['context_holds'])
        self.assertTrue(result['source_review']['boundary_assumption_verified'])
        for f in m.FIELDS:self.assertIn('missing_'+f,result['ni_assets_admission']['reasons'])

    def test_null_value_boundary_still_requires_period_and_locator(self):
        for mutation in ('date','table','headers'):
            e,c=example(missing=True);r=review_row(e);r['boundary_assumption_verified']=True
            r['boundary_evidence']={f:dict(table=1,headers=[{'text':'annual date'}],
                date=e['periods']['comparison' if f=='prior_total_assets' else 'current']['period_end']) for f in m.FIELDS}
            r['boundary_evidence']['prior_total_assets'][mutation]={'date':'2010-12-31','table':True,'headers':[]}[mutation]
            doc=dict(passed=True,plan_sha256='a',extractions_sha256='b',event_reviews=[r])
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                m.verify_independent_review(doc,[e],'a','b')

    def test_nonnull_value_asof_must_equal_independent_boundary(self):
        e,c=example();r=review_row(e)
        e['fields']['annual_net_income']['as_of']='2016-03-31'
        doc=dict(passed=True,plan_sha256='a',extractions_sha256='b',event_reviews=[r])
        with self.assertRaises(ValueError):m.verify_independent_review(doc,[e],'a','b')

    def test_nonpositive_assets_and_nonfinite_rejected_for_use(self):
        for value in ('0','-1','NaN','Infinity'):
            e,c=example();e['fields']['prior_total_assets']['value_decimal']=value
            self.assertFalse(m.assess_event(e,c,review_row(e),sessions())['ni_assets_admission']['admitted'])

    def test_cross_currency_source_time_scale_hold(self):
        for key,value in [('currency_iso','USD'),('currency_literal','EUR'),('scale','1000'),('source',{}),('available_from','2020-01-01'),('as_of','2016-03-31')]:
            e,c=example();e['fields']['current_total_assets'][key]=value
            self.assertFalse(m.assess_event(e,c,review_row(e),sessions())['ni_assets_admission']['admitted'])

    def test_identity_and_hold_changes_fail_hard(self):
        e,c=example();r=review_row(e);r['issuer_id']='other'
        with self.assertRaises(ValueError):m.assess_event(e,c,r,sessions())
        e,c=example();c['context_holds'].append('cannot drop')
        with self.assertRaises(ValueError):m.assess_event(e,c,review_row(e),sessions())

    def test_review_metadata_field_and_boundary_flags(self):
        for kind in ('field','metadata','boundary'):
            e,c=example();r=review_row(e)
            if kind=='field':r['fields']['annual_net_income']['value_and_sign_verified']=False
            if kind=='metadata':r['metadata_verified']=False
            if kind=='boundary':r['boundary_evidence']['annual_net_income']['date']='2017-01-01'
            doc=dict(passed=True,plan_sha256='a',extractions_sha256='b',event_reviews=[r])
            with self.assertRaises(ValueError):m.verify_independent_review(doc,[e],'a','b')

    def test_latest_unprocessed_blocks_old(self):
        e,c=example();old=m.assess_event(e,c,review_row(e),sessions());new=deepcopy(catalogrow(e))
        new.update(event_id='MU_new',publication_date='2017-03-01',available_session='2017-03-02',clock_review=dict(verified=True))
        first=dict(catalogrow(e),clock_review=dict(verified=True))
        out=m.select_latest([first,new],{old['event_id']:old},'MU','2017-03-03')
        self.assertEqual(out['reason'],'latest_annual_event_unprocessed')

    def test_latest_held_and_unverified_block(self):
        e,c=example();held=m.assess_event(e,c,review_row(e),sessions());held['ni_assets_admission']['admitted']=False
        row=dict(catalogrow(e),clock_review=dict(verified=True))
        self.assertEqual(m.select_latest([row],{held['event_id']:held},'MU','2017-03-03')['reason'],'latest_annual_event_held')
        row['clock_review']['verified']=False
        self.assertEqual(m.select_latest([row],{held['event_id']:held},'MU','2017-03-03')['reason'],'latest_annual_clock_unverified')

    def test_550day_boundary(self):
        e,c=example();a=m.assess_event(e,c,review_row(e),sessions());row=dict(catalogrow(e),clock_review=dict(verified=True))
        end=m.day(e['periods']['current']['period_end'])
        self.assertTrue(m.select_latest([row],{a['event_id']:a},'MU',str(end+timedelta(days=550)))['available'])
        self.assertEqual(m.select_latest([row],{a['event_id']:a},'MU',str(end+timedelta(days=551)))['reason'],'latest_annual_event_stale')

    def test_full_synthetic_build_preserves_prior_refs_and_axis(self):
        with tempfile.TemporaryDirectory() as root:
            args=fixture(root);objects,evidence=m.build(root,*args)
            data=objects['DATASET.json'];cov=objects['COVERAGE.json']
            self.assertEqual(len(data['events']),9);self.assertEqual(len(data['latest_event_catalog']),23)
            self.assertEqual(cov['counts']['stock_slots'],3904)
            self.assertEqual(data['prior_batch3_reference']['events'],10)
            self.assertEqual(data['prior_batch2_reference']['events'],19)
            self.assertEqual(data['prior_batch1_reference']['events'],26)
            self.assertEqual(data['original_four_issuer_reference']['events'],44)
            self.assertEqual(objects['ADMISSION.json']['counts']['unprocessed_annual_catalog_events'],14)
            self.assertTrue(all(data[k]==0 for k in ('scores_computed','ratios_computed','returns_computed','new_strategy_trials')))

    def test_write_once_manifest_and_input_stability(self):
        with tempfile.TemporaryDirectory() as root:
            objects,evidence=m.build(root,*fixture(root));manifest=m.publish(root,objects,evidence)
            self.assertEqual(set(manifest['outputs']),set(objects))
            with self.assertRaises(ValueError):m.publish(root,objects,evidence)
            Path(root,'source.html').write_text('tampered')
            with self.assertRaises(ValueError):evidence.stable()

    def test_wrong_pinned_hash_blocks_build(self):
        with tempfile.TemporaryDirectory() as root:
            args=list(fixture(root));args[1]='f'*64
            with self.assertRaises(ValueError):m.build(root,*args)

    def test_detached_accounting_report_blocks_build(self):
        with tempfile.TemporaryDirectory() as root:
            args=fixture(root);p=Path(root,m.ACCOUNTING);data=json.loads(p.read_text());data['events'][0]['basis']='tampered';p.write_text(json.dumps(data))
            with self.assertRaises(ValueError):m.build(root,*args)

    def test_prior_package_tamper_blocks_build(self):
        with tempfile.TemporaryDirectory() as root:
            args=fixture(root);p=Path(root,m.BATCH3+'/DATASET.json');data=json.loads(p.read_text());data['events'].pop();p.write_text(json.dumps(data))
            with self.assertRaises(ValueError):m.build(root,*args)

    def test_context_population_requires_23_and_exact9(self):
        with tempfile.TemporaryDirectory() as root:
            fixture(root);context=json.loads(Path(root,m.CONTEXT).read_text());context['events'].pop()
            with self.assertRaises(ValueError):m.context_map(context)

    def test_census_context_classification_uses_explicit_mapping(self):
        for raw_kind, kind, has_bytes in (
            ('issuer_original_release_full_bytes', 'HTML_original', True),
            ('issuer_distributed_original_prnewswire_release', 'HTML_original', True),
            ('public_index_observation_not_original_bytes', 'index_observation_not_original_report', False),
            ('public_indexed_issuer_release_excerpt', 'index_observation_not_original_report', False),
        ):
            raw=dict(source_kind=raw_kind,event_id='same',evidence_sha256='a'*64)
            ctx=dict(raw,source_kind=kind,original_report_bytes_verified=has_bytes)
            with self.subTest(raw_kind=raw_kind):
                m.validate_census_context(raw,ctx)
                before=deepcopy(raw)
                bad=dict(ctx,original_report_bytes_verified=not has_bytes)
                with self.assertRaises(ValueError):m.validate_census_context(raw,bad)
                bad=dict(ctx,source_kind='HTML_original' if not has_bytes else 'index_observation_not_original_report')
                with self.assertRaises(ValueError):m.validate_census_context(raw,bad)
                for field in ('event_id','evidence_sha256'):
                    with self.assertRaises(ValueError):
                        m.validate_census_context(raw,dict(ctx,**{field:'changed'}))
                self.assertEqual(before,raw)

    def test_unknown_census_source_kind_is_not_silently_ignored(self):
        with self.assertRaises(ValueError):
            m.validate_census_context(dict(source_kind='unknown'),dict(source_kind='HTML_original',
                                                                      original_report_bytes_verified=True))

    def test_policy_fixed_limits_cannot_change(self):
        with tempfile.TemporaryDirectory() as root:
            fixture(root);p=json.loads(Path(root,m.POLICY).read_text());p['known_time']['maximum_age_days']=551
            with self.assertRaises(ValueError):m.policy_contract(p)


    def test_latest_same_publication_tie_blocks_all_candidates(self):
        e,c=example();a=m.assess_event(e,c,review_row(e),sessions())
        first=dict(catalogrow(e),clock_review=dict(verified=True))
        second=deepcopy(first);second['event_id']='distinct_tied_event'
        r=m.select_latest([first,second],{a['event_id']:a},'MU','2017-03-03')
        self.assertEqual(r['reason'],'ambiguous_latest_annual_event')

    def test_future_report_not_known_before_availability(self):
        e,c=example();a=m.assess_event(e,c,review_row(e),sessions())
        row=dict(catalogrow(e),clock_review=dict(verified=True))
        self.assertEqual(m.select_latest([row],{a['event_id']:a},'MU',e['publication_date'])['reason'],'no_known_annual_event')
        self.assertTrue(m.select_latest([row],{a['event_id']:a},'MU',e['available_from'])['available'])

    def test_index_only_latest_kept_and_blocks_old_available_report(self):
        e,c=example('PEP');a=m.assess_event(e,c,review_row(e),sessions())
        row=dict(catalogrow(e),clock_review=dict(verified=True))
        future=dict(row,event_id='PEP_index_only',publication_date='2017-03-01',available_session='2017-03-02',
                    source_kind='index_observation_not_original_report')
        self.assertEqual(m.select_latest([row,future],{a['event_id']:a},'PEP','2017-03-03')['reason'],
                         'latest_annual_event_unprocessed')

    def test_accounting_scope_and_first_public_time_never_implicitly_certified(self):
        e,c=example();a=m.assess_event(e,c,review_row(e),sessions())
        self.assertFalse(a['first_public_time_verified']);self.assertFalse(a['first_public_version_verified'])
        self.assertFalse(a['identity']['certified_daily_security_master'])
        e['first_public_time_verified']=True
        with self.assertRaises(ValueError):m.assess_event(e,c,review_row(e),sessions())

    def test_explicit_clock_claim_matches_original_context(self):
        e,c=example();c['available_session_verified']=False
        with self.assertRaises(ValueError):m.assess_event(e,c,review_row(e),sessions())

    def test_symbol_name_or_source_anchors_cannot_be_changed(self):
        for key,value in [('symbol','OTHER'),('issuer','OTHER'),('source_anchors',['OTHER']),('certified_daily_security_master',True)]:
            e,c=example();e['identity'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):m.assess_event(e,c,review_row(e),sessions())

    def test_gap_and_null_alignment_required(self):
        for action in ('missing_gap','contradictory_gap','wrong_complete'):
            e,c=example(missing=action=='missing_gap')
            if action=='missing_gap':e['gaps']=[]
            if action=='contradictory_gap':e['gaps']=[{'field':'annual_net_income','reason':'contradictory'}]
            if action=='wrong_complete':e['complete_required_fields']=False
            with self.subTest(action=action),self.assertRaises(ValueError):m.assess_event(e,c,review_row(e),sessions())

    def test_catalog_projection_contains_exact_twelve_preserved_fields(self):
        with tempfile.TemporaryDirectory() as root:
            fixture(root);context=json.loads(Path(root,m.CONTEXT).read_text())
            projection=m.catalog_projection(context)
            self.assertEqual(len(projection),23);self.assertEqual(len(projection[0]),12)
            self.assertEqual(set(projection[0]),set(m.CATALOG_KEYS))
            self.assertEqual(projection[-1]['source_kind'],'index_observation_not_original_report')
            self.assertFalse(projection[-1]['original_report_bytes_verified'])
            self.assertEqual(projection[0]['context_holds'],context['events'][0]['context_holds'])

    def test_changed_extraction_catalog_projection_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            args=list(fixture(root));path=Path(root,m.BATCH,'EXTRACTIONS.json')
            obj=json.loads(path.read_text());obj['catalog_metadata'][0]['source_kind']='changed'
            args[1]=write(root,m.BATCH+'/EXTRACTIONS.json',obj)
            with self.assertRaises(ValueError):m.build(root,*args)

    def test_full_build_does_not_mutate_raw_fields_or_prior_files(self):
        with tempfile.TemporaryDirectory() as root:
            args=fixture(root)
            paths=[Path(root,p) for p in (m.OLD_DATA,m.OLD_ADMISSION,m.BATCH1+'/DATASET.json',
                   m.BATCH2+'/DATASET.json',m.BATCH3+'/DATASET.json',m.BATCH+'/EXTRACTIONS.json')]
            before={str(p):p.read_bytes() for p in paths};objects,evidence=m.build(root,*args)
            self.assertEqual(before,{str(p):p.read_bytes() for p in paths})
            raw=json.loads(before[str(paths[-1])])['events']
            for event,result in zip(raw,objects['DATASET.json']['events']):
                for key,value in event.items():
                    if key!='identity':self.assertEqual(result[key],value)
                for key,value in event['identity'].items():self.assertEqual(result['identity'][key],value)
            self.assertEqual(len(objects['DATASET.json']['latest_event_catalog']),23)
            self.assertEqual(objects['ADMISSION.json']['counts']['reviewed_nonnull_fields'],27)
            self.assertEqual(objects['ADMISSION.json']['counts']['reviewed_null_fields'],0)

    def test_stock_axis_identity_set_and_original_order_preserved(self):
        with tempfile.TemporaryDirectory() as root:
            args=fixture(root);objects,_=m.build(root,*args)
            meta=json.loads(Path(root,m.META).read_text());etfs=set(meta['etfs'])
            expected=[(d,s) for d,names in zip(meta['dates'],meta['symbols']) for s in names if s not in etfs]
            rows=objects['COVERAGE.json']['monthly_candidate_coverage']
            self.assertEqual([(r['session'],r['symbol']) for r in rows],expected)
            self.assertEqual(len(rows),3904);self.assertEqual(len({r['symbol'] for r in rows}),73)
            self.assertTrue(all(r['reason']=='outside_new_batch_prior_evidence_preserved' for r in rows if r['symbol'] not in m.SYMBOLS))


if __name__=='__main__':unittest.main(verbosity=2)
