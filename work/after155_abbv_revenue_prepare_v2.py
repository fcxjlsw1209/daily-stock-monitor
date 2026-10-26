"""Versioned ABBV source metadata adapter; preserves all v1 files and holds."""
import argparse
import io
import json
import re
import unittest
from bs4 import BeautifulSoup
import after155_abbv_revenue_prepare as base
import after155_abbv_revenue_audit as audit

ROOT,OUT=base.ROOT,base.OUT
SELF='work/after155_abbv_revenue_prepare_v2.py'
EVENTS='outputs/monthly_earnings_repair/EVENTS.json'
PROFILE=OUT+'EXTRACTION_PROFILE_V2.json'
TEST=OUT+'FABRICATED_TESTS_V2.json'
GAAP=re.compile(r'Reported results were prepared in accordance with (?:GAAP|generally accepted accounting principles in the United States \(GAAP\)) and include all revenue and expenses recognized during the period\.',re.I)

def canonical_event(events,meta):
    hits=[e for e in events if e.get('symbol')=='ABBV' and e.get('event_id')==meta['event_id']]
    if len(hits)!=1: raise ValueError('Missing or ambiguous canonical event')
    e=hits[0]
    for k in ['fiscal_year','source_url','publication_date','official_actual_period_end','available_session']:
        if e[k]!=meta[k]: raise ValueError('Canonical event metadata mismatch: '+k)
    if e['fiscal_quarter']!=4 or e['status']!='ready' or e['source_kind']!='issuer_original_release': raise ValueError('Canonical event scope mismatch')
    if e['evidence_sha256']!=meta['sha256'] or e['evidence_path']!=meta['path']: raise ValueError('Canonical event source mismatch')
    if e['precision']!='date_only' or e['publication_at'] is not None or e['known_by_is_actual_publication']: raise ValueError('Canonical event clock mismatch')
    return {'release_date':e['publication_date'],'fiscal_period_end_actual':e['official_actual_period_end'],
            'evidence_sha256':e['evidence_sha256'],'source_url':e['source_url'],'precision':e['precision'],
            'publication_timestamp':e['publication_at'],'publication_timezone':e['publication_timezone'],
            'publication_timezone_basis':e['publication_timezone_basis']},e

def policy(soup):
    text=base.txt(soup);matches=list(GAAP.finditer(text))
    if len(matches)!=1: raise ValueError('GAAP reported-results policy absent or ambiguous')
    m=matches[0]
    return {'excerpt':m.group(0),'start_normalized_text':m.start(),'end_normalized_text':m.end()}

def freeze():
    prior=base.load(base.PROFILE);tests=base.load(TEST)
    if not tests['successful'] or tests['script']['sha256']!=audit.bind(SELF)['sha256']: raise ValueError('Current fabricated tests missing')
    for b in prior['bindings']:audit.read(b)
    # Metadata validation only: no revenue-cell conversion in profile creation.
    events=base.load(EVENTS);metadata=[]
    for item in prior['sources']:
        s=item['source'];meta=s['metadata'];mapped,e=canonical_event(events,meta)
        soup=BeautifulSoup(audit.read(s['raw']),'html.parser')
        metadata.append({'fiscal_year':meta['fiscal_year'],'canonical_event_id':e['event_id'],
                         'canonical_event_fields':e,'reported_gaap_policy':policy(soup)})
    print(json.dumps(audit.write('EXTRACTION_PROFILE_V2.json',{
        'created_utc':base.now(),'stage':'FROZEN_V2_METADATA_ADAPTER_BEFORE_V2_EXTRACTION',
        'bindings':[audit.bind(p) for p in [SELF,base.PROFILE,base.SELF,EVENTS,TEST,OUT+'EXTRACTION.json']],
        'source_profiles_unchanged':True,'metadata':metadata,
        'reason':'V1 used incomplete batch events and a shortened GAAP phrase. Use the complete canonical event catalog already underlying the coverage census, with exact field-by-field bindings, and either explicit GAAP policy spelling. No source additions, monetary selector changes, value-driven rules, historical backfill or admission changes.',
        'currency_and_accounting_holds_unchanged':True})))

def extract():
    profile=base.load(PROFILE);prior=base.load(base.PROFILE);scope=base.load(base.SCOPE);base.assert_inputs(scope)
    for b in profile['bindings']+prior['bindings']:audit.read(b)
    cal=base.load(base.CALENDAR);sessions=base.load(base.SESSIONS)
    if sessions['plan_sha256']!=audit.bind(base.CALENDAR)['sha256']:raise ValueError('Calendar mismatch')
    for p,h in cal['bindings'].items():
        if audit.bind(p)['sha256']!=h:raise ValueError('Calendar dependency mismatch: '+p)
    events=base.load(EVENTS);reports=[]
    for item in prior['sources']:
        s=item['source'];meta=s['metadata'];year=meta['fiscal_year']
        r={'symbol':'ABBV','fiscal_year':year,'source':s,'extraction_status':'HOLD',
           'revenue_admitted':False,'accounting_chain_admitted':False,'strategy_admitted':False,'financial_features_computed':False}
        try:
            raw=audit.read(s['raw']);receipt=json.loads(audit.read(s['receipt']));base.receipt_check(raw,receipt,meta)
            soup=BeautifulSoup(raw,'html.parser');event,canonical=canonical_event(events,meta)
            clock=base.publication(soup,meta,receipt,event,sessions['sessions'],cal['start'])
            clock['canonical_event_id']=canonical['event_id']
            clock['canonical_event_known_by_at']=canonical['known_by_at']
            clock['known_by_is_actual_publication']=False
            gaap=policy(soup);facts=base.revenue_cells(soup,year,item['statement_profile']);notes=base.note_contexts(soup)
            notes['reported_gaap_policy']=gaap
            for f in facts.values():f['publication_clock']=clock;f['source_raw_sha256']=s['raw']['sha256']
            holds=['CURRENCY_ISO_UNRESOLVED','ACCOUNTING_SCOPE_NOT_INDEPENDENTLY_REVIEWED','HISTORICAL_FIRST_PUBLIC_BYTES_NOT_VERIFIED']
            if clock['available_session'] is None:holds.append('PUBLICATION_PRECEDES_PINNED_CALENDAR')
            r.update({'extraction_status':'EXACT_REPORTED_PAIR_EXTRACTED_ADMISSION_HELD','revenue':facts,'publication_clock':clock,
                      'receipt_verified':True,'statement_profile':item['statement_profile'],'context':notes,'holds':holds})
        except (ValueError,KeyError,IndexError) as e:r.update({'holds':['SOURCE_EXTRACTION_FAILURE'],'error':str(e)})
        reports.append(r)
    successes=sum('revenue' in r for r in reports)
    print(json.dumps(audit.write('EXTRACTION_V2.json',{'created_utc':base.now(),'stage':'SOURCE_ONLY_EXTRACTION_COMPLETE',
        'profile':audit.bind(PROFILE),'source_count':len(reports),'exact_report_pairs':successes,'exact_amount_cells':2*successes,
        'failed_sources':len(reports)-successes,'currency_certified':0,'revenue_admitted':0,'accounting_chains_admitted':0,
        'strategy_trials_added':0,'returns_or_prices_read':False,'reports':reports})))
    print(json.dumps({'sources':len(reports),'pairs':successes,'failures':[{'year':r['fiscal_year'],'error':r.get('error')} for r in reports if 'revenue' not in r]}))

class AdapterTests(unittest.TestCase):
    def test_both_explicit_policy_spellings(self):
        for p in ['GAAP','generally accepted accounting principles in the United States (GAAP)']:
            s=BeautifulSoup('<p>Reported results were prepared in accordance with '+p+' and include all revenue and expenses recognized during the period.</p>','html.parser')
            self.assertEqual(policy(s)['start_normalized_text'],0)
    def test_policy_not_inferred(self):
        with self.assertRaises(ValueError):policy(BeautifulSoup('<p>Adjusted results are non-GAAP.</p>','html.parser'))
    def test_policy_duplicates_fail(self):
        p='Reported results were prepared in accordance with GAAP and include all revenue and expenses recognized during the period.'
        with self.assertRaises(ValueError):policy(BeautifulSoup(p+p,'html.parser'))
    def test_event_exact_binding(self):
        meta={'event_id':'ABBV_2025-01-31','fiscal_year':2024,'source_url':'https://news.abbvie.com/fabricated','publication_date':'2025-01-31','official_actual_period_end':'2024-12-31','available_session':'2025-02-03','sha256':'fabricated','path':'fabricated'}
        event=dict(meta,symbol='ABBV',fiscal_quarter=4,status='ready',source_kind='issuer_original_release',evidence_sha256='fabricated',evidence_path='fabricated',precision='date_only',publication_at=None,known_by_is_actual_publication=False,publication_timezone='America/Chicago',publication_timezone_basis='fabricated')
        self.assertEqual(canonical_event([event],meta)[0]['release_date'],'2025-01-31')
        for key,value in [('fiscal_year',2023),('evidence_sha256','different'),('publication_at','2025-01-31T09:00:00Z')]:
            e=dict(event);e[key]=value
            with self.assertRaises(ValueError):canonical_event([e],meta)
        with self.assertRaises(ValueError):canonical_event([event,event],meta)

def tests():
    stream=io.StringIO();suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(base.Tests),unittest.defaultTestLoader.loadTestsFromTestCase(AdapterTests)])
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite);print(stream.getvalue())
    print(json.dumps(audit.write('FABRICATED_TESTS_V2.json',{'created_utc':base.now(),'script':audit.bind(SELF),
        'base_script':audit.bind(base.SELF),'successful':result.wasSuccessful(),'tests_run':result.testsRun,
        'failures':len(result.failures),'errors':len(result.errors),'fabricated_only':True,'output':stream.getvalue()})))
    if not result.wasSuccessful():raise SystemExit(1)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['test','freeze','extract']);a=p.parse_args()
    {'test':tests,'freeze':freeze,'extract':extract}[a.phase]()
