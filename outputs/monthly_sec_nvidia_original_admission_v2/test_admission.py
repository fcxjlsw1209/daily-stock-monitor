"""Synthetic complete pipeline and source/clock/all-pair rejection boundaries."""
from copy import deepcopy
from datetime import date
import gzip
import importlib.util
from pathlib import Path
import tempfile
import unittest
from test_parser import fixture

def load(name):
    s=importlib.util.spec_from_file_location('nvda_test_'+name,Path(__file__).with_name(name+'.py'))
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

a=load('admission');b=load('build')

def synthetic():
    docs={}
    def add(name,obj):docs[name]=b.output_bytes(obj)
    flags={k:False for k in a.FLAGS};acc='0001045810-22-000001';retrieved='2026-10-16T01:00:00+00:00'
    selected=dict(accession=acc,accessionNumber=acc,form='10-Q',filingDate='2022-08-26',filing_date='2022-08-26',reportDate='2022-07-31',report_date='2022-07-31',first_usable_date='2022-08-29',metadata_issues=[],supported_form=True,sources=[dict(table='recent',row=0)])
    pair=dict(taxonomy='us-gaap',tag='Revenues',unit='USD')
    for i,(role,start,end,value) in enumerate([('current','2022-01-31','2022-07-31',200_000_000_000),('prior','2021-02-01','2021-08-01',180_000_000_000)]):
        pair[role]=dict(start=start,end=end,val=value,accn=acc,filed='2022-08-26',form='10-Q',row_index=i,row_indices=[i],duration_days=182)
    raw=[{k:v for k,v in pair[r].items() if k not in ('row_index','row_indices','duration_days')} for r in ('current','prior')]
    add('facts',dict(cik=1045810,facts={'us-gaap':{'Revenues':dict(units={'USD':raw})}}))
    record=dict(symbol='NVDA',candidate_cik=a.CIK,signal_date='2022-08-31',**flags,catalog=dict(status='selected',coverage_complete=True,selected=selected),pairing=dict(status='paired',structural_pair_available=True,pairs=[pair],**flags))
    add('records',[record]);add('sessions',['2022-08-26','2022-08-29','2022-08-31'])
    recent={k:[selected[k]] for k in ('accessionNumber','filingDate','reportDate','form')}
    archive=dict(accessionNumber=['old'],filingDate=['2019-02-22'],reportDate=['2019-01-27'],form=['10-K'])
    advertised=dict(name='CIK0001045810-submissions-001.json',filingCount=1,filingFrom='2019-02-22',filingTo='2019-02-25')
    add('root',dict(cik=1045810,filings=dict(recent=recent,files=[advertised])));add('archive',archive)
    docs['collector']=b'synthetic collector';add('sec.plan',dict(scope='synthetic'))
    for path,name,kind in [('root','CIK0001045810.json','submissions_root'),('archive','CIK0001045810-submissions-001.json','submissions_archive')]:
        url='https://data.sec.gov/submissions/'+name
        meta=dict(source=url,http_status=200,sha256=a.sha(docs[path]),bytes=len(docs[path]),kind=kind,retrieved_at=retrieved,plan_sha256=a.sha(docs['sec.plan']),collector_sha256=a.sha(docs['collector']))
        request=dict(url=url,filename=name,kind=kind,http_status=200,sha256=meta['sha256'],bytes=meta['bytes'],retrieved_at=retrieved)
        if path=='archive':request.update(fresh_advertised=advertised,observed_archive=dict(filing_count=1,minimum_filing_date='2019-02-22',maximum_filing_date='2019-02-22'))
        add(path+'.meta',meta);add(path+'.request',request)
    collections={};scope=[];calendar=[]
    for key,items in [('main',[(2023,2,fixture())]),('boundary',[(2022,4,fixture(4,2022)),(2021,4,fixture(4,2021))])]:
        docs[key+'/client']=b'synthetic client';entries=[];targets=[]
        for i,(fy,q,html) in enumerate(items):
            path=key+f'/raw/{i}.decoded';docs[path]=html.encode();response=key+f'/raw/{i}.response';docs[response]=docs[path]
            url=f'https://nvidianews.nvidia.com/news/synthetic-{key}-{i}'
            receipt=dict(url=url,final_url=url,status=200,public_unauthenticated=True,request_count=1,content_type='text/html; charset=utf-8',retrieved_at=retrieved,
                response_path=f'raw/{i}.response',response_sha256=a.sha(docs[response]),response_bytes=len(docs[response]),decoded_path=f'raw/{i}.decoded',decoded_sha256=a.sha(docs[path]),decoded_bytes=len(docs[path]),content_encoding=None,decoder='identity')
            meta=key+f'/raw/{i}.meta.json';add(meta,receipt)
            entry=dict(path=path,receipt=receipt)
            released='2022-08-24' if key=='main' else ('2022-02-16' if fy==2022 else '2021-02-24')
            anchor=dict(collection=key,entry_index=i,release_url=url,release_path=path,response_path=response,receipt_path=meta,fiscal_year=fy,fiscal_quarter=q,release_date=released)
            if key=='main':
                target=dict(url=url,accession=acc,filing_date='2022-08-26',report_date='2022-07-31',release_date=released,record_indices=[0]);entry['target']=target;targets.append(target)
                anchor.update(**{k:target[k] for k in ('accession','filing_date','report_date','record_indices')},form='10-Q',first_usable_date='2022-08-29',record_sha256s={'0':a.digest(record)},evidence_pair_hashes={a.digest(dict(accession=acc,pair=pair)):a.digest(pair)})
                scope.append(anchor)
            else:anchor['role']='calendar_boundary_only_not_admission_scope';calendar.append(anchor)
            anchor['entry_sha256']=a.digest(entry);entries.append(entry)
        plan=dict(client_sha256=a.sha(docs[key+'/client']))
        if key=='main':plan['targets']=targets
        else:plan['urls']=[e['receipt']['url'] for e in entries]
        add(key+'/PLAN.json',plan);add(key+'/RESULT.json',dict(plan_sha256=a.sha(docs[key+'/PLAN.json']),entries=entries))
        collections[key]=dict(plan_path=key+'/PLAN.json',result_path=key+'/RESULT.json',client_path=key+'/client')
    hashes={n:a.sha(v) for n,v in docs.items()}
    d=dict(policy_id=a.POLICY,source_class=a.SOURCE_CLASS,evidence_class=a.EVIDENCE_CLASS,allowed_tag_semantics=a.SEMANTICS,expected_record_count=1,expected_accessions=1,expected_rows=1,expected_pairs=1,expected_pair_occurrences=1,expected_calendar_only_documents=2,
        source_hashes=hashes,source_records=dict(path='records',sha256=hashes['records']),facts_path='facts',sessions_path='sessions',filings_root_path='root',filings_root_receipt_path='root.meta',filings_root_request_path='root.request',filings_path='archive',filings_receipt_path='archive.meta',filings_request_path='archive.request',acquisition_plan_path='sec.plan',acquisition_collector_path='collector',collections=collections,scope=scope,calendar_only=calendar,excluded_indices=[])
    return d,docs

class AdmissionTests(unittest.TestCase):
    def setUp(self):self.d,self.docs=synthetic()
    def read(self,n):return a.strict_json(self.docs[n])
    def change(self,n,x):self.docs[n]=b.output_bytes(x);self.d['source_hashes'][n]=a.sha(self.docs[n])
    def run_fixture(self):return {k:a.strict_json(v) for k,v in b.produce(self.d,self.docs).items()}
    def entries(self):return a.source_bundle(self.d,self.docs,self.d['source_hashes'])
    def mutate_receipt(self,field,value,encoding=None):
        anchor=self.d['scope'][0];result=self.read('main/RESULT.json');entry=result['entries'][0];entry['receipt'][field]=value
        if encoding is not None:entry['receipt']['content_encoding']=encoding
        self.change(anchor['receipt_path'],entry['receipt']);self.change('main/RESULT.json',result);anchor['entry_sha256']=a.digest(entry)
    def mutate_html(self,old,new):
        anchor=self.d['scope'][0];blob=self.docs[anchor['release_path']].replace(old.encode(),new.encode())
        for n in (anchor['release_path'],anchor['response_path']):self.docs[n]=blob;self.d['source_hashes'][n]=a.sha(blob)
        for field,value in [('decoded_sha256',a.sha(blob)),('response_sha256',a.sha(blob)),('decoded_bytes',len(blob)),('response_bytes',len(blob))]:self.mutate_receipt(field,value)
    def test_complete_pipeline_truthful_and_immutable(self):
        old=deepcopy((self.d,self.docs));out=self.run_fixture();self.assertEqual((self.d,self.docs),old)
        self.assertEqual(out['COVERAGE.json']['semantic_admitted_count'],1)
        r=out['ADMISSIONS.json']['records'][0];self.assertEqual(r['evidence_class'],'primary_document_reconstruction');self.assertFalse(r['factor_ready'])
        status=r['approval'][0]['source_status'];self.assertFalse(status['raw_facts_provenance_verified']);self.assertFalse(status['facts_http_status_recorded'])
        period=r['approval'][0]['selected_period'];self.assertEqual(period['current']['start'],'2022-01-31');self.assertEqual(period['prior']['duration_weeks'],26)
        self.assertFalse(period['current']['period_start_directly_printed'])
    def test_missing_identity_is_unverified_not_admitted(self):
        self.mutate_html('NASDAQ: NVDA','NASDAQ: OTHER');out=self.run_fixture();self.assertEqual(out['ADMISSIONS.json']['records'],[]);self.assertEqual(out['COVERAGE.json']['unverified_count'],1)
    def test_numeric_mismatch_is_unverified(self):self.mutate_html('200,000','200,001');self.assertEqual(self.run_fixture()['COVERAGE.json']['unverified_count'],1)
    def add_pair(self,wrong=False):
        records=self.read('records');pair=deepcopy(records[0]['pairing']['pairs'][0]);pair['tag']='RevenueFromContractWithCustomerExcludingAssessedTax'
        if wrong:pair['prior']['val']+=1
        records[0]['pairing']['pairs'].append(pair);self.change('records',records);self.d['source_records']['sha256']=a.sha(self.docs['records'])
        facts=self.read('facts');facts['facts']['us-gaap'][pair['tag']]=dict(units={'USD':[{k:v for k,v in pair[r].items() if k not in ('row_index','row_indices','duration_days')} for r in ('current','prior')]});self.change('facts',facts)
        anchor=self.d['scope'][0];anchor['record_sha256s']['0']=a.digest(records[0]);anchor['evidence_pair_hashes'][a.digest(dict(accession=anchor['accession'],pair=pair))]=a.digest(pair);self.d['expected_pairs']=self.d['expected_pair_occurrences']=2
    def test_original_dual_tags_preserved_in_order(self):
        self.add_pair();r=self.run_fixture()['ADMISSIONS.json']['records'][0];self.assertEqual([x['tag'] for x in r['approval']],list(a.SEMANTICS));self.assertEqual(len(set(r['evidence_ids'])),2)
    def test_second_pair_failure_blocks_whole_row(self):self.add_pair(True);self.assertEqual(self.run_fixture()['ADMISSIONS.json']['records'],[])
    def test_future_boundary_not_backfilled(self):
        out=self.run_fixture();parsed=out['EVIDENCE.json']['releases'][self.d['scope'][0]['accession']]['parsed'];cal=out['EVIDENCE.json']['fiscal_calendar_evidence']
        for vals in cal.values():
            for v in vals:v['release_date']='2023-01-01'
        r=a.compare_pair(self.read('records')[0]['pairing']['pairs'][0],6,parsed,cal);self.assertEqual(r['reason'],'missing_prior_fiscal_year_end')
    def test_missing_comparative_boundary_not_approximated(self):
        out=self.run_fixture();ev=out['EVIDENCE.json'];cal=ev['fiscal_calendar_evidence'];del cal['2021']
        self.assertEqual(a.compare_pair(self.read('records')[0]['pairing']['pairs'][0],6,next(iter(ev['releases'].values()))['parsed'],cal)['reason'],'missing_prior_fiscal_year_end')
    def test_wrong_start_or_nonweek_duration_rejected(self):
        ev=self.run_fixture()['EVIDENCE.json'];pair=self.read('records')[0]['pairing']['pairs'][0];pair['current']['start']='2022-02-01';pair['current']['duration_days']=181
        self.assertEqual(a.compare_pair(pair,6,next(iter(ev['releases'].values()))['parsed'],ev['fiscal_calendar_evidence'])['reason'],'fiscal_start_or_duration_mismatch')
    def test_53_week_year_requires_exact_derived_days(self):
        parsed=a.parser.parse_release(fixture(4,2021));pair=dict(current=dict(start='2020-01-27',end='2021-01-31',duration_days=371,val=200_000_000_000),prior=dict(start='2019-01-28',end='2020-01-26',duration_days=364,val=180_000_000_000))
        cal={str(y):[dict(end=end,release_date='2020-02-01',source_path='prior',basis='printed',excerpt='exact')] for y,end in [(2020,'2020-01-26'),(2019,'2019-01-27')]}
        r=a.compare_pair(pair,12,parsed,cal);self.assertEqual(r['status'],'exact_match');self.assertEqual(r['selected_period']['current']['duration_weeks'],53)
    def test_conflicting_fiscal_end_hard_fails(self):
        ev=self.run_fixture()['EVIDENCE.json'];r=deepcopy(next(iter(ev['calendar_only_releases'].values())));r['parsed']['fiscal_evidence']['year_ends'][0]['end']='2022-01-23'
        with self.assertRaisesRegex(ValueError,'conflicting'):a.calendar_index({**ev['calendar_only_releases'],'bad':r})
    def test_http403_never_admitted(self):
        self.mutate_receipt('status',403)
        with self.assertRaisesRegex(ValueError,'failed_source'):self.run_fixture()
    def test_redirected_url_not_original_evidence(self):
        self.mutate_receipt('final_url','https://other.example/x')
        with self.assertRaisesRegex(ValueError,'receipt'):self.run_fixture()
    def test_receipt_path_alias_rejected(self):
        self.mutate_receipt('decoded_path','raw/other.decoded')
        with self.assertRaisesRegex(ValueError,'artifact_path'):self.run_fixture()
    def test_naive_retrieval_rejected(self):
        self.mutate_receipt('retrieved_at','2026-10-16')
        with self.assertRaisesRegex(ValueError,'naive'):self.run_fixture()
    def test_transfer_bytes_bound(self):
        anchor=self.d['scope'][0];self.docs[anchor['response_path']]+=b' ';self.d['source_hashes'][anchor['response_path']]=a.sha(self.docs[anchor['response_path']])
        with self.assertRaisesRegex(ValueError,'transfer_hash'):self.run_fixture()
    def test_gzip_redecoded_with_recorded_decoder(self):
        anchor=self.d['scope'][0];raw=gzip.compress(self.docs[anchor['release_path']]);self.docs[anchor['response_path']]=raw;self.d['source_hashes'][anchor['response_path']]=a.sha(raw)
        for k,v in [('content_encoding','gzip'),('decoder','python:gzip.decompress'),('response_sha256',a.sha(raw)),('response_bytes',len(raw))]:self.mutate_receipt(k,v)
        self.assertEqual(self.run_fixture()['COVERAGE.json']['semantic_admitted_count'],1)
        self.mutate_receipt('decoder','gzip')
        with self.assertRaisesRegex(ValueError,'gzip_transfer'):self.run_fixture()
    def test_collection_plan_targets_exact(self):
        pl=self.read('main/PLAN.json');pl['targets'][0]['report_date']='2022-06-30';self.change('main/PLAN.json',pl)
        res=self.read('main/RESULT.json');res['plan_sha256']=a.sha(self.docs['main/PLAN.json']);self.change('main/RESULT.json',res)
        with self.assertRaisesRegex(ValueError,'targets'):self.run_fixture()
    def test_advertised_end_may_have_no_filing(self):self.assertEqual(self.run_fixture()['COVERAGE.json']['semantic_admitted_count'],1)
    def test_actual_archive_filing_after_coverage_fails(self):
        root=self.read('root');root['filings']['files'][0]['filingTo']='2019-02-21';self.change('root',root)
        for n in ('root.meta','root.request'):
            v=self.read(n);v.update(sha256=a.sha(self.docs['root']),bytes=len(self.docs['root']));self.change(n,v)
        with self.assertRaisesRegex(ValueError,'coverage'):self.run_fixture()
    def test_catalog_collector_hash_required(self):
        meta=self.read('root.meta');meta['collector_sha256']='bad';self.change('root.meta',meta)
        with self.assertRaisesRegex(ValueError,'acquisition'):self.run_fixture()
    def test_full_raw_context_and_duplicate_indices_required(self):
        r=self.read('records')[0];facts=self.read('facts');rows=facts['facts']['us-gaap']['Revenues']['units']['USD'];rows.append(deepcopy(rows[0]))
        with self.assertRaisesRegex(ValueError,'context'):a.validate_pair(r,r['pairing']['pairs'][0],facts,self.read('sessions'),self.d['scope'][0],a.filing_source(self.d,self.docs,self.d['source_hashes']))
    def test_first_usable_strictly_after_filing(self):
        r=self.read('records')[0];r['catalog']['selected']['first_usable_date']='2022-08-26'
        with self.assertRaisesRegex(ValueError,'first_usable'):a.validate_pair(r,r['pairing']['pairs'][0],self.read('facts'),self.read('sessions'),self.d['scope'][0],a.filing_source(self.d,self.docs,self.d['source_hashes']))
    def test_no_original_flag_upgrade(self):
        rows=self.read('records');rows[0]['semantic_verified']=True;self.change('records',rows);self.d['source_records']['sha256']=a.sha(self.docs['records'])
        with self.assertRaisesRegex(ValueError,'flags'):self.run_fixture()
    def test_missing_or_reordered_pair_evidence_hard_fails(self):
        self.add_pair();out=self.run_fixture();r=self.read('records')[0];entries=list(out['EVIDENCE.json']['evidence'].values());ids=out['ADMISSIONS.json']['records'][0]['evidence_ids'];entries=sorted(entries,key=lambda e:ids.index(e['evidence_id']))
        for bad in (entries[:1],entries[::-1]):
            with self.assertRaisesRegex(ValueError,'order_or_count'):a.admit_record(0,r,bad,{'semantic_source_sufficient':True})
    def test_create_once_and_byte_exact_verify(self):
        with tempfile.TemporaryDirectory() as tmp:
            out={n:b'{}\n' for n in b.OUTPUTS};self.assertEqual(b.persist_once(Path(tmp),out),'created_once');self.assertEqual(b.persist_once(Path(tmp),out,True),'verified_existing')
            with self.assertRaisesRegex(ValueError,'mismatch'):b.persist_once(Path(tmp),{**out,'PROOF.json':b'changed'})

class PriorEvidencePreservationTests(unittest.TestCase):
    def setUp(self):
        self.d,self.docs=synthetic();self.old={k:a.strict_json(v) for k,v in b.produce(self.d,self.docs).items()}
    def test_retained_calendar_recomputed_not_replaced_by_new_earlier_source(self):
        prior=deepcopy(self.old['EVIDENCE.json'])
        prior['fiscal_calendar_evidence']['2021']=[v for v in prior['fiscal_calendar_evidence']['2021'] if v['source_path']=='boundary/raw/0.decoded']
        eid=next(iter(prior['evidence']))
        prior['evidence'][eid]['selected_period']['prior']['revenue_usd']=1  # Must not be used as an oracle.
        out={k:a.strict_json(v) for k,v in b.produce(self.d,self.docs,prior).items()}
        result=out['EVIDENCE.json']['evidence'][eid]['selected_period']['prior']
        self.assertEqual(result['fiscal_boundary_evidence']['source_path'],'boundary/raw/0.decoded')
        self.assertEqual(result['revenue_usd'],180_000_000_000)
        self.assertEqual(self.old['EVIDENCE.json']['evidence'][eid]['selected_period']['prior']['fiscal_boundary_evidence']['source_path'],'boundary/raw/1.decoded')
    def test_prior_unverified_pair_gets_new_full_calendar(self):
        prior=deepcopy(self.old['EVIDENCE.json']);prior['fiscal_calendar_evidence']['2021']=[v for v in prior['fiscal_calendar_evidence']['2021'] if v['source_path']=='boundary/raw/0.decoded']
        eid=next(iter(prior['evidence']));prior['evidence'][eid]['status']='unverified'
        out={k:a.strict_json(v) for k,v in b.produce(self.d,self.docs,prior).items()}
        self.assertEqual(out['EVIDENCE.json']['evidence'][eid]['selected_period']['prior']['fiscal_boundary_evidence']['source_path'],'boundary/raw/1.decoded')
    def test_prior_boundary_must_be_refound_exact(self):
        prior=deepcopy(self.old['EVIDENCE.json']);prior['fiscal_calendar_evidence']['2021'][0]['source_sha256']='wrong'
        with self.assertRaisesRegex(ValueError,'not_reconstructed'):b.produce(self.d,self.docs,prior)
    def test_prior_sidecars_exact_and_additions_bounded(self):
        raw={k:b.output_bytes(v) for k,v in self.old.items()}
        self.assertEqual(b.preserve_prior(raw,self.old['ADMISSIONS.json'],self.old['COVERAGE.json']),1)
        changed=deepcopy(self.old['ADMISSIONS.json']);changed['records'][0]['approval'][0]['selected_period']['current']['revenue_usd']+=1
        with self.assertRaisesRegex(ValueError,'sidecar_changed'):b.preserve_prior({**raw,'ADMISSIONS.json':b.output_bytes(changed)},self.old['ADMISSIONS.json'],self.old['COVERAGE.json'])
        changed=deepcopy(self.old['ADMISSIONS.json']);new=deepcopy(changed['records'][0]);new['record_index_0based']=99;changed['records'].append(new)
        with self.assertRaisesRegex(ValueError,'not_prior_unverified'):b.preserve_prior({**raw,'ADMISSIONS.json':b.output_bytes(changed)},self.old['ADMISSIONS.json'],self.old['COVERAGE.json'])

if __name__=='__main__':unittest.main()
