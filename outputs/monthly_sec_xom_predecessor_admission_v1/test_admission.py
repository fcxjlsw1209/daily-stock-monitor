from copy import deepcopy
from datetime import date
import importlib.util,json
from pathlib import Path
import unittest
P=Path(__file__).resolve().parent

def load(name,file):
    s=importlib.util.spec_from_file_location(name,P/file);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
build=load('xom_test_build','build.py');a=build.module();pf=load('xom_test_fixtures','test_parser.py')

def synthetic(dual=False,bad_second=False):
    docs={};d=dict(policy_id=a.POLICY,source_class=a.SOURCE_CLASS,evidence_class=a.EVIDENCE_CLASS,
        expected_record_count=3,expected_rows=2,expected_accessions=2,expected_pairs=2 if dual else 1,
        expected_pair_occurrences=2 if dual else 1,resolved_indices=[0,1],no_pair_indices=[0],post_effective_excluded_indices=[2],
        facts_path='facts.json',filings_root_path='root.json',filings_archive_path='CIK0000034088-submissions-001.json',
        sessions_path='sessions.json',diagnostic_path='diagnostic.json',source_certificates=[])
    def put(path,x):docs[path]=build.output_bytes(x)
    def legacy(path,x,url):
        put(path,x);put(str(Path(path).with_suffix('.provenance.json')),dict(source=url,sha256=a.sha(docs[path]),retrieved_at='2026-10-16T12:00:00+00:00'))
    anchor_acc='0000034088-16-000065';annual='0000034088-20-000010';quarter='0000034088-20-000020'
    recent=dict(accessionNumber=[anchor_acc,annual,quarter],filingDate=['2016-02-24','2020-01-31','2020-05-01'],
        reportDate=['2015-12-31','2019-12-31','2020-03-31'],form=['10-K','10-K','10-Q'],primaryDocument=['anchor.htm','annual.htm','quarter.htm'])
    archive=dict(accessionNumber=['0000034088-10-000001'],filingDate=['2010-01-01'],reportDate=['2009-12-31'],form=['10-K'],primaryDocument=['unused.htm'])
    root=dict(cik=34088,filings=dict(recent=recent,files=[dict(name=d['filings_archive_path'],filingCount=1,filingFrom='2010-01-01',filingTo='2010-01-02')]))
    facts=dict(cik=34088,facts={'us-gaap':{}})
    for tag,vals in [('Revenues',(105000000,95000000))]+([('RevenueFromContractWithCustomerExcludingAssessedTax',(101000000 if bad_second else 100000000,90000000))]if dual else[]):
        facts['facts']['us-gaap'][tag]=dict(units={'USD':[dict(start=f'{y}-01-01',end=f'{y}-03-31',val=v,accn=quarter,form='10-Q',filed='2020-05-01',fy=2020,fp='Q1')for y,v in zip((2020,2019),vals)]})
    legacy(d['facts_path'],facts,'https://data.sec.gov/api/xbrl/companyfacts/CIK0000034088.json')
    legacy(d['filings_root_path'],root,'https://data.sec.gov/submissions/CIK0000034088.json')
    legacy(d['filings_archive_path'],archive,'https://data.sec.gov/submissions/CIK0000034088-submissions-001.json')
    sessions=['2015-01-02','2016-02-25','2020-01-31','2020-02-03','2020-03-02','2020-05-01','2020-05-04','2020-05-05','2026-07-31'];put('sessions.json',sessions)
    records=[dict(symbol='XOM',signal_date=x,candidate_cik=a.OLD_CIK,catalog=dict(status='coverage_unknown',selected=None),
                  pairing=dict(pairs=[],factor_ready=False,identity_verified=False,semantic_verified=False),factor_ready=False,identity_verified=False,semantic_verified=False)
             for x in ('2020-03-02','2020-05-05','2026-07-31')]
    put('records.json',records);d['source_records']=dict(path='records.json',sha256=a.sha(docs['records.json']))
    catalog=a.filings.build_catalog(root,{d['filings_archive_path']:archive},'2026-10-16');diag=[]
    for i,r in enumerate(records):
        cat=a.filings.asof_report(catalog,r['signal_date'],sessions)if i<2 else None
        pairing=a.revenue.pair_latest_report(facts,cat['selected'],r['signal_date'])if cat else None
        diag.append(dict(record_index_0based=i,signal_date=r['signal_date'],original_record_sha256=a.digest(r),original_cik=a.OLD_CIK,catalog=cat,pairing=pairing))
    put('diagnostic.json',dict(rows=diag))
    def source(acc,filing,report,form,name,raw,indices):
        directory='source';path=f'{directory}/raw/{name}.decoded';docs[path]=raw;docs[f'{directory}/raw/{name}.response']=raw
        url=f'https://investor.exxonmobil.com/sec-filings/all-sec-filings/content/{acc}/{name}.htm'
        rp=f'{directory}/raw/{name}.meta.json'
        put(rp,dict(status=200,request_count=1,public_unauthenticated=True,url=url,final_url=url,
            response_path=f'raw/{name}.response',decoded_path=f'raw/{name}.decoded',response_sha256=a.sha(raw),decoded_sha256=a.sha(raw),
            response_bytes=len(raw),decoded_bytes=len(raw),content_encoding=None,decoder='identity',accept_encoding='identity',retrieved_at='2026-10-16T12:00:00+00:00'))
        out=dict(accession=acc,filing_date=filing,report_date=report,form=form,primary_document=name+'.htm',url=url,
            document_path=path,response_path=f'{directory}/raw/{name}.response',receipt_path=rp,record_indices=indices,
            record_sha256s={str(i):a.digest(records[i])for i in indices},evidence_pair_hashes={},semantic_map={})
        for i in indices:
            for pair in diag[i]['pairing']['pairs']:
                out['evidence_pair_hashes'][a.digest(dict(accession=acc,pair=pair))]=a.digest(pair)
                out['semantic_map'][pair['tag']]=dict(semantic_concept='total_revenues_and_other_income'if pair['tag']=='Revenues'else'sales_and_other_operating_revenue',basis='registered_accession_specific_printed_consolidated_row_reconstruction')
        return out
    anchor_raw=pf.fixture(year=2015,months=12).replace(b'</body>',b'<p>The principal market where ExxonMobil common stock (XOM) is traded is the New York Stock Exchange.</p></body>')
    d['identity_anchor']=source(anchor_acc,'2016-02-24','2015-12-31','10-K','anchor',anchor_raw,[])
    d['scope']=[source(annual,'2020-01-31','2019-12-31','10-K','annual',pf.fixture(year=2019,months=12),[0]),
                source(quarter,'2020-05-01','2020-03-31','10-Q','quarter',pf.fixture(),[1])]
    d['source_hashes']={k:a.sha(v)for k,v in docs.items()};return d,docs

class AdmissionTests(unittest.TestCase):
    def test_full_sidecar_retains_original(self):
        d,docs=synthetic();out={k:a.strict_json(v)for k,v in build.produce(d,docs).items()};s=out['ADMISSIONS.json']['records'][0]
        self.assertEqual(s['candidate_cik'],a.OLD_CIK);self.assertEqual(s['resolved_candidate_cik'],a.CIK)
        self.assertEqual(s['original_record'],a.strict_json(docs['records.json'])[1]);self.assertFalse(s['factor_ready'])
        self.assertEqual(out['COVERAGE.json']['no_pair_count'],1);self.assertEqual(out['COVERAGE.json']['post_effective_excluded_indices'],[2])
    def test_ordered_dual_pair(self):
        d,docs=synthetic(dual=True);s=a.strict_json(build.produce(d,docs)['ADMISSIONS.json'])['records'][0]
        self.assertEqual([e['tag']for e in s['approval']],['Revenues','RevenueFromContractWithCustomerExcludingAssessedTax'])
        self.assertNotEqual(s['approval'][0]['semantic_concept'],s['approval'][1]['semantic_concept'])
    def test_second_failure_blocks_whole_record(self):
        d,docs=synthetic(dual=True,bad_second=True);out=build.produce(d,docs)
        self.assertEqual(a.strict_json(out['ADMISSIONS.json'])['records'],[])
        self.assertEqual(a.strict_json(out['COVERAGE.json'])['unverified_count'],1)
    def test_changed_source_fails(self):
        d,docs=synthetic();docs['facts.json']+=b' '
        with self.assertRaisesRegex(ValueError,'hash'):build.produce(d,docs)
    def test_legacy_http_cannot_be_invented(self):
        d,docs=synthetic();p='facts.provenance.json';m=a.strict_json(docs[p]);m['status']=200;docs[p]=build.output_bytes(m);d['source_hashes'][p]=a.sha(docs[p])
        with self.assertRaisesRegex(ValueError,'legacy_http'):build.produce(d,docs)
    def test_original_flag_mutation_fails(self):
        d,docs=synthetic();r=a.strict_json(docs['records.json']);r[0]['factor_ready']=True;docs['records.json']=build.output_bytes(r);d['source_hashes']['records.json']=a.sha(docs['records.json']);d['source_records']['sha256']=d['source_hashes']['records.json']
        with self.assertRaisesRegex(ValueError,'flags'):build.produce(d,docs)
    def test_corrupt_diagnostic_not_used_as_oracle(self):
        d,docs=synthetic();x=a.strict_json(docs['diagnostic.json']);x['rows'][1]['pairing']['pairs'][0]['current']['val']=1;docs['diagnostic.json']=build.output_bytes(x);d['source_hashes']['diagnostic.json']=a.sha(docs['diagnostic.json'])
        with self.assertRaisesRegex(ValueError,'reconstruction'):build.produce(d,docs)
    def test_wrong_url_rejected(self):
        d,docs=synthetic();p=d['scope'][1]['receipt_path'];x=a.strict_json(docs[p]);x['final_url']='https://example.com/';docs[p]=build.output_bytes(x);d['source_hashes'][p]=a.sha(docs[p])
        with self.assertRaisesRegex(ValueError,'url'):build.produce(d,docs)
    def test_missing_ordered_pair_scope_fails(self):
        d,docs=synthetic(dual=True);d['scope'][1]['evidence_pair_hashes'].pop(next(iter(d['scope'][1]['evidence_pair_hashes'])))
        with self.assertRaisesRegex(ValueError,'unregistered_pair'):build.produce(d,docs)
    def test_wrong_semantic_does_not_follow_numeric_match(self):
        d,docs=synthetic();d['scope'][1]['semantic_map']['Revenues']['semantic_concept']='sales_and_other_operating_revenue'
        self.assertEqual(a.strict_json(build.produce(d,docs)['ADMISSIONS.json'])['records'],[])
    def test_archive_coverage_end_not_forced_actual(self):
        d,docs=synthetic();build.produce(d,docs)
    def test_archive_outside_end_rejected(self):
        d,docs=synthetic();p='root.json';x=a.strict_json(docs[p]);x['filings']['files'][0]['filingTo']='2009-12-31';docs[p]=build.output_bytes(x);d['source_hashes'][p]=a.sha(docs[p]);m=a.strict_json(docs['root.provenance.json']);m['sha256']=a.sha(docs[p]);docs['root.provenance.json']=build.output_bytes(m);d['source_hashes']['root.provenance.json']=a.sha(docs['root.provenance.json'])
        with self.assertRaisesRegex(ValueError,'archive'):build.produce(d,docs)
    def test_raw_duplicate_scope_is_complete(self):
        d,docs=synthetic();facts=a.strict_json(docs['facts.json']);pair=a.strict_json(docs['diagnostic.json'])['rows'][1]['pairing']['pairs'][0];facts['facts']['us-gaap']['Revenues']['units']['USD'].append(deepcopy(facts['facts']['us-gaap']['Revenues']['units']['USD'][0]))
        with self.assertRaisesRegex(ValueError,'duplicate'):a.validate_pair(pair,dict(accession=pair['current']['accn'],filingDate='2020-05-01',form='10-Q'),facts)
    def test_no_calendar_or_catalog_fallback(self):
        d,docs=synthetic();sessions=a.strict_json(docs['sessions.json']);sessions.remove('2020-05-04');docs['sessions.json']=build.output_bytes(sessions);d['source_hashes']['sessions.json']=a.sha(docs['sessions.json'])
        with self.assertRaisesRegex(ValueError,'reconstruction'):build.produce(d,docs)
    def test_admission_does_not_upgrade_legacy_grade(self):
        d,docs=synthetic();ev=a.strict_json(build.produce(d,docs)['EVIDENCE.json'])
        status=deepcopy(ev['source_status']);status['raw_facts_provenance_verified']=True
        with self.assertRaisesRegex(ValueError,'source_grade'):
            a.admit(ev['resolved_records'][1],list(ev['evidence'].values()),status,ev['identity_anchor'])
if __name__=='__main__':unittest.main()
