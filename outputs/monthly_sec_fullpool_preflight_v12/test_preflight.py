"""Synthetic count-only boundaries; no real source data or strategy execution."""
from copy import deepcopy
from datetime import date, timedelta
import importlib.util
from pathlib import Path
import unittest

P=Path(__file__).resolve().parent
ROOT=P.parent.parent

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

m=load(P/'preflight.py','v12_merge_test')
v=load(ROOT/'outputs/monthly_sec_fullpool_preflight_v1/preflight.py','v12_axis_test')

def fixture():
    flags={k:False for k in m.FLAGS}
    days=[(date(2020,1,2)+timedelta(days=i)).isoformat() for i in range(122)]
    etfs=['ETF'+str(i) for i in range(17)]
    stocks=['S'+str(i) for i in range(31)]+['INTC']
    meta=dict(train_months=88,early_months=53,dates=days,etfs=etfs,symbols=[stocks+etfs for _ in days])
    sessions=['2020-01-01']+days
    source=[];rows=[];added=[]
    indices=[120*32+31,121*32+31]
    old=set(i for i in range(3904) if i%32!=31)
    old=set(sorted(old)[:834])
    seen=0
    for i in range(3904):
        f=dict(accession='a'+str(i),accessionNumber='a'+str(i),filingDate='2020-01-01',first_usable_date='2020-01-02')
        tags=['SalesRevenueNet']+(['RevenueFromContractWithCustomerExcludingAssessedTax'] if i==indices[1] else [])
        pairs=[dict(taxonomy='us-gaap',tag=t,unit='USD',current={'val':100+j},prior={'val':90+j}) for j,t in enumerate(tags)]
        s=dict(symbol=stocks[i%32],candidate_cik='0000050863' if i%32==31 else 'synthetic',signal_date=days[i//32],catalog=dict(selected=f,status='selected',coverage_complete=True),pairing=dict(pairs=pairs,structural_pair_available=True,**flags),**flags)
        source.append(s)
        ids=['prior'+str(seen%379)] if i in old else []
        if i in old:seen+=1
        row=dict(record_index_0based=i,source_record_sha256=m.digest(s),symbol=s['symbol'],candidate_cik=s['candidate_cik'],signal_date=s['signal_date'],source_flags=deepcopy(flags),factor_ready=False,semantic_admitted=i in old,coverage_counted=i in old,reason='synthetic',evidence_ids=ids)
        rows.append(row)
        if i in indices:
            eids=[m.digest(dict(accession=f['accession'],pair=p)) for p in pairs]
            approval=[dict(evidence_id=e,pair_sha256=m.digest(p),semantic_concept=m.SEMANTICS[p['tag']],selected_filing=deepcopy(f),source_class=m.SOURCE_CLASS,evidence_class=m.EVIDENCE_CLASS,source_status=dict(raw_facts_provenance_verified=False),factor_ready=False,**deepcopy(p)) for e,p in zip(eids,pairs)]
            added.append(dict(record_index_0based=i,source_record_sha256=m.digest(s),symbol=s['symbol'],candidate_cik=s['candidate_cik'],signal_date=s['signal_date'],source_flags=deepcopy(flags),factor_ready=False,semantic_admitted=True,selected_accession=f['accession'],evidence_ids=eids,pair_sha256s=[m.digest(p) for p in pairs],approval=approval,source_class=m.SOURCE_CLASS,evidence_class=m.EVIDENCE_CLASS))
    common=dict(source_records={'path':'synthetic','sha256':'synthetic'},factor_ready_count=0)
    base=dict(policy_id='monthly_sec_fullpool_semantic_coverage_v11',thresholds=m.THRESHOLDS,semantic_admitted_count=834,unique_admitted_pairs=379,factor_ready=False,rows=rows,months=[dict(signal_index_0based=i,signal_date=d,stock_count=32,etfs_not_applicable=etfs) for i,d in enumerate(days)],source_policy_counts={'synthetic_prior':834},**common)
    add=dict(policy_id='monthly_sec_intel_original_admission_v2',semantic_admitted_count=2,records=added,source_class=m.SOURCE_CLASS,evidence_class=m.EVIDENCE_CLASS,raw_facts_provenance_verified=False,**common)
    expected=dict(baseline_rows=834,baseline_unique_pairs=379,additional_rows=2,additional_unique_pairs=3,additional_pair_occurrences=3,combined_rows=836,combined_unique_pairs=382,source_records=3904,signal_dates=122)
    policy=dict(policy_id=m.POLICY,thresholds=deepcopy(m.THRESHOLDS),expected=expected,source_hashes={'outputs/monthly_sec_fullpool_preflight_v11/RESULT.json':'base','outputs/monthly_sec_intel_original_admission_v2/ADMISSIONS.json':'addition'})
    return source,base,add,sessions,policy,meta

class ContractTests(unittest.TestCase):
    def setUp(self):self.records,self.base,self.add,self.sessions,self.policy,self.meta=fixture()
    def run_merge(self):return m.merge(self.records,self.base,self.add,self.sessions,self.policy,v.validate_universe,v.breadth,self.meta)
    def fails(self,text):
        with self.assertRaisesRegex(ValueError,text):self.run_merge()
    def refresh_source(self):
        item=self.add['records'][0];i=item['record_index_0based'];h=m.digest(self.records[i])
        item['source_record_sha256']=self.base['rows'][i]['source_record_sha256']=h
    def test_partial_batch_exact_union_and_inputs_unchanged(self):
        original=deepcopy((self.records,self.base,self.add,self.sessions,self.policy,self.meta));out=self.run_merge()
        self.assertEqual((out['semantic_admitted_count'],out['unique_admitted_pairs']),(836,382))
        self.assertEqual((out['legacy_intel_rows'],out['additional_pair_occurrences']),(2,3))
        self.assertEqual((self.records,self.base,self.add,self.sessions,self.policy,self.meta),original)
        new=set(out['additional_certified_indices'])
        for i,row in enumerate(out['rows']):
            if i not in new:self.assertEqual(row,self.base['rows'][i])
        self.assertFalse(out['factor_ready']);self.assertEqual(out['strategy_trials'],0)
    def test_changed_threshold(self):self.policy['thresholds']['admitted_stocks_per_month_minimum']=2;self.fails('policy')
    def test_wrong_expected_addition(self):self.policy['expected']['additional_rows']=57;self.fails('registered addition')
    def test_missing_second_pair(self):self.add['records'][1]['approval'].pop();self.fails('full pair approval')
    def test_reordered_pairs(self):self.add['records'][1]['evidence_ids'].reverse();self.fails('pair identity/order')
    def test_reordered_pair_hashes(self):self.add['records'][1]['pair_sha256s'].reverse();self.fails('ordered pair hashes')
    def test_concepts_not_merged(self):self.add['records'][1]['approval'][1]['tag']='SalesRevenueNet';self.fails('approved concept/context')
    def test_amount_changed(self):self.add['records'][0]['approval'][0]['current']['val']=999;self.fails('approved concept/context')
    def test_selected_filing_changed(self):self.add['records'][0]['approval'][0]['selected_filing']['filingDate']='2020-01-02';self.fails('approved concept/context')
    def test_facts_not_authenticated(self):self.add['records'][0]['approval'][0]['source_status']['raw_facts_provenance_verified']=True;self.fails('pair source class/provenance')
    def test_row_source_class(self):self.add['records'][0]['source_class']='issuer_original_ir';self.fails('row source class')
    def test_batch_source_class(self):self.add['raw_facts_provenance_verified']=True;self.fails('source class/provenance')
    def test_original_flags(self):self.records[0]['factor_ready']=True;self.base['rows'][0]['source_record_sha256']=m.digest(self.records[0]);self.fails('original flags')
    def test_source_hash(self):self.add['records'][0]['source_record_sha256']='bad';self.fails('addition identity')
    def test_duplicate(self):self.add['records'][1]=deepcopy(self.add['records'][0]);self.fails('duplicate/overlap')
    def test_overlap(self):self.add['records'][0]['record_index_0based']=0;self.fails('duplicate/overlap')
    def test_boolean_index(self):self.add['records'][0]['record_index_0based']=True;self.fails('index')
    def test_nonlatest_accession(self):self.add['records'][0]['selected_accession']='older';self.fails('selected accession')
    def test_same_day_not_usable(self):
        i=self.add['records'][0]['record_index_0based'];self.records[i]['catalog']['selected']['first_usable_date']='2020-01-01';self.refresh_source();self.fails('filing clock')
    def test_source_order(self):self.records[0],self.records[1]=self.records[1],self.records[0];self.fails('record order')
    def test_breadth_boundary(self):
        self.assertFalse(v.breadth([16]*43+[0]*79)['breadth_passed'])
        self.assertTrue(v.breadth([16]*44+[0]*78)['breadth_passed'])
        self.assertFalse(v.breadth([0]*88+[32]*34)['breadth_passed'])

if __name__=='__main__':unittest.main(verbosity=2)
