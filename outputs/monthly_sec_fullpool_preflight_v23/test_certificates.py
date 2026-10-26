"""Completed certificate closure checks, without prior implementation imports."""
import json,hashlib,importlib.util
from pathlib import Path
import unittest
p=Path(__file__).resolve().parent;s=importlib.util.spec_from_file_location('v23_cert_tests',p/'certificates.py');c=importlib.util.module_from_spec(s);s.loader.exec_module(c)
def fixture():
 d='outputs/synthetic/';docs={};hashes={}
 def put(n,x):
  b=(json.dumps(x,sort_keys=True)+'\n').encode();docs[n]=b;hashes[n]=c.sha(b)
 put('work/source',{'source':'fixed'});put('work/producer.py','source');put('work/auditor.py','source')
 source={'work/source':hashes['work/source']};impl={'work/producer.py':hashes['work/producer.py']};policy='synthetic'
 put(d+'DESIGN.json',dict(policy_id=policy,source_hashes=source));put(d+'PLAN.json',dict(policy_id=policy,design_sha256=hashes[d+'DESIGN.json'],actual_executed=False,source_hashes=source,implementation_hashes=impl))
 put(d+'ADMISSIONS.json',dict(records=[{}],semantic_admitted_count=1,factor_ready_count=0));put(d+'EVIDENCE.json',{});put(d+'COVERAGE.json',{})
 common=dict(policy_id=policy,design_sha256=hashes[d+'DESIGN.json'],plan_sha256=hashes[d+'PLAN.json'],source_hashes=source,implementation_hashes=impl)
 outs=['ADMISSIONS.json','EVIDENCE.json','COVERAGE.json'];put(d+'PROOF.json',dict(common,output_hashes={n:hashes[d+n]for n in outs}))
 put('work/review.json',dict(common,passed=True,errors=[],strategy_trials=0,network_requests=0,raw_facts_provenance_verified=False,filing_catalog_http_status_verified=True,auditor_source_hashes={'work/auditor.py':hashes['work/auditor.py']},artifact_hashes={d+n:hashes[d+n]for n in outs+['PROOF.json']}))
 put(d+'COMPLETION.json',dict(policy_id=policy,semantic_admitted_count=1,unique_admitted_pairs=1,repeat_run_verified_existing=True,factor_ready_count=0,strategy_trials=0,status='SEMANTIC_ADMISSION_COMPLETE',independent_review=dict(passed=True,path='work/review.json',sha256=hashes['work/review.json']),raw_facts_provenance_verified=False,filing_catalog_http_status_verified=True,artifact_hashes={n:hashes[d+n]for n in outs+['PROOF.json','DESIGN.json','PLAN.json']}))
 spec=dict(directory=d,policy_id=policy,rows=1,pairs=1,kind='admission',completion_sha256=hashes[d+'COMPLETION.json'],raw_facts=False,catalog_http=True);return spec,docs,hashes
class CertificateTests(unittest.TestCase):
 def setUp(self):self.spec,self.docs,self.hashes=fixture()
 def edit(self,path,key,value):o=json.loads(self.docs[path]);o[key]=value;self.docs[path]=json.dumps(o).encode()
 def check(self):return c.validate(self.spec,self.docs,self.hashes)
 def fails(self,s):
  with self.assertRaisesRegex(ValueError,s):self.check()
 def test_completed_certificate(self):self.assertFalse(self.check()['prior_financial_parsers_reexecuted'])
 def test_completion_hash(self):self.spec['completion_sha256']='bad';self.fails('completion pin')
 def test_failed_review(self):self.edit('work/review.json','passed',False);self.fails('review failed')
 def test_errors_review(self):self.edit('work/review.json','errors',['bad']);self.fails('review failed')
 def test_stale_registration(self):self.edit('work/review.json','plan_sha256','bad');self.fails('registration hash')
 def test_source_dependency_changed(self):self.hashes['work/source']='bad';self.fails('unbound dependency')
 def test_unbound_reviewer(self):del self.hashes['work/auditor.py'];self.fails('unbound dependency')
 def test_output_changed(self):self.hashes[self.spec['directory']+'EVIDENCE.json']='bad';self.fails('artifact pin')
 def test_missing_completion_output(self):
  path=self.spec['directory']+'COMPLETION.json';o=json.loads(self.docs[path]);del o['artifact_hashes']['EVIDENCE.json'];self.docs[path]=json.dumps(o).encode();self.fails('incomplete completed')
 def test_true_catalog_not_erased(self):self.spec['catalog_http']=False;self.fails('source grade')
 def test_no_ready_claim(self):self.edit(self.spec['directory']+'COMPLETION.json','factor_ready_count',1);self.fails('completion status')
 def test_unbound_registration_artifact(self):
  self.edit(self.spec['directory']+'PLAN.json','registration_artifact_hashes',{'outputs/prior/correction.json':'missing'});self.fails('unbound dependency')
 def test_no_relative_escape(self):
  with self.assertRaisesRegex(ValueError,'unsafe'):c.resolve(self.spec['directory'],'../elsewhere')
if __name__=='__main__':unittest.main(verbosity=2)
