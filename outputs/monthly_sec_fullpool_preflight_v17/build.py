"""Validate completed certificates; independently join this delta and count breadth."""
import argparse,hashlib,importlib.util,json
from pathlib import Path
P=Path(__file__).resolve().parent
ROOT=P.parent.parent
DESIGN_SHA='1dc544c01a13c857ae7aee9e832af4fc5b863a8c7b8e8199f08ada262727bff1'
IMPLEMENTATIONS=('build.py','preflight.py','certificates.py','register.py','test_preflight.py','test_certificates.py')
def load(path,name):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def blob(v):return (json.dumps(v,sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
def sha(b):return hashlib.sha256(b).hexdigest()
def require(ok,msg):
    if not ok:raise ValueError(msg)
class Documents:
    def __getitem__(self,path):
        p=Path(path);require(not p.is_absolute()and '..'not in p.parts and ROOT.resolve()in(ROOT/p).resolve().parents,'unsafe path');return(ROOT/p).read_bytes()
def bound(hashes):
    for path,expected in hashes.items():
        h=hashlib.sha256()
        with (ROOT/path).open('rb')as f:
            for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
        require(h.hexdigest()==expected,'source hash changed '+path)
def prepare():
    docs=Documents();raw=(P/'DESIGN.json').read_bytes();plan_raw=(P/'PLAN.json').read_bytes();d=json.loads(raw);p=json.loads(plan_raw)
    require(sha(raw)==DESIGN_SHA==p['design_sha256']and p['actual_executed']is False,'registration')
    require(p['source_hashes']==d['source_hashes']and set(p['implementation_hashes'])=={str((P/n).relative_to(ROOT))for n in IMPLEMENTATIONS},'manifest')
    require(p['tests_passed']>0 and p['independent_tests_passed']>0 and p['tests_sha256']==sha((P/'TESTS.txt').read_bytes()),'tests')
    all_hashes={**d['source_hashes'],**p['implementation_hashes'],**p['independent_review_hashes']};bound(all_hashes)
    cert=load(P/'certificates.py','v17_certificates');receipts=[cert.validate(s,docs,d['source_hashes'])for s in d['certificates']]
    merge=load(P/'preflight.py','v17_merge');validator=load(ROOT/'outputs/monthly_sec_fullpool_preflight_v1/preflight.py','v17_breadth')
    read=lambda path:json.loads(docs[path])
    batches={symbol:{'admissions':read(spec['directory']+'ADMISSIONS.json'),'evidence':read(spec['directory']+'EVIDENCE.json')}for symbol,spec in merge.SPECS.items()}
    result=merge.merge(read(validator.RECORDS),read(merge.BASE+'RESULT.json'),batches,read(validator.SRC+'SESSIONS.json'),d,validator.validate_universe,validator.breadth,read(validator.META))
    data=blob(result);proof=dict(schema_version=1,policy_id=d['policy_id'],design_sha256=DESIGN_SHA,plan_sha256=sha(plan_raw),source_hashes=d['source_hashes'],implementation_hashes=p['implementation_hashes'],independent_review_hashes=p['independent_review_hashes'],output_hashes={'RESULT.json':sha(data)},certificate_validation=receipts,completed_baseline_and_batch_certificates_verified=True,prior_financial_parsers_reexecuted=False,all_dependency_hashes_verified=True,new_delta_reconstructed=True,original3904_records_unchanged=True,prior1238_rows_preserved=True,factor_ready=False,strategy_trials=0,network_requests=0,new_semantic_admissions=0,observer_files_changed=False,independent_review_pending=True)
    bound(all_hashes);require((P/'DESIGN.json').read_bytes()==raw and(P/'PLAN.json').read_bytes()==plan_raw,'registration changed')
    return {'RESULT.json':data,'PROOF.json':blob(proof)}
def persist(outputs,verify=False):
    present=[n for n in outputs if(P/n).exists()]
    if present:
        require(len(present)==len(outputs)and all((P/n).read_bytes()==b for n,b in outputs.items()),'existing output differs');return'verified_existing'
    require(not verify,'nothing to verify')
    for n,b in outputs.items():
        with(P/n).open('xb')as f:f.write(b)
    return'created_once'
if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('action',choices=['build','verify']);args=cli.parse_args();out=prepare();print(persist(out,args.action=='verify'),json.loads(out['RESULT.json'])['semantic_admitted_count'])
