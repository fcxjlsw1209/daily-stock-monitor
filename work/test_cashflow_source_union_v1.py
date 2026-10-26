"""Synthetic-only union contract tests; never open real financial outputs."""
from copy import deepcopy
import io
import os
import sys
import tempfile
import unittest
from unittest.mock import patch
import cashflow_source_union_v1 as m


def binding():
 return dict(schema_version=1,stage='reviewed_source_union_inputs',source_union_authorized=True,admission_authorized=False,
  producers={k:dict(directory='outputs/synthetic_'+k,plan_sha256='a'*64,manifest_sha256='b'*64,
   independent_reviews=[dict(path='synthetic/'+k+'_review.json',sha256='c'*64)]) for k in ('pdf','html')},
  accounting_review=dict(path='synthetic/accounting.json',sha256='d'*64))


def recount(data):
 n=len(data['events']);count=m.counted(data)
 data['counts']=dict(events=n,requested_cells=n*6,extracted_cells=count,gap_cells=n*6-count)
 return data


def dataset(events):
 return recount(dict(events=events,**{k:0 for k in m.ZERO_COUNTS}))


def cell(spec,side,n):
 return dict(value_decimal=str(n),displayed_amount=str(n),scale='1000000',currency_literal='$',currency_iso=None,
  source=deepcopy(spec['source']),period=deepcopy(spec['periods'][side]),available_from=spec['available_from'],
  locator=dict(synthetic_row=n),fact_admitted=False)


def gaps(event):
 event['gaps']=[dict(side=s,field=f,reason='synthetic_missing') for s in m.SIDES for f in m.FIELDS if event['fields'][s][f]is None]


def fixtures():
 specs=[];originals=[];pdf=[];html=[];reviews=[];html_position=0
 for i in range(44):
  symbol='AAPL' if i<10 else ('AMD','MSFT','NVDA')[i%3]
  spec=dict(event_id='synthetic:%s:%02d'%(symbol,i),symbol=symbol,fiscal_year=2030+i,
   release_date='2031-01-15',available_from='2031-01-16',
   periods={'current':dict(period_start='2030-01-01',period_end='2030-12-31',days_inclusive=365),
    'comparison':dict(period_start='2029-01-01',period_end='2029-12-31',days_inclusive=365)},
   source=dict(path='synthetic/source_%d.response'%i,sha256='1'*64,url='https://issuer.example/synthetic/%d'%i),
   identity=dict(synthetic_issuer=symbol),prior_field_admitted=i>=6,
   inherited_rejection_codes=['synthetic_legacy_hold'] if i<6 else [],
   prior_identity_evidence=dict(synthetic=True),locator_inventory={'synthetic':True})
  specs.append(spec)
  old={k:deepcopy(v) for k,v in spec.items() if k!='locator_inventory'}
  old.update(fields={s:{} for s in m.SIDES},gaps=[],metadata_gaps=[dict(reason='old_metadata_flag')],
   accounting_review_flags=[dict(code='old_accounting_flag')],fact_admitted=False,model_admitted=False,strategy_admitted=False)
  new=deepcopy(old);new['metadata_gaps']=[dict(reason='producer_metadata_flag')];new['accounting_review_flags']=[dict(code='producer_flag')]
  for s in m.SIDES:
   for f in m.FIELDS:
    n=i*6+len(old['fields'][s])+1
    before=None if symbol=='AAPL' or html_position>=157 else cell(spec,s,n)
    after=cell(spec,s,n) if symbol=='AAPL' or html_position<162 else None
    old['fields'][s][f]=before;new['fields'][s][f]=after
    if symbol!='AAPL':html_position+=1
  gaps(old);gaps(new);originals.append(old)
  (pdf if symbol=='AAPL' else html).append(new)
  reviews.append(dict(event_id=spec['event_id'],periods=deepcopy(spec['periods']),
   inherited_rejection_codes=deepcopy(spec['inherited_rejection_codes']),finding_ids=['synthetic_note'],
   fact_admitted=False,model_admitted=False,strategy_admitted=False))
 accounting=dict(events=reviews,findings=[dict(id='synthetic_note',remaining_limitations=['unresolved synthetic limitation'])],
  general_limits=['synthetic incomplete history'])
 return dict(events=specs),dataset(originals),dataset(pdf),dataset(html),accounting,binding()


def run_union(f):return m.consolidate(*f)


class UnionTests(unittest.TestCase):
 def setUp(self):self.f=fixtures()
 def test_exact_fixed_census_and_157_conservation(self):
  rows,positions,old=run_union(self.f)
  self.assertEqual((len(rows),len(positions),old),(44,264,157))
  self.assertEqual(sum(p['present'] for p in positions),222)
 def test_producer_event_deep_equality(self):
  rows,_,_=run_union(self.f);producers={e['event_id']:e for data in self.f[2:4] for e in data['events']}
  for row in rows:
   self.assertEqual({k:v for k,v in row.items() if k not in m.ADDED_EVENT_KEYS},producers[row['event_id']])
 def test_no_input_mutation(self):
  before=deepcopy(self.f);run_union(self.f);self.assertEqual(self.f,before)
 def test_original_event_order_retained(self):
  self.f[2]['events'].reverse();self.f[3]['events'].reverse()
  rows,_,_=run_union(self.f)
  self.assertEqual([e['event_id'] for e in rows],[e['event_id'] for e in self.f[0]['events']])
 def test_all_six_old_holds_retained(self):
  rows,_,_=run_union(self.f)
  self.assertEqual(sum(bool(e['inherited_rejection_codes']) for e in rows),6)
 def test_producer_and_baseline_flags_remain_separate(self):
  rows,_,_=run_union(self.f);row=rows[0]
  self.assertEqual(row['accounting_review_flags'],[dict(code='producer_flag')])
  self.assertEqual(row['union_provenance']['frozen_v1_accounting_review_flags'],[dict(code='old_accounting_flag')])
  self.assertEqual(row['union_provenance']['accounting_event_review'],self.f[4]['events'][0])
 def test_old_resolved_gaps_not_erased(self):
  rows,_,_=run_union(self.f)
  self.assertEqual(len(rows[0]['union_provenance']['frozen_v1_gaps']),6)
  self.assertEqual(rows[0]['gaps'],[])
 def test_missing_nonlegacy_cell_stays_null(self):
  new=self.f[2]['events'][0];new['fields']['current']['net_income']=None;gaps(new);recount(self.f[2])
  rows,positions,_=run_union(self.f)
  self.assertIsNone(rows[0]['fields']['current']['net_income'])
  self.assertIn('current.net_income',rows[0]['required_field_completeness']['missing_required_positions'])
 def test_partial_pdf_does_not_require_sixty(self):
  for event in self.f[2]['events']:
   event['fields']['current']['total_assets']=None;gaps(event)
  recount(self.f[2]);rows,positions,_=run_union(self.f)
  self.assertEqual(sum(p['present'] for p in positions),212)
 def test_comparison_flow_missing_does_not_change_required_four(self):
  event=self.f[2]['events'][0]
  for f in ('operating_cash_flow','net_income'):event['fields']['comparison'][f]=None
  gaps(event);recount(self.f[2]);rows,_,_=run_union(self.f)
  self.assertTrue(rows[0]['required_field_completeness']['all_four_required_positions_present'])
 def test_comparison_assets_are_required(self):
  event=self.f[2]['events'][0];event['fields']['comparison']['total_assets']=None;gaps(event);recount(self.f[2])
  rows,_,_=run_union(self.f)
  self.assertFalse(rows[0]['required_field_completeness']['all_four_required_positions_present'])
 def test_current_income_is_required(self):
  event=self.f[2]['events'][0];event['fields']['current']['net_income']=None;gaps(event);recount(self.f[2])
  rows,_,_=run_union(self.f)
  self.assertFalse(rows[0]['required_field_completeness']['all_four_required_positions_present'])
 def test_zero_and_negative_preserved_without_quality_filter(self):
  event=self.f[2]['events'][0]
  event['fields']['current']['net_income']['value_decimal']='-12.5'
  event['fields']['current']['operating_cash_flow']['value_decimal']='0'
  rows,_,_=run_union(self.f)
  self.assertEqual(rows[0]['fields']['current']['net_income']['value_decimal'],'-12.5')
 def test_existing_numeric_reexpression_is_equal_with_original_retained(self):
  event=self.f[3]['events'][0];old=self.f[1]['events'][10]['fields']['current']['net_income']
  event['fields']['current']['net_income']['value_decimal']=old['value_decimal']+'.00'
  rows,_,_=run_union(self.f)
  self.assertEqual(rows[10]['union_provenance']['frozen_v1_existing_cells']['current']['net_income'],old)
 def test_existing_value_conflict_fails(self):
  self.f[3]['events'][0]['fields']['current']['net_income']['value_decimal']='999999'
  with self.assertRaisesRegex(ValueError,'value conflict'):run_union(self.f)
 def test_existing_null_fails_without_fallback(self):
  event=self.f[3]['events'][0];event['fields']['current']['net_income']=None;gaps(event);recount(self.f[3])
  with self.assertRaisesRegex(ValueError,'fallback forbidden'):run_union(self.f)
 def test_existing_scale_change_fails_even_if_value_compensates(self):
  self.f[3]['events'][0]['fields']['current']['net_income']['scale']='1000'
  with self.assertRaisesRegex(ValueError,'scale'):run_union(self.f)
 def test_existing_currency_change_fails(self):
  self.f[3]['events'][0]['fields']['current']['net_income']['currency_iso']='USD'
  with self.assertRaisesRegex(ValueError,'currency_iso'):run_union(self.f)
 def test_source_change_fails(self):
  self.f[3]['events'][0]['fields']['current']['net_income']['source']['sha256']='2'*64
  with self.assertRaisesRegex(ValueError,'source'):run_union(self.f)
 def test_period_change_fails(self):
  self.f[2]['events'][0]['fields']['current']['net_income']['period']['period_end']='2030-12-30'
  with self.assertRaisesRegex(ValueError,'period'):run_union(self.f)
 def test_availability_change_fails(self):
  self.f[2]['events'][0]['fields']['current']['net_income']['available_from']='2031-01-15'
  with self.assertRaisesRegex(ValueError,'availability'):run_union(self.f)
 def test_identity_change_fails(self):
  self.f[2]['events'][0]['identity']={'synthetic_issuer':'OTHER'}
  with self.assertRaisesRegex(ValueError,'metadata'):run_union(self.f)
 def test_old_hold_dropped_fails(self):
  self.f[2]['events'][0]['inherited_rejection_codes']=[]
  with self.assertRaisesRegex(ValueError,'metadata'):run_union(self.f)
 def test_duplicate_producer_id_fails(self):
  self.f[2]['events'].append(deepcopy(self.f[2]['events'][0]));recount(self.f[2])
  with self.assertRaisesRegex(ValueError,'duplicate'):run_union(self.f)
 def test_missing_producer_id_fails(self):
  self.f[2]['events'].pop();recount(self.f[2])
  with self.assertRaisesRegex(ValueError,'partition'):run_union(self.f)
 def test_cross_producer_overlap_fails(self):
  self.f[3]['events'].append(deepcopy(self.f[2]['events'][0]));recount(self.f[3])
  with self.assertRaisesRegex(ValueError,'overlap'):run_union(self.f)
 def test_wrong_partition_fails(self):
  event=self.f[2]['events'].pop();self.f[3]['events'].append(event)
  recount(self.f[2]);recount(self.f[3])
  with self.assertRaisesRegex(ValueError,'partition'):run_union(self.f)
 def test_missing_gap_reason_fails(self):
  self.f[3]['events'][-1]['gaps']=[]
  with self.assertRaisesRegex(ValueError,'explicit gap'):run_union(self.f)
 def test_field_position_dropped_fails(self):
  del self.f[2]['events'][0]['fields']['current']['net_income']
  with self.assertRaises(ValueError):run_union(self.f)
 def test_event_admission_fails(self):
  self.f[2]['events'][0]['fact_admitted']=True
  with self.assertRaisesRegex(ValueError,'admission'):run_union(self.f)
 def test_cell_admission_fails(self):
  self.f[2]['events'][0]['fields']['current']['net_income']['model_admitted']=True
  with self.assertRaisesRegex(ValueError,'admission'):run_union(self.f)
 def test_nonzero_calculation_counter_fails(self):
  self.f[2]['returns_computed']=1
  with self.assertRaisesRegex(ValueError,'counter'):run_union(self.f)
 def test_bool_counter_is_not_zero_integer(self):
  self.f[2]['returns_computed']=False
  with self.assertRaisesRegex(ValueError,'counter'):run_union(self.f)
 def test_accounting_census_must_be_complete(self):
  self.f[4]['events'].pop()
  with self.assertRaisesRegex(ValueError,'Accounting review'):run_union(self.f)
 def test_accounting_cannot_remove_hold(self):
  self.f[4]['events'][0]['inherited_rejection_codes']=[]
  with self.assertRaisesRegex(ValueError,'inherited'):run_union(self.f)
 def test_accounting_cannot_admit_event(self):
  self.f[4]['events'][0]['strategy_admitted']=True
  with self.assertRaisesRegex(ValueError,'not admission'):run_union(self.f)
 def test_union_key_collision_fails(self):
  self.f[2]['events'][0]['union_provenance']={}
  with self.assertRaisesRegex(ValueError,'collides'):run_union(self.f)
 def test_metadata_phase_never_compares_amounts(self):
  with patch.object(m.base.safe,'decimal',side_effect=AssertionError('No amount comparisons')):
   m.metadata_contract(*self.f[:5])


class CertificationTests(unittest.TestCase):
 def test_global_certification_fails(self):
  f=fixtures();f[2]['first_public_version_verified']=True
  with self.assertRaisesRegex(ValueError,'cannot certify'):run_union(f)
 def test_event_certification_fails(self):
  f=fixtures();f[2]['events'][0]['certified_daily_security_master']=True
  with self.assertRaisesRegex(ValueError,'cannot certify'):run_union(f)
 def test_existing_false_certification_preserved(self):
  f=fixtures();f[2]['events'][0]['complete_correction_history_verified']=False
  rows,_,_=run_union(f)
  self.assertIs(rows[0]['complete_correction_history_verified'],False)


class BindingTests(unittest.TestCase):
 def test_explicit_source_only_binding(self):self.assertEqual(len(m.binding_contract(binding())),3)
 def test_no_review_fails(self):
  b=binding();b['producers']['pdf']['independent_reviews']=[]
  with self.assertRaises(ValueError):m.binding_contract(b)
 def test_source_authorization_required(self):
  b=binding();b['source_union_authorized']=False
  with self.assertRaises(ValueError):m.binding_contract(b)
 def test_admission_authorization_forbidden(self):
  b=binding();b['admission_authorized']=True
  with self.assertRaises(ValueError):m.binding_contract(b)
 def test_bad_sha_fails(self):
  b=binding();b['producers']['pdf']['plan_sha256']='abc'
  with self.assertRaises(ValueError):m.binding_contract(b)
 def test_parent_escape_fails(self):
  b=binding();b['producers']['pdf']['directory']='outputs/../elsewhere'
  with self.assertRaises(ValueError):m.binding_contract(b)
 def test_review_collision_fails(self):
  b=binding();b['accounting_review']=deepcopy(b['producers']['pdf']['independent_reviews'][0])
  with self.assertRaises(ValueError):m.binding_contract(b)
 def test_final_version_is_root_bound(self):
  b=binding();b['producers']['pdf']['directory']='outputs/synthetic_pdf_v99'
  self.assertEqual(len(m.binding_contract(b)),3)
 def test_explicit_plan_hash_before_reads(self):
  with patch.object(m.base,'read',side_effect=AssertionError('Must fail first')):
   with self.assertRaises(ValueError):m.products(m.ROOT,None)
 def test_source_schema_has_no_admission(self):
  schema=m.schema();self.assertEqual((schema['events'],schema['positions']),(44,264))
  self.assertIn('No data or strategy admission',schema['prohibitions'])
 def test_exclusive_publish(self):
  with tempfile.TemporaryDirectory() as root:
   m.base.publish(root,'a.json',{'a':1})
   with self.assertRaises(FileExistsError):m.base.publish(root,'a.json',{'a':2})
   self.assertEqual(m.base.read(root,'a.json'),{'a':1})


class BundleAndReviewTests(unittest.TestCase):
 def producer(self,root,kind='pdf',changes=None):
  directory='outputs/synthetic_'+kind+'_v99'
  m.base.publish(root,directory+'/dependency.json',{'fixture':True})
  deps={directory+'/dependency.json':m.base.sha(root,directory+'/dependency.json')}
  plan=dict(stage='frozen_synthetic_source_plan',runtime=sys.version,source_hashes=deps,source_code_hashes=deps)
  if kind=='html':plan.update({k:0 for k in ('financial_calculations','model_fits','ranks_computed','returns_computed','fact_admissions')})
  else:plan.update(financial_calculations_permitted=False,admission_permitted=False,network_requests_permitted=False)
  if changes and 'plan' in changes:plan.update(changes['plan'])
  m.base.publish(root,directory+'/PLAN.json',plan);plan_sha=m.base.sha(root,directory+'/PLAN.json')
  data=fixtures()[2 if kind=='pdf' else 3];data['plan_sha256']=plan_sha
  if kind=='html':data['counts'].update(existing_v1_cells_preserved=157,newly_parsed_cells=5)
  if changes and 'data' in changes:data.update(changes['data'])
  m.base.publish(root,directory+'/RAW_FACTS.json',data);data_sha=m.base.sha(root,directory+'/RAW_FACTS.json')
  audit=dict(passed=True,plan_sha256=plan_sha,data_sha256=data_sha,counts=data['counts'])
  if changes and 'audit' in changes:audit.update(changes['audit'])
  m.base.publish(root,directory+'/AUDIT.json',audit)
  manifest=dict(plan_sha256=plan_sha,output_hashes={'RAW_FACTS.json':data_sha,'AUDIT.json':m.base.sha(root,directory+'/AUDIT.json')})
  if changes and 'manifest' in changes:manifest.update(changes['manifest'])
  m.base.publish(root,directory+'/MANIFEST.json',manifest)
  return directory,plan_sha,m.base.sha(root,directory+'/MANIFEST.json')
 def test_versioned_pdf_bundle_checks_current_schema(self):
  with tempfile.TemporaryDirectory() as root:
   directory,plan,manifest=self.producer(root)
   result=m.bundle(root,'pdf',plan,manifest,directory)
   self.assertEqual(len(result['data']['events']),10)
 def test_html_zero_counter_contract_and_extended_counts(self):
  with tempfile.TemporaryDirectory() as root:
   directory,plan,manifest=self.producer(root,'html')
   result=m.bundle(root,'html',plan,manifest,directory)
   self.assertEqual(result['data']['counts']['existing_v1_cells_preserved'],157)
 def test_changed_dependency_fails(self):
  with tempfile.TemporaryDirectory() as root:
   directory,plan,manifest=self.producer(root)
   m.base.safe.local(root,directory+'/dependency.json').write_text('changed')
   with self.assertRaisesRegex(ValueError,'SHA mismatch'):m.bundle(root,'pdf',plan,manifest,directory)
 def test_wrong_manifest_plan_fails(self):
  with tempfile.TemporaryDirectory() as root:
   directory,plan,manifest=self.producer(root,changes={'manifest':{'plan_sha256':'e'*64}})
   with self.assertRaisesRegex(ValueError,'manifest plan'):m.bundle(root,'pdf',plan,manifest,directory)
 def test_wrong_data_plan_fails(self):
  with tempfile.TemporaryDirectory() as root:
   directory,plan,manifest=self.producer(root,changes={'data':{'plan_sha256':'e'*64}})
   with self.assertRaisesRegex(ValueError,'output plan'):m.bundle(root,'pdf',plan,manifest,directory)
 def test_wrong_audit_data_hash_fails(self):
  with tempfile.TemporaryDirectory() as root:
   directory,plan,manifest=self.producer(root,changes={'audit':{'data_sha256':'e'*64}})
   with self.assertRaisesRegex(ValueError,'audit mismatch'):m.bundle(root,'pdf',plan,manifest,directory)
 def test_failed_audit_fails(self):
  with tempfile.TemporaryDirectory() as root:
   directory,plan,manifest=self.producer(root,changes={'audit':{'passed':False}})
   with self.assertRaisesRegex(ValueError,'audit mismatch'):m.bundle(root,'pdf',plan,manifest,directory)
 def test_unfrozen_plan_fails(self):
  with tempfile.TemporaryDirectory() as root:
   directory,plan,manifest=self.producer(root,changes={'plan':{'stage':'draft_source_plan'}})
   with self.assertRaisesRegex(ValueError,'frozen stage'):m.bundle(root,'pdf',plan,manifest,directory)
 def test_enabled_financial_calculations_fail(self):
  with tempfile.TemporaryDirectory() as root:
   directory,plan,manifest=self.producer(root,changes={'plan':{'financial_calculations_permitted':True}})
   with self.assertRaisesRegex(ValueError,'source-only'):m.bundle(root,'pdf',plan,manifest,directory)
 def review_inputs(self,root):
  b=binding();bundles={};reviews={};deps={}
  for kind in ('pdf','html'):
   directory,plan,manifest=self.producer(root,kind)
   bundles[kind]=m.bundle(root,kind,plan,manifest,directory)
   b['producers'][kind].update(directory=directory,plan_sha256=plan,manifest_sha256=manifest)
   refs=b['producers'][kind]['independent_reviews']
   reviews[refs[0]['path']]=dict(passed=True,input_hashes=bundles[kind]['hashes'])
   deps.update(bundles[kind]['hashes'])
  accounting=dict(stage='source_only_field_scoped_accounting_review',source_only=True,source_admissions=0,
   plan_sha256=m.BASELINE_PINS[m.DIRECTORIES['baseline']+'/PLAN.json'],source_hashes=deps)
  return b,bundles,reviews,accounting
 def test_current_reviews_verify(self):
  with tempfile.TemporaryDirectory() as root:
   m.review_contract(root,*self.review_inputs(root))
 def test_stale_review_data_hash_fails(self):
  with tempfile.TemporaryDirectory() as root:
   b,bundles,reviews,accounting=self.review_inputs(root)
   review=reviews[b['producers']['pdf']['independent_reviews'][0]['path']]
   review.pop('input_hashes');review.update(data_sha256='e'*64,source_manifest_sha256=b['producers']['pdf']['manifest_sha256'])
   with self.assertRaisesRegex(ValueError,'current producer'):m.review_contract(root,b,bundles,reviews,accounting)
 def test_review_with_only_plan_hash_is_insufficient(self):
  with tempfile.TemporaryDirectory() as root:
   b,bundles,reviews,accounting=self.review_inputs(root)
   reviews[b['producers']['pdf']['independent_reviews'][0]['path']]=dict(passed=True,plan_sha256=b['producers']['pdf']['plan_sha256'])
   with self.assertRaisesRegex(ValueError,'current producer'):m.review_contract(root,b,bundles,reviews,accounting)
 def test_failed_independent_review_fails(self):
  with tempfile.TemporaryDirectory() as root:
   b,bundles,reviews,accounting=self.review_inputs(root)
   reviews[b['producers']['pdf']['independent_reviews'][0]['path']]['passed']=False
   with self.assertRaisesRegex(ValueError,'Passing'):m.review_contract(root,b,bundles,reviews,accounting)
 def test_accounting_admission_fails(self):
  with tempfile.TemporaryDirectory() as root:
   b,bundles,reviews,accounting=self.review_inputs(root);accounting['source_admissions']=1
   with self.assertRaisesRegex(ValueError,'Accounting'):m.review_contract(root,b,bundles,reviews,accounting)
 def test_accounting_wrong_baseline_fails(self):
  with tempfile.TemporaryDirectory() as root:
   b,bundles,reviews,accounting=self.review_inputs(root);accounting['plan_sha256']='e'*64
   with self.assertRaisesRegex(ValueError,'Accounting'):m.review_contract(root,b,bundles,reviews,accounting)


if __name__=='__main__':
 stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
 log=stream.getvalue();print(log,end='')
 if '--write-report' in sys.argv:
  for name in ('PLAN.json','PLAN_DRAFT.json'):
   m.require(not m.base.safe.local(m.ROOT,m.OUT+'/'+name).exists(),'Cannot replace tests bound by draft/freeze')
  path=m.base.safe.local(m.ROOT,m.OUT+'/TESTS.json');previous=[m.base.read(m.ROOT,m.OUT+'/TESTS.json')] if path.exists() else []
  report=dict(passed=result.wasSuccessful(),synthetic_only=True,tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
   real_source_unions=0,source_code_hashes=m.code_hashes(m.ROOT),log=log,previous_unfrozen_runs=previous)
  path.parent.mkdir(parents=True,exist_ok=True);fd,temporary=tempfile.mkstemp(prefix='.TESTS-',dir=path.parent)
  try:
   with os.fdopen(fd,'wb') as out:out.write(m.base.safe.encoded(report));out.flush();os.fsync(out.fileno())
   os.replace(temporary,path)
  finally:
   if os.path.exists(temporary):os.unlink(temporary)
 raise SystemExit(0 if result.wasSuccessful() else 1)
