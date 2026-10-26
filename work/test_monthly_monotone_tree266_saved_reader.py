"""Synthetic saved-evidence validation; never open actual266 artifacts.

The one numeric fixture creates only generated six-security inputs to exercise
JSON serialization compatibility. The reader itself imports only stdlib and
never fits or decodes an NPZ. All other cases mutate synthetic saved metadata.
"""
import ast
import copy
from copy import deepcopy
from datetime import date,timedelta
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import monthly_monotone_tree266_saved_reader as reader


def native(value):return json.loads(json.dumps(value,allow_nan=False))
def setup_schemas():reader.SCHEMAS,reader.FLAGS=reader.schema_flags()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

class MetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):setup_schemas()
    def event(self,i=1,width=6):
        return dict(query_index=i,date='2016-08-17' if i==0 else '2016-08-31',initial_partial=i==0,weak=True,ranking_indices=[0,1,2],active_indices=[] if i==0 else [0,1,2],control_changed_delta_indices=[],candidate_changed_delta_indices=[],control_delta=[0.]*width,candidate_delta=[0.]*width,incumbent_choices=[0,1,2],control_choices=[0,1,2],candidate_choices=[0,1,2],episode_budget=.05)
    def test_stdlib_reader_imports_only(self):
        allowed={'argparse','ast','collections','datetime','hashlib','json','math','pathlib','platform','unittest','fractions','copy'}
        for node in ast.walk(ast.parse(Path(reader.__file__).read_text())):
            if isinstance(node,ast.Import):self.assertTrue(all(n.name in allowed for n in node.names))
            if isinstance(node,ast.ImportFrom):self.assertIn(node.module,allowed)
    def test_literal_22_17_14_13_schema(self):
        self.assertEqual([len(reader.SCHEMAS[k]) for k in ('MODEL_COUNT_FIELDS','COUNT_FIELDS','EVENT_KEYS','WORK_FIELDS')],[22,17,14,13])
    def test_fee_helpers_unchanged_exact_ast(self):
        original=ast.parse(reader.FEE_SOURCE.read_text());current=ast.parse(Path(reader.__file__).read_text())
        for name in ('diagnostic_holdings_parity','diagnostic_summary_parity'):
            nodes=[next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name==name) for tree in (original,current)]
            self.assertEqual(ast.dump(nodes[0]),ast.dump(nodes[1]))
    def test_typed_exact_signedzero_bool(self):
        for a,b in [(0.,-0.),(1,True),(1.,1)]:
            with self.subTest(a=a,b=b),self.assertRaises(ValueError):reader.exact(a,b,'fixture')
    def test_decision_counts_full_domain_and_changes(self):
        rows=[self.event(0),self.event(1)];rows[1]['candidate_delta'][2]=.01;rows[1]['candidate_changed_delta_indices']=[2];rows[1]['candidate_choices']=[0,1,3]
        c=reader.decision_counts(rows,2,12,6);self.assertEqual(c['ranking_cells'],6);self.assertEqual(c['weak_applicable_cells'],3);self.assertEqual(c['candidate_control_changed_holdings_events'],1)
        self.assertEqual(reader.decision_counts(rows,0,0,6)['incumbent_episode_calls'],1)
    def test_inactive_delta_or_choice_rejected(self):
        for target in ('delta','choices'):
            rows=[self.event(0)]
            if target=='delta':rows[0]['candidate_delta'][0]=.01;rows[0]['candidate_changed_delta_indices']=[0]
            else:rows[0]['candidate_choices']=[1,2,3]
            with self.subTest(target=target),self.assertRaises(ValueError):reader.decision_counts(rows,1,6,6)
    def test_candidate_delta_outside_originalU_rejected(self):
        rows=[self.event(0),self.event(1)];rows[1]['candidate_delta'][5]=.01;rows[1]['candidate_changed_delta_indices']=[5]
        with self.assertRaises(ValueError):reader.decision_counts(rows,2,12,6)
    def test_oversized_delta_and_eligible_understatement_rejected(self):
        rows=[self.event(0),self.event(1)]
        with self.assertRaises(ValueError):reader.decision_counts(rows,2,5,6)
        rows[1]['candidate_delta'][0]=.0376;rows[1]['candidate_changed_delta_indices']=[0]
        with self.assertRaises(ValueError):reader.decision_counts(rows,2,12,6)
    def test_runtime_exact_roles_years_limit(self):
        rows=[dict(role=role,year=2016,fit_threadpools=[dict(filepath='/synthetic/lib',num_threads=1)],predict_threadpools=[dict(filepath='/synthetic/lib',num_threads=1)]) for role in reader.TREE_ROLES]
        reader.runtime_check(rows,[2016]);reader.runtime_check([],[])
        for mutation in ('role','year','num_threads'):
            bad=deepcopy(rows)
            if mutation=='num_threads':bad[0]['fit_threadpools'][0][mutation]=2
            else:bad[0][mutation]='bad'
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):reader.runtime_check(bad,[2016])
    def test_noquery_projection_never_copies_future_model(self):
        class Poison:
            def __deepcopy__(self,memo):raise AssertionError('Future model copied')
        row=dict(year=2016,cutoff='2015-12-31',lower_bound='2010-12-31',training_row_ids=[1,2],training_months=18,status='fit',query_row_ids=Poison(),weights=Poison(),mean=Poison(),scale=Poison(),beta=Poison(),intercept=Poison(),gram=Poison(),rhs=Poison(),raw_predictions=Poison(),tree=Poison(),manual_predictions=Poison())
        ridge=reader.project_annual(row,[],'no_prediction_rows',True);tree=reader.project_annual(row,[],'no_prediction_rows',False)
        self.assertIsNone(ridge['beta']);self.assertIsNone(tree['tree']);self.assertEqual(tree['raw_predictions'],[])
    def test_query_projection_uses_globalIDs_not_position(self):
        row=dict(year=2016,cutoff='2015-12-31',training_row_ids=[1],status='fit',query_row_ids=[8,12,21],tree={'synthetic':True},raw_predictions=[1.,2.,3.],manual_predictions=[1.,2.,3.])
        self.assertEqual(reader.project_annual(row,[8,21],'fit',False)['raw_predictions'],[1.,3.])
        with self.assertRaises(ValueError):reader.project_annual(row,[99],'fit',False)
    def test_six_zero_rates_exact_including_full(self):
        result=dict(arms={a:{p:dict(metrics=dict(cost_per_side=0.,fees_paid=0.),slices={'full122':dict(cost_per_side=0.,fees_paid=0.)}) for p in reader.PATHS[:2]} for a in reader.ARMS})
        reader.zero_rates(result);result['arms'][reader.ARMS[2]][reader.PATHS[1]]['slices']['full122']['cost_per_side']=1e-100
        with self.assertRaises(ValueError):reader.zero_rates(result)
    def test_three_root_terminals_scope_and_artifact(self):
        plan='1'*64;hashes=dict(prepare='2'*64,select='3'*64,diagnose='4'*64)
        rows={p:dict(session_id=10,exit_code=0,chunk_id='chunk',artifact_path=str(reader.OUT/f),artifact_sha256=hashes[p]) for p,f in [('prepare','prepared/COMPLETION.json'),('select','selected/FROZEN.json'),('diagnose','results/COMPLETION.json')]}
        record=dict(plan_sha256=plan,phases=rows);reader.terminals_check(record,plan,hashes)
        for key,value in [('exit_code',1),('artifact_sha256','0'*64),('artifact_path',str(reader.OUT/'bad.json'))]:
            bad=deepcopy(record);bad['phases']['prepare'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):reader.terminals_check(bad,plan,hashes)
        del record['phases']['select']
        with self.assertRaises(ValueError):reader.terminals_check(record,plan,hashes)


class AdmissionTests(unittest.TestCase):
    def fixture(self):
        def metric(c=.2):return dict(cagr=c,sharpe=1.,max_drawdown=-.2,daily_return_sd=.01,sharpe_status='defined',sharpe_reason=None)
        m={a:{p:{g:metric(.21 if a==reader.ARMS[2] else .2) for g in ('training','early','recent')} for p in reader.PATHS[:2]} for a in reader.ARMS}
        o={a:[dict(symbols=['A','B','C']) for _ in range(122)] for a in reader.ARMS}
        for i in (2,54,60):o[reader.ARMS[2]][i]['symbols']=['A','B','D']
        return m,o
    def test_both_refs_first88_candidate_admitted(self):
        m,o=self.fixture();self.assertTrue(reader.saved_selection(m,o)['admitted'])
    def test_control_never_promoted_and_later_cannot_rescue(self):
        m,o=self.fixture();m[reader.ARMS[2]][reader.PATHS[0]]['training']['cagr']=.1
        for i in range(88,122):o[reader.ARMS[2]][i]['symbols']=['X','Y','Z']
        got=reader.saved_selection(m,o);self.assertFalse(got['admitted']);self.assertEqual(got['primary'],reader.ARMS[0])
    def test_must_beat_genuine_control(self):
        m,o=self.fixture();m[reader.ARMS[1]][reader.PATHS[1]]['training']['cagr']=.3
        self.assertFalse(reader.saved_selection(m,o)['admitted'])
    def test_order_permutation_not_changed_set(self):
        m,o=self.fixture();o[reader.ARMS[2]][60]['symbols']=['C','B','A']
        self.assertFalse(reader.saved_selection(m,o)['admitted'])
    def test_undefined_control_zeroSD_rejects(self):
        m,o=self.fixture();m[reader.ARMS[1]][reader.PATHS[0]]['training'].update(sharpe=None,daily_return_sd=0.,sharpe_status='undefined_sharpe',sharpe_reason='zero_daily_return_sd')
        self.assertFalse(reader.saved_selection(m,o)['admitted'])
    def test_cost_training_input_rejected(self):
        m,o=self.fixture();m[reader.ARMS[2]]['next_open_cost']=m[reader.ARMS[2]][reader.PATHS[0]]
        with self.assertRaises(ValueError):reader.saved_selection(m,o)


class SyntheticModelCompatibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        setup_schemas()
        import monthly_monotone_tree266_kernel as k
        import monthly_monotone_tree266_independent as ind
        import test_monthly_local_distance256_kernel as fixtures
        cls.k=k;cls.ind=ind;cls.args=fixtures.fixture();s,t,m,a=cls.args
        cls.gates={};cls.views={}
        def gate(view,stage):
            result=(ind.verify_calibration_models if stage=='calibration' else ind.verify_control_models)(t,m,a,view)
            cls.gates[stage]=native(result);cls.views[stage]=deepcopy(view);return result
        cls.bundle=k.fit_models(t,m,a,calibration_validator=lambda b:gate(b,'calibration'),control_validator=lambda b:gate(b,'control'))
        cls.full=native(ind.verify_models(t,m,a,cls.bundle))
        cls.orders,_,_,audit=k.decisions(s,a,m['monthly'],cls.bundle)
        cls.audit=native(audit);cls.orders=native(cls.orders)
        cls.ma=native(dict(model=cls.bundle['audit'],labels=cls.bundle['label_preservation'],**{role:cls.bundle[role]['audit'] for role in reader.ROLES}))
        old=k.legacy.fit_models(t,m,a);oo,ss,pp,od=k.legacy.decisions(s,a,m['monthly'],old)
        cls.reference=dict(forecasts={role+'_'+key:old[role][key] for role in ('control','candidate') for key in ('raw','available','rank')},model_audit=dict(model=old['audit'],labels=old['label_preservation'],control=old['control']['audit'],candidate=old['candidate']['audit']),labels=dict(training_row_ids=old['training_row_ids'],label_values=old['label_values']),scores=ss,pre_scores=pp,orders=oo,meta=m['monthly'])
        cls.historical=native(dict(meta=m['monthly'],model=dict(model=old['audit'],labels=old['label_preservation'],control=old['control']['audit'],candidate=old['candidate']['audit']),orders=oo,decision=od))
    def test_real_synthetic_saved_model_and_historical_projection(self):
        _,_,metadata,_=self.args
        summary,_=reader.model_summary(self.ma,metadata['monthly'],metadata,self.audit['events'],6)
        self.assertEqual(summary['counts']['actual_model_fits'],6)
        result=reader.historical_models_check(self.ma,self.historical,metadata['monthly'],self.audit['events'],self.orders)
        self.assertTrue(result['complete_tree_projection_exact'])
    def test_real_independent_stage_receipts_all_work_fields(self):
        cn=sum(len(r['tree']['shape_certificate']['constrained_nodes']) for r in self.ma[reader.ROLES[2]]['annual'] if r['status']=='fit')
        for kind,proof in [('calibration',self.gates['calibration']),('control',self.gates['control']),('models',self.full)]:reader.receipt(proof,4,kind,cn)
        bad=deepcopy(self.full);bad['shape_certificates_checked']-=1
        with self.assertRaises(ValueError):reader.receipt(bad,4,'models',cn)
    def test_annual_maturity_and_query_tamper_rejected(self):
        for key,value in [('training_row_ids',[0]),('query_row_ids',[]),('cutoff','2016-12-31')]:
            bad=deepcopy(self.ma);bad[reader.ROLES[0]]['annual'][0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):reader.model_summary(bad,self.args[2]['monthly'],self.args[2],self.audit['events'],6)
    def test_history_tree_and_signedzero_prediction_tamper(self):
        for target in ('tree','raw'):
            bad=deepcopy(self.ma);row=bad[reader.ROLES[1]]['annual'][0]
            if target=='tree':row['tree']['trees'][0]['nodes'][0]['value']=float.fromhex(row['tree']['trees'][0]['nodes'][0]['value']).__add__(1e-6).hex()
            else:row['raw_predictions'][0]=math.nextafter(row['raw_predictions'][0],math.inf)
            with self.subTest(target=target),self.assertRaises(ValueError):reader.historical_models_check(bad,self.historical,self.args[2]['monthly'],self.audit['events'],self.orders)
    def test_shape_certificate_recomputed_and_constraint_type(self):
        tree=deepcopy(self.ma[reader.ROLES[2]]['annual'][0]['tree']);reader.tree_structure(tree,reader.ROLES[2])
        tree['shape_certificate']['constrained_nodes']=[]
        if self.ma[reader.ROLES[2]]['annual'][0]['tree']['shape_certificate']['constrained_nodes']:
            with self.assertRaises(ValueError):reader.tree_structure(tree,reader.ROLES[2])
        tree=deepcopy(self.ma[reader.ROLES[2]]['annual'][0]['tree']);tree['parameters']['monotonic_cst'][0]=True
        with self.assertRaises(ValueError):reader.tree_structure(tree,reader.ROLES[2])
    def test_descendant_inversion_rejected_even_claimed_pass(self):
        from test_monthly_monotone_tree266_kernel import tiny_tree
        tree=native(tiny_tree(self.ma[reader.ROLES[2]]['annual'][0]['tree']))
        reader.tree_structure(tree,reader.ROLES[2]);tree['trees'][0]['nodes'][1]['value']=3.0.hex()
        with self.assertRaisesRegex(ValueError,'descendant'):reader.tree_structure(tree,reader.ROLES[2])
    def test_layout_digest_matches_host_numpy_representation(self):
        import monthly_monotone_tree266_host as host
        s,t,metadata,a=self.args
        layout=host.reference_layout(a,metadata['monthly'],self.bundle,4)
        ids=[row['ranking_indices'] for row in self.audit['events']]
        self.assertEqual(reader.layout_digest(4,6,metadata['monthly'],layout['annual_cohorts'],ids),host.structured_digest(layout))
    def test_reference_envelopes_counts_match_real_synthetic_helper(self):
        import monthly_monotone_tree266_host as host
        import monthly_monotone_tree266_reference as ref
        s,t,metadata,a=self.args;binding=dict(prepared_completion_sha256='a'*64)
        for stage in ('calibration','control'):
            view=self.views[stage];layout=host.reference_layout(a,metadata['monthly'],view,4)
            cap=ref.bind_verified_inputs(view,layout,self.gates[stage],stage=stage)
            strategy=(self.k.calibration_decisions if stage=='calibration' else self.k.control_decisions)(s,a,metadata['monthly'],view)
            compare=ref.compare_calibration_reference if stage=='calibration' else ref.compare_control_reference
            proof=compare(self.reference,view,layout,independent_receipt=self.gates[stage],verification_binding=cap,orders=strategy[0],scores=strategy[1],pre_scores=strategy[2])
            envelope=native(host.reference_envelope(proof,layout,binding))
            reader.reference_envelope_check(envelope,4,6,stage,binding,self.ma,metadata['monthly'],self.audit['events'],self.ma['model']['counts'])
            bad=deepcopy(envelope);bad['comparison']['counts']['tree_models_compared']+=1
            with self.assertRaises(ValueError):reader.reference_envelope_check(bad,4,6,stage,binding,self.ma,metadata['monthly'],self.audit['events'],self.ma['model']['counts'])

    def test_control_mapping_cannot_use_ridge_or_incumbent(self):
        bad=deepcopy(self.historical);bad['model']['candidate']=bad['model']['control']
        with self.assertRaises(ValueError):reader.historical_models_check(self.ma,bad,self.args[2]['monthly'],self.audit['events'],self.orders)


def holdings(fees=None):
    fees=[1.]+[0.]*121 if fees is None else fees
    start=date(2001,1,1)
    return [dict(signal_date=(start+timedelta(days=31*i)).isoformat(),
        execution_date=(start+timedelta(days=31*i+1)).isoformat(),symbols=['A','B','C'],scores=[1.,.9,.8],fee=fee)
        for i,fee in enumerate(fees)]


def summaries(rows):
    total=0.
    for row in rows:total+=row['fee']
    result={}
    for path in reader.PATHS:
        fee=total if path=='next_open_cost' else 0.
        metrics=dict(fees_paid=fee,cost_per_side=.0005 if path=='next_open_cost' else 0.,
            cagr=.15,sharpe=.8,total_return=2.,max_drawdown=-.3,final_equity=300000.,
            dollar_turnover=1000.,trade_rows=30,execution='open' if path!='same_close_zero' else 'close')
        slices={name:dict(metrics) for name in ('full122','first88','early53','recent35','later34')}
        for name in slices:
            if name!='full122':slices[name]['fees_paid']=0.
        result[path]=dict(metrics=metrics,slices=slices,accounting=dict(exported_fill_ledger_matches=True,terminal_liquidation=False))
    return result


class DiagnosticSummaryParity(unittest.TestCase):
    def setUp(self):
        self.previous_rows=holdings();self.current_rows=copy.deepcopy(self.previous_rows)
        self.current_rows[0]['fee']=math.nextafter(1.,math.inf)
        self.previous=summaries(self.previous_rows);self.current=summaries(self.current_rows)

    def review(self):
        return reader.diagnostic_summary_parity(self.current,self.previous,self.current_rows,self.previous_rows)

    def test_lowbit_fee_only_accepted_with_honest_exact_flag(self):
        review=self.review()
        self.assertFalse(review['exact']);self.assertTrue(review['economic_summary_metrics_exact'])
        self.assertTrue(review['own_totals_exact']);self.assertTrue(review['full122_duplicate_exact'])
        self.assertFalse(review['aggregate_tolerance_used'])
        self.assertEqual(len(review['display_fee_review']['differences']),1)
        self.assertEqual(review['allowed_difference_paths'],['next_open_cost.metrics.fees_paid','next_open_cost.slices.full122.fees_paid'])

    def test_identical_summary_and_fees_are_exact(self):
        self.current=copy.deepcopy(self.previous);self.current_rows=copy.deepcopy(self.previous_rows)
        self.assertTrue(self.review()['exact'])

    def test_identical_totals_can_coexist_with_lowbit_display_difference(self):
        self.previous_rows=holdings([1.,1.]+[0.]*120);self.current_rows=copy.deepcopy(self.previous_rows)
        self.current_rows[0]['fee']=math.nextafter(1.,math.inf)
        self.previous=summaries(self.previous_rows);self.current=summaries(self.current_rows)
        review=self.review()
        self.assertTrue(review['exact']);self.assertFalse(review['display_fee_review']['exact'])

    def test_falsified_own_sum_rejected_even_one_ulp(self):
        for side in ('current','previous'):
            saved=copy.deepcopy(getattr(self,side));bad=copy.deepcopy(saved)
            fee=bad['next_open_cost']['metrics']['fees_paid'];fee=math.nextafter(fee,math.inf)
            bad['next_open_cost']['metrics']['fees_paid']=fee;bad['next_open_cost']['slices']['full122']['fees_paid']=fee
            setattr(self,side,bad)
            with self.subTest(side=side),self.assertRaisesRegex(ValueError,'Exact own chronological'):self.review()
            setattr(self,side,saved)

    def test_full122_duplicate_must_equal_own_sum(self):
        for side in ('current','previous'):
            saved=copy.deepcopy(getattr(self,side));getattr(self,side)['next_open_cost']['slices']['full122']['fees_paid']=0.
            with self.subTest(side=side),self.assertRaisesRegex(ValueError,'Exact own chronological'):self.review()
            setattr(self,side,saved)

    def test_all_nonfee_summary_changes_rejected(self):
        mutations=[('same_close_zero','metrics','fees_paid'),('next_open_zero','metrics','cagr'),
                   ('next_open_cost','metrics','cagr'),('next_open_cost','metrics','dollar_turnover'),
                   ('next_open_cost','slices','first88','fees_paid'),('next_open_cost','slices','full122','cagr')]
        for path in mutations:
            saved=copy.deepcopy(self.current);target=self.current
            for key in path[:-1]:target=target[key]
            target[path[-1]]=math.nextafter(target[path[-1]],math.inf)
            with self.subTest(path=path),self.assertRaisesRegex(ValueError,'Every non-display-fee'):self.review()
            self.current=saved

    def test_path_keys_and_fee_leaf_presence_strict(self):
        for change in ('rename','extra','missing_metrics_fee','missing_duplicate'):
            saved=copy.deepcopy(self.current)
            if change=='rename':self.current['renamed']=self.current.pop('same_close_zero')
            elif change=='extra':self.current['extra']=copy.deepcopy(self.current['same_close_zero'])
            elif change=='missing_metrics_fee':del self.current['next_open_cost']['metrics']['fees_paid']
            else:del self.current['next_open_cost']['slices']['full122']['fees_paid']
            with self.subTest(change=change),self.assertRaises(ValueError):self.review()
            self.current=saved

    def test_fee_rate_not_tolerated_even_both_changed(self):
        for rate in (0.,math.nextafter(.0005,math.inf),True,float('nan'),float('inf'),'0.0005'):
            for side in ('current','previous'):getattr(self,side)['next_open_cost']['metrics']['cost_per_side']=rate
            with self.subTest(rate=rate),self.assertRaises(ValueError):self.review()

    def test_bad_summary_fee_numbers_rejected(self):
        for bad in (True,False,None,'1',-1.,float('nan'),float('inf'),float('-inf')):
            for side in ('current','previous'):
                saved=copy.deepcopy(getattr(self,side))
                for container in ('metrics','full122'):
                    item=getattr(self,side)['next_open_cost'];item=item['metrics'] if container=='metrics' else item['slices']['full122']
                    item['fees_paid']=bad
                    with self.subTest(bad=bad,side=side,container=container),self.assertRaises(ValueError):self.review()
                    setattr(self,side,copy.deepcopy(saved))

    def test_bad_holdings_fee_numbers_and_sum_overflow_rejected(self):
        for bad in (True,None,'1',-1.,float('nan'),float('inf')):
            saved=self.current_rows[0]['fee'];self.current_rows[0]['fee']=bad
            with self.subTest(bad=bad),self.assertRaises(ValueError):self.review()
            self.current_rows[0]['fee']=saved
        huge=float.fromhex('0x1.fffffffffffffp+1023')
        self.previous_rows=holdings([huge,huge]);self.current_rows=copy.deepcopy(self.previous_rows)
        with self.assertRaisesRegex(ValueError,'Finite chronological display fee total'):self.review()

    def test_dates_order_and_nonfee_holdings_are_exact(self):
        for field,value in (('signal_date','2000-12-31'),('execution_date','2001-01-03'),('symbols',['X']),('scores',[1.,1.,1.])):
            saved=copy.deepcopy(self.current_rows);self.current_rows[0][field]=value
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'Non-fee holdings'):self.review()
            self.current_rows=saved
        self.current_rows[0],self.current_rows[1]=self.current_rows[1],self.current_rows[0]
        with self.assertRaisesRegex(ValueError,'Non-fee holdings'):self.review()

    def test_both_reordered_duplicate_or_malformed_dates_rejected(self):
        for change in ('reordered','duplicate','malformed','execution_before_signal'):
            saved=(copy.deepcopy(self.current_rows),copy.deepcopy(self.previous_rows))
            for rows in (self.current_rows,self.previous_rows):
                if change=='reordered':rows[0],rows[1]=rows[1],rows[0]
                elif change=='duplicate':rows[1]['execution_date']=rows[0]['execution_date']
                elif change=='malformed':rows[0]['signal_date']='2001-1-1'
                else:rows[0]['execution_date']='2000-12-31'
            with self.subTest(change=change),self.assertRaises(ValueError):self.review()
            self.current_rows,self.previous_rows=saved

    def test_existing_perperiod_bound_and_zero_rules_unchanged(self):
        for fee in (0.,2.,1.+2**-45):
            self.current_rows[0]['fee']=fee;self.current=summaries(self.current_rows)
            with self.subTest(fee=fee),self.assertRaisesRegex(ValueError,'Zero display|binary64 bound'):self.review()

    def test_chronological_plus_equal_not_fsum(self):
        self.previous_rows=holdings([2.**53,1.,1.]);self.current_rows=copy.deepcopy(self.previous_rows)
        self.previous=summaries(self.previous_rows);self.current=copy.deepcopy(self.previous)
        review=self.review();self.assertEqual(review['own_chronological_fee_totals']['current'],2.**53)
        self.assertNotEqual(math.fsum(row['fee'] for row in self.current_rows),2.**53)
        for side in ('current','previous'):
            getattr(self,side)['next_open_cost']['metrics']['fees_paid']=2.**53+2.
            getattr(self,side)['next_open_cost']['slices']['full122']['fees_paid']=2.**53+2.
        with self.assertRaisesRegex(ValueError,'Exact own chronological'):self.review()

class FormalBindingsTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='266-reader-',dir='/private/tmp');self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name);work=self.root/'work';work.mkdir()
        self.paths={name:work/('monthly_monotone_tree266_'+suffix) for name,suffix in [('CONTEXT','context.json'),('CONTRACT','IMPLEMENTATION_CONTRACT.json'),('TESTS','TESTS.json'),('CODE_REVIEW','CODE_REVIEW.json')]}
        self.code={}
        for stem in ('host','kernel','independent','admission','reference'):
            for name in ('monthly_monotone_tree266_'+stem+'.py','test_monthly_monotone_tree266_'+stem+'.py'):
                p=work/name;p.write_text('# synthetic formal source\n');self.code[str(p)]=sha(p)
        self.reader_path=work/'monthly_monotone_tree266_saved_reader.py';self.reader_path.write_text(Path(reader.__file__).read_text())
        self.fee=work/reader.FEE_SOURCE.name;self.fee.write_text(reader.FEE_SOURCE.read_text())
        for name in ('monthly_weak_tree261_kernel.py','monthly_calendar_clock248_v2_independent.py','test_monthly_monotone_tree266_saved_reader.py'):(work/name).write_text('# synthetic pinned dependency\n')
        self.context=dict(frozen=True,registration_authorized=True,actual_data_started=False,arms=list(reader.ARMS),registered_trials=3,only_promotable=reader.ARMS[2],diagnostic_only=reader.ARMS[1]);self.write('CONTEXT',self.context)
        self.contract=dict(frozen=True,registration_authorized=True,actual_data_started=False,context_sha256=sha(self.paths['CONTEXT']));self.write('CONTRACT',self.contract)
        runtime=dict(python='3.9.6',numpy='2.0.2',pandas='2.3.3')
        self.tests=dict(passed=True,synthetic_only=True,tests=100,runtime=runtime,code_sha256=self.code);self.write('TESTS',self.tests)
        self.review=dict(passed=True,unresolved_blockers=[],actual_returns_computed=False,code_sha256=self.code,context_sha256=sha(self.paths['CONTEXT']),implementation_contract_sha256=sha(self.paths['CONTRACT']));self.write('CODE_REVIEW',self.review)
        self.plan=dict(context=self.context,implementation_contract=self.contract,context_sha256=sha(self.paths['CONTEXT']),implementation_contract_sha256=sha(self.paths['CONTRACT']),runtime=runtime,input_sha256={str(p):sha(p) for p in work.iterdir()})
    def write(self,key,v):self.paths[key].write_text(json.dumps(v,sort_keys=True))
    def check(self):
        with patch.multiple(reader,ROOT=self.root,FIXED={},FEE_SOURCE=self.fee,__file__=str(self.reader_path),**self.paths):
            evidence=reader.Evidence();out=reader.formal_bindings(evidence,self.plan)
            return out,evidence.pins
    def test_registered_ten_files_and_top_level_context_binding(self):
        (context,contract,tests),pins=self.check();self.assertEqual(tests['tests'],100);self.assertEqual(context,self.context);self.assertEqual(contract,self.contract)
        self.assertTrue(all(str(Path(k).relative_to(self.root)) in pins for k in self.code))
    def test_legacy_nested_context_binding_rejected(self):
        contract=dict(self.contract);contract.pop('context_sha256');contract['context']={'sha256':sha(self.paths['CONTEXT'])};self.write('CONTRACT',contract);self.plan.update(implementation_contract=contract,implementation_contract_sha256=sha(self.paths['CONTRACT']));self.plan['input_sha256'][str(self.paths['CONTRACT'])]=sha(self.paths['CONTRACT'])
        with self.assertRaises((KeyError,ValueError)):self.check()
    def test_missing_formal_code_pin_rejected(self):
        self.plan['input_sha256'].pop(next(iter(self.code)))
        with self.assertRaisesRegex(ValueError,'Missing/conflicting'):self.check()
    def test_nine_file_review_not_complete(self):
        bad=deepcopy(self.tests);bad['code_sha256'].pop(next(iter(bad['code_sha256'])));self.write('TESTS',bad)
        review=deepcopy(self.review);review['code_sha256']=bad['code_sha256'];self.write('CODE_REVIEW',review)
        for name in ('TESTS','CODE_REVIEW'):self.plan['input_sha256'][str(self.paths[name])]=sha(self.paths[name])
        with self.assertRaisesRegex(ValueError,'Exactly ten'):self.check()
    def test_unregistered_reader_or_changed_source_rejected(self):
        self.reader_path.write_text(self.reader_path.read_text()+'\n# change\n')
        with self.assertRaisesRegex(ValueError,'Digest mismatch'):self.check()



def empty_preparation_fixture():
    """No models: fabricated cold source for complete metadata/ledger coverage."""
    import calendar
    setup_schemas();dates=['2016-08-17']
    for k in range(121):
        month=(7+k)%12+1;year=2016+(7+k)//12;dates.append('%04d-%02d-%02d'%(year,month,calendar.monthrange(year,month)[1]))
    names=['S'+str(i) for i in range(49)];meta=dict(dates=dates,ends=dates[1:]+['2026-09-24'],symbols=[names[:] for _ in dates],etfs=[])
    events=[dict(query_index=i,date=d,initial_partial=i==0,weak=False,ranking_indices=[],active_indices=[],control_changed_delta_indices=[],candidate_changed_delta_indices=[],control_delta=[0.]*49,candidate_delta=[0.]*49,incumbent_choices=[],control_choices=[],candidate_choices=[],episode_budget=.05) for i,d in enumerate(dates)]
    def counts(n):
        c=dict.fromkeys(reader.SCHEMAS['MODEL_COUNT_FIELDS'],0);c['events']=n;c['ordinary_years']=c['cold_years']=len({d[:4] for d in dates[1:n]});return c
    def audit(n):return dict(schema='monthly_monotone_tree266_model_audit_v1',counts=counts(n),original_cache_parity=dict(passed=True,events=n,original_raw_exact=True,original_availability_exact=True,original_ranks_exact=True,candidate_models_computed=False),original_row_order_preserved=True,original_common_good_preserved=True,all_targets_preserved=True,local_models_reused=True,other_targets_not_fitted=True,weak_only=True,computes_returns=False,execution_runtime=[])
    def receipt(n,kind):
        r=dict(passed=True,checked_events=n,producer_modules_imported=False,computes_returns=False,**dict.fromkeys(reader.WORK_FIELDS,0))
        flag={'calibration':'CALIBRATION_FLAGS','control':'CONTROL_FLAGS','models':'MODEL_FLAGS','incumbent':'INCUMBENT_FLAGS','control_strategy':'CONTROL_SELECTION_FLAGS','selection':'SELECTION_FLAGS'}[kind];r.update({k:True for k in reader.FLAGS[flag]})
        if kind in ('calibration','control','models'):
            r.update(counts=counts(n),producer_equivalent_counts=counts(n),continuous_atol=2e-10,continuous_rtol=2e-10,rank_tolerance=0.,exact_discrete=True,full_bundle_supplied=True,full_calibration_supplied=True,full_control_supplied=kind!='calibration',full_candidate_supplied=kind=='models',full_label_preservation_supplied=True,full_model_audit_supplied=True,candidate_models_computed=kind=='models',shared_tree_trainer=True,independent_tree_training_algorithm=False,independent_numeric_system=reader.SCHEMAS['NUMERIC_SYSTEM'])
        else:
            r.update(full_scores_supplied=True,full_pre_scores_supplied=True,numerical_tolerance=0.,ordered_choices_checked=(3 if kind=='selection' else 1)*n,original_episode_calls=3 if kind=='selection' else 1)
            if kind=='selection':r.update(full_bundle_supplied=True,full_audit_supplied=True,counts=reader.decision_counts(events,n,0))
            else:r['candidate_selection_computed']=False
            if kind=='control_strategy':r['full_control_supplied']=True
        return r
    years=sorted({int(d[:4]) for d in dates[1:]});annual=[dict(year=y,cutoff=f'{y-1}-12-31',lower_bound=f'{y-6}-12-31',training_row_ids=[],query_row_ids=[],training_months=0,status='cold_start',weights=[],mean=None,scale=None,beta=None,intercept=None,gram=None,rhs=None,raw_predictions=[]) for y in years]
    ma=dict(model=audit(122),calibration_ridge=dict(schema='monthly_monotone_tree266_ridge_audit_v1',annual=annual))
    for role in reader.TREE_ROLES:ma[role]=dict(schema='monthly_monotone_tree266_tree_audit_v1',annual=[{k:v for k,v in r.items() if k in ('year','cutoff','training_row_ids','query_row_ids','status','raw_predictions')}|dict(tree=None,manual_predictions=[]) for r in annual])
    binding=dict(prepared_completion_sha256='a'*64)
    def reference(n,stage):
        yy=sorted({int(d[:4]) for d in dates[1:n]});cohorts=[{k:r[k] for k in ('year','cutoff','lower_bound','training_row_ids','training_months')} for r in annual if r['year'] in yy]
        c=dict.fromkeys(reader.SCHEMAS['REFERENCE_COUNT_FIELDS'],0);c.update(forecast_numeric_cells_compared=2*n*49,availability_cells_compared=n*49,annual_metadata_rows_compared=len(yy),score_numeric_cells_compared=2*n*49,ordered_events_compared=n,verification_binding_checks=1)
        if stage=='calibration':c.update(calibration_incumbent_score_cells_compared=2*n*49,calibration_incumbent_ordered_events_compared=n)
        proof=dict(schema='monthly_monotone_tree266_historical_'+stage+'_reference_v1',passed=True,checked_events=n,width=49,reference_role='control' if stage=='calibration' else 'candidate',current_role='calibration_ridge' if stage=='calibration' else 'unconstrained_control',source_layout_independently_verified_by_caller=True,reference_manifest_verified_by_caller=True,receipt_authentication_performed=False,original_query_domain_used=True,prefix_fit_state_from_verified_source=True,full_annual_fit_not_borrowed=True,exact_bits_and_types=True,numeric_tolerance=0.,candidate_models_computed=False,model_fits=0,computes_returns=False,counts=c)
        return dict(reference_prepared_sha256='a'*64,source_layout_sha256=reader.layout_digest(n,49,meta,cohorts,[[] for _ in range(n)]),consumed_years=yy,annual_cohorts=cohorts,original_u_indices=[[] for _ in range(n)],comparison=proof)
    def one(n):
        v={kind:receipt(n,kind) for kind in ('calibration','control','models','incumbent','selection')};v['early_incumbent']=receipt(n,'incumbent')
        v.update(main_model_counts=audit(n),historical_reference_binding=binding,original_reference=dict(passed=True,events=n,source_incumbent_orders_exact=True,prior_incumbent_exact=True,reference261_incumbent_exact=True))
        for stage in ('calibration','control'):
            v[stage+'_strategy']=dict(passed=True,events=n,pre_scores_exact=True,final_scores_exact=True,ordered_choices_exact=True,producer_episode_calls=1,independent=receipt(n,'incumbent' if stage=='calibration' else 'control_strategy'))
            if stage=='calibration':v[stage+'_strategy'].update(matched_ranks_exact=True,delta_exact_zero=True)
            v['reference261_'+stage]=reference(n,stage)
        return v
    main=one(122);physical=[]
    for n in reader.PREFIXES:
        v=one(n);v.update(prefix=n,exact=True,model_recomputed=n!=122,model=v.pop('models'),model_counts=v.pop('main_model_counts'))
        if n==122:v['reuses_main_verified_full']=True
        else:v['input_scope']=dict(events=n,training_months=0,training_rows=0,cutoff=f'{int(dates[n-1][:4])-1}-12-31' if n>1 else None,global_row_ids_start=0,global_row_ids_preserved=True,training_numeric_consumed=False,physically_shortened=True)
        physical.append(v)
    main.update(all123_order_prefixes=True,original_arrays_unchanged=True,original_training_unchanged=True,historical_reference_unchanged=True,physical_prefixes=physical,cached_prefixes=[dict(prefix=n,exact=True,incumbent=receipt(n,'incumbent'),selection=receipt(n,'selection')) for n in range(123)])
    import monthly_monotone_tree266_host as host
    main['fit_execution_accounting']=native(host.execution_accounting(main))
    da=dict(events=events,counts=reader.decision_counts(events,122,0),incumbent_gate=main['incumbent'])
    return main,ma,da,meta,dict(training=dict(dates=[],ends=[],symbols=[])),[]


class CompleteLedgerTests(unittest.TestCase):
    def setUp(self):self.args=empty_preparation_fixture()
    def test_host_and_reader_complete_123_and_reuse_ledger(self):
        got=reader.preparation_evidence(*self.args)
        self.assertEqual(got['model_invocations'],5);self.assertEqual(got['totals']['producer_episode_calls'],399);self.assertEqual(got['totals']['original_episode_calls'],527);self.assertEqual(got['totals']['total_model_fit_calls'],0)
        self.assertEqual(len(got['entries']),5)
    def test_wrong_cached_decision_count_rejected(self):
        self.args[0]['cached_prefixes'][53]['selection']['counts']['weak_applicable_cells']=1
        with self.assertRaises(ValueError):reader.preparation_evidence(*self.args)
    def test_full122_alias_tamper_rejected_not_just_excluded_from_sum(self):
        self.args[0]['physical_prefixes'][-1]['model']['independent_canonical_solves']=9999
        with self.assertRaises(ValueError):reader.preparation_evidence(*self.args)
    def test_reference_layout_or_cohort_tamper_rejected(self):
        for target in ('source_layout_sha256','annual_cohorts'):
            args=deepcopy(self.args);env=args[0]['physical_prefixes'][2]['reference261_control']
            if target=='source_layout_sha256':env[target]='0'*64
            else:env[target][0]['training_row_ids']=[0]
            with self.subTest(target=target),self.assertRaises(ValueError):reader.preparation_evidence(*args)
    def test_missing_model_or_historical_work_rejected(self):
        for kind in ('work','reference'):
            args=deepcopy(self.args)
            if kind=='work':args[0]['fit_execution_accounting']['totals']['original_episode_calls']-=1
            else:args[0]['fit_execution_accounting']['historical_reference_comparison_work']['verification_binding_checks']-=1
            with self.subTest(kind=kind),self.assertRaises(ValueError):reader.preparation_evidence(*args)
    def test_physical_input_boundary_not_relabelled_full(self):
        self.args[0]['physical_prefixes'][2]['input_scope']['physically_shortened']=False
        with self.assertRaises(ValueError):reader.preparation_evidence(*self.args)
    def test_partial_no_model_claim_and_full_before_candidate_scope(self):
        for scope in ('partial','full'):
            args=deepcopy(self.args)
            if scope=='partial':args[0]['physical_prefixes'][1]['calibration']['counts']['training_rows']=1
            else:args[0]['models']['full_candidate_supplied']=False
            with self.subTest(scope=scope),self.assertRaises(ValueError):reader.preparation_evidence(*args)



class HistoricalManifestTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='266-history-',dir='/private/tmp');self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.old=self.root/'outputs/monthly_weak_tree261';prepared=self.old/'prepared';prepared.mkdir(parents=True)
        (self.old/'PLAN.json').write_text('{}');self.plan_hash=sha(self.old/'PLAN.json')
        artifacts={}
        for name in reader.PREP:
            path=prepared/name
            path.write_bytes(b'opaque synthetic npz bytes' if path.suffix=='.npz' else b'{}');artifacts[name]=sha(path)
        self.pc=dict(status='COMPLETE',events=122,new_returns_computed=False,plan_sha256=self.plan_hash,artifacts_sha256=artifacts)
        (prepared/'COMPLETION.json').write_text(json.dumps(self.pc))
        self.c=dict(status='COMPLETED_INDEPENDENTLY_VERIFIED',protected_files=21444,plan_sha256=self.plan_hash,model_audit_sha256=artifacts['MODEL_AUDIT.json'],independent_prepare_sha256=artifacts['INDEPENDENT_PREPARE.json'])
        (self.old/'COMPLETION.json').write_text(json.dumps(self.c));self.refresh()
    def refresh(self):
        files={str(p):sha(p) for p in list((self.old/'prepared').iterdir())+[self.old/'PLAN.json',self.old/'COMPLETION.json']}
        self.plan=dict(input_sha256=files);self.binding=dict(schema='monthly_monotone_tree266_historical_binding_v1',files_sha256=files,completion_sha256=files[str(self.old/'COMPLETION.json')],prepared_completion_sha256=files[str(self.old/'prepared/COMPLETION.json')],plan_sha256=files[str(self.old/'PLAN.json')],prepared_members=sorted(reader.PREP),npz_decoded=False,hash_scope='All13 complete file bytes; hashing is not numerical prefix consumption')
    def check(self):
        with patch.multiple(reader,ROOT=self.root,HISTORICAL=self.old):return reader.reference_chain(reader.Evidence(),self.plan,self.binding)
    def test_complete13_hashes_and_json_only_projection(self):
        result=self.check();self.assertEqual(set(result),{'model','meta','orders','decision'})
        self.assertEqual(len(self.binding['files_sha256']),13)
    def test_changed_opaque_forecast_bytes_rejected(self):
        (self.old/'prepared/FORECASTS.npz').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'Digest mismatch'):self.check()
    def test_caller_passed_flags_cannot_replace_member_inventory(self):
        self.binding['files_sha256'].pop(str(self.old/'prepared/LABELS.npz'))
        with self.assertRaisesRegex(ValueError,'All13'):self.check()
    def test_rehashed_wrong_plan_link_still_rejected(self):
        self.pc['plan_sha256']='a'*64;(self.old/'prepared/COMPLETION.json').write_text(json.dumps(self.pc));self.refresh()
        with self.assertRaisesRegex(ValueError,'PLAN links'):self.check()
    def test_rehashed_missing_manifest_member_rejected(self):
        self.pc['artifacts_sha256'].pop('MODEL_AUDIT.json');(self.old/'prepared/COMPLETION.json').write_text(json.dumps(self.pc));self.refresh()
        with self.assertRaisesRegex(ValueError,'full preparation'):self.check()
    def test_rehashed_closed_model_proof_link_rejected(self):
        self.c['model_audit_sha256']='0'*64;(self.old/'COMPLETION.json').write_text(json.dumps(self.c));self.refresh()
        with self.assertRaisesRegex(ValueError,'model and independent proof'):self.check()


if __name__=='__main__':unittest.main(verbosity=2)
