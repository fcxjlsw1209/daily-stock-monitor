"""Synthetic223 funded auxiliary, causal clocks, lifecycle and selection tests."""
from pathlib import Path
from types import SimpleNamespace
import ast, copy, importlib.util, tempfile, unittest
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd
spec=importlib.util.spec_from_file_location('tested223host',Path(__file__).with_name('monthly_risk_reliability223_host.py'))
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
POLICY=dict(budget=.05,ceiling=5,gap=.02)

def rebuild(v):
    mask=v['available']
    for raw,rank in [('raw_candidate','candidate'),('raw_control','control')]:
        v[raw][~mask]=np.nan
        v[rank]=np.full(mask.shape,np.nan)
        for i in range(len(mask)):
            ids=np.flatnonzero(mask[i]);row=v[raw][i,ids]
            for j in ids:
                x=v[raw][i,j];v[rank][i,j]=(np.count_nonzero(row<x)+(np.count_nonzero(row==x)+1)/2)/len(ids)-.5
    return v

def fixture(weight_shape='column'):
    shape=(6,5);mask=np.ones(shape,bool);mask[2,:4]=False;mask[3]=False;mask[4,4]=False
    v=dict(available=mask,raw_candidate=np.tile([-.9,-.8,-.7,-.6,-.5],(6,1)),raw_control=np.tile([-.5,-.9,-.6,-.7,-.8],(6,1)))
    v['raw_candidate'][5]=v['raw_control'][5]=-.25;rebuild(v)
    calm=np.array([False,False,False,True,True,False]);w=np.where(calm,.5,.52)
    core=w[:,None]*np.tile([.4,.35,.3,.28,.26],(6,1));support=np.tile([.1,.08,.05,.03,.01],(6,1))
    a=dict(baseline=core+(1-w[:,None])*support,core=core,core_weight=w[:,None],eligible=np.ones(shape,bool),effects=np.zeros(shape),state_calm10=calm,state_weak_narrow=np.zeros(6,bool))
    if weight_shape=='vector':a['core_weight']=w.copy()
    if weight_shape=='full':a['core_weight']=np.broadcast_to(w[:,None],shape).copy()
    a['eligible'][:,-1]=False
    return a,dict(dates=['synthetic-%d'%i for i in range(6)],symbols=[list('ABCDE')]*6),v

def counts_fixture():
    return dict(checked_cells=5978,available_cells=3,unavailable_cells=5975,year_security_groups=2,supported_groups=1,cold_start_groups=1,historical_month_pairs=120,source_available_pairs=30,available_pairs=20,valid_price_observations=3300,simple_return_observations=3210,second_moment_estimates=90,error_evaluations=40,calibration_estimates=2,ranked_observations=6,new_model_fits=2,predictive_model_fits=2,new_ridge_fits=0,new_tree_fits=0,available_by_event=[3]+[0]*121)

class SyntheticStudy:
    """Real frozen adapter, but deliberately non-global fake episode choices."""
    def __init__(self):
        self.ep=SimpleNamespace(select_boundary=Mock(side_effect=self.boundary))
        self.adapter=SimpleNamespace(score_and_choose=Mock(side_effect=AssertionError('old hand adapter must not be called')))
        self.rp=SimpleNamespace(make_orders=self.orders)
    @staticmethod
    def configs():return [dict(id=i) for i in range(5)]
    @staticmethod
    def modified_pre_score(specification,a,n):
        assert specification==dict(id=4)
        return a['baseline'].copy()+np.array([.011,.012,.013,.014,.015])
    @staticmethod
    def boundary(pre,eligible,symbols,effects,original,policy):
        choices=[]
        for score,good,ss in zip(pre,eligible,symbols):
            ids=sorted(np.flatnonzero(good).tolist(),key=lambda j:(-score[j],ss[j]))
            choices.append(ids[:2]+[ids[-1]] if len(ids)>=3 else ids if len(ids)==2 else [])
        return choices,pre.copy()
    def choices(self,specification,a,m,n,*,policy):
        return self.boundary(self.modified_pre_score(specification,a,n)[:n],a['eligible'][:n],m['symbols'][:n],
            a['effects'][:n],a,policy)
    @staticmethod
    def orders(choices,scores,meta):
        return [dict(date=meta['dates'][i],symbols=[meta['symbols'][i][j] for j in ids],scores=scores[i].tolist())
                for i,ids in enumerate(choices)]

def selection_fixture():
    metrics={n:{p:{e:dict(cagr=.5,sharpe=1.5,max_drawdown=-.3) for e in ('training','early','recent')}
                for p in ('same_close_zero','next_open_cost')} for n in h.NAMES}
    orders={n:[dict(symbols=['A','B','C']) for _ in range(122)] for n in h.NAMES}
    for path in metrics[h.NAMES[1]]:metrics[h.NAMES[1]][path]['training']['cagr']+=.01
    for i in (1,2,53):orders[h.NAMES[1]][i]['symbols']=['A','B','D']
    return metrics,orders

class SelectionTests(unittest.TestCase):
    def test_later34_and_extra_metrics_never_choose_arm(self):
        metrics,orders=selection_fixture();expected=h.select_first88(metrics,orders)
        self.assertEqual(expected['primary'],h.NAMES[1])
        for name in h.NAMES:
            metrics[name]['full122']=dict(cagr=1000. if name!=h.NAMES[1] else -1.)
            for i in range(88,122):orders[name][i]['symbols']=[name]
        self.assertEqual(h.select_first88(metrics,orders),expected)
    def test_total_control_never_promoted_and_must_be_beaten(self):
        for path in ('same_close_zero','next_open_cost'):
            metrics,orders=selection_fixture();metrics[h.NAMES[2]][path]['training']['cagr']=2.
            result=h.select_first88(metrics,orders)
            self.assertEqual(result['primary'],h.NAMES[0]);self.assertFalse(result['admitted'])
    def test_gain_era_risk_and_set_not_order_guards(self):
        failures=[('same_close_zero','training','cagr',.504),('next_open_cost','training','cagr',.49)]
        failures += [(path,era,key,value) for path in ('same_close_zero','next_open_cost') for era,key,value in
            [('early','cagr',.474),('recent','cagr',.474),('training','sharpe',1.449),('training','max_drawdown',-.331)]]
        for path,era,key,value in failures:
            metrics,orders=selection_fixture();metrics[h.NAMES[1]][path][era][key]=value
            with self.subTest(path=path,era=era,key=key):self.assertFalse(h.select_first88(metrics,orders)['admitted'])
        metrics,orders=selection_fixture()
        for i in (1,2,53):orders[h.NAMES[1]][i]['symbols']=['C','B','A']
        for i in (90,100,110):orders[h.NAMES[1]][i]['symbols']=['A','B','D']
        self.assertEqual(h.select_first88(metrics,orders)['changed_months'],[])
    def test_both_eras_and_three_changed_sets_against_control(self):
        metrics,orders=selection_fixture();orders[h.NAMES[2]]=copy.deepcopy(orders[h.NAMES[1]])
        self.assertFalse(h.select_first88(metrics,orders)['admitted'])
        for i in (1,2,3):orders[h.NAMES[2]][i]['symbols']=['A','E','F']
        self.assertFalse(h.select_first88(metrics,orders)['guards']['changed_support_vs_control'])
        orders[h.NAMES[2]][53]['symbols']=['A','E','F']
        self.assertTrue(h.select_first88(metrics,orders)['admitted'])

class ImmutableContractTests(unittest.TestCase):
    def test_only_exact_registered_external_runtime_is_allowed(self):
        runtime,digest=next(iter(h.EXTERNAL_RUNTIME.items()))
        with patch.object(h,'sha',return_value=digest),patch.object(Path,'is_file',return_value=True):
            h.verify({runtime:digest})
            for path,value in ((runtime,'wrong'),('/Library/Developer/CommandLineTools/other.py',digest),('/tmp/other.py',digest)):
                with self.subTest(path=path,value=value),self.assertRaises(ValueError):h.verify({path:value})
    def test_review_requires_context_fit_scope_and_nonempty_pins(self):
        files=[h.ROOT/'work'/('synthetic'+str(i)) for i in range(6)]
        receipt=dict(passed=True,unresolved_blockers=[],context_sha256=h.CONTEXT_SHA,
            code_sha256={str(p.relative_to(h.ROOT)):'digest' for p in files},input_sha256={'work/fake':'digest'})
        for key in ('actual_features_computed','actual_model_fits_computed','actual_portfolio_returns_computed','production_files_modified','strategy_registered_by_reviewer'):receipt[key]=False
        with patch.object(h,'code_files',return_value=files),patch.object(h,'read',return_value=receipt),patch.object(h,'sha',return_value='digest'),patch.object(h,'bind'),patch.object(h,'verify'):
            h.code_review_binding()
            for key,value in (('context_sha256','wrong'),('actual_model_fits_computed',True),('input_sha256',{})):
                old=receipt[key];receipt[key]=value
                with self.subTest(key=key),self.assertRaises(ValueError):h.code_review_binding()
                receipt[key]=old

class LifecycleTests(unittest.TestCase):
    def test_changed_inherited_pin_fails_without_overwriting(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'synthetic-source';p.write_text('source');pins={};h.bind(pins,p);saved=dict(pins)
            h.bind(pins,p,h.sha(p));self.assertEqual(pins,saved);p.write_text('changed')
            with self.assertRaises(ValueError):h.bind(pins,p)
            self.assertEqual(pins,saved)

    def test_unregistered_prepare_never_loads_real_source(self):
        with patch.object(h,'gate',side_effect=ValueError('unregistered')),patch.object(h,'local') as loader:
            with self.assertRaisesRegex(ValueError,'unregistered'):h.prepare('bad')
        loader.assert_not_called()

    def test_independent_code_review_gate_rejects_missing_or_stale(self):
        with patch.object(h,'read',return_value={'passed':False}):
            with self.assertRaises(ValueError):h.code_review_binding()
        with patch.object(h,'read',return_value={'passed':True,'unresolved_blockers':[],'code_sha256':{}}):
            with self.assertRaises(ValueError):h.code_review_binding()

    def test_source_prefix_zero_physically_empty_and_suffix_removed(self):
        prices=pd.DataFrame({'SPY':[1.,2.,'future poison']},index=pd.to_datetime(['2020-01-02','2020-01-03','2020-01-06']))
        panel={'adj':prices};meta={'etfs':[str(i) for i in range(17)],'dates':['2020-01-03','2020-01-06']}
        self.assertIs(h.price_inputs(panel,meta)['adj'],prices)
        self.assertEqual(h.feature_prefix_panel(panel,meta,0)['adj'].shape,(0,1))
        self.assertEqual(h.feature_prefix_panel(panel,meta,1)['adj'].iloc[-1,0],2.)
        self.assertEqual(len(h.feature_prefix_panel(panel,meta,1)['adj']),2)
        with self.assertRaises(ValueError):h.price_inputs({},meta)

    def test_outcomes_require_all_three_arms_nine_paths_fortyfive_slices(self):
        slices={'full122':{},'first88':{},'early53':{},'recent35':{},'later34':{}}
        check=dict(passed=True,independent_accounts=True,independent_of_production_model_and_engine=True,
            checked_paths=9,checked_slices=45,terminal_liquidation=False,
            accounting_source_sha256=h.sha(h.ROOT/'work/ptr169_results_independent.py'),
            paths={arm:{p[0]:dict(passed=True,monthly_executions=122,slices=copy.deepcopy(slices)) for p in h.ev.PATHS} for arm in h.NAMES})
        h.require_outcome_audit(check)
        for mutation in ('arm','path','slice','terminal'):
            bad=copy.deepcopy(check)
            if mutation=='arm':bad['paths'].pop(h.NAMES[2])
            if mutation=='path':bad['paths'][h.NAMES[1]].pop(h.ev.PATHS[2][0])
            if mutation=='slice':bad['paths'][h.NAMES[1]][h.ev.PATHS[0][0]]['slices'].pop('later34')
            if mutation=='terminal':bad['terminal_liquidation']=True
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):h.require_outcome_audit(bad)

    def test_exact38_result_manifest_missing_extra_and_completion_rejected(self):
        manifest={name:'synthetic-digest' for name in h.RESULT_FILES}
        self.assertEqual(len(manifest),38);self.assertNotIn('COMPLETION.json',manifest)
        h.require_result_manifest(manifest)
        for missing in h.RESULT_FILES:
            bad=dict(manifest);bad.pop(missing)
            with self.subTest(missing=missing),self.assertRaises(ValueError):h.require_result_manifest(bad)
        for extra in ('COMPLETION.json','EXTRA.json'):
            bad=dict(manifest);bad[extra]='synthetic'
            with self.subTest(extra=extra),self.assertRaises(ValueError):h.require_result_manifest(bad)

    def test_mutable_central_rejected_from_review_pins(self):
        review=dict(passed=True,unresolved_blockers=[],context_sha256=h.CONTEXT_SHA,actual_model_fits_computed=False,code_sha256={str(p.relative_to(h.ROOT)):'fake' for p in h.code_files()[:6]},
            input_sha256={str(h.CENTRAL):'fake'},actual_features_computed=False,actual_portfolio_returns_computed=False,
            production_files_modified=False,strategy_registered_by_reviewer=False)
        with patch.object(h,'read',return_value=review),patch.object(h,'sha',return_value='fake'):
            with self.assertRaisesRegex(ValueError,'Mutable central'):h.code_review_binding()

    def test_write_once_and_protection_before_publishing_and_central_cas(self):
        code=Path(h.__file__).read_text();tree=ast.parse(code)
        parts={x.name:ast.get_source_segment(code,x) for x in tree.body if isinstance(x,ast.FunctionDef)}
        register=parts['register'];self.assertLess(register.index('code_review_binding()'),register.index('OUT.mkdir()'))
        self.assertLess(register.index('len(protected) == 21444'),register.index('OUT.mkdir()'))
        self.assertIn('Code changed since tests',register)
        self.assertLess(register.index('Mutable central cannot enter registered input pins'),register.index('OUT.mkdir()'))
        closer=parts['close']
        self.assertLess(closer.index('require_result_manifest('),closer.index('require_outcome_audit('))
        self.assertEqual(parts['commit_state'].count('CENTRAL.read_bytes() == before'),2)
        self.assertIn("not (OUT / 'prepared').exists()",parts['prepare'])
        self.assertIn("not (OUT / 'selected').exists()",parts['select'])
        self.assertIn('Selection escaped first88 cutoff',parts['select'])


    def test_source_context_and_prior_bindings(self):
        raw=h.read(h.LOCAL/'PLAN.json')['raw_source_hashes'];binding=h.price_source_binding(raw)
        self.assertTrue(binding['source_admission_metadata_only']);self.assertFalse(binding['strict_pit'])
        c=h.contract();self.assertEqual(c['budget'],.01);self.assertEqual(c['context']['formation']['history_months'],60)
        self.assertEqual(c['context']['formation']['min_months'],18);self.assertEqual(c['context']['formation']['forecast_returns'],63)
        self.assertEqual(c['context']['arms'],list(h.NAMES));self.assertIn(h.CONTEXT,h.code_files())
        state={'completed':[{'completion':{'path':str(h.PRIOR/'COMPLETION.json'),'sha256':h.PRIOR_EXPECTED['COMPLETION.json']}}]}
        self.assertEqual(h.predecessor_binding(state)['directory'],'monthly_idio_risk222')
        rows=[dict(trials=1,directory='old') for _ in range(220)];rows[0]['trials']=958;rows[-1]['directory']=h.PRIOR.name
        synthetic=dict(completed=rows,best_training_selected_candidate=dict(directory='monthly_local_linear70'))
        h.state_check(synthetic);synthetic['completed'][-1]['directory']='bad'
        with self.assertRaises(ValueError):h.state_check(synthetic)

    def test_source_scope_cannot_claim_actual_values(self):
        raw=h.read(h.LOCAL/'PLAN.json')['raw_source_hashes'];review=h.read(h.SOURCE_REVIEW)
        for k in ['npz_decoded','actual_source_coverage_recomputed','actual_new_features_computed','actual_new_orders_computed','actual_new_returns_computed']:
            bad=copy.deepcopy(review);bad[k]=True
            with patch.object(h,'read',return_value=bad),self.assertRaises(ValueError):h.price_source_binding(raw)
        for k in ['annual_cutoff_calendar_verified','canonical_identity_scope_reviewed','metadata_axes_verified']:
            bad=copy.deepcopy(review);bad[k]=False
            with patch.object(h,'read',return_value=bad),self.assertRaises(ValueError):h.price_source_binding(raw)

class WiringTests(unittest.TestCase):
    def test_funded_allstates_ineligible_and_partial(self):
        a,m,v=fixture();t,d=h.scale_and_delta(v,a,6);support=(a['baseline']-a['core'])/(1-a['core_weight'])
        for arm,key in h.RAW.items():
            e=v[key].copy();e[0]=np.nan;np.testing.assert_array_equal(t[arm],e)
            expected=np.zeros(e.shape);good=np.isfinite(e);expected[good]=.01*(e[good]-support[good])
            np.testing.assert_array_equal(d[arm],expected);self.assertTrue(np.isnan(t[arm][0]).all());self.assertEqual(np.count_nonzero(d[arm][0]),0)
        self.assertNotEqual(d[h.NAMES[1]][1,-1],0);self.assertFalse(a['eligible'][1,-1])
        self.assertTrue(a['state_calm10'][4]);self.assertNotEqual(d[h.NAMES[1]][4,0],0)

    def test_legal_weight_shapes_and_exact_ties(self):
        expected=None
        for kind in ['vector','column','full']:
            a,m,v=fixture(kind);actual=h.scale_and_delta(v,a,6)
            if expected is None:expected=actual
            for j in [0,1]:
                for arm in h.NAMES[1:]:np.testing.assert_array_equal(expected[j][arm],actual[j][arm])
        self.assertEqual(actual[0][h.NAMES[1]][2,4],.5)
        np.testing.assert_array_equal(actual[0][h.NAMES[1]][5],np.full(5,3/5-.5))
        self.assertTrue(np.isnan(actual[0][h.NAMES[1]][3]).all())

    def test_zero_rank_funds_support_missing_does_not(self):
        a,m,v=fixture();v['available'][1,4]=False;rebuild(v);t,d=h.scale_and_delta(v,a,6)
        self.assertEqual(t[h.NAMES[1]][1,1],0);support=(a['baseline']-a['core'])/(1-a['core_weight'])
        self.assertEqual(d[h.NAMES[1]][1,1],-.01*support[1,1]);self.assertEqual(d[h.NAMES[1]][1,4],0)

    def test_corrupt_raw_rank_mask_weights_support_rejected(self):
        for kind in ['rawabove','rawbelow','rank','rawnan','rawinf','rawmissing','mask','shape','extra','weight','calm','support']:
            a,m,v=fixture('full')
            if kind=='rawabove':v['raw_candidate'][1,0]=.01
            if kind=='rawbelow':v['raw_control'][1,0]=-1.1
            if kind=='rank':v['candidate'][1,0]=0
            if kind=='rawnan':v['raw_candidate'][1,0]=np.nan
            if kind=='rawinf':v['raw_candidate'][1,0]=np.inf
            if kind=='rawmissing':v['raw_candidate'][3,0]=0
            if kind=='mask':v['available']=v['available'].astype(int)
            if kind=='shape':v['control']=v['control'][:,:-1]
            if kind=='extra':v['extra']=np.zeros((6,5))
            if kind=='weight':a['core_weight'][1,-1]=.5
            if kind=='calm':a['state_calm10']=a['state_calm10'].astype(int)
            if kind=='support':a['baseline'][1,0]=3
            with self.subTest(kind=kind),self.assertRaises((ValueError,AssertionError)):h.scale_and_delta(v,a,6)

    def test_future_numeric_values_not_read_and_n_validation(self):
        a,m,v=fixture();full=h.scale_and_delta(v,a,6)
        for n in range(7):
            aa,vv=copy.deepcopy(a),copy.deepcopy(v)
            for k in vv:
                if k!='available':vv[k]=vv[k].astype(object);vv[k][n:]='future poison'
            for k in ['baseline','core','core_weight']:aa[k]=aa[k].astype(object);aa[k][n:]='future poison'
            actual=h.scale_and_delta(vv,aa,n)
            for j in [0,1]:
                for arm in h.NAMES[1:]:np.testing.assert_array_equal(actual[j][arm],full[j][arm][:n])
        for n in [-1,7,True,1.5,np.int64(2)]:
            with self.assertRaises(ValueError):h.scale_and_delta(v,a,n)

    def test_real_adapter_single_episode_originalsupport_immutable(self):
        a,m,v=fixture();before=copy.deepcopy(a);vv=copy.deepcopy(v);study=SyntheticStudy();adapter=h.path_adapter();pre=study.modified_pre_score(dict(id=4),a,6)
        with patch.object(h,'read',return_value=POLICY),patch.object(h,'path_adapter',return_value=adapter):orders,t,d=h.decisions(study,a,m,v,6)
        self.assertEqual(study.ep.select_boundary.call_count,2);study.adapter.score_and_choose.assert_not_called()
        for arm,call in zip(h.NAMES[1:],study.ep.select_boundary.call_args_list):
            actual,eligible,symbols,effects,original,policy=call.args
            expected=pre.copy();good=d[arm]!=0;expected[good]+=d[arm][good];np.testing.assert_array_equal(actual,expected)
            for k in ['baseline','core','state_calm10','state_weak_narrow']:np.testing.assert_array_equal(original[k],a[k])
            np.testing.assert_array_equal((original['baseline']-original['core'])/(1-original['core_weight']),(a['baseline']-a['core'])/(1-a['core_weight']))
            choice,score=study.boundary(actual,eligible,symbols,effects,original,policy);self.assertEqual(orders[arm],study.orders(choice,score,m))
            self.assertEqual(orders[arm][0],orders[h.NAMES[0]][0])
        for k in a:np.testing.assert_array_equal(a[k],before[k])
        for k in v:np.testing.assert_array_equal(v[k],vv[k])

    def test_every_orderprefix_and_missing_fallback(self):
        a,m,v=fixture()
        with patch.object(h,'read',return_value=POLICY):
            full,tt,dd=h.decisions(SyntheticStudy(),a,m,v,6)
            for n in range(7):
                out,t,d=h.decisions(SyntheticStudy(),a,m,v,n)
                for arm in h.NAMES:self.assertEqual(out[arm],full[arm][:n])
                for arm in h.NAMES[1:]:np.testing.assert_array_equal(t[arm],tt[arm][:n]);np.testing.assert_array_equal(d[arm],dd[arm][:n])
            v['available'][:]=False;rebuild(v);out,_,_=h.decisions(SyntheticStudy(),a,m,v,6)
            self.assertTrue(all(out[k]==out[h.NAMES[0]] for k in h.NAMES))

    def test_cross_producer_oracle_interface_and_source_groups(self):
        import monthly_risk_reliability223_kernel as kernel
        import monthly_risk_reliability223_independent as independent
        from test_monthly_risk_reliability223_kernel import fixture as pf
        panel,meta,calendar=pf();values,audit=kernel.build_features(panel,meta,calendar)
        check=independent.verify_features(panel,meta,calendar,values,audit);self.assertTrue(check['passed']);self.assertEqual(h.FEATURE_FLAGS,independent.FEATURE_FLAGS);h.require_source_support(audit)
        rows,width=values['available'].shape;calm=np.arange(rows)%2==1
        a=dict(baseline=np.zeros((rows,width)),core=np.zeros((rows,width)),core_weight=np.where(calm,.5,.52)[:,None],state_calm10=calm)
        for n in range(rows+1):
            t,d=h.scale_and_delta(values,a,n);receipt=independent.verify_deltas({k:v[:n] for k,v in values.items()},a,t,d,budget=.01);h.require_delta_audit(receipt,d)
        for key in ['cutoff','history_start_month','history_end_month']:
            bad=copy.deepcopy(audit);bad['group_history'][0][key]='2099-01-01'
            with self.assertRaises(ValueError):h.require_source_support(bad)
        bad=copy.deepcopy(audit);bad['event_history'][0]['symbols'][0]='WRONG'
        with self.assertRaises(ValueError):h.require_source_support(bad)

class AccountingTests(unittest.TestCase):
    def test_calibration_counts_not_zero_model_fits(self):
        c=counts_fixture();counts=h.estimate_counts(c);self.assertEqual(counts['calibration_estimates'],2)
        for key in h.COUNT_FIELDS+('new_model_fits','predictive_model_fits','new_ridge_fits','new_tree_fits'):
            bad=copy.deepcopy(c);bad[key]+=1
            with self.subTest(key=key),self.assertRaises(ValueError):h.estimate_counts(bad)
        for series in [[3]*121,[50]+[0]*121,[True]+[0]*121]:
            bad=copy.deepcopy(c);bad['available_by_event']=series
            with self.assertRaises(ValueError):h.estimate_counts(bad)

    def test_prepared_receipt_requires_all_counts_and_proofs(self):
        c=counts_fixture();mask=np.zeros((122,49),bool);mask[0,:3]=True
        feature=dict(c,**{k:True for k in h.FEATURE_FLAGS});feature.update(events=122,symbols_per_event=49,computes_returns=False,producer_modules_imported=False,numerical_tolerance=0.)
        deltaflags=['passed','funded_aux_formula_checked','original_pct_ranks_preserved','first_partial_unchanged','all_states_preserved','missing_preserves_score','core_weights_checked','no_second_rank','independent_delta_rebuild']
        deltas={arm:np.zeros((122,49)) for arm in h.NAMES[1:]};delta=dict({k:True for k in deltaflags},budget=.01,max_absolute_delta=h.MAX_DELTA,changed_cells={k:0 for k in deltas})
        base={k:c[k] for k in h.ESTIMATE_FIELDS};base.update(new_model_fits=2,predictive_model_fits=2,new_ridge_fits=0,new_tree_fits=0,original_array_count=27,model_fit_accounting=h.MODEL_FIT_ACCOUNTING)
        audit=dict(base,feature=feature,delta=delta,prefixes=123,feature_prefixes=list(h.FEATURE_PREFIXES))
        flags=['feature_prefix_sources_truncated','feature_prefix_metadata_truncated','feature_prefix_independently_verified','source_admission_metadata_only','source_support_exact','original_orders_exact','original_arrays_unchanged','original_core_metadata_and_support_preserved','single_original_episode']
        audit.update({k:True for k in flags});receipt=dict(base,new_returns_computed=False,artifacts_sha256={k:'synthetic' for k in h.PREPARED_FILES})
        h.verify_prepared_evidence(receipt,{'available':mask},deltas,audit)
        for k in flags:
            bad=copy.deepcopy(audit);bad[k]=False
            with self.subTest(k=k),self.assertRaises(ValueError):h.verify_prepared_evidence(receipt,{'available':mask},deltas,bad)
        for k in h.ESTIMATE_FIELDS+('new_model_fits','predictive_model_fits','new_ridge_fits','new_tree_fits'):
            bad=copy.deepcopy(receipt);bad[k]=None
            with self.subTest(k=k),self.assertRaises(ValueError):h.verify_prepared_evidence(bad,{'available':mask},deltas,audit)
        for k in h.FEATURE_FLAGS:
            bad=dict(feature);bad.pop(k)
            with self.assertRaises(ValueError):h.require_feature_audit(bad,{'available':mask})
        for k in deltaflags:
            bad=dict(delta);bad.pop(k)
            with self.assertRaises(ValueError):h.require_delta_audit(bad,deltas)

if __name__=='__main__':unittest.main()
