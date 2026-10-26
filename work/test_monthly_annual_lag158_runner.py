"""Synthetic runner tests only: no actual source dates, prices or phase runs."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch
import json
import tempfile
import unittest

import numpy as np
import pandas as pd
import monthly_annual_lag158_runner as r
from test_monthly_annual_lag158_model import fixture

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = r.base.module('annual_lag158_synthetic_adapter', ROOT/'outputs/monthly_path70_research/incumbent_adapter.py')
POLICY = dict(budget=.05, ceiling=5, gap=.02)


def arrays(n=4):
    shape = (n,49); baseline = np.tile(np.linspace(.2,.1,49), (n,1))
    raw = np.full(shape,np.nan); raw[:,:2] = [-10.,-20.]
    ranks = np.zeros(shape); ranks[:,:2] = [.5,-.5]
    mask = np.zeros(shape,bool); mask[:,:2] = True
    a = dict(baseline=baseline, core=.5*baseline, core_weight=np.full(n,.5),
        eligible=np.ones(shape,bool), effects=np.zeros(shape), state_calm10=np.ones(n,bool),
        state_weak_narrow=np.zeros(n,bool), negative_current_lag=raw.copy(), negative_lag_change=-raw.copy(),
        current_lag_rank=ranks.copy(), lag_change_rank=-ranks.copy(), common_available=mask)
    names = ['KO','WMT']+['S'+str(i) for i in range(30)]+['ETF'+str(i) for i in range(17)]
    dates = [str(day.date()) for day in pd.date_range('2010-01-31',periods=n,freq='ME')]
    meta = dict(dates=dates,ends=dates, symbols=[names[:] for _ in dates],etfs=names[-17:],train_months=88)
    return a,meta


def neutral(a, i):
    a['common_available'][i] = False
    for key in r.model.RAW_NAMES: a[key][i] = np.nan
    for key in r.model.RANK_NAMES: a[key][i] = 0.


class FakeBackend:
    adapter = ADAPTER
    def __init__(self): self.calls = 0
    def pre(self,a,n):
        pre = np.zeros_like(a['baseline']);pre[:n] = a['baseline'][:n];pre[:n,3] += .2
        return pre
    def selector(self,scores,good,symbols,effects,original,policy):
        self.calls += 1
        chosen = [sorted(np.flatnonzero(ok),key=lambda j:(-sc[j],names[j]))[:3]
                  for sc,ok,names in zip(scores,good,symbols)]
        return chosen,scores.copy()
    def baseline(self,a,m,n,policy):
        return self.adapter.score_and_choose('incumbent66',0.,np.full(a['baseline'].shape,np.nan),a,m,n,
            pre_score=self.pre(a,n),effects=a['effects'],episode_policy=policy,boundary_selector=self.selector)
    @staticmethod
    def make_orders(chosen,scores,meta):
        return [dict(date=meta['dates'][i],symbols=[meta['symbols'][i][j] for j in row],
                     scores=[float(scores[i,j]) for j in row]) for i,row in enumerate(chosen)]


def trials():
    return [dict(name=name,training=dict(cagr=.2,sharpe=1.,max_drawdown=-.2),early=dict(cagr=.2),recent=dict(cagr=.2))
            for name in r.NAMES]


def proposal():
    return dict(round=158,arms=list(r.NAMES),primary_challenger=r.NAMES[2],
        status='FINAL_PROSPECTIVE_SPECIFICATION_BEFORE_REGISTRATION',measurement=r.measurement_contract(),
        selection_contract=r.selection_contract(),design=dict(path=r.DESIGN_PATH,sha256=r.DESIGN_SHA),
        previous_completion=dict(path=r.PREVIOUS_COMPLETION,sha256=r.PREVIOUS_COMPLETION_SHA),
        actual_new_lag_values=0,actual_new_features=0,actual_new_model_fits=0,actual_new_orders=0,actual_new_returns=0)


class Scoring(unittest.TestCase):
    def setUp(self): self.a,self.m=arrays();self.backend=FakeBackend()
    def test_same_mask_rank_and_exact_dose_before_single_selector(self):
        _,scores,_=r.choices_all(self.a,self.m,4,self.backend,POLICY)
        self.assertEqual(self.backend.calls,3)
        for name,key in zip(r.NAMES[1:],r.model.RANK_NAMES):
            np.testing.assert_array_equal(scores[name],self.backend.pre(self.a,4)+.01*self.a[key])
            np.testing.assert_array_equal(scores[name][:,2:],scores[r.NAMES[0]][:,2:])
    def test_zero_tie_and_singleton_exact_fallback(self):
        for case in ('empty','tie','single'):
            a=deepcopy(self.a)
            if case=='empty': neutral(a,0)
            elif case=='tie':
                for key in r.model.RAW_NAMES:a[key][0,:2]=5.
                for key in r.model.RANK_NAMES:a[key][0]=0.
            else:
                a['common_available'][0,1]=False
                for key in r.model.RAW_NAMES:a[key][0,1]=np.nan
                for key in r.model.RANK_NAMES:a[key][0]=0.
            chosen,scores,_=r.choices_all(a,self.m,1,self.backend,POLICY)
            for name in r.NAMES[1:]:
                self.assertEqual(chosen[name],chosen[r.NAMES[0]])
                self.assertEqual(scores[name].tobytes(),scores[r.NAMES[0]].tobytes())
    def test_initial_partial_neutral_and_rejects_positive_mask(self):
        self.m['dates'][0]='2016-08-17'
        with self.assertRaises(ValueError):r.choices_all(self.a,self.m,1,self.backend,POLICY)
        neutral(self.a,0)
        _,scores,audit=r.choices_all(self.a,self.m,1,self.backend,POLICY)
        self.assertTrue(audit[0]['initial_partial_fallback'])
        self.assertEqual(scores[r.NAMES[0]].tobytes(),scores[r.NAMES[2]].tobytes())
    def test_disagreed_masks_ineligible_finite_and_wrong_rank_rejected(self):
        mutations=[('negative_lag_change',(0,0),np.nan),('eligible',(0,0),False),
                   ('current_lag_rank',(0,0),.5001),('negative_current_lag',(0,0),np.inf),
                   ('common_available',(0,0),False),('current_lag_rank',(0,2),.1)]
        for key,index,value in mutations:
            a=deepcopy(self.a);a[key][index]=value
            with self.subTest(key=key),self.assertRaises(ValueError):r.ranked_pair(a,1)
    def test_no_unlisted_issuer_or_third_source_candidate(self):
        self.m['symbols'][0][0]='UNKNOWN'
        with self.assertRaises(ValueError):r.choices_all(self.a,self.m,1,self.backend,POLICY)
        self.a['common_available'][0,2]=True
        with self.assertRaises(ValueError):r.ranked_pair(self.a,1)
    def test_future_content_ignored_and_invalid_prefixes_rejected(self):
        expected=r.choices_all(self.a,self.m,1,self.backend,POLICY)[1]
        for key in r.model.RAW_NAMES+r.model.RANK_NAMES:
            self.a[key]=self.a[key].astype(object);self.a[key][1:]='future poison'
        actual=r.choices_all(self.a,self.m,1,self.backend,POLICY)[1]
        for name in r.NAMES:self.assertEqual(actual[name].tobytes(),expected[name].tobytes())
        for n in (-1,5,True,1.5):
            with self.assertRaises(ValueError):r.ranked_pair(self.a,n)
    def test_all123_prefixes_366_orders_no_array_mutation(self):
        a,m=arrays(122);before={k:v.tobytes() for k,v in a.items()}
        orders,scores,audit=r.panels(a,m,FakeBackend(),POLICY)
        self.assertEqual(sum(map(len,orders.values())),366);self.assertEqual(len(audit),122)
        self.assertEqual(set(scores),set(r.NAMES));self.assertEqual(before,{k:v.tobytes() for k,v in a.items()})
    def test_unchanged_holdings_report_no_incremental_evidence(self):
        orders={name:[dict(symbols=['S','KO','WMT'])] for name in r.NAMES}
        audit=[dict(unchanged_current_correction=False,unchanged_change_correction=False)]
        result=r.incremental_audit(orders,audit)
        self.assertTrue(result['primary_no_incremental_evidence'])
        orders[r.NAMES[2]][0]['symbols']=['S','KO','NEW']
        self.assertFalse(r.incremental_audit(orders,audit)['primary_no_incremental_evidence'])


class SourcePreflight(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.original=fixture()
    def setUp(self):
        self.data=deepcopy(self.original);d=self.data
        d['meta'].update(ends=list(d['meta']['dates']),train_months=88)
        self.overlay=dict(schema='after157_ko_wmt_conditional_annual_source_v1',rows=d['rows'],
            source_admission_changed=False,old_admission_flags_changed=False,full_strategy_admission=False,
            numerical_experiment_registered=False,lag_or_feature_values_computed=False)
        self.prior=dict(rows=[deepcopy(row['original_row']) for row in d['rows']],
                        selected_original_and_amendment_metadata=d['event_metadata'])
    def check(self):return r.validate_source_rows(self.data['meta'],self.overlay,self.prior,self.data['calendar'])
    def test_full_grid_metadata_preflight_without_model_build(self):
        with patch.object(r.model,'build',side_effect=AssertionError('not metadata-only')):
            mask=self.check()
        np.testing.assert_array_equal(mask,self.data['source_mask'])
    def test_changed_original_or_missing_grid_row_rejected(self):
        self.overlay['rows'][0]['original_row']['tier3']=True
        with self.assertRaises(ValueError):self.check()
        self.overlay['rows'].pop()
        with self.assertRaises(ValueError):self.check()
    def test_bad_latest_even_originally_ineligible_is_hard_failure(self):
        self.data['eligible'][0,0]=False
        event=self.data['rows'][0]['original_row']['selected_event_ids'][0]
        self.data['event_metadata'][event]['filingDate']='not a date'
        with self.assertRaises(ValueError):self.check()
    def test_admitted_source_cannot_clear_another_blocker(self):
        for row in (self.overlay['rows'][0]['original_row'],self.prior['rows'][0]):row['reasons'].append('TIMING_PENDING')
        with self.assertRaisesRegex(ValueError,'retained gates'):self.check()
    def test_second_session_cutoff_and_wrong_cik_hard_fail(self):
        event=self.data['rows'][0]['original_row']['selected_event_ids'][0]
        self.data['event_metadata'][event]['filingDate']=self.data['rows'][0]['original_row']['information_cutoff_date']
        with self.assertRaisesRegex(ValueError,'availability'):self.check()
        self.data['event_metadata'][event]['cik']='other'
        with self.assertRaisesRegex(ValueError,'identity'):self.check()
    def test_blocked_source_dates_not_promoted_or_substituted(self):
        self.overlay['rows'][0]['conditional_annual_pair_admitted']=False
        event=self.data['rows'][0]['original_row']['selected_event_ids'][0]
        self.data['event_metadata'][event]['filingDate']='not a date'
        self.assertFalse(self.check()[0,0])
    def test_wrong_cutoff_even_nonpilot_row_fails(self):
        for row in (self.overlay['rows'][2]['original_row'],self.prior['rows'][2]):row['information_cutoff_ns']+=1
        with self.assertRaisesRegex(ValueError,'cutoff'):self.check()
    def test_real_pure_model_synthetic_build_prefix_and_coverage(self):
        result=r.build_features(**{k:self.data[k] for k in ('meta','rows','eligible','calendar')},
            events=self.data['event_metadata'],source_mask=self.data['source_mask'])
        self.assertEqual(result['audit']['feature_prefixes_checked'],123)
        self.assertEqual(result['audit']['coverage']['all122']['usable_rows'],244)
        self.assertEqual(result['audit']['coverage']['first88']['candidate_cells'],4312)
        self.assertEqual(result['audit']['coverage']['later34']['months_with_two_usable'],34)
    def test_model_cannot_silently_mask_admitted_contradiction(self):
        event=self.data['rows'][0]['original_row']['selected_event_ids'][0]
        self.data['event_metadata'][event]['form']='10-KT'
        with self.assertRaisesRegex(ValueError,'cannot become missing'):
            r.build_features(**{k:self.data[k] for k in ('meta','rows','eligible','calendar')},
                events=self.data['event_metadata'],source_mask=self.data['source_mask'])


class SelectionAndContracts(unittest.TestCase):
    def test_tie_and_current_control_cannot_be_promoted(self):
        self.assertEqual(r.select(trials(),trials())['primary'],r.NAMES[0])
        a,b=trials(),trials();a[1]['training']['cagr']=b[1]['training']['cagr']=.8
        self.assertEqual(r.select(a,b)['primary'],r.NAMES[0])
    def test_both_controls_cost_and_all_guards_required(self):
        a,b=trials(),trials();a[2]['training']['cagr']=b[2]['training']['cagr']=.3
        self.assertTrue(r.select(a,b)['admitted'])
        for section,key,value in [('training','cagr',.2),('training','sharpe',.949),
                                 ('training','max_drawdown',-.231),('early','cagr',.174),('recent','cagr',.174)]:
            bad=deepcopy(b);bad[2][section][key]=value
            self.assertFalse(r.select(a,bad)['admitted'])
    def test_cost_incumbent_only_noninferiority(self):
        a,b=trials(),trials();a[2]['training']['cagr']=.3;b[1]['training']['cagr']=.1
        self.assertTrue(r.select(a,b)['admitted'])
    def test_nonfinite_bool_metrics_and_wrong_arm_order_rejected(self):
        for value in (np.nan,np.inf,True,'0.3'):
            a=trials();a[2]['training']['cagr']=value
            self.assertFalse(r.select(a,trials())['admitted'])
        with self.assertRaises(ValueError):r.select(trials()[::-1],trials())
    def test_exact_proposal_and_no_fold_contract(self):
        r.validate_proposal(proposal());self.assertFalse(r.selection_contract()['prefix_reselection'])
        self.assertEqual(r.selection_contract()['selection_count'],1)
        self.assertNotIn('WALKFORWARD.json',r.FROZEN_FILES)
        for key in ('actual_new_lag_values','actual_new_features','actual_new_model_fits','actual_new_orders','actual_new_returns'):
            for val in (1,True):
                p=proposal();p[key]=val
                with self.assertRaises(ValueError):r.validate_proposal(p)
        p=proposal();p['measurement']['dose']=.02
        with self.assertRaises(ValueError):r.validate_proposal(p)


class ReviewAndPhases(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve();self.out=self.root/'out';self.out.mkdir()
        self.runner=object.__new__(r.Runner);self.runner.root=self.root;self.runner.out=self.out
        self.code=self.root/'review.py';self.code.write_text('# synthetic reviewer\n')
        self.runner.pins={str(self.code):r.sha(self.code)}
        self.runner.plan={'independent_prepare_source_hashes':dict(self.runner.pins)}
        self.hashes={}
        for name in (*r.CACHE_FILES,'PLAN.json','CACHE.json'):
            (self.out/name).write_text('{}');self.hashes[str(self.out/name)]=r.sha(self.out/name)
        self.runner.plan_path=self.out/'PLAN.json'
        self.runner.plan_sha=self.hashes[str(self.out/'PLAN.json')]
        self.review=dict(passed=True,stage='prepare',producer_modules_imported=False,returns_or_nav_computed=False,
            source_hashes=dict(self.runner.pins),artifact_hashes=dict(self.hashes),
            plan_sha256=self.runner.plan_sha,cache_sha256=self.hashes[str(self.out/'CACHE.json')],
            orders_sha256=self.hashes[str(self.out/'PRETRAIN_ORDERS.json')],orders_checked=366,candidate_cells_checked=5978,
            months_checked=122,prefixes_checked=123,lag_feature_matrices_verified=2,new_predictive_model_fits=0,
            nuisance_estimates=0,**{k:True for k in r.CHECKS})
    def check(self):
        (self.out/'INDEPENDENT_PREPARE.json').write_text(json.dumps(self.review))
        with patch.object(self.runner,'cache',return_value={k:v for k,v in self.hashes.items() if Path(k).name in r.CACHE_FILES}):
            self.runner.require_prepare_review()
    def test_complete_receipt(self):self.check()
    def test_every_check_and_counter_mandatory(self):
        for key in r.CHECKS:
            self.review[key]=False
            with self.assertRaises(ValueError):self.check()
            self.review[key]=True
        self.review['orders_checked']=365
        with self.assertRaises(ValueError):self.check()
    def test_unbound_artifact_reviewer_or_wrong_receipt_fail(self):
        for key,value in (('producer_modules_imported',True),('plan_sha256','f'*64),('artifact_hashes',{})):
            old=self.review[key];self.review[key]=value
            with self.assertRaises(ValueError):self.check()
            self.review[key]=old
        self.code.write_text('# changed')
        with self.assertRaises(ValueError):self.check()
    def test_source_failure_precedes_backend_loading(self):
        for name in (*r.CACHE_FILES,'CACHE.json'):(self.out/name).unlink()
        self.runner.backend_factory=Mock(side_effect=AssertionError('must not load'))
        with patch.object(self.runner,'refresh'),patch.object(self.runner,'require_sources',side_effect=ValueError('source')):
            with self.assertRaisesRegex(ValueError,'source'):self.runner.prepare()
        self.runner.backend_factory.assert_not_called()
    def test_prepare_already_started_rejected(self):
        with patch.object(self.runner,'refresh'):
            with self.assertRaises(ValueError):self.runner.prepare()
    def test_review_gate_precedes_performance_loading(self):
        with patch.object(self.runner,'refresh'),patch.object(self.runner,'require_prepare_review',side_effect=ValueError('gate')),\
                patch.object(self.runner,'loaded') as load:
            with self.assertRaisesRegex(ValueError,'gate'):self.runner.select_first88()
            load.assert_not_called()
    def test_single_first88_choice_no_walkforward_and_no_rerun(self):
        backend=NS(training=Mock(return_value=(trials(),trials())))
        (self.out/'INDEPENDENT_PREPARE.json').write_text('{}')
        with patch.object(self.runner,'refresh'),patch.object(self.runner,'require_prepare_review'),\
                patch.object(self.runner,'loaded',return_value=(backend,{},None,None)),patch.object(r,'select',wraps=r.select) as pick:
            self.runner.select_first88();self.assertEqual(pick.call_count,1)
            with self.assertRaises(ValueError):self.runner.select_first88()
        frozen=json.loads((self.out/'FROZEN.json').read_text())
        self.assertEqual(frozen['selection_count'],1);self.assertFalse((self.out/'WALKFORWARD.json').exists())
    def test_diagnostic_needs_frozen_and_preserves_partial_output(self):
        with patch.object(self.runner,'refresh'),patch.object(self.runner,'require_prepare_review'),patch.object(self.runner,'loaded') as load:
            with self.assertRaises(FileNotFoundError):self.runner.diagnose_later34()
            (self.out/'RESULTS.json').write_text('{}')
            with self.assertRaises(ValueError):self.runner.diagnose_later34()
            load.assert_not_called()
    def test_historical_mutable_pin_only_exact_archive_and_no_live_read(self):
        archive=self.root/r.HISTORICAL_ARCHIVE;archive.parent.mkdir(parents=True);archive.write_text('original')
        digest=r.sha(archive)
        with patch.object(r,'HISTORICAL_SHA',digest):
            got=self.runner.verify({r.HISTORICAL_STATE:digest})
            self.assertEqual(got,{str(archive):digest})
            with self.assertRaises(ValueError):self.runner.verify({r.HISTORICAL_STATE:'0'*64})
    def test_no_path_escape_and_no_unregistered_constructor_loading(self):
        for path in ('../escape','/tmp/outside',str(self.root/'a/../escape')):
            with self.assertRaises(ValueError):self.runner.path(path)
        plan=self.out/'not_registered.json';plan.write_text('{"round":158,"status":"DRAFT"}')
        backend=Mock(side_effect=AssertionError('market data'))
        with self.assertRaises(ValueError):r.Runner(self.root,plan,r.sha(plan),backend)
        backend.assert_not_called()
    def registered_plan(self):
        counters=('actual_feature_calculations_before_registration','actual_returns_before_registration',
                  'actual_label_calculations_before_registration','actual_lag_calculations_before_registration',
                  'actual_model_fits_before_registration','actual_nuisance_estimates_before_registration',
                  'new_market_downloads')
        return dict(round=158,status='REGISTERED',arms=list(r.NAMES),primary_challenger=r.NAMES[2],
                    dose=r.DOSE,measurement=r.measurement_contract(),selection_contract=r.selection_contract(),
                    tolerances=r.TOLERANCES,runtime=dict(python=r.sys.version,numpy=np.__version__,pandas=pd.__version__),
                    no_observer_writes=True,hashes={'synthetic':'0'*64},**dict.fromkeys(counters,0)),counters
    def test_zero_activity_constructor_reaches_pins_without_loading(self):
        plan,_=self.registered_plan();self.runner.plan_path.write_text(json.dumps(plan))
        backend=Mock(side_effect=AssertionError('no actual loading'))
        with patch.object(r.Runner,'verify',side_effect=RuntimeError('synthetic pin boundary')) as verify:
            with self.assertRaisesRegex(RuntimeError,'synthetic pin boundary'):
                r.Runner(self.root,self.runner.plan_path,r.sha(self.runner.plan_path),backend)
        verify.assert_called_once_with(plan['hashes']);backend.assert_not_called()
    def test_every_registration_activity_guard_rejects_missing_or_nonzero_before_pins(self):
        good,counters=self.registered_plan()
        for key in counters:
            for bad in (None,1,True,0.0):
                plan=deepcopy(good)
                if bad is None:del plan[key]
                else:plan[key]=bad
                self.runner.plan_path.write_text(json.dumps(plan))
                with self.subTest(key=key,bad=bad),patch.object(r.Runner,'verify') as verify:
                    with self.assertRaisesRegex(ValueError,'Premature real work'):
                        r.Runner(self.root,self.runner.plan_path,r.sha(self.runner.plan_path))
                    verify.assert_not_called()
    def test_artifact_write_refreshes_once_and_preserves_write_once(self):
        with patch.object(self.runner,'refresh',wraps=self.runner.refresh) as refresh:
            self.runner.write('new.json',{'synthetic':True})
        refresh.assert_called_once_with()
        before=(self.out/'new.json').read_bytes()
        with self.assertRaises(FileExistsError):self.runner.write('new.json',{'changed':True})
        self.assertEqual((self.out/'new.json').read_bytes(),before)
    def test_artifact_write_rejects_changed_plan_or_dependency_before_creating_file(self):
        for path in (self.code,self.runner.plan_path):
            before=path.read_bytes();path.write_text('changed')
            try:
                with self.subTest(path=path.name),self.assertRaises(ValueError):
                    self.runner.write('rejected.json',{})
                self.assertFalse((self.out/'rejected.json').exists())
            finally:path.write_bytes(before)
    def test_changed_pin_during_training_prevents_selection_artifacts(self):
        def train(*args):
            self.code.write_text('# changed during long phase')
            return trials(),trials()
        backend=NS(training=Mock(side_effect=train))
        with patch.object(self.runner,'require_prepare_review'),\
                patch.object(self.runner,'loaded',return_value=(backend,{},None,None)):
            with self.assertRaisesRegex(ValueError,'Changed dependency'):
                self.runner.select_first88()
        backend.training.assert_called_once()
        for name in ('TRAIN_TRIALS.json','TRAIN_OPEN_TRIALS.json','FROZEN.json'):
            self.assertFalse((self.out/name).exists())


class AccountingAdapter(unittest.TestCase):
    def test_training_is_only88_and_never_selects_prefixes(self):
        _,meta=arrays(122);orders={n:[dict(symbols=['KO']) for _ in range(122)] for n in r.NAMES}
        eq=pd.Series(np.arange(4000,dtype=float)+10000,index=pd.date_range('2010-01-01',periods=4000))
        engine=NS(replay=Mock(return_value=({},eq,[],[])))
        rp=NS(paths=Mock(return_value=(None,None,None,None)),fast_returns=Mock(return_value=np.array([.01])),
              stats_rr=Mock(return_value={'cagr':.2}))
        backend=object.__new__(r.Backend);backend.local=NS(rp=rp,old=NS(session=NS(measure=Mock(return_value={}))),
                                                        ex=NS(e=NS(engine=engine)))
        with patch.object(r,'select',side_effect=AssertionError('no choice inside training')):
            closed,opened=backend.training(orders,meta,None,None)
        self.assertEqual([x['name'] for x in closed],list(r.NAMES));self.assertEqual(len(opened),3)
        self.assertEqual(rp.paths.call_args.args[2],88);self.assertEqual(engine.replay.call_count,3)
        for call in engine.replay.call_args_list:
            self.assertEqual(len(call.args[2]),88)
            self.assertEqual(call.kwargs['end'],pd.Timestamp(meta['ends'][87]))
    def test_diagnostics_exact_nine_paths_and_27_ledger_artifacts(self):
        _,meta=arrays(122);idx=pd.to_datetime(meta['dates']);names=meta['symbols'][0]
        prices=pd.DataFrame(100.,index=idx,columns=names);eq=pd.Series(100.,index=idx)
        orders={n:[dict(symbols=['KO']) for _ in range(122)] for n in r.NAMES}
        replay=Mock(side_effect=lambda *a,**kw:({'annual_returns':[{'year':2016},{'year':2026}]},eq,[],[]))
        backend=object.__new__(r.Backend)
        accounting=NS(cuts=lambda e,m:{})
        backend.local=NS(ex=NS(e=NS(engine=NS(replay=replay))),
                        arch=NS(old=NS(prior=NS(old=NS(previous=accounting)))))
        saved={};results,periods=backend.diagnostics(orders,meta,prices,prices,lambda n,v:saved.update({n:v}))
        self.assertEqual(replay.call_count,9);self.assertEqual(len(saved),27);self.assertEqual(len(periods),122)
        self.assertEqual(set(results),set(r.NAMES))


if __name__ == '__main__':
    unittest.main(verbosity=2)
