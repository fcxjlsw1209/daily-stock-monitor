"""Synthetic-only independent161 tests; no producer imports or actual numerical data."""
from copy import deepcopy
import ast
from datetime import date,timedelta
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import rank161_independent_prepare as p
import rank161_independent_outcome as o
import rank161_independent as f


def feature_fixture(count=3,width=3):
    index=pd.bdate_range('2014-01-01','2026-10-02')
    calendar=[str(d.date()) for d in index]
    monthends=[calendar[i] for i in range(len(calendar)-1) if calendar[i][:7]!=calendar[i+1][:7]]
    dates=['2016-08-17']+[d for d in monthends if '2016-08-31'<=d<='2026-08-31']
    dates=dates[:count]
    names=['S%d'%i for i in range(width)]
    training_days=[d for d in monthends if '2016-03-01'<=d<='2026-08-31']
    training=dict(dates=training_days,symbols=[names[:] for _ in training_days])
    meta=dict(dates=dates,symbols=[names[:] for _ in dates])
    prices=pd.DataFrame({s:100*np.exp(np.arange(len(index))*(j+1)*.00005) for j,s in enumerate(names)},index=index)
    return meta,calendar,training,prices


class FeatureTests(unittest.TestCase):
    def test_constant_reversal_and_permutation(self):
        for value in (-.5,-.3,0.,.1,.5):
            residence,trend=p.deltas_from_ranks([value]*6)
            self.assertAlmostEqual(residence,.02*value)
            self.assertEqual(trend,0.)
        values=[-.5,-.3,-.1,.1,.3,.5]
        a=p.deltas_from_ranks(values);b=p.deltas_from_ranks(values[::-1])
        self.assertAlmostEqual(a[0],b[0]);self.assertAlmostEqual(a[1],-b[1])
        perm=p.deltas_from_ranks([values[i] for i in (0,5,1,4,2,3)])
        self.assertAlmostEqual(a[0],perm[0]);self.assertNotEqual(a[1],perm[1])
    def test_exact_bound(self):
        self.assertEqual(p.deltas_from_ranks([-.5]*3+[.5]*3)[1],.01)
        self.assertEqual(p.deltas_from_ranks([.5]*3+[-.5]*3)[1],-.01)
        with self.assertRaises(ValueError):p.deltas_from_ranks([.6]*6)
    def test_basic_rank_and_initial_neutral(self):
        args=feature_fixture();got=p.features_oracle(*args)
        self.assertEqual(got['audit']['event_history'][1]['usable_observations'],[6]*3)
        for value in got['arrays'].values():np.testing.assert_array_equal(value[0],np.zeros(3))
        first=got['audit']['event_history'][0]
        self.assertEqual(first['snapshots'][-1],p.INITIAL)
        self.assertEqual(first['usable_observations'],[6]*3)
        self.assertFalse(any(first['common_support']))
        for row in got['audit']['event_history'][1:]:self.assertNotIn(p.INITIAL,row['snapshots'])
    def test_ties_and_rank_all_endpoint_valid_before_positive(self):
        args=feature_fixture();px=args[3];px['S1']=px['S0'];px['S2']=100/np.exp(np.arange(len(px))*.0001)
        got=p.features_oracle(*args);row=got['audit']['snapshot_audits']['2016-08-31']
        self.assertEqual(row['endpoint_valid_count'],3);self.assertEqual(row['usable_count'],2)
        np.testing.assert_allclose(row['centered_percentile_ranks'],[2.5/3-.5,2.5/3-.5,1/3-.5])
        self.assertIsNone(got['audit']['event_history'][1]['rank_trajectories'][5][2])
    def test_warmup_missing_does_not_shrink_rank_population(self):
        args=feature_fixture();day='2016-08-31';pos=args[1].index(day)
        args[3].loc[args[1][pos-200],'S2']=np.nan
        got=p.features_oracle(*args);row=got['audit']['snapshot_audits'][day]
        self.assertEqual(row['endpoint_valid_count'],3);self.assertEqual(row['centered_percentile_ranks'][2],.5)
        self.assertFalse(row['complete_295_prices'][2]);self.assertFalse(row['usable'][2])
        self.assertEqual(got['arrays'][p.EXTRA[0]][1,2],0.)
    def test_endpoint_missing_shrinks_rank_population(self):
        args=feature_fixture();pos=args[1].index('2016-08-31')
        args[3].loc[args[1][pos-42],'S2']=np.nan
        row=p.features_oracle(*args)['audit']['snapshot_audits']['2016-08-31']
        self.assertEqual(row['endpoint_valid_count'],2)
        self.assertEqual(row['centered_percentile_ranks'],[0.,.5,None])
    def test_missing_historical_membership_neutral_without_fill(self):
        args=feature_fixture();i=args[2]['dates'].index('2016-04-29');args[2]['symbols'][i]=['S0','S1']
        got=p.features_oracle(*args);row=got['audit']['event_history'][1]
        self.assertEqual(row['usable_observations'],[6,6,5]);self.assertFalse(row['common_support'][2])
        self.assertTrue(all(got['arrays'][k][1,2]==0 for k in p.EXTRA))
    def test_missing_month_no_older_substitute(self):
        args=feature_fixture();i=args[2]['dates'].index('2016-04-29')
        for key in args[2]:args[2][key].pop(i)
        got=p.features_oracle(*args);row=got['audit']['event_history'][1]
        self.assertEqual(row['missing_snapshot_months'],['2016-04'])
        self.assertEqual(row['usable_observations'],[5]*3)
    def test_missing_exact_price_session_no_fill(self):
        args=list(feature_fixture());day='2016-08-31';pos=args[1].index(day)
        missing=args[1][pos-200];args[3]=args[3].drop(pd.Timestamp(missing))
        row=p.features_oracle(*args)['audit']['snapshot_audits'][day]
        self.assertIn(missing,row['missing_price_sessions']);self.assertEqual(row['usable_count'],0)
    def test_short_history_neutral(self):
        meta,calendar,training,prices=feature_fixture()
        calendar=calendar[calendar.index('2016-01-01'):]
        got=p.features_oracle(meta,calendar,training,prices)
        self.assertEqual(got['audit']['common_support_cells'],0)
    def test_invalid_monthend_and_conflicts_fail(self):
        args=feature_fixture();args[2]['dates'][1]='2016-04-28'
        with self.assertRaisesRegex(ValueError,'month end'):p.features_oracle(*args)
        args=feature_fixture();i=args[2]['dates'].index('2016-08-31');args[2]['symbols'][i]=['S0','S1','different']
        with self.assertRaisesRegex(ValueError,'Conflicting'):p.features_oracle(*args)
    def test_no_mutation_scale_invariance(self):
        args=feature_fixture();before=args[3].copy();metadata=deepcopy(args[:3])
        a=p.features_oracle(*args);b=p.features_oracle(*args[:3],args[3]*2)
        for k in p.EXTRA:np.testing.assert_allclose(a['arrays'][k],b['arrays'][k],atol=1e-12,rtol=0)
        pd.testing.assert_frame_equal(before,args[3]);self.assertEqual(metadata,args[:3])
    def test_123_prefixes_and_future_price_poison(self):
        args=feature_fixture(122,width=1);full=p.features_oracle(*args)
        for n in range(123):
            boundary=args[0]['dates'][n-1] if n else '1900-01-01'
            poisoned=args[3].copy();poisoned.loc[poisoned.index>pd.Timestamp(boundary)]=np.nan
            part=p.features_oracle(*args[:3],poisoned,n)
            for key in p.EXTRA:self.assertEqual(part['arrays'][key].tobytes(),full['arrays'][key][:n].tobytes())
    def test_valid_future_membership_invariant_schema_conflict_rejected(self):
        args=feature_fixture();base=p.features_oracle(*args,n=2);meta,cal,train,px=args
        meta=deepcopy(meta);train=deepcopy(train);future=meta['dates'][2]
        meta['symbols'][2]=['NEW0','NEW1','NEW2'];train['symbols'][train['dates'].index(future)]=meta['symbols'][2][:]
        after=p.features_oracle(meta,cal,train,px,n=2)
        p.compare(after['audit'],base['audit'])
        train['symbols'][train['dates'].index(future)]=['X']
        with self.assertRaises(ValueError):p.features_oracle(meta,cal,train,px,n=2)
    def test_saved_features_and_states_tamper_fail(self):
        got=p.features_oracle(*feature_fixture());saved=p.state_values(got['audit'])
        p.verify_feature_output(got,got['arrays'],got['audit'],saved)
        saved['common_support'][1][0]=False
        with self.assertRaises(ValueError):p.verify_feature_output(got,got['arrays'],got['audit'],saved)
    def test_nonconstant_cancellation_tolerated_but_neutral_exact(self):
        got=p.features_oracle(*feature_fixture());ranks=[k/49-.5 for k in (1,1,6,1,1,2)]
        row=got['audit']['event_history'][1]
        for k,value in enumerate(ranks):row['rank_trajectories'][k][0]=value
        residence,trend=p.deltas_from_ranks(ranks)
        got['arrays'][p.EXTRA[0]][1,0]=residence;got['arrays'][p.EXTRA[1]][1,0]=trend
        other={k:v.copy() for k,v in got['arrays'].items()}
        other[p.EXTRA[1]][1,0]=.01*(2.5*(ranks[5]-ranks[0])+1.5*(ranks[4]-ranks[1])+.5*(ranks[3]-ranks[2]))/4.5
        p.verify_feature_output(got,other,got['audit'],p.state_values(got['audit']))
        other[p.EXTRA[0]][0,0]=1e-20
        with self.assertRaisesRegex(ValueError,'Exact neutral'):p.verify_feature_output(got,other,got['audit'],p.state_values(got['audit']))
    def test_constant_trend_roundoff_rejected(self):
        got=p.features_oracle(*feature_fixture());other={k:v.copy() for k,v in got['arrays'].items()}
        other[p.EXTRA[1]][1,0]=1e-20
        with self.assertRaisesRegex(ValueError,'Constant rank'):p.verify_feature_output(got,other,got['audit'],p.state_values(got['audit']))
    def test_actual_deltas_hard_bound_not_numeric_tolerance(self):
        got=p.features_oracle(*feature_fixture());other={k:v.copy() for k,v in got['arrays'].items()}
        self.assertEqual(got['arrays'][p.EXTRA[0]][1,2],.01)
        other[p.EXTRA[0]][1,2]=.01+1e-13
        with self.assertRaisesRegex(ValueError,'hard one-percent'):p.verify_feature_output(got,other,got['audit'],p.state_values(got['audit']))
    def test_complete_support_counts_and_corruptions(self):
        args=feature_fixture(122,49);got=p.features_oracle(*args)
        support=o.support_oracle(got['audit'],args[0],got['arrays'])
        self.assertEqual(support['candidate_cells'],5978);self.assertIsNone(support['coverage_threshold'])
        self.assertEqual(support['common_support_cells'],121*49)
        self.assertEqual(support['coverage']['early53']['common_support_cells'],52*49)
        for field in ('common_support','usable_observations'):
            bad=deepcopy(got['audit']);bad['event_history'][1][field][0]=0
            with self.assertRaises(ValueError):o.support_oracle(bad,args[0],got['arrays'])
        bad={k:v.copy() for k,v in got['arrays'].items()};bad[p.EXTRA[0]][0,0]=.001
        with self.assertRaises(ValueError):o.support_oracle(got['audit'],args[0],bad)


def score_fixture():
    shape=(2,49);baseline=np.tile(np.linspace(.20,.10,49),(2,1))
    arrays=dict(baseline=baseline,core=.5*baseline,core_weight=np.full(2,.5),eligible=np.ones(shape,bool),
        state_weak_narrow=np.zeros(2,bool),state_calm10=np.ones(2,bool),pre_score=baseline.copy(),
        effects=np.zeros(shape),forecast_available=np.zeros(shape,bool),rank_log_excess=np.zeros(shape),
        old_ml4=np.zeros(shape),extension_available=np.zeros(shape,bool),model_available=np.zeros(shape,bool),
        raw_local_constant=np.zeros(shape),raw_local_linear=np.zeros(shape),raw_source_log=np.zeros(shape))
    names=['S'+str(i) for i in range(49)]
    meta=dict(symbols=[names[:],names[:]],etfs=names[-17:],dates=['2016-08-17','2016-08-31'],ends=['2016-08-31','2016-09-30'])
    hands={key:np.zeros(shape) for key in p.HANDS}
    hands[p.HANDS[0]][1]=np.linspace(-.01,.01,49)
    hands[p.HANDS[1]][1]=np.linspace(.01,-.01,49)
    return arrays,meta,hands


class ScoreTests(unittest.TestCase):
    def test_all123_score_prefixes(self):
        source,meta,hands=score_fixture();meta['dates']=feature_fixture(122,1)[0]['dates']
        meta['ends']=meta['dates'][1:]+['2026-09-30'];meta['symbols']=[meta['symbols'][0][:] for _ in range(122)]
        arrays={k:np.concatenate([v]*61,axis=0) for k,v in source.items()}
        hands={k:np.concatenate([v]*61,axis=0) for k,v in hands.items()}
        full=p.score_oracle(arrays,meta,hands)
        for n in range(123):
            part=p.score_oracle(arrays,meta,hands,n)
            for name in p.NAMES:
                self.assertEqual(part['scores'][name].tobytes(),full['scores'][name][:n].tobytes())
                self.assertEqual(part['choices'][name],full['choices'][name][:n])
    def test_delta_once_original_episode_and_top2(self):
        arrays,meta,hands=score_fixture();result=p.score_oracle(arrays,meta,hands)
        for name,key in zip(p.NAMES[1:],p.HANDS):
            np.testing.assert_array_equal(result['deltas'][name],hands[key])
            modified=arrays['pre_score'][1]+hands[key][1]
            top=sorted(range(49),key=lambda j:(-modified[j],meta['symbols'][1][j]))[:2]
            self.assertEqual(result['episodes'][name][1]['top2'],[meta['symbols'][1][j] for j in top])
            self.assertTrue(set(top).issubset(result['choices'][name][1]))
    def test_exact_ordered_holdings_fail_on_swapped_or_changed_score(self):
        arrays,meta,hands=score_fixture();orders=p.order_oracle(meta,p.score_oracle(arrays,meta,hands))
        p.verify_orders(orders,orders)
        swapped=deepcopy(orders);swapped[p.NAMES[1]][1]['symbols'].reverse()
        with self.assertRaisesRegex(ValueError,'Exact ordered holdings'):p.verify_orders(swapped,orders)
        changed=deepcopy(orders);changed[p.NAMES[2]][1]['scores'][0]+=1e-8
        with self.assertRaises(AssertionError):p.verify_orders(changed,orders)
    def test_initial_zero_and_input_unchanged(self):
        arrays,meta,hands=score_fixture();before={k:v.tobytes() for k,v in arrays.items()};result=p.score_oracle(arrays,meta,hands)
        for name in p.NAMES[1:]:self.assertEqual(result['scores'][name][0].tobytes(),result['scores'][p.NAMES[0]][0].tobytes())
        self.assertEqual(before,{k:v.tobytes() for k,v in arrays.items()})
        for hand in hands.values():hand[:]=0
        result=p.score_oracle(arrays,meta,hands)
        for name in p.NAMES[1:]:self.assertEqual(result['scores'][name].tobytes(),result['scores'][p.NAMES[0]].tobytes())
    def test_ineligible_and_etf_keep_delta_but_selector_enforces(self):
        arrays,meta,hands=score_fixture();arrays['eligible'][1,0]=False;result=p.score_oracle(arrays,meta,hands)
        for name,key in zip(p.NAMES[1:],p.HANDS):
            self.assertNotIn(0,result['choices'][name][1]);self.assertEqual(result['deltas'][name][1,0],hands[key][1,0])
            self.assertNotEqual(result['deltas'][name][1,-1],0)
    def test_invalid_delta_and_initial_nonzero_fail(self):
        arrays,meta,hands=score_fixture();hands[p.HANDS[0]][1,0]=.01001
        with self.assertRaises(ValueError):p.score_oracle(arrays,meta,hands)
        hands[p.HANDS[0]][1,0]=.01;hands[p.HANDS[0]][0,0]=.01
        with self.assertRaises(ValueError):p.score_oracle(arrays,meta,hands)
    def test_score_future_original_array_poison(self):
        arrays,meta,hands=score_fixture();first=p.score_oracle(arrays,meta,hands,1)
        for value in arrays.values():
            if value.dtype.kind=='f':value[1:]=np.nan
        last=p.score_oracle(arrays,meta,hands,1)
        for name in p.NAMES:self.assertEqual(first['scores'][name].tobytes(),last['scores'][name].tobytes())
    def test_original27_arrays_and_only2_extra(self):
        original={str(i):np.zeros((122,49)) for i in range(26)};original['baseline']=np.zeros((122,49))
        prepared={k:v.copy() for k,v in original.items()};prepared.update({k:np.zeros((122,49)) for k in p.HANDS})
        p.verify_original_arrays(original,prepared);prepared['baseline'][1,1]=1
        with self.assertRaises(ValueError):p.verify_original_arrays(original,prepared)


class SelectionTests(unittest.TestCase):
    def rows(self,gains=(0.,.006,.006)):
        return [dict(name=n,training=dict(cagr=.4+g,sharpe=1.5,max_drawdown=-.2),early=dict(cagr=.4),recent=dict(cagr=.4)) for n,g in zip(p.NAMES,gains)]
    def test_changed_set_first88_and_both_eras(self):
        orders={name:[dict(symbols=['A','B','C']) for _ in range(122)] for name in p.NAMES}
        orders[p.NAMES[1]][0]['symbols'].reverse();orders[p.NAMES[1]][100]['symbols']=['D','E','F']
        self.assertFalse(o.changed_support_oracle(orders,p.NAMES[1])['passed'])
        for i in (1,2,53):orders[p.NAMES[1]][i]['symbols']=['A','B','D']
        self.assertEqual(o.changed_support_oracle(orders,p.NAMES[1])['months'],[1,2,53]);self.assertTrue(o.changed_support_oracle(orders,p.NAMES[1])['passed'])
    def test_fixed_tie_and_source_guard(self):
        rows=self.rows();changed={n:dict(passed=True) for n in p.NAMES[1:]}
        self.assertEqual(o.select_first88_oracle(rows,rows,dict(passed=True),changed)['primary'],p.NAMES[1])
        self.assertEqual(o.select_first88_oracle(rows,rows,dict(passed=False),changed)['primary'],p.NAMES[0])
    def test_minimum_gain_and_open_cost_guard(self):
        changed={n:dict(passed=True) for n in p.NAMES[1:]};rows=self.rows((0,.0049,.004))
        self.assertEqual(o.select_first88_oracle(rows,rows,dict(passed=True),changed)['primary'],p.NAMES[0])
        rows=self.rows();opened=self.rows((0,-.0001,-.0001))
        self.assertEqual(o.select_first88_oracle(rows,opened,dict(passed=True),changed)['primary'],p.NAMES[0])
    def test_exact_epsilon_threshold_order(self):
        rows=self.rows();baseline=rows[0];candidate=deepcopy(rows[1])
        edge=baseline['training']['cagr']+.005-1e-12
        candidate['training']['cagr']=edge
        self.assertTrue(o.metric_guard(candidate,baseline,.005))
        candidate['training']['cagr']=float(np.nextafter(edge,-np.inf))
        self.assertFalse(o.metric_guard(candidate,baseline,.005))
        rows=self.rows((0.,.006,.006));rows[2]['training']['cagr']=rows[1]['training']['cagr']+1e-12
        changed={n:dict(passed=True) for n in p.NAMES[1:]}
        self.assertEqual(o.select_first88_oracle(rows,rows,dict(passed=True),changed)['primary'],p.NAMES[1])
        rows[2]['training']['cagr']=float(np.nextafter(rows[2]['training']['cagr'],np.inf))
        self.assertEqual(o.select_first88_oracle(rows,rows,dict(passed=True),changed)['primary'],p.NAMES[2])
    def test_registered_decimal_epsilon_edge_fixtures(self):
        rows=self.rows();baseline=rows[0];candidate=deepcopy(rows[1])
        baseline['training']['cagr']=.2;candidate['training']['cagr']=.204999999999
        self.assertTrue(o.metric_guard(candidate,baseline,.005))
        self.assertLess(candidate['training']['cagr']-baseline['training']['cagr'],.005-1e-12)
        rows=self.rows();rows[0]['training']['cagr']=.19
        rows[1]['training']['cagr']=.2;rows[2]['training']['cagr']=.20000000000100002
        changed={name:dict(passed=True) for name in p.NAMES[1:]}
        self.assertEqual(o.select_first88_oracle(rows,rows,dict(passed=True),changed)['primary'],p.NAMES[1])
    def test_later_values_irrelevant_and_nonfinite_guard(self):
        rows=self.rows((0,.006,.008));changed={n:dict(passed=True) for n in p.NAMES[1:]}
        for row in rows:row['later34']=dict(cagr=-10000.)
        self.assertEqual(o.select_first88_oracle(rows,rows,dict(passed=True),changed)['primary'],p.NAMES[2])
        rows[2]['training']['sharpe']=float('nan')
        self.assertEqual(o.select_first88_oracle(rows,rows,dict(passed=True),changed)['primary'],p.NAMES[1])


class PhaseAndAccountingTests(unittest.TestCase):
    def test_invalid_plan_before_any_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)/f.OUT;directory.mkdir(parents=True);path=directory/'PLAN.json';path.write_text(json.dumps(dict(round=159,status='REGISTERED',arms=list(p.NAMES))))
            reader=f.Reader(tmp)
            with self.assertRaisesRegex(ValueError,'Fixed registered'):f.bind(reader,hashlib.sha256(path.read_bytes()).hexdigest(),'prepare')
            self.assertEqual(set(reader.seen),{str(path.resolve())})
    def test_explicit_plan_sha_required(self):
        with self.assertRaisesRegex(ValueError,'Explicit PLAN'):f.bind(f.Reader('/private/tmp'),None,'prepare')
    def test_prepare_blocks_prior_outcome(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)/f.OUT;directory.mkdir(parents=True)
            plan=dict(round=161,status='REGISTERED',arms=list(p.NAMES),output_directory=str(directory.resolve()),runtime=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__),measurement=f.measurement_contract(),selection_contract=f.selection_contract(),actual_features_before_registration=0,actual_orders_before_registration=0,actual_returns_before_registration=0,no_observer_writes=True)
            path=directory/'PLAN.json';path.write_text(json.dumps(plan));(directory/'TRAIN_TRIALS.json').write_text('[]')
            with self.assertRaisesRegex(ValueError,'precede all outcomes'):f.bind(f.Reader(tmp),hashlib.sha256(path.read_bytes()).hexdigest(),'prepare')
    def test_exact_queue_and_prior_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();directory=root/f.OUT;directory.mkdir(parents=True);control=root/f.STATE;control.parent.mkdir(parents=True)
            before=dict(completed=[dict(trials=6)]*159+[dict(trials=41)],registered_queue=[])
            raw=json.dumps(before);(directory/'STATE_BEFORE.json').write_text(raw)
            plan=dict(registered_utc='2026-10-21T01:00:00+00:00',registration_precondition=dict(path=str(control),sha256=hashlib.sha256(raw.encode()).hexdigest()))
            entry=dict(round=161,directory='monthly_rank161',plan=dict(path=str(directory/'PLAN.json'),sha256='a'*64),registered_utc=plan['registered_utc'])
            current=dict(completed=before['completed'],registered_queue=[entry]);control.write_text(json.dumps(current));f.registration(root,plan,'a'*64)
            current['best_training_selected_candidate']={'changed':True};control.write_text(json.dumps(current))
            with self.assertRaisesRegex(ValueError,'incumbent metadata'):f.registration(root,plan,'a'*64)
            current.pop('best_training_selected_candidate')
            current['registered_queue']=[];control.write_text(json.dumps(current))
            with self.assertRaisesRegex(ValueError,'registered queue'):f.registration(root,plan,'a'*64)
    def test_duplicate_json_and_hash_change_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'x.json';path.write_text('{"x":1,"x":2}')
            with self.assertRaises(ValueError):f.Reader(tmp).json(path)
            reader=f.Reader(tmp);reader.sha(path);path.write_text('{}')
            with self.assertRaisesRegex(ValueError,'changed'):reader.sha(path)
    def test_forbidden_paths_and_import_guard(self):
        with self.assertRaises(ValueError):f.Reader('/private/tmp').path('/Users/test/OneDrive/x')
        with patch.dict(sys.modules,{'rank161_kernel':object()}):
            with self.assertRaisesRegex(ValueError,'imported'):p.independent_imports()
    def test_closure_and_exact_cache_schema(self):
        names={path.name for path in f.source_files()}
        self.assertTrue({'rank161_independent.py','rank161_independent_prepare.py','rank161_independent_outcome.py','test_rank161_independent.py','monthly_ridge_support70_result_review.py','monthly_paired_target143_outcome_review.py','monthly_latent_state134_prepare_review.py','inputs.py'}<=names)
        self.assertEqual(len(f.CACHE_FILES),8);self.assertIn('STATE_VALUES.json',f.CACHE_FILES)
    def test_source_only_static_host_interfaces(self):
        # Parse source only. Never import or execute the new host/runner/kernel.
        root=Path(__file__).resolve().parent
        host=ast.parse((root/'rank161_host.py').read_text())
        assignments={target.id:node.value for node in host.body if isinstance(node,ast.Assign)
                     for target in node.targets if isinstance(target,ast.Name)}
        self.assertEqual(set(ast.literal_eval(assignments['CACHE'])),set(f.CACHE_FILES))
        self.assertEqual(ast.literal_eval(assignments['PROTECTED_SHA']),'89e82ac5cbec3b998f1d8aab93f1f8fd7c5e19522321b2dd1da2f88dfff922cb')
        runner=ast.parse((root/'rank161_runner.py').read_text())
        declarations=[node for node in runner.body if isinstance(node,ast.FunctionDef) and node.name in ('measurement_contract','selection_contract')]
        # Validate before evaluating declaration-only AST; never execute imports,
        # decorators, defaults, or any producer body other than a literal return.
        for declaration in declarations:
            self.assertEqual(len(declaration.body),1)
            self.assertIsInstance(declaration.body[0],ast.Return)
            self.assertFalse(declaration.decorator_list or declaration.args.args or declaration.args.defaults)
            allowed=(ast.Return,ast.Call,ast.Name,ast.Load,ast.Constant,ast.List,ast.Tuple,ast.keyword,ast.UnaryOp,ast.USub)
            self.assertTrue(all(isinstance(node,allowed) for node in ast.walk(declaration.body[0])))
            calls=[node for node in ast.walk(declaration) if isinstance(node,ast.Call)]
            self.assertTrue(all(isinstance(call.func,ast.Name) and call.func.id in ('dict','list') for call in calls))
        scope={'NAMES':p.NAMES,'EXTRA':p.EXTRA,'MAX_ABS_DELTA':.01}
        exec(compile(ast.Module(body=declarations,type_ignores=[]),'static_literal_contracts','exec'),scope)
        self.assertEqual(scope['measurement_contract'](),f.measurement_contract())
        self.assertEqual(scope['selection_contract'](),f.selection_contract())
        p.independent_imports()
    def test_nine_paths_synthetic_fixed_units_and_tamper(self):
        dates=pd.bdate_range('2020-01-01',periods=245);names=['A','B','C']
        close=pd.DataFrame(100.,index=dates,columns=names);opening=close.copy()
        signals=[str(x.date()) for x in dates[::2][:-1]];ends=[str(x.date()) for x in dates[::2][1:]]
        meta=dict(dates=signals,ends=ends,symbols=[names[:] for _ in signals])
        orders={name:[dict(date=d,signal_date=d,end=e,symbols=names[:],scores=[.3,.2,.1]) for d,e in zip(signals,ends)] for name in p.NAMES}
        result=o.nine_path_oracle(close,opening,orders,meta)
        self.assertEqual((result['paths'],result['orders'],len(result['details'])),(9,366,9))
        artifacts={'RESULTS.json':dict(arms=result['results']),'TRAIN_TRIALS.json':result['training'],'TRAIN_OPEN_TRIALS.json':result['opened']}
        for name,values in result['details'].items():
            for key in ('daily','trades','holdings'):artifacts[name+'_'+key.upper()+'.json']=deepcopy(values[key])
        self.assertEqual(o.compare_nine_paths(result,artifacts)['paths'],9)
        for name in p.NAMES:
            self.assertEqual(result['results'][name]['same_close']['metrics']['total_return'],0.)
            self.assertLess(result['results'][name]['next_open_cost']['metrics']['total_return'],0.)
        artifacts[p.NAMES[0]+'_same_close_DAILY.json'][1]['equity']+=1
        with self.assertRaises(AssertionError):o.compare_nine_paths(result,artifacts)


if __name__=='__main__':
    p.independent_imports();unittest.main(verbosity=2)

