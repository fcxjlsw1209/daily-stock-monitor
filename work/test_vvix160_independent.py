"""Synthetic-only independent160 tests; no producer imports or actual numerical data."""
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
import vvix160_independent_prepare as p
import vvix160_independent_outcome as o
import vvix160_independent as f


def feature_fixture(count=3):
    start=date(2015,1,1)
    cal=[(start+timedelta(days=i)).isoformat() for i in range(1800) if (start+timedelta(days=i)).weekday()<5]
    dates=['2016-08-17','2016-08-31','2016-09-30'][:count]
    if count==122:dates=['2016-08-17']+cal[cal.index('2016-08-31'):cal.index('2016-08-31')+121]
    values={s:{d:str(10+(i%31)/10) for i,d in enumerate(cal)} for s in p.SOURCES}
    vol=np.tile(np.linspace(-.5,.5,49),(len(dates),1))
    return dict(dates=dates),cal,values,vol


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


class FeatureTests(unittest.TestCase):
    def test_last_minimum_and_maximum(self):
        self.assertEqual(p.midrank_state([str(i) for i in range(1,253)]),(1.,252.))
        self.assertEqual(p.midrank_state([str(i) for i in range(252,0,-1)]),(-1.,1.))
    def test_all_equal_exact_zero(self):
        self.assertEqual(p.midrank_state(['12.000']*252),(0.,126.5))
    def test_tied_latest_average_rank(self):
        values=['1']*100+['3']*100+['2']*52
        self.assertEqual(p.midrank_state(values),(0.,126.5))
    def test_sorted_reference_random_ties(self):
        rng=random.Random(160)
        for _ in range(80):
            values=[rng.randrange(1,101) for _ in range(252)]
            ordered=sorted(values);ranks=[i+1 for i,x in enumerate(ordered) if x==values[-1]]
            rank=Fraction(sum(ranks),len(ranks));state=2*(rank-1)/251-1
            self.assertEqual(p.midrank_state(list(map(str,values))),(float(state),float(rank)))
    def test_invalid_required_values_fail(self):
        for bad in ('0','-2','NaN','Infinity','garbage',True):
            with self.subTest(bad=bad),self.assertRaises(ValueError):p.midrank_state(['10']*251+[bad])
        with self.assertRaises(ValueError):p.midrank_state(['10']*251)
    def test_float_collision_fails(self):
        with self.assertRaisesRegex(ValueError,'collision'):
            p.midrank_state(['1']*251+['1.0000000000000000000000000001'])
    def test_csv_preserves_unconsumed_bad_and_weekend(self):
        raw=b'DATE,VVIX\n08/17/2016,10\n08/20/2016,bad\n08/31/2016,NaN\n'
        self.assertEqual(p.csv_values('VVIX',raw)['2016-08-20'],'bad')
        raw=b'DATE,OPEN,HIGH,LOW,CLOSE\n08/17/2016,bad,NaN,-1,12.50\n'
        self.assertEqual(p.csv_values('VIX',raw),{'2016-08-17':'12.50'})
    def test_csv_duplicate_chronology_and_schema_fail(self):
        for raw in (b'DATE,VVIX\n08/17/2016,1\n08/17/2016,2\n',b'DATE,VVIX\n08/18/2016,1\n08/17/2016,2\n',b'DATE,CLOSE\n08/17/2016,1\n'):
            with self.assertRaises(ValueError):p.csv_values('VVIX',raw)
    def test_exact_prior252_excludes_signal(self):
        meta,cal,values,vol=feature_fixture();result=p.features_oracle(meta,cal,values,vol)
        window=p.previous_window(cal,meta['dates'][1]);row=result['audit']['monthly'][1]
        self.assertEqual((row['window_start'],row['window_end'],row['window_size']),(window[0],window[-1],252))
        self.assertLess(window[-1],meta['dates'][1])
    def test_initial_partial_neutral(self):
        meta,cal,values,vol=feature_fixture(1);result=p.features_oracle(meta,cal,{s:{} for s in p.SOURCES},vol)
        self.assertFalse(result['common_available'][0])
        for value in result['arrays'].values():self.assertEqual(value.tobytes(),np.zeros((1,49)).tobytes())
    def test_missing_source_fails_no_fill(self):
        meta,cal,values,vol=feature_fixture();del values['VIX'][p.previous_window(cal,meta['dates'][1])[0]]
        with self.assertRaisesRegex(ValueError,'Missing exact'):p.features_oracle(meta,cal,values,vol)
    def test_short_calendar_and_non_session_fail(self):
        with self.assertRaises(ValueError):p.previous_window(['2016-08-17'],'2016-08-17')
        _,cal,_,_=feature_fixture()
        with self.assertRaises(ValueError):p.previous_window(cal,'2016-08-20')
    def test_noncalendar_values_not_consumed(self):
        args=feature_fixture();expected=p.features_oracle(*args);args[2]['VVIX']['2016-08-20']='bad'
        self.assertEqual(expected['audit'],p.features_oracle(*args)['audit'])
    def test_all_prefixes_and_future_poison(self):
        meta,cal,values,vol=feature_fixture();full=p.features_oracle(meta,cal,values,vol)
        for n in range(4):
            cutoff=meta['dates'][n-1] if n else ''
            poison={s:{d:(v if d<cutoff else object()) for d,v in series.items()} for s,series in values.items()}
            pv=vol.copy();pv[n:]=np.nan
            prefix=p.features_oracle(meta,cal,poison,pv,n)
            for key in p.HANDS:self.assertEqual(prefix['arrays'][key].tobytes(),full['arrays'][key][:n].tobytes())
    def test_once_delta_and_no_rerank(self):
        args=feature_fixture();result=p.features_oracle(*args)
        for source,key in zip(p.SOURCES,p.HANDS):
            expected=-.02*result['states'][source][:,None]*args[3];expected[0]=0
            np.testing.assert_array_equal(result['arrays'][key],expected)
            self.assertLessEqual(np.abs(result['arrays'][key]).max(),.01)
    def test_vol_dtype_bounds_and_initial_future_scope(self):
        meta,cal,values,vol=feature_fixture()
        for bad in (vol.astype(np.float32),np.full(vol.shape,.50001),np.full(vol.shape,np.nan)):
            with self.assertRaises(ValueError):p.features_oracle(meta,cal,values,bad)
        vol[1:]=np.nan;p.features_oracle(meta,cal,values,vol,1)
    def test_saved_feature_tamper_fails(self):
        expected=p.features_oracle(*feature_fixture());saved=dict(states={k:v.tolist() for k,v in expected['states'].items()},common_available=expected['common_available'].tolist())
        p.verify_feature_output(expected,expected['arrays'],expected['audit'],saved)
        saved['states']['VIX'][1]+=.1
        with self.assertRaises(ValueError):p.verify_feature_output(expected,expected['arrays'],expected['audit'],saved)


class ScoreTests(unittest.TestCase):
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
    def test_complete_source_counts(self):
        meta,cal,values,vol=feature_fixture(122);audit=p.features_oracle(meta,cal,values,vol)['audit']
        got=o.support_oracle(audit['monthly'],meta)
        self.assertEqual((got['first88'],got['early53'],got['recent35'],got['later34']),(87,52,35,34));self.assertTrue(got['passed'])
        audit['monthly'][90]['VVIX']['passed']=False;self.assertFalse(o.support_oracle(audit['monthly'],meta)['passed'])
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
            plan=dict(round=160,status='REGISTERED',arms=list(p.NAMES),output_directory=str(directory.resolve()),runtime=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__),measurement=f.measurement_contract(),selection_contract=f.selection_contract(),actual_features_before_registration=0,actual_orders_before_registration=0,actual_returns_before_registration=0,no_observer_writes=True)
            path=directory/'PLAN.json';path.write_text(json.dumps(plan));(directory/'TRAIN_TRIALS.json').write_text('[]')
            with self.assertRaisesRegex(ValueError,'precede all outcomes'):f.bind(f.Reader(tmp),hashlib.sha256(path.read_bytes()).hexdigest(),'prepare')
    def test_exact_queue_and_prior_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();directory=root/f.OUT;directory.mkdir(parents=True);control=root/f.STATE;control.parent.mkdir(parents=True)
            before=dict(completed=[dict(trials=6)]*158+[dict(trials=44)],registered_queue=[])
            raw=json.dumps(before);(directory/'STATE_BEFORE.json').write_text(raw)
            plan=dict(registered_utc='2026-10-21T01:00:00+00:00',registration_precondition=dict(path=str(control),sha256=hashlib.sha256(raw.encode()).hexdigest()))
            entry=dict(round=160,directory='monthly_vvix160',plan=dict(path=str(directory/'PLAN.json'),sha256='a'*64),registered_utc=plan['registered_utc'])
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
        with patch.dict(sys.modules,{'vvix160_kernel':object()}):
            with self.assertRaisesRegex(ValueError,'imported'):p.independent_imports()
    def test_closure_and_exact_cache_schema(self):
        names={path.name for path in f.source_files()}
        self.assertTrue({'vvix160_independent.py','vvix160_independent_prepare.py','vvix160_independent_outcome.py','test_vvix160_independent.py','monthly_ridge_support70_result_review.py','monthly_paired_target143_outcome_review.py','monthly_latent_state134_prepare_review.py','inputs.py'}<=names)
        self.assertEqual(len(f.CACHE_FILES),8);self.assertIn('STATE_VALUES.json',f.CACHE_FILES)
    def test_source_only_static_host_interfaces(self):
        # Parse source only. Never import or execute the new host/runner/kernel.
        root=Path(__file__).resolve().parent
        host=ast.parse((root/'vvix160_host.py').read_text())
        assignments={target.id:node.value for node in host.body if isinstance(node,ast.Assign)
                     for target in node.targets if isinstance(target,ast.Name)}
        self.assertEqual(set(ast.literal_eval(assignments['CACHE'])),set(f.CACHE_FILES))
        self.assertEqual(ast.literal_eval(assignments['PROTECTED_SHA']),'89e82ac5cbec3b998f1d8aab93f1f8fd7c5e19522321b2dd1da2f88dfff922cb')
        runner=ast.parse((root/'vvix160_runner.py').read_text())
        declarations=[node for node in runner.body if isinstance(node,ast.FunctionDef) and node.name in ('measurement_contract','selection_contract')]
        # Validate before evaluating declaration-only AST; never execute imports,
        # decorators, defaults, or any producer body other than a literal return.
        for declaration in declarations:
            self.assertEqual(len(declaration.body),1)
            self.assertIsInstance(declaration.body[0],ast.Return)
            self.assertFalse(declaration.decorator_list or declaration.args.args or declaration.args.defaults)
            allowed=(ast.Return,ast.Call,ast.Name,ast.Load,ast.Constant,ast.List,ast.Tuple,ast.keyword)
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
