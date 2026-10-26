"""Synthetic-only four-arm wiring, original-score placement, and phase gates.

Never loads real earnings inputs, real PLANs, original arrays, or real returns.
The inherited score adapter is exercised only with generated arrays.
"""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json
import tempfile
import unittest
import numpy as np
import pandas as pd
import monthly_peer_earnings139_runner as r

ROOT=Path(r.base.__file__).resolve().parents[1]
ADAPTER=r.module('test_peer139_path',ROOT/'outputs/monthly_path70_research/incumbent_adapter.py')
POLICY=dict(budget=.05,ceiling=5,gap=.02)


def synthetic():
    sessions=pd.bdate_range('2015-01-01',periods=2700)
    selected=[21+22*i for i in range(122)]
    names=['S'+str(i) for i in range(32)]+['E'+str(i) for i in range(17)]
    m=dict(dates=[str(sessions[i].date()) for i in selected],
        ends=[str(sessions[i+1].date()) for i in selected],train_months=88,
        symbols=[names[:] for i in selected],etfs=names[32:])
    raw=np.tile(np.linspace(-.49,.49,49),(122,1));risk=-raw
    calm=np.arange(122)%2==0;cw=np.where(calm,.5,.52)
    core=np.where(calm[:,None],.5*raw,.52*risk)
    a=dict(core=core,core_weight=cw,baseline=core+.01,
        eligible=np.ones(core.shape,bool),effects=np.zeros(core.shape),
        state_calm10=calm,state_weak_narrow=np.zeros(122,bool))
    a.update({'cached'+str(i):np.full(core.shape,float(i)) for i in range(20)})
    t=dict(core=core.copy(),cw=np.broadcast_to(cw[:,None],core.shape).copy(),
        eligible=a['eligible'].copy(),calm_core=.5*raw,noncalm_core=.52*risk)
    price=100.+np.arange(len(sessions))[:,None]*.001*np.arange(1,50)[None,:]
    return m,a,t,pd.DataFrame(price,index=sessions,columns=names),[str(d.date()) for d in sessions]


def trials():
    return [dict(name=n,training=dict(cagr=.2,sharpe=1.,max_drawdown=-.2),
        early=dict(cagr=.2),recent=dict(cagr=.2)) for n in r.NAMES]


class FakeBackend:
    def __init__(self,m,a,px):
        self.root=ROOT;self.meta=m;self.arrays=a;self.px=px;self.adapter=ADAPTER;self.panel={}
        self.loads=self.training_calls=self.diagnostic_calls=self.episodes=0
        self.old130=SimpleNamespace(attribution=lambda periods,results,new,old:
            dict(new=new,old=old,changed_months=1,changes=[dict(date=periods[0]['date'],guidance_events=[])]))
    def load(self):
        self.loads+=1;return self.meta,self.arrays,self.px,self.px
    def pre(self,a,n):
        x=np.zeros(a['baseline'].shape);x[:n]=a['baseline'][:n];return x
    def selector(self,scores,good,symbols,effects,original,policy):
        self.episodes+=1;self.last_pre=scores.copy();self.last_original=deepcopy(original)
        assert policy==POLICY
        return [sorted(np.flatnonzero(valid).tolist(),key=lambda j:(-score[j],names[j]))[:3]
                for score,valid,names in zip(scores,good,symbols)],scores.copy()
    def baseline(self,a,m,n,policy):
        return ADAPTER.score_and_choose('incumbent66',0.,np.full(a['baseline'].shape,np.nan),a,m,n,
            pre_score=self.pre(a,n),effects=a['effects'],episode_policy=policy,boundary_selector=self.selector)
    @staticmethod
    def make_orders(choices,scores,m):
        return [dict(date=m['dates'][i],symbols=[m['symbols'][i][j] for j in row],
            scores=[float(scores[i,j]) for j in row]) for i,row in enumerate(choices)]
    def price_parity(self,*args):return dict(synthetic=True)
    def benchmark_map(self,*args):return {}
    def training(self,orders,m,px):
        self.training_calls+=1;self.training_length=len(orders[r.NAMES[0]][:88])
        rows=trials();rows[3]['training']['cagr']=.21
        return rows,dict(folds=[dict(n=n,end=end,selection=r.select(rows))
            for n,end in ((36,60),(60,76),(76,88))],pristine_oos=False)
    def diagnostics(self,orders,m,px,op,writer):
        self.diagnostic_calls+=1
        results={name:{key:dict(metrics={},slices={'training2016_2023':dict(cagr=.21 if j==3 else .2)})
            for key in ('same_close','next_open_zero','next_open_cost')} for j,name in enumerate(r.NAMES)}
        months=[dict(date=day,end=m['ends'][i],split='early53' if i<53 else 'recent35' if i<88 else 'later34')
            for i,day in enumerate(m['dates'])]
        return results,months


class PlacementTests(unittest.TestCase):
    def setUp(self):
        self.m,self.a,t,px,_=synthetic();self.b=FakeBackend(self.m,self.a,px)
        hand=np.tile(np.linspace(-.49,.5,32),(122,1));full=np.full((122,49),np.nan);full[:,:32]=hand
        self.a.update(raw_peer_recent10=full.copy(),raw_negative_own_event10=full.copy(),raw_peer_event10=full.copy(),
            hand_peer_recent10=full.copy(),hand_negative_own_event10=full.copy(),hand_peer_event10=full.copy(),
            common_available=np.isfinite(full),source_available=np.isfinite(full),peer_count=np.full((122,49),31,dtype=np.int64))
    def call(self,name=None,n=1):return r.choose(name or r.NAMES[3],self.a,self.m,n,self.b,POLICY)
    def test_exact_support_increment_one_episode(self):
        _,score=self.call();support=(self.a['baseline'][0]-self.a['core'][0])/(1-self.a['core_weight'][0])
        delta=.025*(np.clip(support[:32]+self.a['hand_peer_event10'][0,:32],-.5,.5)-support[:32])
        expected=self.a['baseline'][0].copy();expected[:32]+=delta
        np.testing.assert_array_equal(score[0],expected);self.assertEqual(self.b.episodes,1)
        self.assertLessEqual(float(np.max(np.abs(delta))),.0125+1e-12)
        np.testing.assert_array_equal(self.b.last_original['core'],self.a['core'][:1])
    def test_all_three_arms_identical_hands_and_etf_unchanged(self):
        c,s=self.call()
        for name in r.NAMES[1:]:
            d,t=self.call(name);self.assertEqual(c,d);self.assertEqual(s.tobytes(),t.tobytes())
            np.testing.assert_array_equal(t[0,32:],self.a['baseline'][0,32:])
    def test_three_distinct_hands_are_wired_to_the_named_arm_once(self):
        for name,value in zip(r.NAMES[1:],(.05,-.10,.20)):
            self.a['hand_'+name][:,:32]=value
        for name,value in zip(r.NAMES[1:],(.05,-.10,.20)):
            before=self.b.episodes
            _,score=self.call(name)
            self.assertEqual(self.b.episodes,before+1)
            expected=self.a['baseline'][0].copy();expected[:32]+=.025*value
            np.testing.assert_allclose(score[0],expected,rtol=0,atol=1e-16)
        with self.assertRaisesRegex(ValueError,'Unregistered arm'):self.call('unexpected')
    def test_zero_and_missing_exact_baseline(self):
        expected=self.call(r.NAMES[0])
        for value in (0.,np.nan):
            for key in r.factor.HAND_NAMES:self.a[key][:,:32]=value
            for name in r.NAMES[1:]:
                c,s=self.call(name);self.assertEqual(c,expected[0]);self.assertEqual(s.tobytes(),expected[1].tobytes())
    def test_future_poison_ignored(self):
        expected=self.call()
        for key in ('baseline','core','effects','core_weight'):
            self.a[key][1:]=np.nan
        for key in r.factor.HAND_NAMES:
            self.a[key]=self.a[key].astype(object);self.a[key][1:]='future'
        c,s=self.call();self.assertEqual(c,expected[0]);self.assertEqual(s.tobytes(),expected[1].tobytes())
    def test_bad_hand_and_etf_rank_rejected(self):
        for value in (np.inf,.51,True,'0.1'):
            a=self.a['hand_peer_event10'].astype(object);a[0,0]=value
            with patch.dict(self.a,hand_peer_event10=a),self.subTest(value=value),self.assertRaises(ValueError):self.call()
        self.a['hand_peer_event10'][0,32]=0.
        with self.assertRaises(ValueError):self.call()
    def test_all_panels_prefix_and_nonmutation(self):
        old={k:v.tobytes() for k,v in self.a.items()};orders,scores=r.panels(self.a,self.m,self.b,POLICY)
        self.assertEqual([len(v) for v in orders.values()],[122]*4)
        self.assertEqual(old,{k:v.tobytes() for k,v in self.a.items()})
        self.assertEqual([v.shape for v in scores.values()],[(122,49)]*4)
        self.assertEqual(sum(map(len,orders.values())),488)
        self.assertEqual(list(orders),list(r.NAMES))
    def test_panels_detect_backend_input_mutation(self):
        original=self.b.pre
        def mutating(a,n):
            a['cached0'][0,0]+=1.
            return original(a,n)
        with patch.object(self.b,'pre',side_effect=mutating),self.assertRaisesRegex(ValueError,'Input mutation'):
            r.panels(self.a,self.m,self.b,POLICY)
    def test_panels_detect_noncausal_prefix(self):
        original=self.b.pre
        def noncausal(a,n):
            out=original(a,n);out[:n]+=n*.000001;return out
        with patch.object(self.b,'pre',side_effect=noncausal),self.assertRaisesRegex(ValueError,'Prefix mismatch'):
            r.panels(self.a,self.m,self.b,POLICY)


class SelectionTests(unittest.TestCase):
    def test_only_peer_event_can_win(self):
        for control in (1,2):
            rows=trials();rows[control]['training']['cagr']=.9
            self.assertEqual(r.select(rows)['primary'],r.NAMES[0])
            rows[3]['training']['cagr']=.91
            self.assertEqual(r.select(rows)['primary'],r.NAMES[3])
    def test_strict_cagr_against_each_reference(self):
        for ref in (0,1,2):
            rows=trials();rows[3]['training']['cagr']=.21;rows[ref]['training']['cagr']=.21-1e-12
            self.assertFalse(r.select(rows)['admitted'])
            rows[ref]['training']['cagr']=.21-2e-12;self.assertTrue(r.select(rows)['admitted'])
    def test_each_guard_against_all_three(self):
        for ref in (0,1,2):
            for group,key,delta in (('training','sharpe',.051),('training','max_drawdown',.031),('early','cagr',.026),('recent','cagr',.026)):
                rows=trials();rows[3]['training']['cagr']=.21;rows[ref][group][key]+=delta
                with self.subTest(ref=ref,key=key):self.assertFalse(r.select(rows)['admitted'])
    def test_invalid_metrics_and_arm_order(self):
        for x in (None,True,'1',float('nan'),float('inf')):
            rows=trials();rows[3]['training']['cagr']=x;self.assertFalse(r.select(rows)['admitted'])
        with self.assertRaises(ValueError):r.select(trials()[::-1])


class PhaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve();self.out=self.root/'outputs/monthly_peer_earnings139';self.out.mkdir(parents=True)
        self.m,self.a,self.t,px,sessions=synthetic();self.b=FakeBackend(self.m,self.a,px)
        c,s=self.b.baseline(self.a,self.m,122,POLICY)
        values=dict(proposal={'round':139},meta=self.m,policy=POLICY,
            original_orders=self.b.make_orders(c,s,self.m),prices={'synthetic':True},
            sessions=dict(sessions=sessions),protected_manifest={},reviewer={'synthetic':True},raw={'source':'synthetic'},event={'source':'synthetic_event'})
        self.paths={}
        for key,value in values.items():
            self.paths[key]=self.root/(key+'.json');r.dump(self.paths[key],value)
        self.plan=dict(round=139,status='REGISTERED',arms=list(r.NAMES),primary_challenger=r.NAMES[3],
            output_directory=str(self.out),runtime=dict(python=r.sys.version,numpy=np.__version__,pandas=pd.__version__),
            tolerances=r.TOLERANCES,inputs={k:dict(path=str(self.paths[k]),sha256=r.sha(self.paths[k])) for k in r.INPUTS},
            hashes={str(p):r.sha(p) for p in self.paths.values()},
            raw_source_hashes={str(self.paths['raw']):r.sha(self.paths['raw'])},
            event_input_hashes={str(self.paths['event']):r.sha(self.paths['event'])},
            independent_prepare_source_hashes={str(self.paths['reviewer']):r.sha(self.paths['reviewer'])})
        self.plan_path=self.out/'PLAN.json';r.dump(self.plan_path,self.plan)
        for name,value in (('PROPOSAL_SHA',r.sha(self.paths['proposal'])),('PROTECTED_SHA',r.sha(self.paths['protected_manifest']))):
            p=patch.object(r,name,value);p.start();self.addCleanup(p.stop)
        p=patch.object(r,'source_files',return_value=self.plan['independent_prepare_source_hashes']);p.start();self.addCleanup(p.stop)
        def normalized(root,reader,meta,panel,official,mapper):
            self.normalized_calls+=1
            self.assertEqual(reader.json(self.paths['event']), {'source':'synthetic_event'})
            self.assertEqual(official, sessions)
            return dict(meta=meta, synthetic=True), dict(synthetic=True, source_sha256=reader.sha(self.paths['event']))
        def features(**inputs):
            self.assertEqual(set(inputs), {'meta','synthetic'})
            self.assertTrue(inputs['synthetic'])
            meta=inputs['meta']
            arrays={k:np.full((122,49),np.nan) for k in r.factor.RAW_NAMES+r.factor.HAND_NAMES}
            arrays.update(common_available=np.zeros((122,49),bool),source_available=np.zeros((122,49),bool),peer_count=np.zeros((122,49),np.int64))
            cells=[dict(i=i,j=j,signal_date=day,symbol=s,source_available=False,available=False,
                reason='synthetic_missing',peer_count=0,peer_symbols=[],peer_issuer_keys=[],peer_event_ids=[])
                for i,(day,names) in enumerate(zip(meta['dates'],meta['symbols'])) for j,s in enumerate(names)]
            return dict(arrays=arrays,audit=dict(cells=cells))
        self.normalized_calls=0
        p=patch.object(r.event_inputs,'load',side_effect=normalized);p.start();self.addCleanup(p.stop)
        self.feature_patch=patch.object(r.factor,'build_peer_features',side_effect=features)
        self.feature_mock=self.feature_patch.start();self.addCleanup(self.feature_patch.stop)
        self.runner=self.make_runner()
    def make_runner(self):return r.Runner(self.root,self.plan_path,r.sha(self.plan_path),lambda root,runner:self.b)
    def review(self):
        artifacts=dict(self.runner.cache());artifacts.update({str(self.out/x):r.sha(self.out/x) for x in ('PLAN.json','CACHE.json')})
        return dict(passed=True,stage='prepare',producer_modules_imported=False,returns_or_nav_computed=False,
            monthly_label_models_fit=0,source_hashes=self.plan['independent_prepare_source_hashes'],artifact_hashes=artifacts,
            plan_sha256=r.sha(self.plan_path),cache_sha256=r.sha(self.out/'CACHE.json'),
            orders_sha256=r.sha(self.out/'PRETRAIN_ORDERS.json'),**{k:True for k in r.CHECKS})
    def test_all_phases_six_caches_four_arms_and_peer_attribution(self):
        self.runner.prepare();self.assertEqual(self.b.training_calls,0)
        self.assertEqual(self.normalized_calls,1);self.assertEqual(self.feature_mock.call_count,1)
        self.assertEqual(set(r.read(self.out/'CACHE.json')),set(r.CACHE_FILES))
        audit=r.read(self.out/'FEATURE_AUDIT.json')
        self.assertEqual(len(audit['cells']),5978)
        self.assertEqual(audit['new_model_fits'],0)
        self.assertEqual(audit['input_provenance'],dict(synthetic=True,source_sha256=r.sha(self.paths['event'])))
        orders=r.read(self.out/'PRETRAIN_ORDERS.json')
        self.assertEqual(sum(map(len,orders.values())),488)
        with np.load(self.out/'DATA.npz',allow_pickle=False) as z:
            self.assertEqual(set(z.files),set(self.a)|set(r.EXTRA))
            for name,original in self.a.items():
                self.assertEqual(z[name].dtype,original.dtype)
                self.assertEqual(z[name].shape,original.shape)
                self.assertEqual(z[name].tobytes(),original.tobytes())
        r.dump(self.out/'INDEPENDENT_PREPARE.json',self.review());self.runner.select_first88()
        self.assertEqual(self.b.training_length,88);self.assertEqual(self.b.diagnostic_calls,0)
        self.runner.diagnose_later34();self.assertEqual(self.b.diagnostic_calls,1)
        results=r.read(self.out/'RESULTS.json');self.assertEqual(sum(map(len,results.values())),12)
        months=r.read(self.out/'MONTH_COMPARISON.json');self.assertEqual(sum(m['split']=='later34' for m in months),34)
        attrs=r.read(self.out/'ATTRIBUTION.json')['comparisons']
        self.assertEqual(set(attrs),{r.NAMES[3]+'_vs_'+name for name in r.NAMES[:3]})
        for c in attrs.values():
            self.assertEqual(c['changes'][0]['peer_earnings'],months[0]['peer_earnings']);self.assertEqual(c['changes'][0]['guidance_events'],[])
        self.assertEqual(r.read(self.out/'VALIDATION.json')['new_model_fits'],0)
        with self.assertRaises(ValueError):self.runner.prepare()
        with self.assertRaises(ValueError):self.runner.select_first88()
        with self.assertRaisesRegex(ValueError,'Diagnostic stage already started'):self.runner.diagnose_later34()
        self.assertEqual(self.b.training_calls,1);self.assertEqual(self.b.diagnostic_calls,1)
    def test_partial_prepare_output_prevents_any_backend_load(self):
        (self.out/'DATA.npz').write_bytes(b'partial synthetic output')
        with self.assertRaisesRegex(ValueError,'Prepare already started'):self.runner.prepare()
        self.assertEqual(self.b.loads,0);self.assertEqual(self.normalized_calls,0)
    def test_diagnostics_without_frozen_selection_stops_before_replay(self):
        with self.assertRaises(FileNotFoundError):self.runner.diagnose_later34()
        self.assertEqual(self.b.loads,0);self.assertEqual(self.b.diagnostic_calls,0)
    def test_partial_selection_stops_before_training(self):
        self.runner.prepare();r.dump(self.out/'INDEPENDENT_PREPARE.json',self.review())
        r.dump(self.out/'TRAIN_TRIALS.json',{'partial':True})
        with self.assertRaisesRegex(ValueError,'Already selected'):self.runner.select_first88()
        self.assertEqual(self.b.training_calls,0)
    def test_partial_diagnostics_stop_before_replaying_any_return(self):
        self.runner.prepare();r.dump(self.out/'INDEPENDENT_PREPARE.json',self.review());self.runner.select_first88()
        r.dump(self.out/(r.NAMES[0]+'_same_close_DAILY.json'),[])
        with self.assertRaisesRegex(ValueError,'[Dd]iagnos|[Ss]tart|[Aa]lready'):
            self.runner.diagnose_later34()
        self.assertEqual(self.b.diagnostic_calls,0)
    def test_existing_diagnostic_marker_blocks_and_is_preserved(self):
        self.runner.prepare();r.dump(self.out/'INDEPENDENT_PREPARE.json',self.review());self.runner.select_first88()
        marker=self.out/'DIAGNOSE_STARTED.json'
        marker.write_text('{"synthetic_preexisting_marker":true}')
        original=marker.read_bytes()
        with self.assertRaisesRegex(ValueError,'Diagnostic stage already started'):
            self.runner.diagnose_later34()
        self.assertEqual(marker.read_bytes(),original)
        self.assertEqual(self.b.diagnostic_calls,0)
    def test_diagnostic_marker_binds_plan_and_freeze_before_backend(self):
        self.runner.prepare();r.dump(self.out/'INDEPENDENT_PREPARE.json',self.review());self.runner.select_first88()
        def fail_after_marker(*args):
            marker=r.read(self.out/'DIAGNOSE_STARTED.json')
            self.assertEqual(set(marker),{'plan_sha256','frozen_sha256','started_utc'})
            self.assertEqual(marker['plan_sha256'],self.runner.plan_sha)
            self.assertEqual(marker['plan_sha256'],r.sha(self.plan_path))
            self.assertEqual(marker['frozen_sha256'],r.sha(self.out/'FROZEN.json'))
            self.assertIsNotNone(pd.Timestamp(marker['started_utc']).tz)
            self.assertEqual(pd.Timestamp(marker['started_utc']).utcoffset().total_seconds(),0.)
            raise RuntimeError('synthetic interrupted replay')
        with patch.object(self.b,'diagnostics',side_effect=fail_after_marker) as replay:
            with self.assertRaisesRegex(RuntimeError,'synthetic interrupted replay'):
                self.runner.diagnose_later34()
            original=(self.out/'DIAGNOSE_STARTED.json').read_bytes()
            with self.assertRaisesRegex(ValueError,'Diagnostic stage already started'):
                self.runner.diagnose_later34()
            self.assertEqual(replay.call_count,1)
            self.assertEqual((self.out/'DIAGNOSE_STARTED.json').read_bytes(),original)
    def test_event_dependency_tamper_stops_before_backend_load(self):
        self.paths['event'].write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError,'Changed dependency'):self.make_runner()
        self.assertEqual(self.b.loads,0)
    def test_event_dependency_must_also_be_in_plan_hashes(self):
        self.plan['hashes'].pop(str(self.paths['event']))
        self.plan_path.write_text(json.dumps(self.plan))
        with self.assertRaisesRegex(ValueError,'Unbound event_input_hashes'):self.make_runner()
        self.assertEqual(self.b.loads,0)
    def test_input_reader_rejects_unbound_local_file(self):
        unbound=self.root/'unbound.json';unbound.write_text('{}')
        reader=r.InputReader(self.runner)
        for call in (reader.path,reader.sha,reader.json):
            with self.assertRaisesRegex(ValueError,'Unbound dependency'):call(unbound)
    def test_prepared_original_arrays_checked_even_if_cache_rehashed(self):
        self.runner.prepare()
        with np.load(self.out/'DATA.npz',allow_pickle=False) as z:
            changed={name:z[name].copy() for name in z.files}
        changed['cached0'][0,0]+=1
        np.savez_compressed(self.out/'DATA.npz',**changed)
        (self.out/'CACHE.json').write_text(json.dumps({name:r.sha(self.out/name) for name in r.CACHE_FILES}))
        with self.assertRaisesRegex(ValueError,'Original arrays changed'):self.runner.loaded()
    def test_no_review_stops_before_returns(self):
        self.runner.prepare()
        with self.assertRaises(FileNotFoundError):self.runner.select_first88()
        self.assertEqual(self.b.training_calls,0)
    def test_incomplete_or_wrong_review_stops_before_returns(self):
        self.runner.prepare();original=self.review()
        for key,value in (('stage','outcome'),('source_hashes',{}),('artifact_hashes',{}),
            (r.CHECKS[1],False),('producer_modules_imported',True),('monthly_label_models_fit',True),('returns_or_nav_computed',True)):
            bad=deepcopy(original);bad[key]=value;(self.out/'INDEPENDENT_PREPARE.json').write_text(json.dumps(bad))
            with self.subTest(key=key),self.assertRaises(ValueError):self.runner.select_first88()
        self.assertEqual(self.b.training_calls,0)
    def test_cache_tamper_stops_before_returns(self):
        self.runner.prepare();r.dump(self.out/'INDEPENDENT_PREPARE.json',self.review())
        (self.out/'FEATURE_AUDIT.json').write_text('{}')
        with self.assertRaises(ValueError):self.runner.select_first88()
        self.assertEqual(self.b.training_calls,0)
    def test_changed_plan_or_draft_stops_before_load(self):
        self.plan_path.write_text('{}')
        with self.assertRaises(ValueError):self.runner.prepare()
        self.assertEqual(self.b.loads,0)
        self.plan['status']='DRAFT';self.plan_path.write_text(json.dumps(self.plan))
        with self.assertRaises(ValueError):self.make_runner()
    def test_frozen_selection_or_set_tamper_stops_diagnostics(self):
        self.runner.prepare();r.dump(self.out/'INDEPENDENT_PREPARE.json',self.review());self.runner.select_first88()
        path=self.out/'FROZEN.json';original=r.read(path)
        for key in ('selection','hashes'):
            bad=deepcopy(original)
            if key=='selection':bad[key]['primary']=r.NAMES[1]
            else:bad[key].pop('INDEPENDENT_PREPARE.json')
            path.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):self.runner.diagnose_later34()
        self.assertEqual(self.b.diagnostic_calls,0)


class ActualHookSyntheticTests(unittest.TestCase):
    def test_training_only_first88_and_three_declared_folds(self):
        m,a,t,px,s=synthetic();b=r.Backend.__new__(r.Backend);calls=[]
        def paths(px,m,n):calls.append(('paths',n));return None,None,np.arange(89),None
        def fast(c,*args):calls.append(('choices',len(c)));return np.full(88,.01)
        def measure(rr,m,bounds,n=None):
            calls.append(('measure',n));return dict(training=dict(cagr=.2,sharpe=1.,max_drawdown=-.2),early=dict(cagr=.2),recent=dict(cagr=.2))
        b.local=SimpleNamespace(rp=SimpleNamespace(paths=paths,fast_returns=fast,stats_rr=lambda rr,start,end:dict(count=len(rr))),
            old=SimpleNamespace(session=SimpleNamespace(measure=measure)))
        orders={n:[dict(symbols=[names[0]]) for names in m['symbols']] for n in r.NAMES}
        rows,folds=b.training(orders,m,px)
        self.assertEqual(calls[0],('paths',88));self.assertEqual([v for k,v in calls if k=='choices'],[88]*4)
        self.assertEqual([(x['n'],x['end']) for x in folds['folds']],[(36,60),(60,76),(76,88)])
        self.assertEqual(folds['metrics']['count'],52)
    def test_diagnostics_twelve_cost_paths_and_partial_years(self):
        m,a,t,px,s=synthetic();b=r.Backend.__new__(r.Backend);calls=[];written={}
        def replay(px,op,orders,**kwargs):
            calls.append(kwargs);return dict(annual_returns=[dict(year=2016),dict(year=2026)]),pd.Series([1.,1.1],index=px.index[:2]),[],[]
        b.local=SimpleNamespace(ex=SimpleNamespace(e=SimpleNamespace(engine=SimpleNamespace(replay=replay))),
            arch=SimpleNamespace(old=SimpleNamespace(prior=SimpleNamespace(old=SimpleNamespace(previous=SimpleNamespace(cuts=lambda eq,m:{}))))))
        orders={n:[dict(symbols=names[:3]) for names in m['symbols']] for n in r.NAMES}
        results,months=b.diagnostics(orders,m,px,px,lambda k,v:written.update({k:v}))
        self.assertEqual([(c['execution'],c['cost']) for c in calls],[('close',0.),('open',0.),('open',.0005)]*4)
        self.assertEqual(len(written),36);self.assertEqual(len(months),122)
        self.assertTrue(all(y['partial'] for x in results.values() for z in x.values() for y in z['metrics']['annual_returns']))


if __name__=='__main__':unittest.main()
