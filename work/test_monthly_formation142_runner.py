"""Synthetic-only integration of frozen core replacement and phase gates."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json
import tempfile
import unittest
import numpy as np
import pandas as pd
import monthly_formation142_runner as r

ROOT=Path(__file__).resolve().parents[1]
ADAPTER=r.module('test_tail137_path',ROOT/'outputs/monthly_path70_research/incumbent_adapter.py')
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
        rows=trials();rows[2]['training']['cagr']=.21
        return rows,dict(folds=[dict(n=n,end=end,selection=r.select(rows))
            for n,end in ((36,60),(60,76),(76,88))],pristine_oos=False)
    def diagnostics(self,orders,m,px,op,writer):
        self.diagnostic_calls+=1
        results={name:{key:dict(metrics={},slices={'training2016_2023':dict(cagr=.21 if j==2 else .2)})
            for key in ('same_close','next_open_zero','next_open_cost')} for j,name in enumerate(r.NAMES)}
        months=[dict(date=day,end=m['ends'][i],split='early53' if i<53 else 'recent35' if i<88 else 'later34')
            for i,day in enumerate(m['dates'])]
        return results,months


class PlacementTests(unittest.TestCase):
    def setUp(self):
        self.m,self.a,t,px,_=synthetic();self.b=FakeBackend(self.m,self.a,px)
        self.a.update(formation_rank=np.tile(np.linspace(-.45,.5,49),(122,12,1)),
            formation_count=np.full(122,12,dtype=np.int64),formation_ready=np.ones(122,bool),
            formation_valid=np.ones((122,49),bool))
    def test_baseline_only_calls_episode_once_for_three_arms(self):
        c,s,details=r.choices_all(self.a,self.m,1,self.b,POLICY)
        self.assertEqual(self.b.episodes,1)
        self.assertEqual(set(c),set(r.NAMES))
        for name in r.NAMES[1:]:
            np.testing.assert_array_equal(s[name],self.a['baseline'][:1])
            self.assertFalse(details[name][0]['fallback'])
    def test_missing_exact_order_and_final_score_fallback(self):
        self.a['formation_ready'][:]=False
        c,s,d=r.choices_all(self.a,self.m,2,self.b,POLICY)
        for name in r.NAMES[1:]:
            self.assertEqual(c[name],c[r.NAMES[0]])
            self.assertEqual(s[name].tobytes(),s[r.NAMES[0]].tobytes())
    def test_unchanged_core_exact_order_and_final_score_fallback(self):
        self.a['formation_rank'][:]=self.a['core'][:,None,:]/self.a['core_weight'][:,None,None]
        c,s,d=r.choices_all(self.a,self.m,2,self.b,POLICY)
        for name in r.NAMES[1:]:
            self.assertTrue(all(x['fallback'] for x in d[name]))
            self.assertEqual(c[name],c[r.NAMES[0]])
            self.assertEqual(s[name].tobytes(),s[r.NAMES[0]].tobytes())
    def test_future_suffix_not_read(self):
        c,s,_=r.choices_all(self.a,self.m,1,self.b,POLICY)
        for key in ('baseline','core','core_weight','effects'):
            self.a[key][1:]=np.nan
        self.a['formation_rank']=self.a['formation_rank'].astype(object)
        self.a['formation_rank'][1:]='future'
        d,t,_=r.choices_all(self.a,self.m,1,self.b,POLICY)
        self.assertEqual(c,d)
        for name in r.NAMES:self.assertEqual(s[name].tobytes(),t[name].tobytes())
    def test_input_nonmutation_and_prefix(self):
        before={k:v.tobytes() for k,v in self.a.items()}
        c,s,_=r.choices_all(self.a,self.m,3,self.b,POLICY)
        for n in range(4):
            d,t,_=r.choices_all(self.a,self.m,n,self.b,POLICY)
            for name in r.NAMES:
                self.assertEqual(d[name],c[name][:n]);self.assertEqual(t[name].tobytes(),s[name][:n].tobytes())
        self.assertEqual(before,{k:v.tobytes() for k,v in self.a.items()})
    def test_candidate_with_invalid_original_core_falls_back(self):
        self.a['formation_valid'][0,48]=False
        c,s,d=r.choices_all(self.a,self.m,1,self.b,POLICY)
        for name in r.NAMES[1:]:
            self.assertEqual(d[name][0]['reason'],'candidate_history_unavailable')
            self.assertEqual(c[name],c[r.NAMES[0]])
            self.assertEqual(s[name].tobytes(),s[r.NAMES[0]].tobytes())
    def test_nonnumeric_or_boolean_scenario_prefix_fails(self):
        for value in ('0.1',True):
            a=dict(self.a,formation_rank=self.a['formation_rank'].astype(object))
            a['formation_rank'][0,0,0]=value
            with self.assertRaises(ValueError):r.choices_all(a,self.m,1,self.b,POLICY)
    def test_invalid_prefix(self):
        for n in (-1,123,True,.5):
            with self.assertRaises(ValueError):r.choices_all(self.a,self.m,n,self.b,POLICY)


class FeatureIntegrationTests(unittest.TestCase):
    def setUp(self):
        axis=pd.bdate_range('2015-01-01',periods=400)
        rng=np.random.default_rng(142)
        self.px=pd.DataFrame(100*np.exp(np.cumsum(rng.normal(.0005,.01,(400,50)),axis=0)),index=axis,columns=['SPY']+['S'+str(i) for i in range(49)])
        self.m=dict(dates=[str(axis[-1].date())],symbols=[list(self.px.columns[1:])])
        self.sessions=[str(x.date()) for x in axis]
        stock=self.px.iloc[:,1:];sigma=stock.pct_change(fill_method=None).rolling(63).std().iloc[-1]
        calm=bool(self.px.SPY.pct_change(fill_method=None).rolling(63).std().iloc[-1]*np.sqrt(252)<.1)
        cw=.5 if calm else .52
        core=(stock.iloc[-43]/stock.iloc[-253 if calm else -295]-1)
        if not calm:core=core/sigma
        self.a=dict(state_calm10=np.array([calm]),core_weight=np.array([cw]),core=(cw*(core.rank(pct=True).fillna(.5)-.5)).to_numpy()[None,:])
    def test_exact_core_mask_and_historical_window(self):
        result=r.build_features({'adj':self.px},self.m,self.a,self.sessions)
        self.assertTrue(result['arrays']['formation_ready'][0]);self.assertTrue(result['arrays']['formation_valid'].all())
        self.assertEqual(result['arrays']['formation_count'][0],12)
        self.assertEqual(result['audit']['months'][0]['blocks'][-1]['end_price_index'],357)
    def test_rank_parity_failure_stops(self):
        self.a['core'][0,0]+=.01
        with self.assertRaisesRegex(ValueError,'weighted core parity'):r.build_features({'adj':self.px},self.m,self.a,self.sessions)
    def test_calm_parity_failure_stops(self):
        self.a['state_calm10'][0]=not self.a['state_calm10'][0]
        with self.assertRaisesRegex(ValueError,'calm state parity'):r.build_features({'adj':self.px},self.m,self.a,self.sessions)
    def test_calendar_gap_stops(self):
        with self.assertRaisesRegex(ValueError,'official formation'):r.build_features({'adj':self.px.drop(self.px.index[200])},self.m,self.a,self.sessions)
    def test_future_price_suffix_does_not_change_features(self):
        expected=r.build_features({'adj':self.px},self.m,self.a,self.sessions)
        extra=pd.DataFrame(np.nan,index=pd.bdate_range(self.px.index[-1]+pd.Timedelta(days=1),periods=10),columns=self.px.columns)
        got=r.build_features({'adj':pd.concat([self.px,extra])},self.m,self.a,self.sessions)
        for key in r.EXTRA:np.testing.assert_array_equal(got['arrays'][key],expected['arrays'][key])


class SelectionTests(unittest.TestCase):
    def test_only_directional_can_win(self):
        rows=trials();rows[1]['training']['cagr']=.9
        self.assertEqual(r.select(rows)['primary'],r.NAMES[0])
        rows[2]['training']['cagr']=.91
        self.assertEqual(r.select(rows)['primary'],r.NAMES[2])
    def test_strict_cagr_against_each_reference(self):
        for ref in (0,1):
            rows=trials();rows[2]['training']['cagr']=.21;rows[ref]['training']['cagr']=.21-1e-12
            self.assertFalse(r.select(rows)['admitted'])
            rows[ref]['training']['cagr']=.21-2e-12;self.assertTrue(r.select(rows)['admitted'])
    def test_each_guard_against_both(self):
        for ref in (0,1):
            for group,key,delta in (('training','sharpe',.051),('training','max_drawdown',.031),('early','cagr',.026),('recent','cagr',.026)):
                rows=trials();rows[2]['training']['cagr']=.21;rows[ref][group][key]+=delta
                with self.subTest(ref=ref,key=key):self.assertFalse(r.select(rows)['admitted'])
    def test_invalid_metrics_and_arm_order(self):
        for x in (None,True,'1',float('nan'),float('inf')):
            rows=trials();rows[2]['training']['cagr']=x;self.assertFalse(r.select(rows)['admitted'])
        with self.assertRaises(ValueError):r.select(trials()[::-1])


class PhaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve();self.out=self.root/'outputs/monthly_formation142';self.out.mkdir(parents=True)
        self.m,self.a,self.t,px,sessions=synthetic();self.b=FakeBackend(self.m,self.a,px)
        c,s=self.b.baseline(self.a,self.m,122,POLICY)
        values=dict(proposal={'round':142},meta=self.m,policy=POLICY,
            original_orders=self.b.make_orders(c,s,self.m),prices={'synthetic':True},
            sessions=dict(sessions=sessions),protected_manifest={},reviewer={'synthetic':True},raw={'source':'synthetic'})
        self.paths={}
        for key,value in values.items():
            self.paths[key]=self.root/(key+'.json');r.dump(self.paths[key],value)
        self.plan=dict(round=142,status='REGISTERED',arms=list(r.NAMES),primary_challenger=r.NAMES[2],
            output_directory=str(self.out),runtime=dict(python=r.sys.version,numpy=np.__version__,pandas=pd.__version__),
            tolerances=r.TOLERANCES,inputs={k:dict(path=str(self.paths[k]),sha256=r.sha(self.paths[k])) for k in r.INPUTS},
            hashes={str(p):r.sha(p) for p in self.paths.values()},
            raw_source_hashes={str(self.paths['raw']):r.sha(self.paths['raw'])},
            independent_prepare_source_hashes={str(self.paths['reviewer']):r.sha(self.paths['reviewer'])})
        self.plan_path=self.out/'PLAN.json';r.dump(self.plan_path,self.plan)
        for name,value in (('PROPOSAL_SHA',r.sha(self.paths['proposal'])),('PROTECTED_SHA',r.sha(self.paths['protected_manifest']))):
            p=patch.object(r,name,value);p.start();self.addCleanup(p.stop)
        p=patch.object(r,'source_files',return_value=self.plan['independent_prepare_source_hashes']);p.start();self.addCleanup(p.stop)
        def features(panel,meta,old,sessions):
            arrays=dict(formation_rank=np.full((122,12,49),np.nan),formation_count=np.zeros(122,dtype=np.int64),
                formation_ready=np.zeros(122,bool),formation_valid=np.zeros((122,49),bool))
            months=[dict(signal_index=i,date=day,calm=True,horizon=252,ready=False,reason='synthetic_missing',blocks=[],original_finite_mask=[False]*49,selectors={}) for i,day in enumerate(meta['dates'])]
            return dict(arrays=arrays,audit=dict(months=months,new_model_fits=0))
        p=patch.object(r,'build_features',side_effect=features);p.start();self.addCleanup(p.stop)
        self.runner=self.make_runner()
    def make_runner(self):return r.Runner(self.root,self.plan_path,r.sha(self.plan_path),lambda root,runner:self.b)
    def review(self):
        artifacts=dict(self.runner.cache());artifacts.update({str(self.out/x):r.sha(self.out/x) for x in ('PLAN.json','CACHE.json')})
        return dict(passed=True,stage='prepare',producer_modules_imported=False,returns_or_nav_computed=False,
            monthly_label_models_fit=0,source_hashes=self.plan['independent_prepare_source_hashes'],artifact_hashes=artifacts,
            plan_sha256=r.sha(self.plan_path),cache_sha256=r.sha(self.out/'CACHE.json'),
            orders_sha256=r.sha(self.out/'PRETRAIN_ORDERS.json'),**{k:True for k in r.CHECKS})
    def test_all_phases_seven_caches_three_arms_and_state_attribution(self):
        self.runner.prepare();self.assertEqual(self.b.training_calls,0)
        self.assertEqual(set(r.read(self.out/'CACHE.json')),set(r.CACHE_FILES))
        self.assertEqual(len(r.read(self.out/'FEATURE_AUDIT.json')['months']),122)
        with np.load(self.out/'DATA.npz',allow_pickle=False) as z:self.assertEqual(set(z.files),set(self.a)|set(r.EXTRA))
        r.dump(self.out/'INDEPENDENT_PREPARE.json',self.review());self.runner.select_first88()
        self.assertEqual(self.b.training_length,88);self.assertEqual(self.b.diagnostic_calls,0)
        self.runner.diagnose_later34();self.assertEqual(self.b.diagnostic_calls,1)
        results=r.read(self.out/'RESULTS.json');self.assertEqual(sum(map(len,results.values())),9)
        months=r.read(self.out/'MONTH_COMPARISON.json');self.assertEqual(sum(m['split']=='later34' for m in months),34)
        attrs=r.read(self.out/'ATTRIBUTION.json')['comparisons']
        self.assertEqual(set(attrs),{'worst_vs_incumbent','mean_vs_incumbent','worst_vs_mean'})
        for c in attrs.values():
            self.assertEqual(c['changes'][0]['formation_scenario'],months[0]['formation_scenario']);self.assertEqual(c['changes'][0]['guidance_events'],[])
        self.assertEqual(r.read(self.out/'VALIDATION.json')['new_model_fits'],0)
        with self.assertRaises(ValueError):self.runner.prepare()
        with self.assertRaises(ValueError):self.runner.select_first88()
    def test_diagnose_marker_precedes_replay_and_is_bound(self):
        self.runner.prepare();r.dump(self.out/'INDEPENDENT_PREPARE.json',self.review());self.runner.select_first88()
        original=self.b.diagnostics
        def inspect(*args):
            marker=r.read(self.out/'DIAGNOSE_STARTED.json')
            self.assertEqual(marker['plan_sha256'],r.sha(self.plan_path))
            self.assertEqual(marker['frozen_sha256'],r.sha(self.out/'FROZEN.json'))
            self.assertIsNotNone(r.datetime.fromisoformat(marker['started_utc']).tzinfo)
            self.assertFalse((self.out/'RESULTS.json').exists())
            return original(*args)
        with patch.object(self.b,'diagnostics',side_effect=inspect):self.runner.diagnose_later34()
        marker_bytes=(self.out/'DIAGNOSE_STARTED.json').read_bytes()
        with self.assertRaises(ValueError):self.runner.diagnose_later34()
        self.assertEqual((self.out/'DIAGNOSE_STARTED.json').read_bytes(),marker_bytes)
        self.assertEqual(self.b.diagnostic_calls,1)
    def test_failed_replay_cannot_silently_restart(self):
        self.runner.prepare();r.dump(self.out/'INDEPENDENT_PREPARE.json',self.review());self.runner.select_first88()
        with patch.object(self.b,'diagnostics',side_effect=RuntimeError('synthetic interrupted replay')) as call:
            with self.assertRaises(RuntimeError):self.runner.diagnose_later34()
            self.assertTrue((self.out/'DIAGNOSE_STARTED.json').exists())
            self.assertFalse((self.out/'RESULTS.json').exists())
            with self.assertRaises(ValueError):self.runner.diagnose_later34()
            self.assertEqual(call.call_count,1)
    def test_partial_artifacts_cannot_start_replay_without_marker(self):
        self.runner.prepare();r.dump(self.out/'INDEPENDENT_PREPARE.json',self.review());self.runner.select_first88()
        path=self.out/(r.NAMES[0]+'_same_close_DAILY.json');r.dump(path,[])
        with self.assertRaises(ValueError):self.runner.diagnose_later34()
        self.assertFalse((self.out/'DIAGNOSE_STARTED.json').exists())
        self.assertEqual(self.b.diagnostic_calls,0)
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
        self.assertEqual(calls[0],('paths',88));self.assertEqual([v for k,v in calls if k=='choices'],[88]*3)
        self.assertEqual([(x['n'],x['end']) for x in folds['folds']],[(36,60),(60,76),(76,88)])
        self.assertEqual(folds['metrics']['count'],52)
    def test_diagnostics_nine_cost_paths_and_partial_years(self):
        m,a,t,px,s=synthetic();b=r.Backend.__new__(r.Backend);calls=[];written={}
        def replay(px,op,orders,**kwargs):
            calls.append(kwargs);return dict(annual_returns=[dict(year=2016),dict(year=2026)]),pd.Series([1.,1.1],index=px.index[:2]),[],[]
        b.local=SimpleNamespace(ex=SimpleNamespace(e=SimpleNamespace(engine=SimpleNamespace(replay=replay))),
            arch=SimpleNamespace(old=SimpleNamespace(prior=SimpleNamespace(old=SimpleNamespace(previous=SimpleNamespace(cuts=lambda eq,m:{}))))))
        orders={n:[dict(symbols=names[:3]) for names in m['symbols']] for n in r.NAMES}
        results,months=b.diagnostics(orders,m,px,px,lambda k,v:written.update({k:v}))
        self.assertEqual([(c['execution'],c['cost']) for c in calls],[('close',0.),('open',0.),('open',.0005)]*3)
        self.assertEqual(len(written),27);self.assertEqual(len(months),122)
        self.assertTrue(all(y['partial'] for x in results.values() for z in x.values() for y in z['metrics']['annual_returns']))


if __name__=='__main__':unittest.main()
