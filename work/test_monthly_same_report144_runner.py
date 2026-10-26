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
import monthly_same_report144_runner as r

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
    def training(self,orders,m,px,op):
        self.training_calls+=1;self.training_length=len(orders[r.NAMES[0]][:88])
        rows=trials();rows[2]['training']['cagr']=.21
        return rows,deepcopy(rows),dict(folds=[dict(n=n,end=end,selection=r.select(rows,rows))
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
        hand=np.tile(np.linspace(-.49,.5,32),(122,1));full=np.full((122,49),np.nan);full[:,:32]=hand
        self.a.update(gross_profit_growth_sign=full.copy(),gross_profitability_improvement_sign=full.copy(),common_available=np.isfinite(full))
    def call(self,name=None,n=1):return r.choose(name or r.NAMES[2],self.a,self.m,n,self.b,POLICY)
    def test_exact_support_increment_one_episode(self):
        _,score=self.call();support=(self.a['baseline'][0]-self.a['core'][0])/(1-self.a['core_weight'][0])
        delta=.025*(np.clip(support[:32]+self.a['gross_profitability_improvement_sign'][0,:32],-.5,.5)-support[:32])
        expected=self.a['baseline'][0].copy();expected[:32]+=delta
        np.testing.assert_array_equal(score[0],expected);self.assertEqual(self.b.episodes,1)
        self.assertLessEqual(float(np.max(np.abs(delta))),.0125+1e-12)
        np.testing.assert_array_equal(self.b.last_original['core'],self.a['core'][:1])
    def test_both_arms_identical_hands_and_etf_unchanged(self):
        c,s=self.call();d,t=self.call(r.NAMES[1]);self.assertEqual(c,d);self.assertEqual(s.tobytes(),t.tobytes())
        np.testing.assert_array_equal(s[0,32:],self.a['baseline'][0,32:])
    def test_zero_and_missing_exact_baseline(self):
        expected=self.call(r.NAMES[0])
        for value in (0.,np.nan):
            for key in ('gross_profit_growth_sign','gross_profitability_improvement_sign'):self.a[key][:,:32]=value
            for name in r.NAMES[1:]:
                c,s=self.call(name);self.assertEqual(c,expected[0]);self.assertEqual(s.tobytes(),expected[1].tobytes())
    def test_future_poison_ignored(self):
        expected=self.call()
        for key in ('baseline','core','effects','core_weight'):
            self.a[key][1:]=np.nan
        for key in ('gross_profit_growth_sign','gross_profitability_improvement_sign'):
            self.a[key]=self.a[key].astype(object);self.a[key][1:]='future'
        c,s=self.call();self.assertEqual(c,expected[0]);self.assertEqual(s.tobytes(),expected[1].tobytes())
    def test_bad_hand_and_etf_rank_rejected(self):
        for value in (np.inf,.51,True,'0.1'):
            a=self.a['gross_profitability_improvement_sign'].astype(object);a[0,0]=value
            with patch.dict(self.a,gross_profitability_improvement_sign=a),self.subTest(value=value),self.assertRaises(ValueError):self.call()
        self.a['gross_profitability_improvement_sign'][0,32]=0.
        with self.assertRaises(ValueError):self.call()
    def test_all_panels_prefix_and_nonmutation(self):
        old={k:v.tobytes() for k,v in self.a.items()};orders,scores=r.panels(self.a,self.m,self.b,POLICY)
        self.assertEqual([len(v) for v in orders.values()],[122]*3)
        self.assertEqual(old,{k:v.tobytes() for k,v in self.a.items()})
        self.assertEqual([v.shape for v in scores.values()],[(122,49)]*3)


class SelectionTests(unittest.TestCase):
    def test_only_primary_can_win(self):
        rows=trials();op=trials();rows[1]['training']['cagr']=.9
        op[2]['training']['cagr']=.21
        self.assertEqual(r.select(rows,op)['primary'],r.NAMES[0])
        rows[2]['training']['cagr']=.91
        self.assertEqual(r.select(rows,op)['primary'],r.NAMES[2])
    def test_strict_close_cagr_against_both(self):
        for ref in (0,1):
            rows=trials();op=trials();op[2]['training']['cagr']=.22
            rows[2]['training']['cagr']=.21;rows[ref]['training']['cagr']=.21-1e-12
            self.assertFalse(r.select(rows,op)['admitted'])
            rows[ref]['training']['cagr']=.21-2e-12;self.assertTrue(r.select(rows,op)['admitted'])
    def test_open_must_strictly_beat_control_but_can_equal_incumbent(self):
        rows=trials();rows[2]['training']['cagr']=.21;op=trials();op[1]['training']['cagr']=.19
        self.assertTrue(r.select(rows,op)['admitted'])
        op[2]['training']['cagr']=.19;self.assertFalse(r.select(rows,op)['admitted'])
        op[2]['training']['cagr']=.2-2e-12;self.assertFalse(r.select(rows,op)['admitted'])
    def test_each_guard_both_executions_and_references(self):
        for execution in (0,1):
            for ref in (0,1):
                for group,key,delta in (('training','sharpe',.051),('training','max_drawdown',.031),('early','cagr',.026),('recent','cagr',.026)):
                    groups=[trials(),trials()]
                    for rows in groups:rows[2]['training']['cagr']=.21
                    groups[execution][ref][group][key]+=delta
                    with self.subTest(execution=execution,ref=ref,key=key):self.assertFalse(r.select(*groups)['admitted'])
    def test_invalid_metrics_and_arm_order(self):
        for execution in (0,1):
            for value in (None,True,'1',float('nan'),float('inf')):
                groups=[trials(),trials()]
                for rows in groups:rows[2]['training']['cagr']=.21
                groups[execution][2]['training']['cagr']=value
                self.assertFalse(r.select(*groups)['admitted'])
        with self.assertRaises(ValueError):r.select(trials()[::-1],trials())


class PhaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve();self.out=self.root/'outputs/monthly_same_report144';self.out.mkdir(parents=True)
        self.m,self.a,self.t,px,sessions=synthetic();self.b=FakeBackend(self.m,self.a,px)
        c,s=self.b.baseline(self.a,self.m,122,POLICY)
        values=dict(proposal={'round':144},meta=self.m,policy=POLICY,
            original_orders=self.b.make_orders(c,s,self.m),prices={'synthetic':True},
            sessions=dict(sessions=sessions),protected_manifest={},reviewer={'synthetic':True},raw={'source':'synthetic'},
            admitted_dataset=dict(schema_version=1,method=r.factor.METHOD,events=[]),
            source_plan={'source':True},source_policy={'policy':True})
        self.paths={}
        for key,value in values.items():
            self.paths[key]=self.root/(key+'.json');r.dump(self.paths[key],value)
        receipt=dict(stage='conditional_source_admission',passed=True,
            financial_calculations=0,model_fits=0,ranks_computed=0,returns_computed=0,network_requests=0,
            data_sha256=r.sha(self.paths['admitted_dataset']),plan_sha256=r.sha(self.paths['source_plan']),
            policy_sha256=r.sha(self.paths['source_policy']),
            source_hashes={str(self.paths[k]):r.sha(self.paths[k]) for k in ('raw','sessions')},
            source_code_hashes={str(self.paths['reviewer']):r.sha(self.paths['reviewer'])},
            checks={key:True for key in r.ADMISSION_CHECKS},first_public_version_verified=False,
            complete_correction_history_verified=False,certified_daily_security_master=False)
        self.paths['source_admission']=self.root/'source_admission.json';r.dump(self.paths['source_admission'],receipt)
        self.plan=dict(round=144,status='REGISTERED',arms=list(r.NAMES),primary_challenger=r.NAMES[2],
            output_directory=str(self.out),runtime=dict(python=r.sys.version,numpy=np.__version__,pandas=pd.__version__),
            tolerances=r.TOLERANCES,inputs={k:dict(path=str(self.paths[k]),sha256=r.sha(self.paths[k])) for k in r.INPUTS},
            hashes={str(p):r.sha(p) for p in self.paths.values()},
            raw_source_hashes={str(self.paths[k]):r.sha(self.paths[k]) for k in ('raw','sessions')},
            fundamental_source_hashes={str(self.paths[k]):r.sha(self.paths[k]) for k in ('raw','sessions','reviewer','admitted_dataset','source_admission','source_plan','source_policy')},
            independent_prepare_source_hashes={str(self.paths['reviewer']):r.sha(self.paths['reviewer'])},
            independent_outcome_source_hashes={str(self.paths['reviewer']):r.sha(self.paths['reviewer'])})
        self.plan_path=self.out/'PLAN.json';r.dump(self.plan_path,self.plan)
        for name,value in (('PROPOSAL_SHA',r.sha(self.paths['proposal'])),('PROTECTED_SHA',r.sha(self.paths['protected_manifest']))):
            p=patch.object(r,name,value);p.start();self.addCleanup(p.stop)
        p=patch.object(r,'source_files',return_value=self.plan['independent_prepare_source_hashes']);p.start();self.addCleanup(p.stop)
        def features(meta,dataset):
            arrays={k:np.full((122,49),np.nan) for k in r.EXTRA[:-1]};arrays['common_available']=np.zeros((122,49),bool)
            cells=[dict(signal_index=i,pool_index=j,signal_date=day,symbol=s,available=False,
                reason='synthetic_unavailable',gross_profit_growth_sign=None,gross_profitability_improvement_sign=None)
                for i,(day,names) in enumerate(zip(meta['dates'],meta['symbols'])) for j,s in enumerate(names)]
            return dict(arrays=arrays,audit=dict(cells=cells,new_model_fits=0))
        p=patch.object(r.factor,'build_features',side_effect=features);self.factor_call=p.start();self.addCleanup(p.stop)
        self.runner=self.make_runner()
    def make_runner(self):return r.Runner(self.root,self.plan_path,r.sha(self.plan_path),lambda root,runner:self.b)
    def review(self):
        artifacts=dict(self.runner.cache());artifacts.update({str(self.out/x):r.sha(self.out/x) for x in ('PLAN.json','CACHE.json')})
        return dict(passed=True,stage='prepare',producer_modules_imported=False,returns_or_nav_computed=False,
            monthly_label_models_fit=0,source_hashes=self.plan['independent_prepare_source_hashes'],artifact_hashes=artifacts,
            plan_sha256=r.sha(self.plan_path),cache_sha256=r.sha(self.out/'CACHE.json'),
            orders_sha256=r.sha(self.out/'PRETRAIN_ORDERS.json'),**{k:True for k in r.CHECKS})
    def repin(self,key):
        path=self.paths[key];h=r.sha(path)
        self.plan['hashes'][str(path)]=h
        if key in self.plan['inputs']:self.plan['inputs'][key]['sha256']=h
        self.plan_path.write_text(json.dumps(self.plan))

    def test_source_admission_precedes_backend_and_factor(self):
        original=r.read(self.paths['source_admission'])
        for key,value in (('passed',False),('stage','migration'),('data_sha256','0'*64),
                          ('financial_calculations',True),('financial_calculations',1),
                          ('source_hashes',{}),('source_code_hashes',{}),('checks',{}),
                          ('first_public_version_verified',True),('plan_sha256',None)):
            bad=deepcopy(original);bad[key]=value
            self.paths['source_admission'].write_text(json.dumps(bad));self.repin('source_admission')
            with self.subTest(key=key),self.assertRaises(ValueError):self.make_runner()
        self.assertEqual(self.b.loads,0);self.factor_call.assert_not_called()

    def test_inconsistent_source_closure_rejected(self):
        self.plan['fundamental_source_hashes']=self.plan['independent_prepare_source_hashes']
        self.plan_path.write_text(json.dumps(self.plan))
        with self.assertRaisesRegex(ValueError,'closure'):self.make_runner()
        self.assertEqual(self.b.loads,0);self.factor_call.assert_not_called()

    def test_source_mutation_after_initialization_stops_prepare(self):
        self.paths['raw'].write_text('changed evidence')
        with self.assertRaises(ValueError):self.runner.prepare()
        self.assertEqual(self.b.loads,0);self.factor_call.assert_not_called()

    def test_factor_output_invalid_mask_and_sign_stop_before_orders(self):
        original=self.factor_call.side_effect
        for mode in ('shape','mask','continuous','infinity','etf','fit','cell'):
            def invalid(meta,dataset):
                result=original(meta,dataset);a=result['arrays']
                if mode=='shape':a['gross_profit_growth_sign']=np.zeros((1,49))
                elif mode=='mask':a['gross_profit_growth_sign'][0,0]=.5
                elif mode in ('continuous','infinity','etf'):
                    j=32 if mode=='etf' else 0;a['common_available'][0,j]=True
                    for key in r.HAND_KEYS:a[key][0,j]=.1 if mode=='continuous' else float('inf') if mode=='infinity' else .5
                elif mode=='fit':result['audit']['new_model_fits']=1
                elif mode=='cell':result['audit']['cells'][0]['symbol']='WRONG'
                return result
            self.factor_call.side_effect=invalid
            with self.subTest(mode=mode),self.assertRaises(ValueError):self.runner.prepare()
            self.assertFalse((self.out/'DATA.npz').exists())
        self.assertEqual(self.b.training_calls,0)

    def test_all_phases_seven_caches_three_arms_and_state_attribution(self):
        self.runner.prepare();self.assertEqual(self.b.training_calls,0)
        self.assertEqual(set(r.read(self.out/'CACHE.json')),set(r.CACHE_FILES))
        self.assertEqual(len(r.read(self.out/'FEATURE_AUDIT.json')['cells']),5978)
        with np.load(self.out/'DATA.npz',allow_pickle=False) as z:self.assertEqual(set(z.files),set(self.a)|set(r.EXTRA))
        r.dump(self.out/'INDEPENDENT_PREPARE.json',self.review());self.runner.select_first88()
        self.assertEqual(self.b.training_length,88);self.assertEqual(self.b.diagnostic_calls,0)
        self.runner.diagnose_later34();self.assertEqual(self.b.diagnostic_calls,1)
        results=r.read(self.out/'RESULTS.json');self.assertEqual(sum(map(len,results.values())),9)
        months=r.read(self.out/'MONTH_COMPARISON.json');self.assertEqual(sum(m['split']=='later34' for m in months),34)
        attrs=r.read(self.out/'ATTRIBUTION.json')['comparisons']
        self.assertEqual(set(attrs),{'profitability_vs_incumbent','growth_vs_incumbent','profitability_vs_growth'})
        for c in attrs.values():
            self.assertEqual(c['changes'][0]['profitability'],months[0]['profitability']);self.assertEqual(c['changes'][0]['guidance_events'],[])
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
    def test_open_training_uses_prefix_cost_and_declared_split(self):
        m,a,t,px,s=synthetic();b=r.Backend.__new__(r.Backend);calls=[];intervals=[]
        def replay(px,op,orders,**kwargs):
            calls.append((len(orders),kwargs))
            eq=pd.Series(np.linspace(1.,2.,len(px)),index=px.index)
            return {},eq,[],[]
        def stats(rr,lo,hi):
            intervals.append((lo,hi));return dict(cagr=.2,sharpe=1.,max_drawdown=-.2)
        b.local=SimpleNamespace(ex=SimpleNamespace(e=SimpleNamespace(engine=SimpleNamespace(replay=replay))),rp=SimpleNamespace(stats_rr=stats))
        orders={name:[dict(symbols=names[:3]) for names in m['symbols']] for name in r.NAMES}
        for n in (36,60,76,88):
            calls.clear();intervals.clear();rows=b.open_training(orders,m,px,px,n)
            self.assertEqual([count for count,kwargs in calls],[n]*3)
            self.assertTrue(all(kwargs['execution']=='open' and kwargs['cost']==.0005 for count,kwargs in calls))
            split=53 if n==88 else n//2
            self.assertEqual(intervals[:3],[(m['dates'][0],m['ends'][n-1]),(m['dates'][0],m['ends'][split-1]),(m['dates'][split],m['ends'][n-1])])
            self.assertEqual([row['name'] for row in rows],list(r.NAMES))
    def test_training_only_first88_and_three_declared_folds(self):
        m,a,t,px,s=synthetic();b=r.Backend.__new__(r.Backend);calls=[]
        def paths(px,m,n):calls.append(('paths',n));return None,None,np.arange(89),None
        def fast(c,*args):calls.append(('choices',len(c)));return np.full(88,.01)
        def measure(rr,m,bounds,n=None):
            calls.append(('measure',n));return dict(training=dict(cagr=.2,sharpe=1.,max_drawdown=-.2),early=dict(cagr=.2),recent=dict(cagr=.2))
        b.local=SimpleNamespace(rp=SimpleNamespace(paths=paths,fast_returns=fast,stats_rr=lambda rr,start,end:dict(count=len(rr))),
            old=SimpleNamespace(session=SimpleNamespace(measure=measure)))
        orders={n:[dict(symbols=[names[0]]) for names in m['symbols']] for n in r.NAMES}
        b.open_training=lambda orders,m,px,op,n: [dict(name=name,training=dict(cagr=.2,sharpe=1.,max_drawdown=-.2),early=dict(cagr=.2),recent=dict(cagr=.2)) for name in r.NAMES]
        rows,open_rows,folds=b.training(orders,m,px,px)
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
