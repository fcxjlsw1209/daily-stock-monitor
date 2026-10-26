"""Synthetic original65 support funding, matched availability, gates and replay tests."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

P=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('beta_prior_driver_test',P/'study.py')
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)


def fixture(months=4):
    dates=['2016-08-17']+[str(x.date()) for x in pd.date_range('2016-08-31',periods=121,freq=pd.offsets.MonthEnd())]
    ends=dates[1:]+['2026-09-24'];etfs=['E%02d'%i for i in range(17)];names=etfs+['S%02d'%i for i in range(32)]
    meta=dict(dates=dates[:months],ends=ends[:months],symbols=[names.copy() for _ in range(months)],etfs=etfs)
    shape=(months,49);base=np.tile(np.linspace(-.01,.01,49),(months,1))
    cw=np.broadcast_to(np.where(np.arange(months)%3==0,.5,.52)[:,None],shape).copy()
    oldrank=np.tile((np.arange(49)+1.)/49-.5,(months,1));available=np.ones(shape,bool);available[:,:17]=False
    a=dict(baseline=base,core=cw*oldrank,cw=cw,eligible=np.ones(shape,bool),available=available)
    for mode,raw in [('ols',np.arange(49)[::-1]),('sector_prior',np.arange(49)),('shrunk',np.roll(np.arange(49),12))]:
        value=np.tile(raw.astype(float),(months,1));value[~available]=np.nan
        a['raw_'+mode]=value;a['rank_'+mode]=pd.DataFrame(value).rank(axis=1,pct=True).to_numpy()-.5
    original=s.rp.make_orders(s.rp.choose(base,a['eligible'],meta['symbols'],3),base,meta)
    local=copy.deepcopy(original)
    for row in local:row['symbols'][1:]=row['symbols'][1:][::-1];row['scores']=[.41,.32,.37]
    return dict(meta=meta,data=a,source_orders={0:original,1:local})


class PrefixOnly:
    def __init__(self,array,limit):self.array,self.limit=array,limit
    def __len__(self):return len(self.array)
    def __getitem__(self,key):
        if not isinstance(key,slice) or key.start is not None or key.stop>self.limit:raise AssertionError('Future access')
        return self.array[key]


class StudyTests(unittest.TestCase):
    def setUp(self):self.b=fixture()

    def test_design_eight_configs_eleven_arrays_and_fixed_budgets(self):
        self.assertEqual(s.sha(P/'DESIGN.json'),s.DESIGN_SHA256)
        self.assertEqual(json.loads((P/'DESIGN.json').read_text())['configs'],s.configs())
        self.assertEqual([c['mode'] for c in s.configs()[2:]],['ols']*2+['sector_prior']*2+['shrunk']*2)
        self.assertEqual([c['budget'] for c in s.configs()[2:]],[.025,.05]*3)
        self.assertEqual(len(s.DATA_KEYS),11);s.validate_data(self.b['data'],(4,49))

    def test_exact_support_formula_all_modes_states_and_budgets(self):
        a=self.b['data'];use=a['available'];core_before=a['core'].copy()
        for cfg in s.configs()[2:]:
            score=s.modified_score(cfg,a,4);budget=cfg['budget']
            expected=a['core'][use]+(1-budget/(1-a['cw'][use]))*(a['baseline'][use]-a['core'][use])+budget*a['rank_'+cfg['mode']][use]
            np.testing.assert_array_equal(score[use],expected)
            self.assertEqual(score[~use].tobytes(),a['baseline'][~use].tobytes())
            np.testing.assert_allclose(score[use]-a['baseline'][use],
                budget*(a['rank_'+cfg['mode']][use]-(a['baseline'][use]-a['core'][use])/(1-a['cw'][use])),atol=1e-16,rtol=0.)
        np.testing.assert_array_equal(a['core'],core_before)
        self.assertEqual(set(a['cw'][:,0]),{.5,.52})

    def test_zero_factor_is_valid_support_shrink_not_missing_identity(self):
        a=self.b['data'];use=a['available'];a['baseline'][use]=.2;a['core'][use]=.1
        a['rank_ols'][use]=0.;score=s.modified_score(s.configs()[2],a,4)
        expected=.1+(1-.025/(1-a['cw'][use]))*.1
        np.testing.assert_array_equal(score[use],expected)
        self.assertTrue((score[use]<a['baseline'][use]).all())
        a['baseline'][use]=-.2;a['core'][use]=-.1;score=s.modified_score(s.configs()[2],a,4)
        self.assertTrue((score[use]>a['baseline'][use]).all())

    def test_missing_cells_and_whole_missing_keep_baseline_bytes(self):
        a=self.b['data'];a['baseline'][0,0]=-0.;a['available'][:]=False
        for mode in s.ARMS:a['raw_'+mode][:]=np.nan;a['rank_'+mode][:]=np.nan
        s.validate_data(a,(4,49))
        original=s.rp.make_orders(s.rp.choose(a['baseline'],a['eligible'],self.b['meta']['symbols'],3),a['baseline'],self.b['meta'])
        for cfg in s.configs()[2:]:
            self.assertEqual(s.modified_score(cfg,a,4).tobytes(),a['baseline'].tobytes())
            self.assertEqual(s.orders_for(cfg,self.b,4),original)

    def test_true_global_selector_uses_funded_score_for_every_seat(self):
        a=self.b['data'];a['core'][:]=0.;a['baseline'][:]=np.linspace(-.001,.001,49)
        cfg=s.configs()[3];score=s.modified_score(cfg,a,4);got=s.orders_for(cfg,self.b,4)
        self.assertTrue(set(got[0]['symbols']).isdisjoint(self.b['source_orders'][0][0]['symbols']))
        for i,row in enumerate(got):
            expected=sorted(range(49),key=lambda j:(-score[i,j],self.b['meta']['symbols'][i][j]))[:3]
            self.assertEqual(row['symbols'],[self.b['meta']['symbols'][i][j] for j in expected])
            self.assertEqual(row['scores'],[float(score[i,j]) for j in expected])

    def test_rank_population_is_not_recomputed_after_eligibility(self):
        a=self.b['data'];before=a['rank_shrunk'].copy();a['eligible'][:,17:35]=False
        captured={};choose=s.rp.choose
        def spy(score,eligible,symbols,retain):
            captured['score']=score.copy();return choose(score,eligible,symbols,retain)
        with patch.object(s.rp,'choose',side_effect=spy):s.orders_for(s.configs()[7],self.b,4)
        np.testing.assert_array_equal(captured['score'],s.modified_score(s.configs()[7],a,4))
        np.testing.assert_array_equal(a['rank_shrunk'],before)

    def test_two_half_cash_and_lexical_score_ties(self):
        a=self.b['data'];a['available'][:]=False;a['baseline'][:]=0.;a['eligible'][:]=False
        a['eligible'][0,[48,0,40,19]]=True;a['eligible'][1,[48,0]]=True;a['eligible'][2,48]=True
        got=s.orders_for(s.configs()[7],self.b,4)
        self.assertEqual([row['symbols'] for row in got],[['E00','S02','S23'],['E00','S31'],[],[]])

    def test_all_prefixes_no_mutation_and_only_own_rank_read(self):
        before=copy.deepcopy(self.b)
        for cfg in s.configs():
            full=s.orders_for(cfg,self.b,4)
            for n in range(5):self.assertEqual(s.orders_for(cfg,self.b,n),full[:n])
        for cfg in s.configs()[2:]:
            b=copy.deepcopy(self.b);expected=s.orders_for(cfg,b,2)
            used={'baseline','core','cw','available','eligible','rank_'+cfg['mode']}
            for key in b['data']:b['data'][key]=PrefixOnly(b['data'][key],2) if key in used else None
            self.assertEqual(s.orders_for(cfg,b,2),expected)
        for key in s.DATA_KEYS:self.assertEqual(self.b['data'][key].tobytes(),before['data'][key].tobytes())
        self.assertEqual(self.b['meta'],before['meta'])

    def test_data_rejects_mask_divergence_bad_dtypes_and_original_changes(self):
        a=self.b['data'];source=dict(baseline=a['baseline'],core=a['core'],eligible=a['eligible'],core_weight=a['cw'][:,:1])
        s.validate_data(a,(4,49),source)
        for key,value in [('available',np.ones((4,49),int)),('raw_ols',np.full((4,49),np.inf)),('rank_sector_prior',np.zeros((4,49))),('rank_shrunk',np.full((4,49),.6)),('cw',np.full((4,49),.6))]:
            b=copy.deepcopy(a);b[key]=value
            with self.assertRaises(AssertionError):s.validate_data(b,(4,49))
        b=copy.deepcopy(a);b['baseline'][0,0]+=1e-16
        with self.assertRaises(AssertionError):s.validate_data(b,(4,49),source)
        meta=fixture(122)['meta'];s.validate_meta(meta,copy.deepcopy(meta))
        meta['symbols'][0][-1]=meta['symbols'][0][0]
        with self.assertRaises(AssertionError):s.validate_meta(meta,meta)

    def test_bad_configs_prefix_shapes_available_ranks_fail_closed(self):
        for n in (-1,5,True,1.5):
            with self.assertRaises(ValueError):s.orders_for(s.configs()[2],self.b,n)
        with self.assertRaises(ValueError):s.orders_for(dict(id=2,name='new'),self.b,1)
        with self.assertRaises(ValueError):s.modified_score(s.configs()[0],self.b['data'],1)
        for key,value in [('available',np.ones((4,49),int)),('eligible',np.ones((4,48),bool)),('rank_ols',np.full((4,49),np.nan)),('core',np.full((4,49),np.inf))]:
            b=copy.deepcopy(self.b);b['data'][key]=value
            with self.assertRaises(ValueError):s.orders_for(s.configs()[2],b,1)

    def test_prepare_only_old_controls_and_source_copies(self):
        b=fixture(122);a=b['data'];source=dict(meta=b['meta'],data=dict(baseline=a['baseline'],core=a['core'],eligible=a['eligible'],core_weight=a['cw'][:,:1]),orders=b['source_orders'])
        result=dict(meta=b['meta'],data=a,audit={'synthetic':True},source_hashes={});calls=[];original=s.orders_for
        def control_only(cfg,bundle,n):
            self.assertLess(cfg['id'],2);calls.append(cfg['id']);return original(cfg,bundle,n)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);old=root/'old';old.mkdir()
            (old/'META.json').write_text(json.dumps(b['meta'])+'\n');(old/'PRICES.npz').write_bytes(b'fixed synthetic price bytes')
            (root/'PLAN.json').write_text(json.dumps({'raw_source_hashes':{}}))
            for name in ('DESIGN.json','DESIGN.sha256'):(root/name).write_text('synthetic')
            module=SimpleNamespace(prepare_data=lambda:result)
            with patch.object(s,'P',root),patch.object(s,'SOURCE',old),patch.object(s,'PRICE_SOURCE',old/'PRICES.npz'),patch.object(s,'registered'),patch.object(s,'verify_source'),patch.object(s,'load_source',return_value=source),patch.object(s,'feature_module',return_value=module),patch.object(s,'implementations',return_value=()),patch.object(s,'source_files',return_value=()),patch.object(s,'orders_for',side_effect=control_only),patch.object(s,'modified_score',side_effect=AssertionError('no new scores during prepare')):s.prepare()
            self.assertEqual(set(calls),{0,1});audit=json.loads((root/'FEATURE_AUDIT.json').read_text())['driver_validation']
            self.assertFalse(audit['new_candidate_choices']);self.assertTrue(audit['shared_raw_rank_availability'])
            self.assertEqual(audit['prepared_array_count'],11)
            for name in ('META.json','PRICES.npz'):self.assertEqual((root/name).read_bytes(),(old/name).read_bytes())
            self.assertFalse((root/'TRAIN_TRIALS.json').exists())

    def test_train_only88_freezes_eight_and_seventeen_files(self):
        b=fixture(122)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            files=('DESIGN.json','DESIGN.sha256','PLAN.json','GRID.json','DATA.npz','META.json','PRICES.npz','FEATURE_AUDIT.json','PREPARE_VALIDATION.json','ACTUAL_FEATURE_REVIEW.json','INPUT_HASHES.json','CACHE_HASHES.json','PROTECTED.json','TESTS.txt')
            for name in files:(root/name).write_text('synthetic')
            def returns(cfg,orders,bundle,n):
                self.assertEqual(n,88);self.assertEqual(len(orders),88)
                return np.full(88,.001+cfg['id']*.0001),list(range(89))
            with patch.object(s,'P',root),patch.object(s,'registered'),patch.object(s,'verify_actual_review'),patch.object(s,'load',return_value=b),patch.object(s,'returns_for',side_effect=returns),patch.object(s,'implementations',return_value=()),patch.object(s,'source_files',return_value=()),patch.object(s,'orders_for',wraps=s.orders_for) as choose:s.train()
            self.assertEqual([call.args[2] for call in choose.call_args_list],[88]*8)
            frozen=json.loads((root/'FROZEN.json').read_text());self.assertEqual(frozen['trials'],8)
            self.assertEqual(set(frozen['selected']),{'primary'}|{cfg['name'] for cfg in s.configs()})
            self.assertEqual(len(frozen['hashes']),17);self.assertEqual(frozen['primary_default_id'],1)
            wf=json.loads((root/'WALKFORWARD.json').read_text());self.assertEqual([row['training_months'] for row in wf['folds']],[36,60,76])
            for fold in wf['folds']:self.assertEqual(set(fold['candidate_gates']),{cfg['name'] for cfg in s.configs()[2:]})
            self.assertFalse((root/'RESULTS.json').exists())

    def test_exact_eighteen_attribution_pairs_and_undo_diagnostic(self):
        names=[cfg['name'] for cfg in s.configs()]
        month=dict(date='2020-01-31',split='early53',**{name:dict(symbols=['A','B','C'],monthly_return=.01) for name in names})
        month['shrunk_050']=dict(symbols=['A','B','D'],monthly_return=.03)
        results={name:dict(metrics=dict(total_return=.2,years=1.)) for name in names};out=s.attribution([month],results)
        expected={new+'__vs__'+old for new in names[2:] for old in names[:2]}
        for suffix in ('025','050'):
            expected|={a+'_'+suffix+'__vs__'+b+'_'+suffix for a,b in [('shrunk','ols'),('shrunk','sector_prior'),('sector_prior','ols')]}
        self.assertEqual(set(out),expected);self.assertEqual(len(out),18)
        row=out['shrunk_050__vs__ols_050'];self.assertEqual(row['entering_seats'],1)
        self.assertAlmostEqual(row['undo_best1_diagnostic_cagr'],1.2*1.01/1.03-1)

    def rows(self):
        metric=lambda c:dict(cagr=c,sharpe=1.5,max_drawdown=-.25)
        return [dict(spec=cfg,training=metric(.5 if cfg['id']==1 else .49),early=metric(.4),recent=metric(.6)) for cfg in s.configs()]


    def test_both_cached_controls_bypass_new_factors_and_selector(self):
        del self.b['data']
        with patch.object(s.rp,'choose',side_effect=AssertionError('must bypass')):
            for ident in (0,1):self.assertEqual(s.orders_for(s.configs()[ident],self.b,4),self.b['source_orders'][ident])
        got=s.orders_for(s.configs()[1],self.b,4);got[0]['symbols'].reverse()
        self.assertNotEqual(got,self.b['source_orders'][1])


    def test_all_five_guards_and_strict_improvement(self):
        for section,key,value in [('training','cagr',.5),('training','sharpe',1.449999),
            ('training','max_drawdown',-.280001),('early','cagr',.374999),('recent','cagr',.574999)]:
            rows=self.rows();rows[2]['training']['cagr']=.6;rows[2][section][key]=value
            self.assertEqual(s.select(rows)['primary']['spec']['id'],1,(section,key))
        rows=self.rows();rows[2]['training']['cagr']=.5+5e-13
        self.assertEqual(s.select(rows)['primary']['spec']['id'],1)
        rows[2]['training']['cagr']=.500001;rows[0]['training']['cagr']=.99
        self.assertEqual(s.select(rows)['primary']['spec']['id'],2)


    def test_candidate_cagr_tolerance_then_sharpe_then_id(self):
        rows=self.rows();rows[2]['training']['cagr']=.6;rows[3]['training']['cagr']=.6+5e-13
        rows[2]['training']['sharpe']=1.8
        self.assertEqual(s.select(rows)['primary']['spec']['id'],2)
        rows[3]['training']['sharpe']=1.9
        self.assertEqual(s.select(rows)['primary']['spec']['id'],3)
        rows[2]['training']['sharpe']=1.9
        self.assertEqual(s.select(rows)['primary']['spec']['id'],2)
        rows[3]['training']['cagr']=.6+2e-12;rows[3]['training']['sharpe']=1.5
        self.assertEqual(s.select(rows)['primary']['spec']['id'],3)


    def test_original65_and_local_gates_separate_default_local(self):
        rows=self.rows();rows[2]['training']['cagr']=.495
        got=s.select(rows)
        self.assertTrue(rows[2]['original65_admitted']);self.assertFalse(rows[2]['local67_admitted'])
        self.assertEqual(got['primary']['spec']['id'],1)
        self.assertEqual(set(got),{'primary'}|{cfg['name'] for cfg in s.configs()})


    def test_measure_first88_53_35_and_prefix_half(self):
        m=fixture(122)['meta'];rr=np.arange(88)/10000.;bounds=list(range(89));calls=[]
        def stats(values,start,end):calls.append(len(values));return dict(cagr=0,sharpe=0,max_drawdown=0)
        with patch.object(s.rp,'stats_rr',side_effect=stats):s.measure(rr,m,bounds,88)
        self.assertEqual(calls,[88,53,35]);calls.clear()
        with patch.object(s.rp,'stats_rr',side_effect=stats):s.measure(rr,m,bounds,36)
        self.assertEqual(calls,[36,18,18])


    def test_actual_review_pass_and_all_seven_current_artifact_hashes(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for name in s.ACTUAL_BOUND_FILES:(root/name).write_text(name)
            audit=dict(passed=True,artifact_hashes={name:s.sha(root/name) for name in s.ACTUAL_BOUND_FILES})
            (root/'ACTUAL_FEATURE_REVIEW.json').write_text(json.dumps(audit))
            with patch.object(s,'P',root):
                self.assertTrue(s.verify_actual_review()['passed'])
                for name in s.ACTUAL_BOUND_FILES:
                    original=(root/name).read_bytes();(root/name).write_bytes(original+b'change')
                    with self.assertRaises(AssertionError):s.verify_actual_review()
                    (root/name).write_bytes(original)
                audit['passed']=False;(root/'ACTUAL_FEATURE_REVIEW.json').write_text(json.dumps(audit))
                with self.assertRaises(AssertionError):s.verify_actual_review()


    def test_train_audit_gate_precedes_load_or_new_choices(self):
        with patch.object(s,'registered'),patch.object(s,'verify_actual_review',side_effect=AssertionError('audit required')),\
            patch.object(s,'load',side_effect=RuntimeError('must not load')),patch.object(s,'orders_for',side_effect=RuntimeError('must not choose')):
            with self.assertRaisesRegex(AssertionError,'audit required'):s.train()


    def test_frozen_replay_fixed_units_and_nextopen_fee(self):
        idx=pd.date_range('2020-01-01',periods=4)
        close=pd.DataFrame({'A':[100.,110.,121.,133.1],'B':[100.,100.,90.,90.]},index=idx)
        opening=close.copy();opening.loc[idx[1],['A','B']]=[105.,100.]
        orders=[dict(date='2020-01-01',signal_date='2020-01-01',end='2020-01-04',symbols=['A','B'],scores=[.2,.1])]
        _,eq,_,hold=s.engine.replay(close,opening,orders,cost=0.,execution='close',start=idx[0],end=idx[-1])
        np.testing.assert_allclose(eq,100000*close.div(close.iloc[0]).mean(axis=1),rtol=0.,atol=1e-8)
        rebalanced=100000*np.prod(1+close.pct_change(fill_method=None).iloc[1:].mean(axis=1))
        self.assertNotAlmostEqual(eq.iloc[-1],rebalanced,places=5);self.assertEqual(len(hold),1)
        _,eq0,_,h0=s.engine.replay(close,opening,orders,cost=0.,execution='open',start=idx[0],end=idx[-1])
        met,eq5,tr5,h5=s.engine.replay(close,opening,orders,cost=.0005,execution='open',start=idx[0],end=idx[-1])
        self.assertEqual(h0[0]['execution_date'],'2020-01-02');self.assertEqual(h5[0]['signal_date'],'2020-01-01')
        self.assertAlmostEqual(met['fees_paid'],100000*.0005/1.0005,places=7)
        self.assertAlmostEqual(eq5.iloc[-1],eq0.iloc[-1]/1.0005,places=7)
        self.assertTrue(all(t['side']=='buy' for t in tr5))


    def test_missing_held_future_path_fails_without_changing_choice(self):
        b=fixture(1);idx=pd.date_range(b['meta']['dates'][0],b['meta']['ends'][0])
        orders=s.orders_for(s.configs()[2],b,1);held=orders[0]['symbols'][0]
        close=pd.DataFrame(100.,index=idx,columns=b['meta']['symbols'][0])
        close.loc[idx[2],held]=np.nan;b['close']=close
        self.assertEqual(s.orders_for(s.configs()[2],b,1),orders)
        with self.assertRaisesRegex(AssertionError,'Missing held historical price'):s.returns_for(s.configs()[2],orders,b,1)


    def test_streamed_parallel_hashes_cover_every_file_and_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            paths=[];expected={}
            for i in range(19):
                p=Path(folder)/str(i);value=bytes([i])*((1024*1024+7) if i==0 else i)
                p.write_bytes(value);paths.append(p);expected[str(p)]=hashlib.sha256(value).hexdigest()
            self.assertEqual(s.hash_files(paths),expected);s.assert_hashes(expected)
            with self.assertRaises(AssertionError):s.assert_hashes({str(paths[0]):'0'*64})
            with self.assertRaises(FileNotFoundError):s.hash_files(paths+[Path(folder)/'missing'])


    def test_all_six_candidates_participate_in_selection_and_ties(self):
        for ident in range(2,8):
            rows=self.rows();rows[ident]['training']['cagr']=.6
            self.assertEqual(s.select(rows)['primary']['spec']['id'],ident)
        rows=self.rows()
        for row in rows[2:]:row['training']['cagr']=.6
        rows[7]['training']['cagr']+=5e-13;rows[6]['training']['sharpe']=1.7
        self.assertEqual(s.select(rows)['primary']['spec']['id'],6)
        rows[7]['training']['cagr']+=2e-12
        self.assertEqual(s.select(rows)['primary']['spec']['id'],7)



if __name__=='__main__':unittest.main()
