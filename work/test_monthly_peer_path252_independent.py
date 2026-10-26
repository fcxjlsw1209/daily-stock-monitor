"""252 synthetic source/ratio/original-score tests; never loads actual inputs."""
import ast
import copy
from fractions import Fraction as F
from functools import lru_cache
import math
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import monthly_peer_path252_independent as audit


def helper():return audit.load_pinned(audit.PEER_TEST,audit.PEER_TEST_SHA,'path252_old_fixture')


def fixture():return helper().fixture()


def warm_fixture():
    p,m,c,a,r=helper().warm_fixture()
    for symbol in ('SPCX','SKHY'):
        p['adj'][symbol]=np.nan
        bound=audit.peer.BOUNDS[symbol]['not_before']
        p['adj'].loc[pd.Timestamp(bound):,symbol]=128.
    return p,m,c,a,r


def physical(args,n):return helper().physical(args,n)


def run(args,n=None):
    report=audit.independent_source(*args,limit=n)
    v,a=audit.independent_features(*args[:4],report,limit=n)
    return report,v,a


def selection_fixture():
    p,m,c,a,r=helper().selection_fixture()
    report,v,fa=run((p,m,c,a,r))
    return p,m,c,a,r,report,v,fa


@lru_cache(None)
def full_core_fixture():
    h=audit.load_pinned(audit.CORE_TEST,audit.CORE_TEST_SHA,'path252_core_fixture')
    p,m,c,a,r=h.fixture();return {'adj':p['adj']},m,c,a


class ExactMathTests(unittest.TestCase):
    def test_two_peer_reset_counterexample_and_ratio_direction(self):
        focal=[10.,10.,11.]+[11.]*208
        first=[1.,2.,1.]+[1.]*208;second=[1.,.5,1.]+[1.]*208
        s=audit.relative_statistics(focal,[first,second])
        self.assertEqual(s['daily_peer_gross'],F(25,16));self.assertEqual(s['endpoint_peer_gross'],1)
        self.assertEqual(s['path_raw'],F(-37,125));self.assertEqual(s['endpoint_raw'],F(1,10))

    def test_identical_peer_paths_and_exact_median_ties(self):
        p=[float(128+i) for i in range(211)];f=[float(256+3*i) for i in range(211)]
        s=audit.relative_statistics(f,[p,p,p,p])
        self.assertEqual(s['daily_peer_gross'],s['endpoint_peer_gross']);self.assertEqual(s['path_raw'],s['endpoint_raw'])
        self.assertEqual(audit.median_exact([F(1),F(2),F(3),F(8)]),F(5,2))
        self.assertEqual(audit.ranks_exact([F(1)]*3),[float(F(1,6))]*3)
        self.assertEqual(audit.ranks_exact([F(1),F(2),F(3),F(4)]),[-.25,0.,.25,.5])
        self.assertEqual(audit.ranks_exact([]),[])

    def test_representable_per_security_scale_and_peer_permutation(self):
        f=[float(128+i) for i in range(211)]
        pp=[[float(512+(i*(j+1))%71) for i in range(211)] for j in range(4)]
        x=audit.relative_statistics(f,pp)
        y=audit.relative_statistics([v*4 for v in f],[[v*2**j for v in p] for j,p in enumerate(pp)])
        self.assertEqual(x,y);self.assertEqual(x,audit.relative_statistics(f,pp[::-1]))

    def test_extreme_finite_binary64_raw_never_cast_before_ranking(self):
        lo=float(np.nextafter(0.,1.));hi=float(np.finfo(float).max)
        focal=[lo]*210+[hi];peers=[[1.]*211,[1.]*211]
        s=audit.relative_statistics(focal,peers)
        self.assertGreater(s['path_raw'],F.from_float(hi))
        q=audit.ranks_exact([s['path_raw'],s['path_raw']+1])
        self.assertEqual(q,[0.,.5])
        self.assertEqual(audit.rational(audit.pair(s['path_raw'])),s['path_raw'])

    def test_invalid_raw_rational_and_price_inputs(self):
        for value in ({'numerator':'2','denominator':'2'},{'numerator':'01','denominator':'1'},
                      {'numerator':'1','denominator':'0'},1.,True):
            with self.subTest(value=value),self.assertRaises(ValueError):audit.rational(value)
        for bad in (None,True,'1',0.,-1.,np.nan,np.inf):
            p=[1.]*211;p[50]=bad
            with self.subTest(bad=bad),self.assertRaises(ValueError):audit.relative_statistics(p,[[1.]*211]*2)


class SourceTests(unittest.TestCase):
    def test_complete_source_report_masks_counts_and_no_mutation(self):
        args=fixture();before=copy.deepcopy(args);report,v,fa=run(args)
        self.assertEqual(set(report),set(audit.SOURCE_TOP_KEYS))
        self.assertEqual(report['counts'],dict(events=3,stock_cells=24,mature_cells=16,supported_cells=14,price_observations=16*211,new_model_fits=0))
        self.assertTrue(all(audit.verify_source(*args,report)[k] for k in audit.SOURCE_FLAGS))
        receipt=audit.verify_features(*args[:4],report,v,fa)
        self.assertTrue(all(receipt[k] for k in audit.FEATURE_FLAGS))
        pd.testing.assert_frame_equal(args[0]['adj'],before[0]['adj'])
        self.assertEqual((args[1],args[2],args[4]),(before[1],before[2],before[4]))
        audit.exact_array(args[3]['eligible'],before[3]['eligible'],'No E mutation')

    def test_source_before_E_and_even_structurally_unsupported_required(self):
        args=fixture();left=run(args);args[3]['eligible'][:]=False;right=run(args)
        audit.record_check(left[0],right[0]);audit.record_check(left[2],right[2])
        for k in audit.VALUE_KEYS:audit.exact_array(left[1][k],right[1][k],k)
        args=fixture();t=args[2].index(args[1]['dates'][1]);args[0]['adj'].at[pd.Timestamp(args[2][t-100]),'H']=np.nan
        with self.assertRaisesRegex(ValueError,'H'):run(args)

    def test_211_inclusive_window_and_internal_missing_hard(self):
        args=fixture();report,_,_=run(args);t=args[2].index(args[1]['dates'][1]);e=report['events'][1]
        self.assertEqual(e['path_dates'],args[2][t-252:t-41]);self.assertEqual(len(e['path_dates']),211)
        row=next(x for x in report['records'] if x['query_index']==1)
        self.assertEqual([x['date'] for x in row['path_observations']],e['path_dates'])
        for missing in ('row','column','nan'):
            bad=fixture();d=pd.Timestamp(e['path_dates'][80])
            if missing=='row':bad[0]['adj']=bad[0]['adj'].drop(d)
            elif missing=='column':bad[0]['adj']=bad[0]['adj'].drop(columns='A')
            else:bad[0]['adj'].at[d,'A']=np.nan
            with self.subTest(missing=missing),self.assertRaises(ValueError):run(bad)

    def test_all_present_invalids_remain_hard(self):
        for value in (False,'128',-1.,0.,math.inf,-math.inf,F(1,10**1000),10**1000):
            args=fixture();t=args[2].index(args[1]['dates'][1]);args[0]['adj'].at[pd.Timestamp(args[2][t-100]),'A']=value
            with self.subTest(value=repr(value)[:50]),self.assertRaises(ValueError):run(args)

    def test_partial_and_ETFs_have_no_new_price_reads(self):
        args=physical(fixture(),1);args[0]['adj'][:]='poison';r,v,fa=run(args)
        self.assertEqual(r['counts']['price_observations'],0);self.assertTrue(all(not x['path_observations'] for x in r['records']))
        self.assertFalse(v['mature'].any())
        args=fixture();args[0]['adj']['FUND']='unused';run(args)

    def test_warmup_scans_all_points_and_preserves_inventory(self):
        args=warm_fixture();report,v,_=run(args)
        self.assertEqual(report['counts']['price_observations'],15*211)
        self.assertFalse(v['mature'][:,3].any());self.assertFalse(v['mature'][1:,4].any())
        for r in report['records']:
            self.assertEqual(len(r['path_observations']),211)
            if r['history_status']=='official_pre_inception':
                self.assertTrue(all(x['status']=='missing' for x in r['path_observations'] if x['date']<r['history_bound']['not_before']))
        for kind in ('positive_before','malformed_before','missing_after'):
            bad=warm_fixture();frame=bad[0]['adj']
            if kind=='positive_before':frame.at[pd.Timestamp('2025-10-01'),'SPCX']=1.
            elif kind=='malformed_before':frame.at[pd.Timestamp('2025-10-01'),'SPCX']='bad'
            else:frame.at[pd.Timestamp('2026-06-15'),'SPCX']=np.nan
            with self.subTest(kind=kind),self.assertRaises(ValueError):run(bad)

    def test_prebound_missing_row_or_column_reason_is_independent_evidence(self):
        args=physical(warm_fixture(),1)
        row=next(r for r in args[4]['records'] if r['symbol']=='SPCX')
        t=args[2].index(args[1]['dates'][0]);days=[pd.Timestamp(d) for d in args[2][t-252:t-41]]
        frame=args[0]['adj'].drop(days[50])
        observations,_=audit._inspect_path(frame,row,days)
        self.assertEqual(observations[50]['reason'],'missing_row')
        for kind in ('value','column'):
            args=physical(warm_fixture(),1);frame=args[0]['adj']
            if kind=='value':
                target=pd.Timestamp(args[2][args[2].index(args[1]['dates'][0])-252])
                frame.at[target,'SPCX']=pd.NA
            else:args[0]['adj']=frame.drop(columns='SPCX')
            report,_,_=run(args);row=next(r for r in report['records'] if r['symbol']=='SPCX')
            wanted='missing_value' if kind=='value' else 'missing_column'
            self.assertTrue(all(p['reason']==wanted for p in row['path_observations']))

    def test_old_claim_mapping_peer_and_history_tamper_rejected(self):
        for mutation in ('self','omission','duplicate','sector','bool','history'):
            args=fixture();r=args[4]['records'][8]
            if mutation=='self':r['peer_indices'].append(r['column_index'])
            elif mutation=='omission':r['peer_indices']=r['peer_indices'][:-1]
            elif mutation=='duplicate':args[4]['records'][9]['issuer_key']=r['issuer_key']
            elif mutation=='sector':r['sector']='XLE'
            elif mutation=='bool':r['column_index']=False
            else:r['history_status']='official_pre_inception'
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):run(args)

    def test_full_logical_physical_prefixes_and_future_poison(self):
        args=fixture();full=run(args)
        for n in range(4):
            limited=run(args,n);small=run(physical(args,n))
            audit.record_check(limited[0],small[0]);audit.record_check(limited[2],small[2])
            for k in audit.VALUE_KEYS:audit.exact_array(limited[1][k],full[1][k][:n],k)
            v,fa=audit.independent_features(*args[:4],full[0],limit=n)
            audit.record_check(fa,limited[2])
        poisoned=copy.deepcopy(args);poisoned[1]['dates'][2]='future';poisoned[1]['symbols'][2]=None
        poisoned[3]['eligible']=poisoned[3]['eligible'].astype(object);poisoned[3]['eligible'][2]='bad'
        poisoned[4]['events'][2]=None
        poisoned[4]['records']=[r if r['query_index']<2 else {'query_index':2,'poison':object()} for r in poisoned[4]['records']]
        poisoned[0]['adj'].loc[poisoned[0]['adj'].index>args[1]['dates'][1]]='future price'
        audit.record_check(run(poisoned,2)[0],run(args,2)[0])
        class UnreadFuture:
            def __deepcopy__(self,memo):raise AssertionError('Future body must not be copied')
        daily=copy.deepcopy(full[0]);daily['events'][2]=UnreadFuture()
        daily['records']=[r if r['query_index']<2 else {'query_index':2,'poison':UnreadFuture()} for r in daily['records']]
        vv,aa=audit.independent_features(*args[:4],daily,limit=2)
        wanted=run(args,2);audit.record_check(aa,wanted[2])
        for k in audit.VALUE_KEYS:audit.exact_array(vv[k],wanted[1][k],k)

    def test_source_report_and_feature_full_audit_tamper(self):
        args=fixture();report,v,fa=run(args)
        for mutation in ('counts','path','date','keys'):
            bad=copy.deepcopy(report)
            if mutation=='counts':bad['counts']['price_observations']-=1
            elif mutation=='path':bad['records'][8]['path_observations'][80]['status']='missing'
            elif mutation=='date':bad['events'][1]['path_dates'][80]='2000-01-01'
            else:bad['records'][8]['extra']=1
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):audit.verify_source(*args,bad)
        for mutation in ('raw','rank','mask','counts'):
            vv=copy.deepcopy(v);aa=copy.deepcopy(fa)
            if mutation=='raw':aa['events'][1]['cells'][0]['daily_peer_gross']['numerator']='999'
            elif mutation=='rank':vv['path_target'][1,0]=math.nextafter(vv['path_target'][1,0],math.inf)
            elif mutation=='mask':vv['supported'][1,0]=False
            else:aa['counts']['supported_cells']-=1
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):audit.verify_features(*args[:4],report,vv,aa)

    def test_N0_both_arms_exact_incumbent_noop(self):
        p,m,c,a,r,_,_,_=selection_fixture();sectors={s:'XLK' if j<2 else ['XLE','XLF','XLI','XLP','XLV','XLU'][j-2] for j,s in enumerate(list('ABCDEFGH'))}
        r=helper().make_report(m,c,sectors);report,v,fa=run((p,m,c,a,r));self.assertFalse(v['supported'].any())
        scores,ch,pre=audit.selection_oracle(a,m,v)
        for arm in audit.ARMS[1:]:
            audit.exact_array(scores[arm],scores[audit.ARMS[0]],arm);audit.exact_array(pre[arm],pre[audit.ARMS[0]],arm)
            self.assertEqual(ch[arm],ch[audit.ARMS[0]])


class OriginalSelectionTests(unittest.TestCase):
    def test_real_frozen_adapter_parity_full_scores_pre_and_single_episode(self):
        p,m,c,a,r,report,v,fa=selection_fixture();s,ch,pre=audit.selection_oracle(a,m,v)
        study=helper().load('outputs/monthly_local_linear70/study.py','path252_test_study')
        adapter=helper().load('outputs/monthly_path70_research/incumbent_adapter.py','path252_test_adapter')
        base=audit.original_base(a,m,len(m['dates']));calls=[];actual={};orders={}
        for arm,key in zip(audit.ARMS,(None,'path_target','endpoint_target')):
            factor=np.full(a['eligible'].shape,np.nan) if key is None else v[key]
            def boundary(P,E,names,effects,original,policy):
                calls.append(arm);audit.exact_array(P,pre[arm],'Funded pre');audit.exact_array(E,a['eligible'],'Original E')
                return study.ep.select_boundary(P,E,names,effects,original,policy)
            choices,scores=adapter.score_and_choose('incumbent66',.05,factor,a,m,len(m['dates']),pre_score=base['P'],
                effects=a['effects'],episode_policy=dict(budget=.05,ceiling=5,gap=.02),boundary_selector=boundary)
            self.assertEqual(choices,ch[arm]);audit.exact_array(scores,s[arm],'Full final')
            actual[arm]=scores;orders[arm]=study.rp.make_orders(choices,scores,m)
        self.assertEqual(calls,list(audit.ARMS));receipt=audit.verify_selection(a,m,v,orders,scores=actual,pre_scores=pre)
        self.assertTrue(all(receipt[k] for k in audit.SELECTION_FLAGS))
        audit.verify_incumbent(a,m,orders[audit.ARMS[0]],actual[audit.ARMS[0]],pre[audit.ARMS[0]])

    def test_zero_target_funded_and_partial_unchanged(self):
        p,m,c,a,r,report,v,fa=selection_fixture();v['path_target'][1,0]=0.
        s,ch,pre=audit.selection_oracle(a,m,v);base=audit.original_base(a,m,3)
        self.assertEqual(pre[audit.ARMS[1]][1,0],base['P'][1,0]+.05*(0.-base['support'][1,0]))
        for arm in audit.ARMS:
            audit.exact_array(pre[arm][0],base['P'][0],'Partial original pre')
            audit.exact_array(s[arm][0],s[audit.ARMS[0]][0],'Partial original final')

    def test_complete_budget_including_extra_extension_and_no_mutation(self):
        p,m,c,a,r,report,v,fa=selection_fixture();a['extension_available'][:]=True;a['forecast_available'][:]=True
        a['rank_log_excess'][:]=.2;a['eligible'][2,1]=False;before=copy.deepcopy(a)
        s,ch,pre=audit.selection_oracle(a,m,v);receipt=audit.verify_selection(a,m,v,audit.orders_from_choices(s,ch,m),scores=s,pre_scores=pre)
        self.assertEqual(receipt['calm_nonweak_extension_count'],8);self.assertGreaterEqual(receipt['minimum_remaining_support'],.33-1e-12)
        for k in a:audit.exact_array(a[k],before[k],k)

    def test_selection_prefix_cash_halves_and_exact_tamper_rejection(self):
        p,m,c,a,r,report,v,fa=selection_fixture()
        for count in (0,1,2,9):
            a['eligible'][:]=False;a['eligible'][:,:count]=True
            s,ch,pre=audit.selection_oracle(a,m,v)
            for arm in audit.ARMS:self.assertTrue(all(len(row)==(count if count==2 else 3 if count>=3 else 0) for row in ch[arm]))
            for n in range(4):
                sn,cn,pn=audit.selection_oracle(a,m,v,n)
                for arm in audit.ARMS:audit.exact_array(sn[arm],s[arm][:n],'Prefix final')
                audit.verify_selection(a,m,v,audit.orders_from_choices(sn,cn,m),scores=sn,pre_scores=pn,n=n)
        s,ch,pre=audit.selection_oracle(a,m,v);wrong=copy.deepcopy(s);wrong[audit.ARMS[1]][1,-1]+=1e-10
        with self.assertRaises(ValueError):audit.verify_selection(a,m,v,audit.orders_from_choices(s,ch,m),scores=wrong,pre_scores=pre)

    def test_original49_source_gate_scope_parity_and_early_stage(self):
        p,m,c,a=copy.deepcopy(full_core_fixture())
        with patch.object(audit.old,'episode',side_effect=AssertionError('No early episode')),patch.object(audit,'_relative',side_effect=AssertionError('No new math')):
            receipt=audit.verify_original_source(p,m,c,a)
        self.assertTrue(all(receipt[k] for k in audit.ORIGINAL_SOURCE_FLAGS));self.assertEqual(receipt['episode_calls'],0)
        self.assertFalse(receipt['new_target_ranks_computed']);self.assertLessEqual(receipt['maximum_core_source_absolute_error'],1e-12)
        wrong=copy.deepcopy(a);wrong['core'][1,0]+=1e-8
        with self.assertRaisesRegex(ValueError,'core parity'):audit.verify_original_source(p,m,c,wrong)
        # A consumed formal date row cannot be removed, irrespective of future tail.
        missing={'adj':p['adj'].drop(pd.Timestamp(m['dates'][1]))}
        with self.assertRaises(ValueError):audit.verify_original_source(missing,m,c,a)

    def test_source_gate_no_rank_episode_or_features(self):
        args=fixture();report,_,_=run(args)
        with patch.object(audit.old,'episode',side_effect=AssertionError('No episode')),patch.object(audit,'ranks_exact',side_effect=AssertionError('No ranks')):
            result=audit.verify_source(*args,report)
        self.assertEqual(result['episode_calls'],0);self.assertFalse(result['new_target_ranks_computed'])

    def test_dependency_closure_no_new_producer_imports(self):
        files=audit.dependency_files();self.assertIn(audit.PEER,files);self.assertIn(audit.CORE,files);self.assertIn(audit.ACCOUNT_BASE,files)
        tree=ast.parse(audit.Path(audit.__file__).read_text())
        imports=[node for node in ast.walk(tree) if isinstance(node,(ast.Import,ast.ImportFrom))]
        text=' '.join(ast.dump(x) for x in imports)
        self.assertNotIn('monthly_peer_path252_kernel',text);self.assertNotIn('monthly_peer_path252_host',text)


class AccountingTests(unittest.TestCase):
    @staticmethod
    def args():
        h=audit.load_pinned(audit.old.ACCOUNT_TEST,audit.old.ACCOUNT_TEST_SHA,'path252_account_fixture')
        p,o,orders,exports,results,m=h.account_fixture();first=next(iter(orders))
        return p,o,{k:copy.deepcopy(orders[first]) for k in audit.ARMS},{k:copy.deepcopy(exports[first]) for k in audit.ARMS},{k:copy.deepcopy(results[first]) for k in audit.ARMS},m

    def test_nine_paths_fortyfive_slices_eighteen_training_groups(self):
        r=audit.audit_outcomes(*self.args());self.assertEqual((r['checked_paths'],r['checked_slices'],r['checked_training_groups']),(9,45,18))

    def test_zero_stream_and_aggregate_exact_not_near(self):
        for target in ('trade','period','aggregate'):
            args=list(self.args());arm=audit.ARMS[1];path='same_close_zero'
            if target=='trade':args[3][arm][path]['TRADES'][0]['cost']=1e-15
            elif target=='period':args[3][arm][path]['HOLDINGS'][0]['fee']=1e-15
            else:args[4][arm][path]['metrics']['fees_paid']=1e-15
            with self.subTest(target=target),self.assertRaises(ValueError):audit.audit_outcomes(*args)


class BilateralCross(unittest.TestCase):
    def test_both_fixtures_full_source_feature_audit_and_all_prefixes(self):
        import monthly_peer_path252_kernel as producer
        import test_monthly_peer_path252_kernel as fixtures
        self.assertEqual(audit.peer.sha(audit.ROOT/'work/monthly_peer_path252_kernel.py'),'dc7b36c6b16e08c9e85cf2b60de9a5dc3f8dd3f20d5e4c8d5cbb98e8c32c670c')
        self.assertEqual(audit.peer.sha(audit.ROOT/'work/test_monthly_peer_path252_kernel.py'),'5f8245859b51b34c54f814abd01641c6c241fffbfc9deff8067f6da42432d2e6')
        for args in (fixture(),warm_fixture(),fixtures.fixture(),fixtures.warm_fixture()):
            p,m,c,a,r=args
            for n in range(len(m['dates'])+1):
                source=producer.build_source(*args,limit=n);audit.verify_source(*args,source,limit=n)
                v,fa=producer.build_features(p,m,c,a,source,limit=n);audit.verify_features(p,m,c,a,source,v,fa,limit=n)
                small=physical(args,n);s2=producer.build_source(*small)
                audit.record_check(source,s2,'Physical source prefix')
                v2,a2=producer.build_features(*small[:4],s2);audit.verify_features(*small[:4],s2,v2,a2)
                audit.record_check(fa,a2,'Physical full audit')
                for k in audit.VALUE_KEYS:audit.exact_array(v[k],v2[k],'Physical '+k)


class HostIntegration(unittest.TestCase):
    def test_full49_and_empty_prefix_with_independent_expected_incumbent(self):
        # End-to-end host seam on invented49-member data, including empty prefix.
        import monthly_peer_path252_kernel as producer
        import monthly_peer_path252_host as host
        p,m,c,a=copy.deepcopy(full_core_fixture());r=helper().make_report(m,c)
        study=helper().load('outputs/monthly_local_linear70/study.py','path252_cross_original')
        for args in ((p,m,c,a,r),physical((p,m,c,a,r),0)):
            p1,m1,c1,a1,r1=args;n=len(m1['dates'])
            base=audit.original_base(a1,m1,n);original_scores,choices=audit.old.episode(base['P'],base)
            expected=[dict(date=m1['dates'][i],end=m1['ends'][i],signal_date=m1['dates'][i],
              symbols=[m1['symbols'][i][j] for j in ids],scores=[float(original_scores[i,j]) for j in ids])
              for i,ids in enumerate(choices)]
            values,fa,source,orders,scores,pres,checks=host.prepare_one(study,a1,m1,p1,c1,r1,n,producer,audit,
                expected_incumbent_orders=expected)
            self.assertEqual(checks['selection']['ordered_choices_checked'],3*n)
            self.assertTrue(checks['original_source']['original_core_checked'])
            self.assertEqual(checks['daily_source']['episode_calls'],0)


if __name__=='__main__':
    if '--self' in sys.argv:
        sys.argv.remove('--self');suite=unittest.TestSuite()
        for cls in (ExactMathTests,SourceTests,OriginalSelectionTests,AccountingTests,HostIntegration):suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(cls))
        result=unittest.TextTestRunner(verbosity=2).run(suite);raise SystemExit(not result.wasSuccessful())
    unittest.main(verbosity=2)
