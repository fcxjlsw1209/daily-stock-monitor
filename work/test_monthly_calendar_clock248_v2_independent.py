"""Invented source histories, original-score caches and explicit accounting only."""
import copy
import math
import pickle
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import monthly_calendar_clock248_v2_independent as audit


def helper():return audit.load_pinned(audit.PRIOR_TEST,audit.PRIOR_TEST_SHA,'calendar248_v2_original_fixture')


def set_core(args):
    p,m,c,a,r=args;frame=p['adj'];ret=frame.pct_change(fill_method=None);sd=ret.rolling(63).std()
    for i,(d,names) in enumerate(zip(m['dates'],m['symbols'])):
        calm=bool(a['state_calm10'][i]);cw=.5 if calm else .52
        raw=frame.shift(42)/frame.shift(252 if calm else 294)-1.
        if not calm:raw=raw/sd.replace(0.,np.nan)
        rank=raw.loc[d,names].replace([np.inf,-np.inf],np.nan).rank(pct=True).fillna(.5).to_numpy()-.5
        a['core'][i]=cw*rank
    a['baseline']=a['core']+(1.-np.asarray(a['core_weight']).reshape(len(m['dates']),-1))*np.linspace(-.15,.15,len(m['symbols'][0]))
    return args


def fixture():
    a,m=helper().fixture();N=len(m['dates']);W=len(m['symbols'][0]);calm=np.array([i%2==1 for i in range(N)])
    a['state_calm10']=calm;a['core_weight']=np.where(calm,.5,.52)[:,None]
    idx=pd.bdate_range('2010-01-04',max(m['ends']));t=np.arange(len(idx),dtype=float)
    names=sorted(set(s for row in m['symbols'] for s in row))
    frame=pd.DataFrame({s:50.*np.exp((.0001+j*.00002)*t+.12*np.sin((.021+j*.003)*t+j)) for j,s in enumerate(names)},index=idx)
    review=dict(schema='calendar_clock248_v2_metadata_source_review_v1',panel_first_session=str(idx[0].date()),listing_bounds=[],identity_cells=[])
    return set_core(({'adj':frame},m,idx.strftime('%Y-%m-%d').tolist(),a,review))


def physical(args,n):
    p,m,c,a,r=copy.deepcopy(args);a={k:v[:n].copy() for k,v in a.items()}
    m={k:(v[:n] if k in ('dates','ends','symbols') else v) for k,v in m.items()}
    if n:
        t=c.index(m['dates'][-1]);c=c[:t+2];p['adj']=p['adj'].loc[:m['dates'][-1]].copy()
    else:c=[];p['adj']=p['adj'].iloc[:0].copy()
    r['identity_cells']=[x for x in r['identity_cells'] if x['query_index']<n]
    return p,m,c,a,r


def run(args,n=None):
    p,m,c,a,r=args;f=audit.independent_features(p,m,c,a,r,n)
    s,ch,pre,d=audit.selection_oracle(a,m,f,n)
    return f,audit.orders_from_choices(s,ch,m),s,pre,d


def verify(args,out,n=None):
    p,m,c,a,r=args;f,o,s,pre,d=out
    return audit.verify_prepared(p,m,c,a,r,f,o,scores=s,pre_scores=pre,audit=d,n=n)


def bound_for(args,j,not_before='2016-01-01',available_from='2016-01-02'):
    p,m,c,a,r=args;s=m['symbols'][0][j]
    b=dict(symbol=s,issuer_key='synthetic-issuer',not_before=not_before,available_from=available_from,kind='synthetic',evidence_id='synthetic-evidence')
    r['listing_bounds']=[b];r['identity_cells']=[];sessions=[pd.Timestamp(x) for x in c]
    for i,day in enumerate(m['dates']):
        if s not in m['symbols'][i]:continue
        t=c.index(day)
        r['identity_cells'].append(dict(query_index=i,column_index=m['symbols'][i].index(s),date=day,symbol=s,
            issuer_key=b['issuer_key'],available_from=available_from,not_before=not_before,bound_visible=available_from<=day,
            calendar_anchors=audit.calendar_anchors(pd.Timestamp(day),sessions),old_clock_on_official_axis=dict(
                t_minus_42=c[t-42],t_minus_252=c[t-252],t_minus_294=c[t-294],risk63_price_start=c[t-63],
                risk63_price_end=day,risk63_prices=64,risk63_return_intervals=63)))


class SourceTests(unittest.TestCase):
    def test_full_features_scores_audit_and_immutability(self):
        args=fixture();snapshot=pickle.dumps(args);out=run(args);r=verify(args,out)
        self.assertEqual(pickle.dumps(args),snapshot);self.assertTrue(all(r[k] for k in audit.PREPARE_FLAGS))
        self.assertEqual(r['C_source_atol'],1e-12);self.assertEqual(r['C_source_rtol'],0.)
        self.assertEqual(r['counts']['original_episode_calls'],2)
        self.assertEqual(r['ordered_choices_checked'],2*len(args[1]['dates']))

    def test_pandas_mean_ties_missing_zero_singleton(self):
        rank,U=audit.independent_rank([2.,2.,2.,float('nan')]);self.assertEqual(U,[0,1,2])
        self.assertEqual(rank[0],2./3.-.5);self.assertEqual(rank[3],0.)
        rank,U=audit.independent_rank([float('nan'),1.]);self.assertEqual(rank.tolist(),[0.,.5])
        rank,U=audit.independent_rank([1.,2.,3.,4.]);self.assertEqual(rank[1],0.);self.assertIn(1,U)

    def test_original_core_mismatch_hard_failure_not_fallback(self):
        args=fixture();args[3]['core'][1,0]+=1e-8
        with self.assertRaises(ValueError):run(args)

    def test_weighted_core_tolerance_keeps_frozen_delta_anchor(self):
        args=fixture();args[3]['core'][1,0]+=5e-13
        out=run(args);verify(args,out);cell=out[0][1]['cells'][0]
        self.assertGreater(cell['core_abs_error'],0.);self.assertLessEqual(cell['core_abs_error'],1e-12)
        self.assertEqual(cell['delta'],out[0][1]['core_weight']*cell['calendar_rank']-cell['frozen_core'])

    def test_whole_month_missing_ineligible_U0_member(self):
        args=physical(fixture(),2);p,m,c,a,r=args;j=0;a['eligible'][1,j]=False
        end=audit.calendar_anchors(pd.Timestamp(m['dates'][1]),[pd.Timestamp(d) for d in c])['M-12']
        p['adj'].at[pd.Timestamp(end),m['symbols'][1][j]]=np.nan;set_core(args)
        out=run(args);e=out[0][1];self.assertIn(j,e['u0_indices']);self.assertEqual(e['branch'],'missing_calendar')
        self.assertIsNone(e['cells'][j]['calendar_start_price']);self.assertIsNotNone(e['cells'][j]['calendar_end_price'])
        np.testing.assert_array_equal(out[2][audit.ARMS[0]],out[2][audit.ARMS[1]])
        self.assertEqual(out[1][audit.ARMS[0]],out[1][audit.ARMS[1]])
        self.assertTrue(any(c['calendar_raw'] is not None for c in e['cells']))
        self.assertTrue(all(c['calendar_rank'] is None and c['delta']==0. for c in e['cells']));verify(args,out)

    def test_outside_U0_does_not_read_candidate_price_or_expand_domain(self):
        args=physical(fixture(),2);p,m,c,a,r=args;j=0;day=m['dates'][1];t=c.index(day);s=m['symbols'][1][j]
        p['adj'].at[pd.Timestamp(c[t-252]),s]=np.nan;set_core(args)
        old=audit.independent_features(*args[:3],args[3],args[4]);self.assertNotIn(j,old[1]['u0_indices'])
        end=old[1]['calendar_start'];p['adj'].at[pd.Timestamp(end),s]=-7.;set_core(args)
        out=run(args);self.assertEqual(out[0][1]['cells'][j]['calendar_reason'],'outside_U0');verify(args,out)

    def test_missing_does_not_hide_invalid_required_endpoint(self):
        args=physical(fixture(),2);p,m,c,a,r=args;anchors=audit.calendar_anchors(pd.Timestamp(m['dates'][1]),[pd.Timestamp(d) for d in c])
        p['adj'].at[pd.Timestamp(anchors['M-12']),m['symbols'][1][0]]=np.nan
        p['adj'].at[pd.Timestamp(anchors['M-12']),m['symbols'][1][1]]=-3.;set_core(args)
        with self.assertRaises(ValueError):run(args)

    def test_original_column_and_recent_row_missing_are_hard_errors(self):
        args=physical(fixture(),2);args[0]['adj']=args[0]['adj'].drop(columns=[args[1]['symbols'][1][0]])
        with self.assertRaises(ValueError):run(args)
        args=physical(fixture(),2);args[0]['adj']=args[0]['adj'].drop(index=args[0]['adj'].index[-100])
        with self.assertRaises(ValueError):run(args)

    def test_older_calendar_row_missing_is_whole_month_fallback(self):
        args=physical(fixture(),3);p,m,c,a,r=args;a['state_calm10'][2]=False;a['core_weight'][2]=.52
        anchor=audit.calendar_anchors(pd.Timestamp(m['dates'][2]),[pd.Timestamp(d) for d in c])['M-14']
        self.assertLess(c.index(anchor),c.index(m['dates'][2])-294)
        p['adj']=p['adj'].drop(index=pd.Timestamp(anchor))
        a={k:v[2:3].copy() for k,v in a.items()};m={k:(v[2:3] if k in ('dates','ends','symbols') else v) for k,v in m.items()}
        args=set_core((p,m,c,a,r));out=run(args)
        self.assertEqual(out[0][0]['branch'],'missing_calendar');verify(args,out)

    def test_initial_partial_never_reads_source_numeric_values(self):
        args=physical(fixture(),1);args[0]['adj']=args[0]['adj'].astype(object);args[0]['adj'].iloc[:]='poison'
        out=run(args);self.assertEqual(out[0][0]['cells'],[]);self.assertEqual(out[1][audit.ARMS[0]],out[1][audit.ARMS[1]])
        verify(args,out)

    def test_original_inf_sanitization_and_zero_sigma_domain(self):
        args=physical(fixture(),3);p,m,c,a,r=args;s=m['symbols'][2][0]
        p['adj'][s]=100.;set_core(args);out=run(args)
        self.assertEqual(out[0][2]['cells'][0]['sigma63'],0.)
        self.assertFalse(out[0][2]['cells'][0]['original_finite'])
        self.assertTrue(out[0][1]['cells'][0]['original_finite'])
        verify(args,out)

    def test_empty_original_domain_exact_copy(self):
        args=physical(fixture(),3);p,m,c,a,r=args;p['adj'].iloc[:]=100.;set_core(args)
        out=run(args);self.assertEqual(out[0][2]['branch'],'empty_domain')
        np.testing.assert_array_equal(out[2][audit.ARMS[0]][2],out[2][audit.ARMS[1]][2]);verify(args,out)

    def test_original_origin_and_nonreal_history_rejected(self):
        args=physical(fixture(),2);args[4]['panel_first_session']='2010-01-05'
        with self.assertRaises(ValueError):run(args)
        args=physical(fixture(),2);args[0]['adj']=args[0]['adj'].astype(object);args[0]['adj'].iloc[0,0]=True
        with self.assertRaises(ValueError):run(args)

    def test_listing_identity_clock_and_visible_history_guard(self):
        args=physical(fixture(),2);bound_for(args,0)
        with self.assertRaises(ValueError):run(args)
        args=physical(fixture(),2);bound_for(args,0,available_from='2099-01-01');out=run(args);verify(args,out)
        args[4]['identity_cells'][1]['calendar_anchors']['M-2']='2010-01-04'
        with self.assertRaises(ValueError):run(args)

    def test_current_sigma_not_formation_sigma_and_calendar_gap_not_42(self):
        args=fixture();out=run(args);p,m,c,a,r=args
        expected=p['adj'].pct_change(fill_method=None).rolling(63).std()
        gaps=[]
        for i,e in enumerate(out[0]):
            if e['initial_partial']:continue
            gaps.append(e['calendar_lag_sessions'])
            for cell in e['cells']:
                self.assertEqual(cell['sigma63'],float(expected.at[pd.Timestamp(e['date']),cell['symbol']]))
        self.assertTrue(any(g!=42 for g in gaps))

    def test_all_logical_physical_prefixes_and_future_poison(self):
        args=fixture();full=run(args)
        for n in range(len(args[1]['dates'])+1):
            lo=run(args,n);pa=physical(args,n);po=run(pa)
            audit.record_check(lo[0],full[0][:n],'Logical feature prefix');audit.record_check(lo[0],po[0],'Physical feature prefix')
            self.assertEqual(lo[-1]['events'],full[-1]['events'][:n]);verify(args,lo,n);verify(pa,po)
        p,m,c,a,r=copy.deepcopy(args);m['symbols'][-1]=['bad'];p['adj']=p['adj'].astype(object)
        p['adj'].loc[p['adj'].index>args[1]['dates'][1]]='future-invalid'
        run((p,m,c,a,r),2)


class SelectionTests(unittest.TestCase):
    def test_cached_selection_never_rebuilds_source(self):
        args=fixture();out=run(args);f,o,s,p,d=out
        with patch.object(audit,'independent_features',side_effect=AssertionError('No source')):
            audit.verify_selection(args[3],args[1],f,o,scores=s,pre_scores=p,audit=d)

    def test_tampered_rank_fallback_and_full_score_rejected(self):
        args=fixture();out=run(args)
        bad=copy.deepcopy(out);bad[0][1]['cells'][0]['calendar_rank']+=.01
        with self.assertRaises(ValueError):verify(args,bad)
        bad=copy.deepcopy(out);bad[0][1]['branch']='missing_calendar'
        with self.assertRaises(ValueError):verify(args,bad)
        bad=copy.deepcopy(out);bad[2][audit.ARMS[1]][1,-1]+=1e-12
        with self.assertRaises(ValueError):verify(args,bad)

    def test_original_support_single_episode_and_budget_are_preserved(self):
        args=fixture();out=run(args);base=audit.original_base(args[3],args[1],len(args[1]['dates']))
        for arm in audit.ARMS:
            fs,ch=audit.episode(out[3][arm],base);np.testing.assert_array_equal(fs,out[2][arm])
            self.assertTrue(all(len(ids) in (0,2,3) for ids in ch))
        self.assertEqual(out[-1]['counts']['original_episode_calls'],2)
        self.assertTrue(all(min(e['original_remaining_support'])>=.38-1e-12 for e in out[-1]['events']))

    def test_two_or_cash_rules_unchanged(self):
        args=fixture();args[3]['eligible'][1,:]=False;args[3]['eligible'][1,:2]=True
        args[3]['eligible'][2,:]=False;args[3]['eligible'][2,0]=True
        out=run(args);verify(args,out)
        for arm in audit.ARMS:self.assertEqual(len(out[1][arm][1]['symbols']),2);self.assertEqual(out[1][arm][2]['symbols'],[])


class EarlyGateTests(unittest.TestCase):
    def test_source_gate_never_calls_episode_or_candidate_selection(self):
        args=fixture();p,m,c,a,r=args;f=run(args)[0]
        with patch.object(audit,'selection_oracle',side_effect=AssertionError('candidate')), \
             patch.object(audit,'episode',side_effect=AssertionError('episode')):
            receipt=audit.verify_source_features(p,m,c,a,r,f)
        self.assertTrue(receipt['passed']);self.assertEqual(receipt['episode_calls'],0)
        self.assertFalse(receipt['candidate_selection_computed'])
        f[1]['cells'][0]['original_rank']+=.01
        with self.assertRaises(ValueError):audit.verify_source_features(p,m,c,a,r,f)

    def test_incumbent_gate_has_no_candidate_dependency_and_rejects_control_tamper(self):
        args=fixture();p,m,c,a,r=args;f,o,s,pre,d=run(args);inc=audit.ARMS[0]
        with patch.object(audit,'selection_oracle',side_effect=AssertionError('candidate')):
            receipt=audit.verify_incumbent(a,m,o[inc],s[inc],pre[inc])
        self.assertTrue(receipt['passed']);self.assertFalse(receipt['candidate_selection_computed'])
        s[inc][1,0]+=.01
        with self.assertRaises(ValueError):audit.verify_incumbent(a,m,o[inc],s[inc],pre[inc])


class AccountingTests(unittest.TestCase):
    @staticmethod
    def args():
        h=audit.load_pinned(audit.ACCOUNT_TEST,audit.ACCOUNT_TEST_SHA,'calendar248_v2_account_fixture')
        p,o,orders,exports,results,m=h.account_fixture();first=next(iter(orders))
        return p,o,{k:copy.deepcopy(orders[first]) for k in audit.ARMS},{k:copy.deepcopy(exports[first]) for k in audit.ARMS},\
            {k:copy.deepcopy(results[first]) for k in audit.ARMS},m

    def test_six_paths_thirty_slices_and_twelve_training_groups(self):
        report=audit.audit_outcomes(*self.args())
        self.assertEqual((report['checked_paths'],report['checked_slices'],report['checked_training_groups']),(6,30,12))

    def test_zero_cost_fee_and_cost_ledger_mutation_rejected(self):
        args=list(self.args());args[3][audit.ARMS[1]]['same_close_zero']['HOLDINGS'][0]['fee']=1e-15
        with self.assertRaises(ValueError):audit.audit_outcomes(*args)
        args=list(self.args());args[4][audit.ARMS[1]]['next_open_cost']['slices']['later34']['cagr']+=.1
        with self.assertRaises((ValueError,AssertionError)):audit.audit_outcomes(*args)


class BilateralCross(unittest.TestCase):
    def test_both_fixtures_full_artifacts_and_all_prefixes(self):
        import monthly_calendar_clock248_v2_kernel as producer
        import test_monthly_calendar_clock248_v2_kernel as fixtures
        own=fixture();own_args=(helper().real_study(),own[3],own[1],own[0],own[2],own[4])
        for args in (fixtures.fixture(),own_args):
            study,a,m,p,c,r=args
            gate=lambda f,nn=len(m['dates']):audit.verify_source_features(p,m,c,a,r,f,nn)
            incgate=lambda o,s,pre,nn=len(m['dates']):audit.verify_incumbent(a,m,o,s,pre,nn)
            full=producer.prepare(*args,len(m['dates']),source_validator=gate,incumbent_validator=incgate)
            f,o,s,pre,d=full;audit.verify_prepared(p,m,c,a,r,f,o,scores=s,pre_scores=pre,audit=d)
            for n in range(len(m['dates'])+1):
                f,o,s,pre,d=producer.prepare(*args,n,
                    source_validator=lambda f:audit.verify_source_features(p,m,c,a,r,f,n),
                    incumbent_validator=lambda o,s,pre:audit.verify_incumbent(a,m,o,s,pre,n))
                audit.verify_prepared(p,m,c,a,r,f,o,scores=s,pre_scores=pre,audit=d,n=n)
                o,s,pre,d=producer.decisions(study,a,m,full[0],n,
                    incumbent_validator=lambda o,s,pre:audit.verify_incumbent(a,m,o,s,pre,n))
                audit.verify_selection(a,m,full[0],o,scores=s,pre_scores=pre,audit=d,n=n)
                pa=fixtures.physical(args,n);ps,aa,mm,pp,cc,rr=pa
                f,o,s,pre,d=producer.prepare(*pa,n,
                    source_validator=lambda f:audit.verify_source_features(pp,mm,cc,aa,rr,f,n),
                    incumbent_validator=lambda o,s,pre:audit.verify_incumbent(aa,mm,o,s,pre,n))
                audit.verify_prepared(pp,mm,cc,aa,rr,f,o,scores=s,pre_scores=pre,audit=d,n=n)


if __name__=='__main__':
    if '--self-only' in sys.argv:
        sys.argv.remove('--self-only')
        suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(c) for c in (SourceTests,SelectionTests,EarlyGateTests,AccountingTests))
        result=unittest.TextTestRunner(verbosity=2).run(suite);sys.exit(not result.wasSuccessful())
    unittest.main(verbosity=2)
