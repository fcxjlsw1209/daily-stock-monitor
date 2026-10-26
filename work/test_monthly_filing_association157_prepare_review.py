"""Fabricated data only: independent157 arithmetic, masks, and selector checks."""
from copy import deepcopy
import math
import sys
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import monthly_filing_association157_prepare_review as r
from test_monthly_cash_quality146_prepare_review import score_fixture


def partition(t):
    labels=['N']*210;owners=[None]*210;bundles=[]
    for k,a in enumerate((t-251,t-170)):
        bundles.append(dict(anchor=a,event_ids=['original'+str(k)],forms=['10-Q'],amendment=False,
            owned_E_days=list(range(a+2,a+21)),full19=True))
        for day in range(a,a+21):
            labels[day-(t-251)]='B' if day<a+2 else 'E';owners[day-(t-251)]=a
    return dict(first_day=t-251,last_day=t-42,labels=''.join(labels),owner_anchor_indices=owners,
        bundles=bundles,counts={lab:labels.count(lab) for lab in 'EBN'},full19_original_bundles=2)


def fixture(months=2):
    names=['AAPL','MSFT']+['S'+str(i) for i in range(30)]+['E'+str(i) for i in range(17)]
    days=[str(day.date()) for day in pd.bdate_range('2020-01-01',periods=800)]
    positions=[550+21*i for i in range(months)]
    meta=dict(dates=[days[t] for t in positions],symbols=[names[:] for _ in positions],etfs=names[32:])
    changes=np.array([.002*math.sin(i*.3)+.0002 for i in range(len(days)-1)])
    spy=100*np.exp(np.r_[0,np.cumsum(changes)]);stock=70*np.exp(np.r_[0,np.cumsum(1.25*changes+.0006)])
    prices=pd.DataFrame(dict(SPY=spy,AAPL=stock,MSFT=2*stock),index=pd.to_datetime(days))
    cutoffs={day:int(pd.Timestamp(day+'T20:00:00Z').value) for day in meta['dates']};rows=[]
    for i,t in enumerate(positions):
        for j,symbol in enumerate(names):
            usable=j<2
            rows.append(dict(signal_index=i,candidate_index=j,symbol=symbol,decision_date=days[t],
                decision_cutoff=days[t]+'T20:00:00Z',decision_cutoff_ns=cutoffs[days[t]],signal_session_index=t,
                issuer_id='issuer:'+symbol if j<32 else None,research_eligible=usable,reasons=[] if usable else ['source_hold'],
                partition=partition(t) if usable else None,event_ids=['original0','original1'] if usable else [],
                context_original_event_ids=[],source_refs=[]))
    return meta,rows,days,prices,cutoffs


def broad_fixture():
    meta,_,_,_,_=fixture(1);names=meta['symbols'][0]
    days=[str(day.date()) for day in pd.date_range('2010-01-01',periods=122,freq='MS')]
    meta=dict(meta,dates=days,symbols=[names[:] for _ in days]);rows=[]
    for i,day in enumerate(days):
        for j,symbol in enumerate(names):
            rows.append(dict(signal_index=i,candidate_index=j,symbol=symbol,decision_date=day,
                decision_cutoff=day+'T20:00:00Z',decision_cutoff_ns=int(pd.Timestamp(day+'T20:00:00Z').value),
                research_eligible=j<32,reasons=[] if j<32 else ['ETF'],issuer_id='issuer:'+symbol if j<32 else None))
    return meta,rows


class FeatureTests(unittest.TestCase):
    def test_common_mask_known_residual_and_counters(self):
        result=r.features_oracle(*fixture());self.assertEqual(int(result['arrays']['common_available'].sum()),4)
        for key in r.HANDS:np.testing.assert_array_equal(np.isfinite(result['arrays'][key]),result['arrays']['common_available'])
        cell=result['cells'][0];self.assertAlmostEqual(cell['beta'],1.25,places=10)
        self.assertAlmostEqual(cell['raw'][r.HANDS[0]],21*.0006,places=11)
        self.assertAlmostEqual(cell['raw'][r.HANDS[2]],0,places=11)
        self.assertEqual([result[k] for k in r.COUNTERS],[4,4,0])

    def test_prefix_future_isolation_and_consumed_price_cache_invalidation(self):
        args=list(fixture());cache={};full=r.features_oracle(*args,numeric_cache=cache)
        for n in (0,1,2):
            got=r.features_oracle(*args,n=n,numeric_cache=cache)
            for key in r.EXTRA:np.testing.assert_array_equal(got['arrays'][key],full['arrays'][key][:n])
        last=args[2][args[1][0]['signal_session_index']-42]
        args[3].loc[args[3].index>pd.Timestamp(last),'AAPL']=np.nan
        prefix=r.features_oracle(*args,n=1,numeric_cache=cache)
        for key in r.EXTRA:np.testing.assert_array_equal(prefix['arrays'][key],full['arrays'][key][:1])
        args[3].loc[last,'AAPL']=np.nan
        self.assertFalse(r.features_oracle(*args,n=1,numeric_cache=cache)['arrays']['common_available'][0,0])

    def test_required_price_boundaries_share_missing_mask(self):
        for offset in (-504,-252,-251,-42):
            args=list(fixture(1));t=args[1][0]['signal_session_index'];args[3].loc[args[2][t+offset],'AAPL']=np.nan
            result=r.features_oracle(*args);self.assertFalse(result['arrays']['common_available'][0,0])
            self.assertEqual(result['cells'][0][r.COUNTERS[0]],0)
            self.assertTrue(all(np.isnan(result['arrays'][key][0,0]) for key in r.HANDS))

    def test_held_row_has_no_measurement(self):
        args=list(fixture(1));args[1][0].update(research_eligible=False,reasons=['known_conflict'])
        with patch.object(r,'measure_oracle',wraps=r.measure_oracle) as measure:result=r.features_oracle(*args)
        self.assertEqual(measure.call_count,1);self.assertEqual(result['cells'][0]['reason'],'known_conflict')
        self.assertIsNone(result['cells'][0]['price_window'])

    def test_zero_variance_attempted_not_completed(self):
        args=list(fixture(1));args[3]['SPY']=1.;result=r.features_oracle(*args)
        self.assertEqual([result[k] for k in r.COUNTERS],[2,0,0])
        self.assertEqual(result['cells'][0]['reason'],'zero_preformation_market_variance')
        self.assertFalse(result['arrays']['common_available'].any())

    def test_nonpositive_infinite_and_missing_calendar_prices(self):
        for value in (0.,-1.,np.inf):
            args=list(fixture(1));args[3].loc[args[2][46],'AAPL']=value
            self.assertFalse(r.features_oracle(*args)['arrays']['common_available'][0,0])
        args=list(fixture(1));args[3]=args[3].drop(args[3].index[50])
        with self.assertRaises(ValueError):r.features_oracle(*args)

    def test_same_date_wrong_close_and_bad_source_axes(self):
        args=list(fixture(1));row=args[1][0];row['decision_cutoff']=row['decision_date']+'T19:00:00Z'
        row['decision_cutoff_ns']=int(pd.Timestamp(row['decision_cutoff']).value)
        with self.assertRaises(ValueError):r.features_oracle(*args)
        for key,value in (('signal_session_index',551),('candidate_index',1),('symbol','wrong')):
            args=list(fixture(1));args[1][0][key]=value
            with self.subTest(field=key),self.assertRaises(ValueError):r.features_oracle(*args)

    def test_incomplete_grid_and_etf_eligibility_rejected(self):
        args=list(fixture(1));args[1]=args[1][:-1]
        with self.assertRaises(ValueError):r.features_oracle(*args)
        args=list(fixture(1));args[1][-1].update(research_eligible=True,reasons=[],issuer_id='ETF')
        with self.assertRaises(ValueError):r.features_oracle(*args)

    def test_partition_detects_forged_labels_bundles_ids_owners(self):
        for change in ('labels','full','amendment','duplicate','owners'):
            args=list(fixture(1));row=args[1][0];p=row['partition']
            if change=='labels':p['labels']='N'*210
            if change=='full':p['full19_original_bundles']=3
            if change=='amendment':p['bundles'][0]['forms']=['10-Q/A']
            if change=='duplicate':row['event_ids']*=2
            if change=='owners':p['owner_anchor_indices'][2]=None
            with self.subTest(change=change),self.assertRaises(ValueError):r.features_oracle(*args)

    def test_formation_end_changes_signal_without_refitting_beta(self):
        args=list(fixture(1));before=r.features_oracle(*args);t=args[1][0]['signal_session_index']
        args[3].loc[args[2][t-42],'AAPL']*=math.exp(.01);after=r.features_oracle(*args)
        self.assertEqual(before['cells'][0]['beta'],after['cells'][0]['beta'])
        self.assertAlmostEqual(after['cells'][0]['raw'][r.HANDS[0]]-before['cells'][0]['raw'][r.HANDS[0]],.001,places=11)

    def test_inputs_unchanged(self):
        args=fixture(1);before=deepcopy(args);r.features_oracle(*args)
        self.assertEqual(args[:3],before[:3]);pd.testing.assert_frame_equal(args[3],before[3]);self.assertEqual(args[4],before[4])


class CoverageTests(unittest.TestCase):
    def test_eight_source_post_price_floors_and_original_denominator(self):
        meta,rows=broad_fixture();source=r.coverage_oracle(meta,rows)
        self.assertEqual(len(source['checks']),8);self.assertTrue(source['coverage_floor_passed'])
        self.assertEqual(source['eras']['all122']['stock_slots'],3904)
        got=r.coverage_oracle(meta,rows,np.zeros((122,49),bool))
        self.assertFalse(got['coverage_floor_passed']);self.assertFalse(any(got['checks'].values()))

    def test_issuers_are_not_tickers(self):
        meta,rows=broad_fixture()
        for row in rows:
            if row['research_eligible']:row['issuer_id']='ONE_ISSUER'
        result=r.coverage_oracle(meta,rows);self.assertEqual(result['eras']['first88']['distinct_issuers'],1)
        self.assertFalse(result['coverage_floor_passed'])

    def test_later34_cannot_select_and_mask_cannot_expand(self):
        meta,rows=broad_fixture();mask=np.array([row['research_eligible'] for row in rows]).reshape(122,49)
        mask[88:]=False;self.assertTrue(r.coverage_oracle(meta,rows,mask)['coverage_floor_passed'])
        mask[0,-1]=True
        with self.assertRaises(ValueError):r.coverage_oracle(meta,rows,mask)


class ScoreTests(unittest.TestCase):
    def test_point01_pre_score_and_one_episode_per_arm(self):
        a,m,hand=score_fixture();a['eligible'][0,0]=False;helper=r.old.original_helper()
        with patch.object(r.old,'original_helper',return_value=helper),patch.object(helper,'episode',wraps=helper.episode) as episode:
            _,_,deltas=r.score_oracle(a,m,dict.fromkeys(r.HANDS,hand))
        self.assertEqual(episode.call_count,4);expected=np.where(a['eligible'] & np.isfinite(hand),.01*np.nan_to_num(hand),0.)
        np.testing.assert_array_equal(deltas[r.NAMES[0]],np.zeros(hand.shape))
        for name in r.NAMES[1:]:np.testing.assert_array_equal(deltas[name],expected)

    def test_123_prefixes_488_orders_and_exact_zero_missing_fallback(self):
        original,m,hand=score_fixture();a={key:np.concatenate([v]*61,axis=0) for key,v in original.items()}
        hand=np.concatenate([hand]*61,axis=0);dates=pd.date_range('2010-01-01',periods=122,freq='MS')
        m=dict(m,symbols=m['symbols']*61,dates=[str(d.date()) for d in dates],ends=[str((d+pd.offsets.MonthEnd()).date()) for d in dates])
        ss,cc,_=r.score_oracle(a,m,dict.fromkeys(r.HANDS,hand))
        self.assertEqual(sum(len(v) for v in r.order_oracle(m,ss,cc).values()),488)
        for n in range(123):
            ps,pc,_=r.score_oracle(a,m,dict.fromkeys(r.HANDS,hand),n)
            for name in r.NAMES:
                self.assertEqual(pc[name],cc[name][:n]);self.assertEqual(ps[name].tobytes(),ss[name][:n].tobytes())
        for fill in (0.,np.nan):
            empty=np.full(a['core'].shape,np.nan)
            for i,names in enumerate(m['symbols']):
                for j,symbol in enumerate(names):
                    if symbol not in m['etfs']:empty[i,j]=fill
            ps,pc,_=r.score_oracle(a,m,dict.fromkeys(r.HANDS,empty))
            for name in r.NAMES[1:]:
                self.assertEqual(pc[name],cc[r.NAMES[0]]);self.assertEqual(ps[name].tobytes(),ss[r.NAMES[0]].tobytes())

    def test_common_mask_mismatch_and_etf_signal_rejected(self):
        a,m,hand=score_fixture();hands={k:hand.copy() for k in r.HANDS};hands[r.HANDS[-1]][0,0]=np.nan
        with self.assertRaises(ValueError):r.score_oracle(a,m,hands)
        hand[:,-1]=.5
        with self.assertRaises(ValueError):r.score_oracle(a,m,dict.fromkeys(r.HANDS,hand))

    def test_plan_hash_required_before_io(self):
        with patch.object(r.old.Reader,'sha',side_effect=AssertionError('Unexpected read')):
            for bad in (None,'z'*64,'a'*63):
                with self.subTest(hash=bad),self.assertRaises(ValueError):r.audit('/synthetic','plan.json',bad)

    def test_no_new_producer_module_imported(self):
        for name in ('monthly_filing_association157_runner','after156_association_measure','monthly_revenue_persistence156_runner'):
            self.assertNotIn(name,sys.modules)


class SourceReviewContractTests(unittest.TestCase):
    def test_prereview_proposal_windows_masks_and_activity(self):
        measurement=dict(clock_policy_id=r.CLOCK,exact_acceptance_certified=False,beta_prices=[-504,-252],
            formation_prices=[-252,-42],aligned_prices=463,beta='centered_252_return_OLS_slope_no_formation_intercept',
            whole='21*mean(all210_beta_residuals)',density='nE/(nE+nN)',
            contrast='21*(mean(E_beta_residuals)-mean(N_beta_residuals))',return_hand='h(raw/0.05)',
            density_hand='raw_density',h='x/(1+abs(x))',minimum_full19_bundles=2,minimum_E=38,minimum_N=80,
            original_candidate_axis=[122,49],unchanged_original_arrays=27,original_eligibility_unchanged=True,
            placement='add_0.01_h_to_original_pre_score_then_one_original_episode_selector',
            common_mask='same_source_partition_price_beta_availability_for_all_three_hands',
            no_parameter_search=True,new_predictive_model_fits=0,etf_features=False,monthly_equal_weight_positions=3,
            intramonth_trades=False)
        proposal=dict(round=157,arms=list(r.NAMES),primary_challenger=r.NAMES[3],dose=.01,
            status='FINAL_PROSPECTIVE_SPECIFICATION_BEFORE_REGISTRATION',measurement=measurement,
            actual_new_factors=0,actual_new_orders=0,actual_new_returns=0,
            actual_new_nuisance_estimates=0,actual_new_predictive_model_fits=0)
        r.validate_prereview_proposal(proposal)
        for key,value in [('beta_prices',[-503,-252]),('formation_prices',[-251,-42]),('minimum_full19_bundles',1),
                          ('common_mask','separate'),('exact_acceptance_certified',True)]:
            bad=deepcopy(proposal);bad['measurement'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):r.validate_prereview_proposal(bad)
        for key in ('actual_new_factors','actual_new_orders','actual_new_returns',
                    'actual_new_nuisance_estimates','actual_new_predictive_model_fits'):
            bad=deepcopy(proposal);bad[key]=True
            with self.subTest(key=key),self.assertRaises(ValueError):r.validate_prereview_proposal(bad)

    def test_new_producer_import_forbidden_before_review(self):
        r.independent_imports()
        for name in r.FORBIDDEN:
            with patch.dict(sys.modules,{name:object()}),self.subTest(module=name),self.assertRaises(ValueError):
                r.independent_imports()

    def test_source_manifest_rejects_wrong_clock_and_partial_or_extra_outputs(self):
        manifest=dict(schema_version=1,source_only=True,strategy_registration=False,clock_policy_id=r.CLOCK,
            outputs={'ROWS.json':{},'COVERAGE_GATE.json':{}})
        r.validate_source_manifest(manifest)
        for key,value in [('schema_version',True),('clock_policy_id','exact_acceptance'),
                ('source_only',False),('strategy_registration',True),('outputs',{'ROWS.json':{}}),
                ('outputs',dict(manifest['outputs'],EXTRA={}))]:
            broken=deepcopy(manifest);broken[key]=value
            with self.subTest(field=key,value=value),self.assertRaises(ValueError):r.validate_source_manifest(broken)

    def test_research_review_does_not_certify_exact_clock(self):
        review=dict(passed=True,actual_grid_verified=True,research_eligibility_verified=True,exact_acceptance_certified=False,
            checks=dict.fromkeys(r.SOURCE_REVIEW_CHECKS,True));r.validate_source_review(review)
        for key in r.SOURCE_REVIEW_CHECKS:
            broken=deepcopy(review);broken['checks'][key]=False
            with self.subTest(check=key),self.assertRaises(ValueError):r.validate_source_review(broken)
        for key in ('passed','actual_grid_verified','research_eligibility_verified','exact_acceptance_certified'):
            broken=deepcopy(review);broken[key]=not broken[key]
            with self.subTest(check=key),self.assertRaises(ValueError):r.validate_source_review(broken)


if __name__=='__main__':unittest.main()
