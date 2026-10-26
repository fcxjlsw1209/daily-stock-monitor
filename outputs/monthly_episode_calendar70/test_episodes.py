"""Synthetic official-calendar, paired-label and original-estimator tests only."""
import copy
import importlib.util
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

P=Path(__file__).resolve().parent

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value

s=module('calendar_episode_helper_test',P/'episodes.py')
ep=module('calendar_episode_original_test',P.parent/'monthly_daily_episodes/episodes.py')
calmod=module('calendar_episode_calendar_test',P.parent/'monthly_earnings_repair/research_calendar.py')
CAL=calmod.market_calendar('2010-01-01','2030-12-31')


def fixture(start='2018-01-01',end='2022-06-30'):
    dates=s._sessions(CAL,start,end);n=len(dates);t=np.arange(n)
    px=pd.DataFrame(dict(A=100*np.exp(.0007*t+.006*np.sin(t/7)),B=100*np.exp(.0004*t),
                         SPY=100*np.exp(.0002*t),QQQ=100*np.exp(.0003*t)),index=dates)
    panel=dict(adj=px,open=px*1.001,close=px,high=px+1,low=px-1,volume=px*0+1000)
    bundle=([px*0 for _ in range(8)],pd.DataFrame(0.,index=dates,columns=['x','y','z']),
            pd.DataFrame(1,index=dates,columns=px.columns),pd.DataFrame(True,index=dates,columns=px.columns),px*0+.02)
    return panel,bundle


def forecast(events,day,variant,eligible=True,signal=1):
    meta=dict(dates=[day],symbols=[['A']])
    queries=dict(shape=np.zeros((1,1,8)),market=np.zeros((1,3)),signal=np.array([[signal]]),vol=np.full((1,1),.02))
    return s.forecast(events,meta,np.array([[eligible]],bool),queries,variant,ep.estimate)


class EpisodeTests(unittest.TestCase):
    def test_monthends_and_future_endpoints_official_no_lockout(self):
        panel,bundle=fixture('2018-01-01','2026-09-24')
        plans=s.monthend_plans(panel['adj'].index,CAL)
        self.assertEqual(str(plans[-1]['date'].date()),'2026-08-31')
        self.assertNotIn(pd.Timestamp('2026-09-24'),[p['date'] for p in plans])
        counts=set()
        for p in plans:
            self.assertEqual(len(p['path_21'])-1,21)
            self.assertEqual(p['path_21'][-1],p['exit_21'])
            self.assertEqual(p['path_calendar'][-1],p['exit_calendar'])
            self.assertEqual(p['common_maturity'],max(p['exit_21'],p['exit_calendar']))
            counts.add(len(p['path_calendar'])-1)
        self.assertTrue({19,20,22,23}.issubset(counts))
        # Good Friday closes the month early; no observed-row approximation.
        self.assertIn(pd.Timestamp('2024-03-28'),[p['date'] for p in plans])
        events,_=s.build_events(panel,bundle,['A'],CAL,ep.label_path)
        self.assertEqual(len(events['dates']),len(plans))
        self.assertTrue((np.diff(events['anchor_index'])<21).any())

    def test_truncated_future_never_shortens_calendar_or21_label(self):
        panel,bundle=fixture('2026-07-01','2026-09-24')
        events,audit=s.build_events(panel,bundle,['A'],CAL,ep.label_path)
        i=np.flatnonzero(events['dates']==np.datetime64('2026-08-31'))[0]
        self.assertEqual(events['exit_calendar'][i],np.datetime64('2026-09-30'))
        self.assertGreater(events['exit_21'][i],np.datetime64('2026-09-24'))
        self.assertFalse(events['common_good'][i])
        self.assertTrue(np.isnan(events['utility_calendar'][i]))
        self.assertEqual(audit['final_present_monthend'],'2026-08-31')

    def test_anchor_selection_only_current_original_good_and_positive_type(self):
        panel,bundle=fixture('2020-01-01','2020-08-31');days=[p['date'] for p in s.monthend_plans(panel['adj'].index,CAL)]
        bundle[2].loc[days[0],'A']=0;bundle[2].loc[days[1],'A']=-1;bundle[3].loc[days[2],'A']=False
        bundle[2].loc[days[3],'A']=2;bundle[2].loc[days[4],'A']=3
        events,_=s.build_events(panel,bundle,['A'],CAL,ep.label_path)
        np.testing.assert_array_equal(events['dates'],np.array(days[3:],dtype='datetime64[D]'))
        np.testing.assert_array_equal(events['signal'][:2],[2,3])
        self.assertFalse(events['common_good'][-1])

    def test_same_anchors_despite_future_missing_labels_and_coverage(self):
        panel,bundle=fixture();expected,_=s.build_events(panel,bundle,['A'],CAL,ep.label_path)
        changed={key:value.copy() for key,value in panel.items()};changed['adj'].loc['2021-02-01':,'A']=np.nan
        got,_=s.build_events(changed,bundle,['A'],CAL,ep.label_path)
        for key in ('dates','anchor_index','signal','shape','market','entry','exit_21','exit_calendar','common_maturity'):
            np.testing.assert_array_equal(got[key],expected[key])
        self.assertLess(got['common_good'].sum(),expected['common_good'].sum())

    def test_original_utility_exact_for_both_actual_exits(self):
        panel,bundle=fixture();events,_=s.build_events(panel,bundle,['A'],CAL,ep.label_path)
        px=panel['adj'];opening=px*panel['open']/panel['close']
        for i in np.flatnonzero(events['common_good']):
            anchor=int(events['anchor_index'][i])
            for kind in ('21','calendar'):
                end=int(px.index.get_loc(pd.Timestamp(events['exit_'+kind][i])))
                y,u=ep.label_path(px.A.to_numpy(),opening.A.to_numpy(),px.SPY.to_numpy(),opening.SPY.to_numpy(),anchor,end-anchor)
                self.assertEqual(y,events['excess_'+kind][i]);self.assertEqual(u,events['utility_'+kind][i])
        self.assertFalse(np.array_equal(events['utility_21'],events['utility_calendar'],equal_nan=True))

    def test_missing_one_arm_path_excludes_both_without_removing_anchor(self):
        panel,bundle=fixture();panel['adj'].loc['2021-03-01','A']=np.nan
        events,_=s.build_events(panel,bundle,['A'],CAL,ep.label_path)
        i=np.flatnonzero(events['dates']==np.datetime64('2021-01-29'))[0]
        self.assertTrue(events['valid_calendar'][i]);self.assertFalse(events['valid_21'][i])
        self.assertFalse(events['common_good'][i])
        for variant in s.VARIANTS:self.assertTrue(np.isnan(s.estimator_view(events,variant)['utility_21'][i]))

    def test_missing_expected_exchange_session_is_not_skipped(self):
        panel,bundle=fixture();missing=pd.Timestamp('2021-02-01')
        panel={key:value.drop(index=missing) for key,value in panel.items()}
        events,_=s.build_events(panel,bundle,['A'],CAL,ep.label_path)
        i=np.flatnonzero(events['dates']==np.datetime64('2021-01-29'))[0]
        self.assertFalse(events['valid_21'][i]);self.assertFalse(events['valid_calendar'][i])
        self.assertEqual(events['exit_calendar'][i],np.datetime64('2021-02-26'))

    def test_common_maturity_withholds_earlier_finished_arm_and_separate_exit_audit(self):
        panel,bundle=fixture();events,_=s.build_events(panel,bundle,['A'],CAL,ep.label_path)
        index=np.flatnonzero(events['dates']==np.datetime64('2021-01-29'))[0]
        self.assertEqual(events['exit_calendar'][index],np.datetime64('2021-02-26'))
        self.assertEqual(events['common_maturity'][index],np.datetime64('2021-03-02'))
        before=[];after=[]
        for variant in s.VARIANTS:
            before.append(forecast(events,'2021-02-26',variant)[1]['records'][0])
            after.append(forecast(events,'2021-03-02',variant)[1]['records'][0])
        self.assertEqual(before[0]['observations'],before[1]['observations'])
        self.assertEqual(after[0]['observations'],after[1]['observations'])
        self.assertEqual(after[0]['observations'],before[0]['observations']+1)
        self.assertEqual(after[1]['latest_common_maturity'],'2021-03-02')
        self.assertEqual(after[1]['latest_actual_exit'],'2021-02-26')
        self.assertNotIn('latest_label_end',after[1])

    def test_same_evidence_and_exact_original_h21_estimator_formula(self):
        panel,bundle=fixture();events,_=s.build_events(panel,bundle,['A'],CAL,ep.label_path)
        q=dict(symbol='A',date=np.datetime64('2021-03-31'),eligible=True,shape=np.zeros(8),market=np.zeros(3),signal=1,vol=.02)
        for variant in s.VARIANTS:
            view=s.estimator_view(events,variant)
            want=ep.estimate(view,q,s.MODEL21)[0]
            self.assertEqual(forecast(events,'2021-03-31',variant)[0][0,0],want)
            mask=(view['dates']>=q['date']-np.timedelta64(1827,'D'))&(view['dates']<q['date'])&(view['ends_21']<=q['date'])&np.isfinite(view['utility_21'])
            y=view['utility_21'][mask];age=(q['date']-view['dates'][mask]).astype('timedelta64[D]').astype(float)/30.4375
            w=2**(-age/24);mass=w.sum();mu=w@y/mass;var=w@(y-mu)**2/mass;neff=mass**2/(w@w)
            shrunk=np.sign(mu)*max(abs(mu)-np.sqrt(var/neff),0)*mass/(mass+6)
            self.assertAlmostEqual(want,.5*np.tanh(shrunk/max(.02*np.sqrt(21),.01)),places=15)

    def test_future_path_and_label_changes_do_not_change_past_mature_predictions(self):
        panel,bundle=fixture();events,_=s.build_events(panel,bundle,['A'],CAL,ep.label_path)
        changed=copy.deepcopy(events);future=changed['common_maturity']>np.datetime64('2021-03-31')
        changed['utility_21'][future]=999.;changed['utility_calendar'][future]=-999.
        for variant in s.VARIANTS:
            np.testing.assert_array_equal(forecast(events,'2021-03-31',variant)[0],forecast(changed,'2021-03-31',variant)[0])
        cut=pd.Timestamp('2021-03-31');small={key:value.loc[:cut] for key,value in panel.items()}
        trimmed=([value.loc[:cut] for value in bundle[0]],*[value.loc[:cut] for value in bundle[1:]])
        short,_=s.build_events(small,trimmed,['A'],CAL,ep.label_path)
        for key in ('dates','signal','exit_21','exit_calendar','common_maturity'):
            np.testing.assert_array_equal(short[key],events[key][events['dates']<=np.datetime64(cut,'D')])
        for variant in s.VARIANTS:
            np.testing.assert_array_equal(forecast(events,'2021-03-31',variant)[0],forecast(short,'2021-03-31',variant)[0])

    def test_coldstart_no_signal_ineligible_missing_query_and_empty_history_zero(self):
        panel,bundle=fixture();events,_=s.build_events(panel,bundle,['A'],CAL,ep.label_path)
        for variant in s.VARIANTS:
            self.assertEqual(forecast(events,'2018-02-01',variant)[0][0,0],0.)
            self.assertEqual(forecast(events,'2021-03-31',variant,eligible=False)[0][0,0],0.)
            for signal in (-1,0,2,3):self.assertEqual(forecast(events,'2021-03-31',variant,signal=signal)[0][0,0],0.)
        empty={key:value[:0] for key,value in events.items()}
        for variant in s.VARIANTS:self.assertEqual(forecast(empty,'2021-03-31',variant)[0][0,0],0.)
        meta=dict(dates=['2021-03-31'],symbols=[['A']]);eligible=np.ones((1,1),bool)
        queries=dict(shape=np.full((1,1,8),np.nan),market=np.zeros((1,3)),signal=np.ones((1,1),int),vol=np.full((1,1),.02))
        for variant in s.VARIANTS:
            self.assertFalse(s.forecast(events,meta,eligible,queries,variant,ep.estimate)[0].any())
        self.assertTrue(eligible.all())

    def test_original_query_layout_and_reference_forecast_exact(self):
        panel,bundle=fixture();meta=dict(dates=['2021-03-31'],symbols=[['B','A']])
        query=s.query_table(bundle,meta)
        np.testing.assert_array_equal(query['shape'][0],np.column_stack([v.loc['2021-03-31',['B','A']] for v in bundle[0]]))
        dates=np.array(['2020-01-01','2020-02-03','2020-03-02','2020-04-01'],dtype='datetime64[D]')
        events=dict(symbol=np.full(4,'A'),dates=dates,ends_5=dates+np.timedelta64(7,'D'),signal=np.ones(4,int),utility_5=np.full(4,.1))
        got,_=s.forecast(events,meta,np.ones((1,2),bool),query,'reference',ep.estimate)
        for j,symbol in enumerate(meta['symbols'][0]):
            q=dict(symbol=symbol,date=np.datetime64(meta['dates'][0]),eligible=True,shape=query['shape'][0,j],market=query['market'][0],signal=query['signal'][0,j],vol=query['vol'][0,j])
            self.assertEqual(got[0,j],ep.estimate(events,q,s.MODEL5)[0])

    def test_pair_mask_maturity_and_variant_tampering_rejected(self):
        panel,bundle=fixture();events,_=s.build_events(panel,bundle,['A'],CAL,ep.label_path)
        with self.assertRaises(ValueError):s.estimator_view(events,'native5')
        bad=copy.deepcopy(events);bad['common_good'][0]=not bad['common_good'][0]
        with self.assertRaises(ValueError):s.estimator_view(bad,s.VARIANTS[0])
        bad=copy.deepcopy(events);bad['common_maturity'][0]-=np.timedelta64(1,'D')
        with self.assertRaises(ValueError):s.estimator_view(bad,s.VARIANTS[1])


if __name__=='__main__':unittest.main()
