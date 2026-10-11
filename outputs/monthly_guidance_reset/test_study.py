import unittest
import numpy as np
import pandas as pd
from study import calendar_momentum,picks,read,P
from search_rules import evaluate

class StudyTests(unittest.TestCase):
    def test_calendar_skips_two_month_end_prices(self):
        dates=pd.bdate_range('2018-01-01','2021-02-28')
        px=pd.DataFrame({'X':np.exp(np.arange(len(dates))*.001)},index=dates)
        self.assertAlmostEqual(calendar_momentum(px,12).loc['2020-12-31','X'],px.loc['2020-10-30','X']/px.loc['2019-12-31','X']-1)

    def test_calendar_prefix_does_not_read_future(self):
        dates=pd.bdate_range('2018-01-01','2021-12-31')
        px=pd.DataFrame({'X':np.exp(np.arange(len(dates))*.001)},index=dates)
        full=calendar_momentum(px,14)
        for cut in ['2020-11-13','2020-12-31','2021-04-07']:
            self.assertAlmostEqual(calendar_momentum(px.loc[:cut],14).iloc[-1,0],full.loc[cut,'X'])

    def test_later_outcomes_cannot_change_calibration_score(self):
        d=dict(eligible=np.ones((2,4),bool),labels=np.array([[1,1,1,0],[0,1,1,1]],bool),gross=np.ones((2,4))*1.1,complete=np.ones((2,4),bool))
        sc=np.tile(np.arange(4),(2,1));mask=np.array([1,0],bool)
        before=evaluate(sc,d,mask)
        d['labels'][1]=False;d['gross'][1]=999.;d['complete'][1]=False
        after=evaluate(sc,d,mask)
        for key in before:np.testing.assert_equal(before[key],after[key])

    def test_ineligible_large_score_cannot_enter(self):
        ids,good=picks(np.array([[100.,3.,2.,1.]]),np.array([[False,True,True,True]]))
        self.assertEqual(ids.tolist(),[[1,2,3]])
        self.assertTrue(good.all())

    def test_single_eligible_stays_cash(self):
        d=dict(eligible=np.array([[1,0,0,0]],bool),labels=np.ones((1,4),bool),gross=np.ones((1,4))*2.,complete=np.ones((1,4),bool))
        r=evaluate(np.ones((1,4)),d,np.array([True]))
        self.assertEqual(r['log_return'],0.);self.assertEqual(r['shared'],0)

    def test_annual_labels_fully_mature(self):
        for row in read(P/'TRAINING_AUDIT.json'):
            self.assertLessEqual(row['latest_label_end'],row['cutoff'])
            self.assertGreaterEqual(pd.Timestamp(row['train_start']),pd.Timestamp(row['cutoff'])-pd.DateOffset(years=5))
            self.assertLess(row['normal_equation_residual'],1e-7)

    def test_frozen_budget(self):
        for row in read(P/'FROZEN_RULES.json')['leaders'].values():
            r=row['rule'];weights=r['weights'];ml=r.get('ml_weight',0)
            self.assertGreaterEqual(weights[r['core']],.5-1e-8)
            self.assertTrue(all(v>=0 for v in weights.values()))
            self.assertLessEqual(ml,.1)
            self.assertAlmostEqual(sum(weights.values())+ml,1.,places=7)

    def test_no_author_identity_predictor(self):
        names=read(P/'INPUTS.json')['ml_names']
        self.assertFalse(any(any(w in n for w in ['symbol','holdings','author','month_id']) for n in names))

if __name__=='__main__':unittest.main()
