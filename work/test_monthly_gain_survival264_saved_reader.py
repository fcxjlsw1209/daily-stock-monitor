"""Synthetic checks only for root saved-evidence selection; no replay inputs."""
import unittest
import monthly_gain_survival264_saved_reader as reader

class SavedSelection(unittest.TestCase):
    def setUp(self):
        self.p=reader.load_parent()
        def metrics(c):
            m=dict(cagr=c,sharpe=1.,max_drawdown=-.2,daily_return_sd=.01,sharpe_status='defined',sharpe_reason=None)
            return {p:{g:dict(m) for g in ('training','early','recent')} for p in reader.PATHS[:2]}
        self.m={a:metrics(.2 if a!=reader.ARMS[2] else .21) for a in reader.ARMS}
        self.o={a:[dict(symbols=['A','B','C']) for _ in range(122)] for a in reader.ARMS}
        for i in (2,4,60):self.o[reader.ARMS[2]][i]=dict(symbols=['A','B','D'])
    def pick(self):return reader.selection(self.p,self.m,self.o)
    def test_candidate_only_when_both_references_pass(self):
        self.assertTrue(self.pick()['admitted']);self.assertEqual(self.pick()['primary'],reader.ARMS[2])
        self.m[reader.ARMS[1]]['same_close_zero']['training']['cagr']=.3
        self.assertFalse(self.pick()['admitted']);self.assertEqual(self.pick()['primary'],reader.ARMS[0])
    def test_open_zero_must_pass(self):
        self.m[reader.ARMS[2]]['next_open_zero']['training']['cagr']=.19
        self.assertFalse(self.pick()['admitted'])
    def test_risk_guard(self):
        self.m[reader.ARMS[2]]['same_close_zero']['training']['sharpe']=.949
        self.assertFalse(self.pick()['admitted'])
    def test_recent_era_required(self):
        self.m[reader.ARMS[2]]['same_close_zero']['recent']['cagr']=.174
        self.assertFalse(self.pick()['admitted'])
    def test_permutations_do_not_count(self):
        for i in (2,4,60):self.o[reader.ARMS[2]][i]=dict(symbols=['C','A','B'])
        self.assertFalse(self.pick()['admitted'])
    def test_later_only_cannot_rescue(self):
        for i in (2,4,60):self.o[reader.ARMS[2]][i]=dict(symbols=['A','B','C'])
        for i in range(88,122):self.o[reader.ARMS[2]][i]=dict(symbols=['A','B','D'])
        self.assertFalse(self.pick()['admitted'])
    def test_undefined_candidate_sharpe_is_inadmissible(self):
        self.m[reader.ARMS[2]]['same_close_zero']['training'].update(sharpe=None,daily_return_sd=0.,sharpe_status='undefined_sharpe',sharpe_reason='zero_daily_return_sd')
        self.assertFalse(self.pick()['admitted'])
    def test_cost_diagnostic_not_selection_input(self):
        self.m[reader.ARMS[2]]['next_open_cost']={}
        with self.assertRaises(ValueError):self.pick()

if __name__=='__main__':unittest.main(verbosity=2)
