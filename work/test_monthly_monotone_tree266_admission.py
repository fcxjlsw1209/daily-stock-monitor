"""Synthetic admission tests; no market datasets or portfolio replay."""
import copy
import unittest
import monthly_monotone_tree266_admission as h


def training():
    r=dict(training=dict(cagr=.5,sharpe=1.7,max_drawdown=-.3),early=dict(cagr=.4),recent=dict(cagr=.6))
    m={a:{p:copy.deepcopy(r) for p in ('same_close_zero','next_open_zero')} for a in h.ARMS}
    for p in m[h.ARMS[2]]:m[h.ARMS[2]][p]['training']['cagr']=.52
    o={a:[dict(symbols=['A','B','C']) for i in range(122)] for a in h.ARMS}
    for i in (1,54,55):o[h.ARMS[2]][i]['symbols']=['A','B','D']
    return m,o


class AdmissionTests(unittest.TestCase):
    def test_only_monotone_candidate_can_promote(self):
        m,o=training();self.assertEqual(h.select_first88(m,o)['primary'],h.ARMS[2])
        for p in m[h.ARMS[1]]:m[h.ARMS[1]][p]['training']['cagr']=9.
        result=h.select_first88(m,o);self.assertFalse(result['admitted']);self.assertEqual(result['primary'],h.ARMS[0])

    def test_inc_gain_and_open_noninferiority(self):
        for p,v in [('same_close_zero',.504),('next_open_zero',.49)]:
            m,o=training();m[h.ARMS[2]][p]['training']['cagr']=v
            self.assertFalse(h.select_first88(m,o)['admitted'])

    def test_each_reference_requires_three_changes_across_segments(self):
        m,o=training();o[h.ARMS[1]]=copy.deepcopy(o[h.ARMS[2]]);o[h.ARMS[1]][2]['symbols']=['Q']
        self.assertFalse(h.select_first88(m,o)['admitted'])
        for j in (54,55):o[h.ARMS[1]][j]['symbols']=['Q']
        self.assertTrue(h.select_first88(m,o)['admitted'])

    def test_inc_changed_months_require_both_eras(self):
        for ids in ((54,55),(1,2,3),(53,54,55)):
            m,o=training();o[h.ARMS[2]]=copy.deepcopy(o[h.ARMS[0]])
            for i in ids:o[h.ARMS[2]][i]['symbols']=['D']
            self.assertFalse(h.select_first88(m,o)['admitted'])

    def test_order_only_change_not_support(self):
        m,o=training()
        for i in (1,54,55):o[h.ARMS[2]][i]['symbols']=['C','A','B']
        self.assertFalse(h.select_first88(m,o)['admitted'])

    def test_both_references_both_paths_risk_segments(self):
        for ref in (h.ARMS[0],h.ARMS[1]):
            for path in ('same_close_zero','next_open_zero'):
                for group,key,v in [('training','sharpe',1.76),('training','max_drawdown',-.26),('early','cagr',.43),('recent','cagr',.63)]:
                    m,o=training();m[ref][path][group][key]=v
                    self.assertFalse(h.select_first88(m,o)['admitted'])

    def test_cagr_only_roundoff(self):
        m,o=training();m[h.ARMS[2]]['same_close_zero']['training']['cagr']=.505-5e-13
        self.assertTrue(h.select_first88(m,o)['admitted'])
        m[h.ARMS[2]]['same_close_zero']['training']['sharpe']=1.65-5e-13
        self.assertFalse(h.select_first88(m,o)['admitted'])

    def test_undefined_candidate_or_control_rejects(self):
        for arm in h.ARMS[1:]:
            for path in ('same_close_zero','next_open_zero'):
                m,o=training();m[arm][path]['training'].update(sharpe=None,daily_return_sd=0.,sharpe_status='undefined_sharpe',sharpe_reason='zero_daily_return_sd')
                self.assertFalse(h.select_first88(m,o)['admitted'])

    def test_bad_metrics_hardfail(self):
        for value in (None,float('nan'),float('inf'),True):
            m,o=training();m[h.ARMS[2]]['same_close_zero']['training']['cagr']=value
            with self.assertRaises(ValueError):h.select_first88(m,o)
        m,o=training();m[h.ARMS[0]]['same_close_zero']['training'].update(sharpe=None,daily_return_sd=0.,sharpe_status='undefined_sharpe',sharpe_reason='zero_daily_return_sd')
        with self.assertRaises(ValueError):h.select_first88(m,o)

    def test_later_full_cost_cannot_rescue(self):
        m,o=training();m[h.ARMS[2]]['same_close_zero']['training']['cagr']=.1
        expected=h.select_first88(m,o)
        for i in range(88,122):o[h.ARMS[2]][i]['symbols']=['LATER_WINNER']
        self.assertEqual(h.select_first88(m,o),expected)
        m[h.ARMS[2]]['next_open_cost']={'full122':{'cagr':100.}}
        with self.assertRaises(ValueError):h.select_first88(m,o)

    def test_missing_arm_short_order_rejected(self):
        m,o=training();o[h.ARMS[2]]=o[h.ARMS[2]][:87]
        with self.assertRaises(ValueError):h.select_first88(m,o)
        m,o=training();m.pop(h.ARMS[1])
        with self.assertRaises(ValueError):h.select_first88(m,o)

    def test_fixed_three_roles(self):
        self.assertEqual(h.ARMS, ('incumbent_local67', 'unconstrained_weak_tree261_control', 'monotone_raw_weak_tree_candidate'))

    def test_future_orders_not_accessed(self):
        class NoFuture(list):
            def __getitem__(self, index):
                if not isinstance(index, int) or index >= 88:
                    raise AssertionError('Future order accessed')
                return super().__getitem__(index)
        m, o = training()
        expected = h.select_first88(m, o)
        self.assertEqual(h.select_first88(m, {a: NoFuture(v) for a, v in o.items()}), expected)

    def test_inputs_unchanged(self):
        m, o = training(); before = copy.deepcopy((m, o))
        h.select_first88(m, o)
        self.assertEqual((m, o), before)

    def test_each_reference_genuine_open_noninferiority(self):
        for ref in h.ARMS[:2]:
            m, o = training()
            m[ref]['next_open_zero']['training']['cagr'] = .53
            self.assertFalse(h.select_first88(m, o)['admitted'])

    def test_undefined_sharpe_requires_exact_zero_evidence(self):
        for arm in h.ARMS[1:]:
            for value in (None, 1e-30, -1., float('nan')):
                m, o = training()
                m[arm]['same_close_zero']['training'].update(sharpe=None, daily_return_sd=value,
                    sharpe_status='undefined_sharpe', sharpe_reason='zero_daily_return_sd')
                with self.assertRaises(ValueError): h.select_first88(m, o)

    def test_exact_first88_required_no_cost_substitution(self):
        m, o = training(); expected=h.select_first88(m,o)
        self.assertEqual(expected,h.select_first88(m,{a:v[:88] for a,v in o.items()}))
        m[h.ARMS[2]]['same_close_cost']=m[h.ARMS[2]].pop('same_close_zero')
        with self.assertRaises(ValueError): h.select_first88(m,o)


if __name__ == '__main__': unittest.main()
