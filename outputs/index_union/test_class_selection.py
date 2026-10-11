import unittest
from class_selection import select_classes

class ClassSelectionTests(unittest.TestCase):
    def setUp(self):
        self.sessions = ['2026-01-%02d'%d for d in range(1,31)]
        self.day = self.sessions[24]
        self.members = [dict(issuer_id='issuer',security_ids=['B','A'],indexes=['NDX'])]
        self.bars = [dict(security_id=s,date=d,close=10,volume=v,basis='raw',verified=True)
                     for s,v in [('A',100),('B',200)] for d in self.sessions[:25]]
        self.coverage = {s:dict(verified=True,signal_session=self.day,source_sha256='receipt') for s in ['A','B']}
    def run_case(self):
        return select_classes(self.members,self.bars,self.day,self.sessions,self.coverage)
    def test_selects_dollar_volume_not_symbol_preference(self):
        self.assertEqual(self.run_case()[0]['security_id'],'B')
    def test_future_volume_cannot_change_selection(self):
        self.bars.append(dict(security_id='A',date=self.sessions[25],close=1000,volume=1e9,basis='raw',verified=True))
        self.assertEqual(self.run_case()[0]['security_id'],'B')
    def test_tie_uses_permanent_id(self):
        for b in self.bars:b['volume']=100
        self.assertEqual(self.run_case()[0]['security_id'],'A')
    def test_adjusted_units_rejected(self):
        self.bars[0]['basis']='adjusted'
        with self.assertRaises(ValueError):self.run_case()
    def test_missing_class_certificate_rejected(self):
        del self.coverage['A']
        with self.assertRaises(ValueError):self.run_case()
    def test_short_history_rejected(self):
        self.bars=self.bars[10:]
        with self.assertRaises(ValueError):self.run_case()
    def test_single_security_needs_no_class_comparison(self):
        self.members[0]['security_ids']=['A'];self.bars=[];self.coverage={}
        self.assertEqual(self.run_case()[0]['security_id'],'A')
    def test_duplicate_bar_rejected(self):
        self.bars.append(self.bars[0].copy())
        with self.assertRaises(ValueError):self.run_case()

if __name__=='__main__':unittest.main()
