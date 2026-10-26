import unittest
from shell_pool_sensitivity168 import choose_report, next_session, rerank, revalue


class SensitivityTests(unittest.TestCase):
    sessions=['2016-07-28','2016-07-29','2016-08-01','2016-08-02']

    def test_announcement_same_day_excluded(self):
        reports=[{'period':'2016Q2','period_end':'2016-06-30','release':{'stated_date':'2016-07-28'}}]
        with self.assertRaises(ValueError):choose_report(reports,'2016-07-28',self.sessions)
        report,usable=choose_report(reports,'2016-07-29',self.sessions)
        self.assertEqual((report['period'],usable),('2016Q2','2016-07-29'))

    def test_weekend_strict_session(self):
        self.assertEqual(next_session('2016-07-29',self.sessions),'2016-08-01')
        self.assertEqual(next_session('2016-07-30',self.sessions),'2016-08-01')

    def test_future_report_ignored(self):
        old={'period':'old','period_end':'2016-06-30','release':{'stated_date':'2016-07-28'}}
        future={'period':'new','period_end':'2016-07-31','release':{'stated_date':'2016-08-01'}}
        self.assertEqual(choose_report([old,future],'2016-08-01',self.sessions)[0],old)

    def test_classes_and_units(self):
        self.assertEqual(revalue({'A':100,'B':50},{'class_A':{'close':10},'class_B':{'close':20}}),1000)
        with self.assertRaises(ValueError):revalue({'A':100,'B':50},{'class_A':{'close':0},'class_B':{'close':20}})

    def test_insert_absent_issuer_changes_only_boundary(self):
        old=[{'symbol':'S%02d'%i,'market_cap_usd':100-i} for i in range(40)]
        ranked=rerank(old,1000)
        self.assertEqual(ranked[0]['symbol'],'SHEL')
        self.assertEqual({r['symbol'] for r in ranked[:32]}-{r['symbol'] for r in old[:32]},{'SHEL'})
        self.assertEqual(len(old),40)

    def test_existing_issuer_replaced_once_and_others_preserved(self):
        old=[{'symbol':'S%02d'%i,'market_cap_usd':100-i,'source':'x'} for i in range(39)]
        old.insert(0,{'symbol':'SHEL','market_cap_usd':999})
        ranked=rerank(old,1)
        self.assertEqual(sum(r['symbol']=='SHEL' for r in ranked),1)
        self.assertEqual(ranked[:-1],old[1:])


if __name__=='__main__':unittest.main()
