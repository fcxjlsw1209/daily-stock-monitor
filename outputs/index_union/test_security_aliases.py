import unittest
from security_aliases import compile_aliases,resolve

class SecurityAliasTests(unittest.TestCase):
    def docs(self,edges):return [('source.json',{'aliases':edges},'checksum')]
    def edge(self,old,new,day,kind='ticker_change'):
        return dict(old=old,new=new,effective=day,published=day,kind=kind,source='https://issuer.example/release')
    def test_same_equity_chain_preserves_id(self):
        rows=compile_aliases(self.docs([self.edge('RIMM','BBRY','2013-02-04'),self.edge('BBRY','BB','2017-10-16')]))
        a=resolve(rows,'RIMM','2012-01-01');b=resolve(rows,'BB','2018-01-01')
        self.assertEqual(a['security_id'],b['security_id'])
        self.assertIsNone(resolve(rows,'RIMM','2014-01-01'))
        self.assertIsNone(resolve(rows,'BB','2012-01-01'))
        self.assertFalse(a['point_in_time_identity_certified'])
    def test_merger_and_spinoff_not_collapsed(self):
        rows=compile_aliases(self.docs([self.edge('LMCA','STRZA','2013-01-14','parent_rename_after_spinoff'),self.edge('WAG','WBA','2014-12-31','holding_company_reorganization')]))
        self.assertEqual(rows,[])
    def test_branch_rejected(self):
        with self.assertRaises(ValueError):compile_aliases(self.docs([self.edge('A','B','2012-01-01'),self.edge('A','C','2013-01-01')]))
    def test_future_evidence_not_certified_as_contemporaneous(self):
        e=self.edge('A','B','2012-01-01');e['published']='2013-01-01'
        with self.assertRaises(ValueError):compile_aliases(self.docs([e]))

if __name__=='__main__':unittest.main()
