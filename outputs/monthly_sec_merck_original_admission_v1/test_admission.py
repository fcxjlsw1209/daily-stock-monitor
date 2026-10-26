import unittest
from copy import deepcopy
import admission as a
B=b'''<html><p class="date-text">May 1, 2020 6:30 am EDT</p><p>Merck (NYSE: MRK), known as MSD outside the United States and Canada, today announced financial results for the first quarter of 2020.</p><table><tr><td>CONSOLIDATED STATEMENT OF INCOME AMOUNTS IN MILLIONS</td></tr><tr><td></td><td>1Q20</td><td>1Q19</td></tr><tr><td>Sales</td><td>$ 200</td><td>$ 100</td></tr><tr><td>Costs, Expenses and Other</td></tr></table></html>'''
def fixture():
 s={'accession':'x','accessionNumber':'x','form':'10-Q','filing_date':'2020-05-01','filingDate':'2020-05-01','report_date':'2020-03-31','reportDate':'2020-03-31','supported_form':True,'metadata_issues':[],'sources':[{'table':'recent','row':0}],'first_usable_date':'2020-05-04'}
 vals=[dict(start='2020-01-01',end='2020-03-31',val=200000000,accn='x',form='10-Q',filed='2020-05-01',fy=2020,fp='Q1'),dict(start='2019-01-01',end='2019-03-31',val=100000000,accn='x',form='10-Q',filed='2020-05-01',fy=2020,fp='Q1')]
 pair={'taxonomy':'us-gaap','tag':'Revenues','unit':'USD'}
 for i,key in enumerate(['current','prior']):pair[key]={**vals[i],'row_index':i,'row_indices':[i],'duration_days':91-i}
 r={'symbol':'MRK','candidate_cik':a.CIK,'signal_date':'2020-05-29','catalog':{'selected':s,'status':'selected','coverage_complete':True,'coverage_start':'2019-05-01','coverage_end':'2020-05-29','reasons':[],'coverage_reasons':[],'tied_accessions':[]},'pairing':{'status':'paired','structural_pair_available':True,'pairs':[pair],**{k:False for k in a.FLAGS}},**{k:False for k in a.FLAGS}}
 d={'target_accession':'x','expected_report_date':'2020-03-31','expected_filing_date':'2020-05-01'}
 f={'cik':310158,'facts':{'us-gaap':{'Revenues':{'units':{'USD':vals}}}}}
 recent={k:[s[k]] for k in ['accessionNumber','form','filingDate','reportDate']}
 cat={'coverage_ranges':[{'start':'2018-01-01','end':'2021-01-01'}],'holes':[]}
 return [0,r,d,a.parse_release(B),f,recent,cat,['2020-05-01','2020-05-04','2020-05-29']]
class TestAdmission(unittest.TestCase):
 def test_valid(self):self.assertTrue(a.validate_record(*fixture())['semantic_admitted'])
 def reject(self,fn):
  x=fixture();fn(x)
  with self.assertRaises(ValueError):a.validate_record(*x)
 def test_later_release(self):self.reject(lambda x:x[3].update(release_date='2020-05-05'))
 def test_same_day_available(self):self.reject(lambda x:x[1]['catalog']['selected'].update(first_usable_date='2020-05-01'))
 def test_wrong_raw_value(self):self.reject(lambda x:x[4]['facts']['us-gaap']['Revenues']['units']['USD'][0].update(val=199000000))
 def test_wrong_statement_value(self):self.reject(lambda x:x[3]['periods'][0].update(val=199000000))
 def test_wrong_statement_period(self):self.reject(lambda x:x[3]['periods'][0].update(start='2020-01-02'))
 def test_pair_missing(self):self.reject(lambda x:x[1]['pairing'].update(pairs=[]))
 def test_every_pair_required(self):
  def change(x):
   p=deepcopy(x[1]['pairing']['pairs'][0]);p['tag']='SalesRevenueNet';x[1]['pairing']['pairs'].append(p)
  self.reject(change)
 def test_flag_mutation(self):self.reject(lambda x:x[1].update(factor_ready=True))
 def test_wrong_cik(self):self.reject(lambda x:x[1].update(candidate_cik='0000000001'))
 def test_coverage_hole(self):self.reject(lambda x:x[6]['holes'].append({'start':'2020-01-01','end':'2020-01-02'}))
 def test_missing_raw_filing(self):self.reject(lambda x:x[5]['accessionNumber'].__setitem__(0,'other'))
 def test_newer_amendment_blocks(self):
  def change(x):
   for k,v in {'accessionNumber':'y','form':'10-Q/A','filingDate':'2020-05-20','reportDate':'2020-03-31'}.items():x[5][k].append(v)
  self.reject(change)
 def test_ambiguous_same_date_blocks(self):
  def change(x):
   for k,v in {'accessionNumber':'y','form':'10-Q','filingDate':'2020-05-01','reportDate':'2020-03-31'}.items():x[5][k].append(v)
  self.reject(change)
 def parse_reject(self,b):
  with self.assertRaises(ValueError):a.parse_release(b)
 def test_ambiguous_date(self):self.parse_reject(B.replace(b'</html>',b'<p class="date-text">May 1, 2020 6:30 am EDT</p></html>'))
 def test_footer_only_identity(self):self.parse_reject(B.replace(b'Merck (NYSE: MRK)',b'Merck').replace(b'</html>',b'<footer>NYSE: MRK</footer></html>'))
 def test_table_year_mismatch(self):self.parse_reject(B.replace(b'1Q19',b'1Q18'))
 def test_missing_units(self):self.parse_reject(B.replace(b'AMOUNTS IN MILLIONS',b''))
 def test_ambiguous_table(self):self.parse_reject(B.replace(b'</html>',B[B.index(b'<table>'):B.index(b'</table>')+8]+b'</html>'))
 def test_not_full_income(self):self.parse_reject(B.replace(b'Costs, Expenses and Other',b'Profit'))
if __name__=='__main__':unittest.main()
