import unittest
import pandas as pd
from secondary_prices import parse_history,verify_no_actions
class SecondaryTests(unittest.TestCase):
 def html(self,adj='100'):
  return '<table><tr>'+''.join('<th>'+s+'</th>' for s in ['Date','Open','High','Low','Close','Adj. Close','Change','Volume'])+'</tr><tr>'+''.join('<td>'+s+'</td>' for s in ['Sep 23, 2026','100','101','99','100',adj,'0%','1,000'])+'</tr></table>'
 def test_history_and_cutoff(self):
  frame=parse_history(self.html(),'2026-09-23');self.assertEqual(frame.iloc[0]['Volume'],1000)
  with self.assertRaises(ValueError):parse_history(self.html(),'2026-09-22')
 def test_missing_adjusted_not_inferred(self):
  with self.assertRaises(ValueError):parse_history(self.html('-'),'2026-09-23')
 def test_actions_unknown_or_split_blocked(self):
  start=pd.Timestamp('2026-09-22');end=pd.Timestamp('2026-09-23')
  div='There is no dividend history available'
  verify_no_actions(div,'Last Split Date Dec 16, 2024',start,end)
  with self.assertRaises(ValueError):verify_no_actions(div,'Last Split Date Sep 22, 2026',start,end)
  with self.assertRaises(ValueError):verify_no_actions(div,'unknown',start,end)
  with self.assertRaises(ValueError):verify_no_actions('unknown','Last Split Date Dec 16, 2024',start,end)
