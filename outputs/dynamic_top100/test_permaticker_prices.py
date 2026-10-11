import unittest
import pandas as pd
from audit_permaticker_prices import select_identity,eligible_dates,IDENTIFIER
class PermaTests(unittest.TestCase):
 def test_reused_symbol_not_selected(self):
  good=dict(permaTicker=IDENTIFIER,name='Paramount Global - Class B',ticker='PARA',isActive=False)
  self.assertEqual(select_identity([dict(permaTicker='other',ticker='PARA',name='Banzai'),good]),good)
 def test_wrong_name_rejected(self):
  with self.assertRaises(ValueError):select_identity([dict(permaTicker=IDENTIFIER,name='Banzai',ticker='PARA',isActive=False)])
 def test_lifecycle_boundaries(self):
  self.assertEqual(eligible_dates(pd.Series(['2022-02-16','2022-02-17','2025-08-06','2025-08-07'])).tolist(),[False,True,True,False])
if __name__=='__main__':unittest.main()
