import unittest
import pandas as pd
from consolidate_issuer_classes import select_panel
class ClassPanelTests(unittest.TestCase):
 def test_trailing_selection_and_unavailable_class(self):
  idx=pd.bdate_range('2020-01-01',periods=70);cols=['A','B']
  data={k:pd.DataFrame(10.,index=idx,columns=cols) for k in ['close','adj','opening','high','low','volume']}
  data['volume']['B']=20.;data['eligible']=pd.DataFrame(True,index=idx,columns=cols);data['issuers']={'A':'a','B':'b'}
  out,audit=select_panel(data,{'A':'issuer','B':'issuer'})
  self.assertFalse(out['eligible'].iloc[:19].any().any());self.assertTrue(out['eligible']['B'].iloc[19:].all());self.assertFalse(out['eligible']['A'].any())
  data['close'].iloc[-1,1]=float('nan');out2,_=select_panel(data,{'A':'issuer','B':'issuer'})
  self.assertTrue(out2['eligible']['A'].iloc[-1]);pd.testing.assert_frame_equal(out2['eligible'].iloc[:-1],out['eligible'].iloc[:-1])
if __name__=='__main__':unittest.main()
