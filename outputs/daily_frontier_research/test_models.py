import unittest,copy
import numpy as np
import pandas as pd
import models

class LabelBoundaryTests(unittest.TestCase):
 def test_all_new_model_targets_exclude_unmatured_outcomes(self):
  dates=pd.bdate_range('2015-01-01','2019-01-31');rng=np.random.RandomState(55)
  x=pd.DataFrame(rng.uniform(size=(len(dates),6)),index=dates,columns=list('ABCDEF'))
  f=dict(valid=x.notna(),train=x.notna(),event=x.notna(),X=[x,x**2,x*.8,x*.3],groups=[x]*4,y=.1*x-.05)
  cut=dates.get_loc(pd.Timestamp('2019-01-01'))-6
  a,_,audit,_=models.prepare_models(f);changed=copy.deepcopy(f);changed['y'].iloc[cut:]=1000
  b,_,_,_=models.prepare_models(changed)
  for name in a:pd.testing.assert_frame_equal(a[name],b[name])
  self.assertTrue(all(row['latest_label_session']<row['first_prediction'] for rows in audit.values() for row in rows))

if __name__=='__main__':unittest.main()
