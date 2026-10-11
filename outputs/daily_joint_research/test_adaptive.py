import unittest
import numpy as np
import pandas as pd
from adaptive import matured_skill,choices


class AdaptiveTests(unittest.TestCase):
 def test_unmatured_outcome_changes_no_current_choice(self):
  idx=pd.bdate_range('2018-01-01','2019-03-29');rng=np.random.RandomState(7)
  pred=pd.DataFrame(rng.rand(len(idx),8),index=idx);y=pred+rng.normal(0,.02,pred.shape);event=pred.notna()
  before=matured_skill(pred,y,event);cutoff=idx.get_loc(pd.Timestamp('2019-01-01'))-6
  changed=y.copy();changed.iloc[cutoff:]*=-1
  after=matured_skill(pred,changed,event)
  old,_=choices(idx,{'a':before},['a']);new,_=choices(idx,{'a':after},['a'])
  pd.testing.assert_series_equal(old.loc['2019-01'],new.loc['2019-01'])
  self.assertTrue(old.loc['2019-01'].eq('a').all())
 def test_nonpositive_validation_disables_model(self):
  idx=pd.bdate_range('2018-01-01','2019-03-29');skill=pd.Series(-.1,index=idx)
  selected,audit=choices(idx,{'bad':skill},['bad'])
  self.assertTrue(selected.isna().all());self.assertTrue(all(v['selected_on']<v['first_signal'] for v in audit))


if __name__=='__main__':unittest.main()
