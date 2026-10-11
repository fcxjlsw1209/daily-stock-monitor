import copy,unittest
from finalize_available_research import validate_run_status
from union_execution import ACCOUNTS
class AvailableAdmissionTests(unittest.TestCase):
 def setUp(self):
  self.status={'all_six_completed':True,'manifest_sha256':'data','assumptions_sha256':'assumptions','accounts':{k:{'status':'complete','sessions':4215} for k in ACCOUNTS}}
 def test_accepts_complete_and_same_inputs(self):
  validate_run_status(self.status,'data','assumptions')
 def test_rejects_changed_inputs_or_assumptions(self):
  for a,b in [('changed','assumptions'),('data','changed')]:
   with self.assertRaises(ValueError):validate_run_status(self.status,a,b)
 def test_rejects_partial_or_failed_account(self):
  for key,value in [('sessions',4214),('status','blocked')]:
   d=copy.deepcopy(self.status);d['accounts']['baseline'][key]=value
   with self.assertRaises(ValueError):validate_run_status(d,'data','assumptions')
  del self.status['accounts']['baseline']
  with self.assertRaises(ValueError):validate_run_status(self.status,'data','assumptions')
if __name__=='__main__':unittest.main()
