"""Synthetic identity/phase-boundary tests; no real source prices."""
import copy,unittest
import peer239_source_host as h

class IdentityTests(unittest.TestCase):
    def setUp(self):
        self.meta={'dates':['2016-08-17'],'symbols':[['CHL','META','BRK-B']]}
        self.records=[{'symbol':'META','url':'issuer/meta'},{'symbol':'BRK-B','url':'issuer/berkshire'}]
        self.sec={'cik':'0001117795','name':'CHINA MOBILE LTD /ADR/'}
        self.ev={'companies':[{'symbol':'CHL','business_evidence_public_by':'2016-04-27',
                            'source_ids':['chl_2015_20f','chl_20160427_issuer_release']}]}
    def call(self):return h.identity_keys(self.meta,self.records,self.sec,self.ev)
    def test_exact_keys_aliases(self):
        self.assertEqual(self.call(),{'research_equity_FB':'issuer/meta','BRK.B':'issuer/berkshire','CHL':'sec_cik:0001117795'})
    def test_inputs_not_mutated(self):
        before=copy.deepcopy((self.meta,self.records,self.sec,self.ev));self.call()
        self.assertEqual(before,(self.meta,self.records,self.sec,self.ev))
    def test_no_prepublic_chl(self):
        self.meta['dates']=['2016-04-27']
        with self.assertRaises(ValueError):self.call()
    def test_wrong_cik(self):
        self.sec['cik']='1117796'
        with self.assertRaises(ValueError):self.call()
    def test_wrong_issuer_name(self):
        self.sec['name']='unrelated'
        with self.assertRaises(ValueError):self.call()
    def test_cap_override_disallowed(self):
        self.records.append({'symbol':'CHL','url':'invented'})
        with self.assertRaises(ValueError):self.call()
    def test_key_collision(self):
        self.records[0]['url']='sec_cik:0001117795'
        with self.assertRaises(ValueError):self.call()
    def test_conflicting_alias(self):
        self.records.append({'symbol':'research_equity_FB','url':'different'})
        with self.assertRaises(ValueError):self.call()
    def test_missing_evidence(self):
        self.ev['companies']=[]
        with self.assertRaises(ValueError):self.call()
    def test_future_evidence_rejected(self):
        self.ev['companies'][0]['business_evidence_public_by']='2026-01-01'
        with self.assertRaises(ValueError):self.call()

class IdentityScopeTests(unittest.TestCase):
    def setUp(self):
        self.meta={'dates':['2016-08-17','2016-08-31'],'symbols':[['CHL'],['CHL']]}
        self.supp={'issuer_key':'sec_cik:0001117795','valid_from_assumption':'2016-04-28',
                   'observed_query_dates':['2016-08-17','2016-08-31']}
    def test_exact_scope(self):h.validate_identity_scope(self.meta,self.supp)
    def test_scope_expansion_rejected(self):
        self.meta['dates'].append('2026-08-31');self.meta['symbols'].append(['CHL'])
        with self.assertRaises(ValueError):h.validate_identity_scope(self.meta,self.supp)
    def test_scope_omission_rejected(self):
        self.supp['observed_query_dates']=self.supp['observed_query_dates'][:1]
        with self.assertRaises(ValueError):h.validate_identity_scope(self.meta,self.supp)
    def test_prepublic_scope_rejected(self):
        self.meta['dates'][0]=self.supp['observed_query_dates'][0]='2016-04-27'
        with self.assertRaises(ValueError):h.validate_identity_scope(self.meta,self.supp)

if __name__=='__main__':unittest.main()
