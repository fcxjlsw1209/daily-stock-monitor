"""PEP-specific synthetic tests plus shared strict source/geometry tests."""
import unittest
import after152_pep_extract as m
import test_after152_mu_extract as shared


class PEPTests(shared.ExtractionChecks,unittest.TestCase):
    MODULE=m

    def test_53_week_year_not_forced_to_365_days(self):
        out=self.out(year=2016)
        self.assertEqual(out['periods']['current']['days_inclusive'],371)
        self.assertTrue(out['complete_required_fields'],out['gaps'])

    def test_52_week_year_not_forced_to_365_days(self):
        out=self.out(year=2016,short_week=True)
        self.assertEqual(out['periods']['current']['days_inclusive'],364)
        self.assertTrue(out['complete_required_fields'],out['gaps'])

    def test_both_pinned_official_publication_hosts(self):
        for host in m.PROFILE['hosts']:
            raw,c=self.make();url='https://'+host+'/synthetic';c['source']['url']=c['source_url']=url
            for f in c['fields'].values():f['source']['url']=url
            self.assertTrue(m.extract_document(raw,c)['complete_required_fields'])


if __name__=='__main__':unittest.main(verbosity=2)
