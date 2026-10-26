"""Synthetic source indexing only; never loads real filings or market inputs."""
import hashlib
from pathlib import Path
import tempfile
import unittest

import after157_annual_original_body_inventory as target

ACC='0000000001-20-000001'
OTHER='0000000002-20-000002'
DOC='annual.htm'
URL='https://www.sec.gov/Archives/edgar/data/1/000000000120000001/annual.htm'
INFO=dict(cik='0000000001',primaryDocument=DOC)
HTML=b'<html><body>FORM 10-K ANNUAL REPORT PART I text PART II text SIGNATURES</body></html>'


class IdentityTests(unittest.TestCase):
    def test_exact_primary_identity(self):
        self.assertEqual(target.sec_identity(URL),dict(cik='0000000001',accession=ACC,document=DOC))

    def test_ixviewer_decodes_without_network(self):
        url='https://www.sec.gov/ixviewer/ix.html?doc=%2FArchives%2Fedgar%2Fdata%2F1%2F000000000120000001%2Fannual.htm'
        self.assertEqual(target.sec_identity(url),target.sec_identity(URL))

    def test_lookalike_domain_and_prose_are_not_sec_identity(self):
        for bad in (URL.replace('www.sec.gov','sec.gov.evil.example'),'Read '+URL,'https://issuer.example/annual.htm'):
            self.assertIsNone(target.sec_identity(bad))

    def test_end_date_and_ticker_do_not_map_accession(self):
        record=dict(symbol='ABC',reportDate='2019-12-31',source=dict(path='a.pdf',sha256='0'*64))
        self.assertEqual(list(target.records(record,{ACC:INFO})),[])

    def test_exact_url_maps_without_ticker(self):
        rows=list(target.records(dict(url=URL),{ACC:INFO}))
        self.assertEqual(rows[0][2],[ACC])

    def test_record_accession_propagates_to_its_source_only(self):
        rows=list(target.records([dict(accession=ACC,source=dict(path='a.html',sha256='0'*64)),
                                  dict(accession=OTHER,source=dict(path='b.html',sha256='0'*64))],{ACC:INFO}))
        self.assertTrue(any(x[0]=='$[0].source' for x in rows))
        self.assertFalse(any(x[0].startswith('$[1]') for x in rows))

    def test_explicit_cdn_catalog_mapping_locates_without_certifying(self):
        obj=dict(source_identity=dict(catalog_accession=ACC,publisher_chain_verified=False),
                 target=dict(raw_path='year.pdf',raw_sha256='0'*64))
        rows=list(target.records(obj,{ACC:INFO}))
        self.assertTrue(any(x[0]=='$.target' and x[2]==[ACC] for x in rows))

    def test_accession_dictionary_key_locates_nested_record(self):
        rows=list(target.records({ACC:dict(path='a.html',sha256='0'*64)},{ACC:INFO}))
        self.assertEqual(rows[0][2],[ACC])


class BodyTests(unittest.TestCase):
    def test_full_structural_body_candidate_not_period_admission(self):
        value=target.body_shape(HTML)
        self.assertTrue(value['complete_body_candidate'])
        self.assertNotIn('normal_consecutive_annual_admission',value)

    def test_denial_page_and_short_cover_not_complete(self):
        for blob in (b'<html>FORM 10-K ANNUAL REPORT</html>',HTML+b'ACCESS DENIED'):
            self.assertFalse(target.body_shape(blob)['complete_body_candidate'])

    def test_json_excerpts_never_original_body(self):
        self.assertFalse(target.body_shape(b'{"text":"FORM 10-K PART I PART II SIGNATURES"}')['complete_body_candidate'])

    def test_pdf_requires_eof_and_does_not_assert_source_identity(self):
        self.assertFalse(target.body_shape(b'%PDF-1.7 truncated')['complete_body_candidate'])
        self.assertTrue(target.body_shape(b'%PDF-1.7 binary\n%%EOF')['complete_body_candidate'])


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name).resolve()
        self.src=target.Sources(self.root)
    def tearDown(self):self.tmp.cleanup()

    def node(self,blob=HTML,url=URL,**values):
        (self.root/'raw.html').write_bytes(blob)
        node=dict(raw_path='raw.html',raw_sha256=hashlib.sha256(blob).hexdigest(),url=url,http_status=200)
        node.update(values);return node

    def inspect(self,node,acc=ACC):
        urls=[dict(url=node['url'],field='url',sec=target.sec_identity(node['url']))] if 'url' in node else []
        return target.inspect_reference(self.src,'catalog.json','$',node,acc,urls,INFO)[0]

    def test_exact_url_hash_http_and_structure_required(self):
        value=self.inspect(self.node())
        self.assertTrue(value['exact_source_body_candidate'])

    def test_cdn_explicit_mapping_remains_pending(self):
        value=self.inspect(self.node(url='https://issuer.example/year.html'))
        self.assertTrue(value['complete_body_candidate']);self.assertFalse(value['exact_source_body_candidate'])

    def test_http403_cannot_pass_even_if_annual_text_present(self):
        value=self.inspect(self.node(http_status=403))
        self.assertFalse(value['complete_body_candidate'])
        self.assertIn('FAILED_HTTP_OR_EXPLICITLY_UNACCEPTED_RESPONSE',value['reasons'])

    def test_explicit_unaccepted_status_blocks(self):
        self.assertFalse(self.inspect(self.node(accepted_original_source=False))['complete_body_candidate'])

    def test_other_document_at_same_accession_not_original_primary(self):
        value=self.inspect(self.node(url=URL.replace(DOC,'exhibit99.htm')))
        self.assertFalse(value['complete_body_candidate'])

    def test_conflicting_cik_or_accession_blocks(self):
        for url in (URL.replace('/data/1/','/data/2/'),URL.replace('000000000120000001','000000000220000002')):
            value=self.inspect(self.node(url=url))
            self.assertFalse(value['complete_body_candidate'])
            self.assertIn('SOURCE_URL_CONTRADICTS_SELECTED_ACCESSION_OR_CIK',value['reasons'])

    def test_missing_file_is_explicit(self):
        node=self.node();(self.root/'raw.html').unlink()
        value=self.inspect(node)
        self.assertEqual(value['reasons'],['MISSING_LOCAL_BODY'])

    def test_changed_body_hash_is_explicit(self):
        node=self.node();(self.root/'raw.html').write_bytes(b'changed')
        value=self.inspect(node)
        self.assertEqual(value['reasons'],['BODY_SHA_MISMATCH'])

    def test_parent_relative_path_and_ambiguous_roots(self):
        (self.root/'folder').mkdir();(self.root/'folder'/'body.htm').write_bytes(HTML)
        self.assertEqual(target.local_paths(self.root,'folder/catalog.json','body.htm'),['folder/body.htm'])
        (self.root/'body.htm').write_bytes(HTML)
        self.assertEqual(len(target.local_paths(self.root,'folder/catalog.json','body.htm')),2)

    def test_path_escape_never_read(self):
        self.assertEqual(target.local_paths(self.root,'catalog.json','../elsewhere.htm'),[])


class DenominatorTests(unittest.TestCase):
    def test_summary_preserves_unavailable_and_etfs(self):
        rows=[]
        for i in range(122):
            for j in range(49):
                rows.append(dict(signal_index=i,is_etf=j>=32,prior_tier2=j<3,
                    both_exact_source_body_candidates=False,both_any_explicitly_mapped_body_candidates=False,
                    any_explicitly_mapped_body_candidate=j==0))
        summary=target.summarize(rows)
        self.assertEqual(summary['all122']['all_slots'],5978)
        self.assertEqual(summary['all122']['stock_slots'],3904)
        self.assertEqual(summary['first88']['prior_tier2_slots'],264)
        self.assertEqual(summary['later34']['any_explicitly_mapped_body_candidate'],34)


if __name__=='__main__':unittest.main(verbosity=2)
