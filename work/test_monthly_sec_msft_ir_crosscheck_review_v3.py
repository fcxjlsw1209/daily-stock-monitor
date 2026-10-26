import unittest
from test_monthly_sec_msft_ir_crosscheck_review import IndependentOracleTests, fixture
from monthly_sec_msft_ir_crosscheck_review_v3 import income


def internal_fixture():
    html = fixture()
    first = html.index('<caption>'); last = html.index('</caption>') + len('</caption>')
    html = html[:first] + html[last:]
    title = '<tr><td>Microsoft Corporation</td></tr><tr><td>Income Statements</td></tr><tr><td>(In millions, except per share amounts)(Unaudited)</td></tr>'
    return html.replace('<table>', '<div class="col-1-1"><table class="MsoNormalTable">'+title).replace('</table>', '</table></div>')


class V3TitleTests(unittest.TestCase):
    def test_internal_equivalence_and_preserved_row_locator(self):
        a, _ = income(fixture(), '2018-12-31')
        b, _ = income(internal_fixture(), '2018-12-31')
        self.assertEqual(b['caption_location'], 'internal_header_rows')
        self.assertEqual((b['current_usd'], b['prior_usd']), (a['current_usd'], a['prior_usd']))
        self.assertEqual(b['row_index'], a['row_index']+3)

    def test_empty_rows_and_table_sections(self):
        html = internal_fixture().replace('<tr><td>Income Statements', '<tr><td> </td></tr><tr><td>Income Statements')
        html = html.replace('<table class="MsoNormalTable">', '<table class="MsoNormalTable"><tbody>').replace('</table>', '</tbody></table>')
        self.assertIsNotNone(income(html, '2018-12-31')[0])

    def test_reject_mixed_header_cells(self):
        html = internal_fixture().replace('<td>Microsoft Corporation</td>', '<td>Microsoft Corporation</td><td>other</td>')
        self.assertIsNone(income(html, '2018-12-31')[0])

    def test_container_and_table_class_are_mandatory(self):
        for a, b in [('col-1-1','col-1-2'),('MsoNormalTable','other'),('pressreleasecontent','footer')]:
            with self.subTest(a=a):
                self.assertIsNone(income(internal_fixture().replace(a,b), '2018-12-31')[0])

    def test_direct_parent_is_required(self):
        h = internal_fixture().replace('<table class="MsoNormalTable">', '<section><table class="MsoNormalTable">').replace('</table>', '</table></section>')
        self.assertIsNone(income(h, '2018-12-31')[0])

    def test_order_segment_and_unit(self):
        for a,b in [('Microsoft Corporation','Income Statements'),('Income Statements','Segment Income Statements'),('In millions','In billions')]:
            with self.subTest(a=a):
                self.assertIsNone(income(internal_fixture().replace(a,b), '2018-12-31')[0])

    def test_title_prefix_rejects_text_and_foreign_nodes(self):
        for inserted in ['text','<div>text</div>','<table><tr><td>x</td></tr></table>','<span>text</span>']:
            h = internal_fixture().replace('<tr><td>Income Statements', inserted+'<tr><td>Income Statements')
            with self.subTest(inserted=inserted):
                self.assertIsNone(income(h, '2018-12-31')[0])

    def test_pure_empty_span_only(self):
        h = internal_fixture().replace('<tr><td>Income Statements', '<span> <span> </span></span><tr><td>Income Statements')
        self.assertIsNotNone(income(h, '2018-12-31')[0])
        for tag in ['img', 'script', 'br']:
            bad = h.replace('<span> <span> </span></span>', '<span><'+tag+'></'+tag+'></span>')
            self.assertIsNone(income(bad, '2018-12-31')[0])
        bad = internal_fixture().replace('<table class="MsoNormalTable">', '<table class="MsoNormalTable"><tfoot>').replace('</table>', '</tfoot></table>')
        self.assertIsNone(income(bad, '2018-12-31')[0])

    def test_duration_before_title_rejected(self):
        h = internal_fixture().replace('<tr><td>Microsoft Corporation', '<tr><td>Three Months Ended December 31,</td></tr><tr><td>Microsoft Corporation')
        self.assertIsNone(income(h, '2018-12-31')[0])

    def test_nested_table_after_headers_rejected(self):
        h = internal_fixture().replace('32,471', '<table><tr><td>32,471</td></tr></table>')
        self.assertIsNone(income(h, '2018-12-31')[0])

    def test_caption_priority_and_ambiguity(self):
        h = internal_fixture().replace('<table class="MsoNormalTable">', '<table class="MsoNormalTable"><caption>SEGMENT</caption>')
        self.assertIsNone(income(h, '2018-12-31')[0])
        self.assertIsNone(income(internal_fixture()+fixture(), '2018-12-31')[0])

    def test_adjacent_labels_individually_strict(self):
        h = fixture(); a=h.index('<caption>'); b=h.index('</caption>')+len('</caption>'); h=h[:a]+h[b:]
        bad='<p>Microsoft</p><p>Corporation Income Statements</p><p>(In millions, except per share amounts)</p>'
        h=h.replace('<table>',bad+'<div class="c-table"><table>').replace('</table>','</table></div>')
        self.assertIsNone(income(h,'2018-12-31')[0])
        good=h.replace('<p>Microsoft</p><p>Corporation Income Statements</p>','<p>Microsoft Corporation</p><p>Income Statements</p>')
        self.assertEqual(income(good,'2018-12-31')[0]['caption_location'],'adjacent_paragraphs')


if __name__ == '__main__': unittest.main()
