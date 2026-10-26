"""Narrow edits to the new independent review only."""
from pathlib import Path
p=Path('work/after149_batch1_independent_review.py');s=p.read_text()
s=s.replace("title_regex.fullmatch(c['text'].rstrip(':').strip())", "title_regex.search(c['text'].rstrip(':').strip())")
s=s.replace("null_classification='source_value_present_but_frozen_parser_no_candidate',", """null_classification='source_value_present_but_frozen_parser_no_candidate',
                          null_diagnosis='Formal balance-sheet title is inside the source table. An unrelated preceding non-GAAP reconciliation notice contaminates the frozen parser external context and causes its BAD-context rejection before candidate collection.',""")
p.write_text(s)
p=Path('work/after149_batch1_independent_review_test.py');s=p.read_text()
new='''    def test_indented_leftmost_nonempty_label(self):
        soup, context = fixture()
        row = soup.table.find_all('tr')[4]
        blank = soup.new_tag('td'); row.insert(0, blank)
        for item in context['tables'][0]['columns']: item['column'] += 1
        # Align the corresponding date and duration rows with the indentation.
        for r in soup.table.find_all('tr')[:4]: r.insert(0, soup.new_tag('td'))
        self.assertEqual(audit.independently_read_field(soup, context, 'CSCO', 'annual_net_income', '2025-07-26')['value_decimal'], '-1234.50')

    def test_combined_issuer_title_and_units(self):
        soup, context = fixture()
        soup.find(string='CONSOLIDATED STATEMENTS OF CASH FLOWS').replace_with('CISCO SYSTEMS, INC. CONSOLIDATED STATEMENTS OF CASH FLOWS (In millions) (Unaudited)')
        self.assertEqual(audit.independently_read_field(soup, context, 'CSCO', 'annual_net_income', '2025-07-26')['scale'], '1000000')

    def test_external_formal_heading(self):
        soup, context = fixture()
        soup.find(string='CONSOLIDATED STATEMENTS OF CASH FLOWS').replace_with('')
        soup.find(string='(In millions)').replace_with('')
        heading=soup.new_tag('p');heading.string='CONSOLIDATED STATEMENTS OF CASH FLOWS (In millions)'
        soup.table.insert_before(heading)
        self.assertEqual(audit.independently_read_field(soup, context, 'CSCO', 'annual_net_income', '2025-07-26')['scale'], '1000000')

    def test_unrelated_non_gaap_notice_not_inline_title(self):
        soup, context = fixture()
        heading=soup.new_tag('p');heading.string='Additional reconciliations between GAAP and non-GAAP financial measures follow.'
        soup.table.insert_before(heading)
        self.assertEqual(audit.independently_read_field(soup, context, 'CSCO', 'annual_net_income', '2025-07-26')['value_decimal'], '-1234.50')

'''
s=s.replace("    def test_no_io_or_producer_import(self):",new+"    def test_no_io_or_producer_import(self):")
p.write_text(s)
