"""Generated-HTML producer/independent-reader interface checks only."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from bs4 import BeautifulSoup
import after150_batch2_independent_review as audit
import after150_intc_extract as intc
import after150_klac_extract as klac
import test_after150_intc_extract as it
import test_after150_klac_extract as kt


class IndependentParserInterface(unittest.TestCase):
    def compare(self, module, factory, **kwargs):
        raw, context = factory(**kwargs)
        output = module.extract_document(raw, context)
        soup = BeautifulSoup(raw, 'html.parser')
        if context['symbol'] == 'INTC':
            context['context']['statement_tables'] = []
            for table_index, stop in ((0, 3), (1, 2)):
                cells, _ = audit.grid_from_html(soup.find_all('table')[table_index])
                context['context']['statement_tables'].append(dict(
                    table=table_index, header_stop_row=stop,
                    headers=[deepcopy(cell) for (row, _), cell in cells.items() if row < stop]))
        policy = json.loads((Path(__file__).parent / 'after150_batch2_admission_policy.json').read_text())
        period = audit.periods(context, output, policy)
        for field in audit.FIELDS:
            reading = audit.source_field(soup, context, field)
            checked = audit.verify_published_cell(output['fields'][field], reading, field, output, period)
            self.assertTrue(checked['verified'])
            self.assertTrue(checked['source_locator_verified'])

    def test_intc_fixed_title_families(self):
        for year, title in ((2017, 'summary'), (2020, 'plain'), (2021, 'condensed')):
            with self.subTest(year=year):
                self.compare(intc, it.fixture, year=year, title_style=title)

    def test_intc_reversed_narrow_signed_fields(self):
        for reverse, narrow in ((True, False), (False, True), (True, True)):
            with self.subTest(reverse=reverse, narrow=narrow):
                self.compare(intc, it.fixture, reverse=reverse, narrow=narrow)

    def test_intc_zero_and_minus_income(self):
        for ni in ('0', '-25'):
            with self.subTest(ni=ni):
                self.compare(intc, it.fixture, ni=ni)

    def test_klac_layouts(self):
        for layout in ('wide', 'compact', 'compact_gap2'):
            with self.subTest(layout=layout):
                self.compare(klac, kt.fixture, layout=layout)

    def test_klac_signed_merged_fields(self):
        self.compare(klac, kt.fixture, merged=True, ni='(25)')

    def test_klac_transition_year(self):
        self.compare(klac, kt.fixture, year=2019)


if __name__ == '__main__':
    unittest.main(verbosity=2)
