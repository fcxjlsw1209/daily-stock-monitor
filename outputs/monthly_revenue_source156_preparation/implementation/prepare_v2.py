"""Source-only parser revision: distinguish Revenue section label from Total revenue.

The frozen v1 extractor, its specification and its results remain unchanged.
This revision changes only the semantic total-row selector, not source scope,
period, currency, availability or admission rules.
"""
import argparse
import re
import unittest

import after155_revenue_source_prepare as base


ORIGINAL = base.SELF
base.SELF = 'outputs/monthly_revenue_source156_preparation/implementation/prepare_v2.py'
base.OUT = 'outputs/monthly_revenue_source156_preparation/v2'
base.SPEC = base.OUT + '/EXTRACTION_SPEC.json'
base.INPUTS = dict(base.INPUTS, extractor=base.SELF, source_v1=ORIGINAL,
                   prior_v1_spec='outputs/monthly_revenue_source156_preparation/EXTRACTION_SPEC.json',
                   prior_v1_proposal='outputs/monthly_revenue_source156_preparation/SOURCE_ADMISSION_PROPOSAL.json')


def statement_candidates(soup, fiscal_year):
    candidates, rejections = [], []
    wanted = [f'{fiscal_year - 1}-06-30', f'{fiscal_year}-06-30']
    for ti, table in enumerate(soup.find_all('table')):
        context = base.layout.statement_context(table)
        if not re.search(r'INCOME STATEMENTS|CONSOLIDATED STATEMENTS OF (?:INCOME|OPERATIONS)', context, re.I):
            continue
        try:
            text = base.layout.norm(table.get_text(' ', strip=True))
            base.layout.validate_statement(context, text, 'annual_gross_amount', 'MSFT')
            rows, grid, physical = base.layout.table_grid(table)
            stop = base.layout.header_stop(rows)
            totals, plain = [], []
            for ri, row in enumerate(rows):
                labels = [base.layout.norm(c.get_text(' ', strip=True)) for c in row.find_all(['td', 'th'], recursive=False)]
                if any(re.fullmatch(r'total revenues?:?', t, re.I) for t in labels):
                    totals.append(ri)
                elif any(re.fullmatch(r'revenues?:?', t, re.I) for t in labels):
                    plain.append(ri)
            matches = totals if totals else plain
            if len(matches) != 1:
                raise ValueError('Not exactly one consolidated total-revenue row')
            ri = matches[0]
            unit_context = base.layout.norm(context + ' ' + ' '.join(r.get_text(' ', strip=True) for r in rows[:stop]))
            scale = base.layout.scale_from(unit_context)
            if '$' not in text:
                raise ValueError('No statement dollar symbol')
            found = {d: [] for d in wanted}
            for (r, pi), cell in physical.items():
                if r != ri:
                    continue
                try:
                    amount = base.layout.exact_amount(cell['text'])
                except ValueError:
                    continue
                headers = base.layout.headers_for(grid, ri, cell['columns'][0], stop)
                dates = base.layout.header_dates(headers)
                if 'Twelve Months Ended' not in ' '.join(headers) or dates not in [[d] for d in wanted]:
                    continue
                year_end = dates[0]
                cell_tag = rows[ri].find_all(['td', 'th'], recursive=False)[pi]
                raw_cell = str(cell_tag)
                found[year_end].append({
                    'displayed_amount': cell['text'], 'amount_decimal': amount,
                    'scale': scale, 'currency_literal': '$', 'period_end': year_end,
                    'period_start_derived': f'{int(year_end[:4]) - 1}-07-01',
                    'period_start_is_direct_quote': False,
                    'duration_literal': 'Twelve Months Ended',
                    'days_inclusive': (base.date.fromisoformat(year_end) - base.date(int(year_end[:4]) - 1, 7, 1)).days + 1,
                    'header_literals': headers,
                    'locator': {'table_index_zero_based': ti, 'row_index_zero_based': ri,
                                'physical_cell_index_zero_based': pi, 'logical_columns': cell['columns']},
                    'cell_html': raw_cell, 'cell_html_sha256': base.sha(raw_cell.encode()),
                })
            if any(len(v) != 1 for v in found.values()):
                raise ValueError('Annual current/prior revenue cells absent or ambiguous: ' + str({d: len(v) for d, v in found.items()}))
            candidates.append({
                'current': found[wanted[1]][0], 'comparative': found[wanted[0]][0],
                'statement_context': context, 'unit_context': unit_context,
                'row_selection': 'unique_total_revenue' if totals else 'unique_revenue',
                'plain_revenue_category_rows_present': len(plain),
                'revenue_row_text': base.layout.norm(rows[ri].get_text(' ', strip=True)),
                'revenue_row_html': str(rows[ri]), 'table_html': str(table),
                'table_html_sha256': base.sha(str(table).encode()),
            })
        except (ValueError, KeyError, IndexError) as exc:
            rejections.append({'table_index_zero_based': ti, 'reason': str(exc)})
    if len(candidates) != 1:
        raise ValueError('Expected one annual statement candidate; got ' + str(len(candidates)) + '; ' + base.json.dumps(rejections))
    return candidates[0], rejections


base.statement_candidates = statement_candidates


class RevisedTests(base.SyntheticTests):
    def test_total_takes_precedence_over_section_label(self):
        soup = self.sample()
        label = soup.find('td', string='Revenue')
        label.string = 'Total revenue'
        label.parent.insert_before(base.BeautifulSoup('<tr><td>Revenue:</td><td></td><td></td></tr>', 'html.parser').tr)
        item, _ = statement_candidates(soup, 2016)
        self.assertEqual(item['row_selection'], 'unique_total_revenue')
        self.assertEqual(item['current']['amount_decimal'], '12300')
        self.assertEqual(item['current']['locator']['row_index_zero_based'], 3)

    def test_duplicate_total_rejected(self):
        soup = self.sample(extra='<tr><td>Total revenue</td><td>2</td><td>3</td></tr>')
        soup.find('td', string='Revenue').string = 'Total revenue'
        with self.assertRaises(ValueError):
            statement_candidates(soup, 2016)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--freeze', action='store_true')
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--self-test', action='store_true')
    parser.add_argument('--spec-sha')
    args = parser.parse_args()
    if args.self_test:
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RevisedTests))
        raise SystemExit(0 if result.wasSuccessful() else 1)
    if args.freeze:
        base.freeze()
    elif not args.spec_sha:
        parser.error('--prepare requires --spec-sha')
    else:
        base.prepare(args.spec_sha)
