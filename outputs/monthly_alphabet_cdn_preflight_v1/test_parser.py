import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('alphabet_parser', Path(__file__).with_name('parser.py'))
p = importlib.util.module_from_spec(spec); spec.loader.exec_module(p)


def fixture(q=2, modern=False):
    month, day = {1:('March',31),2:('June',30),3:('September',30),4:('December',31)}[q]
    word = ('First','Second','Third','Fourth')[q-1]
    extra = ' and Fiscal Year' if q == 4 else ''
    release = 'February 2, 2023' if q == 4 else {1:'April 26, 2022',2:'July 26, 2022',3:'October 25, 2022'}[q]
    lead = (f'Alphabet Announces {word} Quarter{extra} 2022 Results\n'
            f'MOUNTAIN VIEW, Calif. – {release} – Alphabet Inc. (NASDAQ: GOOG, GOOGL) today announced financial '
            f'results for the quarter{extra.lower()} ended {month} {day}, 2022.\n')
    if modern:
        header = f'Quarter Ended {month} {day},'
        if q > 1: header += f' {"Year Ended" if q == 4 else "Year to Date"} {month} {day},'
    else:
        header = 'Three Months Ended'
        if q > 1: header += f' { {2:"Six",3:"Nine",4:"Twelve"}[q]} Months Ended'
        header += f' {month} {day},' * (1 if q == 1 else 2)
    columns = '2021 2022' if q == 1 else '2021 2022 2021 2022'
    values = '$ 1,000 $ 1,200' if q == 1 else '$ 1,000 $ 1,200 $ 2,345 $ 2,678'
    return (lead + 'Financial highlights\nRevenues $ 999 $ 999\nAlphabet Inc.\n' + p.TITLE +
            '\n(In millions, except per share amounts; unaudited)\n'+header+'\n'+columns+
            '\nRevenues '+values+'\nCosts and expenses:\nCost of revenues 99 100\n')


class ParserTests(unittest.TestCase):
    def test_all_four_quarters_old_and_modern(self):
        for q in range(1,5):
            for modern in (False,True):
                with self.subTest(q=q,modern=modern):
                    got=p.parse_release(fixture(q,modern))
                    self.assertEqual(got['identity']['fiscal_quarter'],q)
                    self.assertEqual(got['statement']['periods'][-1]['duration_months'],3*q)
                    self.assertEqual(got['statement']['periods'][-1]['current']['revenue_usd'],1200000000 if q==1 else 2678000000)

    def test_exact_cfo_subtitle_allowed(self):
        subtitle = 'Announces Plans for Chief Financial Officer Transition and Appointment of President and Chief Investment Officer'
        text = fixture().replace('2022 Results\n', '2022 Results\n' + subtitle + '\n')
        self.assertEqual(p.parse_release(text), p.parse_release(fixture()))

    def test_unrecognized_or_duplicate_frontmatter_rejected(self):
        subtitle = 'Announces Plans for Chief Financial Officer Transition and Appointment of President and Chief Investment Officer'
        for prefix in ('Unrelated text', subtitle + ' changed', subtitle + ' ' + subtitle):
            with self.subTest(prefix=prefix), self.assertRaises(p.ParseError):
                p.parse_release(fixture().replace('2022 Results\n', '2022 Results\n' + prefix + '\n'))

    def test_multiline_cells_and_header(self):
        a=fixture(); b=a.replace('Revenues $ 1,000 $ 1,200 $ 2,345 $ 2,678','Revenues\n$\n1,000\n$\n1,200\n$\n2,345\n$\n2,678')
        self.assertEqual(p.parse_release(a),p.parse_release(b))

    def test_shares_thousands_do_not_change_revenue_unit(self):
        text=fixture().replace('per share amounts; unaudited','share amounts which are reflected in thousands and per share amounts')
        self.assertEqual(p.statement(text)['periods'][1]['prior']['revenue_usd'],2345000000)

    def test_unaudited_year_suffix(self):
        self.assertEqual(p.statement(fixture().replace('2021 2022 2021 2022','2021 2022 2021 2022 (unaudited) (unaudited)'))['periods'],p.statement(fixture())['periods'])

    def test_ungrouped_amount(self):
        self.assertEqual(p.statement(fixture().replace('2,678','2678'))['periods'][1]['current']['revenue_usd'],2678000000)

    def test_zero_reported_amount(self):
        self.assertEqual(p.statement(fixture().replace('2,678','0'))['periods'][1]['current']['revenue_usd'],0)

    def test_nbsp_and_newline(self):
        self.assertEqual(p.parse_release(fixture()),p.parse_release(fixture().replace('July 26','July\u00a026').replace('NASDAQ: GOOG,','NASDAQ:\nGOOG,')))

    def test_reject_mutations(self):
        changes=[('NASDAQ: GOOG, GOOGL','NYSE: GOOG, GOOGL'),('NASDAQ: GOOG, GOOGL','NASDAQ: GOOG'),
            ('Alphabet Inc. (NASDAQ','Google Inc. (NASDAQ'),('MOUNTAIN VIEW, Calif.','OTHER CITY, Calif.'),
            ('July 26, 2022','July 26, 2021'),('Second Quarter','First Quarter'),
            (p.TITLE,'SEGMENT RESULTS'),('(In millions,','(In thousands,'),
            ('Six Months Ended','Nine Months Ended'),('June 30, June 30,','June 30, September 30,'),
            ('2021 2022 2021 2022','2022 2021 2022 2021'),('2021 2022 2021 2022','2020 2022 2020 2022'),
            ('$ 2,678','$ 2,67'),('$ 2,678','$ -2,678'),('$ 2,678','$ (2,678)'),
            ('$ 2,678','$ 2,678 $ 3,000'),('$ 2,678',''),('Costs and expenses:','Net revenues:'),
            ('\nRevenues $ 1,000','\nNet revenues $ 1,000'),('Alphabet Inc.\n'+p.TITLE,'Google LLC\n'+p.TITLE)]
        for old,new in changes:
            with self.subTest(old=old,new=new),self.assertRaises(p.ParseError):p.parse_release(fixture().replace(old,new))

    def test_duplicate_statement_fails(self):
        with self.assertRaises(p.ParseError):p.statement(fixture()+p.TITLE)

    def test_duplicate_identity_fails(self):
        with self.assertRaises(p.ParseError):p.identity(fixture()+fixture())

    def test_table_identity_period_mismatch(self):
        with self.assertRaises(p.ParseError):p.parse_release(fixture().replace('2021 2022 2021 2022','2020 2021 2020 2021'))

    def test_missing_ytd_not_substituted_by_quarter(self):
        with self.assertRaises(p.ParseError):p.statement(fixture().replace(' Six Months Ended','').replace('June 30, June 30,','June 30,').replace('2021 2022 2021 2022','2021 2022').replace(' $ 2,345 $ 2,678',''))


if __name__ == '__main__':unittest.main()
