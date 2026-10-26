"""Synthetic-only source-parser checks; no original report amounts are read."""
import copy
import datetime as dt
import unittest
from bs4 import BeautifulSoup
import after151_lrcx_extract as m


def td(text='', span=1, rowspan=1):
    return '<td colspan="%s" rowspan="%s">%s</td>' % (span, rowspan, text)


def tr(*cells):
    return '<tr>' + ''.join(cells) + '</tr>'


def amount(value, dollar=True):
    if value.startswith('(') and value.endswith(')'):
        return td('$ (' if dollar else '(') + td(value[1:-1]) + td(')') + td()
    return td('$' if dollar else '') + td(value) + td() + td()


def fixture(year=2016, split=True, quarters=3, reverse=False, ni='(1,234.50)',
            current_assets='8,000', prior_assets='7,000', unit='thousands', dollar=True):
    dates = {2013: ('2013-06-30', '2012-06-24', '2013-03-31'),
             2014: ('2014-06-29', '2013-06-30', '2014-03-30'),
             2015: ('2015-06-28', '2014-06-29', '2015-03-29'),
             2016: ('2016-06-26', '2015-06-28', '2016-03-27')}
    current, prior, quarter = dates[year]
    annual = [prior, current] if reverse else [current, prior]
    short = [current, quarter, prior][:quarters]
    ends = short + annual
    bsends = [prior, quarter, current] if reverse else [current, quarter, prior]
    count = 3+4*len(ends)
    def date_cell(end):
        d = dt.date.fromisoformat(end)
        label = d.strftime('%B') + ' ' + str(d.day) + ' ,'
        return td(label if split else label+' '+str(d.year), 2)+td()+td()
    def labels(values):
        return tr(td('', 3), *(date_cell(e) for e in values))
    def years(values):
        return tr(td('', 3), *(td(e[:4], 2)+td()+td() for e in values))
    income = [tr(td('CONDENSED CONSOLIDATED STATEMENTS OF OPERATIONS', count)),
              tr(td('(in '+unit+', except per share data and percentages)', count)),
              tr(td('', 3), td('Three Months Ended', 4*quarters), td('Twelve Months Ended', 8)),
              labels(ends)]
    if split:
        income.append(years(ends))
    stop = len(income)
    income.append(tr(td(), td(), td('Revenues'), td('55', count-3)))
    row = len(income)
    income.append(tr(td(), td(), td('Net income'),
                     *(amount('333', dollar) for e in short),
                     *(amount(ni if e == current else '222', dollar) for e in annual)))
    balance = [tr(td('CONDENSED CONSOLIDATED BALANCE SHEETS', 15)),
               tr(td('(in '+unit+')', 15)), labels(bsends)]
    if split:
        balance.append(years(bsends))
    bsstop = len(balance)
    balance.append(tr(td(), td(), td('Current assets'), td('11', 12)))
    bsrow = len(balance)
    balance.append(tr(td(), td(), td('Total assets'),
                      *(amount(current_assets if e == current else prior_assets if e == prior else '999', dollar)
                        for e in bsends)))
    raw = ('<html><p>Lam Research Corporation (NASDAQ : LRCX)</p><table>' + ''.join(income)
           + '</table><table>' + ''.join(balance) + '</table></html>').encode()
    source = {'path': 'synthetic/source.html', 'sha256': m.sha256(raw),
              'receipt_path': 'synthetic/source.meta.json', 'receipt_sha256': 'a'*64,
              'url': 'https://newsroom.lamresearch.com/synthetic'}
    c = {'symbol': 'LRCX', 'fiscal_year': year, 'fiscal_quarter': 'Q4',
         'event_id': 'LRCX_'+str(year)+'-07-30', 'publication_date': str(year)+'-07-30',
         'available_session': str(year)+'-07-31', 'available_session_verified': year>=2015,
         'official_actual_period_end': current, 'source': source,
         'evidence_path': source['path'], 'evidence_sha256': source['sha256'], 'source_url': source['url'],
         'receipt': {'path': source['receipt_path'], 'sha256': source['receipt_sha256']},
         'identity_anchors': ['LAM RESEARCH CORPORATION', 'NASDAQ : LRCX)'],
         'context_status': m.STATUS, 'format': 'html_or_text',
         'context_holds': ['currency_uncertified', 'start_assumption_unapproved', 'duplicate', 'duplicate'],
         'periods': {'current_end': current, 'prior_end': prior,
                     'proposed_current_start': str(dt.date.fromisoformat(prior)+dt.timedelta(days=1)),
                     'start_assumption': 'same_current_report_prior_annual_end_plus_one_calendar_day',
                     'start_is_source_quoted': False, 'start_assumption_approved': False},
         'fields': {}, 'statement_contexts': []}
    soup = BeautifulSoup(raw, 'html.parser')
    for name, ti, ri, end, index, hs in (
        ('annual_net_income', 0, row, current, quarters+annual.index(current), stop),
        ('current_total_assets', 1, bsrow, current, bsends.index(current), bsstop),
        ('prior_total_assets', 1, bsrow, prior, bsends.index(prior), bsstop)):
        _, _, physical = m.table_grid(soup.find_all('table')[ti])
        c['fields'][name] = {'source': copy.deepcopy(source), 'as_of': end,
            'available_from': c['available_session'], 'currency_literal': '$', 'currency_iso': None,
            'table': ti, 'row': ri, 'label_cell': 2, 'label': 'Net income' if ti==0 else 'Total assets',
            'logical_columns': [3+4*index, 4+4*index],
            'scale': '1000' if unit=='thousands' else '1000000',
            'header_evidence': [{k: v for k, v in x.items() if k in m.GEOMETRY+('text',)}
                                for (r, _), x in physical.items() if r<hs and x['text']],
            'row_geometry_without_values': [{k: x[k] for k in m.GEOMETRY}
                                            for x in m.old._cells(physical, ri)]}
    return raw, c


def changed(raw, c, before, after, refresh_headers=False):
    raw = raw.replace(before.encode(), after.encode())
    c = copy.deepcopy(c)
    c['source']['sha256'] = c['evidence_sha256'] = m.sha256(raw)
    for spec in c['fields'].values():
        spec['source'] = copy.deepcopy(c['source'])
    if refresh_headers:
        soup = BeautifulSoup(raw, 'html.parser')
        for spec in c['fields'].values():
            _, _, physical = m.table_grid(soup.find_all('table')[spec['table']])
            for anchor in spec['header_evidence']:
                cell = physical[(anchor['row'], anchor['physical_cell'])]
                for k in m.GEOMETRY+('text',):
                    anchor[k] = copy.deepcopy(cell[k])
    return raw, c


class LrcxTests(unittest.TestCase):
    def output(self, **kwargs):
        return m.extract_document(*fixture(**kwargs))

    def test_signed_values_and_scope(self):
        out = self.output()
        self.assertEqual([x['value_decimal'] for x in out['fields'].values()], ['-1234.50', '8000', '7000'])
        self.assertEqual(out['gaps'], [])
        self.assertTrue(out['complete_required_fields'])
        for flag in ('fact_admitted', 'model_admitted', 'strategy_admitted', 'first_public_version_verified'):
            self.assertIs(out[flag], False)
        self.assertEqual(out['financial_calculations'], 0)

    def test_all_fixed_years(self):
        for y in m.YEARS:
            self.assertTrue(self.output(year=y)['complete_required_fields'])

    def test_four_and_five_groups_and_full_dates(self):
        for split in (True, False):
            for quarters in (2, 3):
                for reverse in (False, True):
                    out = self.output(split=split, quarters=quarters, reverse=reverse)
                    self.assertTrue(out['complete_required_fields'], out['gaps'])
                    self.assertEqual(out['fields']['annual_net_income']['value_decimal'], '-1234.50')
                    self.assertEqual(out['fields']['prior_total_assets']['value_decimal'], '7000')

    def test_preserve_metadata_and_input_immutability(self):
        raw, c = fixture()
        before = copy.deepcopy(c)
        out = m.extract_document(raw, c)
        self.assertEqual(c, before)
        self.assertEqual(out['source'], c['source'])
        self.assertEqual(out['context_holds'], c['context_holds'])
        for name, cell in out['fields'].items():
            self.assertEqual(cell['source'], c['source'])
            self.assertEqual(cell['available_from'], c['available_session'])
            self.assertEqual(cell['period'], out['periods']['current'] if name=='annual_net_income' else None)
            self.assertEqual(cell['as_of'], c['fields'][name]['as_of'])
            self.assertIsNone(cell['currency_iso'])
        self.assertFalse(out['periods']['start_assumption_certified'])
        self.assertEqual(out['periods']['assumption_id'], 'LRCX_consecutive_annual_boundaries_v1')
        self.assertEqual(out['periods']['current']['days_inclusive'], 364)

    def test_zero_and_minus_and_large_decimal(self):
        for val, expected in [('0', '0'), ('-42', '-42'), ('(0)', '-0'),
            ('(123456789012345678901234567890123456789.123456789)', '-123456789012345678901234567890123456789.123456789')]:
            self.assertEqual(self.output(ni=val)['fields']['annual_net_income']['value_decimal'], expected)

    def test_whitespace_comma_preserved(self):
        raw, c = changed(*fixture(ni='1,234'), '1,234', '1&nbsp; , &nbsp;234')
        cell = m.extract_document(raw, c)['fields']['annual_net_income']
        self.assertEqual(cell['value_decimal'], '1234')
        self.assertIn('\xa0', ''.join(x['original_text'] for x in cell['locator']['physical_cells']))
        self.assertTrue(all(x['markup_sha256']==x['parsed_cell_markup_sha256'] for x in cell['locator']['physical_cells']))

    def test_missing_amount_not_zero(self):
        for val in ('', '-', '—', 'N/A', '12 34', '12%', '(-12)', '$ $ 3'):
            out = self.output(ni=val)
            self.assertIsNone(out['fields']['annual_net_income'])
            self.assertIsNotNone(out['fields']['current_total_assets'])
            self.assertFalse(out['complete_required_fields'])

    def test_raw_hash_mismatch_hard_error(self):
        raw, c = fixture()
        with self.assertRaises(m.SourceError):
            m.extract_document(raw+b' ', c)

    def test_reference_scope_hard_errors(self):
        for key, value in [('symbol','AMD'), ('fiscal_year',2017), ('fiscal_year',True),
                           ('event_id','LRCX_1900-01-01'), ('context_status','unknown')]:
            raw, c = fixture()
            c[key] = value
            with self.assertRaises(m.SourceError): m.extract_document(raw, c)

    def test_source_and_receipt_reference_errors(self):
        for key, value in [('url','http://newsroom.lamresearch.com/x'), ('url','https://example.com/x'),
                           ('url','https://user:pass@newsroom.lamresearch.com/x'), ('path','../source'),
                           ('path','/source'), ('receipt_sha256','bad'), ('sha256','bad')]:
            raw, c = fixture()
            c['source'][key] = value
            with self.assertRaises(m.SourceError): m.extract_document(raw, c)

    def test_conflicting_reference_errors(self):
        for key in ('evidence_path','evidence_sha256','source_url'):
            raw, c = fixture()
            c[key] = 'wrong'
            with self.assertRaises(m.SourceError): m.extract_document(raw, c)
        raw, c = fixture(); c['receipt']['sha256'] = 'f'*64
        with self.assertRaises(m.SourceError): m.extract_document(raw, c)

    def test_wrong_issuer_and_ticker(self):
        for a,b in [('Lam Research Corporation','Other Corporation'), ('NASDAQ : LRCX','NASDAQ : ELSE')]:
            raw, c = changed(*fixture(), a,b)
            with self.assertRaises(m.SourceError): m.extract_document(raw, c)

    def test_bad_metadata_period_is_explicit_gap(self):
        for key, value in [('current_end','2016-07-26'), ('prior_end','2015-03-29'),
                           ('proposed_current_start','2015-07-01'), ('start_assumption_approved',True),
                           ('start_is_source_quoted',True), ('start_assumption','guess')]:
            raw, c = fixture(); c['periods'][key] = value
            out=m.extract_document(raw,c)
            self.assertEqual(len(out['gaps']), 3)
            self.assertTrue(all(v is None for v in out['fields'].values()))
            self.assertEqual(out['context_holds'], c['context_holds'])

    def test_bad_publication_order(self):
        raw,c=fixture(); c['available_session']=c['publication_date']
        self.assertEqual(len(m.extract_document(raw,c)['gaps']),3)

    def test_unverified_early_calendar_remains_metadata(self):
        out=self.output(year=2013)
        self.assertTrue(out['complete_required_fields'])
        self.assertFalse(out['first_public_version_verified'])

    def test_quarter_ni_cannot_replace_annual(self):
        raw,c=fixture(); c['fields']['annual_net_income']['logical_columns']=[3,4]
        out=m.extract_document(raw,c)
        self.assertIsNone(out['fields']['annual_net_income'])
        self.assertIn('not_exclusively_annual_column',[g['reason'] for g in out['gaps']])

    def test_march_assets_cannot_replace_prior_year(self):
        raw,c=fixture(); c['fields']['prior_total_assets']['logical_columns']=[7,8]
        self.assertIsNone(m.extract_document(raw,c)['fields']['prior_total_assets'])

    def test_no_annual_parent_and_overlapping_parent(self):
        for new in ('Six Months Ended','Three Months Ended'):
            raw,c=changed(*fixture(),'Twelve Months Ended',new,True)
            self.assertIsNone(m.extract_document(raw,c)['fields']['annual_net_income'])

    def test_comparison_annual_end_must_match_prior_asset_end(self):
        raw,c=fixture(split=False)
        # Change the prior year in all source headers while period metadata stays fixed.
        raw,c=changed(raw,c,'June 28 , 2015','June 27 , 2015',True)
        self.assertIsNone(m.extract_document(raw,c)['fields']['annual_net_income'])

    def test_parent_net_income_is_not_consolidated(self):
        raw,c=changed(*fixture(),'Net income','Net income attributable to parent',True)
        c['fields']['annual_net_income']['label']='Net income attributable to parent'
        self.assertIsNone(m.extract_document(raw,c)['fields']['annual_net_income'])

    def test_non_gaap_and_missing_title(self):
        for title in ('NON-GAAP CONDENSED CONSOLIDATED STATEMENTS OF OPERATIONS', 'STATEMENTS OF OPERATIONS'):
            raw,c=changed(*fixture(),'CONDENSED CONSOLIDATED STATEMENTS OF OPERATIONS',title,True)
            self.assertIsNone(m.extract_document(raw,c)['fields']['annual_net_income'])

    def test_conflicting_units_and_missing_dollar(self):
        raw,c=changed(*fixture(),'in thousands, except','in millions and in thousands, except',True)
        self.assertIsNone(m.extract_document(raw,c)['fields']['annual_net_income'])
        self.assertTrue(all(v is None for v in self.output(dollar=False)['fields'].values()))

    def test_scale_metadata_mismatch(self):
        raw,c=fixture(); c['fields']['annual_net_income']['scale']='1000000'
        self.assertIsNone(m.extract_document(raw,c)['fields']['annual_net_income'])

    def test_foreign_currency_rejected(self):
        raw,c=changed(*fixture(),'$ (','EUR (')
        self.assertIsNone(m.extract_document(raw,c)['fields']['annual_net_income'])

    def test_bad_field_metadata(self):
        for key,value in [('as_of','2015-06-28'),('available_from','2010-01-01'),
                          ('currency_iso','USD'),('currency_literal','EUR'),('label_cell',0),
                          ('source',{}),('logical_columns',[16,15]),('logical_columns',[True])]:
            raw,c=fixture(); c['fields']['annual_net_income'][key]=value
            self.assertIsNone(m.extract_document(raw,c)['fields']['annual_net_income'])

    def test_fixed_header_and_geometry_checked(self):
        for key in ('header_evidence','row_geometry_without_values'):
            raw,c=fixture(); c['fields']['annual_net_income'][key]=[]
            self.assertIsNone(m.extract_document(raw,c)['fields']['annual_net_income'])

    def test_changed_header_text_cannot_reuse_old_context(self):
        raw,c=changed(*fixture(),'Twelve Months Ended','Years Ended')
        self.assertIsNone(m.extract_document(raw,c)['fields']['annual_net_income'])

    def test_nested_target_table_rejected(self):
        raw,c=changed(*fixture(), '1,234.50','<table><tr><td>1,234.50</td></tr></table>')
        self.assertIsNone(m.extract_document(raw,c)['fields']['annual_net_income'])

    def test_preserves_nonpositive_assets_for_later_admission(self):
        out=self.output(current_assets='0',prior_assets='-2')
        self.assertEqual(out['fields']['current_total_assets']['value_decimal'],'0')
        self.assertEqual(out['fields']['prior_total_assets']['value_decimal'],'-2')
        self.assertFalse(out['fact_admitted'])

    def test_missing_body_boundary(self):
        raw,c=changed(*fixture(),'Revenues','Unknown body line')
        self.assertIsNone(m.extract_document(raw,c)['fields']['annual_net_income'])

    def test_duplicate_exact_ni_row(self):
        raw,c=changed(*fixture(),'Revenues','Net income')
        self.assertIsNone(m.extract_document(raw,c)['fields']['annual_net_income'])

    def test_53_week_source_endpoints(self):
        self.assertEqual(self.output(year=2013)['periods']['current']['days_inclusive'],371)


    def test_narrow_anchor_keeps_its_own_split_sign(self):
        raw,c=fixture()
        c['fields']['annual_net_income']['logical_columns']=[16]
        cell=m.extract_document(raw,c)['fields']['annual_net_income']
        self.assertEqual(cell['value_decimal'],'-1234.50')
        self.assertEqual(len(cell['locator']['physical_cells']),3)

    def test_closing_parenthesis_cannot_cross_group(self):
        raw,c=changed(*fixture(), td(')')+td(), td()+td(')'))
        self.assertIsNone(m.extract_document(raw,c)['fields']['annual_net_income'])

    def test_unconsumed_period_content_is_not_ignored(self):
        raw,c=changed(*fixture(ni='1234.50'), td('1234.50')+td(), td('1234.50')+td('7'))
        out=m.extract_document(raw,c)
        self.assertIsNone(out['fields']['annual_net_income'])
        self.assertIn('unconsumed_content_within_period_group',[g['reason'] for g in out['gaps']])

    def test_body_numeric_year_is_not_header(self):
        raw,c=changed(*fixture(), '55</td>', '2011</td>')
        self.assertTrue(m.extract_document(raw,c)['complete_required_fields'])

    def test_repeated_or_conflicting_date_header_is_gap(self):
        raw,c=changed(*fixture(split=False), 'June 26 , 2016', 'June 26 , 2016 June 26 , 2016',True)
        self.assertIsNone(m.extract_document(raw,c)['fields']['annual_net_income'])

    def test_year_header_conflict_is_gap(self):
        raw,c=changed(*fixture(), '2016</td>', '2016 2017</td>',True)
        self.assertIsNone(m.extract_document(raw,c)['fields']['annual_net_income'])

    def test_foreign_statement_title_no_acceptance(self):
        raw,c=changed(*fixture(),'CONDENSED CONSOLIDATED STATEMENTS OF OPERATIONS',
                      'CONDENSED CONSOLIDATED STATEMENTS OF COMPREHENSIVE INCOME',True)
        self.assertIsNone(m.extract_document(raw,c)['fields']['annual_net_income'])

    def test_pretax_label_no_acceptance(self):
        raw,c=changed(*fixture(),'Net income','Income before taxes')
        c['fields']['annual_net_income']['label']='Income before taxes'
        self.assertIsNone(m.extract_document(raw,c)['fields']['annual_net_income'])

    def test_millions_unit_preserved_without_normalizing_amount(self):
        out=self.output(unit='millions')
        self.assertEqual(out['fields']['annual_net_income']['scale'],'1000000')
        self.assertEqual(out['fields']['annual_net_income']['value_decimal'],'-1234.50')

    def test_no_cross_year_symbol_borrow(self):
        raw,c=fixture()
        # Move the opening sign to the final quarterly group's separator.
        raw,c=changed(raw,c,td('$ (')+td('1,234.50'),td()+td('1,234.50'))
        self.assertIsNone(m.extract_document(raw,c)['fields']['annual_net_income'])

    def test_unsupported_nonstandard_format(self):
        raw,c=fixture(); c['format']='pdf'
        with self.assertRaises(m.SourceError): m.extract_document(raw,c)


if __name__ == '__main__':
    unittest.main(verbosity=2)
