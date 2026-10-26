"""Pure synthetic tests; no actual report, amount or strategy input read."""
import copy
import unittest
from bs4 import BeautifulSoup
import after153_batch5_accounting_review as m


def fixture(symbol='PLTR'):
    source=dict(path='work/synthetic.html',sha256='a'*64)
    period=dict(current_end='2024-12-31',prior_end='2023-12-31',proposed_current_start='2024-01-01',days_inclusive=366)
    ep=dict(current=dict(period_start='2024-01-01',period_end='2024-12-31',days_inclusive=366),comparison=dict(period_end='2023-12-31'))
    fields={name:dict(scale='1000' if symbol=='PLTR' else '1000000',currency_literal='$',currency_iso=None,as_of='2023-12-31' if name=='prior_total_assets' else '2024-12-31') for name in m.FIELDS}
    # Use a declared issuer/year while testing a synthetic endpoint separately.
    year={'PLTR':2024,'PYPL':2018,'TMO':2020}[symbol]
    c=dict(event_id=symbol+'_synthetic',symbol=symbol,fiscal_year=year,source=source,context_holds=['retained'],publication_date='2025-02-03',available_session='2025-02-04',periods=period,fields=fields)
    e={k:copy.deepcopy(c[k]) for k in ('event_id','symbol','fiscal_year','source','context_holds','publication_date')}
    e.update(periods=ep,available_from=c['available_session'],gaps=[],fact_admitted=False,model_admitted=False,strategy_admitted=False)
    e['fields']={name:dict(**copy.deepcopy(f),source=copy.deepcopy(source),available_from=c['available_session'],period=copy.deepcopy(ep['current']) if name=='annual_net_income' else None,fact_admitted=False) for name,f in fields.items()}
    return c,e


class ScopeTests(unittest.TestCase):
    def test_three_declared_issuers(self):
        for s in ('PLTR','PYPL','TMO'):m.assert_scope(*fixture(s))
    def test_week_year_rejected(self):
        c,e=fixture();c['periods']['days_inclusive']=364
        with self.assertRaises(ValueError):m.assert_scope(c,e)
    def test_changed_context_holds(self):
        c,e=fixture();e['context_holds']=[]
        with self.assertRaises(ValueError):m.assert_scope(c,e)
    def test_changed_source(self):
        c,e=fixture();e['source']['sha256']='b'*64
        with self.assertRaises(ValueError):m.assert_scope(c,e)
    def test_changed_asset_date(self):
        c,e=fixture();e['fields']['prior_total_assets']['as_of']='2024-06-30'
        with self.assertRaises(ValueError):m.assert_scope(c,e)
    def test_changed_currency(self):
        c,e=fixture();e['fields']['annual_net_income']['currency_iso']='USD'
        with self.assertRaises(ValueError):m.assert_scope(c,e)
    def test_changed_scale(self):
        c,e=fixture();e['fields']['annual_net_income']['scale']='1000000'
        with self.assertRaises(ValueError):m.assert_scope(c,e)
    def test_null_preserved_with_explicit_gap(self):
        c,e=fixture();e['fields']['annual_net_income']=None;e['gaps']=[dict(field='annual_net_income',reason='not located')]
        m.assert_scope(c,e)
    def test_null_without_gap_fails(self):
        c,e=fixture();e['fields']['annual_net_income']=None
        with self.assertRaises(ValueError):m.assert_scope(c,e)
    def test_premature_admission_fails(self):
        c,e=fixture();e['fact_admitted']=True
        with self.assertRaises(ValueError):m.assert_scope(c,e)


class EvidenceTests(unittest.TestCase):
    def test_repeated_label_locates_selected_cell(self):
        s=BeautifulSoup('<p>Net income</p><table><tr><td>Net income</td><td>DO_NOT_READ_AMOUNT</td></tr></table>','html.parser')
        text,pos=m.visible_document(s);q=m.quote_cell(s.td,dict(path='x',sha256='a'),text,pos,0,0,0)
        self.assertGreater(q['locator']['start'],0);self.assertEqual(q['quote'],'Net income');self.assertNotIn('DO_NOT',q['quote'])
    def test_entities_unicode_and_nested_text(self):
        s=BeautifulSoup('<p>é &amp; <b>scope</b></p><table><tr><td>Total assets</td></tr></table>','html.parser');text,pos=m.visible_document(s)
        q=m.quote_cell(s.td,dict(path='x',sha256='a'),text,pos,0,0,0)
        self.assertEqual(text[q['locator']['start']:q['locator']['end']],'Total assets')
    def test_scripts_ignored(self):
        s=BeautifulSoup('<script>ignored amount</script><p>GAAP</p>','html.parser');text,pos=m.visible_document(s);self.assertEqual(text,'GAAP')
    def test_missing_exact_quote_fails(self):
        with self.assertRaises(ValueError):m.quote_literal('absent',dict(path='x',sha256='a'),'other')
    def test_changed_quote_fails(self):
        source=dict(path='x',sha256='a');q=m.quote_literal('GAAP',source,'GAAP');q['quote']='OTHER'
        item=dict(source=source,status='resolved',scope='current_annual_net_income_same_report_assets',field_checks={f:True for f in m.FIELDS},current_prior_common_basis=True,basis='same report',evidence=[q])
        with self.assertRaises(ValueError):m.verify_item(item,source,'GAAP')


class DecisionTests(unittest.TestCase):
    def previous(self):return dict(event_id='X',findings=['Independently reviewed scope.'],unresolved_questions=[])
    def test_early_clock_not_accounting_rejection(self):
        p=self.previous();p['unresolved_questions']=['Publication lies outside the pinned calendar evidence.']
        self.assertEqual(m.scoped_decision(p,{})['status'],'resolved')
    def test_actual_unresolved_requires_resolution(self):
        p=self.previous();p['unresolved_questions']=['Unknown opening asset basis.']
        with self.assertRaises(ValueError):m.scoped_decision(p,{})
    def test_limited_resolution_used(self):
        p=self.previous();p['unresolved_questions']=['Unknown prior legal identity.']
        r=dict(status='resolved',scope='current_annual_net_income_same_report_assets',field_scope_checks={f:True for f in m.FIELDS},findings=['As reported only.'],limitations=['Not standalone.'])
        answer=m.scoped_decision(p,{'X':r});self.assertEqual(answer['basis'],'As reported only.');self.assertIn('Not standalone.',answer['limitations'])
    def test_unresolved_not_auto_cleared(self):
        r=dict(status='unresolved',scope='current_annual_net_income_same_report_assets',field_scope_checks={f:True for f in m.FIELDS})
        with self.assertRaises(ValueError):m.scoped_decision(self.previous(),{'X':r})
    def test_wrong_source_run_rejected_before_io(self):
        with self.assertRaises(ValueError):m.build('a'*64,'b'*64)

if __name__=='__main__':unittest.main(verbosity=2)
