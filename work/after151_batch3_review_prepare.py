from pathlib import Path
p=Path('work/after151_batch3_independent_review.py');s=p.read_text()
s=s.replace("def statement(soup,context,field):", "def statement(soup,context,field,accounting=None):")
s=s.replace("    require(len(titles)==1 and not BAD.search(titles[0]) and not any(BAD.search(c['text']) for c in headers),'Formal GAAP statement required')", """    require(len(titles)==1 and not BAD.search(titles[0]),'Formal GAAP statement required')
    for c in headers:
        if not BAD.search(c['text']):continue
        # A separately source-verified manual decision can explain only the
        # exact retrospective header. Non-GAAP/pro-forma remains rejected.
        allow=(accounting is not None and accounting.get('status')=='resolved'
               and accounting.get('gaap_retrospective_adjustment_verified') is True
               and re.fullmatch(r'as\\s+adjusted(?:\\s+\\d+)?',c['text'],re.I))
        require(allow,'Formal GAAP statement required')""")
s=s.replace('def source_field(soup,context,field):','def source_field(soup,context,field,accounting=None):').replace('statement(soup,context,field)\n    groups','statement(soup,context,field,accounting)\n    groups')
s=s.replace('def review_position(soup,context,field,event,period):','def review_position(soup,context,field,event,period,accounting=None):').replace('source_field(soup,context,field)','source_field(soup,context,field,accounting)').replace('statement(soup,context,field)\n        groups','statement(soup,context,field,accounting)\n        groups')
s=s.replace('def review_event(reader,row,event,context,sessions,policy):','def review_event(reader,row,event,context,sessions,policy,accounting=None):')
s=s.replace("    receipt=receipt_evidence(reader,context,raw);soup=BeautifulSoup(raw,'html.parser')", "    if accounting is not None:verify_accounting(accounting,event,raw)\n    receipt=receipt_evidence(reader,context,raw);soup=BeautifulSoup(raw,'html.parser')")
s=s.replace('review_position(soup,context,f,event,expected)','review_position(soup,context,f,event,expected,accounting)')
s=s.replace("    if item['status']=='resolved':", """    if item.get('gaap_retrospective_adjustment_verified') is True:
        quotes=' '.join(e['quote'] for e in item['evidence']).lower()
        require(item['status']=='resolved' and
                'change in accounting methodology has been applied retrospectively' in quotes and
                'adjusted all prior period financial information presented herein as required' in quotes and
                'pension plans' in quotes,'Retrospective GAAP exception lacks explicit source proof')
    if item['status']=='resolved':""")
s=s.replace("review_event(reader,row,event,contexts[row['event_id']],sessions,policy)", "review_event(reader,row,event,contexts[row['event_id']],sessions,policy,decisions[event['event_id']])")
p.write_text(s)
p=Path('work/test_after151_batch3_independent_review.py');s=p.read_text();idx=s.index("if __name__=='__main__':")
s=s[:idx]+'''class RetrospectiveTests(unittest.TestCase):
    def adjusted(self,header='As Adjusted 1'):
        raw,c,e,*_=fixture()
        raw=raw.replace(b'Year Ended',('Year Ended</td><td>'+header).encode())
        # Use an extra header inside the existing title cell, preserving columns.
        raw,c,e,*_=fixture()
        raw=raw.replace(b'(In millions, except per share)',('(In millions, except per share)</td></tr><tr><td>'+header).encode())
        # Rebuild row positions after the deliberately inserted header row.
        c['statement_contexts'][0]['header_stop_row']=4
        c['fields'][m.FIELDS[0]]['row']=5
        rebuild_context(c,raw)
        return raw,c,e
    def test_adjusted_requires_explicit_manual_decision(self):
        raw,c,e=self.adjusted()
        with self.assertRaises(ValueError):m.source_field(BeautifulSoup(raw,'html.parser'),c,m.FIELDS[0])
    def test_adjusted_formal_header_explained(self):
        raw,c,e=self.adjusted()
        r=m.source_field(BeautifulSoup(raw,'html.parser'),c,m.FIELDS[0],dict(status='resolved',gaap_retrospective_adjustment_verified=True))
        self.assertEqual(r['value_decimal'],'-31')
    def test_non_gaap_not_exempted(self):
        for text in ['Non-GAAP As Adjusted 1','Reconciliation','Pro forma','As Adjusted forecast']:
            raw,c,e=self.adjusted(text)
            with self.assertRaises(ValueError):m.source_field(BeautifulSoup(raw,'html.parser'),c,m.FIELDS[0],dict(status='resolved',gaap_retrospective_adjustment_verified=True))
    def test_unresolved_does_not_exempt(self):
        raw,c,e=self.adjusted()
        with self.assertRaises(ValueError):m.source_field(BeautifulSoup(raw,'html.parser'),c,m.FIELDS[0],dict(status='unresolved',gaap_retrospective_adjustment_verified=True))
    def test_claim_without_quoted_source_proof_rejected(self):
        raw,c,e,*_=fixture()
        item=dict(event_id=e['event_id'],source=e['source'],status='resolved',scope='current_annual_net_income_same_report_assets',reasons=[],basis='Reviewed',field_checks={f:True for f in m.FIELDS},current_prior_common_basis=True,evidence=[],gaap_retrospective_adjustment_verified=True)
        with self.assertRaisesRegex(ValueError,'Retrospective'):m.verify_accounting(item,e,raw)


'''+s[idx:];p.write_text(s)
