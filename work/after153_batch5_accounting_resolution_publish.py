"""Publish manually assessed limited as-reported scope; not value admission."""
from datetime import datetime, timezone
import hashlib,json,re,sys
from bs4 import BeautifulSoup
import after153_batch5_accounting_resolution as m
CONTEXT='work/after153_next_source_context.json'
CONTEXT_SHA='1bd968ca401a313af2ee6c1e92328ae8b270e7c78e607b14edb61060ecc6f241'
FIELDS=('annual_net_income','current_total_assets','prior_total_assets')


def evidence_at(report,start):
    matched=[e for e in report['passages'] if e['locator']['start']==start]
    m.check(len(matched)==1,'Unique reviewed byte span')
    return matched[0]


def verify_evidence(e,pins):
    raw=m.read(e['path'],pins,e['sha256']);loc=e['locator']
    fragment=raw[loc['start']:loc['end_exclusive']]
    m.check(hashlib.sha256(fragment).hexdigest()==e['raw_span_sha256'],'Evidence bytes unchanged')
    text=m.norm(BeautifulSoup(fragment,'html.parser').get_text(' ',strip=True))
    m.check(m.redact(text)==e['quote_numbers_redacted'],'Redacted evidence exact')
    return text


def synthetic_checks():
    text='é\n<p>Pension &amp; scope <em>unchanged</em>.</p><td>GAAP 2020</td>'
    p=m.Blocks(text);p.feed(text)
    m.check(len(p.blocks)==2,'Synthetic blocks')
    first=sorted(p.blocks)[0];a=len(text[:first[0]].encode());b=len(text[:first[1]].encode())
    m.check(text.encode()[a:b].decode()=='<p>Pension &amp; scope <em>unchanged</em>.</p>','Multibyte byte map')
    m.check(m.norm(BeautifulSoup(text.encode()[a:b],'html.parser').get_text(' ',strip=True))=='Pension & scope unchanged .','Entity normalization')
    m.check(m.redact('GAAP $12,345.67 (20) 2017')=='GAAP $[number] ([number]) [number]','No amounts emitted')
    m.check(not re.search(r'\d',m.redact('123 45.67')),'No residual numerals')
    return 5


def publish(reports=None,pins=None):
    tests=synthetic_checks()
    if reports is None:reports,pins=m.scan()
    ctx=json.loads(m.read(CONTEXT,pins,CONTEXT_SHA));events={e['event_id']:e for e in ctx['events']}
    reviews=[]
    for report in reports:
        eid=report['event_id'];c=events[eid]
        m.check(c['source']==report['source'],'Same pinned report')
        p=c['periods'];m.check(p['days_inclusive']==365 and p['current_end'][5:]=='12-31' and p['prior_end'][5:]=='12-31','Actual selected calendar boundaries')
        m.check(all(c['fields'][f]['scale']=='1000000' and c['fields'][f]['currency_literal']=='$' for f in FIELDS),'Common same-report monetary unit')
        if eid=='PYPL_2016-01-27':
            selected=[evidence_at(report,n) for n in (54677,63864,133163,139148,57127,60051,205934)]
            findings=[
              'Both selected statements explicitly identify the reporting presentation as Combined and Consolidated under PayPal Holdings, Inc.; the release identifies its financial statements as GAAP.',
              'The annual GAAP NI and the December 31, 2015 and 2014 total assets belong to that same presented PayPal reporting business. The combination-to-consolidation transition is disclosed, not an accidental mix with total eBay accounts.',
              'Separation costs, related-party balances, historical allocation effects and the actual capital structure remain in the reported GAAP figures. The separately described non-GAAP pro-forma adjustments must not replace any of the three inputs.',
              'For a measure defined as current reported GAAP income relative to same-report beginning/end assets, these disclosed transition effects are economic limitations rather than a source-basis contradiction. This is not constant-standalone-performance normalization.'
            ]
            limitations=[
              'Does not establish continuous identity as a standalone legal issuer before the separation or constant legal/reporting perimeter throughout the year.',
              'No certification that historical allocations equal standalone costs; resulting cross-period/cross-issuer economic comparability is limited.',
              'No pro-forma, eBay-total, EPS/share-count or subsequent-report substitution is permitted.'
            ]
            issue='Combined/pre-separation and consolidated/post-separation presentation'
            unknown={'continuous_standalone_legal_identity':'not_established_and_not_required_by_this_use','counterfactual_standalone_costs':'not_established_and_not_used'}
        else:
            selected=[evidence_at(report,109515),evidence_at(report,178723)]
            note=verify_evidence(selected[0],pins)
            m.check('The results for 2017 have been restated for the immaterial impacts of adopting new guidance on pension accounting.' in note,'Exact 2017 restatement scope')
            findings=[
              'The release explicitly identifies reported results as GAAP and says the results for 2017 were restated for immaterial impacts of new pension-accounting guidance.',
              'The selected numerator is current 2018 annual total NI, not the 2017 comparison NI, adjusted earnings or a reconstructed pre-adoption NI.',
              'Both selected total-asset endpoints are the December 31, 2018 and 2017 figures presented together in this same January 30, 2019 report. No opening asset value is borrowed from an earlier report.',
              'The source does not say whether pension adoption affected only presentation or also measurement/capitalization. This review therefore does not certify unchanged assets or a classification-only effect.',
              'The disclosed restatement does not contradict the limited as-reported same-report basis. Use of the three current-publication GAAP inputs does not require comparison NI or proof that previous publications contained identical asset values.'
            ]
            limitations=[
              'Classification-only versus measurement/capitalization impact of the pension guidance remains unspecified by the inspected source.',
              'Zero effect on either selected total-asset endpoint is not established. No such claim is needed for the declared as-reported measure.',
              'A differently defined common-policy counterfactual or cross-publication invariant-asset measure would require additional evidence and is not approved here.'
            ]
            issue='Immaterial prior-year pension-guidance restatement'
            unknown={'pension_effect_classification_only':'unknown','selected_assets_zero_impact':'unknown','prior_publication_asset_values_identical':'not_checked_and_not_required'}
        for e in selected:verify_evidence(e,pins)
        reviews.append(dict(event_id=eid,source=report['source'],issue=issue,status='resolved',
          scope='current_annual_net_income_same_report_assets',
          permitted_use='limited_as_reported_consolidated_or_combined_and_consolidated_GAAP_current_NI_and_same_report_asset_endpoints',
          resolution_type='field_scope_accounting_context_only_not_value_admission',
          field_scope_checks={f:True for f in FIELDS},current_prior_common_reported_basis=True,
          common_reported_basis_does_not_mean_invariant_legal_perimeter_or_unchanged_prior_published_amounts=True,
          selected_metadata={f:{k:c['fields'][f][k] for k in ('table','row','label','logical_columns','as_of','scale','currency_literal','currency_iso')} for f in FIELDS},
          periods=p,findings=findings,evidence=selected,unknown_claims=unknown,limitations=limitations,
          selected_values_verified=False,field_admitted=False,fact_admitted=False,model_admitted=False,strategy_admitted=False))
    supplemental=[]
    for issuer,stem,date in (('PYPL','c5dd6b4c3afd11106b43','2015-10-28'),('TMO','c1163eb22b3373bd49f9','2017-07-26')):
        name=f'outputs/monthly_earnings_repair/batch_financial_consumer/{issuer}/raw/{stem}.html'
        receipt=json.loads(m.read(name+'.meta.json',pins));raw=m.read(name,pins,receipt['sha256'])
        m.check(receipt['status']==200,'Supplemental receipt')
        supplemental.append(dict(path=name,sha256=pins[name],receipt_path=name+'.meta.json',receipt_sha256=pins[name+'.meta.json'],publication_date=date,role='Earlier local context inspected; not a replacement for either selected report and not used to assert absent accounting effects'))
    for path in ('work/after153_batch5_accounting_resolution.py','work/after153_batch5_accounting_resolution_publish.py','work/after153_batch5_accounting_resolution_extra.py'):m.read(path,pins)
    for path,digest in list(pins.items()):m.read(path,pins,digest)
    result=dict(schema_version=1,stage='independent_limited_as_reported_accounting_resolution_not_source_admission',passed=True,
      created_utc=datetime.now(timezone.utc).isoformat(),runtime=sys.version,input_hashes=pins,synthetic_checks_passed=tests,
      source_order=['PYPL','TMO'],events=reviews,earlier_local_context_examined=supplemental,
      scope='Resolve only the two identified accounting-basis questions using the selected as-of reports; retain original feasibility record unchanged.',
      policy_interpretation='A same-report as-reported GAAP NI/asset measure accepts disclosed combined-to-consolidated presentation and same-publication restatement. It does not assert counterfactual standalone or unchanged historical values.',
      exclusions=['No future report is used to repair or replace as-of fields.','No amount, ratio, factor, order or return was computed.','This artifact does not replace independent monetary-cell, annual-column, sign, publication-clock or source-admission checks.','Shared dollar literal is not ISO-currency certification. Calendar-year 365/366 policy remains distinct from 364/371-week issuers.','Original context holds/provenance stay preserved; a later explicit source-use gate may consume this scoped resolution.'],
      counts=dict(events_reviewed=2,accounting_scope_resolved=2,accounting_scope_held=0,numeric_fields_verified=0,new_source_admissions=0),
      financial_values_extracted=0,ratios_computed=0,factors_computed=0,orders_or_returns_computed=0)
    with (m.ROOT/m.OUT).open('x') as output:json.dump(result,output,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);output.write('\n')
    check=json.loads((m.ROOT/m.OUT).read_bytes());m.check(check==result,'Round-trip exact')
    print(json.dumps(dict(passed=True,output=m.OUT,sha256=hashlib.sha256((m.ROOT/m.OUT).read_bytes()).hexdigest(),counts=result['counts'],synthetic_checks=tests)))

if __name__=='__main__':publish()
