"""Independent offline review of period-end total-share evidence, not valuation."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, io, json, re, sys

R = Path(__file__).resolve().parents[1]
O = R / 'outputs/monthly_shell_pool_source168'
PLAN_SHA = '2ff82efd4ac8abdca1ca9a0a97addad081ceb4dc8152195b38f873499193d606'
OUTPUT_SHA = '9be66db12b2583526f01f028ae5f07920be5b44f526eda710665a971a1b3688a'
CODE_SHA = '712db8290d755480e324e9ab1afd09d25ff6f5a9b6acf4ba9cfd198bcfd9a5ac'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def norm(text):
    text = re.sub(r'(?<=\w)-\s*\n\s*(?=\w)', '', text)
    return ' '.join(text.split())

def scan():
    assert sha(O/'TOTAL_OUTSTANDING_PLAN.json') == PLAN_SHA
    assert sha(O/'TOTAL_OUTSTANDING.json') == OUTPUT_SHA
    assert sha(R/'work/shell_pool_source168_outstanding.py') == CODE_SHA
    plan = json.loads((O/'TOTAL_OUTSTANDING_PLAN.json').read_text())
    for pin in ['original_plan','pdf_runtime_plan']:
        assert sha(R/plan[pin]['path']) == plan[pin]['sha256']
    for path, digest in plan['protected_existing_outputs'].items():
        assert sha(R/path) == digest
    runtime = json.loads((R/plan['pdf_runtime_plan']['path']).read_text())
    for path, digest in runtime['pdf_runtime_hashes'].items():
        assert sha(path) == digest
    sys.path.append(runtime['pdf_runtime']['bundle_site_packages'])
    import pypdfium2 as pdfium
    from pypdf import PdfReader
    producer = json.loads((O/'TOTAL_OUTSTANDING.json').read_text())
    result = []
    for entry, produced in zip(plan['records'],producer['records']):
        path=R/entry['binary'];assert sha(path)==entry['sha256']
        data=path.read_bytes();doc=pdfium.PdfDocument(data);other=PdfReader(io.BytesIO(data))
        ident='%dQ%d'%(entry['year'],entry['quarter']);assert produced['period']==ident
        assert len(doc)==len(other.pages)==produced['pages_scanned']
        patterns=[r'\boutstanding\b',r'(?:total|number|balance).{0,70}\bshares?\b',
                  r'\bshares?\b.{0,90}(?:period.end|end.of.(?:the.)?period|net.of|excluding|less|in.issue)',
                  r'\bshares?\b.{0,45}\bat\b.{0,15}(?:December|September|June|March)']
        matches=[];exact=[];alt_exact=[];empty=[];eps=[];capital=[]
        for i in range(len(doc)):
            text=norm(doc[i].get_textpage().get_text_range())
            alt=norm(other.pages[i].extract_text())
            if len(text)<100:empty.append(i+1)
            if re.search(r'shares\s+outstanding\s+at\s+the\s+end\s+of\s+the\s+period',text,re.I):exact.append(i+1)
            if re.search(r'shares\s+outstanding\s+at\s+the\s+end\s+of\s+the\s+period',alt,re.I):alt_exact.append(i+1)
            if re.search(r'weighted average.{0,25}shares',text,re.I):eps.append(i+1)
            if re.search(r'share capital|issued and fully paid',text,re.I):capital.append(i+1)
            for pattern in patterns:
                for m in re.finditer(pattern,text,re.I):
                    context=text[max(0,m.start()-80):m.end()+110]
                    row=dict(page=i+1,pattern=pattern,context=context)
                    if row not in matches:matches.append(row)
        assert not empty,(ident,empty)
        assert exact==alt_exact,(ident,exact,alt_exact)
        result.append(dict(period=ident,source=entry,pages_scanned=len(doc),pages_with_less_than_100_text_characters=empty,
                           exact_label_pdfium_pages=exact,exact_label_pypdf_pages=alt_exact,
                           broader_search_matches=matches,EPS_context_pages=eps,capital_context_pages=capital))
        if ident=='2015Q4':
            target_page=doc[9]
            independent_text=norm(target_page.get_textpage().get_text_range())
            assert 'Quarters Millions Full year Q4 2015 Q3 2015 Q4 2014 2015 2014' in independent_text
            assert '6,397.5 6,348.4 6,295.0 Shares outstanding at the end of the period 6,397.5 6,295.0' in independent_text
            assert 'Royal Dutch Shell plc ordinary shares of €0.07 each' in independent_text
            target_page.render(scale=1.6).to_pil().save('/tmp/shell168_outstanding_independent_page10.png')
        doc.close()
    return plan,producer,result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--scan',action='store_true');parser.add_argument('--review',action='store_true')
    args=parser.parse_args();plan,producer,result=scan()
    if args.scan:
        for record in result:print(json.dumps(record,ensure_ascii=False))
    if args.review:
        assert len(result)==len(producer['records'])==20
        checks=[]
        for source,p in zip(result,producer['records']):
            ident=source['period'];expected=[10] if ident=='2015Q4' else []
            labelchecks=dict(
                original_source_hash=p['source']['sha256']==source['source']['sha256'],
                original_source_identity=p['source']['binary']==source['source']['binary'] and p['source']['url']==source['source']['source_url'],
                complete_page_count=p['pages_scanned']==source['pages_scanned'],
                independent_exact_label_pages=source['exact_label_pdfium_pages']==source['exact_label_pypdf_pages']==expected,
                reported_label_pages=p['text_engine_agreement']['pypdf_target_pages']==p['text_engine_agreement']['pdfium_target_pages']==expected,
                trust_basis_unknown=p['trust_exclusion_explicit'] is None,
                treasury_basis_unknown=p['treasury_exclusion_explicit'] is None,
                no_substitution=p['no_interpolation_or_substitution'] is True)
            if not expected:
                labelchecks.update(absent_left_null=p['explicit_period_end_total'] is None,
                    missing_status=p['status']=='missing_explicit_period_end_total',
                    no_outstanding_keyword=not any(x['pattern']==r'\boutstanding\b' for x in source['broader_search_matches']))
            else:
                target=p['explicit_period_end_total'];interval=target['conditional_rounding_interval']
                labelchecks.update(displayed_value=target['displayed_value']=='6,397.5',
                    source_units=target['source_units']=='Millions' and target['security_unit']=='ordinary shares' and target['ads'] is False,
                    correct_page=target['source_page_1based']==10,exact_count_unknown=target['exact_integer_count'] is None,
                    comparison_columns=target['comparator_cells']=={'prior_quarter':'6,348.4','year_ago_quarter':'6,295.0','full_year_current':'6,397.5','full_year_previous':'6,295.0'},
                    conditional_interval_labeled=interval['conditional_only'] is True and interval['source_certified_interval'] is False,
                    conditional_interval_values=interval['displayed_step_millions']=='0.1' and interval['lower_ordinary_shares']=='6397450000.00' and interval['upper_ordinary_shares']=='6397550000.00',
                    conditional_rounding_not_claimed_source_fact='source rounding rule is not stated' in interval['assumption'],
                    status=p['status']=='reported_rounded_total_netting_basis_unresolved')
            assert all(labelchecks.values()),(ident,labelchecks)
            checks.append(dict(period=ident,source=source['source'],pages_scanned=source['pages_scanned'],checks=labelchecks,
                exact_label_pages=expected,broader_match_count=len(source['broader_search_matches']),
                broader_match_pages=sorted(set(x['page'] for x in source['broader_search_matches'])),
                outstanding_contexts=[x for x in source['broader_search_matches'] if x['pattern']==r'\boutstanding\b'],
                EPS_pages=source['EPS_context_pages'],issued_capital_context_pages=source['capital_context_pages'],
                determination='explicit rounded period-end ordinary total' if expected else 'no explicit period-end outstanding total identified'))
        for path,digest in plan['protected_existing_outputs'].items():assert sha(R/path)==digest
        assert sha(O/'TOTAL_OUTSTANDING.json')==OUTPUT_SHA
        assert sha(O/'FOLLOWUP_REVIEW.json')=='b33d4af2bcc8907d629182482d36b76aa9126499fc895183a5a68bb344b07a7e'
        out=dict(schema='shell_pool_source168_total_outstanding_independent_review_v1',
            created_utc=datetime.now(timezone.utc).isoformat(),plan_sha256=PLAN_SHA,producer_output_sha256=OUTPUT_SHA,
            producer_script_sha256=CODE_SHA,reviewer_script_sha256=sha(__file__),
            method='Read every original PDF page with PDFium and pypdf independently of producer code; checked exact label plus wider outstanding/total/number/balance/period-end/net/exclusion contexts. Visually inspected original 2015Q4 physical PDF page 10.',
            summary=dict(documents=20,original_pages_scanned=sum(x['pages_scanned'] for x in checks),
                         reported_period_end_totals=1,missing_period_end_totals=19,source_value_or_unit_mismatches=0,
                         producer_script_review='pass for these twenty pinned documents',status='PASS_SOURCE_SUPPLEMENT_ONLY'),
            records=checks,
            visual_review=dict(period='2015Q4',physical_PDF_page_1based=10,displayed='6,397.5',unit='million ordinary shares',
                statement='Shares outstanding at the end of the period',column='Q4 2015; independently repeated in Full year 2015',
                footnote='Royal Dutch Shell plc ordinary shares of €0.07 each',
                render_path_at_review='/tmp/shell168_outstanding_independent_page10.png',
                render_sha256=sha('/tmp/shell168_outstanding_independent_page10.png')),
            source_script_review=[
                'Producer reads the pinned originals and verifies runtime/source hashes; it does not import an existing cap or earnings denominator as a substitute.',
                'Parser demands the exact row, Millions heading, ordinary-share footnote and Q4 current/prior headers, and rejects ambiguous repeats; its hard-coded Q4 layout is appropriate only to the found 2015Q4 table, not a generic future parser.',
                'Quarter and full-year repeated current total agree; source precision remains one decimal million and exact integer count remains null.',
                'Nearest-rounding interval is explicitly conditional. Inclusive boundaries are conservative under unspecified tie handling; neither interval nor nearest-rounding rule is asserted as source-certified.',
                'Absent entries remain null with missing reasons. Weighted-average EPS and issued A/B counts are never substituted.',
                'Netting against treasury or employee-trust holdings remains unknown; monetary trust balances are not treated as share counts.'],
            absence_scope='Nineteen missing results mean no explicit current-period-end outstanding total was identified in these nineteen specific original QRA PDFs by this source scan. They do not mean zero shares, no other public disclosure, or a proven universal absence of the fact.',
            limitations=['No exact A/B net outstanding split is supplied by the one rounded total.',
                'The ordinary-share footnote specifies security units, not explicit treasury/trust exclusion mechanics.',
                'No period-end total was inferred from EPS-weighted counts, issued-capital sums, dividend records, rounding or later reports.',
                'This source supplement does not establish any price, market capitalization, pool membership or strategy result.'],
            old_outputs_unchanged=True,network_requests=0,market_caps_computed=False,ranks_computed=False,returns_computed=False,pool_modified=False)
        destination=O/'TOTAL_OUTSTANDING_REVIEW.json';assert not destination.exists()
        destination.write_text(json.dumps(out,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n')
        print(json.dumps(dict(summary=out['summary'],review_sha256=sha(destination)),sort_keys=True))

if __name__=='__main__':main()
