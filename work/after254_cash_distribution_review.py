"""Bounded cash-distribution source inventory; no feature or amount arithmetic.

Run only with run_local.sh.  --inspect prints metadata; --write publishes one
new task note.  No strategy modules are imported and no facts' val is consumed.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SELF = 'work/after254_cash_distribution_review.py'
OUT = 'work/after254_cash_distribution_review.json'
SOURCE_PLAN = 'work/after158_basic_shares_source_plan.json'
BRIDGE = 'work/after158_basic_shares_gaap_bridge_v1/'
ADMISSION = 'outputs/after158_basic_shares_admission_v2/EVENTS.json'
TAGS = {
 'repurchase_common': ['PaymentsForRepurchaseOfCommonStock'],
 'dividends_common': ['PaymentsOfDividendsCommonStock'],
 'dividends_broad': ['PaymentsOfDividends'],
 'issuance_common': ['ProceedsFromIssuanceOfCommonStock'],
 'issuance_plan': ['ProceedsFromIssuanceOfSharesUnderIncentiveAndShareBasedCompensationPlansIncludingStockOptions', 'ProceedsFromStockPlans'],
 'issuance_options_only': ['ProceedsFromStockOptionsExercised'],
 'withholding_payment_not_component': ['PaymentsRelatedToTaxWithholdingForShareBasedCompensation'],
 'repurchase_other_not_component': ['PaymentsForRepurchaseOfOtherEquity'],
 'issuance_preferred_not_component': ['ProceedsFromIssuanceOfPreferredStockAndPreferenceStock'],
}
DEFINITIONS = {
 '146': {'path':'work/monthly_cash_quality146_factor.py', 'formula':'NI / average_assets; CFO / average_assets; (CFO - NI) / average_assets', 'relation':'Earnings, operating cash generation and accruals; no shareholder payment measure.'},
 '149': {'path':'work/monthly_stock_comp149_factor.py', 'formula':'(CFO - stock_compensation_addback) / average_assets', 'relation':'Noncash compensation adjustment; not actual cash paid to shareholders or net share issuance.'},
 '159': {'path':'work/basic_shares159_independent.py', 'formula':'1 - current_annual_basic_weighted_average_shares / same_report_prior_annual_basic_weighted_average_shares', 'relation':'EPS denominator change; not end-period outstanding shares and not cash repurchases less cash issuance.'},
 '193': {'path':'work/monthly_cash_ppe193_independent.py', 'formula':'(CFO - cash_PPE_outflow) / average_assets', 'relation':'Cash after reported PPE spending; no financing distributions.'},
}
PINS = {}

def pin(name, expected=None):
    p = Path(name)
    p = p if p.is_absolute() else ROOT / p
    p = p.resolve()
    if not p.is_relative_to(ROOT) or any(s in str(p) for s in ('CloudStorage','OneDrive','stock-observer-runtime')):
        raise ValueError('Outside local research root')
    key = p.relative_to(ROOT).as_posix()
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    if expected is not None and digest != expected:
        raise ValueError('Input SHA differs: '+key)
    if key in PINS and PINS[key] != digest:
        raise ValueError('Input changed: '+key)
    PINS[key] = digest
    return p

def read(name, expected=None):
    return json.loads(pin(name, expected).read_text())

class Rows(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tables = 0; self.rows = []; self.stack = []; self.row = None; self.cell = None
    def handle_starttag(self, tag, attrs):
        if tag == 'table':
            self.tables += 1; self.stack.append([self.tables-1,0])
        elif tag == 'tr' and self.stack:
            self.row = {'table_index_zero_based':self.stack[-1][0], 'row_index_zero_based':self.stack[-1][1], 'cells':[]}
            self.stack[-1][1] += 1
        elif tag in ('td','th') and self.row is not None:
            self.cell = []
    def handle_data(self, value):
        if self.cell is not None: self.cell.append(value)
    def handle_endtag(self, tag):
        if tag in ('td','th') and self.cell is not None and self.row is not None:
            self.row['cells'].append(' '.join(' '.join(self.cell).split())); self.cell = None
        elif tag == 'tr' and self.row is not None:
            self.rows.append(self.row); self.row = None; self.cell = None
        elif tag == 'table' and self.stack: self.stack.pop()

LABEL = re.compile(r'repurchas|dividend|(?:common stock|stock|shares) (?:issued|issuance)|proceeds.{0,100}(?:stock|share)|tax.{0,60}(?:equity|stock|share)', re.I)
PDF_LABEL = re.compile(r'(?:Proceeds from (?:the )?issuance of common stock(?: under [A-Za-z -]+)?|Payments for dividends and dividend equivalents|Repurchases of common stock|Payments for taxes related to net share settlement of equity awards)', re.I)

def original_labels(event):
    src = event['source']; raw = pin(src['path'], src['sha256'])
    pin(src['receipt']['path'],src['receipt']['sha256'])
    text = event.get('existing_text')
    if text:
        body = pin(text['path'],text['sha256']).read_text()
        items = [{'character_offset':m.start(),'label':m.group()} for m in PDF_LABEL.finditer(body)]
        return {'path':text['path'], 'format':'pinned_pdf_derived_text', 'row_candidates':items,
                'limitation':'Lexical row labels only; no fresh PDF geometry or annual column/value certification.'}
    parser = Rows(); parser.feed(raw.read_text(errors='replace'))
    candidates = []
    for row in parser.rows:
        cells = [x for x in row['cells'] if x]
        if not cells: continue
        label = cells[0]
        if len(label) <= 250 and LABEL.search(label):
            candidates.append({k:row[k] for k in ('table_index_zero_based','row_index_zero_based')} | {'label':label})
    return {'path':src['path'],'format':'html','row_candidates':candidates,
            'limitation':'First-cell labels only. Table ancestry, annual column, units, amounts and complete financing-line reconciliation are not admitted.'}

def cash_contexts(cache, selected, period, fiscal_year):
    groups = {}; glossary = {}
    for family, tags in TAGS.items():
        rows = []
        for tag in tags:
            field = cache['facts'].get('us-gaap',{}).get(tag)
            if field is None: continue
            glossary[tag] = {'label':field.get('label'),'description':field.get('description'), 'unit':'USD', 'family':family}
            for index, fact in enumerate(field.get('units',{}).get('USD',[])):
                if (fact.get('accn'),fact.get('filed'),fact.get('form'),fact.get('fp'),fact.get('fy'),fact.get('start'),fact.get('end')) == (
                    selected['selected_accession'],selected['selected_filed'],'10-K','FY',fiscal_year,period['period_start'],period['period_end']):
                    rows.append({'concept':tag,'unit':'USD','json_pointer':'/facts/us-gaap/%s/units/USD/%d'%(tag,index),
                                 **{k:fact[k] for k in ('accn','filed','form','fp','fy','start','end')}})
        groups[family] = {'status':'missing' if not rows else 'unique' if len(rows)==1 else 'multiple_contexts_not_combined', 'contexts':rows}
    return groups,glossary

def label_families(inventory):
    families = {'repurchase_common':[], 'dividends_paid':[], 'issuance_common':[], 'issuance_plan':[]}
    for row in inventory['row_candidates']:
        label = row['label'].lower()
        if ('repurchas' in label and ('common stock' in label or 'shares' in label)
                and not ('tax' in label or 'withhold' in label)):
            families['repurchase_common'].append(row)
        if (label.startswith('payments for dividends') or label in ('dividends paid','common stock cash dividends paid')):
            families['dividends_paid'].append(row)
        if label == 'common stock issued' or label.startswith('proceeds'):
            if re.search(r'(?:stock|equity) plans|stock-based compensation equity plans|employee',label):
                families['issuance_plan'].append(row)
            elif 'common stock' in label:
                families['issuance_common'].append(row)
    common = all(families[x] for x in ('repurchase_common','dividends_paid','issuance_common'))
    proxy = all(families[x] for x in ('repurchase_common','dividends_paid')) and bool(families['issuance_common'] or families['issuance_plan'])
    return {'families':families,'three_family_label_presence_common_issuance':common,
            'three_family_label_presence_allow_plan_proxy':proxy,'amount_or_annual_column_certified':False}

def build():
    plan = read(SOURCE_PLAN); bridge_plan = read(BRIDGE+'PLAN.json'); metadata = read(BRIDGE+'METADATA.json')
    admission = read(ADMISSION)
    if len(plan['events']) != 44: raise ValueError('Expected original 44 reports')
    selected = {r['event_id']:r for r in metadata['records']}
    timed = {r['event_id']:r for r in admission['events']}
    caches = {}; origins = {}
    for symbol, info in bridge_plan['caches'].items():
        source_name = info['path']; absolute = str((ROOT/source_name).resolve())
        expected = bridge_plan['input_hashes'].get(absolute,bridge_plan['input_hashes'].get(source_name))
        if expected is None: raise ValueError('Cache not bound by existing bridge')
        caches[symbol] = read(source_name,expected)
        if caches[symbol]['cik'] != info['cik']: raise ValueError('CIK changed')
        prov = read(info['provenance']) if info['provenance'] else None
        if prov and prov['sha256'] != PINS[source_name]: raise ValueError('Provenance mismatch')
        origins[symbol] = {'path':source_name,'cik':info['cik'],'entity':caches[symbol]['entityName'],
            'origin_sidecar':info['provenance'],'retrieved_at':None if prov is None else prov.get('retrieved_at'),
            'sec_url':None if prov is None else prov.get('source'),'strict_pit':False}
    records = []; glossary = {}
    for e in plan['events']:
        chosen = selected[e['event_id']]; timing = timed[e['event_id']]
        if chosen['symbol'] != e['symbol'] or chosen['periods'] != e['periods']: raise ValueError('Report alignment')
        if timing['gaap_bridge']['selected_accession'] != chosen['selected_accession']: raise ValueError('Timing accession')
        groups,gl = cash_contexts(caches[e['symbol']],chosen,e['periods']['current'],e['fiscal_year']); glossary.update(gl)
        unique = lambda key: groups[key]['status']=='unique'
        dividends = unique('dividends_common') or unique('dividends_broad')
        strict = unique('repurchase_common') and unique('dividends_common') and unique('issuance_common')
        reported = unique('repurchase_common') and dividends and unique('issuance_common')
        plan_proxy = unique('repurchase_common') and dividends and (unique('issuance_common') or unique('issuance_plan'))
        labels = original_labels(e)
        records.append({'event_id':e['event_id'],'symbol':e['symbol'],'fiscal_year':e['fiscal_year'],
            'original_release':e['source'],'current_annual_period':e['periods']['current'],
            'release_date':e['release_date'],'earliest_cached_selected_accession':chosen['selected_accession'],
            'selected_filed_date':chosen['selected_filed'],'selected_metadata_status':chosen['status'],
            'inherited_release_known_at':timing['known_at'],'inherited_later_source_usable_at':timing['usable_at'],
            'clock_is_conditional_reconstruction_not_first_public_proof':True,
            'cash_fact_groups':groups,'strict_common_three_tag_complete':strict,
            'three_cash_families_common_issuance_complete':reported,
            'three_cash_families_allow_plan_issuance_proxy_complete':plan_proxy,
            'complete_economic_net_distribution_admitted':False,
            'original_report_label_inventory':labels,'original_report_label_families':label_families(labels),
            'inherited_holds':e['inherited_rejection_codes'],
            'first_public_version_verified':False,'complete_correction_history_verified':False})
    summary = {}
    for symbol in caches:
        rows = [r for r in records if r['symbol']==symbol]
        summary[symbol] = {'reports':len(rows), 'years':[r['fiscal_year'] for r in rows],
            **{field:[r['fiscal_year'] for r in rows if r[field]] for field in ('strict_common_three_tag_complete','three_cash_families_common_issuance_complete','three_cash_families_allow_plan_issuance_proxy_complete')},
            'missing_by_family':{key:[r['fiscal_year'] for r in rows if r['cash_fact_groups'][key]['status']=='missing'] for key in TAGS},
            'original_release_label_presence_common_issuance_years':[r['fiscal_year'] for r in rows if r['original_report_label_families']['three_family_label_presence_common_issuance']],
            'original_release_label_presence_allow_plan_proxy_years':[r['fiscal_year'] for r in rows if r['original_report_label_families']['three_family_label_presence_allow_plan_proxy']]}
    for info in DEFINITIONS.values(): pin(info['path'])
    for p in ('work/monthly_cash_quality146_proposal.json','work/monthly_stock_comp149_proposal.json',
              'work/capex193_source_scope.json','work/after158_basic_shares_admission_policy_v2.json','work/after158_basic_shares_gaap_bridge.py'):
        pin(p)
    pin(SELF)
    return {'schema':'after254_cash_distribution_bounded_source_review_v1','created_utc':datetime.now(timezone.utc).isoformat(),
        'status':'SOURCE_LEADS_AND_GAPS_ONLY_NOT_ADMISSION','scope':'Exactly44 existing annual release events and four pinned SEC companyfacts caches; no later-accession fallback.',
        'source_admitted':False,'strict_pit':False,'strategy_registered':False,
        'activity':{'reports_inventoried':44,'companyfacts_caches':4,'financial_amounts_used_in_arithmetic':0,'new_features':0,'ranks':0,'orders':0,'returns':0,'downloads':0},
        'comparison_to_existing_definitions':DEFINITIONS,
        'independence':'Cash dividends leave the EPS share denominator unchanged. Repurchase/issuance prices and within-year timing break equivalence to weighted-average share changes. Noncash compensation/M&A/share conversions can change shares without matching cash issuance. This is algebraic non-equivalence, not evidence of predictive independence.',
        'coverage_summary':summary,
        'coverage_totals':{field:sum(bool(r[field]) for r in records) for field in ('strict_common_three_tag_complete','three_cash_families_common_issuance_complete','three_cash_families_allow_plan_issuance_proxy_complete')},
        'bounded_conclusion':'Not equivalent to the four reviewed old definitions, but not source-ready. Hold as a source-semantics backup; do not implement or register a factor from this inventory.',
        'company_findings':{
            'MSFT':'FY2015..2026 all12 selected annual10-K contexts have unique common repurchase/common cash dividend/common issuance USD tags; all12 original releases carry the three cashflow labels. Best bounded lead, not yet field-specific admission.',
            'AAPL':'FY2016..2021 six selected annual10-K contexts and original release labels have all three broad families. FY2022..2025 lack common issuance in both checked sources; no zero inference. Original dividend line includes dividend equivalents; only FY2017 also has the narrow common-dividend SEC tag.',
            'AMD':'No dividend payment tag in these10 selected reports; early common-repurchase gaps and changing common/employee-plan proceeds tags remain. FY2019 is outside the original44-report inventory, not newly filled.',
            'NVDA':'None of the12 selected earliest accessions provides the full common-issuance triple. FY2024..2026 has the narrower employee-plan proceeds proxy plus broad dividends and repurchases. FY2020/FY2023 original releases also show those narrower line families despite selected-accession tag gaps; do not backfill from later accessions. FY2022 repurchase tag is missing, not an established zero.'
        },'cache_provenance':origins,'cash_concept_glossary':glossary,'records':records,
        'implementation_conditions':[
            'A common-stock-only payout definition needs common cash dividends; PaymentsOfDividends can include preferred/noncontrolling distributions and original Apple row includes dividend equivalents.',
            'Plan or option proceeds are narrower than total common equity issuance. Do not splice tags or add overlapping concepts without full statement reconciliation.',
            'Tax withholding cash outflows are separate from common buybacks; SBC addbacks and noncash acquisition shares are not cash paid/raised.',
            'Every tag context must retain selected accession, filed date, full annual start/end, currency, and source pointer. Missing tag or omitted financing row is unknown, never inferred zero.',
            'The copied second-session close after later release/filing is a preexisting conservative research clock, not a newly admitted payout availability clock or certified historical vintage.',
            'Assets and capitalization denominators require independent same-currency field admission; no historical market-cap denominator or full-universe payout coverage was established.',
            'Sparse four-company scope and different issuance concepts require matched support and explicit coverage controls before any prospective experiment.'
        ],
        'limitations':[
            'Metadata completeness is not statement-level amount/annual-column reconciliation; neither strict common tags nor broader families certify the full economic net cash distribution.',
            'Cached SEC facts may omit custom tags and absent standard tags do not prove an absent payment. Earliest cached filing is not certified earliest original filing.',
            'AMD and NVDA lack identified matching origin/retrieval sidecars; AAPL/MSFT sidecars record retrospective retrieval.',
            'Original report label inventory is lexical and can include note/summary labels; no quantities were extracted or combined.',
            'No first-public, amendment history, historical security-master, split/ADR, or expanded universe admission was performed.',
            'Bounded non-equivalence to146/149/159/193 only; exhaustive novelty over the full research archive remains UNKNOWN.'
        ],'input_sha256':dict(sorted(PINS.items()))}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--write',action='store_true');ap.add_argument('--rows',action='store_true');args=ap.parse_args()
    out=build()
    if args.write:
        with (ROOT/OUT).open('x') as f: json.dump(out,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
        print(json.dumps({'path':OUT,'sha256':hashlib.sha256((ROOT/OUT).read_bytes()).hexdigest(),'input_pins':len(PINS)}))
    if args.rows:
        for r in out['records']:
            print(json.dumps({'event':r['event_id'],'filed':r['selected_filed_date'],'usable_at':r['inherited_later_source_usable_at'],
                'labels':r['original_report_label_inventory']['row_candidates']},ensure_ascii=False))
    else: print(json.dumps(out['coverage_summary'],indent=2))

if __name__=='__main__': main()
