"""Synthetic bytes and invented reports only; never runs the actual census."""
import copy
from datetime import date
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import roa_field_scope270_host_v3 as h
import prepare_roa_field_scope270_policy_v3 as builder
import test_roa_field_scope270_host as legacy


def package(root):
    prior, _, docs, oldev = legacy.package(root)
    pins = dict(oldev.pins); refs = copy.deepcopy(prior['inputs'])
    def save(role, value, path=None):
        refs[role] = legacy.put(root, path or ('new/' + role + '.json'), value, pins)
        docs[role] = value
        return refs[role]
    def put(path, value): return legacy.put(root, path, value, pins)
    # Seven declaring documents, each with the same 82 intentionally absent
    # historical binaries. None may be read as an active dependency.
    historical = {'retired/runtime/pypdf/f%d.py' % i:'a'*64 for i in range(55)}
    historical.update({'retired/runtime/other%d.bin' % i:'b'*64 for i in range(27)})
    declarations = []
    d = copy.deepcopy(docs['legacy_review']); d['input_hashes'] = historical
    save('legacy_review', d, refs['legacy_review']['path']); declarations.append(refs['legacy_review'])
    for i in range(6): declarations.append(put('new/declaration%d.json' % i, {'pdf_runtime_hashes':historical}))
    edges = []
    for ref in declarations:
        group = 'input_hashes' if ref == refs['legacy_review'] else 'pdf_runtime_hashes'
        for path, sha in historical.items():
            edges.append(dict(declaring_document=ref, declaration_map=group, declaration_key=path,
                declaration_json_pointer=h.pointer(group,path), runtime_path=path, expected_sha256=sha,
                disposition='historical_anchor_only', replacement_chain_id='new_pdf_render_and_two_reviews'))
    save('runtime_gap', dict(source_only=True, old_runtime_verified=False))
    old_plan_map, entries = {}, []
    for i in range(55):
        name = 'work/capex193_pdf_runtime/site-packages/pypdf/f%d.py' % i
        active = put(name, ('# isolated synthetic %d' % i).encode())
        key = 'retired/runtime/pypdf/f%d.py' % i; old_plan_map[key] = active['sha256']
        entries.append(dict(declaration_key=key, expected_sha256=active['sha256'], active_copy_path=name))
    # Unlisted content in this byte anchor MUST NOT be recursively expanded.
    old_plan_map['retired/unrelated/lib.bin'] = 'c'*64
    old_plan = put('new/old_plan.json', {'pdf_runtime_hashes':old_plan_map})
    runtime_ref = put('new/active_runtime.bin', b'synthetic active PDF runtime')
    pillow = put('new/active_png.bin', b'synthetic active PNG encoder')
    runtime = dict(pdf_runtime_hashes={runtime_ref['path']:runtime_ref['sha256']},
        png_encoder_sha256={pillow['path']:pillow['sha256']},
        executable_path=str(Path(sys.executable).resolve()), executable_sha256=h.digest(Path(sys.executable).read_bytes()))
    pins[runtime['executable_path']] = runtime['executable_sha256']
    base = 'new/render'
    save('pdf_runtime', runtime, base+'/RUNTIME.json')
    auth = dict(execution_authorized=True, financial_admission_authorized=False,
        input_sha256={runtime_ref['path']:runtime_ref['sha256'],pillow['path']:pillow['sha256']},
        events=[dict(event_id=e['event_id'],source=copy.deepcopy(e['source']),pages_1based=[2,3])
                for e in docs['dataset']['events'] if e['fields']['current']['net_income']['locator']['format']=='pdf'])
    save('pdf_authorized', auth)
    rendered, visual, primary, visual_pages = [], [], [], []
    output = {'RUNTIME.json':refs['pdf_runtime']['sha256']}
    events = [e for e in docs['dataset']['events'] if e['fields']['current']['net_income']['locator']['format']=='pdf']
    for i, event in enumerate(events):
        fields, cells, cell_shas, primary_fields = {}, [], {}, {}
        pages = {}
        for page, label in ((2,'Total assets'),(3,'Net income')):
            image = put(base+'/event_%02d_page_%d.png'%(i,page), b'not an actual PNG: synthetic '+str(i).encode())
            boxes=[dict(index_0based=j,text='120120'[j],bbox_pdf_units=[float(j if j<3 else j+7),0.0,float(j+1 if j<3 else j+8),1.0]) for j in range(6)]
            text = dict(event_id=event['event_id'], page_1based=page,
                        pypdf_layout_text=label+' 120 120', pdfium_text=label+' 120 120',
                        image=dict(path=Path(image['path']).name,sha256=image['sha256']),nonwhitespace_character_boxes=boxes)
            tr = put(base+'/event_%02d_page_%d.json'%(i,page), text)
            pages[page] = (image,tr)
            output[Path(image['path']).name]=image['sha256']; output[Path(tr['path']).name]=tr['sha256']
            ends = [date.fromisoformat(event['periods'][s]['period_end']) for s in ('current','comparison')]
            visual_pages.append(dict(event_id=event['event_id'],page_1based=page,image=image,text_geometry=tr,
                pixel_review_performed=True,required_headers_legible=True,required_value_rows_legible=True,
                required_value_or_date_contradiction_found=False,financial_admission=False,
                observed_header_dates_left_to_right=[x.strftime('%B ')+str(x.day)+x.strftime(', %Y') for x in ends],
                observed_unit_literal='In millions',required_currency_glyph_pixels='$',
                annual_header_directly_observed='Twelve Months Ended' if page==3 else None,
                dual_parser_header_lines={k:[dict(line_1based=1,literal_text=text[k])] for k in ('pypdf_layout_text','pdfium_text')}))
        for side, field in h.scope.REQUIRED:
            cell=event['fields'][side][field]; loc=cell['locator']; key=side+'.'+field
            sha=h.scope.canonical_sha256(cell); cell_shas[key]=sha
            page=loc['page_1based']; image,tr=pages[page]
            fields[key]=dict(original_cell_sha256=sha,original_cell=copy.deepcopy(cell),original_locator=copy.deepcopy(loc),
                render_evidence=dict(image=dict(path=Path(image['path']).name,sha256=image['sha256']),
                                     text_geometry=dict(path=Path(tr['path']).name,sha256=tr['sha256'])))
            row=dict(line_1based=1,literal_text=loc['row_label']+' 120 120',numeric_literals_in_original_row_order=['120','120'])
            primary_fields[key]=dict(original_cell_sha256=sha,observed_value_decimal=cell['value_decimal'],
                displayed_amount=cell['displayed_amount'],scale=cell['scale'],currency_literal='$',currency_iso=None,
                page_1based=page,financial_column_index_0based=loc['financial_column_index_0based'],
                observed_endpoint=loc['period_end'],instant_or_duration='annual_duration' if field=='net_income' else 'instant',
                stored_period_preserved=copy.deepcopy(cell['period']),
                header_endpoints=[event['periods'][s]['period_end'] for s in ('current','comparison')],
                independently_reviewed=False,field_admitted=False,currency_iso_certified=False,
                amount_matches_original=True,physical_column_matches_original=True,units_match_original=True,
                endpoint_matches_original=True,annual_or_instant_semantics_match=True,complete_original_cell_equal_to_dataset=True,
                image=fields[key]['render_evidence']['image'],text_geometry=fields[key]['render_evidence']['text_geometry'],
                raw_pdfium_row=row['literal_text'])
            columns=[dict(text='120',char_indices=[0,1,2],bbox=[0.0,0.0,3.0,1.0]),
                     dict(text='120',char_indices=[3,4,5],bbox=[10.0,0.0,13.0,1.0])]
            primary_fields[key].update(geometry_columns=columns,selected_geometry=copy.deepcopy(columns[loc['financial_column_index_0based']]))
            cells.append(dict(field_path=key,original_cell_sha256=sha,matched=True,observed_from_pixels=True,
                physical_row_and_column_visually_matched=True,dual_parser_amount_tokens_match_visual=True,
                consolidated_scope_observed=True,value_decimal=cell['value_decimal'],scale=cell['scale'],
                observed_amount_literal=cell['displayed_amount'],observed_unit_literal='In millions',currency_literal='$',
                iso_currency=None,iso_currency_certified=False,financial_column_index_0based=loc['financial_column_index_0based'],
                page_1based=page,period_end=loc['period_end'],observed_row_label=loc['row_label'],
                period_kind='annual_duration' if field=='net_income' else 'instant',
                observed_duration_header='Twelve Months Ended' if field=='net_income' else None,
                net_income_scope='consolidated indirect cash flow start' if field=='net_income' else None,
                exact_period_start_directly_printed_and_reverified=False,field_admitted=False,chain_admitted=False,
                new_image=image,new_text_geometry=tr,pypdf_row=copy.deepcopy(row),pdfium_row=copy.deepcopy(row)))
        esha=h.scope.canonical_sha256(event)
        rendered.append(dict(event_id=event['event_id'],original_event_sha256=esha,source=copy.deepcopy(event['source']),fields=fields))
        visual.append(dict(event_id=event['event_id'],event_sha256=esha,source=copy.deepcopy(event['source']),cell_sha256=cell_shas,cells=cells))
        primary.append(dict(event_id=event['event_id'],original_event_sha256=esha,source=copy.deepcopy(event['source']),fields=primary_fields))
    evidence=dict(schema='roa_field_scope270_pdf_render_evidence_v1',source_only=True,synthetic_backend=False,
        proposal_sha256=refs['pdf_authorized']['sha256'],counts=dict(actual_assess_calls=0,events=10,html_events_reextracted=0,
            pages_rendered=20,required_fields=30),chain_admitted=False,field_admitted=False,ratios_computed=False,
        iso_currency_certified=False,old_runtime_verified=False,events=rendered,strategy_admitted=False,strict_pit=False,first_public_version_verified=False,
        runtime_manifest=dict(path='RUNTIME.json',sha256=refs['pdf_runtime']['sha256']))
    save('pdf_evidence',evidence,base+'/EVIDENCE.json'); output['EVIDENCE.json']=refs['pdf_evidence']['sha256']
    save('pdf_manifest',dict(schema='roa_field_scope270_pdf_render_manifest_v1',render_complete=True,
        synthetic_backend=False,financial_admission=False,old_runtime_verified=False,output_sha256=output,
        proposal_sha256=refs['pdf_authorized']['sha256']),base+'/MANIFEST.json')
    terminal=dict(session_id=123,exit_code=0,chunk_id='invented-terminal')
    save('pdf_terminal',dict(manifest=refs['pdf_manifest'],evidence=refs['pdf_evidence'],terminal=terminal,
        log=put('new/render.log',b'fabricated metadata fixture'),render_complete=True,financial_admission=False))
    save('pdf_primary',dict(primary_checks_passed=True,review_role='producer_side_self_review',independently_reviewed=False,
        financial_admission=False,events=primary,evidence_chain=dict(render_manifest=refs['pdf_manifest'],render_evidence=refs['pdf_evidence'])))
    save('pdf_independent_visual',dict(schema='roa_field_scope270_pdf_rerender_visual_review_v1',passed=True,source_only=True,
        manifest=refs['pdf_manifest'],evidence=refs['pdf_evidence'],owner_render_terminal_receipt=refs['pdf_terminal'],
        owner_render_terminal=terminal,unresolved_field_fact_discrepancies=[],financial_admission=False,ratios_computed=False,
        chains_admitted=False,iso_currency_certified=False,strict_pit=False,
        independence=dict(pixel_review_performed_by_reviewer=True,other_agent_primary_review_used=False,
            parser_text_used_as_substitute_for_viewing=False,producer_passed_flags_used_as_visual_evidence=False),
        counts=dict(events=10,images_actually_viewed=20,required_fields=30,visually_matched_fields=30,canonical_full_cells_checked=30),events=visual,pages=visual_pages))
    inventory_refs=list(refs[k] for k in h.old.ROLES)+declarations[1:]+[
        docs['legacy_policy']['inputs']['tests'],docs['legacy_policy']['independent_code_review']]
    inventory=[]
    for ref in inventory_refs:
        value=h.old.json.loads((root/ref['path']).read_bytes())
        inventory.append(dict(document=ref,expand_top_level_maps=[k for k in h.old.HASH_GROUPS if k in value]))
    design=dict(schema='roa_field_scope270_source_v3_design_v1',source_only=True,execution_authorized=False,financial_admission=False,
        role_count=29,role_refs=copy.deepcopy(refs),role_map={k:r['path'] for k,r in refs.items()},
        historical_runtime=dict(edges=edges),declared_document_inventory=inventory,
        replacement_chains=dict(new_pdf_render_and_two_reviews={k:refs[k] for k in h.NEW_ROLES if k!='runtime_gap'}),
        new_runtime_source_copy_table=dict(document=old_plan,map='pdf_runtime_hashes',entries=entries))
    dref=put('new/DESIGN.json',design)
    codes={name:put(name,(h.ROOT/name).read_bytes())['sha256'] for name in h.CODE_FILES}
    testref=put('new/TESTS.json',dict(passed=True,synthetic_only=True,tests=1,failures=0,errors=0,
        terminal=dict(exit_code=0,chunk_id='fabricated-synthetic-terminal'),code_sha256=codes))
    reviewref=put('new/REVIEW.json',dict(passed=True,unresolved_blockers=[],code_sha256=codes))
    policy=builder.collect(root,dref,testref,reviewref)
    pref=put('new/POLICY.json',policy)
    ev=h.Evidence(root,{**policy['input_sha256'],pref['path']:pref['sha256']},edges)
    authenticated=h.policy_check(policy,ev)
    return policy,pref,authenticated,ev


class EdgeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.oldpath='retired/libpdfium.bin';self.sha='a'*64
        self.pins={};self.ref=legacy.put(self.root,'doc.json',{'input_hashes':{self.oldpath:self.sha}},self.pins)
        self.edge=dict(declaring_document=self.ref,declaration_map='input_hashes',declaration_key=self.oldpath,
            declaration_json_pointer=h.pointer('input_hashes',self.oldpath),runtime_path=self.oldpath,expected_sha256=self.sha,
            disposition='historical_anchor_only',replacement_chain_id='new_pdf_render_and_two_reviews')
    def tearDown(self):self.temp.cleanup()
    def test_exact_missing_old_binary_edge_retained(self):
        e=h.Evidence(self.root,self.pins,[self.edge]);e.declared(e.read(self.ref));self.assertEqual(e.complete_history(),[self.edge])
        self.assertNotIn(self.oldpath,e.verified)
    def test_duplicate_edge(self):
        with self.assertRaisesRegex(ValueError,'Duplicate'):h.Evidence(self.root,self.pins,[self.edge,self.edge])
    def test_edge_pointer_key_disposition_and_chain(self):
        for key,value in [('declaration_json_pointer','/wrong'),('declaration_key','wrong'),('disposition','verified'),('replacement_chain_id','other')]:
            with self.subTest(key=key):
                x=copy.deepcopy(self.edge);x[key]=value
                with self.assertRaises(ValueError):h.Evidence(self.root,self.pins,[x])
    def test_wrong_document_sha_path_map_target_expected_never_exempts(self):
        for key in ('docsha','docpath','map','target','sha'):
            x=copy.deepcopy(self.edge)
            if key=='docsha':x['declaring_document']['sha256']='b'*64
            elif key=='docpath':x['declaring_document']['path']='other.json'
            elif key=='map':x.update(declaration_map='source_hashes',declaration_json_pointer=h.pointer('source_hashes',self.oldpath))
            elif key=='target':x.update(runtime_path='other.bin',declaration_key='other.bin',declaration_json_pointer=h.pointer('input_hashes','other.bin'))
            else:x['expected_sha256']='b'*64
            e=h.Evidence(self.root,self.pins,[x])
            with self.subTest(key=key),self.assertRaises(ValueError):e.declared(e.read(self.ref))
    def test_unused_allowlist_fails(self):
        with self.assertRaisesRegex(ValueError,'Missing'):h.Evidence(self.root,self.pins,[self.edge]).complete_history()
    def test_same_target_from_other_document_is_active(self):
        ref=legacy.put(self.root,'other.json',{'input_hashes':{self.oldpath:self.sha}},self.pins)
        e=h.Evidence(self.root,self.pins,[self.edge])
        with self.assertRaisesRegex(ValueError,'omitted'):e.declared(e.read(ref))
    def test_other_document_current_hash_is_verified_not_globally_skipped(self):
        current=legacy.put(self.root,self.oldpath,b'current different bytes',self.pins)
        ref=legacy.put(self.root,'other.json',{'input_hashes':{self.oldpath:current['sha256']}},self.pins)
        e=h.Evidence(self.root,self.pins,[self.edge]);e.declared(e.read(ref))
        self.assertEqual(e.verified[self.oldpath],current['sha256']);self.assertFalse(e.historical_seen)
    def test_unissued_or_mutated_document_rejected(self):
        e=h.Evidence(self.root,self.pins,[self.edge]);d=e.read(self.ref)
        with self.assertRaises(ValueError):e.declared(copy.deepcopy(d))
        d['x']=1
        with self.assertRaises(ValueError):e.declared(d)
    def test_nonruntime_dependency_not_ignored(self):
        ref=legacy.put(self.root,'active.json',{'input_hashes':{'financial.bin':'e'*64}},self.pins)
        e=h.Evidence(self.root,self.pins,[self.edge])
        with self.assertRaisesRegex(ValueError,'omitted'):e.declared(e.read(ref))


class PackageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name).resolve()
        cls.p,cls.ref,cls.docs,cls.ev=package(cls.root)
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def test_full_synthetic44_preserves_original_and_routes_pdf10(self):
        with patch.object(h.scope,'assess',wraps=h.scope.assess) as assess:
            r=h.run(self.root,self.ref['path'],self.ref['sha256'])
        self.assertEqual(assess.call_count,44);self.assertTrue(r['passed'])
        self.assertEqual((r['admitted_events'],r['held_events']),(34,10)) # fixture result, not implementation constraint
        result=h.old.json.loads((self.root/h.OUT/'DATASET.json').read_bytes())
        for row,old in zip(result['events'],self.docs['dataset']['events']):
            self.assertEqual({k:v for k,v in row.items() if not k.startswith('roa_field_scope270_v3')},old)
            if old['fields']['current']['net_income']['locator']['format']=='pdf':
                routes=row['roa_field_scope270_v3_evidence']['mapping']['field_mapping']
                self.assertEqual({x['route'] for x in routes},{'pdf_independent_visual'})
        with self.assertRaisesRegex(ValueError,'Write-once'):h.run(self.root,self.ref['path'],self.ref['sha256'])
    def test_package_gate_and_history_complete(self):
        h.pdf_package(self.p,self.docs,self.ev);self.assertEqual(len(self.ev.complete_history()),574)
    def test_pdf_field_tamper(self):
        event=self.docs['dataset']['events'][0]
        changes=dict(value_decimal='999',scale='1',iso_currency='USD',currency_literal='EUR',period_end='2000-01-01',
            page_1based=4,financial_column_index_0based=1,original_cell_sha256='f'*64,matched=False,
            observed_from_pixels=False,consolidated_scope_observed=False,field_admitted=True,
            exact_period_start_directly_printed_and_reverified=True)
        for key,value in changes.items():
            docs=copy.deepcopy(self.docs);docs['pdf_independent_visual']['events'][0]['cells'][0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):h.pdf_cell(event,'current','net_income',docs,self.p['inputs'],self.ev)
    def test_pdf_actual_row_token_and_image_binding(self):
        event=self.docs['dataset']['events'][0]
        for kind in ('row','tokens','image'):
            docs=copy.deepcopy(self.docs);c=docs['pdf_independent_visual']['events'][0]['cells'][0]
            if kind=='row':c['pypdf_row']['literal_text']='different 120 120'
            elif kind=='tokens':c['pdfium_row']['numeric_literals_in_original_row_order']=['121','120']
            else:c['new_image']['sha256']='a'*64
            with self.subTest(kind=kind),self.assertRaises(ValueError):h.pdf_cell(event,'current','net_income',docs,self.p['inputs'],self.ev)
    def test_new_pdf_mandatory_even_three_review(self):
        docs=copy.deepcopy(self.docs);docs['pdf_independent_visual']['events'].pop()
        event=docs['dataset']['events'][31]
        new={r['event_id']:r for r in docs['three_review']['events']}
        with self.assertRaises(ValueError):h.field_review(event,docs,self.p['inputs'],new,self.ev)
    def test_original31_regression_reported_no_forced34(self):
        events,sessions,new=h.old.verify_package(self.p,{k:self.docs[k] for k in h.old.ROLES},self.ev)
        original=h.scope.assess
        def held(event,*args):
            r=original(event,*args)
            if event['event_id']=='SYNTHETIC:0':r['admitted']=False;r['reasons']=['synthetic regression']
            return r
        with patch.object(h.scope,'assess',side_effect=held):
            _,results,reg=h.assess_census(events,sessions,new,self.docs,self.p['inputs'],self.ev)
        self.assertEqual(len(reg),1);self.assertEqual(sum(r['admitted'] for r in results),33)
    def test_unknown_hold_is_not_cleared(self):
        e=self.docs['dataset']['events'][34]
        review,mapping=h.field_review(e,self.docs,self.p['inputs'],{},self.ev)
        self.assertIsNone(review);self.assertFalse(mapping['review_available'])
    def test_active_tamper_rejected_before_assess(self):
        for name in ('new/active_runtime.bin','new/active_png.bin',self.docs['dataset']['events'][12]['source']['path']):
            path=self.root/name;raw=path.read_bytes();path.write_bytes(raw+b'changed')
            try:
                with self.subTest(name=name),self.assertRaisesRegex(ValueError,'SHA mismatch'):
                    h.policy_check(self.p,h.Evidence(self.root,self.p['input_sha256'],self.p['historical_dependency_edges']))
            finally:path.write_bytes(raw)
    def test_omitted_or_duplicate_page_member_rejects(self):
        docs=dict(self.docs);docs['pdf_manifest']=copy.deepcopy(docs['pdf_manifest']);docs['pdf_manifest']['output_sha256'].pop('event_00_page_2.png')
        with self.assertRaisesRegex(ValueError,'42 render'):h.pdf_package(self.p,docs,self.ev)
    def test_authorization_and_inventory_required(self):
        for change in ('auth','roles','codes'):
            p=copy.deepcopy(self.p)
            if change=='auth':p['execution_authorized']=False
            elif change=='roles':p['inputs'].pop('pdf_independent_visual')
            else:p['code_sha256'].pop(h.CODE_FILES[-1])
            with self.subTest(change=change),self.assertRaises(ValueError):h.policy_check(p,self.ev)
    def test_input_mutation_or_wrong_original_pdf_source_rejects(self):
        event=copy.deepcopy(self.docs['dataset']['events'][0]);event['source']['url']='https://different.example/'
        with self.assertRaises(ValueError):h.pdf_cell(event,'current','net_income',self.docs,self.p['inputs'],self.ev)
    def test_old_plan_unlisted_runtime_not_expanded(self):
        self.assertNotIn('retired/unrelated/lib.bin',self.p['input_sha256'])
        self.assertEqual(len([k for k in self.p['input_sha256'] if '/site-packages/pypdf/' in k]),55)
    def test_primary_field_facts_not_only_passed(self):
        for field,value in [('observed_value_decimal','999'),('currency_iso','USD'),('field_admitted',True),('stored_period_preserved',{})]:
            docs=copy.deepcopy(self.docs);docs['pdf_primary']['events'][0]['fields']['current.net_income'][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                h.pdf_cell(docs['dataset']['events'][0],'current','net_income',docs,self.p['inputs'],self.ev)
    def test_primary_geometry_indices_text_bbox_and_selected_column(self):
        for mode in ('index','duplicate','text','bbox','selected','order'):
            docs=copy.deepcopy(self.docs);pc=docs['pdf_primary']['events'][0]['fields']['current.net_income']
            if mode=='index':pc['geometry_columns'][0]['char_indices'][0]=999
            elif mode=='duplicate':pc['geometry_columns'][0]['char_indices']=[0,0,2]
            elif mode=='text':pc['geometry_columns'][0]['text']='121'
            elif mode=='bbox':pc['geometry_columns'][0]['bbox'][0]=1.0
            elif mode=='selected':pc['selected_geometry']=copy.deepcopy(pc['geometry_columns'][1])
            else:pc['geometry_columns'].reverse()
            with self.subTest(mode=mode),self.assertRaises(ValueError):h.pdf_cell(docs['dataset']['events'][0],'current','net_income',docs,self.p['inputs'],self.ev)
    def test_late_receipt_semantic_failure_precedes_all_assessments(self):
        events,sessions,new=h.old.verify_package(self.p,{k:self.docs[k] for k in h.old.ROLES},self.ev)
        events=copy.deepcopy(events);last=list(events)[-1];events[last]['source']['url']='https://wrong.example/'
        with patch.object(h.scope,'assess',side_effect=AssertionError('must not assess')) as spy:
            with self.assertRaisesRegex(ValueError,'Receipt URL'):h.assess_census(events,sessions,new,self.docs,self.p['inputs'],self.ev)
        self.assertEqual(spy.call_count,0)
    def test_late_html_mapping_failure_precedes_all_assessments(self):
        events,sessions,new=h.old.verify_package(self.p,{k:self.docs[k] for k in h.old.ROLES},self.ev)
        docs=copy.deepcopy(self.docs)
        row=h.one(docs['html_review']['events'],event_id='SYNTHETIC:30');row['cells'][0]['independent_source']['value_decimal']='999'
        with patch.object(h.scope,'assess',side_effect=AssertionError('must not assess')) as spy:
            with self.assertRaises(ValueError):h.assess_census(events,sessions,new,docs,self.p['inputs'],self.ev)
        self.assertEqual(spy.call_count,0)
    def test_post_assessment_active_mutation_prevents_publish(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();p,ref,docs,ev=package(root)
            original=h.scope.assess;count=[0]
            def mutate(*args):
                result=original(*args);count[0]+=1
                if count[0]==44:(root/'new/active_runtime.bin').write_bytes(b'changed after assessment')
                return result
            with patch.object(h.scope,'assess',side_effect=mutate):
                with self.assertRaisesRegex(ValueError,'SHA mismatch'):h.run(root,ref['path'],ref['sha256'])
            self.assertEqual(count[0],44);self.assertFalse((root/h.OUT).exists())
    def test_extra_unknown_dependency_requires_active_pin(self):
        doc={'input_hashes':{'retired/unlisted_runtime.bin':'a'*64}}
        pins=dict(self.p['input_sha256']);ref=legacy.put(self.root,'new/extra.json',doc,pins)
        ev=h.Evidence(self.root,pins,self.p['historical_dependency_edges'])
        with self.assertRaisesRegex(ValueError,'omitted'):ev.declared(ev.read(ref))
    def test_copy_table_never_allows_missing_changed_or_duplicate_copy(self):
        original=self.ev.read(self.p['design'])
        for mode in ('missing','duplicate','expected','path'):
            design=copy.deepcopy(original);rows=design['new_runtime_source_copy_table']['entries']
            if mode=='missing':rows.pop()
            elif mode=='duplicate':rows[-1]=copy.deepcopy(rows[0])
            elif mode=='expected':rows[0]['expected_sha256']='e'*64
            else:rows[0]['active_copy_path']='elsewhere.py'
            pins=dict(self.p['input_sha256']);ref=legacy.put(self.root,'new/design-'+mode+'.json',design,pins)
            ev=h.Evidence(self.root,pins,self.p['historical_dependency_edges'])
            with self.subTest(mode=mode),self.assertRaises(ValueError):h.design_check(self.p,ev.read(ref),ev)


if __name__=='__main__':unittest.main(verbosity=2)
