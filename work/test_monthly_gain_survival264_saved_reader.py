"""Synthetic checks only for root saved-evidence selection; no replay inputs."""
import unittest
from copy import deepcopy
import ast
from pathlib import Path
import monthly_gain_survival264_saved_reader as reader


def saved_evidence():
    """Fabricated receipts only: no host, independent module or market input."""
    width=2
    def counts(n):
        h=max(n-1,0)*width
        c=dict(events=n,symbols_per_event=width,new_model_fits=0,predictive_model_fits=0,original_episode_calls=3,
            original_core_cells=h,required_cells=h,price_observations=253*h,gain_return_observations=252*h,gain_statistics=h,
            quality_ratio_evaluations=2*h,positive_gain_anchors=h,survival_comparisons=10*h,endpoint_comparisons=h,
            no_positive_gain_cells=0,nonpositive_original_cells=0,positive_raw_multiplications=2*h,
            available_events=max(n-1,0),endpoint_ranked_cells=h,survival_ranked_cells=h)
        c.update({k:0 for k in reader.ZERO_SOURCE_COUNTS});return c
    def proof(n,kind):
        c=counts(n);formal=max(n-1,0);rows=300+n if formal else 0
        result=dict.fromkeys(reader.WORK_FIELDS,0)
        result.update(passed=True,checked_events=n,producer_modules_imported=False,computes_returns=False)
        if kind=='original':
            events=[]
            for q in range(n):
                e=dict(query_index=q,date='2016-08-17' if q==0 else 'synthetic'+str(q),initial_partial=q==0,calm=False,u0_indices=[] if q==0 else [0,1])
                for key in ('original_raw','original_rank','frozen_core','rebuilt_core','core_abs_error','original_sigma63'):e[key]=None if q==0 else [1.,1.]
                events.append(e)
            result.update(original_events=events,u0_indices=[e['u0_indices'] for e in events],original_checked_cells=width*formal,
                original_source_cells=width*formal,original_source_price_observations=sum((301+q)*width for q in range(1,n)))
        elif kind=='incumbent':result['original_episode_calls']=1
        else:
            full=kind in ('source','full');selecting=kind in ('selection','full')
            # Keep the fixture arithmetic separate from the reader's work helper.
            result.update(full_source_slot_inspections=c['price_observations'] if full else 0,cached_slot_validations=0 if full else c['price_observations'],return_evaluations=c['gain_return_observations'],statistic_evaluations=c['gain_statistics'],ratio_evaluations=c['quality_ratio_evaluations'],rank_evaluations=2*c['available_events'] if selecting else 0,scalar_rank_evaluations=2*c['required_cells'] if selecting else 0,original_episode_calls=3 if selecting else 0)
            for k in ('positive_gain_anchors','survival_comparisons','endpoint_comparisons','positive_raw_multiplications'):result[k]=c[k]
            if kind in ('source','cached_source'):result.update(source_rebuilt=full,feature_counts=c)
            else:result['counts']=c
        if kind in ('original','source','full'):result.update(original_source_snapshot_sha256=reader.canonical_digest(['synthetic snapshot',n]),original_source_snapshot_price_observations=rows*width,original_source_snapshot_hashes=int(formal>0))
        return result
    def item(n,cached=False):
        c=counts(n);original=proof(122 if cached else n,'original');cert_hash=reader.canonical_digest(original)
        x=dict(prefix=n,counts=c,exact=True,gates=dict(incumbent=proof(n,'incumbent'),source=proof(n,'cached_source' if cached else 'source')))
        x['selection' if cached else 'full']=proof(n,'selection' if cached else 'full')
        if cached:x['original_source_certificate_sha256']=cert_hash
        else:
            x.update(original_source=original,original_source_anchor=dict(schema='monthly_gain_survival264_original_source_anchor_v1',checked_events=n,certificate_sha256=cert_hash,original_event_sha256=[reader.canonical_digest(e) for e in original['original_events']]),producer_original_source_work=dict(fullpast_rows=300+n if n>1 else 0,union_symbols=['A','B'] if n>1 else [],original_source_price_observations=width*(300+n) if n>1 else 0,original_source_cells=width*max(n-1,0)),reused_full=False,source_recomputed=True)
            for r in (x['gates']['source'],x['full']):r.update(original_source_proof_sha256=cert_hash,original_source_proof_events=n)
        x['consumed_original_source_sha256']=reader.canonical_digest(dict(schema='monthly_gain_survival264_original_source_prefix_v1',checked_events=n,original_events=original['original_events'][:n],u0_indices=original['u0_indices'][:n]));return x
    main=item(122);physical=[item(n) for n in (0,1,53,88)]+[dict(deepcopy(main),reused_full=True,source_recomputed=False)];cached=[item(n,True) for n in range(123)]
    e=dict(main=main,physical_prefixes=physical,cached_prefixes=cached,full_original_source_certificate_sha256=reader.canonical_digest(main['original_source']),full_source_certificate_sha256=reader.canonical_digest(main['gates']['source']))
    actual=[main]+physical[:-1];producer=dict.fromkeys(reader.WORK_FIELDS,0);independent=dict.fromkeys(reader.WORK_FIELDS,0)
    for it in actual+cached:
        n=it['prefix'];full='full' in it;p=proof(n,'full' if full else 'selection')
        if full:
            p['original_source_price_observations']=it['producer_original_source_work']['original_source_price_observations'];p['original_source_cells']=it['counts']['original_core_cells']
            p['original_source_snapshot_price_observations']=p['original_source_snapshot_hashes']=0
        for k in reader.WORK_FIELDS:producer[k]+=p[k]
        receipts=[it['gates']['incumbent'],it['gates']['source'],it['full' if full else 'selection']]+([it['original_source']] if full else [])
        for r in receipts:
            for k in reader.WORK_FIELDS:independent[k]+=r[k]
    total={k:producer[k]+independent[k] for k in reader.WORK_FIELDS}
    e['execution_accounting']=dict(producer=producer,independent=independent,total=total,physical_preparations=5,cached_decisions=123,original_source_gate_calls=5,original_core_checked_cells=sum(it['counts']['original_core_cells'] for it in actual),return_vector_steps_computed=total['return_evaluations'],new_model_fits=0,predictive_model_fits=0,physical_122_reused=True,source_and_cached_statistic_recomputation_disclosed=True)
    return e


class SavedSourceEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.evidence=saved_evidence()
    def check(self,e=None):return reader.source_evidence_review(self.evidence if e is None else e,width=2)
    def test_original_raw_full_snapshot_counts_are_separate(self):
        r=self.check();self.assertTrue(r['passed']);self.assertEqual(r['independent_numeric_snapshot_passes'],9)
        self.assertEqual(r['independent_source_snapshot_observations'],3*2*sum(300+n for n in (122,53,88)))
        self.assertEqual(r['cached_source_price_reads'],0);self.assertFalse(r['source_snapshot_sha_recomputed_by_reader'])
        self.assertGreater(r['independent_original_reconstruction_observations'],r['independent_source_snapshot_observations'])
    def test_copied_json_receipts_are_supported(self):
        import json
        self.check(json.loads(json.dumps(self.evidence)))
    def test_stable_host_synthetic_receipt_schema_compatible(self):
        # Cross-check the separately constructed synthetic host receipt format;
        # the saved reader itself does not import or delegate to the host.
        from test_monthly_gain_survival264_host import prepared_fixture
        import json
        evidence,*_=prepared_fixture()
        result=reader.source_evidence_review(json.loads(json.dumps(evidence)))
        self.assertTrue(result['passed']);self.assertEqual(result['independent_numeric_snapshot_passes'],12)
    def test_reject_original_raw_full_snapshot_mismatch(self):
        for label in ('original_source','source','full'):
            e=deepcopy(self.evidence);it=e['physical_prefixes'][2];p=it['gates']['source'] if label=='source' else it[label]
            p['original_source_snapshot_sha256']='f'*64
            with self.subTest(label=label),self.assertRaises(ValueError):self.check(e)
    def test_no_zero_or_bool_numeric_snapshot_count_on_formal_prefix(self):
        for field,value in (('original_source_snapshot_price_observations',0),('original_source_snapshot_hashes',0),('original_source_snapshot_hashes',True)):
            e=deepcopy(self.evidence);e['physical_prefixes'][2]['full'][field]=value
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):self.check(e)
    def test_n0_and_partial_have_no_numeric_snapshot_pass(self):
        for i in (0,1):
            self.assertEqual(self.evidence['physical_prefixes'][i]['full']['original_source_snapshot_hashes'],0)
            e=deepcopy(self.evidence);e['physical_prefixes'][i]['full']['original_source_snapshot_hashes']=1
            with self.assertRaises(ValueError):self.check(e)
    def test_cached_and_incumbent_cannot_claim_source_reads(self):
        for location in ('cached','incumbent'):
            e=deepcopy(self.evidence);p=e['cached_prefixes'][4]['selection'] if location=='cached' else e['main']['gates']['incumbent']
            p['original_source_snapshot_price_observations']=1
            with self.subTest(location=location),self.assertRaises(ValueError):self.check(e)
    def test_raw_full_do_not_repeat_pandas_reconstruction(self):
        for key in ('original_source_price_observations','original_source_cells'):
            e=deepcopy(self.evidence);e['physical_prefixes'][2]['full'][key]=1
            with self.assertRaises(ValueError):self.check(e)
    def test_physical_union_size_and_original_binding_checked(self):
        for mode in ('union','proof','prefix','missing'):
            e=deepcopy(self.evidence);it=e['physical_prefixes'][2]
            if mode=='union':it['producer_original_source_work']['union_symbols'].append('C')
            elif mode=='proof':it['full']['original_source_proof_sha256']='f'*64
            elif mode=='prefix':it['consumed_original_source_sha256']='f'*64
            else:it['full'].pop('original_source_snapshot_hashes')
            with self.subTest(mode=mode),self.assertRaises((ValueError,KeyError)):self.check(e)
    def test_full122_reuse_is_not_a_sixth_physical_execution(self):
        e=deepcopy(self.evidence);e['physical_prefixes'][-1]['reused_full']=False
        with self.assertRaises(ValueError):self.check(e)
        e=deepcopy(self.evidence);e['execution_accounting']['physical_preparations']=6
        with self.assertRaises(ValueError):self.check(e)
    def test_owner_totals_cannot_omit_snapshot_or_doublecount_reuse(self):
        for owner in ('producer','independent','total'):
            e=deepcopy(self.evidence);e['execution_accounting'][owner]['original_source_snapshot_price_observations']+=1
            with self.subTest(owner=owner),self.assertRaises(ValueError):self.check(e)
    def test_cached_original_certificate_or_count_tampering(self):
        for mode in ('certificate','counts','snapshot'):
            e=deepcopy(self.evidence);it=e['cached_prefixes'][53]
            if mode=='certificate':it['original_source_certificate_sha256']='f'*64
            elif mode=='counts':it['counts']['price_observations']-=1
            else:it['gates']['source']['original_source_snapshot_hashes']=1
            with self.subTest(mode=mode),self.assertRaises(ValueError):self.check(e)
    def test_reader_has_no_host_producer_or_independent_import(self):
        tree=ast.parse(Path(reader.__file__).read_text());names=[]
        for n in ast.walk(tree):
            if isinstance(n,ast.Import):names.extend(a.name for a in n.names)
            elif isinstance(n,ast.ImportFrom):names.append(n.module or '')
        self.assertFalse(any('264_host' in n or '264_kernel' in n or '264_independent' in n for n in names))

class SavedSelection(unittest.TestCase):
    def setUp(self):
        self.p=reader.load_parent()
        def metrics(c):
            m=dict(cagr=c,sharpe=1.,max_drawdown=-.2,daily_return_sd=.01,sharpe_status='defined',sharpe_reason=None)
            return {p:{g:dict(m) for g in ('training','early','recent')} for p in reader.PATHS[:2]}
        self.m={a:metrics(.2 if a!=reader.ARMS[2] else .21) for a in reader.ARMS}
        self.o={a:[dict(symbols=['A','B','C']) for _ in range(122)] for a in reader.ARMS}
        for i in (2,4,60):self.o[reader.ARMS[2]][i]=dict(symbols=['A','B','D'])
    def pick(self):return reader.selection(self.p,self.m,self.o)
    def test_candidate_only_when_both_references_pass(self):
        self.assertTrue(self.pick()['admitted']);self.assertEqual(self.pick()['primary'],reader.ARMS[2])
        self.m[reader.ARMS[1]]['same_close_zero']['training']['cagr']=.3
        self.assertFalse(self.pick()['admitted']);self.assertEqual(self.pick()['primary'],reader.ARMS[0])
    def test_open_zero_must_pass(self):
        self.m[reader.ARMS[2]]['next_open_zero']['training']['cagr']=.19
        self.assertFalse(self.pick()['admitted'])
    def test_risk_guard(self):
        self.m[reader.ARMS[2]]['same_close_zero']['training']['sharpe']=.949
        self.assertFalse(self.pick()['admitted'])
    def test_recent_era_required(self):
        self.m[reader.ARMS[2]]['same_close_zero']['recent']['cagr']=.174
        self.assertFalse(self.pick()['admitted'])
    def test_permutations_do_not_count(self):
        for i in (2,4,60):self.o[reader.ARMS[2]][i]=dict(symbols=['C','A','B'])
        self.assertFalse(self.pick()['admitted'])
    def test_later_only_cannot_rescue(self):
        for i in (2,4,60):self.o[reader.ARMS[2]][i]=dict(symbols=['A','B','C'])
        for i in range(88,122):self.o[reader.ARMS[2]][i]=dict(symbols=['A','B','D'])
        self.assertFalse(self.pick()['admitted'])
    def test_undefined_candidate_sharpe_is_inadmissible(self):
        self.m[reader.ARMS[2]]['same_close_zero']['training'].update(sharpe=None,daily_return_sd=0.,sharpe_status='undefined_sharpe',sharpe_reason='zero_daily_return_sd')
        self.assertFalse(self.pick()['admitted'])
    def test_cost_diagnostic_not_selection_input(self):
        self.m[reader.ARMS[2]]['next_open_cost']={}
        with self.assertRaises(ValueError):self.pick()

if __name__=='__main__':unittest.main(verbosity=2)
