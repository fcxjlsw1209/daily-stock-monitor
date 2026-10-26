"""Independent synthetic package corruption tests; no actual files are read."""
from copy import deepcopy
from datetime import date,timedelta
import hashlib
import unittest
from pathlib import Path
from unittest.mock import patch
import after156_association_package_review_v2 as r


def fixture():
    sessions=[(date(2000,1,1)+timedelta(days=i)).isoformat() for i in range(700)]
    names=['S%02d'%i for i in range(32)]+['ETF%02d'%i for i in range(17)]
    meta=dict(dates=sessions[550:672],symbols=[names[:] for _ in range(122)],etfs=names[32:])
    original=[];stocks=[];etfs=[];rows=[];delayed=dict(rows=[],snapshots={})
    for i,day in enumerate(meta['dates']):
        t=550+i
        for j,s in enumerate(names):
            stock=j<32;cutoff=946684800000000000+(550+i)*86400*10**9
            prior=dict(signal_index=i,candidate_index=j,symbol=s,decision_date=day,decision_cutoff=day+'T21:00:00+00:00',decision_cutoff_ns=cutoff)
            original.append(prior)
            origin=dict(signal_index=i,candidate_index=j,symbol=s,signal_date=day,research_eligible=stock,reasons=[] if stock else ['ETF_AUXILIARY_NEUTRAL'])
            row=dict(prior,research_eligible=stock,reasons=origin['reasons'][:],issuer_id=None,partition=None,event_ids=[],context_original_event_ids=[],source_refs=[],
                source_review_ref=dict(path=r.REVIEW,sha256=r.REVIEW_SHA,row_kind='stock' if stock else 'etf',signal_index=i,candidate_index=j),
                source_vintage='retrospectively_retrieved_with_explicit_reconstruction_assumptions',exact_acceptance_certified=False,signal_session_index=t)
            if stock:
                source='source/'+s;issuer=str(j+1).zfill(10);ids=[];bundles=[];labels=['N']*210;owners=[None]*210
                catalog=delayed['snapshots'].setdefault(source,dict(cik=issuer,events=[]))
                for n,a in enumerate([t-230,t-100]):
                    eid=s+'_'+str(i)+'_'+str(n);ids.append(eid)
                    catalog['events'].append(dict(event_id=eid,filingDate=sessions[a-2],form='10-Q'))
                    for d in range(a,a+21):labels[d-(t-251)]='B' if d<a+2 else 'E';owners[d-(t-251)]=a
                    bundles.append(dict(anchor=a,event_ids=[eid],forms=['10-Q'],amendment=False,owned_E_days=list(range(a+2,a+21)),full19=True,left_clipped=False,right_clipped=False,hypothetical_anchor_date=sessions[a]))
                partition=dict(first_day=t-251,last_day=t-42,labels=''.join(labels),owner_anchor_indices=owners,bundles=bundles,
                    counts=dict(E=38,B=4,N=168),full19_original_bundles=2,hypothetical_partition_pass=True,failure_reasons=[])
                d=dict(signal_index=i,candidate_index=j,symbol=s,partition=partition)
                ref=dict(path=r.DELAYED,sha256='d'*64,row_index=len(delayed['rows']),labels_sha256=hashlib.sha256(partition['labels'].encode()).hexdigest())
                delayed['rows'].append(d)
                origin.update(signal_session_index=t,decision_cutoff_ns=cutoff,partition_ref=ref,selected_snapshot=source,event_ids=ids,context_original_event_ids=[],
                    expected_ciks=[issuer],assumption_category='SYNTHETIC_CONTINUITY',checks=dict(all=True),source_evidence_refs=[dict(occurrences=[dict(source_path=source,source_sha256='s'*64,receipt_path=source+'.receipt',receipt_sha256='e'*64)])])
                refs={r.REVIEW:r.REVIEW_SHA,r.DELAYED:'d'*64,source:'s'*64,source+'.receipt':'e'*64}
                row.update(partition=deepcopy(partition),issuer_id=issuer,event_ids=ids[:],selected_snapshot=source,assumption_category=origin['assumption_category'],
                    source_gate_checks=dict(all=True),source_refs=[dict(path=p,sha256=h) for p,h in sorted(refs.items())])
                stocks.append(origin)
            else:etfs.append(origin)
            rows.append(row)
    gate=dict(schema_version=1,original_candidate_rows=5978,original_stock_slots=3904,clock_policy_id=r.CLOCK,exact_acceptance_certified=False,
        coverage_floor_passed=True,checks=dict.fromkeys(r.FLOORS,True),eras={})
    review=dict(passed=True,research_eligibility_verified=True,actual_grid_verified=True,exact_acceptance_certified=False,coverage_floor_checks=gate['checks'].copy(),rows=stocks,preserved_etf_slots=etfs,coverage={})
    for name,lo,hi in r.ERAS:
        gate['eras'][name]=dict(stock_slots=(hi-lo)*32,usable_slots=(hi-lo)*32,fraction=1.,distinct_issuers=32,
            months_with_at_least8_issuers=hi-lo,minimum_issuers_per_month=32)
        review['coverage'][name]=dict(original_stock_slots=(hi-lo)*32,research_eligible_slots=(hi-lo)*32,fraction=1.,distinct_issuers=32,months_with_at_least8_issuers=hi-lo)
    return meta,original,review,delayed,dict(sessions=sessions),rows,gate


class PackageReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.base=fixture()
    def setUp(self):self.data=deepcopy(self.base)
    def fails(self):
        with self.assertRaises((ValueError,KeyError,IndexError)):r.validate(*self.data)
    def test_full_5978_roundtrip(self):
        answer=r.validate(*self.data)
        self.assertEqual((answer['rows_verified'],answer['stock_rows_verified'],answer['etf_rows_verified'],answer['partitions_verified']),(5978,3904,2074,3904))
    def test_missing_stock(self):self.data[5].pop(0);self.fails()
    def test_reordered_rows(self):self.data[5][0],self.data[5][1]=self.data[5][1],self.data[5][0];self.fails()
    def test_duplicate_origin(self):self.data[2]['rows'][1]=deepcopy(self.data[2]['rows'][0]);self.fails()
    def test_eligibility_changed(self):self.data[5][0]['research_eligible']=False;self.fails()
    def test_eligibility_integer_not_bool(self):self.data[5][0]['research_eligible']=1;self.fails()
    def test_reason_changed(self):self.data[5][0]['reasons']=['hidden change'];self.fails()
    def test_etf_nonneutral(self):self.data[5][32]['partition']={};self.fails()
    def test_etf_cutoff_changed(self):self.data[5][32]['decision_cutoff_ns']+=1;self.fails()
    def test_stock_cutoff_text_changed(self):self.data[5][0]['decision_cutoff']='late';self.fails()
    def test_context_lost(self):self.data[5][0]['context_original_event_ids']=['future'];self.fails()
    def test_source_replaced(self):self.data[5][0]['selected_snapshot']='favorable';self.fails()
    def test_receipt_omitted(self):self.data[5][0]['source_refs'].pop();self.fails()
    def test_assumption_omitted(self):self.data[5][0]['assumption_category']='certified';self.fails()
    def test_entire_partition_required(self):self.data[5][0]['partition']['failure_reasons']=['changed'];self.fails()
    def test_hidden_price_field(self):self.data[5][0]['return']=.4;self.fails()
    def test_source_review_pointer(self):self.data[5][0]['source_review_ref']['sha256']='0'*64;self.fails()
    def test_counterfeit_bundle_matches_copies_but_not_raw(self):
        self.data[5][0]['partition']['bundles'][0]['anchor']+=1
        self.data[3]['rows'][0]['partition']['bundles'][0]['anchor']+=1
        self.fails()
    def test_density_threshold_gate_corrupted(self):self.data[6]['checks']['first88_fraction']=False;self.fails()
    def test_missing_floor_is_rejected(self):self.data[2]['coverage_floor_checks'].pop('first88_fraction');self.fails()
    def test_later34_cannot_change_first88_floors(self):
        rows=self.data[5];before=r.coverage(rows,set(self.data[0]['etfs']))
        for row in rows:
            if row['signal_index']>=88:row['research_eligible']=False
        after=r.coverage(rows,set(self.data[0]['etfs']))
        self.assertEqual(before['checks'],after['checks'])
        self.assertEqual(before['eras']['first88'],after['eras']['first88'])
    def test_mutation_free(self):
        before=deepcopy(self.data);r.validate(*self.data);self.assertEqual(self.data,before)


class RuntimePathTests(unittest.TestCase):
    def setUp(self):
        self.runtime,self.digest=next(iter(r.EXTERNAL_RUNTIME.items()))
    def test_exact_pinned_external_runtime_allowed(self):
        inputs=r.Inputs()
        with patch.object(Path,'is_symlink',return_value=False), patch.object(Path,'resolve',lambda p:p), patch.object(r,'sha',return_value=self.digest):
            self.assertEqual(inputs.bind(self.runtime,self.digest),self.runtime)
            self.assertEqual(inputs.hashes,{self.runtime:self.digest})
    def test_other_external_paths_rejected(self):
        for value in ('/tmp/anything','/usr/bin/python3',self.runtime+'.other','/Users/other/file'):
            with self.subTest(value=value),self.assertRaises(ValueError):r.Inputs().name(value)
    def test_all_parent_traversals_rejected(self):
        for value in ('../file','work/../file',str(r.ROOT/'work/../file'),self.runtime.replace('/bin/','/bin/../bin/')):
            with self.subTest(value=value),self.assertRaises(ValueError):r.Inputs().name(value)
    def test_external_runtime_final_symlink_rejected(self):
        with patch.object(Path,'is_symlink',return_value=True),self.assertRaises(ValueError):r.Inputs().name(self.runtime)
    def test_external_runtime_parent_symlink_rejected(self):
        with patch.object(Path,'is_symlink',return_value=False),patch.object(Path,'resolve',return_value=Path('/tmp/outside')), self.assertRaises(ValueError):
            r.Inputs().name(self.runtime)
    def test_internal_symlink_escape_rejected(self):
        with patch.object(Path,'is_symlink',return_value=False),patch.object(Path,'resolve',return_value=Path('/tmp/outside')), self.assertRaises(ValueError):
            r.Inputs().name('work/synthetic')
    def test_runtime_digest_mismatch_rejected_even_without_expected(self):
        with patch.object(Path,'is_symlink',return_value=False),patch.object(Path,'resolve',lambda p:p),patch.object(r,'sha',return_value='0'*64):
            for expected in (None,self.digest,'0'*64):
                with self.subTest(expected=expected),self.assertRaises(ValueError):r.Inputs().bind(self.runtime,expected)
    def test_runtime_wrong_caller_digest_rejected(self):
        with patch.object(Path,'is_symlink',return_value=False),patch.object(Path,'resolve',lambda p:p),patch.object(r,'sha',return_value=self.digest),self.assertRaises(ValueError):
            r.Inputs().bind(self.runtime,'0'*64)
    def test_internal_path_behavior_and_digest_preserved(self):
        with patch.object(Path,'is_symlink',return_value=False),patch.object(Path,'resolve',lambda p:p),patch.object(r,'sha',return_value='a'*64):
            self.assertEqual(r.Inputs().bind(str(r.ROOT/'work/synthetic'),'a'*64),'work/synthetic')
            with self.assertRaises(ValueError):r.Inputs().bind('work/synthetic','b'*64)


if __name__=='__main__':unittest.main(verbosity=2)
