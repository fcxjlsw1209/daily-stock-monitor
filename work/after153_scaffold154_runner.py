"""Create prospective round154 runner/proposal templates; no actual factors."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def write(name, text):
    path = ROOT / "work" / name
    with path.open("x") as stream:
        stream.write(text)


src = (ROOT / "work/monthly_financial_coverage153_runner.py").read_text()
src = src.replace("Cumulative coverage replay with immutable149,150,151 and152 earnings controls.",
                  "Cumulative coverage replay with immutable149 through153 earnings controls.")
src = src.replace("MU/PEP", "PLTR/PYPL/TMO")
src = src.replace("monthly_financial_coverage152_runner as previous", "monthly_financial_coverage153_runner as previous")
src = src.replace("monthly_financial_coverage153_", "monthly_financial_coverage154_")
# The previous-module import is deliberately153 rather than154.
src = src.replace("monthly_financial_coverage154_runner as previous", "monthly_financial_coverage153_runner as previous")
src = src.replace('OUT = "outputs/monthly_financial_coverage153"', 'OUT = "outputs/monthly_financial_coverage154"')
src = src.replace('PRIOR_OUT = "outputs/monthly_financial_coverage152"', 'PRIOR_OUT = "outputs/monthly_financial_coverage153"')
src = src.replace('"batch3_available", "expanded_available"', '"batch3_available", "batch4_available", "expanded_available"')
start = src.index("COVERAGE_REFS = ")
stop = src.index("SOURCE_BATCHES = ", start)
src = src[:start] + '''COVERAGE_REFS = dict(previous.COVERAGE_REFS, batch5={
    "dataset": "outputs/monthly_financial_coverage_admission_batch5_v1/DATASET.json",
    "admission": "outputs/monthly_financial_coverage_admission_batch5_v1/ADMISSION.json",
    "coverage": "outputs/monthly_financial_coverage_admission_batch5_v1/COVERAGE.json",
    "manifest": "outputs/monthly_financial_coverage_admission_batch5_v1/MANIFEST.json",
    "policy": "work/after153_batch5_admission_policy.json",
    "review": "work/after153_batch5_independent_review.json",
})
SOURCE_BATCH = dict(previous.SOURCE_BATCH,batch5="outputs/monthly_financial_coverage_batch5_v1")
''' + src[stop:]
src = src.replace("b507b39c3b7362080e954decf95134e487ce43067d6eab3096b25f22c502152d", "3ced8088fa5fed13ec29591e581ed0c38ed5f88916acff5b6a3f9e33b50b190b")
src = src.replace("7b9925053304e38b94e8ad386e0dc74f7d220a26b9884bb3f99e189365e23016", "6828b1dfd341f34cccc77d3d20bb3135d0fb28a3a2f80d8eb6e29c52157dc560")
src = src.replace("all_six_score_matrices", "all_seven_score_matrices")
src = src.replace('    "prior152_batch3_earnings_cache_and_mask_verified", "batch3_source_records_and_holds_preserved",',
'''    "prior152_batch3_earnings_cache_and_mask_verified", "batch3_source_records_and_holds_preserved",
    "prior153_batch4_earnings_cache_and_mask_verified", "batch4_source_records_and_holds_preserved",''')
src = src.replace('new_symbols=["MU", "PEP"], source_events=64, annual_catalog_events=119,',
                  'new_symbols=["PLTR", "PYPL", "TMO"], source_events=85, annual_catalog_events=147,')
src = src.replace('        batch1_events=26,batch2_events=19,batch3_events=10,batch4_events=9,',
'''        prior153_control="verified_final153_CACHE_earnings_expanded_and_expanded_available",
        batch1_events=26,batch2_events=19,batch3_events=10,batch4_events=9,batch5_events=21,''')
src = src.replace('batch4_admission=', 'batch5_admission=').replace('batch4_annual_days=[364,371]', 'batch5_annual_days=[365,366]')
src = src.replace('batch1_admitted=17,batch2_admitted=14,batch3_admitted=5,', 'batch1_admitted=17,batch2_admitted=14,batch3_admitted=5,batch4_admitted=5,')
src = src.replace('distinct_original_batch1_batch2_batch3_expanded_with_all_four_frozen_controls_preserved', 'distinct_original_batch1_batch2_batch3_batch4_expanded_with_all_five_frozen_controls_preserved')
src = src.replace('p.get("round") == 153', 'p.get("round") == 154')
src = src.replace('p.get("primary_challenger") == NAMES[5]', 'p.get("primary_challenger") == NAMES[6]')
# New source validator delegates every earlier batch unchanged.
start = src.index("def validate_source(")
stop = src.index("\ndef validate_features(", start)
segment = src[start:stop]
segment = segment.replace('"batch4"', '"batch5"').replace("Batch4", "Batch5").replace("batch4", "batch5")
segment = segment.replace('["MU", "PEP"]', '["PLTR", "PYPL", "TMO"]')
segment = segment.replace('get("maximum_age_days")', 'get("age_days")')
segment = segment.replace('len(set(ids)) == 9', 'len(set(ids)) == 21')
segment = segment.replace('{"MU": 2, "PEP": 7}', '{"PLTR": 6, "PYPL": 5, "TMO": 10}')
segment = segment.replace("fixed9 event", "fixed21 event")
segment = segment.replace('len(set(catalog_ids)) == 23', 'len(set(catalog_ids)) == 28')
segment = segment.replace('{"MU": 7, "PEP": 16}', '{"PLTR": 6, "PYPL": 11, "TMO": 11}')
segment = segment.replace("fixed23 annual", "fixed28 annual")
segment = segment.replace("len(metadata)==23", "len(metadata)==28")
segment = segment.replace("factor.FIELDS)==27", "factor.FIELDS)==63").replace("Fixed27 published", "Fixed63 published")
segment = segment.replace("len(reviews) == 9", "len(reviews) == 21")
segment = segment.replace("events=9, admitted=admitted, held=9-admitted, annual_catalog_events=23,", "events=21, admitted=admitted, held=21-admitted, annual_catalog_events=28,")
segment = segment.replace("unprocessed_annual_catalog_events=14", "unprocessed_annual_catalog_events=7")
segment = segment.replace('counts.get("reviewed_nonnull_fields") == 27', 'counts.get("reviewed_nonnull_fields") == 63')
segment = segment.replace('counts.get("prior_batch3_held") == 5,', '''counts.get("prior_batch3_held") == 5 and
            counts.get("prior_batch4_events") == 9 and counts.get("prior_batch4_admitted") == 5 and
            counts.get("prior_batch4_held") == 4,''')
segment = segment.replace("return dict(events=9, admitted=admitted, held=9-admitted, catalog_events=23)", "return dict(events=21, admitted=admitted, held=21-admitted, catalog_events=28)")
src = src[:start] + segment + src[stop:]
start = src.index("def validate_features(")
stop = src.index("\ndef panels(", start)
src = src[:start] + '''def validate_features(result,original_earnings,original_mask,batch1_earnings,batch1_mask,batch2_earnings,batch2_mask,batch3_earnings,batch3_mask,batch4_earnings,batch4_mask,meta):
    arrays=result["arrays"];require(set(arrays)==set(EXTRA),"Exact twelve coverage arrays required")
    shape=(len(meta["dates"]),WIDTH)
    masks=EXTRA[6:]
    for hand,mask in zip(HANDS,masks):
        require(arrays[hand].shape==arrays[mask].shape==shape and arrays[hand].dtype==np.float64 and
                arrays[mask].dtype==bool and not np.isinf(arrays[hand]).any() and
                np.array_equal(np.isfinite(arrays[hand]),arrays[mask]),"Feature shape/type/mask mismatch")
        require(np.all(np.abs(arrays[hand][arrays[mask]])<=1),"Bounded earnings required")
    refs=((original_earnings,original_mask),(batch1_earnings,batch1_mask),(batch2_earnings,batch2_mask),(batch3_earnings,batch3_mask),(batch4_earnings,batch4_mask))
    for hand,mask,(values,available) in zip(HANDS,masks,refs):
        require(arrays[hand].tobytes()==values.tobytes() and arrays[mask].tobytes()==available.tobytes(),
                "Frozen149/150/151/152/153 hand or mask changed")
    for i in range(5):
        low,high=arrays[masks[i]],arrays[masks[i+1]]
        require(np.all(~low|high) and arrays[HANDS[i+1]][low].tobytes()==arrays[HANDS[i]][low].tobytes(),
                "Expanded changed previous coverage")
    for i,names in enumerate(meta["symbols"]):
        for j,name in enumerate(names):
            if name not in factor.NEW_SYMBOLS:
                require(arrays[HANDS[4]][i:i+1,j:j+1].tobytes()==arrays[HANDS[5]][i:i+1,j:j+1].tobytes(),
                        "Nonbatch5 coverage changed")


def validate_catalog(dataset,census,batch="batch1"):
    require(batch in COVERAGE_REFS,"Unknown coverage batch")
    if batch!="batch5":return previous.validate_catalog(dataset,census,batch)
    expected=[r for r in census["annual_documents"] if r.get("symbol") in factor.NEW_SYMBOLS]
    catalog=dataset["latest_event_catalog"]
    stripped=[{k:v for k,v in r.items() if k not in {"clock_review","processed_in_batch"}} for r in catalog]
    require(stripped==expected and len(expected)==28 and all(r.get("fiscal_quarter")==4 for r in expected),
            "Full batch5 catalog changed")
    ids={e["event_id"] for e in dataset["events"]}
    require(all(r.get("processed_in_batch") is (r["event_id"] in ids) for r in catalog),"Batch5 sentinel changed")


def combine_sources(first,second,third,fourth,fifth,refs):
    combined=previous.combine_sources(first,second,third,fourth,refs)
    require(fifth["original_four_issuer_reference"]==first["original_four_issuer_reference"] and
            all(fifth["prior_"+key+"_reference"]==fourth["prior_"+key+"_reference"]
                for key in ("batch1","batch2","batch3")),"Prior source references differ")
    prior=fifth["prior_batch4_reference"]
    for key,refkey in (("path","dataset"),("admission_path","admission"),("manifest_path","manifest"),("coverage_path","coverage")):
        require(prior[key]==COVERAGE_REFS["batch4"][refkey] and
                prior["sha256" if key=="path" else key.replace("_path","_sha256")]==refs["batch4"][refkey]["sha256"],
                "Batch5 prior-source reference mismatch")
    require(prior["records_canonical_sha256"]==canonical_hash(fourth["events"]) and
            prior["catalog_canonical_sha256"]==canonical_hash(fourth["latest_event_catalog"]),"Prior batch4 records/catalog changed")
    require(len(fifth["events"])==21 and len(fifth["latest_event_catalog"])==28,"Exact batch5 population required")
    combined["events"].extend(deepcopy(fifth["events"]));combined["latest_event_catalog"].extend(deepcopy(fifth["latest_event_catalog"]))
    require(len(combined["events"])==len({e["event_id"] for e in combined["events"]})==85 and
            len(combined["latest_event_catalog"])==len({e["event_id"] for e in combined["latest_event_catalog"]})==147,
            "Union duplicates or missing events")
    return combined

''' + src[stop:]
src = src.replace('p["round"] == 153', 'p["round"] == 154')
src = src.replace('p["primary_challenger"] == NAMES[5]', 'p["primary_challenger"] == NAMES[6]')
src = src.replace("Registered153 PLAN", "Registered154 PLAN").replace("Dedicated153 directory", "Dedicated154 directory")
src = src.replace('p["prior152"] == proposal["sources"]["prior152"]', 'p["prior153"] == proposal["sources"]["prior153"]')
start = src.index("    def require_prior(self):")
stop = src.index("    def prepare(self):", start)
src = src[:start] + '''    def require_prior(self):
        refs=self.plan["prior153"]
        require(refs["plan"]["sha256"]==PRIOR_PLAN_SHA and
                refs["completion"]["sha256"]==PRIOR_COMPLETION_SHA,"Exact finalized153 predecessor required")
        module=previous
        first=None
        for number in (153,152,151,150):
            directory="outputs/monthly_financial_coverage"+str(number)
            objects,_=self._refs(refs,{k:directory+"/"+v for k,v in LEGACY_FILES.items()})
            if first is None:first=objects
            pp,cache,done,review=(objects[k] for k in ("plan","cache","completion","outcome"))
            require(pp["round"]==done["round"]==number and pp["status"]=="REGISTERED" and pp["arms"]==list(module.NAMES),"Completed predecessor required")
            require(pp["legacy149"]==self.plan["legacy149"],"Prior legacy reference changed")
            if number==150:
                require(pp["coverage_source"]==self.plan["coverage_sources"]["batch1"],"Prior150 sources changed")
            else:
                require(pp["coverage_sources"]=={k:self.plan["coverage_sources"][k] for k in ("batch1","batch2","batch3","batch4")[:number-149]},
                        "Prior coverage sources changed")
            require(done["plan_sha256"]==review["plan_sha256"]==refs["plan"]["sha256"] and
                    done["independent_outcome_sha256"]==refs["outcome"]["sha256"] and
                    review["cache_sha256"]==refs["cache"]["sha256"] and review.get("passed") is True and
                    review.get("stage")=="evaluate","Final predecessor audit/cache chain")
            self._closure(pp["hashes"]);require(set(cache)==set(CACHE_FILES),"Exact predecessor cache required")
            cp=self._closure({directory+"/"+k:v for k,v in cache.items()});audited=self._closure(review["artifact_hashes"])
            require(all(audited.get(k)==v for k,v in cp.items()) and review["orders_sha256"]==cache["PRETRAIN_ORDERS.json"],"Unaudited predecessor cache/order")
            self._closure(review["source_hashes"])
            for key in INPUTS:
                if key!="proposal":require(self.inputs[key]==self.path(pp["inputs"][key]["path"]) and sha(self.inputs[key])==pp["inputs"][key]["sha256"],"Prior input changed")
            require(self.plan["raw_source_hashes"]==pp["raw_source_hashes"],"Prior price sources changed")
            if number>150:
                refs=pp["prior"+str(number-1)]
                module=module.previous
        return first

    def require_coverage(self):
        require(set(self.plan["coverage_sources"])==set(COVERAGE_REFS),"Five exact source packages required")
        return combine_sources(*(self.require_batch(k) for k in ("batch1","batch2","batch3","batch4","batch5")),self.plan["coverage_sources"])

    def reference_arrays(self,meta,original):
        # Read-only predecessor helpers authenticate each old frozen control;
        # require_prior binds all four finalized coverage chains separately.
        *values,scores,orders=previous.Runner.reference_arrays(self,meta,original)
        directory=self.root/PRIOR_OUT
        require(read(directory/"META.json")==meta,"Prior153 metadata changed")
        with np.load(directory/"DATA.npz",allow_pickle=False) as cache:
            require(all(k in cache.files and cache[k].dtype==v.dtype and cache[k].shape==v.shape and
                        cache[k].tobytes()==v.tobytes() for k,v in original.items()),"Prior153 original27 changed")
            for hand,mask,index in zip(HANDS[:4],EXTRA[6:10],range(4)):
                for key,value in ((hand,values[2*index]),(mask,values[2*index+1])):
                    require(cache[key].dtype==value.dtype and cache[key].shape==value.shape and cache[key].tobytes()==value.tobytes(),"Prior153 frozen control changed")
            batch4,b4mask=cache["earnings_expanded"].copy(),cache["expanded_available"].copy()
        require(batch4.dtype==np.float64 and batch4.shape==values[0].shape and b4mask.dtype==bool and
                b4mask.shape==values[1].shape and not np.isinf(batch4).any() and np.array_equal(np.isfinite(batch4),b4mask),"Prior153 hand/mask invalid")
        with np.load(directory/"SCORES.npz",allow_pickle=False) as cache:
            for name in NAMES[:5]:require(cache[name].tobytes()==scores[name].tobytes(),"Prior153 frozen scores differ")
            scores[NAMES[5]]=cache["expanded_earnings"].copy()
        prior_orders=read(directory/"PRETRAIN_ORDERS.json")
        require(all(prior_orders[n]==orders[n] for n in NAMES[:5]),"Prior153 frozen orders differ")
        orders[NAMES[5]]=prior_orders["expanded_earnings"]
        return (*values,batch4,b4mask,scores,orders)

''' + src[stop:]
src = src.replace("batch3, b3mask, ref_scores", "batch3, b3mask, batch4, b4mask, ref_scores")
src = src.replace("batch3, self.require_coverage()", "batch3, batch4, self.require_coverage()")
src = src.replace("batch3, b3mask, meta)", "batch3, b3mask, batch4, b4mask, meta)")
src = src.replace("for name in NAMES[:5]:\n            require(orders[name]", "for name in NAMES[:6]:\n            require(orders[name]")
src = src.replace('legacy149=self.plan["legacy149"], prior152=self.plan["prior152"],', 'legacy149=self.plan["legacy149"], prior153=self.plan["prior153"],')
src = src.replace("prior152_control_scores_and_orders_byte_exact=True)", '''prior152_control_scores_and_orders_byte_exact=True,
            prior153_batch4_earnings_cache_and_mask_verified=True,
            prior153_control_scores_and_orders_byte_exact=True)''')
src = src.replace("batch3, b3mask, _, _ =", "batch3, b3mask, batch4, b4mask, _, _ =")
src = src.replace('("orders_checked",732)', '("orders_checked",854)')
write("monthly_financial_coverage154_runner.py", src)

p = (ROOT/"work/prepare_financial_coverage153_proposal.py").read_text()
p = p.replace("153", "154").replace("152", "153").replace("962", "968")
p = p.replace("b507b39c3b7362080e954decf95134e487ce43067d6eab3096b25f22c502154d", "3ced8088fa5fed13ec29591e581ed0c38ed5f88916acff5b6a3f9e33b50b190b")
# Explicitly replace constants by line so version-string substitutions cannot damage hashes.
p = "\n".join('PREVIOUS_PLAN = "3ced8088fa5fed13ec29591e581ed0c38ed5f88916acff5b6a3f9e33b50b190b"' if line.startswith("PREVIOUS_PLAN = ") else
              'PREVIOUS_DONE = "6828b1dfd341f34cccc77d3d20bb3135d0fb28a3a2f80d8eb6e29c52157dc560"' if line.startswith("PREVIOUS_DONE = ") else line
              for line in p.splitlines())+"\n"
p = p.replace("MU/PEP", "PLTR/PYPL/TMO").replace("experiment.NAMES[5]", "experiment.NAMES[6]")
p = p.replace("batch3 final153", "batch3 final152 and batch4 final153")
p = p.replace("26+19+10+9 batch records", "26+19+10+9+21 batch records")
p = p.replace("Three prior32-event catalogs and the new23-event catalog", "Three prior32-event catalogs, the prior23-event catalog and the new28-event catalog")
p = p.replace("PLTR/PYPL/TMO52/53-week start dates", "PLTR/PYPL/TMO January1 start dates")
p = p.replace("all four earnings controls", "all five earnings controls")
p = p.replace("All six arms and18 execution paths", "All seven arms and21 execution paths")
write("prepare_financial_coverage154_proposal.py", p)
