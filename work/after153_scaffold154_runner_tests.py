"""Prepare new synthetic tests; no source data or returns are loaded."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
src=(ROOT/"work/test_monthly_financial_coverage153_runner.py").read_text()
src=src.replace("monthly_financial_coverage153_", "monthly_financial_coverage154_")
src=src.replace("round=153", "round=154").replace("r.NAMES[5]", "r.NAMES[6]")
start=src.index("def feature_fixture():")
stop=src.index("\ndef source2_fixture():",start)
src=src[:start]+'''def feature_fixture():
    values=[];current=np.full((3,49),np.nan)
    for i in range(6):
        current=current.copy();current[:,i]=(-1 if i==2 else 1)*.1*(i+1);values.append(current)
    names=["AAPL","AMZN","INTC","KO","MU","PLTR"]+["X"+str(i) for i in range(43)]
    meta=dict(dates=["2020-01-31","2020-02-28","2020-03-31"],symbols=[names.copy() for _ in range(3)])
    arrays={k:v.copy() for k,v in zip(r.HANDS,values)}
    arrays.update({k:np.isfinite(v) for k,v in zip(r.EXTRA[6:],values)})
    args=[dict(arrays=arrays)]
    for v in values[:5]:args.extend((v,np.isfinite(v)))
    return (*args,meta)

''' +src[stop:]
src=src.replace("for column in (0, 5):","for column in (0, 6):")
src=src.replace("self.assertEqual(len(r.NAMES), 6)", "self.assertEqual(len(r.NAMES), 7)")
start=src.index("    def test_reference_arrays_read_four_frozen_hands")
stop=src.index("    def test_batch2_complete_source_contract",start)
src=src[:start]+'''    def test_reference_arrays_read_five_frozen_hands_without_factor_call(self):
        with tempfile.TemporaryDirectory() as directory:
            obj=object.__new__(r.Runner);obj.root=Path(directory)
            obj.require_legacy=Mock();obj.require_prior=Mock()
            meta=dict(synthetic=True);originals={"old"+str(i):np.zeros((122,49)) for i in range(27)}
            values=[];a=np.full((122,49),np.nan)
            for i in range(5):a=a.copy();a[:,i]=.1*(i+1);values.append(a)
            scores=[np.full((122,49),float(i)) for i in range(6)]
            for round_ in range(149,154):
                folder=r.LEGACY_OUT if round_==149 else "outputs/monthly_financial_coverage"+str(round_)
                out=obj.root/folder;out.mkdir(parents=True);(out/"META.json").write_text(json.dumps(meta))
                extra={"earnings_bounded":values[0],"common_available":np.isfinite(values[0])} if round_==149 else {"earnings_original":values[0],"original_available":np.isfinite(values[0]),"earnings_expanded":values[round_-149],"expanded_available":np.isfinite(values[round_-149])}
                for j in range(1,round_-149):
                    extra["earnings_batch"+str(j)]=values[j];extra["batch"+str(j)+"_available"]=np.isfinite(values[j])
                np.savez(out/"DATA.npz",**originals,**extra)
                names=("incumbent_local67","earnings_control") if round_==149 else (*r.NAMES[:round_-149+1],"expanded_earnings")
                np.savez(out/"SCORES.npz",**dict(zip(names,scores)))
                (out/"PRETRAIN_ORDERS.json").write_text(json.dumps({name:["old"+str(i)] for i,name in enumerate(names)}))
            with patch.object(r.factor,"build",side_effect=AssertionError("New factor forbidden")),patch.object(r.previous.factor,"build",side_effect=AssertionError("Prior factor forbidden")),patch.object(r.old.factor,"build_features",side_effect=AssertionError("Old factor forbidden")):
                got=obj.reference_arrays(meta,originals)
            self.assertEqual(len(got),12)
            for i in range(5):self.assertEqual(got[2*i].tobytes(),values[i].tobytes())
            self.assertEqual(got[-1][r.NAMES[5]],["old5"])
            self.assertEqual(got[-2][r.NAMES[5]].tobytes(),scores[5].tobytes())
            originals["old0"][0,0]=1
            with self.assertRaises(ValueError):obj.reference_arrays(meta,originals)

''' +src[stop:]
start=src.index("    def test_union_preserves_all_four_packages")
stop=src.index("    def test_prepare_receipt_counts_are_exact",start)
src=src[:start]+'''    def test_union_preserves_all_five_packages_and_refs(self):
        packages=[source_fixture()[0],source2_fixture()[0],source3_fixture()[0],source4_fixture()[0],source5_fixture()[0]]
        for i,data in enumerate(packages):
            for e in data["events"]+data["latest_event_catalog"]:e["event_id"]="batch"+str(i)+"_"+e["event_id"]
        refs={b:{k:dict(path=p,sha256="a"*64) for k,p in paths.items()} for b,paths in r.COVERAGE_REFS.items()}
        def reference(data,batch):
            result=dict(records_canonical_sha256=r.canonical_hash(data["events"]),catalog_canonical_sha256=r.canonical_hash(data["latest_event_catalog"]))
            for k,rk in (("path","dataset"),("admission_path","admission"),("coverage_path","coverage"),("manifest_path","manifest")):
                result[k]=r.COVERAGE_REFS[batch][rk];result["sha256" if k=="path" else k.replace("_path","_sha256")]="a"*64
            return result
        for i,data in enumerate(packages):
            for j in range(i):
                data["prior_batch"+str(j+1)+"_reference"]=reference(packages[j],"batch"+str(j+1))
        result=r.combine_sources(*packages,refs)
        self.assertEqual(result["events"],sum((p["events"] for p in packages),[]))
        self.assertEqual(len(result["latest_event_catalog"]),147)
        result["events"][-1]["symbol"]="changed"
        self.assertNotEqual(packages[-1]["events"][-1]["symbol"],"changed")
        for key in ("records_canonical_sha256","catalog_canonical_sha256","admission_sha256"):
            ref=packages[-1]["prior_batch4_reference"];saved=ref[key];ref[key]="wrong"
            with self.assertRaises(ValueError):r.combine_sources(*packages,refs)
            ref[key]=saved

    def test_prior153_and_nested152_151_150_finalized_chain(self):
        obj=object.__new__(r.Runner);obj.root=Path("/synthetic");obj.inputs={k:obj.root/k for k in r.INPUTS}
        obj.path=lambda p:Path(p) if Path(p).is_absolute() else obj.root/p
        refs={n:{k:dict(sha256=str(n)+"_"+k) for k in r.LEGACY_FILES} for n in range(150,154)}
        refs[153]["plan"]["sha256"]=r.PRIOR_PLAN_SHA;refs[153]["completion"]["sha256"]=r.PRIOR_COMPLETION_SHA
        obj.plan=dict(prior153=refs[153],legacy149={"fixed":True},coverage_sources={k:{"fixed":k} for k in r.COVERAGE_REFS},raw_source_hashes={"fixed":"raw"})
        groups={};module=r.previous
        for n in (153,152,151,150):
            rf=refs[n];folder="outputs/monthly_financial_coverage"+str(n)
            cache={k:k+"_hash" for k in r.CACHE_FILES};pins={str(obj.root/folder/k):v for k,v in cache.items()}
            plan=dict(round=n,status="REGISTERED",arms=list(module.NAMES),hashes={"old":"hash"},legacy149=obj.plan["legacy149"],inputs={k:dict(path=str(v),sha256="digest") for k,v in obj.inputs.items()},raw_source_hashes=obj.plan["raw_source_hashes"])
            if n>150:
                plan["prior"+str(n-1)]=refs[n-1]
                plan["coverage_sources"]={k:obj.plan["coverage_sources"][k] for k in ("batch1","batch2","batch3","batch4")[:n-149]}
                module=module.previous
            else:plan["coverage_source"]=obj.plan["coverage_sources"]["batch1"]
            done=dict(round=n,plan_sha256=rf["plan"]["sha256"],independent_outcome_sha256=rf["outcome"]["sha256"])
            rev=dict(plan_sha256=rf["plan"]["sha256"],cache_sha256=rf["cache"]["sha256"],passed=True,stage="evaluate",artifact_hashes=pins,source_hashes={"independent":"hash"},orders_sha256=cache["PRETRAIN_ORDERS.json"])
            groups[n]=dict(plan=plan,cache=cache,completion=done,outcome=rev)
        obj._refs=lambda rf,*args:(next(groups[n] for n in groups if rf is refs[n]),{})
        obj._closure=lambda d:{str(obj.path(k)):v for k,v in d.items()}
        with patch.object(r,"sha",return_value="digest"):
            obj.require_prior()
            for group_ in groups.values():
                for key,value in (("cache_sha256","wrong"),("orders_sha256","wrong"),("passed",False)):
                    before=group_["outcome"][key];group_["outcome"][key]=value
                    with self.assertRaises(ValueError):obj.require_prior()
                    group_["outcome"][key]=before
            groups[150]["plan"]["coverage_source"]={"swapped":True}
            with self.assertRaises(ValueError):obj.require_prior()

''' +src[stop:]
src=src.replace("orders_checked=732", "orders_checked=854")
src=src.replace('self.assertEqual(set(r.factor.NEW_SYMBOLS),{"MU","PEP"})','self.assertEqual(set(r.factor.NEW_SYMBOLS),{"PLTR","PYPL","TMO"})')
src=src.replace("(6,10,49)","(7,12,49)")
src=src.replace('a[-1]["symbols"][0][4]="KO"','a[-1]["symbols"][0][5]="KO"')
src=src.replace("all_four_immutable_controls", "all_five_immutable_controls")
src=src.replace("prior152={},coverage_sources={}", "prior153={},coverage_sources={}")
src=src.replace("for name in r.NAMES[1:5]:", "for name in r.NAMES[1:6]:")
src=src.replace("self.assertEqual(len(build.call_args.args),6)", "self.assertEqual(len(build.call_args.args),7)")
src=src.replace("for i in range(4):self.assertIs", "for i in range(5):self.assertIs")
src=src.replace('self.assertTrue(result["audit"]["prior152_control_scores_and_orders_byte_exact"])', 'self.assertTrue(result["audit"]["prior153_control_scores_and_orders_byte_exact"])')
src=src.replace('self.assertTrue(result["audit"]["prior152_batch3_earnings_cache_and_mask_verified"])', 'self.assertTrue(result["audit"]["prior153_batch4_earnings_cache_and_mask_verified"])')
insert=src.index("\nclass RunnerTests(")
src=src[:insert]+'''
def source5_fixture(admitted_count=19):
    args=source4_fixture()
    data,adm,policy,review,manifest,cov,refs,old,ext,plan=args
    templates=[deepcopy(obj[key][0]) for obj,key in ((data,"events"),(ext,"events"),(review,"event_reviews"))]
    data["events"]=[];ext["events"]=[];review["event_reviews"]=[]
    for i in range(21):
        e,raw,rr=deepcopy(templates)
        symbol="PLTR" if i<6 else "PYPL" if i<11 else "TMO"
        event_id=symbol+":"+str(i)
        e.update(event_id=event_id,symbol=symbol);raw.update(event_id=event_id,symbol=symbol)
        rr.update(event_id=event_id,assumption_id=symbol+"_consecutive_annual_boundaries_v1")
        e["identity"]["security_id"]=symbol
        ok=i<admitted_count
        e["ni_assets_admission"].update(admitted=ok,reasons=[] if ok else ["hold"])
        rr["clock_verified"]=ok;e["clock_review"]["verified"]=ok
        data["events"].append(e);ext["events"].append(raw);review["event_reviews"].append(rr)
    plan["records"]=[dict(event_id=e["event_id"]) for e in data["events"]]
    data["latest_event_catalog"]=[dict(event_id=e["event_id"],symbol=e["symbol"]) for e in data["events"]]+[dict(event_id="PYPL_missing"+str(i),symbol="PYPL") for i in range(6)]+[dict(event_id="TMO_missing",symbol="TMO")]
    policy["source_scope"]["symbols"]=["PLTR","PYPL","TMO"]
    policy["known_time"]={"age_days":550}
    metadata=[{k:deepcopy(row.get(k)) for k in r.CATALOG_KEYS} for row in data["latest_event_catalog"]]
    policy["fixed_catalog_metadata"]=deepcopy(metadata)
    plan["catalog_metadata"]=deepcopy(metadata);ext["catalog_metadata"]=deepcopy(metadata)
    adm["counts"].update(events=21,admitted=admitted_count,held=21-admitted_count,annual_catalog_events=28,
                        unprocessed_annual_catalog_events=7,prior_batch4_events=9,prior_batch4_admitted=5,
                        prior_batch4_held=4,reviewed_nonnull_fields=63,reviewed_null_fields=0)
    return args

''' +src[insert:]
insert=src.index('\n\nif __name__ == "__main__":')
src=src[:insert]+'''

class Batch5Tests(unittest.TestCase):
    def test_complete_dynamic_count(self):
        for n in (0,1,19,21):
            self.assertEqual(r.validate_source(*source5_fixture(n),batch="batch5"),dict(events=21,admitted=n,held=21-n,catalog_events=28))

    def test_all_three_issuers_and_full_catalog_required(self):
        for change in ("missing","duplicate","wrong_symbol"):
            a=source5_fixture()
            if change=="missing":a[0]["latest_event_catalog"].pop()
            elif change=="duplicate":a[0]["events"][-1]=deepcopy(a[0]["events"][0])
            else:a[0]["events"][-1]["symbol"]="MU"
            with self.assertRaises(ValueError):r.validate_source(*a,batch="batch5")

    def test_age_days_policy_is_exact(self):
        for value in (549,551,None):
            a=source5_fixture();a[2]["known_time"]["age_days"]=value
            with self.assertRaises(ValueError):r.validate_source(*a,batch="batch5")

    def test_batch4_counts_and_both_clock_holds_preserved(self):
        a=source5_fixture();a[1]["counts"]["prior_batch4_held"]=3
        with self.assertRaises(ValueError):r.validate_source(*a,batch="batch5")
        a=source5_fixture();a[0]["events"][-1]["ni_assets_admission"].update(admitted=True,reasons=[])
        a[1]["counts"].update(admitted=20,held=1)
        with self.assertRaises(ValueError):r.validate_source(*a,batch="batch5")

    def test_accounting_boundary_and_money_field_reviews_required(self):
        for change in ("accounting","boundary","amount"):
            a=source5_fixture();rr=a[3]["event_reviews"][0]
            if change=="accounting":rr["accounting_review"]["current_prior_common_basis"]=False
            elif change=="boundary":rr["boundary_assumption_verified"]=False
            else:rr["fields"][r.factor.FIELDS[0]]["source_value_present"]=False
            with self.assertRaises(ValueError):r.validate_source(*a,batch="batch5")

    def test_exact_sixty_three_fields_cannot_be_dropped(self):
        a=source5_fixture(0);name=r.factor.FIELDS[0]
        a[0]["events"][0]["fields"][name]=None;a[8]["events"][0]["fields"][name]=None
        with self.assertRaisesRegex(ValueError,"Fixed63"):r.validate_source(*a,batch="batch5")

    def test_all_catalog_projections_bound(self):
        for index in (2,8,9):
            a=source5_fixture();key="fixed_catalog_metadata" if index==2 else "catalog_metadata"
            a[index][key][-1]["context_holds"]=["changed"]
            with self.assertRaisesRegex(ValueError,"projection"):r.validate_source(*a,batch="batch5")

    def test_frozen_batch4_values_and_mask_cannot_change(self):
        for key in ("earnings_batch4","batch4_available"):
            a=feature_fixture();a[0]["arrays"][key][0,4]=0
            with self.assertRaises(ValueError):r.validate_features(*a)

    def test_new_catalog_no_sentinel_loss(self):
        a=source5_fixture();data=a[0]
        for row in data["latest_event_catalog"]:row.update(fiscal_quarter=4,publication_date="2020-01-01")
        census=dict(annual_documents=deepcopy(data["latest_event_catalog"]))
        for row in data["latest_event_catalog"]:row.update(clock_review={},processed_in_batch=row["event_id"] in {e["event_id"] for e in data["events"]})
        r.validate_catalog(data,census,"batch5")
        data["latest_event_catalog"][-1]["processed_in_batch"]=True
        with self.assertRaises(ValueError):r.validate_catalog(data,census,"batch5")
''' +src[insert:]
with (ROOT/"work/test_monthly_financial_coverage154_runner.py").open("x") as stream:stream.write(src)
