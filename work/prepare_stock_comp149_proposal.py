"""Prepare a write-once149 proposal; no financial fields, scores or returns."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import monthly_stock_comp149_runner as r

ROOT = Path(__file__).resolve().parents[1]
PRIOR = "work/monthly_cash_quality146_proposal.json"
NOVELTY = "work/after148_next_mechanism_review.json"
NOVELTY_SHA = "dc4237f5c4382f40800bb8cbb2b8268c069e80ba10703fac60d860408f70c1de"


def checked_ref(root, name):
    root = Path(root).resolve()
    path = root/name
    r.require(path.resolve() == path and path.is_relative_to(root) and path.is_file(),
              "Existing local regular non-symlink input required")
    return dict(path=name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def build_proposal(prior, references, created_utc):
    """Pure schema construction; referenced source packages remain external."""
    r.require(set(references) == set(r.SBC_REFS), "All four new source references required")
    for key, ref in references.items():
        r.require(type(ref) is dict and set(ref) == {"path", "sha256"}
                  and ref["path"] == r.SBC_REFS[key] and type(ref["sha256"]) is str
                  and re.fullmatch("[0-9a-f]{64}", ref["sha256"]),
                  "Exact final SBC source reference required")
    p = deepcopy(prior)
    r.require(all(key in p.get("sources", {}) for key in ("dataset", "admission", "policy")),
              "Original146 source references required")
    p.update(round=149, created_utc=created_utc, arms=list(r.NAMES),
             status="FINAL_PROSPECTIVE_SPECIFICATION_BEFORE_REGISTRATION",
             primary_challenger=r.NAMES[3], dose=r.DOSE,
             actual_new_factors=0, actual_new_orders=0, actual_new_returns=0,
             concept=dict(path=NOVELTY, sha256=NOVELTY_SHA),
             novelty=dict(path=NOVELTY, sha256=NOVELTY_SHA),
             measurement=r.measurement_contract(), selection_contract=r.selection_contract(),
             hypothesis="An explicit reported stock-compensation addback adjustment may improve long-momentum selection beyond the same-mask earnings and reported operating-cashflow controls.")
    p["sources"].update(deepcopy(references))
    p["interpretation"] = [
        "The adjusted value is a research measure, not reported GAAP operating cashflow or cash available to shareholders.",
        "A reported noncash compensation addback is not a measured dilution cost; negative reported amounts retain their sign only under explicit source admission.",
        "All44 source events and the original13 holds remain. The original31 eligible events can only be narrowed by the new field-specific source gate.",
        "Sparse four-issuer coverage may introduce covered-issuer exposure. Three controls share exactly the same narrowed mask.",
        "Same-report dollar literal is an assumption, not new ISO-currency certification. First-public versions, complete revision histories and continuous security master remain unverified.",
        "Only current publication availability is used; no historical comparative value is backfilled to an earlier release.",
        "All arms use a fixed0.01 dose without volatility or turnover normalization. Long momentum, skip42days, month-end EW3 and no intramonth trades remain unchanged.",
        "First88 and next-open5bp gates decide selection. Later34 are diagnostics and cannot rescue a rejected challenger.",
        "Prior experiments and later history have been inspected repeatedly; no pristine out-of-sample or global novelty claim.",
        "No dose, sign, source-year or threshold search is authorized by this proposal."
    ]
    r.validate_proposal(p)
    return p


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--root", type=Path, default=ROOT)
    args = cli.parse_args()
    root = args.root.resolve()
    references = {key: checked_ref(root, path) for key, path in r.SBC_REFS.items()}
    r.require(checked_ref(root, NOVELTY)["sha256"] == NOVELTY_SHA, "Source hypothesis changed")
    prior_ref = checked_ref(root, PRIOR)
    prior = json.loads((root/prior_ref["path"]).read_text())
    proposal = build_proposal(prior, references, datetime.now(timezone.utc).isoformat())
    target = root/"work/monthly_stock_comp149_proposal.json"
    with target.open("x") as stream:
        json.dump(proposal, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(checked_ref(root, str(target.relative_to(root)))))


if __name__ == "__main__":
    main()
