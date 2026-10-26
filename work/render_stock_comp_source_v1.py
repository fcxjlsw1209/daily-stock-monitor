"""Render pinned PDF source pages for manual accounting-table review only."""
import argparse
from pathlib import Path
import stock_comp_source_v1_runner as runner


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--plan-sha", required=True)
    args = cli.parse_args()
    root = runner.ROOT
    plan = runner.load_plan(root, args.plan_sha)
    manifest = runner.base.read(root, runner.OUT + "/MANIFEST.json")
    runner.require(manifest["plan_sha256"] == args.plan_sha, "Source manifest changed")
    runner.safe.verify_hashes(root, {
        runner.OUT + "/" + k: v for k, v in manifest["output_hashes"].items()
    })
    _, pdfium = runner.base.configure_pdf(plan)
    pages = []
    for event in plan["events"]:
        inventory = event["locator_inventory"]
        if inventory["format"] != "pdf":
            continue
        runner.require(len(inventory["candidates"]) == 1 and not inventory["gaps"],
                       "PDF locator ambiguous; inspect its source gap separately")
        locator = inventory["candidates"][0]
        source = event["source"]
        raw = runner.safe.local(root, source["path"]).read_bytes()
        runner.require(runner.safe.digest(raw) == source["sha256"], "PDF source changed")
        rel = runner.OUT + "/visual/" + event["symbol"] + "_FY" + str(event["fiscal_year"]) + ".png"
        target = runner.safe.local(root, rel)
        runner.require(not target.exists(), "Visual output is write-once")
        target.parent.mkdir(parents=True, exist_ok=True)
        doc = pdfium.PdfDocument(raw)
        page = doc[locator["page_1based"] - 1]
        bitmap = page.render(scale=1.6)
        image = bitmap.to_pil()
        try:
            image.save(target, format="PNG")
        finally:
            image.close()
            bitmap.close()
            page.close()
            doc.close()
        pages.append(dict(event_id=event["event_id"], source=source,
                          period=event["periods"]["current"], locator=locator,
                          image=dict(path=rel, sha256=runner.base.sha(root, rel)),
                          manual_visual_review_pending=True))
    record = dict(plan_sha256=args.plan_sha, pages=pages,
                  renderer=dict(path="work/render_stock_comp_source_v1.py",
                                sha256=runner.base.sha(root, "work/render_stock_comp_source_v1.py")))
    runner.base.publish(root, runner.OUT + "/VISUAL_PAGES.json", record)
    print(runner.safe.encoded(dict(pages=len(pages), output=runner.OUT + "/VISUAL_PAGES.json")).decode())


if __name__ == "__main__":
    main()
