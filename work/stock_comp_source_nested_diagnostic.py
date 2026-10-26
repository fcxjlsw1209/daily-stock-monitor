"""Read-only structural diagnostic; never parses or emits monetary values."""
import json
from pathlib import Path
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
data = json.loads((ROOT / "outputs/monthly_stock_comp_source_v1/RAW_FACTS.json").read_bytes())
results = []
for event in data["events"]:
    value = event["stock_compensation"]["value"]
    if value is None or value["locator"]["format"] != "html":
        continue
    loc = value["locator"]
    soup = BeautifulSoup((ROOT / event["source"]["path"]).read_bytes(), "html.parser")
    for node in soup(["script", "style"]):
        node.decompose()
    table = soup.find_all("table")[loc["table_index_zero_based"]]
    rows = [row for row in table.find_all("tr") if row.find_parent("table") is table]
    nested = []
    for index, row in enumerate(rows):
        for cell_index, cell in enumerate(row.find_all(["th", "td"], recursive=False)):
            if cell.find("table") is not None:
                nested.append(dict(row=index, cell=cell_index,
                                   contains_target_row=index == loc["row_index_zero_based"],
                                   inside_NI_CFO=loc["indirect_start_row_zero_based"] <= index <= loc["operating_total_row_zero_based"]))
    if nested:
        results.append(dict(event_id=event["event_id"], locator=loc, nested=nested))
print(json.dumps(results, indent=2))
