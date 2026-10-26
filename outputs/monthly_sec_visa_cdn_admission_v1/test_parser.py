from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("visa_parser", Path(__file__).with_name("parser.py"))
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


def payload(quarter=3, year=2020):
    labels = {1: "First", 2: "Second", 3: "Third", 4: "Fourth"}
    ends = {1: "December 31", 2: "March 31", 3: "June 30", 4: "September 30"}
    dates = {1: f"January 28, {year}", 2: f"April 28, {year}",
             3: f"July 28, {year}", 4: f"October 28, {year}"}
    month = {1: 1, 2: 4, 3: 7, 4: 10}[quarter]
    cy = year - 1 if quarter == 1 else year
    title = f"Visa Inc. Reports Fiscal {labels[quarter]} Quarter {year} Results"
    cover = title + f"\nSan Francisco, CA, {dates[quarter]} – Visa Inc. (NYSE: V)\nUSD YoY Change"
    header = f"Three Months Ended\n{ends[quarter]},\n"
    years = f"{cy} {cy-1}"
    row = "$ 1,100 $ 1,000"
    if quarter != 1:
        header += {2: "Six", 3: "Nine", 4: "Twelve"}[quarter] + f" Months Ended\n{ends[quarter]},\n"
        years += f" {cy} {cy-1}"
        row += " $ 4,400 $ 4,000"
    table = ("Visa Inc. Consolidated Statements of Operations (unaudited)\n" + header + years +
             "\n(in millions, except per share data)\nNet revenues " + row + "\nOperating expenses 12 11")
    return {"engine": "pdfplumber", "version": "0.11.9", "total_pages": 3,
            "metadata": {"CreationDate": f"D:{year}{month:02}28010000-07'00'",
                         "ModDate": f"D:{year}{month:02}28080000-07'00'"},
            "pages": [{"page": 1, "text": cover}, {"page": 2, "text": table},
                      {"page": 3, "text": "Legal disclosures"}]}


def change(page, old, new, quarter=3):
    x = payload(quarter)
    x["pages"][page-1]["text"] = x["pages"][page-1]["text"].replace(old, new)
    return x


class ParserTests(unittest.TestCase):
    def test_three_and_nine_months_distinct(self):
        x = p.parse_pdf(payload())
        q, ytd = x["statement"]["periods"]
        self.assertEqual((q["current"]["start"], ytd["current"]["start"]), ("2020-04-01", "2019-10-01"))
        self.assertEqual((q["current"]["revenue_usd"], ytd["current"]["revenue_usd"]), (1100000000, 4400000000))
        self.assertEqual(ytd["current"]["duration_days"], 274)
        self.assertFalse(x["semantic_admission"])
        self.assertFalse(x["catalog_bound"])

    def test_q1_previous_calendar_year(self):
        x = p.parse_pdf(payload(1))["statement"]["periods"]
        self.assertEqual(len(x), 1)
        self.assertEqual((x[0]["current"]["start"], x[0]["current"]["end"]), ("2019-10-01", "2019-12-31"))

    def test_q2_six_months_crosses_year(self):
        x = p.parse_pdf(payload(2))["statement"]["periods"][1]
        self.assertEqual((x["months_printed"], x["current"]["duration_days"]), (6, 183))

    def test_q4_fiscal_year_not_calendar_year(self):
        x = p.parse_pdf(payload(4))["statement"]["periods"][1]
        self.assertEqual((x["current"]["start"], x["current"]["end"]), ("2019-10-01", "2020-09-30"))
        self.assertEqual(x["current"]["duration_days"], 366)

    def test_operating_revenue_label_and_no_dollars(self):
        x = change(2, "Net revenues $ 1,100 $ 1,000 $ 4,400 $ 4,000", "Net operating revenues 1100 1000 4400 4000")
        self.assertEqual(p.parse_pdf(x)["statement"]["periods"][0]["row_label"], "Net operating revenues")

    def test_wrong_ticker(self):
        with self.assertRaisesRegex(p.ParseError, "dateline"):
            p.parse_pdf(change(1, "NYSE: V", "NYSE: MA"))

    def test_ticker_substring(self):
        with self.assertRaisesRegex(p.ParseError, "dateline"):
            p.parse_pdf(change(1, "NYSE: V", "NYSE: VXX"))

    def test_wrong_issuer_heading(self):
        with self.assertRaisesRegex(p.ParseError, "heading"):
            p.parse_pdf(change(1, "Visa Inc. Reports", "Another Inc. Reports"))

    def test_ambiguous_release_date(self):
        x = payload(); x["pages"][0]["text"] += "\nSan Francisco, CA, July 29, 2020 – Visa Inc. (NYSE: V)"
        with self.assertRaisesRegex(p.ParseError, "dateline"):
            p.parse_pdf(x)

    def test_missing_currency(self):
        with self.assertRaisesRegex(p.ParseError, "usd_reporting"):
            p.parse_pdf(change(1, "USD", "EUR"))

    def test_release_before_report(self):
        with self.assertRaisesRegex(p.ParseError, "sequence"):
            p.parse_pdf(change(1, "July 28, 2020", "May 28, 2020"))

    def test_late_republication_not_original_quarter(self):
        with self.assertRaisesRegex(p.ParseError, "sequence"):
            p.parse_pdf(change(1, "July 28, 2020", "July 28, 2021"))

    def test_annual_document_not_accepted(self):
        with self.assertRaisesRegex(p.ParseError, "heading"):
            p.parse_pdf(change(1, "Third Quarter", "Fiscal Year"))

    def test_reverse_year_columns(self):
        with self.assertRaisesRegex(p.ParseError, "year_column_order"):
            p.parse_pdf(change(2, "2020 2019 2020 2019", "2019 2020 2019 2020"))

    def test_shifted_ytd_year_columns(self):
        with self.assertRaisesRegex(p.ParseError, "year_column_order"):
            p.parse_pdf(change(2, "2020 2019 2020 2019", "2020 2019 2019 2018"))

    def test_wrong_quarter_header(self):
        with self.assertRaisesRegex(p.ParseError, "quarter_ytd"):
            p.parse_pdf(change(2, "Nine Months", "Six Months"))

    def test_two_different_ends(self):
        with self.assertRaisesRegex(p.ParseError, "inconsistent_period"):
            p.parse_pdf(change(2, "Nine Months Ended\nJune 30", "Nine Months Ended\nSeptember 30"))

    def test_all_end_dates_disagree_with_release(self):
        with self.assertRaisesRegex(p.ParseError, "statement_report_end"):
            p.parse_pdf(change(2, "June 30", "September 30"))

    def test_thousands_wrong_scale(self):
        with self.assertRaisesRegex(p.ParseError, "statement_unit"):
            p.parse_pdf(change(2, "in millions", "in thousands"))

    def test_extra_revenue_column(self):
        with self.assertRaisesRegex(p.ParseError, "revenue_column_count"):
            p.parse_pdf(change(2, "$ 4,000", "$ 4,000 $ 3,900"))

    def test_malformed_comma(self):
        with self.assertRaisesRegex(p.ParseError, "invalid_amount"):
            p.parse_pdf(change(2, "1,100", "11,00"))

    def test_negative_amount_is_not_silently_absolute(self):
        with self.assertRaisesRegex(p.ParseError, "invalid_amount"):
            p.parse_pdf(change(2, "1,100", "(1,100)"))

    def test_percent_is_not_currency(self):
        with self.assertRaisesRegex(p.ParseError, "invalid_amount"):
            p.parse_pdf(change(2, "1,100", "1100%"))

    def test_duplicate_revenue_row(self):
        with self.assertRaisesRegex(p.ParseError, "unique_net_revenue"):
            p.parse_pdf(change(2, "Operating expenses", "Net revenues 5 4 8 7\nOperating expenses"))

    def test_gross_row_does_not_substitute(self):
        with self.assertRaisesRegex(p.ParseError, "unique_net_revenue"):
            p.parse_pdf(change(2, "Net revenues", "Gross revenues"))

    def test_duplicate_statement_page(self):
        x=payload();x["pages"][2]["text"]=x["pages"][1]["text"]
        with self.assertRaisesRegex(p.ParseError, "unique_statement_page"):
            p.parse_pdf(x)

    def test_page_order(self):
        x=payload();x["pages"][1]["page"]=3
        with self.assertRaisesRegex(p.ParseError, "page_order"):
            p.parse_pdf(x)

    def test_incomplete_pages(self):
        x=payload();x["pages"].pop()
        with self.assertRaisesRegex(p.ParseError, "document_extent"):
            p.parse_pdf(x)

    def test_future_pdf_modification(self):
        x=payload();x["metadata"]["ModDate"]="D:20200729080000-07'00'"
        with self.assertRaisesRegex(p.ParseError, "document_version_after"):
            p.parse_pdf(x)

    def test_metadata_not_publication_evidence(self):
        x=p.parse_pdf(payload())
        self.assertFalse(x["version_consistency"]["metadata_is_publication_evidence"])

    def test_compare_does_not_admit_or_relabel_tag(self):
        x=p.parse_pdf(payload()); q=x["statement"]["periods"][0]
        pair={"taxonomy":"us-gaap","unit":"USD","tag":"UnverifiedRevenueTag"}
        for side in ("current","prior"):
            pair[side]={k:q[side][k]for k in ("start","end","duration_days")}
            pair[side]["val"]=q[side]["revenue_usd"]
        r=p.compare_pair(pair,x)
        self.assertTrue(r["numeric_and_period_match"])
        self.assertFalse(r["semantic_admission"])
        pair["current"]["start"]="2019-10-01"
        self.assertFalse(p.compare_pair(pair,x)["numeric_and_period_match"])


class RowLayoutTests(unittest.TestCase):
    def rowwise(self, quarter=3):
        x = payload(quarter)
        ending = {2:"March 31",3:"June 30",4:"September 30"}[quarter]
        period = {2:"Six",3:"Nine",4:"Twelve"}[quarter]
        old = f"Three Months Ended\n{ending},\n{period} Months Ended\n{ending},"
        new = f"Three Months Ended {period} Months Ended\n{ending}, {ending},"
        x["pages"][1]["text"] = x["pages"][1]["text"].replace(old,new)
        return x

    def test_rowwise_matches_interleaved_all_quarters(self):
        for q in (2,3,4):
            with self.subTest(q=q):
                a=p.parse_pdf(self.rowwise(q))["statement"]
                b=p.parse_pdf(payload(q))["statement"]
                self.assertEqual(a["header_layout"],"rowwise")
                for aa,bb in zip(a["periods"],b["periods"]):
                    for side in ("current","prior"):
                        self.assertEqual(aa[side],bb[side])

    def test_rowwise_reversed_durations_rejected(self):
        x=self.rowwise();x["pages"][1]["text"]=x["pages"][1]["text"].replace(
            "Three Months Ended Nine Months Ended","Nine Months Ended Three Months Ended")
        with self.assertRaisesRegex(p.ParseError,"first_period"):
            p.parse_pdf(x)

    def test_rowwise_different_ends_rejected(self):
        x=self.rowwise();x["pages"][1]["text"]=x["pages"][1]["text"].replace(
            "June 30, June 30,","June 30, September 30,")
        with self.assertRaisesRegex(p.ParseError,"inconsistent_period"):
            p.parse_pdf(x)

    def test_rowwise_missing_end_rejected(self):
        x=self.rowwise();x["pages"][1]["text"]=x["pages"][1]["text"].replace(
            "June 30, June 30,","June 30,")
        with self.assertRaisesRegex(p.ParseError,"complete_period_header"):
            p.parse_pdf(x)

    def test_rowwise_missing_year_columns_rejected(self):
        x=self.rowwise();x["pages"][1]["text"]=x["pages"][1]["text"].replace(
            "2020 2019 2020 2019","2020 2019")
        with self.assertRaisesRegex(p.ParseError,"year_column_count"):
            p.parse_pdf(x)

    def test_extra_header_text_not_ignored(self):
        x=self.rowwise();x["pages"][1]["text"]=x["pages"][1]["text"].replace(
            "June 30, June 30,","June 30, June 30, OTHER PERIOD")
        with self.assertRaisesRegex(p.ParseError,"complete_period_header"):
            p.parse_pdf(x)

    def test_unit_word_split_is_recorded(self):
        x=self.rowwise(4);x["pages"][1]["text"]=x["pages"][1]["text"].replace("except","excep t")
        table=p.parse_pdf(x)["statement"]
        self.assertTrue(table["unit_word_split_normalized"])
        self.assertEqual(table["unit_phrase_raw"],"(in millions, excep t per share data)")
        self.assertEqual(table["periods"][1]["current"]["revenue_usd"],4400000000)

    def test_other_word_split_not_normalized(self):
        with self.assertRaisesRegex(p.ParseError,"statement_unit"):
            p.parse_pdf(change(2,"millions","mill ions"))

    def test_split_amount_not_normalized(self):
        with self.assertRaisesRegex(p.ParseError,"revenue_column_count"):
            p.parse_pdf(change(2,"1,100","1,1 00"))

    def test_duplicate_unit_phrase_rejected(self):
        with self.assertRaisesRegex(p.ParseError,"statement_unit"):
            p.parse_pdf(change(2,"(in millions, except per share data)",
                               "(in millions, except per share data) (in millions, excep t per share data)"))


class AdditionalValidation(unittest.TestCase):
    def test_full_year_non_q4_rejected(self):
        for q in (1,2,3):
            x=payload(q);x['pages'][0]['text']=x['pages'][0]['text'].replace(' Quarter ',' Quarter and Full-Year ')
            with self.assertRaisesRegex(p.ParseError,'full_year_requires'):p.parse_pdf(x)
    def test_bad_metadata_clock_rejected(self):
        x=payload();x['metadata']['ModDate']="D:20200728259900-07'00'"
        with self.assertRaisesRegex(p.ParseError,'metadata_clock'):p.parse_pdf(x)
    def test_bad_timezone_rejected(self):
        x=payload();x['metadata']['ModDate']="D:20200728080000+0060"
        with self.assertRaisesRegex(p.ParseError,'metadata_timezone'):p.parse_pdf(x)

if __name__ == "__main__":
    unittest.main()
