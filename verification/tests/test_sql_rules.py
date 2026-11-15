import json
import sqlite3
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import Mock

from datamate.business_sql import sales_totals
from datamate.inference import answer_kind, clean_generated_sql, execute_sql, infer
from datamate.periods import DatePredicateError, validate_date_predicates
from datamate.sql_rules import month_to_date_periods, product_hints, real_percent_sql, sales_query


class SalesRules(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.db = Path(self.temp.name) / "test.sqlite"
        columns = ("Mã đơn hàng", "Ngày đặt hàng", "Trạng thái đơn hàng", "Tên khách hàng",
                   "Tên sản phẩm", "Số lượng sản phẩm", "Đơn vị tính", "Tổng tiền")
        with sqlite3.connect(self.db) as con:
            con.execute('CREATE TABLE sales (' + ', '.join(f'"{c}"' for c in columns) + ')')
            con.executemany('INSERT INTO sales VALUES (?,?,?,?,?,?,?,?)', [
                ("a", "2025-11-01", "Hoàn thành", "Khách lẻ", "Mặt hàng thử", 1, "kg", 100),
                ("b", "2025-11-30", "Hoàn thành", "Khách lẻ", "Khác", 1, "kg", 900),
                ("c", "2025-12-02", "Hoàn thành", "Lan", "Mặt hàng thử", 1, "túi", 99),
                ("d", "2025-12-02", "Hoàn thành", "Khách lẻ", "Khác", 1, "kg", 1),
                ("e", "2025-12-02", "Hoàn thành", "Bình", "Mặt hàng thử", 1, "kg", 500),
                ("e", "2025-12-03", "Trả một phần", "Bình", "Khác", 1, "kg", 10),
                ("f", "2025-12-03", "Đã hủy", "Bình", "Mặt hàng thử", 1, "kg", 600),
            ])
            con.execute("CREATE TABLE unit_factors(product_name, unit, base_unit, factor)")
            con.execute("INSERT INTO unit_factors VALUES ('Mặt hàng thử', 'túi', 'kg', 3)")
        self.catalog = {"tables": [
            {"name": "sales", "fields": [
                {"name": c, **({"range": ["2025-11-01", "2025-12-03"]} if c == columns[1] else {})}
                for c in columns
            ]},
            {"name": "unit_factors", "fields": [{"name": c} for c in ("product_name", "unit", "base_unit", "factor")]},
        ]}

    def solve(self, question):
        q = {"id": "fixture", "question": question, "as_of_date": "2025-12-03"}
        hints = product_hints(self.db, self.catalog, question)
        kind, sql = sales_query(q, self.catalog, hints)
        return kind, execute_sql(self.db, sql)["rows"]

    def model_fixture(self):
        for table in self.catalog["tables"]:
            table["sources"] = []
            table["rows"] = 7
            for field in table["fields"]:
                field.update(type="TEXT", kind="text", note="")
        self.catalog["joins"] = []
        context = Mock()
        context.dates.return_value = "Hôm nay 2025-12-03."
        context.rule_excerpt.return_value = "Dùng quy tắc của bộ dữ liệu."
        return context, Mock()

    def test_month_to_date_and_cancel_rate(self):
        self.assertEqual(month_to_date_periods(date(2024, 3, 31))[-1], date(2024, 2, 29))
        self.assertEqual(self.solve("Doanh thu từ đầu tháng này tăng bao nhiêu % so với cùng kỳ tháng trước?"),
                         ("percent", [[0.0]]))
        kind, rows = self.solve("Tỉ lệ hủy đơn tháng 12/2025 bao nhiêu %?")
        self.assertEqual(kind, "percent")
        self.assertEqual(rows, [[25.0]])

    def test_grounded_product_quantity_and_known_customers(self):
        self.assertEqual(self.solve("Tiền bán Mat hang thu tháng 12/2025 là bao nhiêu?"), ("number", [[99]]))
        self.assertEqual(self.solve("Mặt hàng thử tháng 12/2025 bán được bao nhiêu kg?"), ("number", [[3]]))
        self.assertEqual(self.solve("Tháng 12/2025 ai là khách mua nhiều lần nhất?"), ("string", [["Lan"]]))

    def test_percent_rewrite_does_not_touch_literals_comments(self):
        query = """SELECT ((99 - 100) / 100) * 100, '2026/01' -- leave / alone"""
        changed = real_percent_sql(query)
        self.assertEqual(execute_sql(self.db, changed)["rows"], [[-1.0, "2026/01"]])
        self.assertTrue(changed.endswith("-- leave / alone"))

    def test_output_shape_can_repair_or_trigger_retry(self):
        q = {"question": "Tháng 12/2025 ai là khách mua nhiều lần nhất?"}
        self.assertEqual(answer_kind(q, "table", {"columns": ["name"], "rows": [["Lan"]]}), "string")
        with self.assertRaisesRegex(ValueError, "requires string"):
            answer_kind(q, "table", {"columns": ["name", "count"], "rows": [["Lan", 1]]})
        q = {"question": "Có bao nhiêu sản phẩm có mặt trong ít nhất 145 bill?"}
        self.assertEqual(answer_kind(q, "table", {"columns": ["count"], "rows": [[11]]}), "number")
        with self.assertRaisesRegex(ValueError, "requires number"):
            answer_kind(q, "table", {"columns": ["product"], "rows": [["A"], ["B"]]})

    def test_month_filter_uses_date_column_and_entire_month(self):
        q = {"question": "Tổng doanh thu tháng 12/2025 là bao nhiêu?", "as_of_date": "2025-12-03"}
        for predicate in (
                """date(h."Ngày đặt hàng") = '2025-12'""",
                """DATE(h."Mã đơn hàng") = '2025-12-01'""",
                """date(h."Ngày đặt hàng") = '2025-12-01'""",
                """h."Ngày đặt hàng" = '2025-12-01'""",
        ):
            with self.subTest(predicate=predicate), self.assertRaises(DatePredicateError):
                validate_date_predicates(q, "SELECT 1 FROM sales h WHERE " + predicate, self.catalog)
        validate_date_predicates(
            q, """SELECT 1 FROM sales h WHERE h."Ngày đặt hàng" >= '2025-12-01'
                  AND h."Ngày đặt hàng" < '2026-01-01' -- date(h."Mã đơn hàng")""", self.catalog)
        validate_date_predicates(
            {**q, "question": "Doanh thu ngày đầu tháng 12/2025 là bao nhiêu?"},
            """SELECT 1 FROM sales h WHERE date(h."Ngày đặt hàng") = '2025-12-01'""", self.catalog)

    def test_unanswerable_requires_evidence_or_repairs_then_abstains(self):
        context, client = self.model_fixture()
        self.catalog["tables"][0]["fields"][1]["timestamp_examples"] = ["2025-12-02 08:30:00"]
        q = {"id": "fixture", "question": "Ca sáng có tổng cộng bao nhiêu đơn trong tháng 12/2025?",
             "as_of_date": "2025-12-03"}
        claim = ('{"type":"unanswerable","reason":"Không có timestamp"}', None, None)
        response = (json.dumps({"type": "number", "sql": "SELECT 1"}), None, None)
        client.complete.side_effect = [claim, response]
        answer, log = infer(q, self.db, self.catalog, "", context, client)
        self.assertEqual(answer, {"id": "fixture", "type": "number", "answer": 1})
        self.assertIn("unanswerable_rejected", log["attempts"][0])
        second_prompt = client.complete.call_args_list[1].args[1]
        self.assertIn("2025-12-02 08:30:00", second_prompt)
        self.assertIn("2025-11-01 đến 2025-12-03", second_prompt)
        client.complete.side_effect = [claim, claim]
        answer, log = infer(q, self.db, self.catalog, "", context, client)
        self.assertEqual(answer, {"id": "fixture", "type": "abstain"})
        self.assertIn("Chưa có bằng chứng", log["abstain_reason"]["detail"])
        client.reset_mock()
        answer, log = infer({**q, "question": "Doanh thu tháng 10/2025 là bao nhiêu?"},
                            self.db, self.catalog, "", context, client)
        self.assertEqual(answer["type"], "unanswerable")
        self.assertEqual(log["attempts"][0]["source"], "catalog_evidence")
        client.complete.assert_not_called()

    def test_sql_repair_receives_exact_identifiers(self):
        context, client = self.model_fixture()
        q = {"id": "fixture", "question": "Có bao nhiêu trạng thái?", "as_of_date": "2025-12-03"}
        client.complete.side_effect = [
            (json.dumps({"type": "number", "sql": "SELECT COUNT(TRANGTHAITD) FROM sales"}), None, None),
            (json.dumps({"type": "number", "sql": 'SELECT COUNT(DISTINCT "Trạng thái đơn hàng") FROM sales'}), None, None),
        ]
        answer, log = infer(q, self.db, self.catalog, "", context, client)
        self.assertEqual(answer["answer"], 3)
        self.assertIn("no such column", log["attempts"][0]["error"])
        repair = client.complete.call_args_list[1].args[1]
        self.assertIn('Danh sách định danh SQLite chính xác để sửa SQL:\n"sales": "Mã đơn hàng"', repair)

    def test_generated_sql_removes_only_trailing_wrapper_brace(self):
        context, client = self.model_fixture()
        q = {"id": "fixture", "question": "Có bao nhiêu trạng thái?", "as_of_date": "2025-12-03"}
        sql = 'SELECT COUNT(DISTINCT "Trạng thái đơn hàng") FROM sales}'
        client.complete.return_value = (json.dumps({"type": "number", "sql": sql}), None, None)
        answer, log = infer(q, self.db, self.catalog, "", context, client)
        self.assertEqual(answer["answer"], 3)
        self.assertEqual(log["attempts"][0]["generated_sql"], sql)
        self.assertEqual(log["code"], sql[:-1])
        client.complete.assert_called_once()
        for valid_or_incomplete_sql in ("SELECT '}'", "SELECT 1 -- }", "SELECT 1 /* }", "SELECT 'x}"):
            with self.subTest(sql=valid_or_incomplete_sql):
                self.assertEqual(clean_generated_sql(valid_or_incomplete_sql), valid_or_incomplete_sql)

    def test_readme_roles_average_sums_per_order_and_does_not_duplicate_lines(self):
        with sqlite3.connect(self.db) as con:
            con.executescript("""
                CREATE TABLE headers ("Ref" TEXT, "Booked" TEXT, "State" TEXT, "Due" NUMERIC);
                INSERT INTO headers VALUES ('a','2025-12-02','xong',500),
                    ('a','2025-12-02','xong',500), ('b','2025-12-02','xong',400),
                    ('c','2025-12-02','Hủy',900);
                CREATE TABLE detail ("Invoice" TEXT, "ID" TEXT, "Net" NUMERIC);
                INSERT INTO detail VALUES ('a','p',100), ('a','q',300), ('b','q',200), ('c','q',900);
            """)
        catalog = {"tables": [
            {"name": "headers", "fields": [
                {"name": "Ref", "note": "mã đơn"},
                {"name": "Booked", "note": "ngày bán", "range": ["2025-12-02", "2025-12-02"]},
                {"name": "State", "note": "trạng thái đơn"},
                {"name": "Due", "note": "tổng tiền đơn"},
            ]},
            {"name": "detail", "fields": [
                {"name": "Invoice", "note": "mã đơn"},
                {"name": "ID", "note": "mã hàng"},
                {"name": "Net", "note": "thành tiền sau giảm"},
            ]},
        ], "joins": [{"left": ["headers", "Ref"], "right": ["detail", "Invoice"]}]}
        q = {"id": "fixture", "question": "Hôm qua giá trị trung bình một đơn là bao nhiêu?",
             "as_of_date": "2025-12-03"}
        row, _ = infer(q, self.db, catalog, "", None, None)
        self.assertEqual(row, {"id": "fixture", "type": "number", "answer": 300})
        self.assertIsNone(sales_totals(q, {**catalog, "joins": []}, []))

    def test_readme_roles_support_sales_lines_without_order_ids(self):
        with sqlite3.connect(self.db) as con:
            con.executescript("""
                CREATE TABLE shop ("Day" TEXT, "State" TEXT, "Item" TEXT, "Net" NUMERIC);
                INSERT INTO shop VALUES ('2025-12-02','xong','A',500),
                    ('2025-12-02','trả hàng','B',400), ('2025-12-03','xong','A',900);
            """)
        catalog = {"tables": [{"name": "shop", "fields": [
            {"name": "Day", "note": "ngày bán", "kind": "date", "range": ["2025-12-02", "2025-12-03"]},
            {"name": "State", "note": "trạng thái đơn"},
            {"name": "Item", "note": "tên hàng"},
            {"name": "Net", "note": "thành tiền sau giảm"},
        ]}]}
        q = {"id": "fixture", "question": "Hôm qua doanh thu bao nhiêu?", "as_of_date": "2025-12-03"}
        row, _ = infer(q, self.db, catalog, "", None, None)
        self.assertEqual(row["answer"], 500)
        self.assertIsNone(sales_totals({**q, "question": "Hôm qua có bao nhiêu đơn?"}, catalog, []))
        with self.assertRaisesRegex(DatePredicateError, "text literal"):
            validate_date_predicates(q, "SELECT SUM(Net) FROM shop WHERE date('Day') = '2025-12-02'", catalog)
        validate_date_predicates(q, """SELECT SUM(Net) FROM shop WHERE date("Day") = date('2025-12-02')""", catalog)

    def test_rule_result_is_identical_when_logprob_enabled(self):
        q = {"id": "fixture", "question": "Tiền bán Mat hang thu tháng 12/2025 là bao nhiêu?",
             "as_of_date": "2025-12-03"}
        for log_confidence in (False, True):
            with self.subTest(log_confidence=log_confidence):
                row, log = infer(q, self.db, self.catalog, "", None, None, log_confidence=log_confidence)
                self.assertEqual(row["answer"], 99)
                self.assertEqual(log["attempts"][0]["source"], "rulebook_sql")
                self.assertIsNone(log["confidence"])
