"""Verify supported data gaps from catalog evidence before returning unanswerable."""

import re

from .business_sql import date_range, has_roles, order_scope, scoped_tables
from .normalize import key
from .periods import explicit_days, question_period
from .sql_rules import is_mtd_comparison


def sales_only(catalog):
    order = order_scope(catalog)
    if order is None or not order[1]["product"]:
        return False
    other_tables = [t for t in catalog["tables"] if t["fields"] is not order[2]]
    conversion_fields = {"ten san pham", "ten hang", "don vi quy doi", "ti le quy doi",
                         "don vi co ban", "product name", "unit", "base unit", "factor"}
    return all({key(f["name"]) for f in t["fields"]} <= conversion_fields for t in other_tables)


def verified_gap(q, catalog):
    text = key(q["question"])
    tables = list(scoped_tables(catalog))
    if not tables:
        return None
    # Only prove gaps from a catalog whose observed sources support this conclusion.
    if sales_only(catalog):
        if not any(has_roles(t, "stock") for t in tables) and re.search(r"\b(?:ton kho|hang con trong kho|gia tri hang ton)\b", text):
            return "Nguồn chỉ có giao dịch bán và bảng quy đổi, thiếu tồn kho hoặc dữ liệu để suy ra tồn kho."
        payment_fields = {"khach da tra", "con no", "tien no", "so tien no", "da thanh toan", "tien da thanh toan"}
        if not any(key(f["name"]) in payment_fields for t in tables for f in t[2]) and re.search(
                r"\b(?:khach con thieu|khach con no|cong no)\b", text):
            return "Nguồn chỉ có giao dịch bán và bảng quy đổi, thiếu khoản phải thu và giao dịch thanh toán."

    order = order_scope(catalog)
    coverage = date_range(order) if order else None
    sales_question = re.search(r"\b(?:doanh thu|dthu|ban (?:ra|duoc|dc)|so don)\b", text)
    complex_period = (is_mtd_comparison(text) or len(explicit_days(q["question"])) >= 1
                      or re.search(r"\b(?:tu ngay|den het|trong .* ngay|truoc|sau|nam nay)\b", text))
    if (coverage and sales_question and not complex_period
            and re.search(r"\bthang \d{1,2} (?:nam )?\d{4}\b", text)
            and not re.search(r"\b(?:nhap|ton|kho|no|thu no|khuyen mai)\b", text)):
        period = question_period(q["question"], q["as_of_date"])
        if period and (period[1].isoformat() < coverage[0] or period[0].isoformat() > coverage[1]):
            return f"Kỳ hỏi {period[0]} đến {period[1]} nằm hoàn toàn ngoài dữ liệu bán hàng {coverage[0]} đến {coverage[1]}."
    return None


def evidence_for_retry(q, catalog):
    order = order_scope(catalog)
    ranges = []
    if order and date_range(order):
        start, end = date_range(order)
        ranges.append(f"Ngày bán có dữ liệu {start} đến {end}.")
    timestamps = []
    for table in catalog.get("tables", ()):
        for field in table["fields"]:
            if field.get("timestamp_examples"):
                timestamps.append(f'{table["name"]}.{field["name"]}: {field["timestamp_examples"][0]}')
    if timestamps:
        ranges.append("Timestamp đã quan sát: " + "; ".join(timestamps[:5]) + ".")
    ranges.append("Ngày, giá và sự kiện nêu sẵn trong câu hỏi có thể dùng làm điều kiện.")
    return " ".join(ranges)
