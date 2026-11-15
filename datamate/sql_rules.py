"""Ground a small set of unambiguous sales queries in the loaded SQLite schema."""

import calendar
import re
from datetime import date
from contextlib import closing
from functools import lru_cache

from .data import open_readonly, quote
from .normalize import identifier_key, key


SQL_TOKEN = re.compile(r"""'(?:''|[^'])*'|"(?:""|[^"])*"|`(?:``|[^`])*`|\[[^\]]*\]|--[^\n]*|/\*[\s\S]*?\*/|/""")
PRODUCT_FIELDS = {"ten san pham", "ten hang", "ten hang hoa", "ten mat hang", "mat hang",
                  "ten mon", "ten hang ban", "san pham", "ten sp", "hang ban", "ten goi", "mon", "hang hoa"}
SALES_ROLES = {
    "order": ("ma don hang", "ma hoa don", "ma don", "ma bill", "order id", "don so", "ma giao dich"),
    "date": ("ngay dat hang", "ngay dat", "ngay ban", "thoi gian", "ngay lap", "ngay hd"),
    "status": ("trang thai don hang", "trang thai don", "trang thai", "tinh trang", "trang thai giao"),
    "product": ("ten san pham", "ten hang", "ten hang hoa", "ten mat hang", "mat hang", "hang ban", "ten sp",
                "mon", "ten goi", "hang hoa"),
    "amount": ("thanh tien", "tong tien"),
    "order_amount": ("tong tien hang",),
    "unit": ("don vi tinh", "dvt", "don vi", "donvi", "d vi", "dv"),
    "quantity": ("so luong san pham", "so luong", "sl ban", "soluong", "sl", "slg", "s l"),
    "customer": ("ten khach hang", "khach hang", "ten khach", "ten kh", "khach hang ten"),
    "customer_id": ("ma kh", "ma khach hang", "ma khach", "ma kh mua", "customer id"),
}
ROLE_NOTES = {
    "order": r"ma (?:cua )?(?:don(?: hang| dat hang)?|hoa don|bill)\b",
    "date": r"(?:ngay (?:ban|mua|dat(?: hang)?|lap don)|ngay gio dat hang)\b",
    "status": r"(?:trang thai|tinh trang) (?:cua )?(?:don(?: hang)?|hoa don)\b",
    "product": r"ten (?:cua )?(?:hang(?: hoa)?|san pham|mat hang|mon)\b",
    "amount": r"thanh tien\b",
    "unit": r"don vi tinh\b",
    "quantity": r"so luong(?: ban)?$",
    "customer": r"ten khach(?: hang)?\b",
    "customer_id": r"ma khach(?: hang)?\b",
    "item": r"ma (?:mat hang|hang(?: hoa)?|san pham)\b",
    "category": r"(?:nhom|danh muc) (?:hang|san pham|mat hang)\b",
    "price": r"don gia(?: ban)?$",
    "base_qty": r"so luong (?:da doi ra don vi co ban|quy doi|quy ve don vi co ban)\b",
    "method": r"hinh thuc thanh toan cua don\b",
    "channel": r"kenh ban\b",
    "balance": r"(?:tong (?:tien )?don|so khach phai tra cho ca don)\b",
    "payment_id": r"ma phieu thu\b",
    "paid": r"so tien (?:tra|thu)\b",
}


def real_percent_sql(sql):
    # Multiplying neither operand avoids changing unary or nested arithmetic precedence.
    return SQL_TOKEN.sub(lambda match: " / 1.0 / " if match[0] == "/" else match[0], sql)


def month_range(year, month):
    if not 1 <= month <= 12 or not 1 <= year <= 9999:
        return None
    return date(year, month, 1), date(year, month, calendar.monthrange(year, month)[1])


def month_to_date_periods(today):
    previous_year, previous_month = (today.year - 1, 12) if today.month == 1 else (today.year, today.month - 1)
    last_day = calendar.monthrange(previous_year, previous_month)[1]
    return today.replace(day=1), today, date(previous_year, previous_month, 1), date(
        previous_year, previous_month, min(today.day, last_day))


def is_mtd_comparison(text):
    return bool(re.search(r"\bcung ky (?:thang|th) truoc\b", text) and (
        re.search(r"\b(?:thang nay|th nay|dau thang|dau th)\b", text) or
        re.search(r"\b(?:hom nay|hnay|bua nay)\b", text)))


def expected_type(question):
    text = key(question)
    if re.search(r"\b(?:bao nhieu|bao nhiu|bnhiu|bn|may) (?:phan tram|%)\b|\b(?:ti le|ty le)\b", text) or "%" in question:
        return "percent"
    if re.search(r"\b(?:co )?(?:bao nhieu|bao nhiu|nhieu|bnhiu|bn|may) (?:mat hang|san pham|sp|khach|don|bill)\b", text):
        return "number"
    if re.search(r"\b(?:thang|ngay) nao\b", text) and not re.search(r"\b(?:nhung|cac) (?:thang|ngay) nao\b", text):
        return "date"
    if re.search(r"\b(?:co )?tang lien tuc (?:khong|hok|ko|k)\b", text):
        return "boolean"
    if re.search(r"\b(?:ai la (?:khach|nguoi|nhan vien)|khach nao|nguoi nao|nhan vien nao)\b", text):
        if re.search(r"\b(?:top\s*\d+|\d+ (?:khach|nguoi|nhan vien))\b", text):
            return "list"
        if re.search(r"\bnhat\b", text) and not re.search(r"\b(?:kem|va bao nhieu|cung voi)\b", text):
            return "string"
    return None


def sql_literal(value):
    return "'" + str(value).replace("'", "''") + "'"



@lru_cache(maxsize=96)
def column_values(path, table, column):
    with closing(open_readonly(path)) as con:
        values = con.execute(f"SELECT DISTINCT {quote(column)} FROM {quote(table)} "
                             f"WHERE typeof({quote(column)}) = 'text' LIMIT 1001").fetchall()
    return tuple((key(v), v) for v, in values if 4 <= len(key(v)) <= 150) if len(values) <= 1000 else ()


def product_hints(path, catalog, question):
    text = " " + key(question) + " "
    result = []
    for table in catalog["tables"]:
        for field in table["fields"]:
            name = identifier_key(field["name"])
            if name == "ten goi" and not cols_for(table)["product"]:
                continue
            if name not in PRODUCT_FIELDS and not re.match(ROLE_NOTES["product"], key(field.get("note"))):
                continue
            matched = [(table["name"], field["name"], value)
                       for normalized, value in column_values(path, table["name"], field["name"])
                       if " " + normalized + " " in text]
            result.extend(matched)
    return sorted(result, key=lambda r: (-len(key(r[2])), r))[:16]


def order_status(value):
    text = key(value)
    if re.search(r"\b(?:huy|cancel|cancelled|canceled)\b", text):
        return "canceled"
    if re.search(r"\b(?:khong giao duoc|giao (?:hang )?that bai|delivery failed|undelivered)\b", text):
        return "failed"
    if re.search(r"\b(?:hoan tra|hoan hang|doi tra|tra hang|tra lai|tra mot phan|hang tra ve|returned|refunded)\b", text) or text == "da tra":
        return "returned"
    return "other"


def known_customer(value):
    return int(key(value) not in {"", "le", "khach le", "khach vang lai", "vang lai", "khach khong luu", "walk in"})



def named_rollup(hints):
    matches = {key(value): value for _, _, value in hints if key(value)}
    longest = max((len(name) for name in matches), default=0)
    if not longest:
        return None
    candidates = [name for name in matches if len(name) == longest]
    if len(candidates) != 1 or any(not re.search(rf"\b{re.escape(name)}\b", candidates[0])
                                   for name in matches if name != candidates[0]):
        return None
    return candidates[0]


def role_column(table, role, aliases):
    fields = table["fields"]
    by_name = {key(field["name"]): field["name"] for field in fields}
    aliases_by_words = {identifier_key(field["name"]): field["name"] for field in fields}
    for alias in aliases:
        if alias in by_name or alias in aliases_by_words:
            return quote(by_name.get(alias) or aliases_by_words[alias])
    pattern = ROLE_NOTES.get(role)
    if pattern:
        matches = [field for field in fields if re.match(pattern, key(field.get("note")))]
        if len(matches) == 1:
            return quote(matches[0]["name"])
    return None


def cols_for(table):
    columns = {role: role_column(table, role, aliases) for role, aliases in SALES_ROLES.items()}
    if columns["product"] and identifier_key(columns["product"]) == "ten goi":
        if not role_column(table, "item", ("ma hang hoa", "ma hang", "ma san pham", "ma sku")):
            columns["product"] = None
    if columns["amount"] is None:
        columns["amount"] = role_column(table, "amount", ("thanh tien d", "amount", "tri gia"))
    return columns


def sales_query(q, catalog, hints):
    """Apply fixed rulebook arithmetic only for question shapes with an unambiguous SQL plan."""
    text = key(q["question"])
    mtd = is_mtd_comparison(text) and expected_type(q["question"]) == "percent"
    months = re.findall(r"\b(?:thang|th) (\d{1,2}) (?:nam )?(\d{4})\b", text)
    if not mtd and len(months) != 1:
        return None
    values = named_rollup(hints)
    revenue = bool(re.search(r"\b(?:doanh thu|dthu|tien ban|ban (?:duoc|dc)(?: tong cong)? (?:total )?(?:bao nhieu|bao nhiu|bnhiu|bn|may) tien)\b", text))
    top_customer = bool(expected_type(q["question"]) == "string" and
                        re.search(r"\bmua nhieu lan nhat\b", text))
    quantity = re.search(r"\b(?:bao nhieu|bao nhiu|bnhiu|bn|may) (kg|lon|chai|goi|hop|ly|phan|cai)\b", text)
    cancel_rate = bool(expected_type(q["question"]) == "percent" and re.search(r"\b(?:ti le|ty le) huy don\b", text))
    if mtd:
        if not revenue or values or re.search(r"\b(?:nhom|khach|chuyen khoan|tien mat|san pham|hang)\b", text):
            return None
    elif re.search(r"\b(?:nhom|hoa hong|loi nhuan|gia von|so sanh|so voi|cung ky|kho|nhap|kenh|tuan|tet|tra bang|ngay nao|kenh nao|bao nhieu don|may don|chuyen khoan|vi dien tu|COD)\b", text, re.I):
        return None
    elif not (top_customer or cancel_rate or (values and (revenue or quantity))):
        return None
    if (not mtd and not cancel_rate and not top_customer and not values) or (values and revenue and quantity):
        return None
    if not mtd and not cancel_rate and expected_type(q["question"]) not in (None, "string"):
        return None
    if not mtd:
        period = month_range(int(months[0][1]), int(months[0][0]))
        if period is None:
            return None
    candidates = []
    for table in catalog["tables"]:
        columns = cols_for(table)
        if not all(columns[name] for name in ("order", "date", "status")):
            continue
        order_table = columns["order_amount"] and not columns["product"]
        line_table = columns["amount"] and columns["product"]
        if not (order_table or line_table):
            continue
        if values and (not line_table or not any(t == table["name"] and quote(col) == columns["product"]
                                                      and key(v) == values for t, col, v in hints)):
            continue
        if top_customer and not columns["customer"]:
            continue
        if quantity and not all(columns[field] for field in ("quantity", "unit", "product")):
            continue
        candidates.append((table, columns, bool(order_table)))
    if len(candidates) != 1:
        return None

    table, c, order_table = candidates[0]
    source = "main." + quote(table["name"])
    qualifier = quote(table["name"])
    order, date_col, status = (c[name] for name in ("order", "date", "status"))
    amount = c["order_amount"] if order_table else c["amount"]
    dates = next((field.get("range") for field in table["fields"] if quote(field["name"]) == date_col), None)
    if not dates:
        return None
    canceled = f"vn_order_status({status}) = 'canceled'"
    invalid = f"vn_order_status({status}) IN ('canceled', 'returned', 'failed')"
    valid = f"{order} IS NOT NULL AND {order} NOT IN (SELECT {order} FROM {source} WHERE {order} IS NOT NULL AND {invalid})"
    def within(start, end):
        if end.isoformat() < dates[0] or start.isoformat() > dates[1]:
            return None
        return f"date({date_col}) BETWEEN '{start}' AND '{end}'"
    if mtd:
        current_start, current_end, previous_start, previous_end = month_to_date_periods(date.fromisoformat(q["as_of_date"]))
        current_where = within(current_start, current_end)
        previous_where = within(previous_start, previous_end)
        if not current_where or not previous_where:
            return None
        sql = (f"WITH totals AS (SELECT SUM(CASE WHEN {current_where} THEN {amount} ELSE 0 END) AS current,"
               f" SUM(CASE WHEN {previous_where} THEN {amount} ELSE 0 END) AS previous"
               f" FROM {source} WHERE ({current_where} OR {previous_where}) AND {valid})"
               " SELECT 100.0 * (current - previous) / NULLIF(previous, 0) FROM totals")
        return "percent", sql
    date_where = within(*period)
    if not date_where:
        return None
    if cancel_rate:
        return "percent", (f"SELECT 100.0 * COUNT(DISTINCT CASE WHEN {canceled} THEN {order} END)"
                           f" / NULLIF(COUNT(DISTINCT {order}), 0) FROM {source} WHERE {date_where}")
    if top_customer:
        customer = c["customer"]
        identity = c["customer_id"] or f"vn_key({customer})"
        return "string", (f"SELECT MIN({customer}) FROM {source} WHERE {date_where} AND {valid}"
                          f" AND vn_known_customer({customer}) AND trim({identity}) <> ''"
                          f" GROUP BY {identity} ORDER BY COUNT(DISTINCT {order}) DESC, {identity} LIMIT 1")
    product = c["product"]
    # The entity comes from actual source values, never from an answer file.
    product_where = f"vn_key({product}) = {sql_literal(values)}"
    if quantity:
        unit = sql_literal(quantity[1])
        u, qty = c["unit"], c["quantity"]
        factor = "NULL"
        if any(x["name"] == "unit_factors" for x in catalog["tables"]):
            factor = (f"(SELECT CASE WHEN COUNT(DISTINCT factor) = 1 THEN MAX(factor) END FROM main.unit_factors"
                      f" WHERE vn_key(product_name) = vn_key({qualifier}.{product})"
                      f" AND vn_key(unit) = vn_key({qualifier}.{u}) AND vn_key(base_unit) = {unit})")
        multiplier = f"CASE WHEN vn_key({qualifier}.{u}) = {unit} THEN 1 ELSE {factor} END"
        return "number", (f"WITH sales AS (SELECT {qty} AS qty, {multiplier} AS factor FROM {source}"
                          f" WHERE {date_where} AND {valid} AND {product_where})"
                          " SELECT CASE WHEN COUNT(*) = COUNT(qty * factor) THEN COALESCE(SUM(qty * factor), 0) END FROM sales")
    if revenue:
        return "number", (f"SELECT COALESCE(SUM({amount}), 0) FROM {source}"
                          f" WHERE {date_where} AND {valid} AND {product_where}")
    return None
