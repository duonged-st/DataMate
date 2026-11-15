"""Compute well-specified sales and accounting queries from catalog roles."""

import re
from datetime import date, timedelta

from .data import quote
from .normalize import key
from .periods import explicit_days, question_period
from .sql_rules import cols_for, expected_type, named_rollup, role_column, sql_literal

DETAIL = {
    "item": ("ma hang", "ma san pham", "ma sp", "ma hang hoa", "ma sku", "ma hh", "ma mon", "id sp", "ma h hoa"),
    "product": ("ten san pham", "ten hang", "ten hang hoa", "ten mat hang"),
    "category": ("nhom hang 3 cap", "nhom hang", "nhom san pham", "nhom mat hang", "danh muc", "phan loai"),
    "price": ("don gia", "gia san pham", "gia don vi", "don gia d"),
    "qty": ("so luong", "so luong san pham", "sl", "slg"),
    "base_qty": ("so luong quy doi", "so luong co ban", "sl quy doi", "sl co ban", "sl dv co ban", "so luong le"),
    "unit": ("don vi tinh", "dvt", "don vi"),
    "method": ("phuong thuc thanh toan", "hinh thuc thanh toan", "phuong thuc tt", "tra bang"),
    "channel": ("kenh ban", "kenh ban hang", "kenh", "nguon don"),
    "timestamp": ("thoi gian tao", "gio tao don", "gio tao", "ngay gio dat hang", "thoi gian lap", "thoi gian"),
    "balance": ("tong sau giam gia", "tong thanh toan", "tong don"),
    "payment_id": ("ma phieu", "ma phieu thu", "phieu thu", "ma thanh toan", "ma gd"),
    "paid": ("gia tri", "so tien thu", "so tien thanh toan", "tien thu"),
    "payment_date": ("ngay gio thu", "thoi gian thu", "ngay thu", "ngay tt", "thoi gian", "ngay"),
    "stock": ("ton cuoi ky", "so luong ton", "ton kho", "ton cuoi"),
    "ingredient": ("hoat chat", "thanh phan chinh"),
    "prescription": ("thuoc ke don", "ke don", "ban theo don"),
    "employee": ("ma nhan vien", "ma nv ban", "ma nv"),
}
ORDER_COUNT = r"(?:don(?: hang)?|hoa don|bill)"
COUNT_REQUEST = r"(?:bao nhieu|bao nhiu|nhieu|bnhiu|bn|may)"
SALES_WORDS = r"(?:doanh thu|dthu|tien ban|ban (?:duoc|dc)(?: tong cong)? (?:total )?(?:bao nhieu|bao nhiu|bnhiu|bn|may) tien)"
CASH_QUESTION = re.compile(rf"\b{SALES_WORDS}\b|\b(?:ban|thu) (?:dc|duoc) (?:tong cong )?(?:bn|bao nhiu|bao nhieu) tien\b")
COUNT_THRESHOLD = re.compile(r"\b(?:it nhat|tu[wf]*) (\d{1,5}) " + ORDER_COUNT + r"\b")
GROUP_SHARE = re.compile(r"\bnhom (?:hang )?(.+?) (?:chiem|dong gop|la bao nhieu)\b")
PER_ORDER_AVERAGE = re.compile(r"\b(?:binh quan|trung binh)\s+(?:moi|mot|tren(?: mot)?) " + ORDER_COUNT + r"\b")
CANCEL_RATE = re.compile(r"\b(?:ti le|ty le) (?:don (?:bi )?)?huy\b|\b(?:don (?:bi )?huy|huy don)\b")


def scoped_tables(catalog):
    for table in catalog.get("tables", ()):
        columns = {**cols_for(table), **{
            role: role_column(table, role, aliases)
            for role, aliases in DETAIL.items() if role not in {"product", "unit"}
        }}
        columns["line_amount"] = role_column(table, "amount", ("thanh tien", "thanh tien d", "amount", "tri gia"))
        if columns["payment_id"] and columns["paid"] and not columns["status"]:
            columns["date"] = columns["date"] or columns["payment_date"]
        if not columns["balance"] and not columns["product"] and columns["method"]:
            columns["balance"] = role_column(table, "balance", ("tong tien",))
        yield "main." + quote(table["name"]), columns, table["fields"]


def has_roles(table, *roles):
    return all(table[1].get(role) for role in roles)


def order_scope(catalog):
    candidates = [t for t in scoped_tables(catalog) if has_roles(t, "order", "date", "status")
                  and (t[1]["product"] or t[1]["order_amount"] or t[1]["balance"])]
    if len(candidates) != 1:
        return None
    return candidates[0]


def date_range(table, role="date"):
    field = table[1].get(role)
    return next((f.get("range") for f in table[2] if quote(f["name"]) == field), None)


def inclusive_predicate(column, start, end):
    return f"date({column}) BETWEEN '{start}' AND '{end}'"


def validity(table, alias="h", include_returns=False):
    source, c, _ = table
    denied = "('canceled','failed')" if include_returns else "('canceled','returned','failed')"
    if c["product"]:
        return (f"{alias}.{c['order']} IS NOT NULL AND {alias}.{c['order']} NOT IN"
                f" (SELECT invalid.{c['order']} FROM {source} invalid"
                f" WHERE invalid.{c['order']} IS NOT NULL AND vn_order_status(invalid.{c['status']}) IN {denied})")
    return f"{alias}.{c['order']} IS NOT NULL AND vn_order_status({alias}.{c['status']}) NOT IN {denied}"


def joined_lines(catalog, orders):
    header, hc, _ = orders
    if hc["product"] and hc["amount"]:
        return orders, "h", f"{header} h"
    line_tables = [t for t in scoped_tables(catalog) if t[0] != header
                   and has_roles(t, "order", "line_amount") and (t[1]["product"] or t[1]["item"])
                   and linked(catalog, orders, "order", t, "order")]
    if len(line_tables) != 1:
        return None
    detail = line_tables[0]
    return detail, "d", f"{header} h JOIN {detail[0]} d ON d.{detail[1]['order']} = h.{hc['order']}"


def linked(catalog, a, a_role, b, b_role):
    expected = {(a[0], a[1][a_role]), (b[0], b[1][b_role])}
    return any({("main." + quote(table), quote(column)) for table, column in (link["left"], link["right"])}
               == expected for link in catalog.get("joins", ()))


def count_intent(text, noun):
    return bool(re.search(rf"\b{COUNT_REQUEST} (?:{noun})\b", text))


def cancellation_rate(q, catalog):
    text = key(q["question"])
    period = question_period(q["question"], q["as_of_date"])
    percent = expected_type(q["question"]) == "percent"
    count = bool(re.search(r"\b(?:dem|so don|bao nhieu don|may don)\b", text))
    if (not period or not (percent or count) or not CANCEL_RATE.search(text)
            or re.search(r"\b(?:so voi|so vs|cung ky|kenh|theo|loai tru|nhom|khach|nhan vien)\b", text)):
        return None
    order = order_scope(catalog)
    coverage = date_range(order) if order else None
    if not coverage or period[1].isoformat() < coverage[0] or period[0].isoformat() > coverage[1]:
        return None
    source, hc, fields = order
    statuses = next(f.get("examples", []) for f in fields if quote(f["name"]) == hc["status"])
    has_explicit_cancel = any(re.search(r"\b(?:huy|cancel|canceled|cancelled)\b", key(s)) for s in statuses)
    denied = "('canceled')" if has_explicit_cancel or not statuses else "('canceled','failed')"
    oid = hc["order"]
    numerator = f"COUNT(DISTINCT CASE WHEN vn_order_status({hc['status']}) IN {denied} THEN {oid} END)"
    result = f"100.0 * {numerator} / NULLIF(COUNT(DISTINCT {oid}),0)" if percent else numerator
    return "percent" if percent else "number", f"SELECT {result} FROM {source} WHERE {inclusive_predicate(hc['date'], *period)}"


def best_sellers(q, catalog, products):
    text = key(q["question"])
    period = question_period(q["question"], q["as_of_date"])
    if (not period or products
            or not re.search(r"\b(?:ban chay nhat|ban (?:duoc |dc )?(?:nhieu|nhiu) nhat|hot nhat|hang nao chay nhat)\b", text)
            or not re.search(r"\b(?:mat hang|san pham|mon|hang|top)\b", text)
            or re.search(r"\b(?:nhom|kenh|theo|cung voi|kem|doanh thu|dthu|so don|bao nhieu (?:tien|kg|ly|goi|chai))\b", text)):
        return None
    requested = re.search(r"\b(?:top\s*(\d{1,2})|(\d{1,2}) (?:mat hang|san pham|mon|hang))\b", text)
    limit = int(requested[1] or requested[2]) if requested else 1
    if not 1 <= limit <= 25:
        return None
    order = order_scope(catalog)
    if order is None:
        standalone = [t for t in scoped_tables(catalog) if has_roles(t, "date", "status", "product", "line_amount", "base_qty")
                      and not t[1]["order"]]
        if len(standalone) != 1:
            return None
        order = standalone[0]
    coverage = date_range(order)
    if not coverage or period[1].isoformat() < coverage[0] or period[0].isoformat() > coverage[1]:
        return None
    lines = joined_lines(catalog, order)
    if not lines:
        return None
    detail, alias, joined = lines
    dc = detail[1]
    _, hc, _ = order
    if not dc["base_qty"]:
        return None
    name = f"{alias}.{dc['product']}" if dc["product"] else None
    if name is None:
        dimensions = [t for t in scoped_tables(catalog) if has_roles(t, "item", "product") and not t[1]["order"]
                      and not t[1]["stock"]
                      and dc["item"] and linked(catalog, detail, "item", t, "item")]
        if len(dimensions) != 1:
            return None
        dimension, attrs, _ = dimensions[0]
        name = "p.name"
        joined += (f" JOIN (SELECT {attrs['item']} AS item, MIN({attrs['product']}) AS name FROM {dimension}"
                   f" GROUP BY {attrs['item']} HAVING COUNT(DISTINCT vn_key({attrs['product']}))=1) p"
                   f" ON p.item={alias}.{dc['item']}")
    validity_filter = validity(order) if hc["order"] else f"vn_order_status(h.{hc['status']})='other'"
    qty = f"{alias}.{dc['base_qty']}"
    return "list" if requested else "string", (
        f"WITH items AS (SELECT {name} AS name, {qty} AS qty FROM {joined}"
        f" WHERE {inclusive_predicate('h.' + hc['date'], *period)}"
        f" AND {validity_filter} AND vn_key({name}) NOT IN ('','phi phuc vu')),"
        f" ranked AS (SELECT MIN(name) AS name, SUM(qty) AS quantity FROM items GROUP BY vn_key(name))"
        f" SELECT CASE WHEN EXISTS (SELECT 1 FROM items WHERE qty IS NULL) THEN NULL ELSE name END"
        f" FROM ranked ORDER BY quantity DESC, vn_key(name) LIMIT {limit}")


def filtered_revenue(q, catalog, products):
    """Use linked receipt and item tables for explicitly requested sales filters."""
    text = key(q["question"])
    period = question_period(q["question"], q["as_of_date"])
    if (not period or not CASH_QUESTION.search(text) or products or expected_type(q["question"]) not in (None, "number")
            or re.search(r"\b(?:so voi|so vs|cung ky|binh quan|trung binh|tang|giam|chia|loi nhuan|von|tung|theo|nhieu nhat|nhap|kho)\b", text)):
        return None
    employee = re.search(r"\bnhan vien ma ([a-z0-9]+)\b", text)
    shift = re.search(r"\bca (sang|chieu)\b", text)
    if re.search(r"\bca sang\b.*\bca chieu\b|\bca chieu\b.*\bca sang\b", text):
        return None
    ingredient = re.search(r"\bhoat chat (.+?)(?=\b(?:thang|hom|tuan|quy|ban|doanh thu|la|co|duoc|bao)\b|$)", text)
    if ingredient and not re.fullmatch(r"[a-z0-9 ]{2,50}", ingredient[1].strip()):
        return None
    non_prescription = bool(re.search(r"\b(?:khong can don bac si|khong ke don)\b", text))
    prescription = non_prescription or bool(re.search(r"\bthuoc ke don\b", text))
    category = re.search(r"\bnhom (?:hang )?(.+?)(?=\b(?:thang|hom|tuan|quy|ban|bao|thu dc|duoc|la)\b|$)", text)
    if category and not re.search(r"\b(?:nhom khach|khach nhom|nhom si|nhom than thiet|nhom le)\b", text):
        category = category[1].strip()
    else:
        category = None
    if sum(bool(selector) for selector in (employee, shift, ingredient, prescription, category)) != 1:
        return None
    order = order_scope(catalog)
    coverage = date_range(order) if order else None
    if not coverage or period[1].isoformat() < coverage[0] or period[0].isoformat() > coverage[1]:
        return None
    lines = joined_lines(catalog, order)
    if not lines:
        return None
    _, columns, fields = order
    detail, alias, joined = lines
    dc = detail[1]
    amount = f"{alias}.{dc['line_amount'] or dc['amount']}"
    where = f"{inclusive_predicate('h.' + columns['date'], *period)} AND {validity(order)}"
    if employee and columns["employee"]:
        where += f" AND vn_key(h.{columns['employee']}) = {sql_literal(employee[1])}"
    elif shift and columns["timestamp"] and any(f.get("timestamp_examples") and quote(f["name"]) == columns["timestamp"] for f in fields):
        hours = ("06:00:00", "14:00:00") if shift[1] == "sang" else ("14:00:00", "22:00:00")
        clock = f"time(h.{columns['timestamp']})"
        where += f" AND {clock} >= '{hours[0]}' AND {clock} < '{hours[1]}'"
    elif ingredient or prescription or category:
        role = "ingredient" if ingredient else "prescription" if prescription else "category"
        candidates = [t for t in scoped_tables(catalog) if has_roles(t, "item", role) and not t[1]["order"]
                      and dc["item"] and linked(catalog, detail, "item", t, "item")]
        if len(candidates) != 1:
            return None
        dimension, dd, _ = candidates[0]
        field = f"attr.{dd[role]}"
        if ingredient:
            constraint = f"vn_key({field}) = {sql_literal(ingredient[1].strip())}"
        elif prescription:
            values = "('0','false','khong','no')" if non_prescription else "('1','true','co','yes')"
            constraint = f"vn_key(CAST({field} AS TEXT)) IN {values}"
        else:
            constraint = f"vn_key({field}) = {sql_literal(category)}"
        where += (f" AND EXISTS (SELECT 1 FROM {dimension} attr WHERE attr.{dd['item']} = {alias}.{dc['item']}"
                  f" AND {constraint})")
    else:
        return None
    return "number", f"SELECT SUM({amount}) FROM {joined} WHERE {where}"


def sales_totals(q, catalog, products):
    """Use a unique transaction source and sum at line grain when available."""
    text = key(q["question"])
    period = question_period(q["question"], q["as_of_date"])
    day_request = bool(re.search(r"\bngay nao[f]?\b", text))
    peak_day = bool(day_request
                    and re.search(r"\b(?:cao nha{1,2}t[s]?|nhieu tien nhat|nhiu tien nhat|ban dat nhat)\b", text)
                    and re.search(r"\b(?:doanh thu|dthu|revenue|tien)\b", text))
    if not period or re.search(r"\b(?:so voi|so vs|cung ky|hon|kem|tang|giam|nhom|kenh|khach|thanh toan|tra bang|tra no|ghi no|ck|cod|chuyen khoan|tien mat|vi dien tu|ca sang|ca chieu|loi nhuan|von|nhap|kho|tuan nao|moi thang|chiem|dong gop|loai tru|it nhat|tro len|tro leen|don bi huy|nhan vien|ke don|hoat chat|qua dien thoai|tai quay|lan luot)\b", text):
        return None
    if not peak_day and (day_request or re.search(r"\b(?:theo|cao nhat|nhieu nhat)\b", text)):
        return None
    order = order_scope(catalog)
    average = bool(PER_ORDER_AVERAGE.search(text))
    if re.search(r"\b(?:binh quan|trung binh)\b", text) and not average:
        return None
    if not peak_day and expected_type(q["question"]) not in (None, "number"):
        return None
    count = bool(re.search(rf"\b{COUNT_REQUEST} {ORDER_COUNT}\b", text))
    if count and (average or products or re.search(r"\b(?:huy|tra|tien|doanh thu|dthu|hang nao|mat hang|san pham|co chua|co mua)\b", text)):
        return None
    if not (count or average or peak_day or re.search(SALES_WORDS, text)):
        return None
    if not average and re.search(r"\b(?:moi don|mot don|tren don)\b", text):
        return None
    if order is None:
        standalone = [t for t in scoped_tables(catalog) if has_roles(t, "date", "status", "product", "line_amount")]
        if len(standalone) != 1 or count or average or standalone[0][1]["order"]:
            return None
        order = standalone[0]
    source, hc, _ = order
    coverage = date_range(order)
    if not coverage or period[1].isoformat() < coverage[0] or period[0].isoformat() > coverage[1]:
        return None
    validity_filter = validity(order) if hc["order"] else f"vn_order_status(h.{hc['status']}) NOT IN ('canceled','returned','failed')"
    where = f"{inclusive_predicate('h.' + hc['date'], *period)} AND {validity_filter}"
    if count:
        return "number", f"SELECT COUNT(DISTINCT h.{hc['order']}) FROM {source} h WHERE {where}"

    lines = joined_lines(catalog, order)
    group_id = f"h.{hc['order']}" if hc["order"] else None
    date_col = f"h.{hc['date']}"
    if lines:
        detail, alias, joined = lines
        dc = detail[1]
        amount = f"{alias}.{dc['line_amount'] or dc['amount']}"
        if detail[0] != source and not peak_day:
            group_id = f"{alias}.{dc['order']}"
            joined = f"{detail[0]} {alias}"
            where = f"{group_id} IN (SELECT h.{hc['order']} FROM {source} h WHERE {where})"
    elif hc["order_amount"] and not products:
        amount = f"h.{hc['order_amount']}"
        joined = f"{source} h"
    else:
        return None
    if products:
        product = named_rollup(products)
        if not lines or not product or average:
            return None
        dimensions = [t for t in scoped_tables(catalog) if has_roles(t, "item", "product") and not t[1]["order"]
                      and not t[1]["stock"]
                      and dc["item"] and linked(catalog, detail, "item", t, "item")
                      and any(t[0] == "main." + quote(name) and t[1]["product"] == quote(col) and key(value) == product
                              for name, col, value in products)]
        if dc["product"] and any(detail[0] == "main." + quote(name) and dc["product"] == quote(col) and key(value) == product
                                 for name, col, value in products):
            where += f" AND vn_key({alias}.{dc['product']}) = {sql_literal(product)}"
        elif dc["item"] and len(dimensions) == 1:
            dim, cols, _ = dimensions[0]
            where += (f" AND EXISTS (SELECT 1 FROM {dim} p WHERE p.{cols['item']} = {alias}.{dc['item']}"
                      f" AND vn_key(p.{cols['product']}) = {sql_literal(product)})")
        else:
            return None
    elif re.search(r"\b(?:cua mat hang|tien ban (?!hang\b)|(?:doanh thu|dthu) cua (?!cua hang\b))", text):
        return None
    if average:
        return "number", (f"SELECT ROUND(AVG(revenue), 0) FROM (SELECT SUM({amount}) AS revenue"
                          f" FROM {joined} WHERE {where} GROUP BY {group_id})")
    if peak_day:
        return "date", (f"SELECT date({date_col}) FROM {joined} WHERE {where}"
                        f" GROUP BY date({date_col}) ORDER BY SUM({amount}) DESC, date({date_col}) LIMIT 1")
    return "number", f"SELECT COALESCE(SUM({amount}), 0) FROM {joined} WHERE {where}"


def basic_sales(q, catalog, products):
    text = key(q["question"])
    order = order_scope(catalog)
    if order is None:
        return None
    source, hc, _ = order
    coverage = date_range(order)
    if not coverage:
        return None
    period = question_period(q["question"], q["as_of_date"])
    kind = expected_type(q["question"])
    product = named_rollup(products)
    valid = validity(order)
    lines = joined_lines(catalog, order)

    if re.search(r"\b(?:gia ban|don gia ban)\b", text) and product and len(explicit_days(q["question"])) == 1 and lines:
        detail, alias, joined = lines
        dc = detail[1]
        if not has_roles(detail, "price", "product", "unit"):
            return None
        day = explicit_days(q["question"])[0]
        if not coverage[0] <= day.isoformat() <= coverage[1]:
            return None
        unit = re.search(r"\b(?:mot|1) (ly|kg|chai|lon|hop|phan|cai|goi|bo|loc|thung)\b", text)
        # The published rule uses the base unit and includes canceled and returned orders.
        if not unit or not dc["base_qty"] or not dc["qty"]:
            return None
        return "number", (f"SELECT CASE WHEN COUNT(DISTINCT {alias}.{dc['price']}) = 1"
                          f" THEN MIN({alias}.{dc['price']}) END FROM {joined}"
                          f" WHERE date(h.{hc['date']}) = '{day}'"
                          f" AND vn_key({alias}.{dc['product']}) = {sql_literal(product)}"
                          f" AND vn_key({alias}.{dc['unit']}) = {sql_literal(unit[1])}"
                          f" AND {alias}.{dc['qty']} = {alias}.{dc['base_qty']}")

    if not period or period[1].isoformat() < coverage[0] or period[0].isoformat() > coverage[1]:
        return None
    period_sql = inclusive_predicate(f"h.{hc['date']}", *period)
    where = f"{period_sql} AND {valid}"

    if product and lines and kind == "percent":
        detail, alias, joined = lines
        dc = detail[1]
        days = explicit_days(q["question"])
        windows = re.findall(r"\b(\d{1,2}) ngay\b", text)
        event = re.search(r"\b(?:doi gia|tang gia|gia len)\b", text)
        unit = re.search(r"\b(?:so|theo) (kg|ly|chai|lon|hop|phan|cai|goi|bo|loc|thung)\b", text)
        if (event and unit and len(days) == 1 and len(windows) >= 2 and len(set(windows)) == 1
                and int(windows[0]) > 0 and has_roles(detail, "product", "qty", "unit")
                and re.search(r"\b(?:ke tu|ngay do)\b", text) and re.search(r"\btruoc\b", text)):
            day, length = days[0], int(windows[0])
            previous = day - timedelta(days=length)
            end = day + timedelta(days=length-1)
            if (day - timedelta(days=1)).isoformat() >= coverage[0]:
                qualifier = f"{alias}.{dc['product']}"
                quantity = f"{alias}.{dc['qty']}"
                sold_unit = f"{alias}.{dc['unit']}"
                multiplier = "NULL"
                if any(t["name"] == "unit_factors" for t in catalog["tables"]):
                    multiplier = (f"(SELECT CASE WHEN COUNT(DISTINCT factor)=1 THEN MAX(factor) END FROM main.unit_factors"
                                  f" WHERE vn_key(product_name)=vn_key({qualifier}) AND vn_key(unit)=vn_key({sold_unit})"
                                  f" AND vn_key(base_unit)={sql_literal(unit[1])})")
                amount = f"{quantity} * CASE WHEN vn_key({sold_unit})={sql_literal(unit[1])} THEN 1 ELSE {multiplier} END"
                return "percent", (f"WITH sales AS (SELECT date(h.{hc['date']}) AS day, {amount} AS qty"
                                   f" FROM {joined} WHERE {inclusive_predicate('h.' + hc['date'], previous, end)}"
                                   f" AND {valid} AND vn_key({qualifier})={sql_literal(product)})"
                                   f" SELECT CASE WHEN COUNT(*)=COUNT(qty) THEN 100.0 *"
                                   f" (SUM(CASE WHEN day>='{day}' THEN qty ELSE 0 END)-SUM(CASE WHEN day<'{day}' THEN qty ELSE 0 END))"
                                   f" / NULLIF(SUM(CASE WHEN day<'{day}' THEN qty ELSE 0 END),0) END FROM sales")

    if re.search(r"\b(?:tong cong|tong so) bao nhieu " + ORDER_COUNT + r"\b", text) and re.search(r"\b\d+ ngay gan nhat\b", text):
        if not re.search(r"\b(?:khach|mat hang|san pham|huy|tra|kenh|ca sang|ca chieu|tien)\b", text):
            return "number", f"SELECT COUNT(DISTINCT h.{hc['order']}) FROM {source} h WHERE {where}"

    if (kind == "percent" and re.search(r"\bso don\b", text)
            and re.search(r"\bca chieu so voi ca sang\b", text) and hc["timestamp"]
            and any(quote(f["name"]) == hc["timestamp"] and f.get("timestamp_examples") for f in order[2])):
        clock = f"time(h.{hc['timestamp']})"
        oid = f"h.{hc['order']}"
        morning = f"COUNT(DISTINCT CASE WHEN {clock} >= '06:00:00' AND {clock} < '14:00:00' THEN {oid} END)"
        afternoon = f"COUNT(DISTINCT CASE WHEN {clock} >= '14:00:00' AND {clock} < '22:00:00' THEN {oid} END)"
        return "percent", (f"SELECT 100.0 * ({afternoon} - {morning}) / NULLIF({morning}, 0)"
                           f" FROM {source} h WHERE {where}")

    threshold = COUNT_THRESHOLD.search(text)
    if (threshold and re.search(r"\b(?:trong it nhat|xuat hien|co mat|tro (?:len|leen)|it nhat)\b", text)
            and not re.search(r"\b(?:ca sang|ca chieu|kenh|nhom|theo|loai tru|khong tinh|so voi|da huy)\b", text)):
        order_id = f"h.{hc['order']}"
        if count_intent(text, "khach") and hc["customer_id"] and not product:
            identity = f"h.{hc['customer_id']}"
            return "number", (f"SELECT COUNT(*) FROM (SELECT {identity} FROM {source} h WHERE {where}"
                              f" AND vn_known_customer({identity}) GROUP BY {identity}"
                              f" HAVING COUNT(DISTINCT {order_id}) >= {int(threshold[1])})")
        if count_intent(text, "mat hang|san pham|sp") and lines and lines[0][1]["product"] and not product:
            detail, alias, joined = lines
            dc = detail[1]
            identity = f"{alias}.{dc['item'] or dc['product']}"
            return "number", (f"SELECT COUNT(*) FROM (SELECT {identity} FROM {joined} WHERE {where}"
                              f" AND vn_key({alias}.{dc['product']}) NOT IN ('', 'phi phuc vu')"
                              f" AND {identity} IS NOT NULL AND trim({identity}) <> '' GROUP BY {identity}"
                              f" HAVING COUNT(DISTINCT {order_id}) >= {int(threshold[1])})")

    group = GROUP_SHARE.search(text)
    if group and kind == "percent" and re.search(SALES_WORDS, text) and lines:
        detail, alias, joined = lines
        dc = detail[1]
        amount = dc["line_amount"] or dc["amount"]
        category = f"{alias}.{dc['category']}" if dc["category"] else None
        if not category:
            categories = [t for t in scoped_tables(catalog) if has_roles(t, "item", "category") and not t[1]["order"]
                          and dc["item"] and linked(catalog, detail, "item", t, "item")]
            if len(categories) != 1 or not dc["item"]:
                return None
            table, cols, _ = categories[0]
            # A grouped dimension prevents multiplying detail rows on repeated item IDs.
            joined += (f" LEFT JOIN (SELECT {cols['item']} AS item, MIN(vn_key({cols['category']})) AS category"
                       f" FROM {table} GROUP BY {cols['item']}"
                       f" HAVING COUNT(DISTINCT vn_key({cols['category']})) = 1) dim"
                       f" ON dim.item = {alias}.{dc['item']}")
            category = "dim.category"
        return "percent", (f"SELECT CASE WHEN COUNT(CASE WHEN vn_key({category}) = {sql_literal(group[1])}"
                           f" THEN 1 END) > 0 THEN 100.0 * SUM(CASE WHEN vn_key({category}) = {sql_literal(group[1])}"
                           f" THEN {alias}.{amount} ELSE 0 END) / NULLIF(SUM({alias}.{amount}), 0)"
                           f" END FROM {joined} WHERE {where}")

    return None


def debt_query(q, catalog):
    text = key(q["question"])
    count_customers = bool(re.search(r"\b(?:so khach|bao nhieu khach|bao nhiu khach|may khach)\b", text)
                           and re.search(r"\b(?:con cong no|chua tra het no|con no)\b", text))
    total_debt = bool(re.search(r"\b(?:khach(?: hang)? con no (?:cua hang|tong cong)|so tien khach con thieu|tong cong no|tong so no)\b", text))
    if (not (count_customers or total_debt)
            or not re.search(r"\b(?:den het|tinh den|cuoi ngay)\b", text)
            or re.search(r"\b(?:nhom|tuoi no|tra no|thu no|top|nhat|nhung khach|theo|hon|qua han)\b", text)):
        return None
    orders = order_scope(catalog)
    payments = [t for t in scoped_tables(catalog)
                if has_roles(t, "order", "date", "payment_id", "paid")
                and (t[1]["method"] or re.search(r"so ?quy|phieu ?thu|thanh ?toan|tra ?no", key(t[0])))
                and orders and linked(catalog, orders, "order", t, "order")]
    if orders is None or not has_roles(orders, "balance", "method") or orders[1]["product"] or len(payments) != 1:
        return None
    days = explicit_days(q["question"])
    if len(days) != 1 and not re.search(r"\b(?:hom nay|hnay)\b", text):
        return None
    as_of = days[0] if len(days) == 1 else date.fromisoformat(q["as_of_date"])
    if not date_range(orders) or as_of.isoformat() < date_range(orders)[0] or not date_range(payments[0]):
        return None
    source, hc, _ = orders
    payments_source, pc, _ = payments[0]
    channel = f"AND vn_key(h.{hc['channel']}) NOT IN ('san tmdt','shopee','lazada','tiktok shop','tiki')" if hc["channel"] else ""
    balance = f"h.{hc['balance']} - COALESCE(paid.amount,0)"
    if count_customers:
        identity = hc["customer_id"] or hc["customer"]
        if not identity:
            return None
        customer = f"h.{identity}" if hc["customer_id"] else f"vn_key(h.{identity})"
        result = f"COUNT(DISTINCT CASE WHEN {balance} > 0 AND vn_known_customer({customer}) THEN {customer} END)"
    else:
        result = f"COALESCE(SUM(MAX({balance},0)),0)"
    return "number", (f"WITH paid AS (SELECT {pc['order']} AS order_id, SUM(vn_money({pc['paid']})) AS amount"
                      f" FROM {payments_source} WHERE date({pc['date']}) <= '{as_of}' GROUP BY {pc['order']})"
                      f" SELECT {result} FROM {source} h LEFT JOIN paid ON paid.order_id = h.{hc['order']}"
                      f" WHERE date(h.{hc['date']}) <= '{as_of}' AND {validity(orders, include_returns=True)}"
                      f" AND vn_key(h.{hc['method']}) NOT LIKE '%cod%'"
                      f" AND vn_key(h.{hc['method']}) NOT LIKE '%tmdt%' {channel}")
