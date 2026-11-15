"""Compute well-specified sales and accounting queries from catalog roles."""

import re
from datetime import date, timedelta

from .data import quote
from .normalize import key
from .periods import explicit_days, question_period
from .sql_rules import cols_for, expected_type, named_rollup, role_column, sql_literal

DETAIL = {
    "item": ("ma hang", "ma san pham", "ma sp"),
    "product": ("ten san pham", "ten hang", "ten hang hoa", "ten mat hang"),
    "category": ("nhom hang 3 cap", "nhom hang", "nhom san pham", "nhom mat hang"),
    "price": ("don gia", "gia san pham"),
    "qty": ("so luong", "so luong san pham"),
    "base_qty": ("so luong quy doi", "so luong co ban"),
    "unit": ("don vi tinh", "dvt", "don vi"),
    "method": ("phuong thuc thanh toan", "hinh thuc thanh toan"),
    "channel": ("kenh ban", "kenh ban hang", "kenh"),
    "timestamp": ("thoi gian tao", "gio tao", "ngay gio dat hang", "thoi gian"),
    "balance": ("tong sau giam gia", "tong thanh toan"),
    "payment_id": ("ma phieu", "ma phieu thu", "ma giao dich"),
    "paid": ("gia tri", "so tien thu", "so tien thanh toan"),
    "stock": ("ton cuoi ky", "so luong ton", "ton kho", "ton cuoi"),
}
ORDER_COUNT = r"(?:don(?: hang)?|hoa don|bill)"
COUNT_REQUEST = r"(?:bao nhieu|bao nhiu|nhieu|bnhiu|bn|may)"
SALES_WORDS = r"(?:doanh thu|dthu|tien ban|ban (?:duoc|dc)(?: tong cong)? (?:total )?(?:bao nhieu|bao nhiu|bnhiu|bn|may) tien)"
COUNT_THRESHOLD = re.compile(r"\b(?:it nhat|tu[wf]*) (\d{1,5}) " + ORDER_COUNT + r"\b")
GROUP_SHARE = re.compile(r"\bnhom (?:hang )?(.+?) (?:chiem|dong gop|la bao nhieu)\b")
PER_ORDER_AVERAGE = re.compile(r"\b(?:binh quan|trung binh)\s+(?:moi|mot|tren(?: mot)?) " + ORDER_COUNT + r"\b")


def scoped_tables(catalog):
    for table in catalog.get("tables", ()):
        columns = {**cols_for(table), **{
            role: role_column(table, role, aliases)
            for role, aliases in DETAIL.items() if role not in {"product", "unit"}
        }}
        columns["line_amount"] = role_column(table, "amount", ("thanh tien", "thanh tien d", "amount"))
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
    denied = "('canceled')" if include_returns else "('canceled','returned')"
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


def sales_totals(q, catalog, products):
    """Use a unique transaction source and sum at line grain when available."""
    text = key(q["question"])
    period = question_period(q["question"], q["as_of_date"])
    if not period or re.search(r"\b(?:so voi|so vs|cung ky|hon|kem|tang|giam|nhom|kenh|khach|thanh toan|tra bang|tra no|ghi no|ck|cod|chuyen khoan|tien mat|vi dien tu|ca sang|ca chieu|loi nhuan|von|theo|nhap|kho|tuan nao|moi thang|cao nhat|nhieu nhat|chiem|dong gop|loai tru|it nhat|tro len|tro leen|don bi huy)\b", text):
        return None
    order = order_scope(catalog)
    average = bool(PER_ORDER_AVERAGE.search(text))
    if re.search(r"\b(?:binh quan|trung binh)\b", text) and not average:
        return None
    if expected_type(q["question"]) not in (None, "number"):
        return None
    count = bool(re.search(rf"\b{COUNT_REQUEST} {ORDER_COUNT}\b", text))
    if count and (average or products or re.search(r"\b(?:huy|tra|tien|doanh thu|dthu|hang nao|mat hang|san pham|co chua|co mua)\b", text)):
        return None
    if not (count or average or re.search(SALES_WORDS, text)):
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
    validity_filter = validity(order) if hc["order"] else f"vn_order_status(h.{hc['status']}) NOT IN ('canceled','returned')"
    where = f"{inclusive_predicate('h.' + hc['date'], *period)} AND {validity_filter}"
    if count:
        return "number", f"SELECT COUNT(DISTINCT h.{hc['order']}) FROM {source} h WHERE {where}"

    lines = joined_lines(catalog, order)
    group_id = f"h.{hc['order']}" if hc["order"] else None
    if lines:
        detail, alias, joined = lines
        dc = detail[1]
        amount = f"{alias}.{dc['line_amount'] or dc['amount']}"
        if detail[0] != source:
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
    if (not re.search(r"\b(?:cong no|con no|con thieu)\b", text)
            or not re.search(r"\b(?:khach con no (?:cua hang |tong cong)|so tien khach con thieu|tong cong no|tong so no)\b", text)
            or not re.search(r"\b(?:den het|tinh den|cuoi ngay)\b", text)
            or re.search(r"\b(?:nhom|tuoi no|tra no|thu no|top|nhat|nhung khach|theo)\b", text)):
        return None
    orders = order_scope(catalog)
    payments = [t for t in scoped_tables(catalog)
                if has_roles(t, "order", "date", "payment_id", "paid")
                and (t[1]["method"] or re.search(r"so ?quy|phieu ?thu|thanh ?toan|tra ?no", key(t[0])))]
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
    return "number", (f"WITH paid AS (SELECT {pc['order']} AS order_id, SUM({pc['paid']}) AS amount"
                      f" FROM {payments_source} WHERE date({pc['date']}) <= '{as_of}' GROUP BY {pc['order']})"
                      f" SELECT COALESCE(SUM(MAX(h.{hc['balance']} - COALESCE(paid.amount,0),0)),0)"
                      f" FROM {source} h LEFT JOIN paid ON paid.order_id = h.{hc['order']}"
                      f" WHERE date(h.{hc['date']}) <= '{as_of}' AND {validity(orders, include_returns=True)}"
                      f" AND vn_key(h.{hc['method']}) NOT LIKE '%cod%'"
                      f" AND vn_key(h.{hc['method']}) NOT LIKE '%tmdt%' {channel}")
