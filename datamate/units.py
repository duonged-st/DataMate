"""Resolve unit hierarchies from source tables while preserving raw values."""

import re
from collections import defaultdict
from .normalize import key, as_number

PRODUCT = ("ten san pham", "ten hang", "ten hang hoa", "hang", "product", "tenhang", "ten sp")
LARGE = ("don vi quy doi", "loai dv", "dv lon")
BASE = ("don vi co ban", "dv goc", "dv nho")
FACTOR = ("ti le quy doi", "he so", "he so qd")


def make_unit_table(con, tables, readme, unique_name, quote):
    chains = defaultdict(dict)
    conflicts = set()
    for table in tables:
        columns = {key(c): c for c in table.get("columns", [])}
        names = [next((columns[x] for x in aliases if x in columns), None)
                 for aliases in (PRODUCT, LARGE, BASE, FACTOR)]
        if not all(names):
            continue
        sql = f"SELECT {', '.join(quote(n) for n in names)} FROM {quote(table['name'])}"
        for product, large, base, factor in con.execute(sql):
            if not all(x is not None for x in (product, large, base, factor)):
                continue
            amount = as_number(factor, "factor")
            target = base
            if amount is None and (m := re.fullmatch(r"\s*(\d+(?:[.,]\d+)?)\s+(.+?)\s*", str(factor))):
                amount = as_number(m[1], "factor")
                target = m[2]
            if amount and amount > 0:
                identity = key(product), key(large)
                mapping = (str(product), str(large), str(base), str(target), amount)
                if identity in chains[key(product)] and chains[key(product)][identity][2:] != mapping[2:]:
                    conflicts.add(identity)
                chains[key(product)][identity] = mapping
    for line in readme.splitlines():
        m = re.fullmatch(r"\s*[-*]\s*(.+?):\s*1\s+(.+?)\s*=\s*(\d+(?:[.,]\d+)?)\s+(.+?)\s*", line)
        if m:
            product, unit, amount, target = m.groups()
            identity = key(product), key(unit)
            chains[key(product)].setdefault(identity, (product, unit, target, target, as_number(amount, "factor")))
    rows = set()
    for product, units in chains.items():
        for identity, (product_name, unit, base, target, factor) in units.items():
            if identity in conflicts or not factor:
                continue
            seen = {identity}
            next_key = product, key(target)
            while next_key in units and next_key not in seen and next_key not in conflicts:
                seen.add(next_key)
                _, _, _, target, multiplier = units[next_key]
                factor *= multiplier
                next_key = product, key(target)
            if next_key in seen or next_key in conflicts or key(target) != key(base):
                continue
            rows.add((product_name, unit, base, factor))
            rows.add((product_name, base, base, 1))
    if not rows:
        return None
    name = unique_name("unit_factors")
    con.execute(f"CREATE TABLE {quote(name)} (product_name TEXT, unit TEXT, base_unit TEXT, factor NUMERIC)")
    con.executemany(f"INSERT INTO {quote(name)} VALUES (?, ?, ?, ?)", sorted(rows))
    columns = ["product_name", "unit", "base_unit", "factor"]
    notes = {
        "product_name": "Tên sản phẩm theo dữ liệu gốc; nối theo tên và unit",
        "unit": "Đơn vị được ghi trong dòng hàng",
        "base_unit": "Đơn vị cơ bản",
        "factor": "Nhân số lượng dòng hàng với hệ số này để ra số lượng đơn vị cơ bản",
    }
    return {"name": name, "columns": columns, "sources": [{"file": "README.md + unit tables"}],
            "kinds": {**dict.fromkeys(columns, "text"), "factor": "factor"},
            "rows": len(rows), "notes": notes}
