"""Load a split and build one SQLite database and catalog per business."""

import csv
import hashlib
import io
import json
import re
import sqlite3
import zipfile
from collections import Counter
from contextlib import closing, contextmanager
from dataclasses import dataclass
from itertools import chain, islice
from pathlib import Path, PurePosixPath

import openpyxl

from .normalize import ORDER_KEY, ORDER_META, column_kind, key, normalize, quantity_separators, slug, year_hint
from .units import make_unit_table

DATA_EXTENSIONS = {".csv", ".xlsx", ".sqlite", ".db", ".sqlite3"}
CACHE_VERSION = "v5.8"


def quote(name):
    return '"' + name.replace('"', '""') + '"'


def decode(data):
    for encoding in ("utf-8-sig", "cp1258", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("Cannot decode the text file")


def display_name(path):
    name = PurePosixPath(path).name
    if any(c in name for c in "├┐║╗ß"):
        try:
            return name.encode("cp437").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    return name


class Dataset:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.archive = zipfile.ZipFile(self.path) if self.path.is_file() else None
        if self.archive:
            members = self.archive.namelist()
            roots = [name[:-len("questions.jsonl")] for name in members
                     if PurePosixPath(name).name == "questions.jsonl"
                     and name[:-len("questions.jsonl")] + "rulebook.md" in members]
            if len(roots) != 1:
                self.archive.close()
                raise ValueError("Expected one split in the ZIP")
            self.prefix = roots[0]
            self.members = {name[len(self.prefix):] for name in members
                            if name.startswith(self.prefix) and not name.endswith("/")}
        else:
            if not (self.path / "questions.jsonl").is_file():
                candidates = [p.parent for p in self.path.glob("*/questions.jsonl")]
                if len(candidates) != 1:
                    raise FileNotFoundError(f"questions.jsonl not found uniquely under {self.path}")
                self.path = candidates[0]
            self.members = {p.relative_to(self.path).as_posix() for p in self.path.rglob("*") if p.is_file()}

    def read(self, name):
        if name not in self.members:
            raise FileNotFoundError(name)
        return self.archive.read(self.prefix + name) if self.archive else (self.path / name).read_bytes()

    def jsonl(self, name):
        return [json.loads(line) for line in decode(self.read(name)).splitlines() if line.strip()]

    def files(self, business):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", business):
            raise ValueError("Invalid business path component")
        folder = PurePosixPath("data", business)
        return sorted(p for p in self.members if PurePosixPath(p).is_relative_to(folder))

    def fingerprint(self, files):
        digest = hashlib.sha256(CACHE_VERSION.encode())
        for name in sorted(files):
            if self.archive:
                info = self.archive.getinfo(self.prefix + name)
                stamp = (info.CRC, info.file_size)
            else:
                stat = (self.path / name).stat()
                stamp = (stat.st_size, stat.st_mtime_ns)
            digest.update(repr((name, stamp)).encode())
        return digest.hexdigest()[:20]

    def __enter__(self):
        return self

    def __exit__(self, *_):
        if self.archive:
            self.archive.close()


COLUMN = re.compile(r"^\s*[-*]\s+\*\*(.+?)\*\*\s*:\s*(.*)$", re.M)


def descriptions(readme):
    table_notes = []
    section = ""
    for line in readme.splitlines():
        if line.startswith("#"):
            section = line
        match = COLUMN.match(line)
        if match:
            table_notes.append((key(section), key(match[1]), match[2]))
    return table_notes


def column_notes(readme, filename, sheet, headers):
    result = {}
    sections = descriptions(readme)
    source = key(Path(filename).stem)
    sheet_key = key(sheet)
    for name in headers:
        candidates = [(int(bool(source and source in s)) * 2 + int(bool(sheet_key and sheet_key in s)), d)
                      for s, k, d in sections if k == key(name)]
        if candidates:
            best = max(score for score, _ in candidates)
            texts = list(dict.fromkeys(d for score, d in candidates if score == best))
            if len(texts) == 1:
                result[name] = texts[0]
    return result


def is_filled(value):
    return value is not None and (not isinstance(value, str) or bool(value.strip()))


def header_score(row, known):
    filled = [str(v).strip() for v in row if is_filled(v)]
    if len(filled) < 2:
        return 0
    keys = [key(v) for v in filled]
    labels = sum(bool(re.fullmatch(r"[^0-9]{1,70}", v)) for v in filled)
    evidence = sum(k in known or k in ORDER_KEY or k in ORDER_META or
                   column_kind(k) != "text" or k.startswith(("ma ", "ten ", "trang thai"))
                   for k in keys)
    if labels * 2 < len(filled) or evidence < 1:
        return 0
    return 3 * evidence + labels + len(set(keys)) / len(keys) - (len(filled) - labels)


def headers_for(rows, known):
    ranked = [(header_score(row, known), i) for i, row in enumerate(rows[:18])]
    score, position = max(ranked, key=lambda x: (x[0], -x[1]), default=(0, 0))
    if score <= 0:
        return None
    row = list(rows[position])
    while row and not is_filled(row[-1]):
        row.pop()
    used = set()
    columns = []
    for i, name in enumerate(row, 1):
        label = str(name).strip() if is_filled(name) else f"column_{i}"
        candidate = label
        serial = 2
        while candidate.casefold() in used:
            candidate = f"{label}_{serial}"
            serial += 1
        columns.append(candidate)
        used.add(candidate.casefold())
    return position, columns


def file_family(filename):
    name = Path(filename).stem
    name = re.sub(r"(?i)[ _-]+(?:thang\s*0?\d{1,2}|t0?\d{1,2})[ _-]+20\d{2}$", "", name)
    return re.sub(r"[ _-]+(?:20\d{2}|\d{1,2})$", "", name)


def is_period_sheet(sheet):
    return bool(re.fullmatch(r"(?:(?:thang|t|m)\s*)?\d{1,2} 20\d{2}|20\d{2} \d{1,2}", key(sheet)))


@dataclass
class Fragment:
    filename: str
    sheet: str
    columns: list
    rows: object
    year: int | None
    epoch: object
    header: int


@contextmanager
def workbook(data):
    # Prevent stale dimensions from truncating rows exported by third-party tools.
    book = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    try:
        yield book
    finally:
        book.close()


def fragments(dataset, files, readme, warnings):
    known = {name for _, name, _ in descriptions(readme)}
    for path in files:
        suffix = PurePosixPath(path).suffix.lower()
        name = display_name(path)
        if suffix == ".csv":
            content = decode(dataset.read(path))
            try:
                dialect = csv.Sniffer().sniff(content[:16384], delimiters=",;\t|")
            except csv.Error:
                dialect = csv.excel
            rows = csv.reader(io.StringIO(content), dialect)
            columns = next(rows, None)
            if columns is None:
                warnings.append({"file": name, "error": "Empty CSV"})
                continue
            _, columns = headers_for([columns], known) or (0, columns)
            yield Fragment(name, "", columns, rows, year_hint(name), None, 1)
        elif suffix == ".xlsx":
            with workbook(dataset.read(path)) as book:
                for sheet in book:
                    sheet.reset_dimensions()
                    rows = sheet.iter_rows(values_only=True)
                    head = list(islice(rows, 22))
                    found = headers_for(head, known)
                    if not found:
                        warnings.append({"file": name, "sheet": sheet.title, "error": "Header not detected"})
                        continue
                    i, columns = found
                    clues = " ".join(str(c) for row in head[:i] for c in row if is_filled(c))
                    yield Fragment(name, sheet.title, columns, chain(head[i+1:], rows),
                                   year_hint(sheet.title, name, clues), book.epoch, i+1)


class CatalogBuilder:
    def __init__(self, con, readme, used_names, warnings):
        self.con = con
        self.readme = readme
        self.used = set(used_names)
        self.warnings = warnings
        self.groups = {}

    def name(self, base):
        base = slug(base) or "data_table"
        if base.startswith("sqlite_"):
            base = "data_" + base
        name = base
        serial = 2
        while name.casefold() in self.used:
            name = f"{base}_{serial}"
            serial += 1
        self.used.add(name.casefold())
        return name

    def add(self, fragment):
        f = fragment
        family = file_family(f.filename)
        sheet_group = "" if is_period_sheet(f.sheet) or key(f.sheet) in {"sheet1", "sheet 1", ""} else f.sheet
        signature = tuple(key(c) for c in f.columns)
        group = key(family), key(sheet_group), tuple(sorted(signature))
        if group not in self.groups:
            name = self.name("_".join(part for part in (family, sheet_group) if part))
            columns = f.columns
            notes = column_notes(self.readme, f.filename, f.sheet, columns)
            kinds = {c: column_kind(c, notes.get(c, "")) for c in columns}
            schema = ", ".join(f"{quote(c)} {('NUMERIC' if kinds[c] in {'money', 'quantity'} else 'TEXT' if kinds[c] != 'factor' else '')}" for c in columns)
            self.con.execute(f"CREATE TABLE {quote(name)} ({schema})")
            self.groups[group] = {
                "name": name, "kinds": kinds, "notes": notes, "columns": columns,
                "rows": 0, "sources": [], "parse_failures": Counter(), "duplicates": 0,
            }
        table = self.groups[group]
        rows, dropped, failures, quantity_formats = self.insert(table, f)
        table["rows"] += rows
        table["duplicates"] += dropped
        table["parse_failures"].update(failures)
        table["sources"].append({"file": f.filename, "sheet": f.sheet or None,
                                 "header": f.header, "rows": rows, "duplicates_removed": dropped,
                                 "quantity_separators": quantity_formats})

    def insert(self, table, f):
        columns = table["columns"]
        indices = [f.columns.index(c) if c in f.columns else next(i for i, x in enumerate(f.columns) if key(x) == key(c)) for c in columns]
        labels = [key(c) for c in columns]
        kinds = [table["kinds"][c] for c in columns]
        order = next((i for i, k in enumerate(labels) if k in ORDER_KEY), None)
        sales_lines = any(k in {"hang", "hang ban", "hang hoa", "mat hang", "ten hang", "ten hang hoa",
                                "ten san pham", "ten sp", "ma hang", "ma hang hoa"} for k in labels)
        phone_lines = "imei" in labels and "stt" in labels and sales_lines
        if order is None and phone_lines:
            order = labels.index("stt")
        line_id = next((i for i, k in enumerate(labels) if (k in {"stt", "dong", "so dong"} or columns[i].casefold() == "no.") and i != order), None)
        sales_lines = sales_lines and (any(k in ORDER_META for k in labels) or line_id is not None)
        metadata = [i for i, k in enumerate(labels) if k in ORDER_META and i != order] if sales_lines else []
        item = next((i for i, k in enumerate(labels) if k in {"ten san pham", "ten hang", "ten hang hoa", "hang ban", "hang", "hang hoa", "ten sp", "mat hang"}), None)
        sql = f"INSERT INTO {quote(table['name'])} VALUES ({','.join('?' for _ in columns)})"
        failures = Counter()
        batch = []
        count = dropped = 0
        active = None
        remembered = {}
        exact_seen = set()
        if line_id is not None and order is not None:
            # The same exact line key and values can recur in split exports.
            # Keep differing rows so that source ambiguity stays visible.
            if table["rows"]:
                old = self.con.execute(f"SELECT * FROM {quote(table['name'])}")
                exact_seen.update(hashlib.blake2b(repr(tuple(str(v) if v is not None else None for v in row)).encode(), digest_size=16).digest() for row in old)
        rows = list(f.rows) if "quantity" in kinds else f.rows
        formats = {col: quantity_separators(row[i] for row in rows if i < len(row))
                   for col, i, kind in zip(columns, indices, kinds) if kind == "quantity"}
        for row in rows:
            raw = [row[i] if i < len(row) else None for i in indices]
            if not any(map(is_filled, raw)):
                active, remembered = None, {}
                continue
            if key(raw[0]) == labels[0] and all(key(v) == label for v, label in zip(raw, labels)):
                active, remembered = None, {}
                continue
            leading = next((key(x) for x in raw if is_filled(x)), "")
            if leading in {"tong cong", "tong ket", "tong thang", "cong", "tong", "subtotal", "total"} or (leading.startswith(("ghi chu ", "luu y ")) and sum(map(is_filled, raw)) == 1):
                active, remembered = None, {}
                continue
            if order is not None and sales_lines:
                order_value = raw[order]
                if is_filled(order_value):
                    order_value = str(order_value).strip()
                    if active != order_value:
                        active, remembered = order_value, {}
                elif active and item is not None and is_filled(raw[item]) and any(is_filled(x) and k in {"quantity", "money"} for x, k in zip(raw, kinds)):
                    raw[order] = active
                else:
                    active, remembered = None, {}
                if active:
                    for i in metadata:
                        if is_filled(raw[i]):
                            remembered[i] = raw[i]
                        elif i in remembered:
                            raw[i] = remembered[i]
            normalized = []
            for col, value, kind in zip(columns, raw, kinds):
                value, failed = normalize(value, kind, f.year, f.epoch, formats.get(col))
                if failed:
                    failures[col] += 1
                normalized.append(value)
            if line_id is not None and order is not None and normalized[order] is not None and normalized[line_id] is not None:
                token = hashlib.blake2b(repr(tuple(str(v) if v is not None else None for v in normalized)).encode(), digest_size=16).digest()
                if token in exact_seen:
                    dropped += 1
                    continue
                exact_seen.add(token)
            batch.append(normalized)
            count += 1
            if len(batch) >= 4000:
                self.con.executemany(sql, batch)
                batch.clear()
        if batch:
            self.con.executemany(sql, batch)
        return count, dropped, failures, formats


def table_names(con):
    return [r[0] for r in con.execute(
        "SELECT name FROM sqlite_schema WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%' ORDER BY name"
    )]


def profile(con, table):
    name = table["name"]
    schema = con.execute(f"PRAGMA table_info({quote(name)})").fetchall()
    fields = []
    key_candidates = {}
    for _, column, sql_type, _, _, pk in schema:
        col = quote(column)
        ck = key(column)
        note = table.get("notes", {}).get(column, "")
        kind = table.get("kinds", {}).get(column) or column_kind(column, note)
        field = {"name": column, "type": sql_type or kind.upper(), "kind": kind, "note": note}
        if kind == "quantity" and table.get("parse_failures", {}).get(column):
            field["missing_quantities"] = table["parse_failures"][column]
        is_identifier = (pk or ck in ORDER_KEY or ck in {"stt", "imei", "phone", "sdt", "barcode"}
                         or ck.startswith(("ma ", "so phieu")) or (column[:2] == "Ma" and column[2:3].isupper())
                         or bool(re.match(r"ma (?:don|hoa don|hang|mat hang|san pham|khach)\b", key(note))))
        if is_identifier:
            field["pk"] = bool(pk)
            key_candidates[column] = True
        if kind == "date":
            # Fast for a single pass even without an index. ISO timestamps sort chronologically.
            first, last = con.execute(
                f"SELECT MIN(substr({col},1,10)),MAX(substr({col},1,10)) FROM {quote(name)} WHERE {col} GLOB '20[0-9][0-9]-[0-9][0-9]-[0-9][0-9]*'"
            ).fetchone()
            if first:
                field["range"] = [first, last]
                sample = con.execute(
                    f"SELECT {col} FROM {quote(name)} WHERE length({col}) >= 16 "
                    f"AND (substr({col},11,1) IN (' ', 'T')) "
                    f"AND substr({col},14,1) = ':' LIMIT 1"
                ).fetchone()
                if sample:
                    field["timestamp_examples"] = [sample[0]]
        if kind == "text" and (ck in ORDER_META or "trang thai" in key(note) or any(w in ck for w in ("trang thai", "phuong thuc", "don vi", "nhom", "kenh", "pttt", "dvt"))):
            if not is_identifier and (ck not in {"khach", "khach hang", "khach hang ten", "ten khach hang", "tenkh", "ho ten khach", "ghi chu"}):
                vals = con.execute(
                    f"SELECT {col} FROM {quote(name)} WHERE {col} IS NOT NULL AND {col} != '' GROUP BY {col} ORDER BY COUNT(*) DESC, {col} LIMIT 9"
                ).fetchall()
                field["examples"] = [str(x[0])[:75] for x in vals]
        fields.append(field)
    return fields, key_candidates


def join_hints(con, catalogs, candidates):
    relationships = []
    profiles = []
    for table, columns in zip(catalogs, candidates):
        name = table["name"]
        for column in columns:
            col = quote(column)
            records = con.execute(f"SELECT {col} FROM {quote(name)} WHERE {col} IS NOT NULL AND {col} != '' LIMIT 160").fetchall()
            sample = {str(r[0]) for r in records}
            if sample:
                profiles.append((name, column, sample, len(sample) == len(records)))
        for _, _, other, col, target, *_ in con.execute(f"PRAGMA foreign_key_list({quote(name)})"):
            if target:
                relationships.append({"left": [name, col], "right": [other, target], "confidence": "declared FK"})
    for i, (left, lc, a, au) in enumerate(profiles):
        for right, rc, b, bu in profiles[i+1:]:
            if left == right or min(len(a), len(b)) < 3:
                continue
            common = len(a & b)
            overlap = common / min(len(a), len(b))
            if common < 3 or overlap < 0.3:
                continue
            if all(v.isdigit() for v in a | b) and key(lc) != key(rc):
                continue
            confidence = "sample overlap; one side unique in sample" if au != bu else "sample overlap; check cardinality"
            relationships.append({"left": [left, lc], "right": [right, rc],
                                  "confidence": confidence, "overlap": round(overlap, 3)})
    return sorted(relationships, key=lambda r: -r.get("overlap", 1))[:32]


def open_readonly(path):
    return sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)


def build_business(dataset, business, work):
    files = dataset.files(business)
    prefix = f"data/{business}/"
    readme_name = prefix + "README.md"
    readme = decode(dataset.read(readme_name)) if readme_name in files else ""
    data_files = [f for f in files if PurePosixPath(f).suffix.lower() in DATA_EXTENSIONS]
    if not data_files:
        raise FileNotFoundError(f"No structured data under {prefix}")
    cache = Path(work) / dataset.fingerprint(files)
    cache.mkdir(parents=True, exist_ok=True)
    db_path = cache / f"{business}.sqlite"
    catalog_path = cache / f"{business}.json"
    if db_path.is_file() and catalog_path.is_file():
        return db_path, json.loads(catalog_path.read_text(encoding="utf-8")), readme
    tmp = cache / f"{business}.building.sqlite"
    tmp.unlink(missing_ok=True)
    native = [f for f in data_files if PurePosixPath(f).suffix.lower() in {".sqlite", ".db", ".sqlite3"}]
    warnings = [] if readme else [{"error": "README.md is missing"}]
    if native:
        # Use a workspace copy of the supplied SQLite database, retaining its schema.
        tmp.write_bytes(dataset.read(native[0]))
    with closing(sqlite3.connect(tmp)) as con, con:
        con.execute("PRAGMA temp_store=MEMORY")
        native_catalog = [{"name": t, "sources": [{"file": display_name(native[0])}],
                           "rows": con.execute(f"SELECT COUNT(*) FROM {quote(t)}").fetchone()[0]}
                          for t in table_names(con)] if native else []
        builder = CatalogBuilder(con, readme, [t["name"].casefold() for t in native_catalog], warnings)
        for i, source in enumerate(native[1:], 1):
            extra = cache / f"source_{i}.sqlite"
            extra.write_bytes(dataset.read(source))
            with closing(open_readonly(extra)) as other:
                for table in table_names(other):
                    schema = other.execute(f"PRAGMA table_info({quote(table)})").fetchall()
                    columns = [str(c[1]) for c in schema]
                    name = builder.name(f"{Path(display_name(source)).stem}_{table}")
                    con.execute(f"CREATE TABLE {quote(name)} ({', '.join(quote(c) for c in columns)})")
                    vals = other.execute(f"SELECT {', '.join(quote(c) for c in columns)} FROM {quote(table)}")
                    count = con.executemany(f"INSERT INTO {quote(name)} VALUES ({','.join('?' for _ in columns)})", vals).rowcount
                    native_catalog.append({"name": name, "sources": [{"file": display_name(source)}], "rows": count})
            extra.unlink()
        for fragment in fragments(dataset, data_files, readme, warnings):
            builder.add(fragment)
        tables = native_catalog + list(builder.groups.values())
        if not tables:
            raise ValueError("No usable tables were found")
        conversions = make_unit_table(con, tables, readme, builder.name, quote)
        if conversions:
            tables.append(conversions)
        keys = []
        for t in tables:
            t["fields"], candidates = profile(con, t)
            keys.append(candidates)
        catalog = {"tables": tables, "joins": join_hints(con, tables, keys), "warnings": warnings}
    tmp.replace(db_path)
    staging = catalog_path.with_suffix(".tmp")
    staging.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    staging.replace(catalog_path)
    return db_path, catalog, readme
