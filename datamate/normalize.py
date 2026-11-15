"""Conservative conversions for Vietnamese spreadsheet exports."""

import re
import unicodedata
from datetime import date, datetime, time
from decimal import Decimal, InvalidOperation

from openpyxl.utils.datetime import from_excel


def fold(value):
    text = unicodedata.normalize("NFKD", str(value or "")).casefold().replace("đ", "d")
    return "".join(c for c in text if not unicodedata.combining(c))


def key(value):
    return re.sub(r"[^a-z0-9]+", " ", fold(value)).strip()


def slug(value):
    return key(value).replace(" ", "_")


MONEY = set("""
    amount|cong|thanh tien|thanh tien d|tt|tri gia|trigia|tien|tong tien|tongtien|
    tong phieu|tong don|tong bill|tong thanh toan|tongthanhtoan|tt don|gia tri hang|
    tong tien hang|tien hang truoc giam|tong sau giam gia|tong tien phieu|
    gia|dg|don gia|don gia d|gia 1 dv|gia san pham|gia von|gia von luc ban|
    gia von hien tai|gia ban|gia le|gia hien tai|gia niem yet|gia nhap|gia cu|gia moi|
    gia thu|gia tri|tien thu|so tien|da thu|da tra|no|cong no|con no|khach da tra|
    khach tra|da gop|gop thang|tra truoc|bot|thu may cu|tien giam|giam gia|
    giam d|ck|giam gia hd|phi ship|phi van chuyen|tien chi|can tra ncc
    """.replace("\n", "").split("|"))
MONEY = {x.strip() for x in MONEY}
QUANTITY = set("""
    sl|s l|slg|sl ban|sl nhap|sl quy doi|sl dv co ban|sl le|soluong|sluong|so luong|
    so luong san pham|so luong hang|so luong quy doi|so luong le|so luong nhap|
    so luong giao|so luong tra|sl thuc giao|quy ra dv le|ton kho|ton cuoi|ton cuoi ky|
    ton thuc|so khach|so ky|luot giao|ky han
    """.replace("\n", "").split("|"))
QUANTITY = {x.strip() for x in QUANTITY}
DATE_PREFIX = ("ngay", "date", "thoi gian", "gio", "han su dung", "han tra", "tu ngay", "den ngay", "sinh nhat")
ORDER_KEY = {
    "ma don", "ma don hang", "ma hoa don", "ma hoa don ban", "ma bill", "bill",
    "don so", "so don", "so ct", "ma chung tu", "so hoa don", "hd",
}
ORDER_META = {
    "tt don", "note", "nhan vien ban", "ma nv ban", "tong tien hang", "tong tien",
    "giam gia hd", "hinh thuc thanh toan", "thanh toan bang", "so khach",
    "khach", "khach hang", "khach hang ten", "ten khach hang", "tenkh", "ho ten khach",
    "ma kh", "ma khach", "ma khach mua", "ma nguoi mua", "ma the", "sdt", "phone",
    "xu ly", "trang thai", "trang thai don", "trang thai don hang", "trang thai xu ly",
    "trang thai giao", "tinh trang", "tien do", "ket qua", "ghi chu", "kenh", "nguon",
    "kenh ban", "kenh dat", "tong phieu", "tong thanh toan", "phi ship", "phi van chuyen",
    "khach tra", "cong no", "con no", "ngay", "ngay dat hang", "ngay hd", "ngay xuat",
    "ngay lap", "ngay giao dich", "thoi gian", "gio", "gio ban", "gio tao don",
}


def column_kind(name, description=""):
    k = key(name)
    d = key(description)
    if k in ORDER_KEY or k.startswith(("ma ", "so phieu", "so chung tu")) or k in {"stt", "imei", "phone", "sdt", "barcode"} or str(name).strip().casefold() == "no.":
        return "text"
    if "trang thai" in d or k in {"tt don", "note", "ghi chu tt", "ket qua", "tien do", "xu ly", "tinh trang"}:
        return "text"
    if k in {"he so", "he so qd", "ti le quy doi", "quy doi", "bang"}:
        return "factor"
    if k.startswith("tien mat") or k in MONEY or d.startswith(("thanh tien", "tong tien", "tien hang", "don gia", "so tien", "so khach phai tra")):
        return "money"
    if k in QUANTITY or d.startswith(("so luong", "ton kho")):
        return "quantity"
    if (k == "han gop ngay" or k == "han no ngay"):
        return "quantity"
    if any(k == prefix or k.startswith(prefix + " ") for prefix in DATE_PREFIX) or d in {"ngay ban", "ngay nhap", "ngay lap", "ngay tao", "ngay thu", "thoi diem thu", "ngay phai tra"}:
        return "date"
    return "text"


def year_hint(*values):
    for value in values:
        matches = set(re.findall(r"(?<!\d)20\d{2}(?!\d)", value))
        if len(matches) == 1:
            return int(matches.pop())
    return None


def as_date(value, year=None, epoch=None):
    if isinstance(value, datetime):
        return value.date().isoformat() if value.time() == time(0) else value.isoformat(sep=" ", timespec="seconds")
    if isinstance(value, (date, time)):
        return value.isoformat()
    text = str(value).strip()
    if re.fullmatch(r"[3456]\d{4}(?:\.\d+)?", text):
        if epoch is None:
            from openpyxl.utils.datetime import WINDOWS_EPOCH
            epoch = WINDOWS_EPOCH
        raw = float(text)
        return as_date(from_excel(round(raw * 86400) / 86400, epoch=epoch))
    text = re.sub(r"\s+thg\s+", "/", fold(text))
    text = re.sub(r"^(\d{1,2}/\d{1,2})\s+(\d{4})(?=\b)", r"\1/\2", text)
    text = text.replace("t", " ") if re.match(r"^\d{4}-\d\d-\d\dt", text) else text
    if re.fullmatch(r"\d{1,2}[:h]\d{2}(?::\d{2})?", text):
        text = text.replace("h", ":")
        try:
            return time.fromisoformat(("0" + text if text[1:2] == ":" else text)).isoformat()
        except ValueError:
            return None
    if year and re.fullmatch(r"\d{1,2}[/.-]\d{1,2}", text):
        text = re.sub(r"[.-]", "/", text) + f"/{year}"
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%d/%m/%Y", "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M", "%d-%m-%Y", "%d.%m.%Y", "%d/%m/%y"):
        try:
            d = datetime.strptime(text, fmt)
            return d.isoformat(sep=" ", timespec="seconds") if "%H" in fmt else d.date().isoformat()
        except ValueError:
            continue
    return None


QUANTITY_TOKEN = re.compile(r"([+-]?\d+(?:[.,]\d+)*)\s*[a-z][a-z0-9 /.-]*")
AMBIGUOUS_QUANTITY = re.compile(r"[+-]?[1-9]\d{0,2}([.,])\d{3}")


def quantity_token(text):
    match = QUANTITY_TOKEN.fullmatch(text)
    return match[1] if match else text


def quantity_separators(values):
    """Only infer separators from unambiguous strings in the same source column."""
    votes = {".": set(), ",": set()}
    for value in values:
        if not isinstance(value, str) or ("." not in value and "," not in value):
            continue
        text = quantity_token(fold(value).strip().replace("−", "-"))
        if match := re.fullmatch(r"[+-]?\d{1,3}([.,])\d{3}(?:\1\d{3})*(?!\1)([.,])\d+", text):
            thousands, decimal = match.groups()
            votes[thousands].add("group")
            votes[decimal].add("decimal")
        elif match := re.fullmatch(r"[+-]?\d{1,3}([.,])\d{3}(?:\1\d{3})+", text):
            votes[match[1]].add("group")
        elif (match := re.fullmatch(r"[+-]?\d+([.,])\d+", text)) and not AMBIGUOUS_QUANTITY.fullmatch(text):
            votes[match[1]].add("decimal")
    return {mark: next(iter(roles)) for mark, roles in votes.items() if len(roles) == 1}


def as_number(value, kind, separators=None):
    if isinstance(value, (int, float)):
        return value
    t = fold(value).strip().replace("−", "-").replace("\xa0", " ")
    t = re.sub(r"\s*(?:vnd|dong|d|₫)\s*$", "", t).strip()
    if kind == "money":
        t = re.sub(r"\s+(?=\d)", "", t)
        if m := re.fullmatch(r"([+-]?)(\d+)[ ]*tr[ ]*(\d{1,3})[ ]*k?", t):
            sign, millions, thousands = m.groups()
            return (-1 if sign == "-" else 1) * (int(millions) * 1000000 + int(thousands.ljust(3, "0")) * 1000)
        if m := re.fullmatch(r"([+-]?\d+(?:[.,]\d+)?)\s*(k|tr|trieu|nghin|ngan)", t):
            value, suffix = m.groups()
            return float(Decimal(value.replace(",", ".")) * (1000 if suffix in {"k", "nghin", "ngan"} else 1000000))
    elif kind == "quantity":
        t = quantity_token(t)
        if match := AMBIGUOUS_QUANTITY.fullmatch(t):
            role = (separators or {}).get(match[1])
            if role is None:
                return None
            if role == "group":
                t = t.replace(match[1], "")
    if re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+\.\d+", t):
        t = t.replace(",", "")
    elif re.fullmatch(r"[+-]?\d{1,3}(?:\.\d{3})+,\d+", t):
        t = t.replace(".", "").replace(",", ".")
    if re.fullmatch(r"[+-]?\d{1,3}(?:[.,]\d{3})+", t) and (kind == "money" or t.count(".") + t.count(",") > 1):
        t = t.replace(".", "").replace(",", "")
    elif re.fullmatch(r"[+-]?\d+[,.]\d+", t):
        t = t.replace(",", ".")
    elif not re.fullmatch(r"[+-]?\d+", t):
        return None
    try:
        d = Decimal(t)
        return int(d) if d == d.to_integral_value() else float(d)
    except InvalidOperation:
        return None


def normalize(value, kind, year, epoch, separators=None):
    if value is None or (isinstance(value, str) and not value.strip()):
        return None, False
    if kind == "date":
        parsed = as_date(value, year, epoch)
    elif kind in {"money", "quantity"}:
        parsed = as_number(value, kind, separators)
    elif kind == "factor":
        parsed = as_number(value, "factor")
        return (parsed if parsed is not None else str(value).strip()), False
    elif isinstance(value, (datetime, date, time)):
        return as_date(value), False
    else:
        return str(value).strip() if isinstance(value, str) else value, False
    if parsed is not None:
        return parsed, False
    # SQLite silently converts number-like strings during aggregation.
    return (None if kind == "quantity" else str(value).strip()), True
