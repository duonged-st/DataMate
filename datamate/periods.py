"""Resolve explicit question periods and check SQL date predicates."""

import re
from datetime import date, timedelta

from .normalize import key
from .sql_rules import month_range

MONTH = re.compile(r"\bthang[s]? (\d{1,2}) (?:nam )?(\d{4})\b")
QUARTER = re.compile(r"\bquy ([1-4]) (?:nam )?(\d{4})\b")
DAY = re.compile(r"(?<!\d)(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})(?!\d)")
IDENT = r'(?:"(?:[^"]|"")+"|\[[^\]]+\]|`[^`]+`|[^\W\d][\w$]*)'
DATE_CALL = re.compile(r"\b(?:date|datetime)\s*\(\s*((?:" + IDENT + r"\s*\.\s*)?" + IDENT + r")\s*\)", re.I)
DATE_LITERAL_CALL = re.compile(r"\b(?:date|datetime)\s*\(\s*'((?:[^']|'')*)'\s*[,)]", re.I)
BARE_COLUMN = re.compile(r"(?:" + IDENT + r"\s*\.\s*)?" + IDENT)
EQUAL_LITERAL = re.compile(r"""\s*(?:==|=|IS\b)\s*(['"])(\d{4}-\d{2}(?:-\d{2})?)\1""", re.I)
LITERAL_OR_COMMENT = re.compile(r"""'(?:''|[^'])*'|"(?:""|[^"])*"|`(?:``|[^`])*`|\[[^\]]*\]|--[^\n]*|/\*[\s\S]*?\*/""")
ORDER_ID = {"ma don", "ma don hang", "ma hoa don", "ma bill", "ma khach", "ma khach hang", "ma hang", "ma san pham"}


class DatePredicateError(ValueError):
    pass


def explicit_days(question):
    result = []
    for day, month, year in DAY.findall(question):
        try:
            result.append(date(int(year), int(month), int(day)))
        except ValueError:
            continue
    return list(dict.fromkeys(result))


def question_period(question, as_of_date):
    """Return an inclusive period for the simple, single-period questions."""
    text = key(question)
    today = date.fromisoformat(as_of_date)
    days = explicit_days(question)
    if days:
        return (days[0], days[0]) if len(days) == 1 and not MONTH.search(text) else None
    months = MONTH.findall(text)
    if len(months) == 1:
        if QUARTER.search(text) or re.search(r"\b(?:cung ky|tu ngay|den ngay|ngay dau|\d+ ngay dau|\d+ ngay cuoi)\b", text):
            return None
        month, year = map(int, months[0])
        return month_range(year, month)
    if months:
        return None
    quarters = QUARTER.findall(text)
    if quarters:
        if len(quarters) != 1:
            return None
        quarter, year = map(int, quarters[0])
        start = month_range(year, 3 * quarter - 2)
        end = month_range(year, 3 * quarter)
        return (start[0], end[1]) if start and end else None
    match = re.search(r"\b(\d{1,3}) ngay gan nhat\b", text)
    if match and int(match[1]) > 0:
        return today - timedelta(days=int(match[1]) - 1), today
    monday = today - timedelta(days=today.weekday())
    if re.search(r"\btuan (?:truoc|roi|trc)\b", text):
        return monday - timedelta(days=7), monday - timedelta(days=1)
    if "tuan nay" in text:
        return monday, today
    if "thang nay" in text or "dau thang" in text:
        return today.replace(day=1), today
    if re.search(r"\b(?:nam nay|dau nam)\b", text):
        return today.replace(month=1, day=1), today
    if re.search(r"\b(?:hom qua|hqua|bua qua)\b", text):
        previous = today - timedelta(days=1)
        return previous, previous
    if re.search(r"\b(?:hom nay|hnay|bua nay)\b", text):
        return today, today
    return None


def validate_date_predicates(q, sql, catalog):
    text = key(q["question"])
    period = question_period(q["question"], q["as_of_date"]) if q.get("as_of_date") else None
    whole_month = (period and period[0] < period[1] and period[0].day == 1
                   and (bool(MONTH.search(text)) or "thang nay" in text)
                   and not re.search(r"\b(?:ngay nao|ngay dau|dau thang|so voi|cung ky|ke tu|trong .* ngay)\b", text))
    date_columns = {key(f["name"]) for t in catalog.get("tables", [])
                    for f in t["fields"] if f.get("kind") == "date" or f.get("range")}
    skip = list(LITERAL_OR_COMMENT.finditer(sql))
    for function in DATE_LITERAL_CALL.finditer(sql):
        if not any(m.start() <= function.start() < m.end() for m in skip):
            column = function[1].replace("''", "'")
            if key(column) in date_columns:
                raise DatePredicateError(f'DATE_FILTER: {column!r} is a text literal. Use the catalog column in double quotes.')
    if whole_month:
        for field in BARE_COLUMN.finditer(sql):
            if any(m.start() < field.start() < m.end() for m in skip):
                continue
            name = key(re.split(r"\s*\.\s*", field[0])[-1].strip('"`[]'))
            if name not in date_columns:
                continue
            condition = EQUAL_LITERAL.match(sql, field.end())
            if condition and condition[2] in {period[0].isoformat(), period[0].isoformat()[:7]}:
                raise DatePredicateError("DATE_FILTER: the question asks for the whole month. Use the full date range.")
    for function in DATE_CALL.finditer(sql):
        if any(m.start() <= function.start() < m.end() for m in skip):
            continue
        column = re.split(r"\s*\.\s*", function[1])[-1].strip('"`[]')
        name = key(column)
        if name in ORDER_ID and name not in date_columns:
            raise DatePredicateError(f"DATE_FILTER: {column} is an ID, not a transaction date. Join the source with the real date.")
        condition = EQUAL_LITERAL.match(sql, function.end())
        if condition and len(condition[2]) == 7:
            raise DatePredicateError("DATE_FILTER: date(...) returns YYYY-MM-DD. Use a month range with >= first day and < next month.")
        if condition and whole_month and condition[2] == period[0].isoformat():
            raise DatePredicateError("DATE_FILTER: the question asks for the whole period. Do not restrict to its first day.")
