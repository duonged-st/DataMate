"""Local implementation of the published DataMate rulebook."""

import math
import re
import unicodedata
from decimal import Decimal
from datetime import date


def normalize_name(text):
    return "".join(c for c in unicodedata.normalize("NFC", text).casefold()
                   if c.isalnum() or unicodedata.category(c).startswith("M"))


def number(value):
    return type(value) in (int, float) and (type(value) is int or math.isfinite(value))


def valid_value(value, kind):
    if kind in {"number", "percent"}:
        if not number(value):
            return False
        if kind == "number" or type(value) is int:
            return True
        decimal = Decimal(str(value))
        return abs(decimal - round(decimal, 2)) <= Decimal("1e-9") if abs(value) < 1e20 else True
    if kind == "boolean":
        return type(value) is bool
    if kind == "date":
        if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}(-\d{2})?", value):
            return False
        try:
            date.fromisoformat(value if len(value) == 10 else value + "-01")
        except ValueError:
            return False
        return True
    if kind == "string":
        return isinstance(value, str)
    if kind == "list":
        return isinstance(value, list)
    if kind == "table":
        if not isinstance(value, dict) or set(value) != {"columns", "rows"}:
            return False
        columns, rows = value["columns"], value["rows"]
        return (isinstance(columns, list) and bool(columns) and all(isinstance(c, str) for c in columns)
                and isinstance(rows, list) and all(isinstance(row, list) and len(row) == len(columns) for row in rows))
    return False


def match_cell(pred, gold):
    if type(gold) is bool:
        return type(pred) is bool and pred == gold
    if number(gold):
        if not number(pred):
            return False
        if type(pred) is int and type(gold) is int:
            return pred == gold
        return pred == gold or abs(Decimal(str(pred)) - Decimal(str(gold))) <= Decimal("1e-9")
    if isinstance(gold, str):
        return isinstance(pred, str) and normalize_name(pred) == normalize_name(gold)
    if isinstance(gold, list):
        return isinstance(pred, list) and len(pred) == len(gold) and all(
            match_cell(p, g) for p, g in zip(pred, gold))
    return pred is None and gold is None


def item_aliases(aliases, index):
    return aliases[index] if index < len(aliases) and isinstance(aliases[index], list) else []


def match_table(pred, gold, aliases):
    if not valid_value(gold, "table"):
        return False
    actual_names = [normalize_name(col) for col in pred["columns"]]
    expected_names = [normalize_name(col) for col in gold["columns"]]
    if len(actual_names) != len(expected_names) or len(pred["rows"]) != len(gold["rows"]):
        return False
    order = range(len(expected_names))
    if len(set(actual_names)) == len(actual_names) and set(actual_names) == set(expected_names):
        order = [actual_names.index(name) for name in expected_names]
    actual_rows = [[row[i] for i in order] for row in pred["rows"]]
    graph = []
    for index, expected in enumerate(gold["rows"]):
        graph.append([
            k for k, row in enumerate(actual_rows)
            if (match_cell(row[0], expected[0]) or any(match_cell(row[0], alias)
                                                       for alias in item_aliases(aliases, index)))
            and all(match_cell(a, b) for a, b in zip(row[1:], expected[1:]))
        ])
    owners = {}

    def assign(index, seen):
        for k in graph[index]:
            if k in seen:
                continue
            seen.add(k)
            if k not in owners or assign(owners[k], seen):
                owners[k] = index
                return True
        return False

    return all(assign(i, set()) for i in range(len(graph)))


def approx_match_value(pred, gold, kind, aliases=None):
    aliases = aliases or []
    if not valid_value(pred, kind):
        return False
    if kind == "table":
        return match_table(pred, gold, aliases)
    if kind == "list":
        return isinstance(gold, list) and len(pred) == len(gold) and all(
            any(match_cell(p, candidate) for candidate in [g, *item_aliases(aliases, i)])
            for i, (p, g) in enumerate(zip(pred, gold)))
    if kind == "date":
        return pred in [gold, *aliases]
    return any(match_cell(pred, candidate) for candidate in [gold, *aliases])


def approx_btc_score(submission_line, gold_row):
    kind = submission_line["type"]
    if kind == "abstain":
        return 0
    if kind != gold_row["type"]:
        return -1
    if kind == "unanswerable":
        return 1 if "answer" not in submission_line else -1
    return 1 if approx_match_value(submission_line.get("answer"), gold_row["answer"],
                                   kind, gold_row.get("alias")) else -1
