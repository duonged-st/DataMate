"""Run one Qwen request and one repair at most, then execute read-only SQL."""

import http.client
import json
import math
import re
import sqlite3
import time
import urllib.error
import urllib.request
from contextlib import closing

from .abstention import CONFIDENCE_SOURCE
from .answer import ALLOWED_TYPES, btc_answer_row
from .business_sql import basic_sales, debt_query, sales_totals
from .data import open_readonly
from .normalize import as_date, as_number, key
from .periods import DatePredicateError, validate_date_predicates
from .prompt import SYSTEM, question_message
from .server import ModelUnavailable
from .sql_rules import expected_type, known_customer, order_status, product_hints, real_percent_sql, sales_query
from .unanswerable import evidence_for_retry, verified_gap


class LlamaClient:
    def __init__(self, base_url, server=None, request_timeout=45):
        self.base = base_url.rstrip("/")
        self.http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.model = None
        self.server = server
        self.request_timeout = request_timeout

    def request(self, path, body=None, timeout=5):
        request = urllib.request.Request(
            self.base + path,
            data=json.dumps(body).encode() if body is not None else None,
            headers={"Content-Type": "application/json"},
        )
        try:
            with self.http.open(request, timeout=timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read(900).decode(errors="replace")
            except (OSError, http.client.HTTPException) as read_error:
                raise ModelUnavailable(f"Cannot read llama-server HTTP {exc.code}: {read_error}") from exc
            finally:
                exc.close()
            if exc.code in (400, 422) and body is not None:
                raise ValueError(f"llama-server rejected request ({exc.code}): {detail}") from exc
            raise ModelUnavailable(f"llama-server HTTP {exc.code}: {detail}") from exc
        except (OSError, http.client.HTTPException, json.JSONDecodeError) as exc:
            raise ModelUnavailable(f"llama-server unavailable: {exc}") from exc

    def complete(self, system, message, deadline, log_confidence=False):
        if self.server is not None:
            self.server.ensure_running(deadline)
        if self.model is None:
            models = self.request("/v1/models", timeout=remaining_seconds(deadline, 5))
            try:
                self.model = models["data"][0]["id"]
            except (KeyError, IndexError, TypeError) as exc:
                raise ModelUnavailable("Invalid response from /v1/models") from exc
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": message}],
            "temperature": 0,
            "seed": 20261112,
            "max_tokens": 900,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if log_confidence:
            body["logprobs"] = True
            body["post_sampling_probs"] = False
        try:
            response = self.request("/v1/chat/completions", body,
                                    timeout=remaining_seconds(deadline, self.request_timeout))
        except ValueError:
            if not log_confidence:
                raise
            body.pop("logprobs")
            body.pop("post_sampling_probs")
            response = self.request("/v1/chat/completions", body,
                                    timeout=remaining_seconds(deadline, self.request_timeout))
        try:
            item = response["choices"][0]["message"]
            if not isinstance(item, dict):
                raise ValueError("message is not an object")
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ModelUnavailable("Invalid response from /v1/chat/completions") from exc
        raw = item.get("content") or ""
        thinking = item.get("reasoning_content") or item.get("reasoning")
        embedded = re.search(r"<think>(.*?)</think>", raw, re.I | re.S)
        if embedded:
            if not thinking:
                thinking = embedded[1] or None
            elif embedded[1] and embedded[1] != thinking:
                thinking = [thinking, embedded[1]]
            raw = raw[:embedded.start()] + raw[embedded.end():]
        evidence = None
        if log_confidence:
            logprobs = response["choices"][0].get("logprobs")
            evidence = token_confidence(logprobs) if "logprobs" in body else None
            if evidence is None:
                evidence = {"status": "request_rejected" if "logprobs" not in body else
                            "missing_logprobs" if logprobs is None else "invalid_logprobs"}
        return raw.strip(), thinking, evidence


    def recover(self, deadline):
        self.model = None
        if self.server is not None:
            self.server.recover(deadline)


def parse_plan(text):
    clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    value = json.loads(clean)
    if not isinstance(value, dict) or value.get("type") not in ALLOWED_TYPES:
        raise ValueError("The response must be a JSON object with a valid type")
    if value["type"] not in {"unanswerable", "abstain"}:
        sql = value.get("sql")
        if not isinstance(sql, str) or not re.match(r"^\s*(SELECT|WITH)\b", sql, flags=re.I):
            raise ValueError("The response needs a SELECT/WITH query")
    return value


def clean_generated_sql(sql):
    """Remove one stray JSON closing brace after a complete SQL statement."""
    end = sql.rstrip()
    if end.endswith("}"):
        candidate = end[:-1].rstrip()
        if sqlite3.complete_statement(candidate.rstrip(";") + ";"):
            return candidate
    return sql


def remaining_seconds(deadline, cap):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise ModelUnavailable("question time budget expired")
    return min(remaining, cap)


def authorize(action, arg1, arg2, database, trigger):
    if action == sqlite3.SQLITE_FUNCTION:
        return sqlite3.SQLITE_DENY if (arg2 or arg1 or "").casefold() in {
            "load_extension", "readfile", "writefile", "fts3_tokenizer"
        } else sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_READ:
        return sqlite3.SQLITE_OK if database in (None, "main") and arg1 not in {"sqlite_schema", "sqlite_master"} else sqlite3.SQLITE_DENY
    if action in {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_RECURSIVE}:
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


def execute_sql(path, sql, limit=250, seconds=12):
    deadline = time.monotonic() + seconds
    with closing(open_readonly(path)) as con:
        con.execute("PRAGMA query_only=ON")
        if hasattr(con, "setconfig") and hasattr(sqlite3, "SQLITE_DBCONFIG_DQS_DML"):
            con.setconfig(sqlite3.SQLITE_DBCONFIG_DQS_DML, False)
        con.create_function("vn_key", 1, lambda x: key(x) if x is not None else None, deterministic=True)
        con.create_function("vn_money", 1, lambda x: as_number(x, "money") if x is not None else None, deterministic=True)
        con.create_function("vn_quantity", 1, lambda x: as_number(x, "quantity") if x is not None else None, deterministic=True)
        con.create_function("vn_date", 1, lambda x: as_date(x) if x is not None else None, deterministic=True)
        con.create_function("vn_order_status", 1, order_status, deterministic=True)
        con.create_function("vn_known_customer", 1, known_customer, deterministic=True)
        con.set_authorizer(authorize)
        con.set_progress_handler(lambda: int(time.monotonic() > deadline), 5000)
        cursor = con.execute(sql)
        columns = [r[0] for r in cursor.description]
        rows = cursor.fetchmany(limit+1)
        if len(rows) > limit:
            raise ValueError(f"SQL returned more than {limit} rows")
        return {"columns": columns, "rows": [list(r) for r in rows]}


def answer_kind(q, kind, result):
    expected = expected_type(q["question"])
    if not expected or kind == expected:
        return kind
    rows = result["rows"]
    if expected == "list" and rows and len(result["columns"]) == 1:
        return expected
    if len(rows) == 1 and len(result["columns"]) == 1:
        return expected
    raise ValueError(f"Question requires {expected}; SQL returned a shape for {kind}")


def answer_from_result(q, kind, result):
    rows = result["rows"]
    columns = result["columns"]
    if kind == "table":
        value = result
    elif kind == "list":
        if not rows or len(columns) == 1:
            value = [r[0] for r in rows]
        elif len(rows) == 1:
            value = rows[0]
        else:
            raise ValueError("List requires one column or one row")
        if any(v is None for v in value):
            raise ValueError("List contains a missing value")
    else:
        if len(rows) != 1 or len(columns) != 1 or rows[0][0] is None:
            raise ValueError(f"{kind} requires exactly one non-null cell")
        value = rows[0][0]
        if kind == "date" and isinstance(value, str):
            normalized = as_date(value)
            if normalized and len(normalized) >= 10:
                value = normalized[:10]
    return btc_answer_row(q, kind, value)


def loggable_result(result):
    def cell(value):
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        if isinstance(value, bytes):
            return {"blob_hex": value.hex()}
        return value
    return {"columns": result["columns"], "rows": [[cell(value) for value in row] for row in result["rows"]]}


def token_confidence(logprobs):
    tokens = logprobs.get("content") if isinstance(logprobs, dict) else None
    if not isinstance(tokens, list) or not tokens:
        return None
    values = [t.get("logprob") if isinstance(t, dict) else None for t in tokens]
    if not all(type(v) in (int, float) and -1e10 < v <= 0 for v in values):
        return None
    mean = math.fsum(values) / len(values)
    return {"token_count": len(values), "mean_token_logprob": mean,
            "confidence": math.exp(mean)}


def infer(q, path, catalog, readme, context, client, question_timeout=120, log_confidence=False):
    start = time.monotonic()
    deadline = start + question_timeout
    attempts = []
    transport_retry_used = False
    thoughts = []
    error = None
    final = None
    evidence = None
    abstain_reason = None
    products = product_hints(path, catalog, q["question"]) if catalog.get("tables") else []
    gap = verified_gap(q, catalog) if catalog.get("tables") else None
    if gap:
        final = {"id": q["id"], "type": "unanswerable", "reason": gap}
        attempts.append({"source": "catalog_evidence", "missing_data": gap})
    rule = None
    if final is None and catalog.get("tables"):
        rule = (sales_query(q, catalog, products) or basic_sales(q, catalog, products)
                or sales_totals(q, catalog, products) or debt_query(q, catalog))
    if rule:
        kind, sql = rule
        record = {"source": "rulebook_sql", "code": sql, "execution_started": True}
        try:
            result = execute_sql(path, sql, seconds=remaining_seconds(deadline, 12))
            record["output"] = loggable_result(result)
            final = answer_from_result(q, kind, result)
        except (ValueError, sqlite3.Error, TypeError, ModelUnavailable) as exc:
            record["error"] = str(exc)
            error = f"Truy vấn từ quy tắc chưa trả được kết quả. SQL: {sql}\nLỗi: {exc}"
        attempts.append(record)
    for _ in range(2):
        if final is not None:
            break
        record = {}
        try:
            message = question_message(q, context, catalog, readme, error, products)
            while True:
                try:
                    raw, thinking, token_evidence = client.complete(SYSTEM, message, deadline, log_confidence)
                    break
                except ModelUnavailable as exc:
                    attempts.append({"model_error": str(exc)})
                    if transport_retry_used or time.monotonic() >= deadline:
                        raise
                    transport_retry_used = True
                    client.recover(deadline)
            if isinstance(thinking, list):
                thoughts.extend(thinking)
            elif thinking is not None:
                thoughts.append(thinking)
            record["response"] = raw
            if token_evidence is not None:
                record["confidence_evidence"] = token_evidence
            plan = parse_plan(raw)
            kind = plan["type"]
            if kind in {"unanswerable", "abstain"}:
                reason = str(plan.get("reason", "")).strip()
                if kind == "unanswerable":
                    record["unanswerable_rejected"] = reason or "No reason supplied"
                    raise ValueError("Chưa có bằng chứng dữ liệu không đủ. " + evidence_for_retry(q, catalog))
                final = btc_answer_row(q, kind, None)
                abstain_reason = {"category": "abstain", "detail": reason or "Model requested abstain"}
            else:
                record["generated_sql"] = plan["sql"]
                sql = clean_generated_sql(plan["sql"])
                if kind == "percent" or expected_type(q["question"]) == "percent":
                    sql = real_percent_sql(sql)
                validate_date_predicates(q, sql, catalog)
                record["code"] = sql
                seconds = remaining_seconds(deadline, 12)
                record["execution_started"] = True
                result = execute_sql(path, sql, seconds=seconds)
                record["output"] = loggable_result(result)
                final = answer_from_result(q, answer_kind(q, kind, result), result)
            if log_confidence and final["type"] != "abstain" and token_evidence is not None:
                evidence = token_evidence
            attempts.append(record)
            break
        except ModelUnavailable as exc:
            record["model_error"] = str(exc)
            abstain_reason = {"category": "execution", "detail": str(exc)}
            if not attempts or record != attempts[-1]:
                attempts.append(record)
            break
        except (ValueError, sqlite3.Error, TypeError) as exc:
            record["error"] = str(exc)
            if "unanswerable_rejected" in record:
                category = "abstain"
            elif isinstance(exc, DatePredicateError):
                category = "sql_generation"
            elif "output" in record:
                category = "formatting"
            elif isinstance(exc, sqlite3.Error):
                if re.search(r"no such (column|table)|ambiguous column", str(exc), re.I):
                    category = "schema_linking"
                elif re.search(r"syntax error|incomplete input|one statement at a time|unrecognized token", str(exc), re.I):
                    category = "sql_generation"
                else:
                    category = "execution"
            elif record.get("execution_started"):
                category = "execution"
            else:
                category = "sql_generation"
            abstain_reason = {"category": category, "detail": str(exc)}
            attempts.append(record)
            error = f"Phản hồi trước: {record.get('response', '')}\nLỗi: {exc}"
    if final is None:
        final = {"id": q["id"], "type": "abstain"}
    executed = [a for a in attempts if a.get("execution_started")]
    last = executed[-1] if executed else {}
    log = {"id": q["id"], "answer": {k: v for k, v in final.items() if k != "id"},
           "elapsed_ms": int((time.monotonic() - start) * 1000),
           "thinking": thoughts[0] if len(thoughts) == 1 else thoughts or None,
           "code": last.get("code"), "output": last.get("output"),
           "attempts": attempts,
           "candidate_answer": {k: v for k, v in final.items() if k != "id"} if final["type"] != "abstain" else None,
           "abstain_reason": abstain_reason if final["type"] == "abstain" else None,
           "confidence": evidence.get("confidence") if evidence else None,
           "confidence_source": CONFIDENCE_SOURCE if evidence and "confidence" in evidence else None,
           "confidence_evidence": evidence}
    return final, log
