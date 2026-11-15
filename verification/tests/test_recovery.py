import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from datamate.abstention import CONFIDENCE_SOURCE
from datamate.inference import LlamaClient, infer
from datamate.server import LocalServer


FIXTURE = '''
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

class Handler(BaseHTTPRequestHandler):
    def send_json(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self.send_json({"data": [{"id": "fixture"}]})

    def do_POST(self):
        marker = Path(sys.argv[2])
        if not marker.exists():
            marker.touch()
            os._exit(1)
        self.send_json({"choices": [{"message": {"content": '{"type":"number","sql":"SELECT 7"}'}}]})

HTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
'''


class Recovery(unittest.TestCase):
    def test_http_read_timeout_is_model_unavailable(self):
        from datamate.inference import ModelUnavailable

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                time.sleep(0.2)

        with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as http:
            thread = threading.Thread(target=http.serve_forever, daemon=True)
            thread.start()
            try:
                client = LlamaClient(f"http://127.0.0.1:{http.server_port}")
                with self.assertRaises(ModelUnavailable):
                    client.request("/v1/models", timeout=0.05)
            finally:
                http.shutdown()

    def test_restart_process_and_retry_current_question(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            script = root / "server.py"
            script.write_text(FIXTURE)
            with socket.socket() as port_socket:
                port_socket.bind(("127.0.0.1", 0))
                port = port_socket.getsockname()[1]
            endpoint = f"http://127.0.0.1:{port}"
            server = LocalServer([sys.executable, str(script), str(port), str(root / "marker")],
                                 endpoint, root / "server.log", startup_timeout=5)
            import sqlite3
            db = root / "test.sqlite"
            sqlite3.connect(db).close()
            from unittest.mock import patch
            with server.running(), patch("datamate.inference.question_message", return_value="Question"):
                client = LlamaClient(endpoint, server, request_timeout=1)
                row, log = infer({"id": "q", "question": "Bao nhiêu?"}, db, {}, "", None, client, question_timeout=8)
                self.assertEqual(row, {"id": "q", "type": "number", "answer": 7})
                self.assertEqual(server.starts, 2)
                self.assertIn("model_error", log["attempts"][0])
            self.assertIsNone(server.process)

    def test_cli_keeps_all_rows_when_model_fails(self):
        class Handler(BaseHTTPRequestHandler):
            received = 0

            def do_GET(self):
                self.send_json({"data": [{"id": "fixture"}]})

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                Handler.received += 1
                if Handler.received == 1:
                    self.send_error(504)
                    return
                if Handler.received in (3, 4):
                    self.close_connection = True
                    return
                self.send_json({"choices": [{"message": {"content": '{"type":"number","sql":"SELECT 7"}'}}]})

            def send_json(self, obj):
                body = json.dumps(obj).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        with tempfile.TemporaryDirectory() as directory, ThreadingHTTPServer(("127.0.0.1", 0), Handler) as http:
            root = Path(directory)
            data = root / "data"
            biz = data / "data" / "biz"
            biz.mkdir(parents=True)
            (biz / "sales.csv").write_text("Order,Quantity\nA,1\n")
            (biz / "README.md").write_text("Sales")
            (data / "rulebook.md").write_text("## 1. Sales")
            questions = [{"id": f"q{i}", "business_id": "biz", "as_of_date": "2026-01-01", "question": "Bao nhiêu?"}
                         for i in range(500)]
            (data / "questions.jsonl").write_text("".join(json.dumps(q) + "\n" for q in questions))
            thread = threading.Thread(target=http.serve_forever, daemon=True)
            thread.start()
            out = root / "out"
            try:
                subprocess.run([sys.executable, "-m", "datamate", "predict", "--data", str(data), "--output",
                                str(out), "--work", str(root / "cache"), "--endpoint",
                                f"http://127.0.0.1:{http.server_port}"],
                               check=True, capture_output=True, text=True, timeout=30)
            finally:
                http.shutdown()
            rows = [json.loads(line) for line in (out / "submission.jsonl").read_text().splitlines()]
            logs = [json.loads(line) for line in (out / "logs/questions.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 500)
            self.assertEqual(len(logs), 500)
            self.assertEqual([r["type"] for r in rows[:3]], ["number", "abstain", "number"])
            self.assertEqual(rows[-1]["id"], "q499")
            self.assertIn("model_error", logs[1]["attempts"][-1])


    def test_frozen_threshold_through_predict_cli(self):
        class Handler(BaseHTTPRequestHandler):
            requests = []

            def send_json(self, data):
                body = json.dumps(data).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                self.send_json({"data": [{"id": "fixture"}]})

            def do_POST(self):
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                Handler.requests.append(request)
                i = len(Handler.requests) - 1
                choice = {"message": {"content": '{"type":"number","sql":"SELECT 7"}'}}
                if i < 2:
                    choice["logprobs"] = {"content": [{"logprob": [-1.0, 0.0][i]}]}
                self.send_json({"choices": [choice]})

            def log_message(self, *args):
                pass

        with tempfile.TemporaryDirectory() as directory, ThreadingHTTPServer(("127.0.0.1", 0), Handler) as http:
            root = Path(directory)
            biz = root / "data" / "biz"
            biz.mkdir(parents=True)
            (biz / "sales.csv").write_text("Order,Quantity\nA,1\n")
            (root / "rulebook.md").write_text("## 1. Sales")
            questions = [{"id": f"q{i}", "business_id": "biz", "as_of_date": "2026-01-01", "question": "How many?"}
                         for i in range(3)]
            (root / "questions.jsonl").write_text("".join(json.dumps(q) + "\n" for q in questions))
            config = root / "policy.json"
            config.write_text(json.dumps({"format_version": 1, "confidence_source": CONFIDENCE_SOURCE,
                                          "missing_confidence_policy": "keep_original", "threshold": 0.5}))
            thread = threading.Thread(target=http.serve_forever, daemon=True)
            thread.start()
            out = root / "out"
            try:
                subprocess.run([sys.executable, "-m", "datamate", "predict", "--data", str(root), "--output",
                                str(out), "--work", str(root / "cache"), "--endpoint",
                                f"http://127.0.0.1:{http.server_port}", "--threshold-config", str(config)],
                               check=True, capture_output=True, text=True, timeout=15)
            finally:
                http.shutdown()
            rows = [json.loads(line) for line in (out / "submission.jsonl").read_text().splitlines()]
            logs = [json.loads(line) for line in (out / "logs/questions.jsonl").read_text().splitlines()]
            self.assertTrue(all(request["logprobs"] is True for request in Handler.requests))
            self.assertEqual(rows[0], {"id": "q0", "type": "abstain"})
            self.assertEqual(logs[0]["candidate_answer"], {"type": "number", "answer": 7})
            self.assertEqual(logs[0]["abstain_reason"]["detail"], "confidence_below_threshold")
            self.assertEqual(rows[1:], [{"id": "q1", "type": "number", "answer": 7},
                                       {"id": "q2", "type": "number", "answer": 7}])
            self.assertIsNone(logs[2]["confidence"])
