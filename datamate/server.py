"""Own the offline llama-server process and recover it after transport failure."""

import os
import subprocess
from contextlib import contextmanager
from time import monotonic, sleep
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener


class ModelUnavailable(RuntimeError):
    pass


class LocalServer:
    def __init__(self, command, endpoint, log_path, startup_timeout=90, cooldown=15, env=None):
        self.command = command
        self.endpoint = endpoint
        self.log_path = log_path
        self.startup_timeout = startup_timeout
        self.cooldown = cooldown
        self.http = build_opener(ProxyHandler({}))
        self.process = None
        self.retry_after = 0
        self.starts = 0
        self.env = {**os.environ, **(env or {}), "CUDA_VISIBLE_DEVICES": "0"}

    def health(self, timeout=2):
        try:
            with self.http.open(self.endpoint + "/health", timeout=timeout) as response:
                return response.status == 200
        except (OSError, URLError):
            return False

    def ensure_running(self, deadline):
        if self.process is not None and self.process.poll() is None:
            return
        if monotonic() < self.retry_after:
            raise ModelUnavailable("llama-server startup is in cooldown")
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise ModelUnavailable("question time budget expired")
        stop_at = monotonic() + min(remaining, self.startup_timeout)
        self.close()
        try:
            with self.log_path.open("a", encoding="utf-8") as log:
                print(f"Starting llama-server (attempt {self.starts + 1})", file=log, flush=True)
                self.process = subprocess.Popen(self.command, env=self.env,
                                                stdout=log, stderr=subprocess.STDOUT)
                self.starts += 1
            while self.process.poll() is None and monotonic() < stop_at:
                if self.health(min(1, max(0.05, stop_at - monotonic()))) and self.process.poll() is None:
                    return
                sleep(min(0.25, max(0, stop_at - monotonic())))
            raise ModelUnavailable(f"llama-server did not become ready (exit {self.process.poll()}); see {self.log_path}")
        except (OSError, ModelUnavailable) as exc:
            self.close()
            self.retry_after = monotonic() + self.cooldown
            raise ModelUnavailable(str(exc)) from exc

    def recover(self, deadline):
        if self.process is not None:
            self.close()
        self.ensure_running(deadline)

    def close(self):
        process, self.process = self.process, None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    @contextmanager
    def running(self):
        try:
            yield self
        finally:
            self.close()
