"""Start llama.cpp's llama-server for the local model, with its chat template's date pinned.

SmolLM3's chat template writes today's date into every prompt, so the same question would
get a different prompt (and sometimes a different answer) each day. A regression test needs
identical prompts, so the template is read from the server once, its date call replaced by a
fixed date, and the server started with that copy.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import os
import re
import socket
import subprocess
import time
from pathlib import Path

import httpx

from .config import CACHE_DIR, env_path

PINNED_DATE = dt.date(2026, 1, 15)
LOG_DIR = CACHE_DIR / "logs"
TEMPLATE_DIR = CACHE_DIR / "templates"


class ServerError(RuntimeError):
    pass


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def pin_template_date(template: str, day: dt.date = PINNED_DATE) -> str:
    """Replace strftime_now("<format>") calls in a Jinja chat template with a fixed date."""
    return re.sub(
        r"strftime_now\(\s*(['\"])(.*?)\1\s*\)",
        lambda m: '"' + day.strftime(m.group(2)) + '"',
        template,
    )


class LlamaServer:
    """One llama-server process. Use as a context manager; it stops the server on exit."""

    def __init__(self, exe: Path | None = None, model: Path | None = None, ctx: int = 8192, gpu_layers: int = 99):
        self.exe = exe or env_path("BANKRAG_LLAMA_SERVER")
        self.model = model or env_path("BANKRAG_LOCAL_GGUF")
        if not self.exe or not self.exe.exists():
            raise ServerError("set BANKRAG_LLAMA_SERVER in .env to the path of llama-server(.exe)")
        if not self.model or not self.model.exists():
            raise ServerError("set BANKRAG_LOCAL_GGUF in .env to the path of a .gguf model")
        self.ctx = ctx
        self.gpu_layers = gpu_layers
        self.proc: subprocess.Popen | None = None
        self.port = 0
        self._log = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def _start(self, template_file: Path | None, timeout: float = 180) -> None:
        self.port = _free_port()
        cmd = [
            str(self.exe), "-m", str(self.model),
            "-c", str(self.ctx), "-ngl", str(self.gpu_layers), "-np", "1",
            "--jinja", "--host", "127.0.0.1", "--port", str(self.port),
        ]
        if template_file:
            cmd += ["--chat-template-file", str(template_file)]
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self._log = open(LOG_DIR / "llama-server.log", "w", encoding="utf-8", errors="replace")
        self.proc = subprocess.Popen(cmd, stdout=self._log, stderr=subprocess.STDOUT)
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout:
            if self.proc.poll() is not None:
                raise ServerError(f"llama-server exited with code {self.proc.returncode}; see {LOG_DIR / 'llama-server.log'}")
            try:
                if httpx.get(f"{self.base_url}/health", timeout=2).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.2)
        self.stop()
        raise ServerError(f"llama-server did not load the model within {timeout:.0f} s")

    def _pinned_template(self) -> Path:
        stat = self.model.stat()
        key = hashlib.sha1(f"{self.model.resolve()}|{stat.st_size}|{PINNED_DATE}".encode()).hexdigest()[:12]
        path = TEMPLATE_DIR / f"{self.model.stem}-{key}.jinja"
        if path.exists():
            return path
        self._start(template_file=None)
        try:
            template = httpx.get(f"{self.base_url}/props", timeout=10).json()["chat_template"]
        finally:
            self.stop()
        TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(pin_template_date(template), encoding="utf-8", newline="\n")
        return path

    def start(self) -> "LlamaServer":
        self._start(template_file=self._pinned_template())
        return self

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None
        if self._log:
            self._log.close()
            self._log = None

    def __enter__(self) -> "LlamaServer":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


def external_url() -> str | None:
    """An OpenAI-compatible server that is already running (Ollama, LM Studio, llama-server)."""
    return os.environ.get("BANKRAG_LOCAL_URL") or None
