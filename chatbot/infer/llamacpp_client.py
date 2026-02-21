from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional


class LlamaCppError(RuntimeError):
    """Raised when the local llama.cpp server call fails."""


@dataclass
class LlamaCppClient:
    """Small OpenAI-compatible client for local llama-server."""

    base_url: str = "http://127.0.0.1:8081"
    model: str = "medical-chatbot"
    timeout_seconds: float = 60.0

    def _url(self, path: str) -> str:
        return urllib.parse.urljoin(self.base_url.rstrip("/") + "/", path.lstrip("/"))

    def _request(self, method: str, path: str, payload: Optional[Dict] = None) -> Dict:
        data = None
        headers = {"Content-Type": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(
            self._url(path),
            data=data,
            headers=headers,
            method=method,
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
                if not body:
                    return {}
                return json.loads(body)
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="ignore")
            raise LlamaCppError(f"llama-server HTTP {exc.code}: {details}") from exc
        except urllib.error.URLError as exc:
            raise LlamaCppError(f"Could not reach llama-server at {self.base_url}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise LlamaCppError(f"Invalid JSON from llama-server: {exc}") from exc

    def health(self) -> Dict:
        """Returns runtime status for local inference service."""
        try:
            data = self._request("GET", "/health")
            return {"ok": True, "provider": "health", "details": data}
        except LlamaCppError:
            # Not all builds expose /health; fall back to models endpoint.
            try:
                data = self._request("GET", "/v1/models")
                return {"ok": True, "provider": "v1/models", "details": data}
            except LlamaCppError as exc:
                return {"ok": False, "error": str(exc)}

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        data = self._request("POST", "/v1/chat/completions", payload)
        choices = data.get("choices") or []
        if not choices:
            raise LlamaCppError(f"No completion choices returned: {data}")

        message = choices[0].get("message") or {}
        content = message.get("content", "").strip()
        if not content:
            raise LlamaCppError("Received empty response from llama-server")

        return content
