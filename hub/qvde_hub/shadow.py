"""
Local shadow (sombra local) of the digital twin's desired state.

The cloud publishes a retained "desejado" message per house with the care
contexts (YAML text + version), the active contexts and the model versions.
The Hub stores the last applied state on disk and keeps operating with it
while offline (NFR1). Model files are downloaded over HTTPS and accepted only
if their SHA-256 matches; otherwise the previous model stays in use.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path

from qvde_common.contexts import ContextError, load_context_dir, parse_context


class LocalShadow:
    def __init__(self, directory: str | Path, bundled_contexts: str | Path | None = None):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "contexts").mkdir(exist_ok=True)
        (self.dir / "models").mkdir(exist_ok=True)
        self.bundled = Path(bundled_contexts) if bundled_contexts else None
        self.state = self._read()

    # ---------------------------------------------------------------- state
    def _read(self) -> dict:
        f = self.dir / "shadow.json"
        if f.exists():
            return json.loads(f.read_text(encoding="utf-8"))
        return {"versao_estado": 0, "contextos": {}, "contextos_ativos": None, "modelos": {}}

    def _write(self) -> None:
        tmp = self.dir / "shadow.json.tmp"
        tmp.write_text(json.dumps(self.state, indent=1, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.dir / "shadow.json")

    @property
    def version(self) -> int:
        return int(self.state.get("versao_estado", 0))

    def reported(self) -> dict:
        return {
            "versao_estado": self.version,
            "contextos": {cid: c["versao"] for cid, c in self.state["contextos"].items()},
            "modelos": {name: m.get("versao") for name, m in self.state["modelos"].items()},
        }

    # ------------------------------------------------------------- contexts
    def contexts(self) -> dict[str, dict]:
        """Contexts from the shadow; bundled files are the fallback before the first sync."""
        if self.state["contextos"]:
            return {cid: parse_context(c["yaml"]) for cid, c in self.state["contextos"].items()}
        return load_context_dir(self.bundled) if self.bundled else {}

    def active_contexts(self) -> list[str] | None:
        return self.state.get("contextos_ativos")

    def model_path(self, name: str) -> str | None:
        m = self.state["modelos"].get(name)
        return m.get("arquivo") if m else None

    # ---------------------------------------------------------------- apply
    def apply_desired(self, desired: dict) -> dict:
        """Apply a desired state; returns what changed. Invalid parts are rejected, not applied."""
        changed = {"contextos": [], "modelos": [], "erros": []}
        if int(desired.get("versao_estado", 0)) <= self.version:
            return changed

        new_contexts = dict(self.state["contextos"])
        for cid, item in desired.get("contextos", {}).items():
            current = new_contexts.get(cid)
            if current and current["versao"] == item["versao"]:
                continue
            try:
                ctx = parse_context(item["yaml"])
                if ctx["id"] != cid or ctx["versao"] != item["versao"]:
                    raise ContextError("id/versao in YAML differ from the envelope")
            except Exception as exc:
                changed["erros"].append(f"contexto {cid}: {exc}")
                continue
            new_contexts[cid] = {"versao": item["versao"], "yaml": item["yaml"]}
            (self.dir / "contexts" / f"{cid}.yaml").write_text(item["yaml"], encoding="utf-8")
            changed["contextos"].append(cid)

        new_models = dict(self.state["modelos"])
        for name, item in desired.get("modelos", {}).items():
            current = new_models.get(name, {})
            if current.get("versao") == item.get("versao") and current.get("arquivo"):
                continue
            try:
                path = self._fetch_model(name, item)
            except Exception as exc:
                changed["erros"].append(f"modelo {name}: {exc}")
                continue
            new_models[name] = {"versao": item["versao"], "sha256": item.get("sha256"), "arquivo": path}
            changed["modelos"].append(name)

        self.state = {
            "versao_estado": int(desired["versao_estado"]),
            "contextos": new_contexts,
            "contextos_ativos": desired.get("contextos_ativos", self.state.get("contextos_ativos")),
            "modelos": new_models,
        }
        self._write()
        return changed

    def _fetch_model(self, name: str, item: dict) -> str:
        url, digest = item.get("url"), item.get("sha256")
        if not url or not digest:
            raise ValueError("model entry needs url and sha256")
        if not url.startswith(("https://", "file://")):
            raise ValueError("only https:// (or file:// for tests) model URLs are accepted")
        suffix = Path(url).suffix or ".bin"
        target = self.dir / "models" / f"{name}_{item['versao']}{suffix}"
        with tempfile.NamedTemporaryFile(dir=self.dir / "models", delete=False) as tmp:
            with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 (scheme checked above)
                shutil.copyfileobj(resp, tmp)
            tmp_path = tmp.name
        h = hashlib.sha256(Path(tmp_path).read_bytes()).hexdigest()
        if h != digest:
            os.unlink(tmp_path)
            raise ValueError("sha256 mismatch; download discarded")
        os.replace(tmp_path, target)
        return str(target)
