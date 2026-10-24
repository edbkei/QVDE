"""
Report job of the deliberative loop: snapshot from the twin -> deterministic
findings per care context -> LLM write-up -> numeric fidelity F(y) -> stored.

    python -m qvde_cloud.report_job --paciente PAC-0007 [--perfil contexts/perfis/geriatra.yaml]
                                    [--dias 7] [--saida relatorios/] [--somente-contexto]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
from datetime import datetime
from pathlib import Path

from psycopg2.extras import Json

from . import relatorio as R
from .db import connect
from .snapshot import build_snapshot


def generate(conn, patient_id: str, profile_path: str, days: int = 7, contexts_dir: str | None = None,
             call_llm: bool = True) -> dict:
    perfil = R.carregar_config(profile_path)
    if os.environ.get("OLLAMA_HOST"):
        perfil["llm"]["host"] = os.environ["OLLAMA_HOST"]
    snapshot = build_snapshot(conn, patient_id, days=days)
    ids, ignored = R.contextos_efetivos(perfil, snapshot)
    results, _ = R.executar_contextos(snapshot, ids, contexts_dir or str(R.CONTEXTOS_DIR))
    pac = snapshot["paciente"]
    llm_context = {"paciente": {k: pac.get(k) for k in ("idade", "sexo", "condicoes", "medicamentos")},
                   "contextos": {r["nome"]: (r["achados"] if r["status"] == "avaliado"
                                             else ["não avaliado"] + r["achados"]) for r in results.values()}}
    messages = R.montar_mensagens(perfil, llm_context)
    text, metrics = "*Modelo de linguagem não chamado.*", {}
    if call_llm:
        try:
            text, metrics = R.chamar_llm(messages, perfil["llm"])
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            text = f"*Análise do modelo indisponível: {exc}*"
    f_y, suspects = R.fidelidade(text, llm_context, snapshot, perfil) if metrics else (None, [])
    report = R.montar_relatorio(snapshot, perfil, results, ignored, text, metrics, suspects)
    if f_y is not None:
        metrics["fidelidade_F"] = f_y
    return {"perfil": perfil, "snapshot": snapshot, "markdown": report, "texto_llm": text,
            "metricas": metrics, "fidelidade": f_y, "suspeitos": suspects, "mensagens": messages}


def store(conn, patient_id: str, out: dict) -> int:
    p = out["perfil"]
    with conn, conn.cursor() as cur:
        cur.execute(
            """INSERT INTO relatorios (paciente_id, perfil, modelo_llm, prompt_versao, snapshot, texto,
                   fidelidade, suspeitos, metricas) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
            (patient_id, p["id"], p["llm"]["modelo"], p["prompt"]["versao"], Json(out["snapshot"]),
             out["markdown"], out["fidelidade"], out["suspeitos"], Json(out["metricas"])))
        return cur.fetchone()[0]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paciente", required=True)
    ap.add_argument("--perfil", default=str(R.PERFIL_PADRAO))
    ap.add_argument("--dias", type=int, default=7)
    ap.add_argument("--saida", default="relatorios")
    ap.add_argument("--somente-contexto", action="store_true", help="print the prompt; no LLM call, nothing stored")
    args = ap.parse_args(argv)

    conn = connect()
    out = generate(conn, args.paciente, args.perfil, args.dias, call_llm=not args.somente_contexto)
    if args.somente_contexto:
        for m in out["mensagens"]:
            print(f"----- {m['role']} -----\n{m['content']}\n")
        return 0
    rid = store(conn, args.paciente, out)
    Path(args.saida).mkdir(parents=True, exist_ok=True)
    path = Path(args.saida) / f"relatorio_{args.paciente}_{datetime.now():%Y%m%d_%H%M}.md"
    path.write_text(out["markdown"], encoding="utf-8")
    print(f"report {rid} saved to {path}", file=sys.stderr)
    print(json.dumps(out["metricas"], ensure_ascii=False, indent=2), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
