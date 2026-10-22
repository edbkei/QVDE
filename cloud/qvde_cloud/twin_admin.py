"""
Digital-twin administration (the Orchestrator's governance functions).

    python -m qvde_cloud.twin_admin registrar-residencia --house house_001 --paciente PAC-0007 \
        --idade 81 --sexo F --contextos risco_queda,pressao_arterial,hipotensao_ortostatica_queda
    python -m qvde_cloud.twin_admin publicar-contextos --dir contexts
    python -m qvde_cloud.twin_admin registrar-modelo --nome rl --versao rl-ctx-v1 \
        --url https://.../rl.json --arquivo models/rl.json
    python -m qvde_cloud.twin_admin definir-desejado --house house_001 --modelo rl=rl-ctx-v1
    python -m qvde_cloud.twin_admin importar-pa --paciente PAC-0007 --csv pa.csv
    python -m qvde_cloud.twin_admin status --house house_001

The desired state is published as a retained MQTT message, so a Hub that was
offline receives it as soon as it reconnects.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import ssl
import sys
from pathlib import Path

from psycopg2.extras import Json

from qvde_common import protocol
from qvde_common.contexts import load_context_dir

from .db import apply_schema, connect


def register_residence(conn, house, patient, age=None, sex=None, conditions=(), meds=(), contexts=()):
    with conn, conn.cursor() as cur:
        cur.execute(
            """INSERT INTO pacientes (paciente_id, idade, sexo, condicoes, medicamentos, contextos_ativos)
               VALUES (%s,%s,%s,%s,%s,%s)
               ON CONFLICT (paciente_id) DO UPDATE SET idade=EXCLUDED.idade, sexo=EXCLUDED.sexo,
                 condicoes=EXCLUDED.condicoes, medicamentos=EXCLUDED.medicamentos,
                 contextos_ativos=EXCLUDED.contextos_ativos""",
            (patient, age, sex, list(conditions), list(meds), list(contexts)))
        cur.execute("""INSERT INTO residencias (house_id, paciente_id) VALUES (%s,%s)
                       ON CONFLICT (house_id) DO UPDATE SET paciente_id=EXCLUDED.paciente_id""",
                    (house, patient))


def publish_contexts(conn, directory) -> list[str]:
    """Store new context versions. Changing a published version without bumping it is refused."""
    published = []
    contexts = load_context_dir(directory)
    with conn, conn.cursor() as cur:
        for cid, ctx in contexts.items():
            text = (Path(directory) / f"{cid}.yaml").read_text(encoding="utf-8")
            cur.execute("SELECT yaml FROM contextos_repo WHERE id=%s AND versao=%s", (cid, ctx["versao"]))
            row = cur.fetchone()
            if row is None:
                cur.execute("INSERT INTO contextos_repo (id, versao, yaml) VALUES (%s,%s,%s)",
                            (cid, ctx["versao"], text))
                published.append(f"{cid} v{ctx['versao']}")
            elif row[0] != text:
                raise SystemExit(f"{cid} v{ctx['versao']} already published with different "
                                 "content: increase 'versao' in the YAML")
    return published


def register_model(conn, name, version, url, file=None, sha256=None, metrics=None):
    if file:
        sha256 = hashlib.sha256(Path(file).read_bytes()).hexdigest()
    if not sha256:
        raise SystemExit("give --arquivo (to compute the hash) or --sha256")
    if not url.startswith("https://"):
        raise SystemExit("model URLs must use https://")
    with conn, conn.cursor() as cur:
        cur.execute("""INSERT INTO modelos_repo (nome, versao, url, sha256, metricas) VALUES (%s,%s,%s,%s,%s)
                       ON CONFLICT (nome, versao) DO NOTHING""",
                    (name, version, url, sha256, Json(metrics or {})))
    return sha256


def build_desired(conn, house, models: dict[str, str] | None = None) -> dict:
    """Desired state: latest version of each active context (+ dependencies) and chosen models."""
    with conn.cursor() as cur:
        cur.execute("""SELECT p.contextos_ativos FROM residencias r JOIN pacientes p USING (paciente_id)
                       WHERE r.house_id=%s""", (house,))
        row = cur.fetchone()
        if not row:
            raise SystemExit(f"unknown residence {house}")
        active = list(row[0])
        cur.execute("""SELECT DISTINCT ON (id) id, versao, yaml FROM contextos_repo
                       ORDER BY id, versao DESC""")
        latest = {cid: (ver, text) for cid, ver, text in cur.fetchall()}
        missing = [c for c in active if c not in latest]
        if missing:
            raise SystemExit(f"contexts not published yet: {missing}")
        cur.execute("SELECT versao_estado, modelos FROM estado_desejado WHERE house_id=%s", (house,))
        prev = cur.fetchone()
        chosen = dict(prev[1]) if prev else {}
        chosen.update(models or {})
        model_items = {}
        for name, ver in chosen.items():
            cur.execute("SELECT url, sha256 FROM modelos_repo WHERE nome=%s AND versao=%s", (name, ver))
            m = cur.fetchone()
            if not m:
                raise SystemExit(f"model {name} {ver} not registered")
            model_items[name] = {"versao": ver, "url": m[0], "sha256": m[1]}
    return {
        "versao_estado": (prev[0] if prev else 0) + 1,
        "contextos_ativos": active,
        "contextos": {cid: {"versao": latest[cid][0], "yaml": latest[cid][1]} for cid in active},
        "modelos": model_items,
    }


def save_desired(conn, house, desired) -> None:
    with conn, conn.cursor() as cur:
        cur.execute(
            """INSERT INTO estado_desejado (house_id, versao_estado, contextos, modelos, atualizado)
               VALUES (%s,%s,%s,%s,now())
               ON CONFLICT (house_id) DO UPDATE SET versao_estado=EXCLUDED.versao_estado,
                 contextos=EXCLUDED.contextos, modelos=EXCLUDED.modelos, atualizado=now()""",
            (house, desired["versao_estado"],
             Json({c: v["versao"] for c, v in desired["contextos"].items()}),
             Json({n: m["versao"] for n, m in desired["modelos"].items()})))


def publish_desired(house, desired) -> None:  # pragma: no cover - network
    from qvde_common.mqtt_compat import make_client
    client = make_client(f"qvde-twin-admin-{os.getpid()}")
    if os.environ.get("MQTT_CA"):
        client.tls_set(ca_certs=os.environ["MQTT_CA"], tls_version=ssl.PROTOCOL_TLS_CLIENT)
    if os.environ.get("MQTT_USER"):
        client.username_pw_set(os.environ["MQTT_USER"], os.environ.get("MQTT_PASSWORD"))
    client.connect(os.environ.get("MQTT_HOST", "localhost"), int(os.environ.get("MQTT_PORT", 1883)))
    client.loop_start()
    client.publish(protocol.desired_topic(house), json.dumps(desired, ensure_ascii=False),
                   qos=1, retain=True).wait_for_publish(timeout=10)
    client.loop_stop()
    client.disconnect()


def import_bp(conn, patient, csv_path) -> int:
    """CSV columns: timestamp,sistolica,diastolica,fc,posicao,origem"""
    n = 0
    with open(csv_path, encoding="utf-8") as f, conn, conn.cursor() as cur:
        for row in csv.DictReader(f):
            cur.execute(
                """INSERT INTO observacoes_pa (paciente_id, ts, sistolica, diastolica, fc, posicao, origem)
                   VALUES (%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING""",
                (patient, row["timestamp"], int(row["sistolica"]), int(row["diastolica"]),
                 int(row["fc"]) if row.get("fc") else None, row.get("posicao") or "sentado",
                 row.get("origem")))
            n += cur.rowcount
    return n


def status(conn, house) -> dict:
    with conn.cursor() as cur:
        cur.execute("SELECT versao_estado, contextos, modelos FROM estado_desejado WHERE house_id=%s", (house,))
        desired = cur.fetchone()
        cur.execute("SELECT estado, atualizado FROM estado_reportado WHERE house_id=%s", (house,))
        reported = cur.fetchone()
    return {
        "desejado": dict(zip(("versao_estado", "contextos", "modelos"), desired)) if desired else None,
        "reportado": reported[0] if reported else None,
        "reportado_em": reported[1].isoformat() if reported else None,
        "sincronizado": bool(desired and reported and reported[0].get("versao_estado") == desired[0]),
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="twin_admin", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("registrar-residencia")
    r.add_argument("--house", required=True); r.add_argument("--paciente", required=True)
    r.add_argument("--idade", type=int); r.add_argument("--sexo")
    r.add_argument("--condicoes", default=""); r.add_argument("--medicamentos", default="")
    r.add_argument("--contextos", default="risco_queda")
    p = sub.add_parser("publicar-contextos"); p.add_argument("--dir", default="contexts")
    m = sub.add_parser("registrar-modelo")
    m.add_argument("--nome", required=True); m.add_argument("--versao", required=True)
    m.add_argument("--url", required=True); m.add_argument("--arquivo"); m.add_argument("--sha256")
    d = sub.add_parser("definir-desejado")
    d.add_argument("--house", required=True)
    d.add_argument("--modelo", action="append", default=[], help="nome=versao")
    d.add_argument("--sem-publicar", action="store_true")
    i = sub.add_parser("importar-pa"); i.add_argument("--paciente", required=True); i.add_argument("--csv", required=True)
    s = sub.add_parser("status"); s.add_argument("--house", required=True)
    sub.add_parser("aplicar-esquema")
    args = ap.parse_args(argv)

    conn = connect()
    split = lambda s: [x.strip() for x in s.split(",") if x.strip()]  # noqa: E731
    if args.cmd == "aplicar-esquema":
        apply_schema(conn); print("schema applied")
    elif args.cmd == "registrar-residencia":
        register_residence(conn, args.house, args.paciente, args.idade, args.sexo,
                           split(args.condicoes), split(args.medicamentos), split(args.contextos))
        print(f"{args.house} -> {args.paciente}")
    elif args.cmd == "publicar-contextos":
        print("\n".join(publish_contexts(conn, args.dir)) or "nothing new")
    elif args.cmd == "registrar-modelo":
        print(register_model(conn, args.nome, args.versao, args.url, args.arquivo, args.sha256))
    elif args.cmd == "definir-desejado":
        desired = build_desired(conn, args.house, dict(x.split("=", 1) for x in args.modelo))
        save_desired(conn, args.house, desired)
        if not args.sem_publicar:
            publish_desired(args.house, desired)
        print(f"desired state v{desired['versao_estado']} for {args.house}")
    elif args.cmd == "importar-pa":
        print(f"{import_bp(conn, args.paciente, args.csv)} observations imported")
    elif args.cmd == "status":
        print(json.dumps(status(conn, args.house), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
