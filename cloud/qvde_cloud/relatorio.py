#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[Cloud copy] Same engine as the standalone qvde_relatorio.py; contexts are read from
the repository-level contexts/ directory, shared with the Hub. Used by report_job.py.

qvde_relatorio.py (v3) — relatório para especialista orientado a CONTEXTOS DE CUIDADO.

Configuração (pasta config/):
  contextos/<id>.yaml   o que é monitorado: fontes, limiares, contexto situacional, ações.
                        O bloco "gemeo:" descreve o gêmeo digital (futuro) e é ignorado aqui.
  relatorios/<id>.yaml  perfil de relatório (lado do LLM): destinatário, contextos, modelo, prompt.

Fluxo:
  snapshot JSON do gêmeo -> contextos ativos (perfil ∩ paciente) -> processador de cada
  contexto (determinístico) -> achados agrupados por contexto -> LLM interpreta -> validação
  de números -> relatório Markdown.

Uso:
  python3 qvde_relatorio.py exemplo_entrada.json
  python3 qvde_relatorio.py exemplo_entrada.json --perfil config/relatorios/geriatra.yaml
  python3 qvde_relatorio.py exemplo_entrada.json --modelo qwen2.5:3b --saida rel_3b.md
  python3 qvde_relatorio.py exemplo_entrada.json --somente-contexto

Requer Python 3.8+ e PyYAML (pip install pyyaml); arquivos de configuração .json também são aceitos.
"""

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean
from string import Template

try:
    import yaml
except ImportError:
    yaml = None

PASTA_SCRIPT = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[2]
CONTEXTOS_DIR = REPO / "contexts"
PERFIL_PADRAO = CONTEXTOS_DIR / "perfis" / "geriatra.yaml"


# =============================================================================
# 1. UTILITÁRIOS
# =============================================================================
def carregar_config(caminho):
    caminho = Path(caminho)
    with open(caminho, encoding="utf-8") as f:
        if caminho.suffix in (".yaml", ".yml"):
            if yaml is None:
                sys.exit("Erro: instale o PyYAML (pip install pyyaml) ou use arquivos .json.")
            return yaml.safe_load(f)
        return json.load(f)


def carregar_contexto(pasta, cid):
    for ext in (".yaml", ".yml", ".json"):
        p = Path(pasta) / f"{cid}{ext}"
        if p.exists():
            return carregar_config(p)
    sys.exit(f"Erro: contexto '{cid}' não encontrado em {pasta}")


def obter_campo(dados, caminho):
    atual = dados
    for parte in caminho.split("."):
        if not isinstance(atual, dict) or parte not in atual:
            return None
        atual = atual[parte]
    return atual


def ts(texto):
    return datetime.fromisoformat(texto)


def inclinacao(xs, ys):
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else 0.0


def tabela_md(cabecalho, linhas):
    out = ["| " + " | ".join(cabecalho) + " |", "|" + "---|" * len(cabecalho)]
    out += ["| " + " | ".join(str(c) for c in linha) + " |" for linha in linhas]
    return "\n".join(out)


# =============================================================================
# 2. CONTEXTO SITUACIONAL (Dey, 2001)
# =============================================================================
def periodo_do_dia(momento, faixas):
    for nome, (ini, fim) in faixas.items():
        if ini <= momento.hour < fim:
            return nome
    return None


def descrever_situacao(evento, cfg_sit):
    if not cfg_sit:
        return ""
    partes = []
    if "periodo_do_dia" in cfg_sit:
        p = periodo_do_dia(ts(evento["timestamp"]), cfg_sit["periodo_do_dia"])
        if p:
            partes.append(f"período: {p}")
    for campo in cfg_sit.get("campos_evento", []):
        if evento.get(campo):
            partes.append(f"{campo.replace('_', ' ')}: {evento[campo]}")
    return "; ".join(partes)


# =============================================================================
# 3. PROCESSADORES DE CONTEXTO (determinísticos)
#    Cada um recebe (entrada, cfg_contexto, resultados_anteriores) e devolve
#    {"achados": [...], "tabelas": [...], "dados": {...}}
# =============================================================================
def proc_risco_queda(entrada, ctx, _resultados):
    lim = ctx.get("limiares", {})
    faixas = lim.get("classe_risco", {"moderado": 0.3, "alto": 0.6})
    prob = entrada["probabilidade"]
    classe = "alto" if prob >= faixas["alto"] else "moderado" if prob >= faixas["moderado"] else "baixo"
    inc = entrada.get("incerteza")

    if entrada.get("horizonte_horas"):
        texto = f"Probabilidade de queda nas próximas {entrada['horizonte_horas']} h: {prob}"
    else:  # proxy: daily aggregate of the Hub's detection, not a risk forecast
        texto = (f"Indicador diário de detecção ({entrada.get('modelo', 'proxy')}), "
                 f"sem horizonte de predição: {prob}")
    if inc is not None:
        texto += f" (incerteza ± {inc})"
    texto += f", classificada como risco {classe}"
    serie = sorted(entrada.get("tendencia", []), key=lambda p: p["data"])
    if len(serie) >= 2:
        t0 = ts(serie[0]["data"])
        xs = [(ts(p["data"]) - t0).days for p in serie]
        ys = [p["probabilidade"] for p in serie]
        direcao = "subindo" if ys[-1] > ys[0] else "caindo" if ys[-1] < ys[0] else "estável"
        texto += (f"; {direcao} de {ys[0]} para {ys[-1]} em {len(serie)} dias "
                  f"(cerca de {round(inclinacao(xs, ys), 3)} por dia)")
    achados = [texto + "."]

    feats = entrada.get("features", {})
    if feats:
        achados.append("Marcha (smartphone no bolso): " + "; ".join(
            f"{k.replace('_', ' ')} = {v}" for k, v in feats.items()) + ".")
    uso, uso_min = feats.get("horas_uso_dispositivo"), lim.get("horas_uso_minimas")
    if uso is not None and uso_min and uso < uso_min:
        achados.append(f"Cobertura baixa: smartphone em uso por {uso} h (mínimo configurado {uso_min} h); "
                       f"variáveis de marcha pouco confiáveis.")

    sit = ctx.get("contexto_situacional")
    for ev in entrada.get("eventos", []):
        s = descrever_situacao(ev, sit)
        achados.append(f"Evento '{ev.get('tipo')}' em {ev['timestamp'][:16].replace('T', ' ')}"
                       + (f" ({s})" if s else "") + ".")

    tabelas = []
    if serie:
        tabelas.append(tabela_md(["Data", "Probabilidade"], [[p["data"], p["probabilidade"]] for p in serie]))
    if entrada.get("eventos"):
        tabelas.append(tabela_md(
            ["Data/hora", "Tipo", "Pico (g)", "Vel. angular (°/s)", "Situação"],
            [[e["timestamp"][:16], e.get("tipo"), e.get("pico_aceleracao_g", "—"),
              e.get("velocidade_angular_max_dps", "—"), descrever_situacao(e, sit) or "—"]
             for e in entrada["eventos"]]))
    return {"achados": achados, "tabelas": tabelas,
            "dados": {"classe_risco": classe, "eventos": entrada.get("eventos", [])}}


def classificar_pa(sist, diast, faixas):
    for rotulo, smin, dmin in faixas:
        if sist >= smin or diast >= dmin:
            return rotulo
    return faixas[-1][0]


def proc_pressao_arterial(entrada, ctx, _resultados):
    lim = ctx.get("limiares", {})
    faixas = lim["classificacao"]["faixas"]
    regs = sorted(({**r, "_ts": ts(r["timestamp"]), "posicao": r.get("posicao", "sentado")} for r in entrada),
                  key=lambda r: r["_ts"])
    for r in regs:
        r["categoria"] = classificar_pa(r["sistolica"], r["diastolica"], faixas)

    base = [r for r in regs if r["posicao"] == "sentado"] or regs
    cats = Counter(r["categoria"] for r in base)
    texto = (f"PA sentado: média {round(mean(r['sistolica'] for r in base))}/"
             f"{round(mean(r['diastolica'] for r in base))} mmHg em {len(base)} medidas ("
             + ", ".join(f"{n} em {c}" for c, n in cats.items()) + ")")
    if len(base) >= 4:
        meio = len(base) // 2
        texto += (f"; sistólica média passou de {round(mean(r['sistolica'] for r in base[:meio]))} "
                  f"para {round(mean(r['sistolica'] for r in base[meio:]))} mmHg")
    achados = [texto + "."]

    pares = []
    for i, r in enumerate(regs):
        if r["posicao"] != "sentado":
            continue
        for q in regs[i + 1:]:
            if (q["_ts"] - r["_ts"]).total_seconds() / 60 > lim["janela_pareamento_min"]:
                break
            if q["posicao"] == "em_pe":
                ds, dd = r["sistolica"] - q["sistolica"], r["diastolica"] - q["diastolica"]
                ho = ds >= lim["ho_queda_sistolica"] or dd >= lim["ho_queda_diastolica"]
                pares.append({"data": r["timestamp"][:16], "sentado": f"{r['sistolica']}/{r['diastolica']}",
                              "em_pe": f"{q['sistolica']}/{q['diastolica']}", "queda": f"{ds}/{dd}",
                              "criterio_HO": ho, "_ts_em_pe": q["_ts"]})
                achados.append(f"{r['timestamp'][:10]}: sentado {r['sistolica']}/{r['diastolica']} -> em pé "
                               f"{q['sistolica']}/{q['diastolica']} (queda {ds}/{dd} mmHg), "
                               f"{'ATINGE' if ho else 'NÃO atinge'} critério de hipotensão ortostática.")
                break

    tabelas = [tabela_md(["Data/hora", "Posição", "PA (mmHg)", "FC", "Classificação", "Origem"],
                         [[r["timestamp"][:16], r["posicao"], f"{r['sistolica']}/{r['diastolica']}",
                           r.get("fc", "—"), r["categoria"], r.get("origem", "—")] for r in regs])
               + f"\n\n*Classificação: {lim['classificacao']['referencia']}.*"]
    if pares:
        tabelas.append(tabela_md(["Data/hora", "Sentado", "Em pé", "Queda", "Critério HO"],
                                 [[p["data"], p["sentado"], p["em_pe"], p["queda"],
                                   "sim" if p["criterio_HO"] else "não"] for p in pares]))
    return {"achados": achados, "tabelas": tabelas, "dados": {"pares": pares}}


def proc_ho_queda(_entrada, ctx, resultados):
    eventos = resultados["risco_queda"]["dados"]["eventos"]
    pares = resultados["pressao_arterial"]["dados"]["pares"]
    janela = ctx.get("limiares", {}).get("janela_evento_min", 15)
    sit = ctx.get("contexto_situacional")
    achados = []
    for ev in eventos:
        t_ev = ts(ev["timestamp"])
        for p in pares:
            dt = (t_ev - p["_ts_em_pe"]).total_seconds() / 60
            if 0 <= dt <= janela:
                s = descrever_situacao(ev, sit)
                extra = "; critério de hipotensão ortostática atingido" if p["criterio_HO"] else ""
                achados.append(f"Evento '{ev.get('tipo')}' em {ev['timestamp'][:16].replace('T', ' ')} "
                               f"ocorreu {round(dt)} min após medida em pé (queda de PA {p['queda']} mmHg{extra})"
                               + (f" [{s}]" if s else "") + ".")
    if pares:
        n_ho = sum(p["criterio_HO"] for p in pares)
        achados.append(f"{n_ho} de {len(pares)} pares sentado/em pé atingiram critério de hipotensão ortostática.")
    if not any("após medida em pé" in a for a in achados):
        achados.append(f"Nenhum evento ocorreu até {janela} min após medida em pé.")
    return {"achados": achados, "tabelas": [], "dados": {}}


PROCESSADORES = {
    "risco_queda": proc_risco_queda,
    "pressao_arterial": proc_pressao_arterial,
    "ho_queda": proc_ho_queda,
}


# =============================================================================
# 4. ORQUESTRAÇÃO DOS CONTEXTOS
# =============================================================================
def executar_contextos(snapshot, ids, pasta_contextos):
    configs = {cid: carregar_contexto(pasta_contextos, cid) for cid in ids}
    ordem = sorted(ids, key=lambda c: configs[c].get("tipo") == "composto")  # simples primeiro
    resultados = {}
    for cid in ordem:
        ctx = configs[cid]
        base = {"nome": ctx.get("nome", cid), "versao": ctx.get("versao")}
        proc = PROCESSADORES.get(ctx.get("processador"))
        if proc is None:
            resultados[cid] = {**base, "status": "não avaliado", "tabelas": [], "dados": {},
                               "achados": [f"Contexto não avaliado: processador '{ctx.get('processador')}' inexistente."]}
            continue
        if ctx.get("tipo") == "composto":
            faltam = [d for d in ctx.get("depende_de", []) if resultados.get(d, {}).get("status") != "avaliado"]
            if faltam:
                resultados[cid] = {**base, "status": "não avaliado", "tabelas": [], "dados": {},
                                   "achados": [f"Contexto não avaliado: dependências indisponíveis {faltam}."]}
                continue
            entrada = None
        else:
            entrada = obter_campo(snapshot, ctx["snapshot"]["campo"])
            if not entrada:
                resultados[cid] = {**base, "status": "não avaliado", "tabelas": [], "dados": {},
                                   "achados": [f"Contexto não avaliado: dado '{ctx['snapshot']['campo']}' ausente no snapshot."]}
                continue
        resultados[cid] = {**base, "status": "avaliado", **proc(entrada, ctx, resultados)}
    return {cid: resultados[cid] for cid in ids}, configs


def contextos_efetivos(perfil, snapshot):
    ativos = obter_campo(snapshot, "paciente.contextos_ativos")
    ids = [c for c in perfil["contextos"] if ativos is None or c in ativos]
    ignorados = [c for c in perfil["contextos"] if c not in ids]
    return ids, ignorados


# =============================================================================
# 5. LLM (Ollama /api/chat) E VALIDAÇÃO
# =============================================================================
def montar_mensagens(perfil, contexto_llm):
    pr = perfil["prompt"]
    valores = {
        "contexto": json.dumps(contexto_llm, ensure_ascii=False, separators=(",", ":")),
        "secoes": "\n".join(f"{i}. {s}" for i, s in enumerate(pr["secoes"], 1)),
        "max_palavras": pr["max_palavras"],
        "destinatario": perfil.get("destinatario", "o especialista"),
    }
    return [{"role": "system", "content": Template(pr["sistema"]).safe_substitute(valores)},
            {"role": "user", "content": Template(pr["usuario"]).safe_substitute(valores)}]


def chamar_llm(mensagens, cfg):
    payload = {"model": cfg["modelo"], "messages": mensagens, "stream": False, "options": cfg["opcoes"]}
    if cfg.get("think") is not None:
        payload["think"] = cfg["think"]
    req = urllib.request.Request(cfg["host"].rstrip("/") + "/api/chat",
                                 data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=cfg["timeout_s"]) as resp:
        r = json.loads(resp.read().decode("utf-8"))
    m = {
        "tokens_prompt": r.get("prompt_eval_count", 0),
        "tempo_prompt_s": round(r.get("prompt_eval_duration", 0) / 1e9, 1),
        "tokens_gerados": r.get("eval_count", 0),
        "tempo_geracao_s": round(r.get("eval_duration", 0) / 1e9, 1),
        "tempo_total_s": round(r.get("total_duration", 0) / 1e9, 1),
        "motivo_fim": r.get("done_reason"),
    }
    if m["tempo_prompt_s"]:
        m["taxa_prompt_tok_s"] = round(m["tokens_prompt"] / m["tempo_prompt_s"], 1)
    if m["tempo_geracao_s"]:
        m["taxa_geracao_tok_s"] = round(m["tokens_gerados"] / m["tempo_geracao_s"], 1)
    return r["message"]["content"].strip(), m


_NUM = re.compile(r"\d+(?:[.,]\d+)?(?:/\d+)?")


def fidelidade(texto, *fontes):
    """F(y) = 1 - |N(y) minus N(K u D_t)| / |N(y)| (Equation of numeric fidelity)."""
    citados = [x.replace(",", ".") for x in _NUM.findall(texto)]
    citados = [n for n in citados if not (n.isdigit() and int(n) <= 10)]
    if not citados:
        return 1.0, []
    suspeitos = validar_numeros(texto, *fontes)
    fora = sum(1 for n in citados if n in suspeitos)
    return round(1 - fora / len(citados), 4), suspeitos


def validar_numeros(texto, *fontes):
    permitidos = set()
    for f in fontes:
        permitidos |= {n.replace(",", ".") for n in _NUM.findall(json.dumps(f, ensure_ascii=False, default=str))}
    suspeitos = []
    for n in (x.replace(",", ".") for x in _NUM.findall(texto)):
        if n not in permitidos and not (n.isdigit() and int(n) <= 10) and n not in suspeitos:
            suspeitos.append(n)
    return suspeitos


# =============================================================================
# 6. RELATÓRIO
# =============================================================================
def montar_relatorio(snapshot, perfil, resultados, ignorados, texto, metricas, suspeitos):
    pid = obter_campo(snapshot, "paciente.id") or "sem id"
    partes = [f"# Relatório QVDE — paciente {pid} — {perfil.get('destinatario', '')}",
              f"Gerado em {datetime.now():%Y-%m-%d %H:%M}. "
              f"**Documento de apoio à decisão; não substitui avaliação clínica.**", "",
              "## 1. Achados por contexto de cuidado (determinísticos)", ""]
    for cid, r in resultados.items():
        partes += [f"### {r['nome']} (`{cid}` v{r['versao']}) — {r['status']}", ""]
        partes += [f"- {a}" for a in r["achados"]] + [""]
        for t in r["tabelas"]:
            partes += [t, ""]
    partes += ["## 2. Análise redigida pelo modelo de linguagem", "", texto, ""]
    if metricas.get("motivo_fim") == "length":
        partes += ["> ⚠️ Texto interrompido pelo limite de tokens (num_predict).", ""]
    if suspeitos:
        partes += [f"> ⚠️ Valores citados pelo modelo que NÃO constam nos dados: {', '.join(suspeitos)}. "
                   f"Conferir com a seção 1.", ""]
    cfg = perfil["llm"]
    partes += ["## 3. Metadados de execução", "",
               f"- Perfil: `{perfil['id']}` | Prompt: `{perfil['prompt']['versao']}` | Modelo: `{cfg['modelo']}`",
               "- Contextos: " + ", ".join(f"`{c}` v{r['versao']} ({r['status']})" for c, r in resultados.items())]
    if ignorados:
        partes.append(f"- Contextos do perfil não ativos para o paciente: {', '.join(ignorados)}")
    partes.append(f"- Opções: `{json.dumps(cfg['opcoes'])}`")
    if metricas:
        partes.append(f"- Métricas: `{json.dumps(metricas, ensure_ascii=False)}`")
    return "\n".join(partes) + "\n"


# =============================================================================
# 7. LINHA DE COMANDO
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="Relatório QVDE orientado a contextos de cuidado")
    ap.add_argument("entrada", help="snapshot JSON do gêmeo digital ('-' para entrada padrão)")
    ap.add_argument("--perfil", default=str(PERFIL_PADRAO))
    ap.add_argument("--contextos-dir", default=str(CONTEXTOS_DIR))
    ap.add_argument("--saida", help="arquivo Markdown de saída")
    ap.add_argument("--modelo", help="sobrescreve o modelo do perfil")
    ap.add_argument("--host", help="sobrescreve o host do Ollama")
    ap.add_argument("--sem-raciocinio", action="store_true", help="envia think=false (ex.: qwen3)")
    ap.add_argument("--somente-contexto", action="store_true", help="mostra o prompt e não chama o LLM")
    args = ap.parse_args()

    perfil = carregar_config(args.perfil)
    if os.environ.get("OLLAMA_HOST"):
        perfil["llm"]["host"] = os.environ["OLLAMA_HOST"]
    if args.modelo:
        perfil["llm"]["modelo"] = args.modelo
    if args.host:
        perfil["llm"]["host"] = args.host
    if args.sem_raciocinio:
        perfil["llm"]["think"] = False

    if args.entrada == "-":
        snapshot = json.load(sys.stdin)
    else:
        with open(args.entrada, encoding="utf-8") as f:
            snapshot = json.load(f)

    ids, ignorados = contextos_efetivos(perfil, snapshot)
    resultados, _ = executar_contextos(snapshot, ids, args.contextos_dir)

    pac = snapshot.get("paciente", {})
    contexto_llm = {
        "paciente": {k: pac.get(k) for k in ("idade", "sexo", "condicoes", "medicamentos")},
        "contextos": {r["nome"]: (r["achados"] if r["status"] == "avaliado" else ["não avaliado"] + r["achados"])
                      for r in resultados.values()},
    }
    if snapshot.get("dados_sinteticos"):
        contexto_llm["observacao"] = "Dados sintéticos para teste."
    mensagens = montar_mensagens(perfil, contexto_llm)

    if args.somente_contexto:
        for m in mensagens:
            print(f"----- {m['role']} -----\n{m['content']}\n")
        return

    cfg = perfil["llm"]
    estados = ", ".join(f"{c} ({r['status']})" for c, r in resultados.items())
    print(f"Contextos: {estados}", file=sys.stderr)
    print(f"Chamando {cfg['modelo']} em {cfg['host']} ...", file=sys.stderr)
    try:
        texto, metricas = chamar_llm(mensagens, cfg)
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        print(f"Aviso: falha ao chamar o LLM ({e}). Verifique 'systemctl status ollama'.", file=sys.stderr)
        texto, metricas = f"*Análise do modelo indisponível: {e}*", {}

    suspeitos = validar_numeros(texto, contexto_llm, snapshot, perfil) if metricas else []
    if metricas.get("motivo_fim") == "length":
        print("Aviso: resposta cortada pelo limite num_predict.", file=sys.stderr)
    if suspeitos:
        print(f"Aviso: valores não encontrados nos dados: {suspeitos}", file=sys.stderr)

    relatorio = montar_relatorio(snapshot, perfil, resultados, ignorados, texto, metricas, suspeitos)
    saida = args.saida or f"relatorio_{pac.get('id', 'paciente')}_{perfil['id']}_{datetime.now():%Y%m%d_%H%M}.md"
    with open(saida, "w", encoding="utf-8") as f:
        f.write(relatorio)
    print(f"Relatório salvo em {saida}", file=sys.stderr)
    if metricas:
        print(json.dumps(metricas, ensure_ascii=False, indent=2), file=sys.stderr)


if __name__ == "__main__":
    main()
