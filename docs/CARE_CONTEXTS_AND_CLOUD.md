# Care contexts, reactive Hub and cloud digital twin

This document describes the components added for the architecture of the
thesis (Figure 4.2: reactive loop on the Hub, digital twin and deliberative
loop in the cloud). They run **next to** the existing services: nothing in
`src/`, `desktop/` or `docker-compose.yml` was changed.

## Mapping to the thesis

| Thesis | Code |
|---|---|
| Care context C = ⟨q, S_req, S_opc, V, M, Θ, Σ_C, α⟩ (Def. 1, Table 4.2) | `contexts/*.yaml` |
| Coverage c_s(W), availability disp(C,t) (Def. 2) | `qvde_common/contexts.py` · `coverage`, `is_available` |
| Evaluation operator, explicit degradation (Def. 3, Prop. 1) | `hub/qvde_hub/service.py` · `_evaluate` (status `nao_avaliado`) |
| Dependency DAG, topological order (Prop. 2) | `qvde_common/contexts.py` · `topological_order` |
| Composite context ρ_Δ, HO × near-falls (Def. 4, Ex. 1) | `cloud/qvde_cloud/relatorio.py` · `proc_ho_queda` |
| RL state s_t = (κ, d, u, status, σ_per) (Sec. 4.6.5, 144 states) | `hub/qvde_hub/state.py` |
| Actions a1..a5 masked by α, situational overrides | `hub/qvde_hub/rl.py`, `contexts.admissible_actions` |
| Raw buffer, configurable retention, cleanup after confirmation (Sec. 4.2.2) | `hub/qvde_hub/buffer.py` |
| Local shadow of contexts and models (Sec. 4.2.2) | `hub/qvde_hub/shadow.py` |
| Sync with the cloud, offline queue, Hub-initiated TLS | `hub/qvde_hub/uplink.py`, `qvde_common/protocol.py` |
| Digital twin storage (PostgreSQL + TimescaleDB) | `cloud/db/schema.sql`, `cloud/qvde_cloud/ingest.py` |
| Governance: versioned contexts/models, desired state (Orchestrator) | `cloud/qvde_cloud/twin_admin.py` |
| Separation of calculation and write-up (Prop. 3), fidelity F(y) | `cloud/qvde_cloud/report_job.py`, `relatorio.fidelidade` |

## Data flow

```
smartphone/simulator ──MQTT──▶ Hub (Jetson)
   iot/{house}/fall_detection/...      │ buffer (raw, retention N days)
                                       │ context evaluator → LSTM → s_t → RL (α)
                                       │ actuation: qvde/local/{house}/acao, ai/fall_prediction
                                       ▼ outbox (derived records, events, episodes)
                       MQTT over TLS, Hub-initiated ──▶ cloud broker (8883, per-house ACL)
                                                            │ ingest → PostgreSQL/TimescaleDB
                       ◀── qvde/v1/{house}/down/ack ────────┘ (Hub may now delete expired raw data)
                       ◀── qvde/v1/{house}/down/desejado (retained: contexts + model versions)
                                                     report_job: snapshot → findings → LLM → F(y)
```

Only derived data leaves the residence. Raw IMU samples go up only inside
labelled episodes (±`janela_evento_s` around an event), used for retraining.

## Hub (Jetson)

```bash
docker compose -f docker-compose.yml -f docker-compose.hub.yml up -d qvde-hub
docker logs -f qvde-hub
```

Configuration (environment, prefix `QVDE_`): `HOUSE_ID`, `LOCAL_BROKER`,
`INTERVAL_S`, `CLOUD_BROKER`, `CLOUD_PORT`, `CLOUD_CA`, `CLOUD_USER`,
`CLOUD_PASSWORD`, `LSTM_PATH`, `RL_PATH`. Without `QVDE_CLOUD_BROKER` the Hub
runs offline and keeps everything queued.

Detection model: the newest `models/lstm_v*.pkl`; if none is present, an
SMV-peak heuristic (`heuristic-smv-v1`) keeps the pipeline running. The model
version travels in every derived record, so the substitution is visible.

## Cloud (e.g. the Locaweb VM)

```bash
cp .env.cloud.example .env.cloud                     # set the passwords
bash scripts/gerar_certificados.sh <broker-host> house_001
docker compose --env-file .env.cloud -f docker-compose.cloud.yml up -d

# governance (one-shot jobs)
alias twin='docker compose --env-file .env.cloud -f docker-compose.cloud.yml run --rm twin-admin'
twin registrar-residencia --house house_001 --paciente PAC-0007 --idade 81 --sexo F \
     --contextos risco_queda,pressao_arterial,hipotensao_ortostatica_queda
twin publicar-contextos --dir contexts
twin definir-desejado --house house_001
twin status --house house_001

# report to the specialist
docker compose --env-file .env.cloud -f docker-compose.cloud.yml run --rm reports --paciente PAC-0007
```

Changing a context: edit the YAML, **increase `versao`**, then
`publicar-contextos` and `definir-desejado`. Publishing changed content under
an existing version is refused, so every decision can be traced to one
context version.

## Tests

```bash
pip install -r requirements-dev.txt
pytest                                    # unit tests (no services needed)
QVDE_TEST_DSN=postgresql://user@host/db pytest     # + PostgreSQL tests
# the end-to-end test also needs a `mosquitto` binary on PATH
```

## Known limitations

* **Fall-risk model (72 h)**: not implemented yet. Until `predicoes_risco` is
  filled, the report uses a labelled proxy (daily P95 of the Hub's detection
  probability) and says that it has no prediction horizon.
* **Uncertainty u**: the custom LSTM has no probabilistic output; u is the
  dispersion of the last five predictions (a proxy, documented in `state.py`).
* **Pending episodes** are kept in memory: an event in the last
  `janela_evento_s` before a restart produces the event record but no episode.
* **Model files**: the LSTM is a pickle; the Hub accepts a downloaded model only
  if its SHA-256 matches the desired state received over the authenticated TLS
  channel. Keep the cloud broker credentials and the model host protected.
* **RL agent**: starts untrained and follows the specialist's preference order
  in `acoes`; it must be trained from labelled episodes before claims about
  learned behaviour.
