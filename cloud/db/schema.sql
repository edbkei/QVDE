-- =============================================================================
-- QVDE cloud digital twin (PostgreSQL 14+, TimescaleDB optional)
-- Only pseudonymous identifiers are stored: identifying data stays outside.
-- =============================================================================

-- Residence (house_id = MQTT username of its Hub) and the pseudonymous patient
CREATE TABLE IF NOT EXISTS pacientes (
    paciente_id      TEXT PRIMARY KEY,                 -- pseudonym, e.g. PAC-0007
    idade            INT,
    sexo             TEXT,
    condicoes        TEXT[] NOT NULL DEFAULT '{}',
    medicamentos     TEXT[] NOT NULL DEFAULT '{}',
    contextos_ativos TEXT[] NOT NULL DEFAULT '{}'      -- A(p)
);

CREATE TABLE IF NOT EXISTS residencias (
    house_id    TEXT PRIMARY KEY,
    paciente_id TEXT NOT NULL UNIQUE REFERENCES pacientes(paciente_id),
    criado      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Derived records sent by the Hub after each inference (reactive loop)
CREATE TABLE IF NOT EXISTS derivados (
    ts                 TIMESTAMPTZ NOT NULL,
    msg_id             UUID NOT NULL,
    house_id           TEXT NOT NULL REFERENCES residencias(house_id),
    sensor_id          TEXT,
    contexto           TEXT NOT NULL,
    versao_contexto    INT,
    avaliado           BOOLEAN NOT NULL,
    cobertura          REAL,
    p                  REAL,
    inclinacao         REAL,
    u                  REAL,
    classe             TEXT,
    tendencia          TEXT,
    incerteza          TEXT,
    periodo            TEXT,
    acao               TEXT,
    acoes_obrigatorias TEXT[] NOT NULL DEFAULT '{}',
    modelo_versao      TEXT,
    rl_versao          TEXT,
    intervalo_s        REAL,
    PRIMARY KEY (msg_id, ts)
);
CREATE INDEX IF NOT EXISTS derivados_house_ts ON derivados (house_id, ts DESC);

CREATE TABLE IF NOT EXISTS eventos (
    msg_id           UUID PRIMARY KEY,
    ts               TIMESTAMPTZ NOT NULL,
    house_id         TEXT NOT NULL REFERENCES residencias(house_id),
    sensor_id        TEXT,
    contexto         TEXT NOT NULL,
    tipo_evento      TEXT NOT NULL,                    -- queda | quase_queda
    p                REAL,
    periodo          TEXT,
    local            TEXT,
    atividade_previa TEXT,
    modelo_versao    TEXT
);
CREATE INDEX IF NOT EXISTS eventos_house_ts ON eventos (house_id, ts DESC);

-- Labelled raw windows around events, used for retraining and audit
CREATE TABLE IF NOT EXISTS episodios (
    msg_id       UUID PRIMARY KEY,
    evento_id    UUID,
    house_id     TEXT NOT NULL REFERENCES residencias(house_id),
    sensor_id    TEXT,
    contexto     TEXT NOT NULL,
    rotulo       TEXT,
    rotulado_por TEXT,
    t_evento     TIMESTAMPTZ,
    campos       TEXT[] NOT NULL,
    amostras     JSONB NOT NULL,
    recebido     TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Clinical record: blood pressure (FHIR Observation LOINC 85354-9, simplified)
CREATE TABLE IF NOT EXISTS observacoes_pa (
    id          BIGSERIAL PRIMARY KEY,
    paciente_id TEXT NOT NULL REFERENCES pacientes(paciente_id),
    ts          TIMESTAMPTZ NOT NULL,
    sistolica   INT NOT NULL,
    diastolica  INT NOT NULL,
    fc          INT,
    posicao     TEXT NOT NULL DEFAULT 'sentado',       -- sentado | em_pe | deitado
    origem      TEXT,
    UNIQUE (paciente_id, ts, posicao)
);

-- Output of the (future) daily fall-risk model of the deliberative loop
CREATE TABLE IF NOT EXISTS predicoes_risco (
    paciente_id   TEXT NOT NULL REFERENCES pacientes(paciente_id),
    data          DATE NOT NULL,
    modelo        TEXT NOT NULL,
    janela_h      INT,
    horizonte_h   INT,
    probabilidade REAL NOT NULL,
    incerteza     REAL,
    PRIMARY KEY (paciente_id, data, modelo)
);

-- Versioned repositories governed by the Orchestrator
CREATE TABLE IF NOT EXISTS contextos_repo (
    id         TEXT NOT NULL,
    versao     INT NOT NULL,
    yaml       TEXT NOT NULL,
    publicado  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (id, versao)
);

CREATE TABLE IF NOT EXISTS modelos_repo (
    nome      TEXT NOT NULL,                           -- lstm | rl
    versao    TEXT NOT NULL,
    url       TEXT NOT NULL,
    sha256    TEXT NOT NULL,
    metricas  JSONB NOT NULL DEFAULT '{}',
    publicado TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (nome, versao)
);

-- Desired state (published retained to the Hub) and state reported by the Hub
CREATE TABLE IF NOT EXISTS estado_desejado (
    house_id      TEXT PRIMARY KEY REFERENCES residencias(house_id),
    versao_estado INT NOT NULL,
    contextos     JSONB NOT NULL,                      -- {id: versao}
    modelos       JSONB NOT NULL DEFAULT '{}',         -- {nome: versao}
    atualizado    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS estado_reportado (
    house_id   TEXT PRIMARY KEY REFERENCES residencias(house_id),
    estado     JSONB NOT NULL,
    atualizado TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Reports to the specialist (deliberative loop), with numeric fidelity F(y)
CREATE TABLE IF NOT EXISTS relatorios (
    id            BIGSERIAL PRIMARY KEY,
    paciente_id   TEXT NOT NULL REFERENCES pacientes(paciente_id),
    perfil        TEXT NOT NULL,
    gerado        TIMESTAMPTZ NOT NULL DEFAULT now(),
    modelo_llm    TEXT,
    prompt_versao TEXT,
    snapshot      JSONB NOT NULL,
    texto         TEXT NOT NULL,
    fidelidade    REAL,
    suspeitos     TEXT[] NOT NULL DEFAULT '{}',
    metricas      JSONB NOT NULL DEFAULT '{}'
);

-- TimescaleDB, when available: hypertable + retention for derived records
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_available_extensions WHERE name = 'timescaledb') THEN
        CREATE EXTENSION IF NOT EXISTS timescaledb;
        PERFORM create_hypertable('derivados', 'ts', if_not_exists => TRUE, migrate_data => TRUE);
        PERFORM add_retention_policy('derivados', INTERVAL '400 days', if_not_exists => TRUE);
    END IF;
END
$$;
