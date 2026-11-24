## Extended Home Quality of Life (QVDE - Qualidade de Vida Domiciliar Estendida)

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/docker-required-blue.svg)](https://www.docker.com/)
[![NVIDIA Jetson](https://img.shields.io/badge/NVIDIA-Jetson%20Orin-76B900.svg)](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/)

A local-first, multi-agent AI system for real-time fall detection and health monitoring in smart homes, running on edge computing infrastructure.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Technology Stack](#technology-stack)
- [Research Context](#research-context)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Monitoring & Verification](#monitoring--verification)
- [Performance](#performance)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

---

## 🎯 Overview

This system monitors accelerometer and gyroscope data from wearable devices used by elderly residents in their homes to detect falls or fall tendencies in real-time. Upon detection, it triggers appropriate actions such as notifying family members, caregivers, or emergency services.

### Why This Matters

**Local-First Architecture**: Continuous health monitoring requires uninterrupted operation, even during internet outages. This system runs entirely on local edge computing infrastructure, ensuring reliability and privacy.

**Multi-Agent AI Paradigm**: The system implements an adaptive agentic architecture using digital twins for representation and learning, addressing the lack of systemic integration between isolated robotic devices, IoT sensors, and home automation solutions in healthcare settings.

**Generalizable Framework**: While demonstrated with fall detection, the architecture is designed to support various health monitoring applications:
- Blood pressure tracking
- Blood glucose monitoring
- Sleep disorder detection
- Body temperature monitoring
- Heart rate variability analysis
- Medication adherence tracking
etc.

By focusing on a specific use case (fall detection), we establish a proven framework that can be theoretically reused for a wide variety of medical applications.

---

## ✨ Key Features

### 🏠 Multi-House Support
- Monitor multiple smart homes simultaneously
- Unique identification per house, room, and sensor
- Scalable architecture for residential care facilities

### 🧠 Intelligent Fall Detection
- **Custom LSTM Neural Network**: 5x faster than standard Keras implementation
- **Reinforcement Learning Agent**: Adaptive decision-making for alert prioritization
- **Real-time Inference**: <2ms prediction latency on NVIDIA Jetson Orin Nano
- **87.88% Validation Accuracy** (v1.1.0 model)

### 🔐 Privacy & Reliability
- **Edge-first Processing**: Fall detection, decision and actuation run on-premise
- **Edge Computing**: All emergency AI inference runs on the Hub
- **Offline Operation**: Continues functioning during internet outages
- **Data Minimization**: Raw sensor data stays on the Hub for a configurable period; only derived data and labelled event episodes go to the optional cloud digital twin

> **New (care contexts + cloud digital twin):** see [docs/CARE_CONTEXTS_AND_CLOUD.md](docs/CARE_CONTEXTS_AND_CLOUD.md).

### 📊 Production-Ready Infrastructure
- **Docker-based Deployment**: Easy setup and updates
- **Time-Series Database**: InfluxDB for efficient sensor data storage
- **Real-time Messaging**: MQTT broker for low-latency communication
- **Visualization Dashboard**: Grafana for monitoring and analytics
- **Model Versioning**: Semantic versioning with metadata tracking

### 🔄 Operational Modes
- **Training Mode**: Collect labeled data for model improvement
- **Inference Mode**: Real-time fall detection and alerting
- **Validation Mode**: Model performance evaluation

---

## 🏗️ System Architecture

### High-Level Architecture

```
┌─ Residência (rede local) ───────────────────────────────────────────────────────────┐
│                                                                                     │
│  ┌───────────────┐  ┌───────────────────┐   ┌──────────────────┐  ┌──────────────┐  │
│  │   Smartphone  │  │    Simulador de   │   │  Interface Web   │  │  Outros IoT  │  │
│  │    no bolso   │  │ sensores (Jupyter)│   │de Operação (8501)│  │  (ex.: PA)   │  │
│  │IMU acel./giro.│  │    -- Laptop --   │   │   -- Laptop --   │  │              │  │
│  └───────┬───────┘  └─────────┬─────────┘   └────────┬─────────┘  └──────┬───────┘  │
│          │                    │                      │                   │          │
│          └────────────────────┴───────────┬──────────┴───────────────────┘          │
│                                           │ MQTT 1883 (somente na rede local)       │
├─ Jetson Orin Nano — Hub (laço reativo) ───┼─────────────────────────────────────────┤
│                                           │ ms–s; continua operando sem a nuvem     │
│                       ┌───────────────────▼───────────────────┐                     │
│                       │  Mosquitto local (1883)  -- Docker -- │                     │
│                       └───────────────────┬───────────────────┘                     │
│                                           │                                         │
│  ┌────────────────────────────────────────▼──────────────────────────────────────┐  │
│  │  qvde-hub  -- Docker --                │                                      │  │
│  │  ┌──────────────────────┐   ┌──────────▼──────────┐   ┌─────────────────────┐ │  │
│  │  │ Sombra local:        │   │ Adaptadores e       ├──►┤ Buffer bruto SQLite │ │  │
│  │  │ contextos e modelos v│   │ payload X(t) ∈ R⁶   │   │ retenção de 7 dias; │ │  │
│  │  │ (estado desejado)    │   └──────────┬──────────┘   │ limpa após ACK      │ │  │
│  │  └──────────┬───────────┘              │              └─────────────────────┘ │  │
│  │             │                          │                                      │  │
│  │  ┌──────────▼──────────────────────────▼───────────────────────────────────┐  │  │
│  │  │ Avaliador de contextos (cobertura; σ(t) no fuso local) ► LSTM_v ► RL_v  │  │  │
│  │  └────────────────┬─────────────────────────────────────────┬──────────────┘  │  │
│  │                   │                                         │                 │  │
│  │  ┌────────────────▼────────────────┐         ┌──────────────▼──────────────┐  │  │
│  │  │ Atuação: alerta local, cuidador,│         │ Outbox (fila até o ACK):    │  │  │
│  │  │ robô (ROS) · qvde/local/+/acao  │         │ derivados, eventos,         │  │  │
│  │  └─────────────────────────────────┘         │ episódios rotulados, estado │  │  │
│  │                                              └──────────────┬──────────────┘  │  │
│  └─────────────────────────────────────────────────────────────┼─────────────────┘  │
│                                                                │                    │
├─ MQTT sobre TLS 8883, conexão iniciada pelo Hub ───────────────┼────────────────────┤
│   ▲ ack · estado desejado (retido) → Sombra local              ▼ só dados derivados │
├─ Nuvem — VM Locaweb, 2 vCPU / 4 GB (laço deliberativo) ────────┼────────────────────┤
│                                                                │                    │
│  ┌──────────────────────────┐                   ┌──────────────▼──────────────┐     │
│  │ Twin-admin (Orquestrador)├──────────────────►┤     Mosquitto TLS (8883)    │     │
│  │ contextos e modelos      │                   │         ACL por casa        │     │
│  │ versionados → gêmeo;     │                   └──────────────┬──────────────┘     │
│  │ estado desejado → MQTT   │                                  │                    │
│  └──────────────────────────┘                   ┌──────────────▼──────────────┐     │
│                                                 │  Ingest (idempotente, ack)  │     │
│   Grafana 127.0.0.1:3000 (painéis do gêmeo)     └──────────────┬──────────────┘     │
│                                                                │                    │
│                                                 ┌──────────────▼──────────────┐     │
│  ┌────────────────────────┐    ┌────────────┐   │ TimescaleDB (PostgreSQL 16) │     │
│  │ Ollama nativo (11434)  ├◄───┤ Relatórios ├◄──┤ gêmeo digital: derivados,   │     │
│  │ qwen2.5:1.5b -- CPU -- │    │(report_job)│   │ eventos, episódios, PA,     │     │
│  └────────────────────────┘    └────────────┘   │ relatórios e F(y)           │     │
│                                                 └─────────────────────┬───────┘     │
│                                                                       │             │
├─ Laptop — i7-13650HX · 40 GB · RTX 4050 6 GB (reforço deliberativo) ──┼─────────────┤
│                                                túnel SSH 15432 → 5432 │             │
│  ┌────────────────────────┐                       ┌───────────────────┴───────┐     │
│  │ Ollama (11434)         ├◄──────────────────────┤ Relatórios (report_job)   │     │
│  │ qwen2.5:7b -- GPU --   │                       │ perfil geriatra_7b        │     │
│  └────────────────────────┘                       └───────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────────────────────┘

```

### Data Flow

```
Laço reativo (Hub — Jetson, opera sem a nuvem)
Smartphone IMU → MQTT local → Adaptador → Buffer SQLite → Avaliador de contexto → LSTM_v → RL_v → Ação
   (10 Hz)        (1883)      (X(t)∈R⁶)   (bruto, 7 d)    (cobertura, σ(t))     (10 amostras) (5 ações) (alerta/robô)
                                                          └────────── ciclo a cada 3 s ──────────┘

Laço deliberativo (nuvem)
Outbox do Hub → MQTT/TLS 8883 → Ingest → TimescaleDB (gêmeo) → Relatório: achados → LLM → F(y)
(derivados,      (Hub inicia;     (ack)    (derivados, eventos,   (determinísticos)  (7B laptop: ~12–17 s
 eventos)         fila offline)             PA, relatórios)                           1,5B VM: ~60 s)
```

### Component Interaction

```
┌─────────────────────────────────────────────────────────────────┐
│          Sensor Simulator / smartphone (Windows/Laptop)         │
└────────────────────────────────┬────────────────────────────────┘
                                 │ MQTT 1883 (JSON, 10 Hz)
                                 ↓
┌─────────────────────────────────────────────────────────────────┐
│            Mosquitto local (Jetson, só rede da casa)            │
│  Entrada: iot/{house_id}/fall_detection/{tipo}/{sensor_id}      │
│  Saída:   qvde/local/{house_id}/acao · ai/fall_prediction       │
└───────────────┬─────────────────────────────────▲───────────────┘
                │ leituras                        │ ações
                ↓                                 │
┌─────────────────────────────────────────────────┴───────────────┐
│                     qvde-hub (Jetson, Docker)                   │
│  - Adaptador: valida casa/sensor, ignora dados de treino        │
│  - Buffer SQLite: bruto por 7 d; limpa após ACK da nuvem        │
│  - Avaliador: cobertura ≥ 0,6, σ(t) no fuso da casa             │
│  - LSTM_v (janela de 10) → RL_v (ação admissível + override)    │
│  - Sombra local: contextos e modelos ativos (versão v)          │
│  - Outbox: derivados, eventos, episódios ±30 s, estado          │
└───────────────┬─────────────────────────────────▲───────────────┘
                │ up/{derivado|evento|            │ down/ack
                │    episodio|estado}             │ down/desejado (retido)
                ↓                                 │   MQTT/TLS 8883 (o Hub inicia)
┌─────────────────────────────────────────────────┴───────────────┐
│        Mosquitto nuvem (VM Locaweb) · TLS · ACL por casa        │
│  Tópicos: qvde/v1/{house_id}/up/... e qvde/v1/{house_id}/down/  │
└───────────────┬─────────────────────────────────▲───────────────┘
                ↓                                 │ publica desejado
┌──────────────────────────────┐  ┌───────────────┴───────────────┐
│  Ingest                      │  │  Twin-admin (Orquestrador)    │
│  - Valida o envelope         │  │  - Registra casa e paciente   │
│  - Grava sem duplicar        │  │  - Publica contextos/modelos  │
│    (msg_id)                  │  │    versionados                │
│  - Devolve ACK ao Hub        │  │  - Importa PA (CSV)           │
└──────────────┬───────────────┘  └───────────────┬───────────────┘
               ↓                                  ↓
┌─────────────────────────────────────────────────────────────────┐
│               TimescaleDB / PostgreSQL 16 (gêmeo)               │
│  Tabelas:                                                       │
│  - derivados (hypertable), eventos, episodios                   │
│  - observacoes_pa, pacientes, residencias (fuso_horario)        │
│  - contextos_repo, modelos_repo, estado_desejado/reportado      │
│  - relatorios (texto, F(y), métricas)                           │
└────────────────────────────────┬────────────────────────▲───────┘
                                 ↓ snapshot               │ grava
┌────────────────────────────────┴────────────────────────┴───────┐
│                    Relatórios (report_job)                      │
│  - Achados determinísticos por contexto de cuidado              │
│  - LLM interpreta: qwen2.5:1.5b (VM) ou 7b (laptop, túnel SSH)  │
│  - Calcula F(y) e marca números suspeitos                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Technology Stack

QVDE runs on three tiers: the **Hub** in the home (reactive loop), a **cloud VM** with the digital twin (deliberative loop) and an optional **GPU workstation** for larger language models.

- **NVIDIA Jetson Orin Nano (8 GB)**: AI-accelerated edge device
  - 6-core Arm Cortex-A78AE CPU
  - 1024-core NVIDIA Ampere GPU with 32 Tensor Cores
  - 8 GB LPDDR5 RAM
  - Ubuntu 24.04 LTS (JetPack 7.2 / Jetson Linux 39.2); Ubuntu 22.04 with JetPack 6 also works
  - NVMe SSD for the operating system, containers and data

### Cloud Platform (Digital Twin)
- **Linux VM (Locaweb)**: 2 vCPU, 4 GB RAM, no GPU, Ubuntu, Docker
- Only derived data leaves the home; raw sensor data stays on the Hub

### Core Infrastructure
- **Docker & Docker Compose v2**: Containerized deployment on the Hub and in the cloud
- **Mosquitto (2.0.x)**: MQTT broker
  - Hub: local broker on port 1883, home network only
  - Cloud: TLS on port 8883, one login and ACL per house (username = `house_id`)
- **SQLite**: Hub raw-data buffer (7-day retention, deleted only after the cloud acknowledges the derived data) and upload queue (outbox)
- **PostgreSQL 16 + TimescaleDB 2.17**: Digital twin in the cloud (derived records, events, labelled episodes, blood pressure, versioned care contexts and models, desired/reported state, reports)
- **Grafana (11.x)**: Twin dashboards in the cloud, bound to `127.0.0.1:3000` (reached through an SSH tunnel) -- being removed
- **Streamlit (1.x)**: Operation web interface to control the training/inference phase of the ML/AI models

### AI/ML Stack
- **Python 3.10+**: Core programming language
- **NumPy**: Numerical computing
- **Custom LSTM implementation (NumPy, CPU)**: Fall detection on the Hub
  - Input: 10-sample window of 6-axis IMU data (accelerometer + gyroscope)
  - Lightweight alternative to Keras/TensorFlow for edge deployment
- **Tabular Q-learning agent**: Chooses one of 5 actions (`alerta_local`, `notificar_remoto`, `acionar_robo`, `solicitar_dados`, `aguardar`) from a discrete state, restricted to the actions each care context allows
- **Care contexts (YAML)**: Versioned, configurable definitions of sources, thresholds, situational context and admissible actions (`contexts/`)
- **Ollama + Qwen2.5**: Local language models for the specialist report (no patient data sent to third-party APIs)
  - `qwen2.5:1.5b` on the cloud VM (CPU, ~9 tokens/s)
  - `qwen2.5:7b` on a GPU  (RTX 4050 6 GB, ~28 tokens/s)
- **Numeric fidelity metric F(y)**: Flags numbers in the LLM text that are not in the computed findings

### Communication Protocols
- **MQTT (QoS 1)**: Sensors → Hub on the home network
- **MQTT over TLS (8883)**: Hub ↔ cloud; the connection is always opened by the Hub, so the home exposes no inbound ports
  - Up: `qvde/v1/{house_id}/up/{derivado|evento|episodio|estado}`
  - Down: `qvde/v1/{house_id}/down/ack` and `down/desejado` (retained desired state)
- **HTTP/REST**: Report job → Ollama API
- **SSH tunnel**: Workstation → cloud database for remote report generation

### Development Tools
- **Jupyter Notebook**: Sensor simulation and experimentation
- **Streamlit**: Operation web interface
- **pytest**: Unit, PostgreSQL and end-to-end MQTT tests
- **Git / GitHub**: Version control
- **Docker**: Containerization

---

## 🔬 Research Context

This project is part of doctoral research at UNICAMP on **Extended Home Quality of Life (QVDE - Qualidade de Vida Domiciliar Estendida)**, focusing on adaptive agentic system paradigms for home healthcare technology integration.

### Theoretical Framework

The research proposes a hybrid reactive-deliberative architecture (Murphy, 2000): a **reactive loop** on the Hub (milliseconds to seconds, works offline) and a **deliberative loop** in the cloud (digital twin, long-term analysis, language-model reports). The agent properties below follow the weak notion of agency of Wooldridge & Jennings (1995) and the learning agent of Russell & Norvig:

- ✅ **Autonomy**: The Hub decides and acts without human intervention or cloud connectivity
- ✅ **Reactivity**: Each care context is evaluated every few seconds; alerts are issued locally
- 🔶 **Proactivity**: Requests more data when coverage is low and escalates by time of day (e.g., mandatory remote notification at night); fall-risk forecasting is planned
- 🔶 **Sociability**: Communicates with caregivers, the robotic agent (ROS) and the cloud twin; multi-agent coordination is future work
- ✅ **Adaptability**: RL agent learns from rewards; care contexts and models are updated from the cloud
- ✅ **Goal-Orientation**: Fall detection and response, with orthostatic hypotension as a composite context

### Digital Twin Paradigm

The digital twin lives in the cloud and is used for:
- **Representation**: Pseudonymous patient, house, derived sensor data, events and blood-pressure records
- **Governance**: Versioned care contexts and models, desired state pushed to each Hub
- **Learning**: Labelled episodes (±30 s around events) for retraining
- **Prediction**: Daily fall-risk indicator (currently a proxy from the Hub's detections; a dedicated risk model is planned)
- **Explanation**: Specialist reports with deterministic findings plus language-model interpretation

### Key Research Questions Addressed

1. How can isolated IoT devices be integrated into a cohesive healthcare system?
2. How should processing be split between the home and the cloud for health monitoring?
3. How can reinforcement learning, constrained by care contexts, improve alert accuracy and reduce false positives?
4. What architectural patterns enable scalable multi-house monitoring?
5. Can small, locally hosted language models write reliable specialist reports from computed findings?

---

## 🚀 Getting Started

### Prerequisites

**Hardware Requirements:**
- NVIDIA Jetson Orin Nano (or compatible edge device)
- NVMe SSD (recommended; QVDE uses a 256 GB NVMe for everything) or a 64 GB+ microSD card. NVMe is considerably faster than microSD for containers and databases.
- Network connectivity (Ethernet or Wi-Fi)
- A smartphone or wearable with accelerometer and gyroscope (or the sensor simulator)
- Cloud VM for the digital twin: 2 vCPU, 4 GB RAM, ~40 GB disk, public IP for MQTT/TLS
- (Optional) Workstation with an NVIDIA GPU (6 GB+ VRAM) for the 7B language model

**Software Requirements:**
- Jetson: Ubuntu 24.04 (JetPack 7.2) or Ubuntu 22.04 (JetPack 6), ARM64
- Docker 24.0+ and Docker Compose v2
- Python 3.10+
- Streamlit 1.x
- Ollama (cloud VM and/or GPU workstation)
- OpenSSL (TLS certificates via `scripts/gerar_certificados.sh`)


### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/iot-fall-detection.git
   cd iot-fall-detection
   ```

2. **Configure Environment**
   ```bash
   # Copy example configuration
   cp .env.example .env
   
   # Edit configuration with your settings
   nano .env
   ```

3. **Start the System**
   ```bash
   # Navigate to deployment directory
   cd /mnt/nvme/iot-stack
   
   # Start all services
   sudo docker-compose up -d
   
   # Verify all containers are running
   sudo docker-compose ps
   ```

4. **Verify Installation**
   ```bash
   # Check MQTT broker
   sudo docker logs mosquitto
   
   # Check InfluxDB
   sudo docker logs influxdb
   
   # Check LSTM service
   sudo docker logs custom-lstm-detector

   # Check bridge logs
   docker logs mqtt-influx-bridge

   # restart MQTT if needed
   docker-compose restart mosquitto
   ```

5. **Install Streamlit**
   ```bash
   # Install dependencies
   pip install -r requirements-operator.txt
   # if needed, pip install streamlit paho-mqtt pandas plotly

   # if new deployment
   docker-compose up -d
   
   # create venv virtual environment: https://docs.python.org/3/library/venv.html
   python -m venv c:\ ... project\fall-detection-operator

   cd ... \project\fall-detection-operator
   
   # Activate venv virtual environment
   venv\Script\activate
   
   # Deactivate venv virtual environment
   venv\Script\deactivate

   # Open Operation Web Interface
   streamlit run operator_interface.py

      You can now view your Streamlit app in your browser.

      Local URL: http://localhost:8501
      Network URL: http://xxx.xxx.xxx.xxx:8501 
   
   # Check URL://xxx.xxx.xxx.xxx:8501

   # In new terminal, start Jupyter
   jupyter notebook or Anaconda/Jupyter/ choose sensor_simulator.ipynb
   
   ```
6. **Everyday update of any script in github.com/edbkei/QVDE from Jetson**
 ```bash
      # In VScode, for instance, update the file scripts/enhanced_integrated_lstm_service.py

      # In Jetson:
      cd /mnt/nvme/iot-stack

      # 1. See what changed
      git status
      git diff

      # 2. stage and commit
      git add scripts/enhanced_integrated_lstm_service.py
      git commit -m "Fix inference field names: accel_x/y/z -> x/y/z
      comments comments ..."

      # 3. publish
      git push origin main

      # Note:
      # git diff before staging is hte habit worth building - it shows exactly what you´re about to record,
      # and catches the stray debug line you forgot to remove

      # Prefer git add <file> over git add -A
      # -A is what pulled .env in earlier. Name the files, or use git add -p to step through changes hunk by hunk and approve each one. 
      # When wanting everything, run git status --short, first and read it.

      # 4. Deploying a change
      # The script lives in the image, not a bind mount, so editing a file doesn´t affect the running container:

      sudo docker compose build enhanced_integrated_lstm_service
      sudo docker compose up -d enhanced _integrated_lstm_service 
      sudo docker compoe logs --tail 30 enhanced_integrated_lstm_service

      # 5. Tag anything it is cited
      # To produce a number that goes in the thesis:

      git tag -a v1.7.0-fieldfix -m "Post field-name fix: F1=0.xx, recall=0.xx"
      git push origin --tags
      # note: the tag can be referenced in thesis, and the exact code is recoverable.

      # 6. Useful when things go sideways

      git diff HEAD~1              # what did my last commit change?
      git checkout -- <file>       # discard uncommited edits to on file
      git log --oneline -10        # recent history
      git log -p -- scripts/foo.py # full history of one file

      # 7. The pre-commit reflex
      # Before every git push, do:

      git grep --cached -n "mZYXd\|password\|token"

      # Note: Takes a second a would have caught the .env problem. Worth doing  util it´s automatic

      # 8. Frequent update in any file

      cd /mnt/nvme/iot-stack

      git pull origin main        # start of session
      # ... edit files ...        using VSCODE, for instance
      git status                  # what changed
      git diff                    # how it was changed
      git add <files>             # stage deliberately, not -A
      git commit -m "..."         # one logical change per commit
      git push origin main        # publish

      # In case push fails due to example Github has A-B-C and laptop/jetson has A-B-C, do first, before git push
      # git add also need first (check)
      git pull --rebase origin main

      # example
      git diff README.md  # review what changed
      git add README.md
      git commit -m "Update README with <what was changed>"
      git push origin main

      # Note: If git push is rejected with "fetch first" or "behind", the remote moved since last pull:

      git pull origin main
      git push origin main

      # Note: With pull.rebase true set, that replays commit on top of the remote´s and kepps history linear.
      # Run git diff before staging. It´s the moment to catch a paragraph that has been delted by accident.
      # And before pusing, the secret check - one line, and it´s what would have saved, earlier:

      git grep --cached -n "mZYXd\|password\|token"
```
7. **Everyday update of any script in github.com/edbkei/QVDE from Jetson**
 ```bash

      # Evidence that all connections are good
      cd /mnt/nvme/iot-stack

      docker logs --tail=20 mqtt-influx-bridge

      # Check LSTM service logs
      docker logs --tail=30 custom-lstm-detector

      # from operator interface send list_models
      # List of models in the /app/models directory

      # Note: List of available models, status response back to MQTT.

      # Test sensor simulator in inference mode
      # Send a single test reading
      simulator.send_reading(is_fall=False, operational_mode="inference")

      # check the  bridge logs:
      # should see: 
      #  - DEBUG: Received message
      #  - Topic: io/house_001/fall_detection?accelerometer/sensor_
      #  - Has sensor_data: True
      #  - Write successful

      # 📊 Understanding Log Output

      # Breaking down what each part means:

      # DEBUG: Received message
      #   Topic: iot/model/fall_detection/command

      # ↳ **Good**: Bridge received MQTT message on command topic
      #   Payload length: 119 bytes
      #   First 200 chars: b'{"command": "list_models", ...}'

      # ↳ **Good**: Valid JSON command from Operator Interface

      # 📋 Model Command: list_models
      #   Parameters: {}

      # ↳ **Good**: Bridge correctly parsed the command

      # ?? Attempting write to bucket: sensors
      #   Point (first 300 chars): model_commands,application=fall_detection,command=list_models parameters="{}"
      # ✅ Write successful to sensors!


      # 1. Operator Interface is Connected
      # Topic: iot/model/fall_detection/command
      # "command": "list_models"
      # "source": "operator_interface"

      # Note: This shows operator interface successfully sent the list_models command to Jetson

      # 2. MQTT Bridge is Working
      # 📋 Model Command: list_models
      #   Parameters: {}
      # ✅ Write successful to sensors!

      # Note: The bridge received the command and logged it to InfluxDB

      # 3. Communication flow is Active
      # Operator Interface (Desktop) 
      #    → MQTT Command 
      #    → Jetson MQTT Broker 
      #    → MQTT Bridge 
      #    → Logged Successfully
      # Note: Commands being received and processed
```
8. **Everyday update of any script in github.com/edbkei/QVDE from Windows**
 ```bash
      # In VScode, for instance, update the file README.md

      # In Windows using Powershell:
      cd $HOME\QVDE
      git pull origin main # start of every session
      # error: cannot pull with rebase: You have unstaged changes.
      # error: Please commit or stash them.
      # if this happens do:
      git status
      git stash
      git pull
      git stash pop
      git add README.md
      git commit -m "Update README.md from Windows"
      git push origin main
      # end if

      # ...edit files in desktop\ ...

      git status
      git diff
      git diff README.md
      git add README.md
      git commit -m "Update README.md from Windows"
      git push origin main

      # ... add new file ...

      cd $HOME\QVDE
      git checkout main
      git pull
      git status                       # README.CLOUD.md deve aparecer em "Untracked files"
      git add README.CLOUD.md
      git commit -m "Add cloud maintenance guide"
      git push

```

9. **Everyday update of any script in github.com/edbkei/QVDE in Ubuntu/LOCAWEB, this is a working up LLM**
 ```bash
      # Using mRemoteNG
      # qvde_relatorio.py — Generate an output with LLM report for health specialist from an input synthetic data produced by a "possible digital twin"
 
      # Read JSON with:
      # (1) prediction of fall detection from digital twin (accelerometer and gyroscope);
      # (2) records from blood pressure medical charts. 
      # The figures are pre-processed deterministically (averages, classification, orthostatic hypotension, temporal correlation with events), and a local LLM  (Ollama) drafts the report.
      # Thus, the small model interprets the data but does not perform calculations - and the prompt remains short (important for CPU usage).

      # Only once
      ollama list
      ollama pull qwen2.5:3b

      # everyday for an example
      cd ~/Projects/llm_example
      ls -la
      # possibilities:
      python3 qvde_relatorio.py exemplo_entrada.json --modelo qwen2.5:3b --saida rel_3b.md
      # other possibilities:
      python3 qvde_relatorio.py exemplo_entrada.json --modelo qwen2.5:3b --saida relatorio.md
      python3 qvde_relatorio.py exemplo_entrada.json --prompt-arquivo prompt_exemplo.json
      python3 qvde_relatorio.py exemplo_entrada.json --somente-contexto     # não chama o LLM
      cat dados.json | python3 qvde_relatorio.py -

```

10. **Report Production**
 ```bash
      # Using mRemoteNG, open cmd
      # 10.1 Find IP address in VM, terminal 1
      docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' qvde-timescaledb-1
      R: <IP>

      # 10.2 Using Windows Powershell in desktop, terminal 2, tunnel openning towards VM
      ssh -N -L 15432:<IP>:5432 statuser01@vps55112.publiccloud.com.br
      
      # R: no answer means that a tunnel was successfully created

      # 10.3 Using Windows Powershell in desktop, terminal 3
      conda activate qvde
      cd $HOME\QVDE
      # pip install -r cloud\requirements.txt
      $pw = ssh statuser01@vps55112.publiccloud.com.br "grep ^POSTGRES_PASSWORD= ~/Projects/QVDE/.env.cloud | cut -d= -f2-"
      $env:DATABASE_URL = "postgresql://qvde:$pw@127.0.0.1:15432/qvde"
      Remove-Variable pw
      $env:OLLAMA_HOST = "http://localhost:11434"
      $env:PYTHONPATH = "$HOME\QVDE;$HOME\QVDE\cloud"

      # 10.4 Using Windows Powershell in desktop, terminal 3, quick test before reports
      python -m qvde_cloud.report_job --paciente PAC-0001 --perfil contexts\perfis\geriatra_7b.yaml --somente-contexto


```


### First-Time Setup

1. **Access Grafana Dashboard**
   - URL: `http://<jetson-ip>:3000`
   - Default credentials: `admin` / `admin`

2. **Configure InfluxDB Data Source** (in Grafana)
   - URL: `http://influxdb:8086`
   - Organization: `nvme_influxdb_org0001`
   - Token: (from `.env` file)
   - Default bucket: `sensors`

3. **Load Pre-trained Model**
   ```bash
   # Models are automatically loaded from /app/models
   # Check available models
   sudo docker exec custom-lstm-detector ls -lh /app/models
   ```

---
### **Training phase**
### Check logs
1. Open Operator Interface -> Training Control tab
2. Set parameters:
   - Data Window (hours): 720
   - Training Epochs: 50
   - Validation Split: 0.2
3.  Click "🚀 Start Training"
4.  Watch the logs:

docker logs -f custom-lstm-detector
```
**See something like:**

================================================================================
🎓 STARTING TRAINING FROM INFLUXDB
================================================================================
  Data source: InfluxDB training_data bucket
  Time range: Last 720 hours  ← THIS WILL WORK!
  Epochs: 50
  Validation split: 20.0%
================================================================================

📊 Fetching training data from InfluxDB...
✅ Found 42309 training samples
📊 Preparing training dataset...
   Training samples: 33847
   Validation samples: 8462
   
📊 Training LSTM model...
Epoch 1/50: loss=0.2345, val_loss=0.1987
Epoch 2/50: loss=0.1876, val_loss=0.1654
...
Epoch 50/50: loss=0.0234, val_loss=0.0287

================================================================================
📊 VALIDATION RESULTS
================================================================================
Accuracy: 95.8%
Precision: 94.2%
Recall: 96.1%
F1 Score: 0.951

Confusion Matrix:
                 Predicted Normal  Predicted Fall
Actual Normal            7854              144
Actual Fall               164              300

True Positives: 300
False Positives: 144
True Negatives: 7854
False Negatives: 164
================================================================================

✅ Model v1.5.0 saved successfully!
🏆 New best model: v1.5.0 (Accuracy: 95.8%)
```
---
### **Inference Mode (test model)**
### Automatically swift to Inference
In Jupyter Notebook (sensor_simulator.ipynb)

### Generate inference test data
```
generate_inference_data(
    duration=60,          # 60 seconds of testing
    fall_probability=0.1  # 10% falls
)
```

In SSH terminal to HUB NVIDIA Jetson nano:

docker logs -f custom-lstm-detector
```
Should see:

[1] ✅ Normal | Risk: 8.2% | RL Action: do_nothing | Time: 1.2ms
[2] ✅ Normal | Risk: 12.4% | RL Action: do_nothing | Time: 1.1ms
[3] ⚠️ FALL DETECTED! | Risk: 87.5% | RL Action: alert | Time: 1.5ms
[4] ✅ Normal | Risk: 9.1% | RL Action: do_nothing | Time: 1.0ms

Note:
RL Action <- RL algorithm (LSTM current prediction, LSTM prediction history (recent last n-predictions))

state <- action(LSTM current prediction), trend(LSTM prediction history), event frequency)
action <- Q-table[state, action]

possible actions: do_nothing, notify_low_priority, notify_high_priority, alert

```
---
## 📊 Understanding Which Model is Better
Key Metrics (in order of importance):

| Metric |What It Means | Target | Meaning |
|--------|--------------|--------|---------|
| Recall | % of real falls detected | >90% (Most Important!) | Catches most falls (most import) |
| Accuracy | % of correct predictions overall | >90% | Model is good |
| Precision | % of fall alerts that are real | >85% | Few false alarms (few False Positives) |
|F1 Score | Balance of precision & recall | >0.90 | Overall excellent |

Example Comparison between models:

- v1.2.0: Accuracy 96.5%, Recall 95.8%, F1 0.950, 1000 samples
- v1.1.0: Accuracy 92.0%, Recall 90.0%, F1 0.892, 500 samples

Winner: v1.2.0 🏆 (Better in every metric!)

Note: If metrics are good ==> Proceed to inference testing, otherwise are bad ==> Retrain with more/better data.

## 📊 Usage

### 1. Turn on/off NVIDIA Jetson Orin Nano
```
# Turn on NVIDIA Jetson
Check red light on & fan running.

# Turn off NVIDIA Jetson
sudo shutdown now
Check red light off & fan stopped.

```

### 2. Sensor Simulation (Development/Testing)

Use the provided Jupyter notebook to simulate sensor data:

```python
# Open sensor_simulator_clean.ipynb on your laptop

# Configure connection
JETSON_IP = "xxx.xxx.xxx.xxx"  # Your Jetson IP

# Run continuous simulation
generate_inference_data(
    duration=120,
    fall_probability=0.10
)
```

### 3. Real Sensor Integration

For production deployment with real sensors:

```python
import paho.mqtt.client as mqtt
import json
from datetime import datetime

# Connect to MQTT broker
client = mqtt.Client()
client.connect("xxx.xxx.xxx.xxx", 1883, 60)

# Publish sensor reading
payload = {
    "metadata": {
        "house_id": "house_001",
        "sensor_id": "sensor_accel_bedroom_001",
        "application": "fall_detection",
        "timestamp": datetime.now().isoformat(),
        "operational_mode": "inference"
    },
    "device_info": {
        "sensor_type": "accelerometer",
        "location": "master_bedroom",
        "power_level": 85.2,
        "sensor_status": "active"
    },
    "sensor_data": {
        "accel_x": 0.1234,
        "accel_y": -0.0567,
        "accel_z": 9.8123,
        "gyro_roll": 2.3456,
        "gyro_pitch": -1.2345,
        "gyro_yaw": 0.5678
    }
}

topic = "iot/house_001/fall_detection/accelerometer/sensor_accel_bedroom_001"
client.publish(topic, json.dumps(payload), qos=1)
```

### 4. Model Training

Generate labeled training data:

```python
# In sensor simulator notebook
generate_training_data(
    num_samples=1000,
    fall_percentage=5.0
)
```

Trigger model training via MQTT:

```bash
# Publish training command
mosquitto_pub -h localhost \
  -t 'iot/model/fall_detection/command' \
  -m '{"command":"start_training","parameters":{"hours":24,"epochs":100,"validation_split":0.2}}'
```

### 5. Model Management

```bash
# List available models
mosquitto_pub -h localhost \
  -t 'iot/model/fall_detection/command' \
  -m '{"command":"list_models"}'

# Load best performing model
mosquitto_pub -h localhost \
  -t 'iot/model/fall_detection/command' \
  -m '{"command":"load_best_model"}'

# Load latest model
mosquitto_pub -h localhost \
  -t 'iot/model/fall_detection/command' \
  -m '{"command":"load_latest_model"}'

# Validate current model
mosquitto_pub -h localhost \
  -t 'iot/model/fall_detection/command' \
  -m '{"command":"validate_model"}'
```

---

## 🔍 Monitoring & Verification

### Real-Time 3-Terminal Monitoring

See `monitoring_guide.md` for complete instructions.

**Terminal 1 - MQTT Messages:**
```bash
sudo docker exec -it mosquitto mosquitto_sub -h localhost -t 'iot/#' -v
```

**Terminal 2 - Bridge Processing:**
```bash
sudo docker logs -f mqtt-influx-bridge | grep -E "sensor_data|Write successful"
```

**Terminal 3 - Model Predictions:**
```bash
sudo docker logs -f custom-lstm-detector | grep -E "Normal|FALL|Risk"
```

### Expected Output

When system is functioning correctly:

```
Terminal 3: [796] ✅ Normal | Risk: 46.9% | RL Action: do_nothing | Time: 1.3ms | Model: v1.1.0
Terminal 3: [797] 🚨 FALL DETECTED | Risk: 89.2% | RL Action: notify_high_priority | Time: 1.5ms
```

### Verification Commands

```bash
# Check system status
sudo docker-compose ps

# Verify data in InfluxDB
sudo docker exec influxdb influx query '
from(bucket: "sensors")
  |> range(start: -1m)
  |> filter(fn: (r) => r["_measurement"] == "fall_detection_accelerometer")
  |> count()
'

# Monitor system health
sudo docker stats

# View logs
sudo docker-compose logs -f

# View models with metrics
docker logs custom-lstm-detector | grep -A 50 "AVAILABLE MODELS"

# Newer command format
docker compose ps
docker ps

docker compose logs --tail=20 mqtt-influx-bridge
docker compose logs --tail=30 custom-lstm-detector

# docker-compose version
docker-compose --version

# docker compose restart
docker compose restart

# if docker ps fails. docker daemon probably not runninig or permission issue. Check.
# Start Docker
sudo systemctl start docker

# Check status
sudo systemctl status docker

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# How to Restart Properly
# Stop all services
docker compose down

# Wait a moment
sleep 2

# Start all services fresh
docker compose up -d

# Verify all started correctly
docker compose ps
```

### Expected Output:
```
NAME                    STATUS
influxdb               Up
grafana                Up  
mosquitto              Up
mqtt-influx-bridge     Up
custom-lstm-detector   Up

# Check services directly (bypasses docker-compose)
# Use docker directly (bypasses docker-compose)
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Or simpler
docker ps

# Check logs
docker logs --tail=20 mqtt-influx-bridge
docker logs --tail=30 custom-lstm-detector

# Quick Test. to verify docker is working
# Test 1: Docker daemon
docker ps

# Test 2: Your containers
docker ps | grep -E "(mqtt|lstm|influx)"

# Test 3: Logs work
docker logs --tail=5 mqtt-influx-bridge

# This should work regardless of docker-compose issue
docker logs custom-lstm-detector | grep -A 50 "AVAILABLE MODELS"

# Check current model
docker logs custom-lstm-detector | grep "CURRENT"

# Check MQTT bridge
docker logs --tail=20 mqtt-influx-bridge

# View full model list with all details
docker logs custom-lstm-detector | grep -A 50 "AVAILABLE MODELS"

# If that doesn't show enough, try:
docker logs custom-lstm-detector | tail -200 | grep -A 50 "AVAILABLE"
```

This will show something like:
```
================================================================================
AVAILABLE MODELS
================================================================================
v1.2.0
  Created: 2025-11-02T11:42:08
  Trained samples: 39200
  Accuracy: XX.X%
  ...
  
v1.1.0 👉 CURRENT
  Created: 2025-11-02T11:36:36
  Trained samples: XXXXX
  Accuracy: XX.X%
  ...
================================================================================


```


---

## 📈 Performance

### Model Performance (v1.1.0)

| Metric | Value |
|--------|-------|
| Validation Accuracy | 87.88% |
| Precision | 13.33% |
| Recall | 25.00% |
| F1 Score | 17.39% |
| Inference Time | <2ms |
| Model Size | 39.87 KB |

### System Performance

| Metric | Value |
|--------|-------|
| End-to-End Latency | <15ms |
| MQTT Message Rate | 10 Hz (configurable) |
| InfluxDB Write Rate | 10+ writes/sec |
| Model Inference Rate | Every 3 seconds |
| CPU Usage (Idle) | ~15% |
| Memory Usage | ~2GB |
| Storage per Day | ~500MB (retention: 30 days) |

### Scalability

- **Sensors per House**: 50+ (tested)
- **Simultaneous Houses**: 10+ (tested)
- **Data Points per Second**: 600+ (tested)
- **Model Switching**: <500ms

---

## 📁 Project Structure

In NVIDIA Jetson Orin Nano, in NVMe storage, at directory /mnt/nvme/iot-stack:
```
iot-fall-detection/
├── README.md                          # This file
├── LICENSE                            # License information
├── docker-compose.yml                 # Service orchestration
├── .env.example                       # Environment template
│
├── docs/                              # Documentation
│   ├── monitoring_guide.md            # 3-terminal monitoring guide
│   ├── monitoring_cheatsheet.md       # Quick reference
│   ├── architecture.md                # System architecture
│   └── api_reference.md               # API documentation
│
├── sensor_simulator/                  # Sensor simulation
│   ├── sensor_simulator_clean.ipynb   # Production simulator
│   └── requirements.txt               # Python dependencies
│
├── mqtt-influx-bridge/                # MQTT to InfluxDB bridge
│   ├── enhanced_mqtt_bridge.py        # Main bridge script
│   ├── Dockerfile                     # Container definition
│   └── requirements.txt               # Python dependencies
│
├── custom-lstm-detector/              # LSTM fall detection service
│   ├── enhanced_integration.py        # Main service script
│   ├── custom_lstm.py                 # Custom LSTM implementation
│   ├── rl_agent.py                    # Reinforcement learning agent
│   ├── models/                        # Trained models directory
│   │   ├── lstm_v1.1.0_*.pkl         # Model files
│   │   ├── rl_agent_v1.1.0_*.pkl     # RL agent files
│   │   └── *_metadata.json           # Model metadata
│   ├── Dockerfile                     # Container definition
│   └── requirements.txt               # Python dependencies
│
├── grafana/                           # Grafana dashboards
│   ├── dashboards/                    # Dashboard JSON files
│   └── provisioning/                  # Auto-provisioning configs
│
├── config/                            # Configuration files
│   ├── mosquitto.conf                 # MQTT broker config
│   └── influxdb.conf                  # InfluxDB config
│
└── scripts/                           # Utility scripts
    ├── setup.sh                       # Initial setup script
    ├── backup.sh                      # Backup models and data
    └── deploy.sh                      # Deployment automation
```

At Windows/Laptop
```
Project\fall-detection-operator:
├── operator_interface.py              # Operation Web Interface, Streamlit
├── sensor_simulator_improved.ipynb    # IoT traffic generatorm, MQTT - Jupyter Notebook
├── START_HERE.txt                     # Quick system introduction
├── DEPLOYMENT_GUIDE.md                # Operation overview

```
---

### Development Guidelines

- Follow PEP 8 style guide for Python code
- Add docstrings to all functions and classes
- Update documentation for new features
- Include tests when possible
- Keep commits atomic and well-described

### Areas for Contribution

- 🔬 **Research**: New algorithms, model architectures
- 🏗️ **Infrastructure**: Scalability improvements, monitoring
- 📱 **Sensors**: Integration with new wearable devices
- 🎨 **Visualization**: Grafana dashboards, alerting
- 📚 **Documentation**: Tutorials, guides, translations
- 🧪 **Testing**: Unit tests, integration tests, benchmarks

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📚 Citation

If you use this system in your research, please cite:

```bibtex
@software{iot_fall_detection_2025,
  author = {Eduardo S. Ito},
  title = {QVDE, Qualidade de Vida Estendida},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/edbkei/QVDE},
  note = {Doctoral research on adaptive agentic systems for home healthcare}
}
```

### Related Research

- Russell, S., & Norvig, P. (2021). *Artificial Intelligence: A Modern Approach* (4th ed.). Pearson.
- [Add your publications here]

---

## 🙏 Acknowledgments

- **NVIDIA**: For Jetson platform and GPU acceleration
- **Anthropic**: For AI research support
- **Open Source Community**: For foundational tools and libraries
- **Research Advisors**: None
- **Funding**: None

---

## 📞 Contact & Support

- **Author**: Eduardo iTO
- **Email**: [eduardoito2010@yahoo.com]
- **Institution**: [UNICAMP]
- **GitHub**: [edbkei](https://github.com/edbkei)

### Reporting Issues

Found a bug? Have a feature request?
- 🐛 [Report an Issue](https://github.com/edbkei/QVDE/)
---

## 🗺️ Roadmap

### Current Version: 2.0

**Completed:**
- ✅ Multi-agent AI support
- ✅ Custom LSTM implementation
- ✅ Reinforcement learning integration
- ✅ Docker-based deployment
- ✅ Real-time monitoring
- ✅ Model versioning
- ✅ Grafana visualization

### Version 2.1 (Q1 2025)

- [ ] WebSocket-based real-time alerts
- [ ] Mobile app for notifications
- [ ] Enhanced RL agent with more states
- [ ] Automated model retraining pipeline
- [ ] Multi-sensor fusion (camera + wearable)

### Version 3.0 (Q2 2025)

- [ ] Federated learning across multiple houses
- [ ] Privacy-preserving analytics
- [ ] Caregiver dashboard
- [ ] Voice assistant integration
- [ ] Extended health monitoring (BP, glucose, etc.)

### Long-term Vision

- Commercialization for residential care facilities
- Integration with electronic health records (EHR)
- Regulatory compliance (HIPAA, GDPR)
- Clinical validation studies
- White-label solution for healthcare providers

---

## 📖 Documentation

- [Complete Monitoring Guide](docs/monitoring_guide.md)
- [Quick Reference Cheatsheet](docs/monitoring_cheatsheet.md)
- [System Architecture Deep Dive](docs/architecture.md)
- [API Reference](docs/api_reference.md)
- [Sensor Integration Guide](docs/sensor_integration.md)
- [Model Training Guide](docs/model_training.md)
- [Deployment Guide](docs/deployment.md)
- [Troubleshooting](docs/troubleshooting.md)

---


