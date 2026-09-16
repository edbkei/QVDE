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
- **100% Local Processing**: No cloud dependencies
- **Edge Computing**: All AI inference runs on-premise
- **Offline Operation**: Continues functioning during internet outages
- **Data Sovereignty**: Health data never leaves the premises

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
┌───────────────────────────────────────────────────────────────────────────────┐
│                               Smart Home Network                              │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌──────────────┐  ┌──────────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │  Wearable    │  │    Sensor        │  │ Operation Web|  │  Other IoT   |   |
│  │  Sensors     │  │   Simulator      │  │  Interface   |  |  Devices     |   |
│  │ (Accel/Gyro) │  │(Jupyter Notebook)|  | (Port 8501)  │  |              │   │
|  |              |  |   -- Laptop --   |  | -- Laptop -- |  |              |   | 
│  └──────┬───────┘  └──────────┬───────┘  └──────┬───────┘  └──────┬───────┘   |
│         │                     │                 │                 │           |
│         └─────────────────────┴────────┼────────┘─────────────────┘           │
│                                        │                                      │
:                                        :                                      :
├────────────────────────────────────────┼──────────────────────────────────────┤
│                                    MQTT Broker                                │
│                               (Mosquitto - Port 1883)                         │
│                                        │                                      │
│            ┌───────────────────────────┼─────────────────────────┐            │
│            │                           │                         │            │
│  ┌─────────▼────────┐  ┌───────────────▼────────────┐  ┌─────────▼─────────┐  │
│  │  MQTT-InfluxDB   │  │        Custom LSTM -       │  │ Grafana Dashboard │  │
│  │      Bridge      │  │   Fall Detection Service   │  |    (Port 3000)    │  │
│  │  -- Docker --    │  │       -- Docker --         │  │   -- Docker --    │  │
│  └────┬─────────────┘  └──┬──────────┬───────────┬──┘  └───────────────────┘  │
│       │                   │          |           |                            │
│  ┌────▼───────────────────▼───┐ ┌────▼────┐ ┌────▼────┐                       │
│  │        InfluxDB            │ | ML/RL   | |         |                       │
│  │   Time-Series Database     │ | Models  | |   Logs  |                       │
│  │       (Port 8086)          │ | Version | |         |                       │
|  |      -- Docker --          | | V1...Vn | |         |                       |
│  └────────────────────────────┘ └─────────┘ └─────────┘                       │
│                  NVIDIA Jetson Orin Nano (Edge Computing Hub)                 │
└───────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Wearable Sensor → MQTT → Bridge → InfluxDB → LSTM Model → RL Agent → Action
    (10 Hz)      (1ms)   (5ms)     (10ms)      (1ms)       (0.5ms)   (Alert)
```

### Component Interaction

```
┌─────────────────────────────────────────────────────────────────┐
│                     Sensor Simulator                            │
│                    (Windows/Laptop)                             │
└──────────────────────────┬──────────────────────────────────────┘
                           │ MQTT (JSON)
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│                  Mosquitto MQTT Broker                          │
│              Topics: iot/{house_id}/{app}/{type}                │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                  ┌────────┴────────┐
                  ↓                 ↓
┌──────────────────────────┐  ┌─────────────────────────────┐
│  MQTT-InfluxDB Bridge    │  │  LSTM Detection Service     │
│  - Parse MQTT messages   │  │  - Load model               │
│  - Tag extraction        │  │  - Query InfluxDB           │
│  - Write to InfluxDB     │  │  - Run inference            │
└──────────┬───────────────┘  │  - RL decision-making       │
           │                  └─────────┬───────────────────┘
           ↓                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                         InfluxDB                                │
│  Buckets:                                                       │
│  - sensors (production data)                                    │
│  - training_data (labeled data)                                 │
│  Retention: 30 days (configurable)                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Technology Stack

### Edge Computing Platform
- **NVIDIA Jetson Orin Nano**: AI-accelerated edge device
  - 6-core ARM CPU
  - 1024-core NVIDIA Ampere GPU
  - 8GB RAM
  - Ubuntu 24.04 LTS

### Core Infrastructure
- **Docker & Docker Compose**: Containerized deployment
- **Mosquitto (v2.0+)**: MQTT message broker
- **InfluxDB (v2.7+)**: Time-series database
- **Grafana (v10+)**: Visualization and monitoring
- **Streamlit (3.13+)**: Operation web interface to control the training/inference phase of a ML/AI model.

### AI/ML Stack
- **Python 3.8+**: Core programming language
- **NumPy**: Numerical computing
- **Custom LSTM Implementation**: Optimized for edge devices
  - 5x faster inference than Keras LSTM
  - 127x smaller model size
  - Hand-optimized for ARM/GPU architecture

### Communication Protocols
- **MQTT**: Low-latency IoT messaging (QoS 1)
- **HTTP/REST**: API communication with InfluxDB
- **WebSocket**: Real-time Grafana updates

### Development Tools
- **Jupyter Notebook**: Sensor simulation and experimentation
- **Streamlit**: Operation web interface
- **Git**: Version control
- **Docker**: Containerization

---

## 🔬 Research Context

This project is part of doctoral research on **Extended Home Quality of Life (QVDE - Qualidade de Vida Domiciliar Estendida)**, focusing on adaptive agentic system paradigms for home healthcare technology integration.

### Theoretical Framework

The research proposes an intelligent hub architecture that satisfies the formal requirements for agentic systems as defined by Russell & Norvig's "Artificial Intelligence: A Modern Approach":

- ✅ **Autonomy**: Self-directed operation without human intervention
- ✅ **Reactivity**: Real-time response to environmental changes
- ✅ **Proactivity**: Goal-directed behavior and planning
- ✅ **Sociability**: Multi-agent communication and coordination
- ✅ **Adaptability**: Learning from experience (RL agent)
- ✅ **Goal-Orientation**: Fall detection and prevention objectives

### Digital Twin Paradigm

The system uses digital twins for:
- **Representation**: Virtual models of physical sensors and residents
- **Learning**: Continuous model improvement from operational data
- **Prediction**: Anticipating falls before they occur
- **Optimization**: Refining alert thresholds and response strategies

### Key Research Questions Addressed

1. How can isolated IoT devices be integrated into a cohesive healthcare system?
2. What are the trade-offs between local and cloud processing for health monitoring?
3. How can reinforcement learning improve alert accuracy and reduce false positives?
4. What architectural patterns enable scalable multi-house monitoring?

---

## 🚀 Getting Started

### Prerequisites

**Hardware Requirements:**
- NVIDIA Jetson Orin Nano (or compatible edge device)
- 16GB+ microSD card or NVMe SSD. In case of QVDE, all apps are installed under NVMe SSD, as its performance is 10x faster than microSD, and it supports 256 GB storage, and microSD only 128 GB.
- Network connectivity (Ethernet or WiFi)
- (Optional) Wearable accelerometer sensors

**Software Requirements:**
- Ubuntu 22.04+ (ARM64)
- Docker 24.0+
- Docker Compose 2.0+
- Python 3.8+
- Streamlit 3.13+

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
6. **Everyday update of any script in github.com/edbkei/QVDE**
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
7. **Everyday update of any script in github.com/edbkei/QVDE**
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


