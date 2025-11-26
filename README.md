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
┌─────────────────────────────────────────────────────────────────┐
│                      Smart Home Network                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐ │
│  │  Wearable    │      │   Motion     │      │  Other IoT   │ │
│  │  Sensors     │      │   Sensors    │      │   Devices    │ │
│  │ (Accel/Gyro) │      │              │      │              │ │
│  └──────┬───────┘      └──────┬───────┘      └──────┬───────┘ │
│         │                     │                     │          │
│         └─────────────────────┴─────────────────────┘          │
│                              │                                  │
│                         MQTT Broker                             │
│                    (Mosquitto - Port 1883)                      │
│                              │                                  │
│         ┌────────────────────┼────────────────────┐            │
│         │                    │                    │            │
│    ┌────▼─────┐      ┌───────▼────────┐   ┌──────▼──────┐    │
│    │  MQTT-   │      │   Custom LSTM  │   │   Grafana   │    │
│    │ InfluxDB │      │ Fall Detection │   │  Dashboard  │    │
│    │  Bridge  │      │     Service    │   │ (Port 3000) │    │
│    └────┬─────┘      └───────┬────────┘   └─────────────┘    │
│         │                    │                                 │
│    ┌────▼────────────────────▼─────┐                          │
│    │        InfluxDB                │                          │
│    │   Time-Series Database         │                          │
│    │      (Port 8086)               │                          │
│    └────────────────────────────────┘                          │
│                                                                 │
│                NVIDIA Jetson Orin Nano                          │
│              (Edge Computing Hub)                               │
└─────────────────────────────────────────────────────────────────┘
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
- 16GB+ microSD card or NVMe SSD
- Network connectivity (Ethernet or WiFi)
- (Optional) Wearable accelerometer sensors

**Software Requirements:**
- Ubuntu 22.04+ (ARM64)
- Docker 24.0+
- Docker Compose 2.0+
- Python 3.8+

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

## 📊 Usage

### 1. Sensor Simulation (Development/Testing)

Use the provided Jupyter notebook to simulate sensor data:

```python
# Open sensor_simulator_clean.ipynb on your laptop

# Configure connection
JETSON_IP = "192.168.15.116"  # Your Jetson IP

# Run continuous simulation
generate_inference_data(
    duration=120,
    fall_probability=0.10
)
```

### 2. Real Sensor Integration

For production deployment with real sensors:

```python
import paho.mqtt.client as mqtt
import json
from datetime import datetime

# Connect to MQTT broker
client = mqtt.Client()
client.connect("192.168.15.116", 1883, 60)

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

### 3. Model Training

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

### 4. Model Management

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

---

## 🤝 Contributing

Contributions are welcome! This is an active research project.

### How to Contribute

1. **Fork the Repository**
2. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make Your Changes**
4. **Test Thoroughly**
   - Verify Docker containers start correctly
   - Test data flow end-to-end
   - Validate model predictions
5. **Commit with Clear Messages**
   ```bash
   git commit -m "feat: add support for blood pressure monitoring"
   ```
6. **Push to Your Fork**
   ```bash
   git push origin feature/your-feature-name
   ```
7. **Open a Pull Request**

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
  title = {Multi-agent AI System for Extended Home Quality of Life},
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
- **Research Advisors**: [Add your advisors]
- **Funding**: [Add funding sources if applicable]

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

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/iot-fall-detection&type=Date)](https://star-history.com/#yourusername/iot-fall-detection&Date)

---

<div align="center">

**Built with ❤️ for safer, smarter homes**

[⬆ Back to Top](#-multi-house-iot-fall-detection-system)

</div>
