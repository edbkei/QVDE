# Complete Step-by-Step Deployment Guide

## 📋 Overview

This guide will help you deploy the **complete system** in the correct order.

**What you'll have:**
- Desktop: Jupyter Notebook for data collection
- Jetson: Docker services (MQTT, InfluxDB, Grafana, Custom LSTM, RL Agent)

**Time required:** ~30-40 minutes

---

## Part A: Jetson Orin Nano Setup

### Step 1: Create Directory Structure

```bash
# SSH into Jetson
ssh eduardo@JETSON_IP

# Create main directory
sudo mkdir -p /mnt/nvme/iot-stack
cd /mnt/nvme/iot-stack

# Set ownership
sudo chown -R $USER:$USER /mnt/nvme/iot-stack

# Create subdirectories
mkdir -p influxdb/{data,config}
mkdir -p grafana/{data,provisioning/datasources}
mkdir -p mosquitto/{config,data,log}
mkdir -p scripts
mkdir -p models
mkdir -p logs
```

### Step 2: Create Configuration Files

#### 2.1 Mosquitto Config

```bash
nano mosquitto/config/mosquitto.conf
```

**Content:**
```ini
listener 1883
allow_anonymous true
persistence true
persistence_location /mosquitto/data/
log_dest file /mosquitto/log/mosquitto.log
log_dest stdout
```

Save: `Ctrl+X`, `Y`, `Enter`

#### 2.2 Grafana Datasource

```bash
mkdir -p grafana/provisioning/datasources
nano grafana/provisioning/datasources/influxdb.yml
```

**Content:**
```yaml
apiVersion: 1
datasources:
  - name: InfluxDB
    type: influxdb
    access: proxy
    url: http://influxdb:8086
    jsonData:
      version: Flux
      organization: myorg
      defaultBucket: sensors
      tlsSkipVerify: true
    secureJsonData:
      token: my-super-secret-auth-token
    editable: true
```

Save: `Ctrl+X`, `Y`, `Enter`

#### 2.3 System Config (Optional - for multiple houses)

```bash
nano system_config.yaml
```

**Content:**
```yaml
global:
  mqtt_broker: "mosquitto"
  mqtt_port: 1883
  influxdb_url: "http://influxdb:8086"
  influxdb_token: "my-super-secret-auth-token"
  influxdb_org: "myorg"
  influxdb_bucket: "sensors"

houses:
  - house_id: "house_001"
    name: "My Home"
    address: "123 Main St"
    residents:
      - name: "User"
        age: 70
    sensors:
      - sensor_id: "sensor_accel_001"
        sensor_type: "accelerometer"
        application: "fall_detection"
        location: "bedroom"
        enabled: true
```

Save: `Ctrl+X`, `Y`, `Enter`

### Step 3: Create Python Scripts

#### 3.1 Custom LSTM (Your proven implementation)

```bash
nano scripts/custom_lstm_persistence.py
```

**Copy the ENTIRE content from the document you provided** (the SimpleLSTM class with save/load functionality)

Save: `Ctrl+X`, `Y`, `Enter`

#### 3.2 Enhanced MQTT Bridge

```bash
nano scripts/enhanced_mqtt_bridge.py
```

**Content:** Copy from the "Enhanced MQTT Bridge with Training/Inference Modes" artifact

Save: `Ctrl+X`, `Y`, `Enter`

#### 3.3 Custom LSTM Service

```bash
nano scripts/integrated_custom_lstm_service.py
```

**Content:** Copy from the "Integrated Custom LSTM Service with Training/Inference Modes" artifact

Save: `Ctrl+X`, `Y`, `Enter`

### Step 4: Create Requirements Files

#### 4.1 Bridge Requirements

```bash
nano requirements-bridge.txt
```

**Content:**
```txt
paho-mqtt==1.6.1
influxdb-client==1.38.0
python-dateutil>=2.8.0
```

#### 4.2 Custom LSTM Requirements

```bash
nano requirements-lstm.txt
```

**Content:**
```txt
paho-mqtt==1.6.1
influxdb-client==1.38.0
numpy>=1.24.0,<2.0.0
python-dateutil>=2.8.0
```

### Step 5: Create Dockerfiles

#### 5.1 Bridge Dockerfile

```bash
nano Dockerfile.bridge
```

**Content:**
```dockerfile
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY requirements-bridge.txt .
RUN pip install --no-cache-dir -r requirements-bridge.txt

COPY scripts/enhanced_mqtt_bridge.py /app/

RUN mkdir -p /app/logs

CMD ["python", "-u", "enhanced_mqtt_bridge.py"]
```

#### 5.2 Custom LSTM Dockerfile

```bash
nano Dockerfile.lstm
```

**Content:**
```dockerfile
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY requirements-lstm.txt .
RUN pip install --no-cache-dir -r requirements-lstm.txt

COPY scripts/custom_lstm_persistence.py /app/
COPY scripts/integrated_custom_lstm_service.py /app/

RUN mkdir -p /app/models /app/logs

CMD ["python", "-u", "integrated_custom_lstm_service.py"]
```

### Step 6: Create docker-compose.yml

```bash
nano docker-compose.yml
```

**Content:**
```yaml
version: '3.8'

services:
  influxdb:
    image: influxdb:latest
    container_name: influxdb
    restart: unless-stopped
    ports:
      - "8086:8086"
    volumes:
      - ./influxdb/data:/var/lib/influxdb2
      - ./influxdb/config:/etc/influxdb2
    environment:
      - DOCKER_INFLUXDB_INIT_MODE=setup
      - DOCKER_INFLUXDB_INIT_USERNAME=admin
      - DOCKER_INFLUXDB_INIT_PASSWORD=${INFLUXDB_ADMIN_PASSWORD}
      - DOCKER_INFLUXDB_INIT_ORG=myorg
      - DOCKER_INFLUXDB_INIT_BUCKET=sensors
      - DOCKER_INFLUXDB_INIT_ADMIN_TOKEN=my-super-secret-auth-token
    networks:
      - iot-network

  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    restart: unless-stopped
    ports:
      - "3000:3000"
    volumes:
      - ./grafana/data:/var/lib/grafana
      - ./grafana/provisioning:/etc/grafana/provisioning
    environment:
      - GF_SECURITY_ADMIN_USER=admin
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_ADMIN_PASSWORD}
    depends_on:
      - influxdb
    networks:
      - iot-network

  mosquitto:
    image: eclipse-mosquitto:latest
    container_name: mosquitto
    restart: unless-stopped
    ports:
      - "1883:1883"
      - "9001:9001"
    volumes:
      - ./mosquitto/config:/mosquitto/config
      - ./mosquitto/data:/mosquitto/data
      - ./mosquitto/log:/mosquitto/log
    networks:
      - iot-network

  mqtt-influx-bridge:
    build:
      context: .
      dockerfile: Dockerfile.bridge
    container_name: mqtt-influx-bridge
    restart: unless-stopped
    depends_on:
      - mosquitto
      - influxdb
    environment:
      - MQTT_BROKER=mosquitto
      - MQTT_PORT=1883
      - INFLUXDB_URL=http://influxdb:8086
      - INFLUXDB_TOKEN=my-super-secret-auth-token
      - INFLUXDB_ORG=myorg
      - INFLUXDB_BUCKET_PRODUCTION=sensors
      - INFLUXDB_BUCKET_TRAINING=training_data
    volumes:
      - ./logs:/app/logs
    networks:
      - iot-network

  custom-lstm-detector:
    build:
      context: .
      dockerfile: Dockerfile.lstm
    container_name: custom-lstm-detector
    restart: unless-stopped
    depends_on:
      - mqtt-influx-bridge
    environment:
      - MQTT_BROKER=mosquitto
      - MQTT_PORT=1883
      - INFLUXDB_URL=http://influxdb:8086
      - INFLUXDB_TOKEN=my-super-secret-auth-token
      - INFLUXDB_ORG=myorg
      - INFLUXDB_BUCKET_PRODUCTION=sensors
      - INFLUXDB_BUCKET_TRAINING=training_data
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
    networks:
      - iot-network

networks:
  iot-network:
    driver: bridge
```

### Step 7: Set Permissions

```bash
cd /mnt/nvme/iot-stack

# Grafana
sudo chown -R 472:472 grafana/data

# Mosquitto
sudo chown -R 1883:1883 mosquitto
```

### Step 8: Build and Start Services

```bash
cd /mnt/nvme/iot-stack

# Build all services (takes 5-10 minutes first time)
docker-compose build

# Start all services
docker-compose up -d

# Wait 30 seconds for initialization
sleep 30

# Check status
docker-compose ps
```

**Expected output:**
```
NAME                   STATUS
influxdb               Up
grafana                Up
mosquitto              Up
mqtt-influx-bridge     Up
custom-lstm-detector   Up
```

### Step 9: Verify Services

```bash
# Check logs
docker-compose logs influxdb --tail 20
docker-compose logs mqtt-influx-bridge --tail 20
docker-compose logs custom-lstm-detector --tail 20

# Get Jetson IP
hostname -I
```

Note your Jetson IP (e.g., `192.168.1.100`)

---

## Part B: Desktop Setup (Jupyter Notebook)

### Step 1: Create Desktop Directory

```bash
# On your desktop
mkdir ~/iot-fall-detection
cd ~/iot-fall-detection
```

### Step 2: Create requirements.txt

```bash
nano requirements.txt
```

**Content:**
```txt
paho-mqtt==1.6.1
numpy>=1.24.0
jupyter>=1.0.0
ipython>=8.12.0
```

### Step 3: Create Virtual Environment

```bash
# Create environment
python3 -m venv venv

# Activate
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate  # Windows

# Install packages
pip install -r requirements.txt
```

### Step 4: Create Jupyter Notebook

```bash
jupyter notebook
```

This will open your browser. Create a new notebook called `fall_detection_simulator.ipynb`

### Step 5: Add Simulator Code to Notebook

**Cell 1: Imports and Configuration**

```python
import paho.mqtt.client as mqtt
import numpy as np
import json
import time
from datetime import datetime
import random

# Configuration - CHANGE THIS!
JETSON_IP = "192.168.1.100"  # ⚠️ Change to your Jetson IP
MQTT_PORT = 1883
HOUSE_ID = "house_001"
```

**Cell 2: Simple Simulator Class**

```python
class SimpleFallDetectionSimulator:
    """Simple simulator for testing"""
    
    def __init__(self, broker_ip, house_id="house_001"):
        self.client = mqtt.Client(client_id=f"simulator_{house_id}")
        self.broker = broker_ip
        self.house_id = house_id
        self.is_connected = False
        self.power_level = 100.0
        
        # MQTT callbacks
        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect
    
    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print(f"✅ Connected to MQTT Broker at {self.broker}")
            self.is_connected = True
        else:
            print(f"❌ Connection failed: {rc}")
    
    def on_disconnect(self, client, userdata, rc):
        print("⚠️  Disconnected")
        self.is_connected = False
    
    def connect(self):
        try:
            self.client.connect(self.broker, MQTT_PORT, 60)
            self.client.loop_start()
            time.sleep(2)
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def disconnect(self):
        self.client.loop_stop()
        self.client.disconnect()
    
    def generate_sensor_data(self, simulate_fall=False):
        """Generate sensor data"""
        if simulate_fall:
            # Fall pattern
            x, y, z = np.random.normal(0, 15, 3)
            z = np.random.normal(-20, 10)
            roll, pitch, yaw = np.random.normal(0, 180, 3)
        else:
            # Normal pattern
            x, y = np.random.normal(0, 0.5, 2)
            z = np.random.normal(-9.81, 0.3)
            roll, pitch, yaw = np.random.normal(0, 5, 3)
        
        magnitude = np.sqrt(x**2 + y**2 + z**2)
        
        # Update power
        self.power_level = max(0, min(100, self.power_level + np.random.normal(-0.5, 1)))
        
        return {
            "metadata": {
                "house_id": self.house_id,
                "sensor_id": "sensor_accel_001",
                "application": "fall_detection",
                "timestamp": datetime.now().isoformat(),
                "version": "1.0"
            },
            "device_info": {
                "sensor_type": "accelerometer",
                "sensor_status": "online",
                "power_level": round(self.power_level, 2),
                "location": "bedroom"
            },
            "sensor_data": {
                "x": round(x, 4),
                "y": round(y, 4),
                "z": round(z, 4),
                "magnitude": round(magnitude, 4),
                "gyro_roll": round(roll, 4),
                "gyro_pitch": round(pitch, 4),
                "gyro_yaw": round(yaw, 4)
            }
        }
    
    def publish_data(self):
        """Publish sensor data"""
        if not self.is_connected:
            print("⚠️  Not connected")
            return
        
        # 5% chance of fall
        simulate_fall = random.random() < 0.05
        
        data = self.generate_sensor_data(simulate_fall)
        
        topic = f"iot/{self.house_id}/fall_detection/accelerometer/sensor_accel_001"
        
        self.client.publish(topic, json.dumps(data), qos=1)
        
        status = "🚨 FALL!" if simulate_fall else "✅ Normal"
        power_emoji = "🔋" if self.power_level > 50 else "🪫"
        
        print(f"\r{status} | Power: {self.power_level:.1f}% {power_emoji}", end='', flush=True)
    
    def run(self, duration=60, interval=0.1):
        """Run simulator"""
        print(f"🚀 Starting Simulator")
        print(f"📡 Broker: {self.broker}")
        print(f"🏠 House: {self.house_id}")
        print(f"⏱️  Duration: {duration}s, Interval: {interval}s")
        print("-" * 60)
        
        self.connect()
        
        if not self.is_connected:
            print("❌ Failed to connect")
            return
        
        start_time = time.time()
        count = 0
        
        try:
            while (time.time() - start_time) < duration:
                self.publish_data()
                count += 1
                time.sleep(interval)
        
        except KeyboardInterrupt:
            print("\n⏹️  Stopped by user")
        
        finally:
            print(f"\n📊 Published {count} messages")
            self.disconnect()
```

**Cell 3: Run Simulator**

```python
# Create simulator
simulator = SimpleFallDetectionSimulator(
    broker_ip=JETSON_IP,
    house_id=HOUSE_ID
)

# Run for 60 seconds (10 readings/second)
simulator.run(duration=60, interval=0.1)
```

---

## Part C: Testing the System

### Test 1: Verify MQTT Messages

**On Jetson:**
```bash
# Subscribe to all messages
mosquitto_sub -h localhost -t 'iot/#' -v
```

**On Desktop:**
Run Cell 3 in Jupyter

You should see messages appearing on Jetson terminal.

### Test 2: Check InfluxDB

**On Jetson:**
```bash
# Wait 30 seconds after running simulator

# Check data
docker-compose exec influxdb influx query '
from(bucket:"sensors")
  |> range(start: -5m)
  |> filter(fn: (r) => r["house_id"] == "house_001")
  |> count()
'
```

You should see data counts.

### Test 3: Access Grafana

**On Desktop Browser:**
- Open: `http://JETSON_IP:3000`
- Login: `admin` / (see .env)
- Go to Explore → Select InfluxDB datasource
- Query:
```flux
from(bucket: "sensors")
  |> range(start: -5m)
  |> filter(fn: (r) => r["house_id"] == "house_001")
```

You should see data.

### Test 4: Check Custom LSTM

**On Jetson:**
```bash
# Check LSTM logs
docker-compose logs custom-lstm-detector --tail 50

# Should see predictions
mosquitto_sub -h localhost -t 'ai/fall_prediction' -v
```

---

## Part D: Training the Model

### Step 1: Generate Training Data

**In Jupyter (Cell 4):**

```python
# Training mode simulator
print("🎓 Collecting Training Data...")
print("Press 'F' when you see a fall to label it")
print("-" * 60)

# Run for 10 minutes to collect data
simulator.run(duration=600, interval=0.1)
```

### Step 2: Train Model

**On Jetson:**
```bash
# Send training command
mosquitto_pub -h localhost \
  -t "iot/model/fall_detection/command" \
  -m '{
    "command": "start_training",
    "parameters": {
      "episodes": 1000,
      "hours": 1
    }
  }'

# Watch training
docker-compose logs custom-lstm-detector -f
```

### Step 3: Save Model

**After training completes:**
```bash
mosquitto_pub -h localhost \
  -t "iot/model/fall_detection/command" \
  -m '{"command": "save_model"}'
```

---

## 📋 Quick Command Reference

### Jetson Commands

```bash
# Start/Stop
cd /mnt/nvme/iot-stack
docker-compose up -d
docker-compose down

# Status
docker-compose ps

# Logs
docker-compose logs <service> --tail 50
docker-compose logs <service> -f

# Rebuild
docker-compose build <service>
docker-compose up -d <service>

# MQTT Test
mosquitto_sub -h localhost -t 'iot/#' -v
mosquitto_pub -h localhost -t 'test' -m 'hello'
```

### Desktop Commands

```bash
# Activate environment
source venv/bin/activate

# Start Jupyter
jupyter notebook

# Check Python packages
pip list
```

---

## 🐛 Troubleshooting

### Issue: Cannot connect from desktop

```bash
# On Jetson - check firewall
sudo ufw status
sudo ufw allow 1883/tcp

# Check MQTT is listening
sudo netstat -tlnp | grep 1883
```

### Issue: Docker services won't start

```bash
# Check logs
docker-compose logs <service>

# Check permissions
ls -la influxdb/data
ls -la grafana/data

# Fix permissions
sudo chown -R 472:472 grafana/data
sudo chown -R 1883:1883 mosquitto
```

### Issue: No data in InfluxDB

```bash
# Check bridge is running
docker-compose logs mqtt-influx-bridge

# Verify MQTT messages
mosquitto_sub -h localhost -t 'iot/#' -C 10
```

---

## ✅ Verification Checklist

- [ ] Jetson directory created
- [ ] All config files created
- [ ] All Python scripts created
- [ ] Dockerfiles created
- [ ] docker-compose.yml created
- [ ] Permissions set
- [ ] Services built successfully
- [ ] All 5 containers running
- [ ] Desktop environment created
- [ ] Jupyter notebook working
- [ ] Simulator connects to Jetson
- [ ] Data appears in InfluxDB
- [ ] Grafana accessible
- [ ] LSTM making predictions

**If all checked ✅ - You're ready to go!** 🎉
