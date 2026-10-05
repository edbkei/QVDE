# Deployment Guide

## 📦 Prerequisites

### Hardware Requirements

**Jetson Orin Nano** (or similar):
- 8GB RAM minimum
- 32GB storage minimum
- Network connectivity
- Ubuntu 20.04/22.04

**Desktop/Laptop** (for Operator Interface & Sensor Simulator):
- 4GB RAM minimum
- Python 3.8+
- Web browser
- Network connectivity to Jetson

### Software Requirements

**On Jetson**:
- Docker 20.10+
- Docker Compose 1.29+
- Git (optional)

**On Desktop/Laptop**:
- Python 3.8 or higher
- pip (Python package manager)
- Jupyter Notebook
- Web browser (Chrome, Firefox, Edge, Safari)

## 🚀 Installation Steps

### Step 1: Prepare Jetson Device

```bash
# Update system
sudo apt update
sudo apt upgrade -y

# Install Docker (if not already installed)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose (if not already installed)
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Verify installation
docker --version
docker-compose --version

# Log out and back in for group changes to take effect
```

### Step 2: Deploy Project Files to Jetson

```bash
# Create project directory
mkdir -p ~/fall-detection-system
cd ~/fall-detection-system

# Copy or upload these files to Jetson:
# - docker-compose.yml
# - Dockerfile.bridge
# - Dockerfile.lstm
# - enhanced_mqtt_bridge.py
# - enhanced_integrated_lstm_service.py
# - integrated_custom_lstm_service.py
# - custom_lstm_persistence.py
# - requirements-bridge.txt
# - requirements-lstm.txt

# Create necessary directories
mkdir -p influxdb/data influxdb/config
mkdir -p grafana/data grafana/provisioning
mkdir -p mosquitto/config mosquitto/data mosquitto/log
mkdir -p models logs

# Set permissions
chmod -R 755 mosquitto
chmod -R 755 models
chmod -R 755 logs
```

### Step 3: Configure Mosquitto MQTT Broker

```bash
# Create Mosquitto configuration
cat > mosquitto/config/mosquitto.conf << EOF
listener 1883
allow_anonymous true
persistence true
persistence_location /mosquitto/data/
log_dest file /mosquitto/log/mosquitto.log
log_dest stdout
EOF
```

### Step 4: Start Services on Jetson

```bash
# Build and start all services
docker-compose up -d

# Verify all services are running
docker-compose ps

# Expected output:
# NAME                    STATUS
# influxdb               Up
# grafana                Up
# mosquitto              Up
# mqtt-influx-bridge     Up
# custom-lstm-detector   Up

# Check logs for any errors
docker-compose logs -f

# Press Ctrl+C to exit logs
```

### Step 5: Verify InfluxDB Setup

```bash
# Check if InfluxDB is accessible
curl http://localhost:8086/health

# Access InfluxDB UI
# Open browser and navigate to: http://<JETSON_IP>:8086
# Login with:
#   Username: admin
#   Password: <see .env>
#   Organization: nvme_influxdb_org0001

# Verify buckets exist:
# - sensors (production data)
# - training_data (labeled training data)
```

### Step 6: Setup Desktop/Laptop Environment

```bash
# On your desktop/laptop

# Create project directory
mkdir -p ~/fall-detection-operator
cd ~/fall-detection-operator

# Copy these files from the project:
# - sensor_simulator.ipynb
# - operator_interface.py
# - requirements-operator.txt

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Jupyter and dependencies
pip install jupyter numpy pandas matplotlib paho-mqtt

# Install Operator Interface dependencies
pip install -r requirements-operator.txt

# Verify installation
python -c "import streamlit; print('Streamlit installed successfully')"
jupyter --version
```

### Step 7: Configure Network Settings

#### Find Jetson IP Address

```bash
# On Jetson
ip addr show

# Or
hostname -I

# Note the IP address (e.g., 192.168.15.116)
```

#### Update Configuration Files

**1. Update Jupyter Notebook** (`sensor_simulator.ipynb`):

Open the notebook and edit the configuration cell:
```python
JETSON_IP = "192.168.15.116"  # Replace with your Jetson IP
```

**2. Operator Interface**:

The Operator Interface allows you to enter the Jetson IP in the web UI, but you can also set a default:

Edit `operator_interface.py`:
```python
DEFAULT_JETSON_IP = "192.168.15.116"  # Your Jetson IP
```

### Step 8: Test the System

#### 8.1 Test MQTT Connectivity

```bash
# On desktop/laptop
# Install mosquitto-clients
# Ubuntu/Debian: sudo apt install mosquitto-clients
# macOS: brew install mosquitto
# Windows: Download from https://mosquitto.org/download/

# Test connection
mosquitto_sub -h <JETSON_IP> -t "iot/#" -v

# Leave this running and proceed to next step
```

#### 8.2 Start Operator Interface

```bash
# On desktop/laptop (in project directory)
streamlit run operator_interface.py

# Browser should open automatically
# If not, navigate to: http://localhost:8501

# In the interface:
# 1. Enter Jetson IP in sidebar
# 2. Click "Connect"
# 3. Verify connection shows "🟢 Connected"
# 4. Click "Get Status" to verify system is responding
```

#### 8.3 Test Sensor Simulator

```bash
# On desktop/laptop (in project directory)
jupyter notebook

# 1. Open sensor_simulator.ipynb
# 2. Update Jetson IP in configuration cell
# 3. Run cells in order
# 4. Connect to MQTT broker
# 5. Send a test reading

# You should see the message in:
# - The mosquitto_sub terminal
# - MQTT Bridge logs on Jetson
# - InfluxDB (check in UI)
```

## 🔧 Configuration Options

### Environment Variables (Jetson)

Edit `docker-compose.yml` to customize:

```yaml
# MQTT Broker
MQTT_BROKER: mosquitto
MQTT_PORT: 1883

# InfluxDB
INFLUXDB_URL: http://influxdb:8086
INFLUXDB_TOKEN: [your-token]
INFLUXDB_ORG: nvme_influxdb_org0001
INFLUXDB_BUCKET_PRODUCTION: sensors
INFLUXDB_BUCKET_TRAINING: training_data
```

### Change Default Passwords

**InfluxDB**:
```yaml
# In docker-compose.yml
DOCKER_INFLUXDB_INIT_USERNAME: admin
DOCKER_INFLUXDB_INIT_PASSWORD: [your-secure-password]
```

**Grafana**:
```yaml
# In docker-compose.yml
GF_SECURITY_ADMIN_USER: admin
GF_SECURITY_ADMIN_PASSWORD: [your-secure-password]
```

## 📊 Verify Deployment

### Checklist

- [ ] All Docker containers running
- [ ] InfluxDB accessible and buckets created
- [ ] MQTT broker accepting connections
- [ ] Operator Interface connects to Jetson
- [ ] Sensor Simulator can send data
- [ ] Data appears in InfluxDB
- [ ] LSTM service responds to commands
- [ ] Predictions are generated

### Health Checks

```bash
# On Jetson

# 1. Check Docker containers
docker-compose ps

# 2. Check MQTT Bridge
docker logs mqtt-influx-bridge | tail -n 20

# 3. Check LSTM Service
docker logs custom-lstm-detector | tail -n 20

# 4. Check InfluxDB
curl http://localhost:8086/health

# 5. Check disk space
df -h

# 6. Check memory usage
free -h
```

## 🐛 Troubleshooting Deployment

### Issue: Docker containers won't start

**Symptoms**: `docker-compose up -d` fails

**Solutions**:
```bash
# Check logs
docker-compose logs

# Check for port conflicts
sudo netstat -tulpn | grep -E '8086|3000|1883'

# Rebuild containers
docker-compose down
docker-compose build --no-cache
docker-compose up -d

# Check disk space
df -h
```

### Issue: InfluxDB not accessible

**Symptoms**: Cannot access http://JETSON_IP:8086

**Solutions**:
```bash
# Check if container is running
docker ps | grep influxdb

# Check logs
docker logs influxdb

# Check if port is bound
sudo netstat -tulpn | grep 8086

# Restart container
docker-compose restart influxdb

# Check firewall
sudo ufw status
sudo ufw allow 8086/tcp
```

### Issue: MQTT connection refused

**Symptoms**: Operator Interface or Sensor Simulator cannot connect

**Solutions**:
```bash
# Check Mosquitto container
docker ps | grep mosquitto

# Check logs
docker logs mosquitto

# Test locally on Jetson
mosquitto_sub -h localhost -t test -v

# Check firewall
sudo ufw allow 1883/tcp

# Verify configuration
cat mosquitto/config/mosquitto.conf
```

### Issue: Data not appearing in InfluxDB

**Symptoms**: Sensor data sent but not in database

**Solutions**:
```bash
# Check MQTT Bridge logs
docker logs -f mqtt-influx-bridge

# Verify topics match
# In Jupyter: TOPIC_SENSOR_DATA value
# In bridge logs: subscribed topics

# Test MQTT directly
mosquitto_pub -h <JETSON_IP> -t "iot/test" -m '{"test": "message"}'

# Check InfluxDB permissions
# Verify token and organization match in docker-compose.yml
```

### Issue: LSTM service errors

**Symptoms**: Training fails or inference not working

**Solutions**:
```bash
# Check logs
docker logs -f custom-lstm-detector

# Verify models directory permissions
ls -la models/

# Check if training data exists in InfluxDB
# Access InfluxDB UI and query training_data bucket

# Restart service
docker-compose restart custom-lstm-detector
```

## 🔄 Updates and Maintenance

### Update Code

```bash
# On Jetson
cd ~/fall-detection-system

# Stop services
docker-compose down

# Update code files
# (copy new versions of Python scripts)

# Rebuild containers
docker-compose build

# Start services
docker-compose up -d

# Verify
docker-compose ps
docker-compose logs -f
```

### Backup Data

```bash
# Backup InfluxDB data
docker-compose exec influxdb influx backup /backup
docker cp influxdb:/backup ./influxdb_backup_$(date +%Y%m%d)

# Backup models
tar -czf models_backup_$(date +%Y%m%d).tar.gz models/

# Backup Grafana dashboards
tar -czf grafana_backup_$(date +%Y%m%d).tar.gz grafana/
```

### Clean Up

```bash
# Remove old containers
docker-compose down

# Remove volumes (WARNING: deletes all data)
docker-compose down -v

# Clean up Docker system
docker system prune -a
```

## 📈 Scaling Considerations

### Multiple Sensors

To support multiple sensors:

1. Update device configurations in Sensor Simulator
2. Use different `sensor_id` for each device
3. Monitor MQTT topics: `iot/+/fall_detection/accelerometer/+`

### Performance Optimization

```yaml
# In docker-compose.yml, add resource limits:
services:
  custom-lstm-detector:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
        reservations:
          memory: 512M
```

### High Availability

- Use external MQTT broker cluster
- Deploy InfluxDB in clustered mode
- Use load balancer for multiple Operator Interfaces

## 🔒 Security Hardening

### 1. Enable MQTT Authentication

```bash
# Create password file
mosquitto_passwd -c mosquitto/passwd operator

# Update mosquitto.conf
cat >> mosquitto/config/mosquitto.conf << EOF
allow_anonymous false
password_file /mosquitto/config/passwd
EOF

# Restart Mosquitto
docker-compose restart mosquitto
```

### 2. Enable TLS/SSL

Generate certificates and update configurations for encrypted connections.

### 3. Firewall Configuration

```bash
# Allow only necessary ports
sudo ufw enable
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 1883/tcp  # MQTT
sudo ufw allow 8086/tcp  # InfluxDB
sudo ufw allow 3000/tcp  # Grafana (optional)
```

### 4. Regular Updates

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Update Docker images
docker-compose pull
docker-compose up -d
```

## ✅ Post-Deployment Checklist

- [ ] All services running and healthy
- [ ] Passwords changed from defaults
- [ ] Backups configured
- [ ] Firewall rules in place
- [ ] Monitoring set up
- [ ] Documentation updated
- [ ] Team trained on Operator Interface
- [ ] Test scenarios validated
- [ ] Incident response plan ready
- [ ] Maintenance schedule established

## 📞 Support

For deployment issues:
1. Check this guide
2. Review Docker logs
3. Verify network connectivity
4. Check resource utilization
5. Consult main README.md

---

**Deployment Guide Version**: 1.0
**Last Updated**: 2025-11-02
**Compatible with**: Fall Detection System v2.0
