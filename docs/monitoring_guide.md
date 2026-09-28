# 🖥️ IoT Fall Detection System - 3-Terminal Monitoring Guide

## Overview

This guide shows you how to monitor your complete IoT fall detection pipeline in real-time using 3 terminals on your Jetson.

**Data Flow:**
```
Windows Simulator → MQTT Broker → Bridge → InfluxDB → LSTM Model → Predictions
     Terminal 1         Terminal 2                      Terminal 3
```

---

## 🚀 Quick Start

### Prerequisites
- SSH into your Jetson Orin Nano (or use 3 local terminal windows)
- Docker containers running: `sudo docker-compose ps` (all should show "Up")
- Sensor simulator ready on Windows laptop

---

## 📺 Terminal 1: MQTT Message Monitor

### Purpose
Monitor all MQTT messages flowing through the broker in real-time.

### Command
```bash
sudo docker exec -it mosquitto mosquitto_sub -h localhost -t 'iot/#' -v
```

### What This Does
- Subscribes to ALL topics under `iot/`
- Shows topic name and message payload
- Real-time display of every message

### ✅ What You Should See (Success)

When sensor simulator sends data:
```
iot/house_001/fall_detection/accelerometer/sensor_accel_bedroom_001 {"metadata":{"house_id":"house_001","sensor_id":"sensor_accel_bedroom_001","application":"fall_detection","timestamp":"2025-11-24T17:35:16.555","operational_mode":"inference","data_purpose":"production"},"device_info":{"sensor_type":"accelerometer","location":"master_bedroom","power_level":85.2,"firmware_version":"2.0.0","sensor_status":"active"},"sensor_data":{"accel_x":0.1234,"accel_y":-0.0567,"accel_z":9.8123,"gyro_roll":2.3456,"gyro_pitch":-1.2345,"gyro_yaw":0.5678}}

iot/house_001/fall_detection/accelerometer/sensor_accel_bedroom_001 {"metadata":{...},"device_info":{...},"sensor_data":{...}}
```

**Good signs:**
- ✅ Messages appear continuously (every 100ms if 10 Hz)
- ✅ Topic includes `house_001/fall_detection/accelerometer/sensor_accel_bedroom_001`
- ✅ Payload contains `metadata`, `device_info`, `sensor_data`
- ✅ `accel_z` values around 9.8 (gravity) for normal readings
- ✅ `accel_z` values far from 9.8 for fall readings

### ❌ What Indicates Problems

**No messages appear:**
```
(empty - no output)
```
→ **Problem**: Simulator not sending data or wrong MQTT broker address
→ **Fix**: Check simulator connection, verify JETSON_IP in notebook

**Wrong topic structure:**
```
iot/sensors/data {"value": 123}
```
→ **Problem**: Old simulator or different topic structure
→ **Fix**: Use the new clean simulator notebook

### 🔍 How to Interpret

| Observation | Meaning |
|-------------|---------|
| Steady stream of messages | ✅ Simulator working |
| Messages every ~100ms | ✅ Correct 10 Hz rate |
| Topic ends with sensor_accel_bedroom_001 | ✅ Correct device |
| `"operational_mode":"inference"` | ✅ Inference data |
| `"operational_mode":"training"` | ✅ Training data |
| Large accel values (>10) | 🚨 Fall event |

### 💡 Tips
- Press Ctrl+C to stop monitoring
- Redirect to file: `... > mqtt_log.txt` to save messages
- Filter falls only: `... | grep -i "fall"`

---

## 🌉 Terminal 2: Bridge Processing Monitor

### Purpose
Verify the MQTT-InfluxDB bridge is receiving and storing data correctly.

### Command
```bash
sudo docker logs -f mqtt-influx-bridge 2>&1 | grep --line-buffered -E "DEBUG|sensor_data|Write successful|Error|error"
```

### What This Does
- Shows bridge logs in real-time
- Filters for important events (data processing, writes, errors)
- Displays whether data is successfully written to InfluxDB

### ✅ What You Should See (Success)

When processing sensor data:
```
?? DEBUG: Received message
   Topic: iot/house_001/fall_detection/accelerometer/sensor_accel_bedroom_001
   Payload length: 514 bytes
   First 200 chars: b'{"metadata": {"house_id": "house_001", "sensor_id": "sensor_accel_bedroom_001", "application": "fall_detection", "timestamp": "2025-11-24T17:09:35.796415", "operational_mode": "inference", "data_purpo'
   Parsed JSON keys: ['metadata', 'device_info', 'sensor_data']
   Has metadata: True
   Has device_info: True
   Has sensor_data: True

?? Attempting write to bucket: sensors
   Point (first 300 chars): fall_detection_accelerometer,application=fall_detection,house_id=house_001,location=master_bedroom,operational_mode=inference,sensor_id=sensor_accel_bedroom_001,sensor_status=active,sensor_type=accelerometer accel_x=0.2787,accel_y=-0.1005,accel_z=9.6743,gyro_pitch=3.0204,gyro_roll=2.5049,gyro_yaw=0.

? Write successful to sensors!

📊 Messages: 20 (Training: 0, Inference: 20, Errors: 0)
```

**Good signs:**
- ✅ "Has sensor_data: True" for every message
- ✅ "Write successful to sensors!" after each message
- ✅ Message counter increases: "Messages: 20, 21, 22..."
- ✅ "Errors: 0" (no errors)
- ✅ Correct bucket: "sensors" (inference) or "training_data" (training)

### ❌ What Indicates Problems

**Wrong payload structure:**
```
Has metadata: True
Has device_info: True
Has sensor_data: False    ← Problem!
```
→ **Problem**: Bridge doesn't recognize payload format
→ **Fix**: Use new clean simulator notebook

**Write failures:**
```
❌ Error writing to InfluxDB: Connection refused
```
→ **Problem**: InfluxDB not reachable
→ **Fix**: Check `sudo docker ps | grep influxdb`, restart if needed

**No messages at all:**
```
(empty - no output)
```
→ **Problem**: Bridge not receiving MQTT messages or not subscribed correctly
→ **Fix**: Check bridge logs: `sudo docker logs mqtt-influx-bridge | tail -30`

### 🔍 How to Interpret

| Observation | Meaning |
|-------------|---------|
| "Has sensor_data: True" | ✅ Correct payload format |
| "Write successful" appears | ✅ Data stored in InfluxDB |
| Messages counter increasing | ✅ Bridge processing data |
| "Errors: 0" | ✅ No processing errors |
| "Training: X" increasing | ✅ Training data being received |
| "Inference: X" increasing | ✅ Inference data being received |

### 💡 Tips
- Each message should show both "DEBUG: Received" and "Write successful"
- Time between messages should match simulator rate (~100ms for 10 Hz)
- If "Write successful" is missing, InfluxDB has issues

---

## 🧠 Terminal 3: Model Predictions Monitor

### Purpose
Watch the LSTM model make real-time fall detection predictions.

### Command
```bash
sudo docker logs -f custom-lstm-detector 2>&1 | grep --line-buffered -E "Normal|FALL|Risk|Insufficient|predictions"
```

### What This Does
- Shows model predictions in real-time
- Displays risk percentage and RL agent actions
- Shows when insufficient data is available

### ✅ What You Should See (Success)

When model has enough data:
```
[796] ✅ Normal | Risk: 46.9% | RL Action: do_nothing | Time: 1.3ms | Model: v1.1.0
[797] ✅ Normal | Risk: 45.2% | RL Action: do_nothing | Time: 1.2ms | Model: v1.1.0
[798] ✅ Normal | Risk: 48.1% | RL Action: do_nothing | Time: 1.4ms | Model: v1.1.0
[799] 🚨 FALL DETECTED | Risk: 87.4% | RL Action: notify_high_priority | Time: 1.5ms | Model: v1.1.0
[800] ✅ Normal | Risk: 43.7% | RL Action: do_nothing | Time: 1.3ms | Model: v1.1.0
```

**Good signs:**
- ✅ Predictions appear every 3 seconds (model's inference interval)
- ✅ Counter [N] increases continuously
- ✅ Normal readings show "✅ Normal"
- ✅ Fall events show "🚨 FALL DETECTED"
- ✅ Risk percentage provided (0-100%)
- ✅ RL Action shows what system recommends
- ✅ Inference time is fast (<5ms typically)

### ❌ What Indicates Problems

**Insufficient data warning:**
```
⚠️  Insufficient data (0 points)
⚠️  Insufficient data (3 points)
⚠️  Insufficient data (7 points)
```
→ **Problem**: Model needs more data (typically 10+ samples in window)
→ **Fix**: Keep simulator running continuously, not just bursts

**No predictions at all:**
```
(empty - no output except startup messages)
```
→ **Problem**: No recent data in InfluxDB, or model can't query InfluxDB
→ **Fix**: Check if data is too old, keep sending continuous data

**Database connection errors:**
```
❌ Error fetching production data: Failed to resolve 'influxdb'
```
→ **Problem**: Docker networking broken
→ **Fix**: `sudo docker-compose down && sudo docker-compose up -d`

### 🔍 How to Interpret

| Observation | Meaning |
|-------------|---------|
| ✅ Normal | No fall detected, normal activity |
| 🚨 FALL DETECTED | Fall event detected |
| Risk: 0-40% | High confidence in normal activity |
| Risk: 40-60% | Uncertain/moderate risk |
| Risk: 60-100% | High risk of fall |
| do_nothing | RL agent: no action needed |
| notify_low_priority | RL agent: slight concern |
| notify_high_priority | RL agent: should alert caregiver |
| emergency_call | RL agent: critical fall, call 911 |
| Time: X ms | Inference speed (should be <5ms) |
| [N] counter | Total predictions made this session |

### 💡 Tips
- Predictions every 3 seconds is normal (model's inference interval)
- "Insufficient data" for first 10 seconds after starting simulator is normal
- If you see only "Insufficient data", simulator probably stopped

---

## 🎯 Complete Monitoring Workflow

### Step 1: Start All 3 Terminals

**Terminal 1:**
```bash
sudo docker exec -it mosquitto mosquitto_sub -h localhost -t 'iot/#' -v
```

**Terminal 2:**
```bash
sudo docker logs -f mqtt-influx-bridge 2>&1 | grep --line-buffered -E "DEBUG|sensor_data|Write successful|Error"
```

**Terminal 3:**
```bash
sudo docker logs -f custom-lstm-detector 2>&1 | grep --line-buffered -E "Normal|FALL|Risk|Insufficient"
```

### Step 2: Start Simulator on Windows

In your Jupyter notebook:
```python
# Run continuous simulation
generate_inference_data(duration=120, fall_probability=0.10)
```

### Step 3: Verify Each Terminal

#### Terminal 1 Checklist:
- [ ] Messages appear immediately when simulator starts
- [ ] Messages appear every ~100ms (10 Hz rate)
- [ ] Topic structure is correct: `iot/house_001/fall_detection/...`
- [ ] Payload contains sensor_data section

#### Terminal 2 Checklist:
- [ ] "DEBUG: Received message" appears for each message
- [ ] "Has sensor_data: True" for each message
- [ ] "Write successful to sensors!" after each message
- [ ] Message counter increases
- [ ] "Errors: 0" stays at zero

#### Terminal 3 Checklist:
- [ ] Shows "Insufficient data" for first 5-10 seconds (normal)
- [ ] Then predictions start appearing
- [ ] Predictions appear every ~3 seconds
- [ ] Counter [N] increases
- [ ] Sees both Normal and FALL events (with 10% fall probability)

---

## 📊 Example of Healthy System

### Terminal 1:
```
iot/house_001/fall_detection/accelerometer/sensor_accel_bedroom_001 {"metadata":{...},"sensor_data":{...}}
iot/house_001/fall_detection/accelerometer/sensor_accel_bedroom_001 {"metadata":{...},"sensor_data":{...}}
iot/house_001/fall_detection/accelerometer/sensor_accel_bedroom_001 {"metadata":{...},"sensor_data":{...}}
(messages continue at 10 Hz)
```

### Terminal 2:
```
?? DEBUG: Received message
   Has sensor_data: True
? Write successful to sensors!
?? DEBUG: Received message
   Has sensor_data: True
? Write successful to sensors!
📊 Messages: 100 (Training: 0, Inference: 100, Errors: 0)
```

### Terminal 3:
```
⚠️  Insufficient data (5 points)
⚠️  Insufficient data (8 points)
[1] ✅ Normal | Risk: 42.3% | RL Action: do_nothing | Time: 1.2ms | Model: v1.1.0
[2] ✅ Normal | Risk: 38.7% | RL Action: do_nothing | Time: 1.3ms | Model: v1.1.0
[3] ✅ Normal | Risk: 44.1% | RL Action: do_nothing | Time: 1.1ms | Model: v1.1.0
[4] 🚨 FALL DETECTED | Risk: 89.2% | RL Action: notify_high_priority | Time: 1.5ms | Model: v1.1.0
[5] ✅ Normal | Risk: 41.8% | RL Action: do_nothing | Time: 1.2ms | Model: v1.1.0
```

**This is PERFECT!** All 3 terminals show data flowing correctly.

---

## 🚨 Common Issues and Solutions

### Issue 1: No Messages in Terminal 1

**Symptom:** Terminal 1 shows nothing
**Cause:** Simulator not sending or wrong IP
**Solution:**
1. Verify JETSON_IP in notebook configuration
2. Test network: `ping 192.168.15.116` from Windows
3. Check simulator connection status in notebook
4. Re-run connection cell in notebook

### Issue 2: Terminal 2 Shows "Has sensor_data: False"

**Symptom:** Bridge receives messages but doesn't recognize them
**Cause:** Old simulator with wrong payload format
**Solution:**
1. Use the new clean simulator notebook I created
2. Re-run all cells from the beginning
3. Verify payload structure in Terminal 1

### Issue 3: Terminal 3 Only Shows "Insufficient data"

**Symptom:** Model never makes predictions
**Cause:** Data is old or simulator stopped
**Solution:**
1. Don't send data in bursts - use continuous mode
2. Keep simulator running while monitoring
3. Model needs data from last 10 seconds

### Issue 4: Terminal 3 Shows Database Errors

**Symptom:** "Failed to resolve 'influxdb'"
**Cause:** Docker networking broken
**Solution:**
```bash
cd /mnt/nvme/iot-stack
sudo docker-compose down
sudo docker-compose up -d
# Wait 30 seconds, then re-run monitoring
```

### Issue 5: Bridge Not Writing to InfluxDB

**Symptom:** "Write failed" or connection errors
**Cause:** InfluxDB not reachable
**Solution:**
```bash
# Check InfluxDB is running
sudo docker ps | grep influxdb

# Check bridge can reach InfluxDB
sudo docker exec mqtt-influx-bridge ping -c 3 influxdb

# If fails, restart stack
sudo docker-compose restart
```

---

## 📝 Quick Reference Commands

### Check System Status
```bash
# All containers running?
sudo docker-compose ps

# Restart everything
sudo docker-compose down && sudo docker-compose up -d

# Check individual container
sudo docker logs --tail=30 <container-name>
```

### Verify Data in InfluxDB
```bash
# Count recent sensor data
sudo docker exec influxdb influx query '
from(bucket: "sensors")
  |> range(start: -1m)
  |> filter(fn: (r) => r["_measurement"] == "fall_detection_accelerometer")
  |> count()
'

# See latest data points
sudo docker exec influxdb influx query '
from(bucket: "sensors")
  |> range(start: -1m)
  |> filter(fn: (r) => r["_measurement"] == "fall_detection_accelerometer")
  |> limit(n: 5)
'
```

### Test MQTT Publishing from Jetson
```bash
# Send test message
sudo docker exec mosquitto mosquitto_pub \
  -h localhost \
  -t 'iot/test' \
  -m '{"test": "message"}'

# Should appear in Terminal 1 if monitoring is working
```

---

## ✅ Success Criteria Checklist

Your system is working correctly when:

- [ ] Terminal 1: Messages flow continuously at ~10 Hz
- [ ] Terminal 2: Every message shows "Write successful"
- [ ] Terminal 3: Predictions appear every 3 seconds
- [ ] No error messages in any terminal
- [ ] Fall events are detected when simulator sends them
- [ ] Model shows reasonable risk percentages (not always 0% or 100%)
- [ ] Message counters increase steadily
- [ ] All Docker containers show "Up" status

---

## 🎓 Understanding the Data Flow

```
1. Simulator (Windows)
   └─> Creates sensor reading (JSON)
       └─> Sends via MQTT
           │
           ├─> Terminal 1: You see the raw MQTT message
           │
           ↓
2. Mosquitto Broker (Jetson)
   └─> Routes message to subscribers
       │
       ↓
3. MQTT-InfluxDB Bridge
   └─> Receives message
       └─> Parses JSON
           └─> Writes to InfluxDB
               │
               ├─> Terminal 2: You see "Write successful"
               │
               ↓
4. InfluxDB
   └─> Stores time-series data
       │
       ↓
5. LSTM Model
   └─> Queries last 10 seconds of data
       └─> Runs inference
           └─> Makes prediction
               │
               └─> Terminal 3: You see the prediction
```

---

## 💡 Pro Tips

1. **Use tmux or screen** to keep all 3 terminals visible at once
2. **Save logs** to files for later analysis:
   ```bash
   sudo docker logs -f custom-lstm-detector > model_predictions.log
   ```
3. **Monitor specific houses**: Add filter to Terminal 1:
   ```bash
   mosquitto_sub ... -t 'iot/house_001/#'
   ```
4. **Watch for falls only** in Terminal 3:
   ```bash
   ... | grep --color=always "FALL"
   ```
5. **Create a monitoring dashboard**: Open Grafana at `http://192.168.15.116:3000`

---

## 📞 Still Having Issues?

If after following this guide your system still isn't working:

1. Check Docker containers: `sudo docker-compose ps` (all should be "Up")
2. Check Docker logs for each service: `sudo docker logs <container-name>`
3. Restart the entire stack: `sudo docker-compose down && sudo docker-compose up -d`
4. Verify network connectivity between Windows and Jetson
5. Check firewall settings on Jetson

---

**Last Updated:** November 24, 2025
**System Version:** IoT Fall Detection System v2.0
