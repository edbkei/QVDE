# 🖥️ 3-Terminal Monitoring Cheatsheet

## Quick Setup (30 seconds)

### Terminal 1: MQTT Monitor
```bash
sudo docker exec -it mosquitto mosquitto_sub -h localhost -t 'iot/#' -v
```
**Expect:** MQTT messages every ~100ms with sensor data

### Terminal 2: Bridge Monitor  
```bash
sudo docker logs -f mqtt-influx-bridge 2>&1 | grep --line-buffered -E "DEBUG|sensor_data|Write successful"
```
**Expect:** "Has sensor_data: True" + "Write successful to sensors!" for each message

### Terminal 3: Model Predictions
```bash
sudo docker logs -f custom-lstm-detector 2>&1 | grep --line-buffered -E "Normal|FALL|Risk|Insufficient"
```
**Expect:** Predictions every 3 seconds: `[N] ✅ Normal | Risk: XX% | RL Action: ...`

---

## ✅ Success Checklist

| Terminal | What to See | What It Means |
|----------|-------------|---------------|
| **1** | Messages every 100ms | Simulator sending data |
| **2** | "Write successful" for each message | Data reaching InfluxDB |
| **3** | Predictions every 3 seconds | Model working |

---

## 🚨 Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| Terminal 1: Empty | Check simulator connection, verify JETSON_IP |
| Terminal 2: "Has sensor_data: False" | Use new clean simulator notebook |
| Terminal 3: Only "Insufficient data" | Keep simulator running continuously (not bursts) |
| Terminal 3: Database errors | Restart: `sudo docker-compose down && up -d` |

---

## 🎯 Healthy System Example

**Terminal 1:**
```
iot/house_001/fall_detection/accelerometer/sensor_accel_bedroom_001 {"metadata":{...},"sensor_data":{...}}
```

**Terminal 2:**
```
?? DEBUG: Received message
   Has sensor_data: True
? Write successful to sensors!
📊 Messages: 100 (Inference: 100, Errors: 0)
```

**Terminal 3:**
```
[5] ✅ Normal | Risk: 41.8% | RL Action: do_nothing | Time: 1.2ms | Model: v1.1.0
[6] 🚨 FALL DETECTED | Risk: 89.2% | RL Action: notify_high_priority | Time: 1.5ms
```

---

## 📊 Understanding Risk Levels

| Risk % | Meaning | RL Action |
|--------|---------|-----------|
| 0-40% | High confidence normal | do_nothing |
| 40-60% | Uncertain | notify_low_priority |
| 60-80% | Likely fall | notify_high_priority |
| 80-100% | High confidence fall | emergency_call |

---

## 🔧 Quick Commands

```bash
# Check all containers
sudo docker-compose ps

# Restart everything
sudo docker-compose down && sudo docker-compose up -d

# Check recent data in InfluxDB
sudo docker exec influxdb influx query '
from(bucket: "sensors") |> range(start: -1m) 
|> filter(fn: (r) => r["_measurement"] == "fall_detection_accelerometer") 
|> count()'

# View model logs (all output)
sudo docker logs --tail=50 custom-lstm-detector
```

---

## 💡 Remember

- Model needs **continuous data flow** (not bursts)
- First 10 seconds shows "Insufficient data" (normal)
- Predictions appear every **3 seconds** (model's cycle time)
- Simulator sends at **10 Hz** (100ms intervals)

---

**Full Guide:** monitoring_guide.md
