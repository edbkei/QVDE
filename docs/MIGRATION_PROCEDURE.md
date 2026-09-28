# Step-by-Step Migration Procedure
## From Single Jupyter Notebook to Separated Operator Interface

**Target System**: Existing Fall Detection System (Already Deployed)
**Migration Type**: Frontend Only - Backend Unchanged
**Estimated Time**: 30-45 minutes
**Downtime Required**: None (backend continues running)

---

## ⚠️ Pre-Migration Checklist

Before starting, verify your current system:

```bash
# On Jetson - Verify all services are running
docker-compose ps

# Expected output - all services should show "Up"
# - influxdb
# - grafana
# - mosquitto
# - mqtt-influx-bridge
# - custom-lstm-detector
```

**If all services are UP**, proceed with migration. ✅

**If any service is DOWN**, fix it first before migrating. ❌

---

## 📋 Migration Overview

### What Changes:
- ❌ OLD: `fall_detection_control_center.ipynb` (replaced)
- ✅ NEW: `operator_interface.py` (web application)
- ✅ NEW: `sensor_simulator.ipynb` (simplified notebook)

### What Stays the Same:
- ✅ All Docker containers on Jetson
- ✅ InfluxDB data and configuration
- ✅ MQTT topics and structure
- ✅ Model files and storage
- ✅ Grafana dashboards

**Result**: Backend keeps running - you just change how you control it!

---

## 🚀 Step-by-Step Migration

### STEP 1: Backup Current System (5 minutes)

**On your desktop/laptop:**

```bash
# 1.1 Create backup directory
mkdir -p ~/fall-detection-backup
cd ~/fall-detection-backup

# 1.2 Backup old notebook
cp /path/to/fall_detection_control_center.ipynb ./fall_detection_control_center.ipynb.backup

# 1.3 Save current configuration
# Note your current Jetson IP (you'll need it)
echo "JETSON_IP=192.168.15.116" > config_backup.txt

# 1.4 Document current state
echo "Backup created on $(date)" >> backup_info.txt
echo "Old notebook backed up" >> backup_info.txt
```

**On Jetson (optional but recommended):**

```bash
# 1.5 Backup models directory
cd ~/fall-detection-system  # Or your project directory
tar -czf models_backup_$(date +%Y%m%d).tar.gz models/

# 1.6 Note current system state
docker-compose ps > system_state_backup.txt
docker-compose logs --tail=50 > logs_backup.txt
```

**✅ Checkpoint**: You have backups of notebook and models

---

### STEP 2: Install New Tools on Desktop/Laptop (10 minutes)

**2.1 Create new project directory:**

```bash
# On your desktop/laptop
mkdir -p ~/fall-detection-operator
cd ~/fall-detection-operator
```

**2.2 Download/copy the new files** (from the outputs I provided):

Copy these files to `~/fall-detection-operator/`:
- `operator_interface.py`
- `sensor_simulator.ipynb`
- `requirements-operator.txt`

```bash
# If you have the files locally, copy them:
cp /path/to/operator_interface.py .
cp /path/to/sensor_simulator.ipynb .
cp /path/to/requirements-operator.txt .

# Verify files are present
ls -l
# Should show all 3 files
```

**2.3 Create Python virtual environment (recommended):**

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # On Linux/Mac
# OR
venv\Scripts\activate  # On Windows

# Your prompt should change to show (venv)
```

**2.4 Install dependencies:**

```bash
# Install Operator Interface dependencies
pip install -r requirements-operator.txt

# Install Jupyter and sensor simulator dependencies
pip install jupyter numpy pandas matplotlib paho-mqtt

# Verify installations
python -c "import streamlit; print('✅ Streamlit:', streamlit.__version__)"
python -c "import paho.mqtt.client; print('✅ MQTT:', paho.mqtt.client.__version__)"
jupyter --version
```

**Expected output:**
```
✅ Streamlit: 1.28.0 (or higher)
✅ MQTT: 1.6.1
Selected Jupyter core packages...
```

**✅ Checkpoint**: All new tools installed successfully

---

### STEP 3: Configure New Tools (5 minutes)

**3.1 Get your Jetson IP address:**

```bash
# On Jetson
hostname -I
# Example output: 192.168.15.116 172.17.0.1
# Use the first IP (192.168.15.116)
```

**3.2 Configure Sensor Simulator:**

```bash
# On desktop/laptop
jupyter notebook
```

This opens Jupyter in your browser. Then:

1. Navigate to `sensor_simulator.ipynb`
2. Open the notebook
3. Find the **Configuration** cell (near the top)
4. Update the `JETSON_IP` value:

```python
# Change this line:
JETSON_IP = "192.168.15.116"  # ⚠️ PUT YOUR JETSON IP HERE
```

5. **Save the notebook** (Ctrl+S or Cmd+S)
6. **Close Jupyter** for now (we'll test it later)

**3.3 Configure Operator Interface (optional):**

The Operator Interface lets you enter the IP in the web UI, but you can set a default:

```bash
# Optional: Edit operator_interface.py
nano operator_interface.py  # or use your favorite editor

# Find this line (around line 18):
DEFAULT_JETSON_IP = "192.168.15.116"

# Change to your Jetson IP
DEFAULT_JETSON_IP = "192.168.15.116"  # Your actual IP

# Save and exit
```

**✅ Checkpoint**: New tools configured with your Jetson IP

---

### STEP 4: Test Operator Interface (5 minutes)

**4.1 Start the Operator Interface:**

```bash
# On desktop/laptop
cd ~/fall-detection-operator
source venv/bin/activate  # If not already activated

# Start the interface
streamlit run operator_interface.py
```

**Expected output:**
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.x:8501
```

Browser should open automatically showing the Operator Interface.

**4.2 Connect to Jetson:**

In the Operator Interface:

1. **Sidebar** → Find "Connection" section
2. Enter your **Jetson IP** (e.g., 192.168.15.116)
3. Verify **MQTT Port** is 1883
4. Click **"🔗 Connect"** button
5. Wait 2-3 seconds

**Expected result:**
- Green message: "✅ Connected to Jetson"
- Sidebar shows: **"🟢 Connected"**

**If connection fails**, see troubleshooting at end of this document.

**4.3 Test system status:**

1. In sidebar, click **"📊 Get Status"** button
2. Go to **"📊 Dashboard"** tab
3. Check if you see system information

**Expected result:**
- Status information appears
- Model version shown (if model loaded)
- No errors

**✅ Checkpoint**: Operator Interface connected and working

---

### STEP 5: Test Sensor Simulator (5 minutes)

**5.1 Start Jupyter:**

```bash
# Open new terminal (keep Operator Interface running)
cd ~/fall-detection-operator
source venv/bin/activate  # If using venv

jupyter notebook
```

**5.2 Open and test the simulator:**

1. Click on `sensor_simulator.ipynb` to open it
2. **Run cells in order** (Shift+Enter for each cell):
   - Configuration cell
   - Import libraries cell
   - SensorSimulator class cell
   - Connect to MQTT cell

**Expected output after connecting:**
```
✅ Connected to MQTT Broker at 192.168.15.116:1883
✅ Sensor simulator ready
```

**5.3 Send a test reading:**

Navigate to **"Mode 3: Manual Single Reading"** section and run:

```python
# Send a single normal reading for inference
simulator.send_reading(is_fall=False, operational_mode="inference")
print("✅ Sent normal inference reading")
```

**Expected output:**
```
✅ Sent normal inference reading
```

**5.4 Verify data flow:**

Check in **Operator Interface** → **Dashboard** tab:
- Should see message count increase
- If in inference mode with model loaded, may see prediction

**Alternative verification** - Check MQTT Bridge logs:

```bash
# On Jetson
docker logs --tail=20 mqtt-influx-bridge

# Should see: "✅ Normal | Messages: X"
```

**✅ Checkpoint**: Sensor Simulator connected and sending data

---

### STEP 6: Verify End-to-End Workflow (5-10 minutes)

Now test a complete workflow to ensure everything works.

**6.1 Quick Training Test (Optional - if you want to verify training):**

In **Sensor Simulator**:

```python
# Generate a small batch of training data
generate_training_data(
    num_samples=50,      # Small batch for testing
    fall_percentage=10.0,
    labeled_by="migration_test"
)
```

Wait 30 seconds, then in **Operator Interface**:

1. Go to **Training Control** tab
2. Set epochs to 10 (quick test)
3. Click **"🚀 Start Training"**
4. Check Jetson logs:

```bash
# On Jetson
docker logs -f custom-lstm-detector

# Watch for training progress
# Press Ctrl+C when you see training complete
```

**6.2 Inference Test:**

In **Operator Interface**:

1. Go to **Inference Control** tab
2. Click **"⭐ Load Best Model"**
3. Click **"🔮 Switch to Inference Mode"**

In **Sensor Simulator**:

```python
# Generate inference data
generate_inference_data(
    duration=30,           # 30 seconds test
    fall_probability=0.1   # 10% fall rate for testing
)
```

In **Operator Interface**:

1. Go to **Live Monitoring** tab
2. Click **"🔄 Refresh Data"**
3. Watch predictions appear in real-time

**Expected result:**
- Predictions appear in the monitoring tab
- Graph updates with prediction data
- Statistics show prediction counts

**✅ Checkpoint**: End-to-end workflow verified

---

### STEP 7: Decommission Old Notebook (2 minutes)

**7.1 Archive the old notebook:**

```bash
# On desktop/laptop
cd ~/fall-detection-backup

# Rename with date
mv fall_detection_control_center.ipynb.backup \
   fall_detection_control_center.ipynb.$(date +%Y%m%d).deprecated

# Create README
cat > README.txt << 'EOF'
This directory contains the old Jupyter notebook control center.

DEPRECATED: This notebook is no longer used.
REPLACED BY: operator_interface.py + sensor_simulator.ipynb

Migration Date: $(date)

DO NOT USE the old notebook - use the new Operator Interface instead!
EOF

echo "✅ Old notebook archived"
```

**7.2 Update your workflow documentation:**

Create a reminder file:

```bash
# In your project directory
cat > ~/fall-detection-operator/USAGE.txt << 'EOF'
=== FALL DETECTION SYSTEM - USAGE ===

NEW WORKFLOW:

1. Start Operator Interface:
   $ cd ~/fall-detection-operator
   $ source venv/bin/activate
   $ streamlit run operator_interface.py

2. Start Sensor Simulator (in separate terminal):
   $ cd ~/fall-detection-operator
   $ jupyter notebook
   # Open sensor_simulator.ipynb

3. Use Operator Interface to control system
   - Training Control tab for model training
   - Inference Control tab for predictions
   - Live Monitoring tab to watch results

OLD SYSTEM (DEPRECATED):
   - fall_detection_control_center.ipynb is no longer used
   - Archived in ~/fall-detection-backup/
EOF

cat ~/fall-detection-operator/USAGE.txt
```

**✅ Checkpoint**: Old system archived, new system documented

---

## ✅ Migration Complete Checklist

Verify all items:

### Backend (Jetson):
- [ ] All Docker containers still running
- [ ] InfluxDB accessible at http://JETSON_IP:8086
- [ ] MQTT broker accepting connections
- [ ] No error messages in logs
- [ ] Models directory intact

### New Tools (Desktop):
- [ ] Operator Interface installed
- [ ] Operator Interface connects to Jetson
- [ ] Sensor Simulator installed
- [ ] Sensor Simulator can send data
- [ ] Can see data in InfluxDB/logs

### End-to-End:
- [ ] Can generate training data
- [ ] Can trigger training (tested or verified)
- [ ] Can load models
- [ ] Can generate inference data
- [ ] Can see predictions in Live Monitoring
- [ ] All workflows documented

### Cleanup:
- [ ] Old notebook archived
- [ ] Backups created
- [ ] New usage documented
- [ ] Team informed of changes

---

## 🎓 Post-Migration Training

Now that migration is complete, train your team:

### For Operators:
1. Read **OPERATOR_GUIDE.md** (45 minutes)
2. Practice training workflow (1 hour)
3. Practice inference workflow (30 minutes)
4. Review troubleshooting section (30 minutes)

### Quick Reference Card:

```
DAILY WORKFLOW:

Morning (Start of Day):
1. Start Operator Interface: streamlit run operator_interface.py
2. Start Jupyter: jupyter notebook → open sensor_simulator.ipynb
3. Connect Operator Interface to Jetson
4. Verify system status

Operations:
- Generate data: Use Sensor Simulator
- Control system: Use Operator Interface
- Monitor: Use Live Monitoring tab

End of Day:
- Can leave Operator Interface running or close
- Jetson services run 24/7 (no action needed)
```

---

## 🐛 Troubleshooting

### Issue: Operator Interface won't connect

**Symptoms**: Red "🔴 Disconnected" status

**Solutions**:

```bash
# 1. Verify Jetson IP is correct
ping <JETSON_IP>

# 2. Check MQTT broker on Jetson
docker ps | grep mosquitto

# 3. Check if port is accessible
telnet <JETSON_IP> 1883
# Or
nc -zv <JETSON_IP> 1883

# 4. Check firewall on Jetson
sudo ufw status

# 5. Test MQTT locally on Jetson
docker exec -it mosquitto mosquitto_sub -h localhost -t test -v
```

### Issue: Sensor Simulator won't connect

**Symptoms**: "Connection failed" error

**Solutions**:

```bash
# 1. Verify JETSON_IP in notebook is correct
# Open sensor_simulator.ipynb and check configuration cell

# 2. Restart MQTT broker
docker-compose restart mosquitto

# 3. Check mosquitto logs
docker logs mosquitto

# 4. Test with mosquitto_pub/sub
mosquitto_sub -h <JETSON_IP> -t test -v
```

### Issue: No predictions appearing

**Symptoms**: Data sent but no predictions in monitoring

**Solutions**:

```bash
# 1. Verify model is loaded
# In Operator Interface → Dashboard → Check "Model Version"

# 2. Verify system in inference mode
# In Operator Interface → Dashboard → Check "System Mode"

# 3. Check LSTM service logs
docker logs -f custom-lstm-detector

# 4. Manually load model
# In Operator Interface → Inference Control → Click "Load Best Model"

# 5. Manually switch mode
# In Operator Interface → Inference Control → Click "Switch to Inference"
```

### Issue: Old notebook still being used

**Solution**: Educate team!

- Show team the new Operator Interface
- Demonstrate easier workflows
- Archive old notebook clearly
- Update documentation everywhere

---

## 📊 Comparison: Old vs New

### Training a Model:

**OLD WAY** (fall_detection_control_center.ipynb):
```python
# Run cell 1: Configure
# Run cell 2: Import libraries
# Run cell 3: Create controller
# Run cell 4: Create training generator
# Run cell 5: Generate training data
# Run cell 6: Start training
# Run cell 7: Monitor (manually)
# Run cell 8: Validate
# Run cell 9: Load model
```
**Time**: 15 minutes, 9 code cells to run

**NEW WAY** (Operator Interface + Sensor Simulator):
```
Sensor Simulator: Run "generate_training_data()" → 1 cell
Wait 30 seconds
Operator Interface: Click "Start Training" → 1 click
Wait for completion
Operator Interface: Click "Validate" → 1 click
Operator Interface: Click "Load Best Model" → 1 click
```
**Time**: 10 minutes, 1 code cell + 3 clicks

### Running Inference:

**OLD WAY**:
```python
# Run cell 1: Load model
# Run cell 2: Switch to inference
# Run cell 3: Create simulator
# Run cell 4: Run simulation
# Run cell 5: Monitor (manually refresh)
```
**Time**: Multiple cells, manual monitoring

**NEW WAY**:
```
Operator Interface: Click "Load Best Model" → 1 click
Operator Interface: Click "Switch to Inference" → 1 click
Sensor Simulator: Run "generate_inference_data()" → 1 cell
Operator Interface: Watch Live Monitoring → automatic updates
```
**Time**: Faster, automatic real-time monitoring

---

## 🎉 Success!

Your migration is complete! You now have:

✅ **Separated concerns**: Data generation ≠ System control
✅ **Professional interface**: Point-and-click operations
✅ **Easier workflows**: Fewer steps, clearer process
✅ **Better monitoring**: Real-time dashboard
✅ **Unchanged backend**: All your data and models preserved

### Next Steps:

1. **Today**: Familiarize team with new tools
2. **This Week**: Use new interface for all operations
3. **Monitor**: Verify everything working smoothly
4. **Feedback**: Note any issues or improvements needed

### Getting Help:

- **Architecture**: README.md
- **Daily Operations**: OPERATOR_GUIDE.md
- **Deployment Issues**: DEPLOYMENT_GUIDE.md
- **What Changed**: MIGRATION_SUMMARY.md

---

## 📝 Migration Log Template

Keep track of your migration:

```bash
cat > ~/fall-detection-operator/MIGRATION_LOG.txt << 'EOF'
=== MIGRATION LOG ===

Migration Date: [DATE]
Performed By: [YOUR NAME]
Jetson IP: [IP ADDRESS]

Pre-Migration:
[ ] Verified all services running
[ ] Created backups
[ ] Noted current configuration

Installation:
[ ] Created project directory
[ ] Downloaded new files
[ ] Installed dependencies
[ ] Configured tools

Testing:
[ ] Operator Interface connected
[ ] Sensor Simulator connected
[ ] End-to-end workflow verified
[ ] Training tested (optional)
[ ] Inference tested

Cleanup:
[ ] Old notebook archived
[ ] Documentation updated
[ ] Team trained

Post-Migration Status:
System Status: [WORKING / ISSUES]
Notes: [ANY OBSERVATIONS]

Issues Encountered:
[DESCRIBE ANY ISSUES]

Resolutions:
[HOW ISSUES WERE FIXED]
EOF
```

---

**Migration Procedure Version**: 1.0
**Date**: 2025-11-02
**Status**: Ready for Use

**Questions?** Refer to the troubleshooting section or main documentation files!
