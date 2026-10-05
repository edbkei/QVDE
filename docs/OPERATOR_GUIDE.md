# Operator User Guide

## 👋 Welcome

This guide will help you operate the Fall Detection System using the **Operator Interface** and **Sensor Simulator**. No programming knowledge is required for daily operations.

## 📚 Table of Contents

1. [Getting Started](#getting-started)
2. [Operator Interface](#operator-interface)
3. [Sensor Simulator](#sensor-simulator)
4. [Training Workflow](#training-workflow)
5. [Inference Workflow](#inference-workflow)
6. [Monitoring & Troubleshooting](#monitoring--troubleshooting)
7. [Best Practices](#best-practices)
8. [FAQ](#faq)

## 🚀 Getting Started

### What You Need

1. **Jetson Device**: Should be running and connected to network
2. **Your Computer**: Desktop or laptop with:
   - Web browser (Chrome, Firefox, Edge, Safari)
   - Jupyter Notebook installed
   - Python installed
3. **Network Connection**: Both computers on same network

### System Overview

The system has two main tools you'll use:

1. **🎯 Operator Interface** (Web Application)
   - Control training and inference
   - Monitor predictions
   - Manage models
   - View system status

2. **📡 Sensor Simulator** (Jupyter Notebook)
   - Generate simulated sensor data
   - Create labeled training data
   - Simulate real-time sensor readings

## 🎯 Operator Interface

### Starting the Operator Interface

1. Open terminal/command prompt
2. Navigate to project directory:
   ```bash
   cd ~/QVDE/desktop
   ```
3. Run the interface:
   ```bash
   streamlit run operator_interface.py
   ```
4. Browser opens automatically to: `http://localhost:8501`

### Connecting to Jetson

1. **In the sidebar**, find "Connection" section
2. Enter **Jetson IP Address** (e.g., `192.168.15.116`)
3. Verify **MQTT Port** is `1883`
4. Click **"🔗 Connect"**
5. Wait for confirmation: **"🟢 Connected"**

### Interface Layout

#### Sidebar
- **Connection**: Connect/disconnect from Jetson
- **System Mode**: View current operational mode
- **Quick Actions**: Common operations

#### Main Tabs
1. **📊 Dashboard**: System overview and status
2. **🎓 Training Control**: Manage model training
3. **🔮 Inference Control**: Manage predictions
4. **📈 Live Monitoring**: Real-time predictions

### Dashboard Tab

Shows system overview:
- **Operational Mode**: Training/Inference
- **System Mode**: Current system state
- **Model Version**: Active model
- **Last Status**: Recent system message
- **Recent Predictions**: Latest fall detections

**Actions**:
- Click **"📊 Get Status"** to refresh system information

### Training Control Tab

Used to train new models.

**Before Training**:
1. Ensure you have generated labeled training data
2. Wait 30 seconds after data generation

**Starting Training**:
1. Navigate to **Training Control** tab
2. Configure parameters:
   - **Data Window**: Hours of data to use (default: 24)
   - **Training Epochs**: Number of training cycles (default: 100)
   - **Validation Split**: Percentage for validation (default: 20%)
3. Click **"🚀 Start Training"**
4. Monitor progress in Jetson logs

**During Training**:
- Training takes several minutes
- Do not send new data during training
- Do not disconnect

**After Training**:
1. Click **"✅ Validate Model"**
2. Review validation metrics
3. Load the new model

### Inference Control Tab

Used to run predictions on live data.

**Preparing for Inference**:
1. Navigate to **Inference Control** tab
2. Click **"⭐ Load Best Model"** (recommended)
   - Or **"🕐 Load Latest Model"**
   - Or enter specific version and click **"📦 Load Specific Version"**
3. Click **"🔮 Switch to Inference Mode"**
4. Verify status shows "inference" mode

**During Inference**:
- System processes data in real-time
- Predictions published continuously
- Monitor in Live Monitoring tab

### Live Monitoring Tab

Shows real-time system activity.

**Features**:
- **Predictions Timeline**: Graph of recent predictions
- **Prediction Confidence**: Confidence levels over time
- **Statistics**:
  - Total predictions
  - Falls detected
  - Average confidence
  - Current status

**Using the Monitor**:
1. Navigate to **Live Monitoring** tab
2. Click **"🔄 Refresh Data"** to update
3. Graphs update automatically as new predictions arrive

## 📡 Sensor Simulator

### Starting the Sensor Simulator

1. Open terminal/command prompt
2. Start Jupyter:
   ```bash
   jupyter notebook
   ```
3. Browser opens automatically
4. Open **`sensor_simulator.ipynb`**

### Configuring the Simulator

1. Find the **Configuration** cell
2. Update **Jetson IP**:
   ```python
   JETSON_IP = "192.168.15.116"  # Your Jetson IP
   ```
3. Run the cell (Shift+Enter)

### Running Cells

To execute notebook cells:
- Click cell and press **Shift+Enter**
- Or click **▶️ Run** button in toolbar

### Connecting to MQTT

1. Find **"Connect to MQTT Broker"** section
2. Run the cell
3. Wait for: **"✅ Connected to MQTT Broker"**

## 🎓 Training Workflow

Complete workflow for training a new model.

### Step 1: Generate Training Data

In **Sensor Simulator** (Jupyter):

1. Navigate to **"Mode 1: Training Data Generation"**
2. Run the function cell to define `generate_training_data()`
3. Run the execution cell:
   ```python
   generate_training_data(
       num_samples=500,      # Number of samples
       fall_percentage=5.0,  # Percentage that are falls
       labeled_by="operator" # Your name/ID
   )
   ```
4. Wait for completion
5. See summary:
   - Total samples
   - Falls generated
   - Normal samples

**⏰ Important**: Wait 30 seconds for data to be stored in database!

### Step 2: Start Training

In **Operator Interface**:

1. Navigate to **Training Control** tab
2. Configure training parameters:
   - **Data Window**: 24 hours (use recent data)
   - **Epochs**: 100 (adjust based on dataset size)
   - **Validation Split**: 0.2 (20% for testing)
3. Click **"🚀 Start Training"**
4. See confirmation: "Training started!"

### Step 3: Monitor Training

Training takes several minutes. To monitor:

```bash
# On Jetson
docker logs -f custom-lstm-detector
```

Look for:
- "Starting training..."
- Epoch progress (1/100, 2/100, ...)
- Validation metrics
- "Training complete!"

### Step 4: Validate Model

In **Operator Interface**:

1. Wait for training to complete
2. Click **"✅ Validate Model"**
3. Check Jetson logs for validation results
4. Review metrics:
   - Accuracy
   - Precision
   - Recall
   - F1 Score

### Step 5: Load New Model

In **Operator Interface**:

1. Navigate to **Inference Control** tab
2. Click **"⭐ Load Best Model"**
3. Wait for confirmation
4. Model is now ready for inference

## 🔮 Inference Workflow

Complete workflow for running predictions.

### Step 1: Prepare System

In **Operator Interface**:

1. Navigate to **Inference Control** tab
2. Ensure model is loaded:
   - Click **"⭐ Load Best Model"** if needed
3. Switch mode:
   - Click **"🔮 Switch to Inference Mode"**
4. Verify:
   - Dashboard shows "inference" mode
   - Status is "Ready"

### Step 2: Generate Sensor Data

In **Sensor Simulator** (Jupyter):

1. Navigate to **"Mode 2: Inference Data Generation"**
2. Run the function cell to define `generate_inference_data()`
3. Run the execution cell:
   ```python
   generate_inference_data(
       duration=60,           # Run for 60 seconds
       fall_probability=0.05  # 5% chance of fall
   )
   ```
4. Data streams to system in real-time

### Step 3: Monitor Predictions

In **Operator Interface**:

1. Navigate to **Live Monitoring** tab
2. Watch predictions appear:
   - **Green (Normal)**: No fall detected
   - **Red (Fall)**: Fall detected
3. View graphs:
   - Prediction timeline
   - Confidence levels
4. Check statistics:
   - Total predictions
   - Falls detected
   - Average confidence

### Step 4: Respond to Falls

When fall is detected:
1. **🚨 Alert** appears in Live Monitoring
2. Check confidence level
3. Take appropriate action based on your procedures

## 🔍 Monitoring & Troubleshooting

### Checking System Health

In **Operator Interface** sidebar:

1. Click **"📊 Get Status"**
2. Review Dashboard tab
3. Check:
   - Connection status: 🟢 Connected
   - Operational mode: Matches your intention
   - Model version: As expected

### Common Issues

#### Issue: Cannot Connect to Jetson

**Symptoms**: Connection fails, red "🔴 Disconnected" status

**Solutions**:
1. Verify Jetson IP address is correct
2. Check Jetson is powered on and connected
3. Verify both computers on same network
4. Try ping: `ping <JETSON_IP>`
5. Check Jetson services: `docker-compose ps`

#### Issue: No Predictions Appearing

**Symptoms**: Sensor data sent but no predictions in Live Monitoring

**Solutions**:
1. Verify system is in **inference mode**
2. Check model is loaded (Dashboard tab)
3. Verify sensor data is being sent (check Jupyter output)
4. Click "Get Status" to refresh
5. Check Jetson logs for errors

#### Issue: Training Not Starting

**Symptoms**: Click "Start Training" but nothing happens

**Solutions**:
1. Verify training data was generated
2. Ensure 30 seconds passed after data generation
3. Check Jetson logs: `docker logs -f custom-lstm-detector`
4. Verify InfluxDB has data in `training_data` bucket
5. Try generating more training data

#### Issue: Low Prediction Confidence

**Symptoms**: Predictions have low confidence (<80%)

**Solutions**:
1. Train with more data
2. Increase training epochs
3. Ensure training data is balanced (mix of falls and normal)
4. Validate model before using
5. Consider retraining with better data

### Viewing Jetson Logs

To see detailed system logs:

```bash
# SSH to Jetson or open terminal on Jetson

# View all services
docker-compose logs -f

# View specific service
docker logs -f mqtt-influx-bridge
docker logs -f custom-lstm-detector

# Press Ctrl+C to exit
```

### Checking Data in InfluxDB

1. Open browser to: `http://<JETSON_IP>:8086`
2. Login:
   - Username: `admin`
   - Password: `<see .env>`
3. Navigate to **Data Explorer**
4. Select bucket:
   - `training_data`: Labeled training data
   - `sensors`: Production inference data
5. Build query to view data

## 📝 Best Practices

### Training Best Practices

1. **Generate Adequate Data**:
   - Minimum 500 samples for initial training
   - 1000+ samples for better accuracy
   - Maintain 5-10% fall percentage

2. **Label Consistently**:
   - Use same `labeled_by` identifier
   - Review generated data quality
   - Balance normal and fall samples

3. **Training Parameters**:
   - Start with 100 epochs
   - Increase if validation accuracy is low
   - Use 20% validation split
   - Use 24 hours of recent data

4. **Validate Before Deployment**:
   - Always validate new models
   - Compare with previous model
   - Test with sample data

### Inference Best Practices

1. **Use Best Model**:
   - Load best performing model
   - Not necessarily latest model
   - Check validation metrics

2. **Monitor Regularly**:
   - Check Live Monitoring tab periodically
   - Review confidence levels
   - Note any anomalies

3. **Retrain Periodically**:
   - Retrain weekly or monthly
   - Use accumulated production data
   - Improve accuracy over time

4. **Document Changes**:
   - Note when models are trained
   - Record validation metrics
   - Track any issues

### Data Generation Best Practices

1. **Realistic Simulation**:
   - Use appropriate fall probability (3-5%)
   - Don't oversimulate falls
   - Generate consistent data

2. **Sufficient Duration**:
   - Training: Generate 500+ samples
   - Inference: Run for at least 60 seconds
   - Long-term: Consider hours of data

3. **Clean Data**:
   - Avoid interrupting generation
   - Complete full batches
   - Wait for confirmation

## ❓ FAQ

### General Questions

**Q: What is the difference between training and inference?**
A: **Training** creates a new model from labeled data. **Inference** uses a trained model to make predictions on new data.

**Q: How long does training take?**
A: Typically 5-15 minutes depending on dataset size and epochs. Larger datasets take longer.

**Q: How often should I retrain the model?**
A: Weekly or monthly, or when you notice declining performance. More frequent retraining improves accuracy.

**Q: What is a good confidence level for predictions?**
A: Above 85% is generally good. Below 70% may indicate the model needs retraining.

### Training Questions

**Q: How much training data do I need?**
A: Minimum 500 samples. 1000+ samples recommended for better accuracy. More data = better model.

**Q: What if training fails?**
A: Check Jetson logs for errors. Verify training data exists. Try generating more data and training again.

**Q: Can I stop training early?**
A: Yes, click "Stop Training" in Operator Interface. The model up to that point will be saved.

**Q: What percentage of falls should be in training data?**
A: 5-10% is realistic. Too many falls can bias the model, too few makes it hard to learn.

### Inference Questions

**Q: Why am I not seeing predictions?**
A: Ensure: (1) Model is loaded, (2) System in inference mode, (3) Sensor data being sent, (4) Connection is active.

**Q: What does "Normal" vs "FALL" mean?**
A: **Normal** = no fall detected. **FALL** = fall detected. Check confidence level.

**Q: Can the model be wrong?**
A: Yes, no model is 100% accurate. Monitor confidence levels. Retrain with more data to improve.

**Q: Should I act on every fall detection?**
A: Depends on confidence. High confidence (>90%) = likely real. Low confidence (<70%) = verify manually.

### Technical Questions

**Q: What if Jetson loses power?**
A: Services restart automatically. Models are preserved. Reconnect Operator Interface when Jetson is back.

**Q: Can I use multiple sensors?**
A: Yes, but requires configuration. Contact system administrator.

**Q: How do I backup models?**
A: Models stored in `/models` directory on Jetson. Copy this directory for backup.

**Q: What if I need to update the system?**
A: Contact system administrator. Updates require rebuilding Docker containers.

## 🆘 Getting Help

### Self-Service

1. Check this guide
2. Review Dashboard in Operator Interface
3. Check Jetson logs
4. Try restarting services
5. Verify network connection

### Escalation

If issue persists:
1. Note exact error message
2. Document steps to reproduce
3. Check when issue started
4. Contact system administrator with details

### Emergency Contacts

[Add your team's contact information here]

---

## 📋 Quick Reference Card

### Starting Work Session

1. Start Operator Interface: `streamlit run operator_interface.py`
2. Connect to Jetson (enter IP, click Connect)
3. Start Jupyter: `jupyter notebook`
4. Open `sensor_simulator.ipynb`
5. Connect simulator to MQTT

### Training a Model

1. Generate 500+ labeled samples in Jupyter
2. Wait 30 seconds
3. In Operator Interface: Training Control → Start Training
4. Wait for completion (check logs)
5. Validate model
6. Load best model

### Running Inference

1. In Operator Interface: Inference Control → Load Best Model
2. Switch to Inference Mode
3. In Jupyter: Generate inference data
4. Monitor predictions in Live Monitoring tab

### Quick Checks

- **Connection**: Sidebar shows 🟢 Connected
- **System Status**: Dashboard tab, click "Get Status"
- **Model Version**: Dashboard tab, check Model Version metric
- **Recent Activity**: Live Monitoring tab, check statistics

---

**User Guide Version**: 1.0
**Last Updated**: 2025-11-02
**For System Version**: 2.0 - Operator Interface Edition

**Remember**: This is manual mode. The operator (you) controls everything. Automatic mode coming in future updates!
