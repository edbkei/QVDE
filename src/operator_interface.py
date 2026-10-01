"""
Operator Interface for Fall Detection System
Web-based control panel for managing training/inference operations
"""

import streamlit as st
import paho.mqtt.client as mqtt
import json
from datetime import datetime
import time
from collections import deque
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Fall Detection - Operator Interface",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
DEFAULT_JETSON_IP = "192.168.15.116"
DEFAULT_MQTT_PORT = 1883

# MQTT Topics
TOPIC_MODEL_COMMAND = "iot/model/fall_detection/command"
TOPIC_MODEL_STATUS = "iot/model/fall_detection/status"
TOPIC_PREDICTION = "ai/fall_prediction"


class OperatorInterface:
    """Main operator interface controller"""
    
    def __init__(self, broker_ip, port=1883):
        self.broker = broker_ip
        self.port = port
        self.client = None
        self.connected = False
        self.status_buffer = deque(maxlen=10)
        self.prediction_buffer = deque(maxlen=100)
        self.last_status = None
        
    def connect(self):
        """Connect to MQTT broker"""
        if not self.connected:
            try:
                self.client = mqtt.Client(client_id="operator_interface")
                self.client.on_connect = self.on_connect
                self.client.on_message = self.on_message
                self.client.connect(self.broker, self.port, 60)
                self.client.loop_start()
                time.sleep(2)
                self.connected = True
                return True, "✅ Connected to Jetson"
            except Exception as e:
                return False, f"❌ Connection failed: {e}"
        return True, "Already connected"
    
    def disconnect(self):
        """Disconnect from MQTT broker"""
        if self.connected and self.client:
            self.client.loop_stop()
            self.client.disconnect()
            self.connected = False
    
    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            client.subscribe(TOPIC_MODEL_STATUS)
            client.subscribe(TOPIC_PREDICTION)
    
    def on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
            
            if msg.topic == TOPIC_MODEL_STATUS:
                self.status_buffer.append(payload)
                self.last_status = payload
            elif msg.topic == TOPIC_PREDICTION:
                self.prediction_buffer.append(payload)
        except Exception as e:
            st.error(f"Error processing message: {e}")
    
    def send_command(self, command, parameters=None):
        """Send command to model service"""
        if not self.connected:
            return False, "Not connected to Jetson"
        
        payload = {
            "command": command,
            "parameters": parameters or {},
            "timestamp": datetime.now().isoformat(),
            "source": "operator_interface"
        }
        
        try:
            self.client.publish(TOPIC_MODEL_COMMAND, json.dumps(payload), qos=1)
            return True, f"Command '{command}' sent successfully"
        except Exception as e:
            return False, f"Failed to send command: {e}"


# Initialize session state
if 'interface' not in st.session_state:
    st.session_state.interface = None
if 'connected' not in st.session_state:
    st.session_state.connected = False
if 'jetson_ip' not in st.session_state:
    st.session_state.jetson_ip = DEFAULT_JETSON_IP
if 'operational_mode' not in st.session_state:
    st.session_state.operational_mode = "inference"


# ==================== HEADER ====================
st.title("🎯 Fall Detection System - Operator Interface")
st.markdown("**Control Panel for Training & Inference Operations**")
st.divider()


# ==================== SIDEBAR - CONNECTION ====================
with st.sidebar:
    st.header("🔌 Connection")
    
    jetson_ip = st.text_input(
        "Jetson IP Address",
        value=st.session_state.jetson_ip,
        help="IP address of your Jetson device"
    )
    
    mqtt_port = st.number_input(
        "MQTT Port",
        value=DEFAULT_MQTT_PORT,
        min_value=1,
        max_value=65535
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔗 Connect", width="stretch"):
            st.session_state.jetson_ip = jetson_ip
            st.session_state.interface = OperatorInterface(jetson_ip, mqtt_port)
            success, message = st.session_state.interface.connect()
            st.session_state.connected = success
            
            if success:
                st.success(message)
            else:
                st.error(message)
    
    with col2:
        if st.button("🔌 Disconnect", width="stretch"):
            if st.session_state.interface:
                st.session_state.interface.disconnect()
                st.session_state.connected = False
                st.info("Disconnected from Jetson")
    
    # Connection status
    if st.session_state.connected:
        st.success("🟢 Connected")
    else:
        st.error("🔴 Disconnected")
    
    st.divider()
    
    # System Mode
    st.header("⚙️ System Mode")
    mode = st.radio(
        "Operational Mode",
        ["inference", "training", "validation"],
        index=0 if st.session_state.operational_mode == "inference" else 1,
        help="Select the current operational mode"
    )
    st.session_state.operational_mode = mode
    
    st.divider()
    
    # Quick Actions
    st.header("⚡ Quick Actions")
    
    if st.button("📊 Get Status", width="stretch"):
        if st.session_state.connected:
            success, message = st.session_state.interface.send_command("get_status")
            if success:
                st.success(message)
        else:
            st.warning("Please connect first")
    
    if st.button("🔄 Refresh Models", width="stretch"):
        if st.session_state.connected:
            success, message = st.session_state.interface.send_command("list_models")
            if success:
                st.success(message)
        else:
            st.warning("Please connect first")


# ==================== MAIN CONTENT ====================

# Check connection
if not st.session_state.connected:
    st.warning("⚠️ Please connect to the Jetson device using the sidebar")
    st.stop()

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dashboard", 
    "🎓 Training Control", 
    "🔮 Inference Control", 
    "📈 Live Monitoring"
])

# ==================== TAB 1: DASHBOARD ====================
with tab1:
    st.header("System Dashboard")
    
    # Status display
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Operational Mode",
            st.session_state.operational_mode.upper(),
            delta="Active"
        )
    
    with col2:
        if st.session_state.interface.last_status:
            current_mode = st.session_state.interface.last_status.get("mode", "unknown")
            st.metric("System Mode", current_mode.upper())
        else:
            st.metric("System Mode", "UNKNOWN")
    
    with col3:
        if st.session_state.interface.last_status:
            model_version = st.session_state.interface.last_status.get("model_version", "N/A")
            st.metric("Model Version", model_version)
        else:
            st.metric("Model Version", "N/A")
    
    st.divider()
    
    # Last status
    st.subheader("📡 Last System Status")
    if st.session_state.interface.last_status:
        st.json(st.session_state.interface.last_status)
    else:
        st.info("No status received yet. Click 'Get Status' in the sidebar.")
    
    # Recent predictions
    st.subheader("🔮 Recent Predictions")
    if len(st.session_state.interface.prediction_buffer) > 0:
        predictions = list(st.session_state.interface.prediction_buffer)[-10:]
        df = pd.DataFrame([{
            'Timestamp': p.get('timestamp', 'N/A'),
            'Prediction': 'FALL' if p.get('prediction', 0) == 1 else 'Normal',
            'Confidence': f"{p.get('confidence', 0)*100:.1f}%"
        } for p in predictions])
        st.dataframe(df, width="stretch")
    else:
        st.info("No predictions received yet.")


# ==================== TAB 2: TRAINING CONTROL ====================
with tab2:
    st.header("🎓 Training Control - Manual Mode")
    
    st.info("""
    **Manual Training Mode**
    1. Use the Sensor Simulator (Jupyter notebook) to generate labeled training data
    2. Configure training parameters below
    3. Click 'Start Training' to begin model training
    4. Monitor progress in the Jetson logs
    """)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Training Parameters")
        
        hours = st.number_input(
            "Data Window (hours)",
            min_value=1,
            max_value=168,
            value=24,
            help="How many hours of historical data to use for training"
        )
        
        epochs = st.number_input(
            "Training Epochs",
            min_value=10,
            max_value=1000,
            value=100,
            help="Number of training iterations"
        )
        
        validation_split = st.slider(
            "Validation Split",
            min_value=0.1,
            max_value=0.5,
            value=0.2,
            step=0.05,
            help="Percentage of data to use for validation"
        )
    
    with col2:
        st.subheader("Actions")
        
        if st.button("🚀 Start Training", width="stretch", type="primary"):
            params = {
                "hours": hours,
                "epochs": epochs,
                "validation_split": validation_split
            }
            success, message = st.session_state.interface.send_command("start_training", params)
            if success:
                st.success(message)
                st.info("Training started! Monitor progress with: `docker logs -f custom-lstm-detector`")
            else:
                st.error(message)
        
        if st.button("⏹️ Stop Training", width="stretch"):
            success, message = st.session_state.interface.send_command("stop_training")
            if success:
                st.success(message)
            else:
                st.error(message)
        
        st.markdown("---")
        
        # Validation with configurable time range
        st.markdown("**✅ Model Validation**")
        validation_hours = st.number_input(
            "Validation Data Window (hours)",
            min_value=1,
            max_value=168,
            value=24,
            help="How many hours of training data to use for validation (default: 24)"
        )
        
        if st.button("🔍 Validate Model", width="stretch"):
            params = {"hours": validation_hours}
            success, message = st.session_state.interface.send_command("validate_model", params)
            if success:
                st.success(message)
                st.info(f"💡 Validating with last {validation_hours} hours of training data")
            else:
                st.error(message)
    
    st.divider()
    
    # Training guidelines
    st.subheader("📋 Training Guidelines")
    st.markdown("""
    **Before Training:**
    1. Generate at least 500 labeled samples using the Sensor Simulator
    2. Ensure data has been ingested (wait ~30 seconds after generation)
    3. Verify training data exists in InfluxDB
    
    **During Training:**
    - Training takes several minutes depending on dataset size
    - Monitor Jetson logs for progress
    - Do not send new data during training
    
    **After Training:**
    1. Validate the model performance
    2. Load the best model
    3. Switch to inference mode
    """)
    
    st.divider()
    
    # Automatic mode placeholder
    st.subheader("🤖 Automatic Mode (Future)")
    st.warning("""
    **Automatic mode will include:**
    - Scheduled automatic training
    - Auto-labeling based on confidence thresholds
    - Automatic model selection and deployment
    - Continuous learning pipeline
    
    *This feature is planned for future deployment.*
    """)


# ==================== TAB 3: INFERENCE CONTROL ====================
with tab3:
    st.header("🔮 Inference Control - Manual Mode")
    
    st.info("""
    **Manual Inference Mode**
    1. Ensure a trained model is loaded
    2. Use the Sensor Simulator to generate sensor data
    3. Monitor predictions in real-time
    """)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Selection")
        
        # Request model list for comparison
        if st.button("📋 Refresh Model List", width="stretch"):
            success, message = st.session_state.interface.send_command("list_models")
            if success:
                st.success("Model list requested - check LSTM service logs")
                st.info("Tip: Check Dashboard tab or Jetson logs for model details")
        
        st.markdown("---")
        
        # Load Best Model (based on performance)
        st.markdown("**⭐ Best Performing Model**")
        st.caption("Automatically selects model with best validation metrics")
        if st.button("🏆 Load Best Model", width="stretch", type="primary"):
            success, message = st.session_state.interface.send_command("load_best_model")
            if success:
                st.success(message)
                st.info("💡 Best model selected based on validation accuracy")
            else:
                st.error(message)
        
        st.markdown("---")
        
        # Load Latest Model
        st.markdown("**🕐 Most Recent Model**")
        st.caption("Loads the most recently trained model")
        if st.button("📅 Load Latest Model", width="stretch"):
            success, message = st.session_state.interface.send_command("load_latest_model")
            if success:
                st.success(message)
                st.info("💡 Latest model loaded (may not be best performing)")
            else:
                st.error(message)
        
        st.markdown("---")
        
        # Load Specific Version
        st.markdown("**📦 Specific Model Version**")
        model_version = st.text_input(
            "Model Version",
            placeholder="e.g., v1.2.3",
            help="Enter exact version to load specific model"
        )
        
        if st.button("🎯 Load Specific Version", width="stretch"):
            if model_version:
                params = {"version": model_version}
                success, message = st.session_state.interface.send_command("load_model_version", params)
                if success:
                    st.success(message)
                else:
                    st.error(message)
            else:
                st.warning("Please enter a model version")
    
    with col2:
        st.subheader("Mode Control")
        
        if st.button("🔮 Switch to Inference Mode", width="stretch", type="primary"):
            success, message = st.session_state.interface.send_command("switch_to_inference")
            if success:
                st.success(message)
                st.session_state.operational_mode = "inference"
            else:
                st.error(message)
        
        if st.button("📋 List Available Models", width="stretch"):
            success, message = st.session_state.interface.send_command("list_models")
            if success:
                st.success(message)
            else:
                st.error(message)
    
    st.divider()
    
    # Model Comparison and Performance Metrics
    st.subheader("📊 Model Comparison & Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📋 View Available Models**")
        st.info("""
        **To see model list with metrics:**
        
        1. Click "List Available Models" (above right)
        2. Check Dashboard tab for status
        3. Or view Jetson logs:
           ```
           docker logs custom-lstm-detector | tail -50
           ```
        
        **Models show:**
        - Version number (e.g., v1.2.0)
        - Creation date
        - Training samples count
        - Current model indicator (👉)
        """)
    
    with col2:
        st.markdown("**🏆 Best vs Latest Model**")
        st.info("""
        **Best Model** (Recommended):
        - Highest validation accuracy
        - Best F1 score
        - Proven performance
        - ✅ Use for production
        
        **Latest Model**:
        - Most recently trained
        - May or may not be best
        - ⚠️ Use for testing only
        """)
    
    st.divider()
    
    # Performance Metrics Guide
    st.subheader("📈 Understanding Model Performance")
    
    with st.expander("ℹ️ How to Compare Models & View Metrics - Click to Expand"):
        st.markdown("""
        ### 🎯 Key Performance Metrics
        
        When comparing models, look for these metrics:
        
        | Metric | What It Means | Target |
        |--------|---------------|--------|
        | **Accuracy** | % of correct predictions | >90% |
        | **Precision** | Of predicted falls, how many were real? | >85% |
        | **Recall** | Of real falls, how many detected? | >90% |
        | **F1 Score** | Balance of precision & recall | >0.90 |
        
        ---
        
        ### 📊 Where to View Model Metrics
        
        #### Method 1: Jetson Logs (Most Detailed)
        
        **View all models with metrics:**
        ```bash
        docker logs custom-lstm-detector | grep -A 30 "AVAILABLE MODELS"
        ```
        
        **View validation results:**
        ```bash
        docker logs custom-lstm-detector | grep -A 10 "Validation Results"
        ```
        
        **View specific model info:**
        ```bash
        docker logs custom-lstm-detector | grep -A 5 "v1.2.0"
        ```
        
        #### Method 2: During Training
        
        Watch training in real-time:
        ```bash
        docker logs -f custom-lstm-detector
        ```
        
        Look for these sections:
        - Training progress (epoch by epoch)
        - Validation results (at end of training)
        - Model saved confirmation with metrics
        
        #### Method 3: Via This Interface
        
        1. Click "List Available Models" button
        2. Go to Dashboard tab
        3. Check "Last System Status" section
        4. Models listed with basic info
        
        ---
        
        ### 🔍 Example: Model Comparison
        
        ```
        ================================================================================
        AVAILABLE MODELS
        ================================================================================
        v1.2.0
          Created: 2025-11-02T14:30:00
          Trained samples: 1000
          Accuracy: 96.5%
          Precision: 94.2%
          Recall: 95.8%
          F1 Score: 0.950
          
        v1.1.0 👉 CURRENT
          Created: 2025-11-02T10:15:00
          Trained samples: 500
          Accuracy: 92.0%
          Precision: 88.5%
          Recall: 90.0%
          F1 Score: 0.892
        ================================================================================
        
        🏆 BEST MODEL: v1.2.0
        📊 ANALYSIS:
        - v1.2.0 has higher accuracy (+4.5%)
        - Better precision (+5.7%)
        - Better recall (+5.8%)
        - Trained on 2x more data
        
        ✅ RECOMMENDATION: Load v1.2.0 for production
        ```
        
        ---
        
        ### 📝 How "Best Model" is Selected
        
        The system automatically ranks models by:
        
        1. **Primary**: Validation Accuracy (highest wins)
        2. **Secondary**: F1 Score (if accuracy tied)
        3. **Tertiary**: Training samples (more data better)
        4. **Final**: Most recent (if all else equal)
        
        ---
        
        ### ⚠️ Important Notes
        
        **1. Best ≠ Latest**
        - "Best Model" = highest performance
        - "Latest Model" = most recent training
        - Always use "Best Model" for production
        
        **2. Training Data Matters**
        - Minimum 500 samples for testing
        - Recommended 1000+ for production
        - Keep 5-10% fall samples (balanced)
        
        **3. Validate Before Use**
        - After training, always validate
        - Check metrics before loading
        - Compare with previous best model
        
        **4. Monitor Performance**
        - Track predictions over time
        - Retrain if accuracy drops
        - Use more data for better models
        
        ---
        
        ### 🚀 Quick Commands
        
        ```bash
        # List all models with full details
        docker logs custom-lstm-detector | grep -A 50 "AVAILABLE MODELS"
        
        # Check current model
        docker logs custom-lstm-detector | grep "CURRENT"
        
        # View validation metrics
        docker logs custom-lstm-detector | grep -A 15 "Validation Results"
        
        # Watch training live
        docker logs -f custom-lstm-detector
        ```
        """)
    
    st.divider()
    
    # Inference guidelines (simplified)
    st.subheader("📋 Quick Inference Guide")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**1️⃣ Load Model**")
        st.markdown("""
        - Click "🏆 Load Best Model"
        - Or load specific version
        - Wait for confirmation
        """)
    
    with col2:
        st.markdown("**2️⃣ Switch Mode**")
        st.markdown("""
        - Click "Switch to Inference"
        - Verify in Dashboard
        - System ready
        """)
    
    with col3:
        st.markdown("**3️⃣ Monitor**")
        st.markdown("""
        - Start Sensor Simulator
        - Watch Live Monitoring
        - Check predictions
        """)
    
    st.divider()
    
    # Automatic mode placeholder
    st.subheader("🤖 Automatic Mode (Future)")
    st.warning("""
    **Automatic mode will include:**
    - Auto-selection of best performing model
    - Automatic failover to backup models
    - Performance monitoring and alerting
    - Automatic model updates based on performance metrics
    
    *This feature is planned for future deployment.*
    """)


# ==================== TAB 4: LIVE MONITORING ====================
with tab4:
    st.header("📈 Live Monitoring")
    
    # Refresh button
    if st.button("🔄 Refresh Data"):
        st.rerun()
    
    # Predictions over time
    st.subheader("Predictions Timeline")
    
    if len(st.session_state.interface.prediction_buffer) > 0:
        predictions = list(st.session_state.interface.prediction_buffer)
        
        # Extract data
        timestamps = [p.get('timestamp', '') for p in predictions]
        pred_values = [p.get('prediction', 0) for p in predictions]
        confidences = [p.get('confidence', 0) for p in predictions]
        
        # Create plot
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Fall Detection Predictions', 'Prediction Confidence'),
            vertical_spacing=0.15
        )
        
        # Predictions
        fig.add_trace(
            go.Scatter(
                y=pred_values,
                mode='lines+markers',
                name='Prediction',
                line=dict(color='red' if pred_values[-1] == 1 else 'green', width=2),
                marker=dict(size=6)
            ),
            row=1, col=1
        )
        
        # Confidence
        fig.add_trace(
            go.Scatter(
                y=confidences,
                mode='lines',
                name='Confidence',
                line=dict(color='blue', width=2),
                fill='tozeroy'
            ),
            row=2, col=1
        )
        
        fig.update_yaxes(title_text="Fall Detected", row=1, col=1)
        fig.update_yaxes(title_text="Confidence", row=2, col=1)
        fig.update_xaxes(title_text="Sample", row=2, col=1)
        
        fig.update_layout(height=600, showlegend=False)
        
        st.plotly_chart(fig, width="stretch")
        
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_predictions = len(predictions)
            st.metric("Total Predictions", total_predictions)
        
        with col2:
            falls_detected = sum(pred_values)
            st.metric("Falls Detected", falls_detected)
        
        with col3:
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            st.metric("Avg Confidence", f"{avg_confidence*100:.1f}%")
        
        with col4:
            if pred_values:
                current_status = "🚨 FALL" if pred_values[-1] == 1 else "✅ Normal"
                st.metric("Current Status", current_status)
        
    else:
        st.info("No predictions received yet. Start generating sensor data to see predictions.")
    
    st.divider()
    
    # Status history
    st.subheader("System Status History")
    if len(st.session_state.interface.status_buffer) > 0:
        status_list = list(st.session_state.interface.status_buffer)
        df = pd.DataFrame(status_list)
        st.dataframe(df, width="stretch")
    else:
        st.info("No status messages received yet.")


# ==================== FOOTER ====================
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p><strong>Fall Detection System - Operator Interface</strong></p>
    <p>Use the Sensor Simulator (Jupyter notebook) to generate data | Monitor Jetson logs for detailed information</p>
    <p><em>Manual Mode: Operator controls all operations | Automatic Mode: Coming in future deployment</em></p>
</div>
""", unsafe_allow_html=True)