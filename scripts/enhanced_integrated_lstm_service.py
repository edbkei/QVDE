"""
Enhanced Integrated Custom LSTM Fall Detection Service
- Proper model versioning with metadata
- Clear separation of training vs inference
- Best model selection
- Improved RL decision making with feedback
"""

import numpy as np
import pickle
import json
import os
from datetime import datetime, timedelta
from collections import defaultdict, deque
import paho.mqtt.client as mqtt
from influxdb_client import InfluxDBClient
import glob
import re

# Import Custom LSTM
from custom_lstm_persistence import SimpleLSTM, RLAgentWithPersistence

# Configuration
INFLUXDB_URL = os.getenv("INFLUXDB_URL", "http://localhost:8086")
INFLUXDB_TOKEN = os.getenv("INFLUXDB_TOKEN", "")
INFLUXDB_ORG = os.getenv("INFLUXDB_ORG", "")
INFLUXDB_BUCKET_PRODUCTION = os.getenv("INFLUXDB_BUCKET_PRODUCTION", "sensors")
INFLUXDB_BUCKET_TRAINING = os.getenv("INFLUXDB_BUCKET_TRAINING", "training_data")

MQTT_BROKER = os.getenv("MQTT_BROKER", "mosquitto")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
MQTT_TOPIC_PREDICTION = "ai/fall_prediction"
MQTT_TOPIC_MODEL_COMMAND = "iot/model/fall_detection/command"
MQTT_TOPIC_MODEL_STATUS = "iot/model/fall_detection/status"

# Model paths
MODEL_DIR = "/app/models"
MODELS_ARCHIVE_DIR = os.path.join(MODEL_DIR, "archive")

# Model parameters
INPUT_SIZE = 6  # accel_x, accel_y, accel_z, gyro_roll, gyro_pitch, gyro_yaw
HIDDEN_SIZE = 32
SEQUENCE_LENGTH = 10


class ModelVersion:
    """Model version management with metadata"""
    
    def __init__(self, major=1, minor=0, patch=0, metadata=None):
        self.major = major
        self.minor = minor
        self.patch = patch
        self.metadata = metadata or {}
        self.created_at = datetime.now().isoformat()
    
    def __str__(self):
        return f"v{self.major}.{self.minor}.{self.patch}"
    
    def to_dict(self):
        return {
            "version": str(self),
            "major": self.major,
            "minor": self.minor,
            "patch": self.patch,
            "created_at": self.created_at,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_string(cls, version_str):
        """Parse version string like 'v1.2.3' or '1.2.3'"""
        match = re.match(r'v?(\d+)\.(\d+)\.(\d+)', version_str)
        if match:
            major, minor, patch = map(int, match.groups())
            return cls(major, minor, patch)
        return None


class OperationalMode:
    TRAINING = "training"
    INFERENCE = "inference"
    VALIDATION = "validation"


class EnhancedLSTMFallDetectionService:
    """
    Enhanced LSTM service with:
    - Proper model versioning
    - Training from InfluxDB only
    - Best model selection
    - RL feedback loop
    """
    
    def __init__(self):
        # Models
        self.lstm_model = None
        self.rl_agent = None
        self.current_model_version = None
        
        # Operational state
        self.operational_mode = OperationalMode.INFERENCE
        self.is_training = False
        
        # Model registry (tracks all available models)
        self.model_registry = {}
        
        # InfluxDB clients
        self.influx_client = InfluxDBClient(
            url=INFLUXDB_URL,
            token=INFLUXDB_TOKEN,
            org=INFLUXDB_ORG
        )
        self.query_api = self.influx_client.query_api()
        
        # MQTT client
        self.mqtt_client = mqtt.Client(client_id="enhanced_lstm_fall_detector")
        self.mqtt_client.on_connect = self.on_mqtt_connect
        self.mqtt_client.on_message = self.on_mqtt_message
        self.mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        self.mqtt_client.loop_start()
        
        # Prediction buffer for RL agent
        self.recent_predictions = deque(maxlen=10)
        
        # RL feedback for improvement
        self.rl_feedback_buffer = []
        
        # Create directories
        os.makedirs(MODEL_DIR, exist_ok=True)
        os.makedirs(MODELS_ARCHIVE_DIR, exist_ok=True)
        
        # Load model registry
        self.load_model_registry()
        
        # Load best available model
        self.load_best_model()
    
    def on_mqtt_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("✅ Connected to MQTT Broker")
            client.subscribe(MQTT_TOPIC_MODEL_COMMAND)
            print(f"📡 Subscribed to: {MQTT_TOPIC_MODEL_COMMAND}")
        else:
            print(f"❌ MQTT connection failed: {rc}")
    
    def on_mqtt_message(self, client, userdata, msg):
        """Handle model commands"""
        try:
            command_data = json.loads(msg.payload.decode())
            command = command_data.get("command")
            parameters = command_data.get("parameters", {})
            
            print(f"\n📋 Received command: {command}")
            print(f"   Parameters: {parameters}")
            
            if command == "start_training":
                self.start_training_from_influxdb(**parameters)
            elif command == "stop_training":
                self.stop_training_mode()
            elif command == "load_best_model":
                self.load_best_model()
            elif command == "load_latest_model":
                self.load_latest_model()
            elif command == "load_model_version":
                version = parameters.get("version")
                self.load_model_by_version(version)
            elif command == "switch_to_inference":
                self.switch_to_inference_mode()
            elif command == "get_status":
                self.publish_status()
            elif command == "list_models":
                self.list_available_models()
            elif command == "validate_model":
                hours = parameters.get("hours", 1)
                self.validate_current_model(hours=hours)
            
        except Exception as e:
            print(f"❌ Error handling command: {e}")
            import traceback
            traceback.print_exc()
    
    # ==================== MODEL VERSIONING ====================
    
    def load_model_registry(self):
        """Load registry of all available models"""
        print("📚 Loading model registry...")
        
        # Find all model files
        lstm_files = glob.glob(os.path.join(MODEL_DIR, "lstm_v*.pkl"))
        lstm_files += glob.glob(os.path.join(MODELS_ARCHIVE_DIR, "lstm_v*.pkl"))
        
        for filepath in lstm_files:
            try:
                # Extract version from filename
                filename = os.path.basename(filepath)
                match = re.search(r'v(\d+)\.(\d+)\.(\d+)', filename)
                if match:
                    version_str = f"v{match.group(1)}.{match.group(2)}.{match.group(3)}"
                    
                    # Load metadata
                    metadata_file = filepath.replace('.pkl', '_metadata.json')
                    metadata = {}
                    if os.path.exists(metadata_file):
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                    
                    self.model_registry[version_str] = {
                        'filepath': filepath,
                        'metadata': metadata,
                        'created_at': metadata.get('created_at', 'unknown')
                    }
            except Exception as e:
                print(f"⚠️  Could not register model {filepath}: {e}")
        
        print(f"✅ Found {len(self.model_registry)} models in registry")
        return self.model_registry
    
    def get_best_model_version(self):
        """
        Determine best model based on metrics
        Priority: accuracy > F1-score > latest
        """
        if not self.model_registry:
            return None
        
        best_version = None
        best_score = -1
        
        for version, info in self.model_registry.items():
            metadata = info['metadata']
            
            # Calculate composite score
            accuracy = metadata.get('validation_accuracy', 0)
            f1_score = metadata.get('validation_f1_score', 0)
            score = (accuracy * 0.6) + (f1_score * 0.4)
            
            if score > best_score:
                best_score = score
                best_version = version
        
        return best_version
    
    def get_latest_model_version(self):
        """Get the most recently created model"""
        if not self.model_registry:
            return None
        
        latest_version = None
        latest_time = None
        
        for version, info in self.model_registry.items():
            created_at = info.get('created_at', '')
            if created_at and (latest_time is None or created_at > latest_time):
                latest_time = created_at
                latest_version = version
        
        return latest_version
    
    def load_best_model(self):
        """Load the best performing model"""
        best_version = self.get_best_model_version()
        
        if best_version:
            print(f"🏆 Loading best model: {best_version}")
            return self.load_model_by_version(best_version)
        else:
            print("⚠️  No models found. Initializing new model...")
            return self.initialize_new_models()
    
    def load_latest_model(self):
        """Load the most recent model"""
        latest_version = self.get_latest_model_version()
        
        if latest_version:
            print(f"🕐 Loading latest model: {latest_version}")
            return self.load_model_by_version(latest_version)
        else:
            print("⚠️  No models found. Initializing new model...")
            return self.initialize_new_models()
    
    def load_model_by_version(self, version_str):
        """Load a specific model version"""
        if version_str not in self.model_registry:
            print(f"❌ Model version {version_str} not found")
            return False
        
        try:
            model_info = self.model_registry[version_str]
            lstm_path = model_info['filepath']
            rl_path = lstm_path.replace('lstm_', 'rl_agent_')
            
            print(f"📂 Loading model {version_str}...")
            self.lstm_model = SimpleLSTM.load(lstm_path)
            
            if os.path.exists(rl_path):
                self.rl_agent = RLAgentWithPersistence.load(rl_path)
            else:
                print("⚠️  No RL agent found, initializing new one")
                self.rl_agent = RLAgentWithPersistence()
            
            self.current_model_version = ModelVersion.from_string(version_str)
            print(f"✅ Model {version_str} loaded successfully")
            self.publish_status()
            return True
            
        except Exception as e:
            print(f"❌ Error loading model {version_str}: {e}")
            return False
    
    def list_available_models(self):
        """List all available models with their metrics"""
        print("\n" + "="*80)
        print("AVAILABLE MODELS")
        print("="*80)
        
        if not self.model_registry:
            print("No models available")
            return
        
        # Sort by version
        sorted_versions = sorted(self.model_registry.keys(), reverse=True)
        
        for version in sorted_versions:
            info = self.model_registry[version]
            metadata = info['metadata']
            
            is_current = (self.current_model_version and 
                         str(self.current_model_version) == version)
            marker = "👉 CURRENT" if is_current else ""
            
            print(f"\n{version} {marker}")
            print(f"  Created: {info.get('created_at', 'unknown')}")
            print(f"  Trained samples: {metadata.get('trained_samples', 'unknown')}")
            print(f"  Validation accuracy: {metadata.get('validation_accuracy', 'N/A'):.4f}" 
                  if 'validation_accuracy' in metadata else "  Validation: Not performed")
            print(f"  F1 Score: {metadata.get('validation_f1_score', 'N/A'):.4f}"
                  if 'validation_f1_score' in metadata else "")
        
        print("\n" + "="*80)
    
    def initialize_new_models(self):
        """Initialize brand new models"""
        print("🔧 Initializing new models...")
        
        self.lstm_model = SimpleLSTM(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            sequence_length=SEQUENCE_LENGTH
        )
        
        self.rl_agent = RLAgentWithPersistence(
            learning_rate=0.1,
            discount_factor=0.9,
            epsilon=0.1
        )
        
        self.current_model_version = ModelVersion(major=1, minor=0, patch=0)
        print("✅ New models initialized")
        return True
    
    # ==================== TRAINING FROM INFLUXDB ====================
    
    def start_training_from_influxdb(self, hours=24, epochs=100, 
                                     validation_split=0.2, **kwargs):
        """
        Start training using ONLY data from InfluxDB training bucket
        
        Args:
            hours: Hours of historical data to fetch
            epochs: Number of training epochs
            validation_split: Fraction of data for validation
        """
        print("\n" + "="*80)
        print("🎓 STARTING TRAINING FROM INFLUXDB")
        print("="*80)
        print(f"  Data source: InfluxDB training_data bucket")
        print(f"  Time range: Last {hours} hours")
        print(f"  Epochs: {epochs}")
        print(f"  Validation split: {validation_split*100}%")
        print("="*80 + "\n")
        
        self.is_training = True
        self.operational_mode = OperationalMode.TRAINING
        self.publish_status()
        
        try:
            # 1. Fetch training data from InfluxDB
            training_samples = self.fetch_training_data_from_influxdb(hours)
            
            if not training_samples or len(training_samples) < 10:
                print(f"❌ Insufficient training data: {len(training_samples)} samples")
                print("   Please ensure labeled training data is available in InfluxDB")
                self.stop_training_mode()
                return
            
            print(f"✅ Fetched {len(training_samples)} labeled samples from InfluxDB")
            
            # 2. Split into training and validation
            split_idx = int(len(training_samples) * (1 - validation_split))
            train_data = training_samples[:split_idx]
            val_data = training_samples[split_idx:]
            
            print(f"📊 Training set: {len(train_data)} samples")
            print(f"📊 Validation set: {len(val_data)} samples")
            
            # 3. Initialize new model for training
            print("\n🔨 Creating new model for training...")
            self.lstm_model = SimpleLSTM(
                input_size=INPUT_SIZE,
                hidden_size=HIDDEN_SIZE,
                sequence_length=SEQUENCE_LENGTH
            )
            
            # 4. Train LSTM
            print("\n🏋️  Training LSTM model...")
            for epoch in range(epochs):
                epoch_loss = 0
                for sequence, label in train_data:
                    self.lstm_model.simple_train(sequence, label)
                    prediction = self.lstm_model.predict(sequence)
                    error = abs((100 if label else 0) - prediction)
                    epoch_loss += error
                
                avg_loss = epoch_loss / len(train_data)
                
                if epoch % 10 == 0:
                    print(f"  Epoch {epoch}/{epochs}: Avg Loss = {avg_loss:.2f}")
            
            # 5. Validate model
            print("\n🎯 Validating model...")
            val_results = self.validate_model(val_data)
            
            # 6. Train RL agent with validated predictions
            print("\n🤖 Training RL agent...")
            self.train_rl_agent(train_data)
            
            # 7. Save new versioned model
            print("\n💾 Saving new model version...")
            self.save_new_model_version(
                metadata={
                    'trained_samples': len(train_data),
                    'epochs': epochs,
                    'validation_accuracy': val_results['accuracy'],
                    'validation_precision': val_results['precision'],
                    'validation_recall': val_results['recall'],
                    'validation_f1_score': val_results['f1_score'],
                    'training_hours': hours,
                    'data_source': 'influxdb_training_bucket'
                }
            )
            
            print("\n" + "="*80)
            print("✅ TRAINING COMPLETED SUCCESSFULLY")
            print("="*80)
            print(f"  Model version: {self.current_model_version}")
            print(f"  Training samples: {len(train_data)}")
            print(f"  Validation accuracy: {val_results['accuracy']:.4f}")
            print(f"  F1 Score: {val_results['f1_score']:.4f}")
            print("="*80 + "\n")
            
        except Exception as e:
            print(f"\n❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.is_training = False
            self.switch_to_inference_mode()
    
    def fetch_training_data_from_influxdb(self, hours: int = 24):
        """
        Fetch ONLY labeled training data from InfluxDB training bucket
        Returns: list of (sensor_sequence, label) tuples
        """
        try:
            print(f"📊 Fetching training data from InfluxDB...")
            print(f"   Bucket: {INFLUXDB_BUCKET_TRAINING}")
            print(f"   Time range: Last {hours} hours")
            
            query = f'''
            from(bucket: "{INFLUXDB_BUCKET_TRAINING}")
              |> range(start: -{hours}h)
              |> filter(fn: (r) => r["_measurement"] =~ /training_fall_detection/)
              |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            '''
            
            result = self.query_api.query(query=query)
            
            # Parse data points
            data_points = []
            for table in result:
                for record in table.records:
                    point = {
                        'time': record.get_time(),
                        'sensor_x': record.values.get('sensor_x', 0),
                        'sensor_y': record.values.get('sensor_y', 0),
                        'sensor_z': record.values.get('sensor_z', 0),
                        'sensor_gyro_roll': record.values.get('sensor_gyro_roll', 0),
                        'sensor_gyro_pitch': record.values.get('sensor_gyro_pitch', 0),
                        'sensor_gyro_yaw': record.values.get('sensor_gyro_yaw', 0),
                        'label_is_fall': record.values.get('label_is_fall', 0)
                    }
                    data_points.append(point)
            
            if not data_points:
                print("⚠️  No training data found in InfluxDB")
                return []
            
            print(f"   Found {len(data_points)} data points")
            
            # Group into sequences
            sequences = self.create_sequences_from_points(data_points)
            
            print(f"   Created {len(sequences)} training sequences")
            return sequences
            
        except Exception as e:
            print(f"❌ Error fetching training data: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def create_sequences_from_points(self, data_points):
        """Convert individual data points into sequences"""
        sequences = []
        
        # Sort by time
        data_points.sort(key=lambda x: x['time'])
        
        # Create sliding window sequences
        for i in range(len(data_points) - SEQUENCE_LENGTH + 1):
            sequence_points = data_points[i:i+SEQUENCE_LENGTH]
            
            # Extract sensor features
            sequence = []
            for point in sequence_points:
                feature_vector = [
                    point['sensor_x'],
                    point['sensor_y'],
                    point['sensor_z'],
                    point['sensor_gyro_roll'],
                    point['sensor_gyro_pitch'],
                    point['sensor_gyro_yaw']
                ]
                sequence.append(feature_vector)
            
            # Get label (use last point's label)
            label = bool(sequence_points[-1]['label_is_fall'])
            
            sequences.append((np.array(sequence), label))
        
        return sequences
    
    def validate_model(self, validation_data):
        """Validate model on hold-out data"""
        if not validation_data:
            return {
                'accuracy': 0, 'precision': 0, 
                'recall': 0, 'f1_score': 0
            }
        
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0
        
        for sequence, true_label in validation_data:
            prediction = self.lstm_model.predict(sequence)
            predicted_label = prediction > 50  # threshold at 50%
            
            if predicted_label and true_label:
                true_positives += 1
            elif predicted_label and not true_label:
                false_positives += 1
            elif not predicted_label and not true_label:
                true_negatives += 1
            else:  # not predicted_label and true_label
                false_negatives += 1
        
        # Calculate metrics
        total = len(validation_data)
        accuracy = (true_positives + true_negatives) / total if total > 0 else 0
        
        precision = (true_positives / (true_positives + false_positives) 
                    if (true_positives + false_positives) > 0 else 0)
        
        recall = (true_positives / (true_positives + false_negatives)
                 if (true_positives + false_negatives) > 0 else 0)
        
        f1_score = (2 * precision * recall / (precision + recall)
                   if (precision + recall) > 0 else 0)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'true_negatives': true_negatives,
            'false_negatives': false_negatives
        }
        
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1_score:.4f}")
        print(f"  TP={true_positives}, FP={false_positives}, "
              f"TN={true_negatives}, FN={false_negatives}")
        
        return results
    
    def train_rl_agent(self, training_data):
        """Train RL agent using LSTM predictions"""
        if not self.rl_agent:
            self.rl_agent = RLAgentWithPersistence()
        
        for sequence, true_label in training_data:
            prediction = self.lstm_model.predict(sequence)
            action, state = self.rl_agent.choose_action(prediction, [])
            
            # Simulate outcome and reward
            false_alarm = prediction > 70 and not true_label
            reward = self.rl_agent.get_reward(action, prediction, false_alarm, true_label)
            
            self.rl_agent.update_q_table(state, action, reward)
            self.rl_agent.decay_epsilon()
        
        print(f"  RL agent trained: {self.rl_agent.total_episodes} episodes")
    
    def save_new_model_version(self, metadata=None):
        """Save model with new version number and metadata"""
        # Increment version
        if self.current_model_version:
            new_version = ModelVersion(
                major=self.current_model_version.major,
                minor=self.current_model_version.minor + 1,
                patch=0,
                metadata=metadata
            )
        else:
            new_version = ModelVersion(major=1, minor=0, patch=0, metadata=metadata)
        
        version_str = str(new_version)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save LSTM model
        lstm_filename = f"lstm_{version_str}_{timestamp}.pkl"
        lstm_path = os.path.join(MODEL_DIR, lstm_filename)
        self.lstm_model.save(lstm_path, format='pickle')
        
        # Save RL agent
        rl_filename = f"rl_agent_{version_str}_{timestamp}.pkl"
        rl_path = os.path.join(MODEL_DIR, rl_filename)
        self.rl_agent.save(rl_path)
        
        # Save metadata
        metadata_filename = f"lstm_{version_str}_{timestamp}_metadata.json"
        metadata_path = os.path.join(MODEL_DIR, metadata_filename)
        
        full_metadata = new_version.to_dict()
        with open(metadata_path, 'w') as f:
            json.dump(full_metadata, f, indent=2)
        
        # Update current version
        self.current_model_version = new_version
        
        # Reload registry
        self.load_model_registry()
        
        print(f"✅ Model saved as {version_str}")
        print(f"   LSTM: {lstm_path}")
        print(f"   RL Agent: {rl_path}")
        print(f"   Metadata: {metadata_path}")
    
    def stop_training_mode(self):
        """Stop training mode"""
        print("⏹️  Stopping training mode")
        self.is_training = False
        self.operational_mode = OperationalMode.INFERENCE
        self.publish_status()
    
    def switch_to_inference_mode(self):
        """Switch to inference mode"""
        print("🔄 Switching to inference mode")
        self.operational_mode = OperationalMode.INFERENCE
        self.is_training = False
        self.publish_status()
    
    def validate_current_model(self, hours=1):
        """Validate current model on recent training data"""
        print("\n🎯 Validating current model...")
        
        # Fetch recent training data
        validation_data = self.fetch_training_data_from_influxdb(hours=hours)
        
        if not validation_data:
            print("❌ No validation data available")
            return
        
        results = self.validate_model(validation_data)
        
        # Publish results
        validation_report = {
            'model_version': str(self.current_model_version),
            'validation_results': results,
            'timestamp': datetime.now().isoformat()
        }
        
        self.mqtt_client.publish(
            "iot/model/fall_detection/validation",
            json.dumps(validation_report),
            qos=1
        )
    
    # ==================== INFERENCE ====================
    
    def fetch_recent_production_data(self, minutes=1):
        """Fetch recent production data for inference"""
        try:
            query = f'''
            from(bucket: "{INFLUXDB_BUCKET_PRODUCTION}")
              |> range(start: -{minutes}m)
              |> filter(fn: (r) => r["_measurement"] =~ /fall_detection/)
              |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            '''
            
            result = self.query_api.query(query=query)
            
            data_points = []
            for table in result:
                for record in table.records:
                    point = {
                        'time': record.get_time(),
                        'x': record.values.get('accel_x', 0),
                        'y': record.values.get('accel_y', 0),
                        'z': record.values.get('accel_z', 0),
                        'gyro_roll': record.values.get('gyro_roll', 0),
                        'gyro_pitch': record.values.get('gyro_pitch', 0),
                        'gyro_yaw': record.values.get('gyro_yaw', 0)
                    }
                    data_points.append(point)
            
            return data_points
            
        except Exception as e:
            print(f"❌ Error fetching production data: {e}")
            return []
    
    def prepare_sequence(self, data_points):
        """Prepare sequence for prediction"""
        if len(data_points) < SEQUENCE_LENGTH:
            return None
        
        # Take last SEQUENCE_LENGTH points
        recent_points = data_points[-SEQUENCE_LENGTH:]
        
        sequence = []
        for point in recent_points:
            feature_vector = [
                point['x'],
                point['y'],
                point['z'],
                point['gyro_roll'],
                point['gyro_pitch'],
                point['gyro_yaw']
            ]
            sequence.append(feature_vector)
        
        return np.array(sequence)
    
    def real_time_inference(self, interval=3):
        """
        Real-time inference loop using best available model
        RL agent makes final decisions
        """
        print("\n" + "="*80)
        print("🔄 STARTING REAL-TIME INFERENCE")
        print("="*80)
        print(f"  Model: {self.current_model_version}")
        print(f"  Inference interval: {interval} seconds")
        print(f"  Data source: {INFLUXDB_BUCKET_PRODUCTION}")
        print("="*80 + "\n")
        
        if not self.lstm_model:
            print("⚠️  No model loaded. Loading best model...")
            self.load_best_model()
        
        self.operational_mode = OperationalMode.INFERENCE
        self.publish_status()
        
        import time
        
        try:
            iteration = 0
            while True:
                iteration += 1
                
                if self.operational_mode != OperationalMode.INFERENCE:
                    print("⏸️  Paused (not in inference mode)", end='\r', flush=True)
                    time.sleep(interval)
                    continue
                
                # Fetch recent data
                data_points = self.fetch_recent_production_data(minutes=1)
                
                if len(data_points) < SEQUENCE_LENGTH:
                    print(f"⚠️  Insufficient data ({len(data_points)} points)", 
                          end='\r', flush=True)
                    time.sleep(interval)
                    continue
                
                # Prepare sequence
                sequence = self.prepare_sequence(data_points)
                if sequence is None:
                    time.sleep(interval)
                    continue
                
                # LSTM prediction
                start_time = time.time()
                fall_probability = self.lstm_model.predict(sequence)
                inference_time = (time.time() - start_time) * 1000
                
                # RL decision (final decision maker)
                action, state = self.rl_agent.choose_action(
                    fall_probability,
                    list(self.recent_predictions)
                )
                
                self.recent_predictions.append(fall_probability)
                
                # Publish prediction
                prediction_data = {
                    'fall_detected': fall_probability > 50,
                    'fall_probability': round(fall_probability / 100, 4),
                    'fall_percentage': round(fall_probability, 2),
                    'confidence': round(fall_probability / 100 if fall_probability > 50 
                                      else 1 - fall_probability / 100, 4),
                    'rl_action': action,
                    'rl_state': state,
                    'timestamp': datetime.now().isoformat(),
                    'model': 'Enhanced_Custom_LSTM',
                    'model_version': str(self.current_model_version),
                    'inference_time_ms': round(inference_time, 2),
                    'iteration': iteration
                }
                
                self.mqtt_client.publish(
                    MQTT_TOPIC_PREDICTION,
                    json.dumps(prediction_data),
                    qos=1
                )
                
                # Display status
                status = "🚨 FALL RISK!" if fall_probability > 50 else "✅ Normal"
                print(f"[{iteration}] {status} | Risk: {fall_probability:.1f}% | "
                      f"RL Action: {action:20s} | "
                      f"Time: {inference_time:.1f}ms | "
                      f"Model: {self.current_model_version}",
                      end='\r', flush=True)
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n\n⏹️  Stopping inference...")
        finally:
            self.cleanup()
    
    def publish_status(self):
        """Publish model status"""
        lstm_info = self.lstm_model.get_info() if self.lstm_model else {}
        
        status = {
            "application": "fall_detection",
            "mode": self.operational_mode,
            "model_type": "Enhanced_Custom_LSTM",
            "model_loaded": self.lstm_model is not None,
            "model_version": str(self.current_model_version) if self.current_model_version else "none",
            "available_models": len(self.model_registry),
            "lstm_trained_samples": lstm_info.get('training', {}).get('trained_samples', 0),
            "rl_total_episodes": self.rl_agent.total_episodes if self.rl_agent else 0,
            "rl_q_table_size": len(self.rl_agent.q_table) if self.rl_agent else 0,
            "is_training": self.is_training,
            "timestamp": datetime.now().isoformat()
        }
        
        self.mqtt_client.publish(
            MQTT_TOPIC_MODEL_STATUS,
            json.dumps(status),
            qos=1
        )
        
        print(f"\n📊 Status published: {self.operational_mode}, Model: {self.current_model_version}")
    
    def cleanup(self):
        """Cleanup resources"""
        self.mqtt_client.loop_stop()
        self.mqtt_client.disconnect()
        self.influx_client.close()
        print("\n👋 Cleanup complete")


if __name__ == "__main__":
    print("="*80)
    print("ENHANCED LSTM FALL DETECTION SERVICE")
    print("="*80)
    print("\nStarting service...")
    
    service = EnhancedLSTMFallDetectionService()
    
    # Start inference mode by default
    service.real_time_inference(interval=3)
