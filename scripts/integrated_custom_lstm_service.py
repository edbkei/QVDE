"""
Integrated Custom LSTM Fall Detection Service
Uses the proven Custom LSTM (better performance than Keras)
Supports training and inference modes with MQTT control
"""

import numpy as np
import pickle
import json
import os
from datetime import datetime, timedelta
from collections import defaultdict, deque
import paho.mqtt.client as mqtt
from influxdb_client import InfluxDBClient

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
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, "custom_lstm_fall_model.pkl")
RL_AGENT_PATH = os.path.join(MODEL_DIR, "custom_rl_agent.pkl")

# Model parameters (from your proven configuration)
INPUT_SIZE = 6  # accel_x, accel_y, accel_z, gyro_roll, gyro_pitch, gyro_yaw
HIDDEN_SIZE = 32
SEQUENCE_LENGTH = 10  # Your proven sequence length


class OperationalMode:
    TRAINING = "training"
    INFERENCE = "inference"
    VALIDATION = "validation"


class CustomLSTMFallDetectionService:
    """
    Custom LSTM service with training/inference separation
    Uses your proven SimpleLSTM implementation
    """
    
    def __init__(self):
        # Custom LSTM model (your proven implementation)
        self.lstm_model = None
        self.rl_agent = None
        
        # Operational state
        self.operational_mode = OperationalMode.INFERENCE
        self.is_training = False
        self.model_version = "custom_v1.0.0"
        
        # InfluxDB clients
        self.influx_client = InfluxDBClient(
            url=INFLUXDB_URL,
            token=INFLUXDB_TOKEN,
            org=INFLUXDB_ORG
        )
        self.query_api = self.influx_client.query_api()
        
        # MQTT client
        self.mqtt_client = mqtt.Client(client_id="custom_lstm_fall_detector")
        self.mqtt_client.on_connect = self.on_mqtt_connect
        self.mqtt_client.on_message = self.on_mqtt_message
        self.mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        self.mqtt_client.loop_start()
        
        # Prediction buffer for RL agent
        self.recent_predictions = deque(maxlen=10)
        
        # Try to load existing models
        self.load_models_if_exist()
    
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
            
            if command == "start_training":
                self.start_training_mode(**parameters)
            elif command == "stop_training":
                self.stop_training_mode()
            elif command == "save_model":
                self.save_models()
            elif command == "load_model":
                self.load_models()
            elif command == "switch_to_inference":
                self.switch_to_inference_mode()
            elif command == "get_status":
                self.publish_status()
            
        except Exception as e:
            print(f"❌ Error handling command: {e}")
    
    def initialize_models(self):
        """Initialize new Custom LSTM and RL Agent"""
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
        
        print("✅ Models initialized")
    
    def load_models_if_exist(self):
        """Try to load existing models"""
        if os.path.exists(LSTM_MODEL_PATH) and os.path.exists(RL_AGENT_PATH):
            print("📂 Loading existing models...")
            try:
                self.lstm_model = SimpleLSTM.load(LSTM_MODEL_PATH)
                self.rl_agent = RLAgentWithPersistence.load(RL_AGENT_PATH)
                print("✅ Models loaded successfully")
                self.publish_status()
                return True
            except Exception as e:
                print(f"⚠️  Could not load models: {e}")
                self.initialize_models()
                return False
        else:
            print("⚠️  No existing models found. Initializing new models...")
            self.initialize_models()
            return False
    
    def load_models(self):
        """Load models from disk"""
        try:
            self.lstm_model = SimpleLSTM.load(LSTM_MODEL_PATH)
            self.rl_agent = RLAgentWithPersistence.load(RL_AGENT_PATH)
            print("✅ Models reloaded")
            self.publish_status()
        except Exception as e:
            print(f"❌ Error loading models: {e}")
    
    def save_models(self):
        """Save models to disk"""
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        if self.lstm_model:
            self.lstm_model.save(LSTM_MODEL_PATH, format='pickle')
            
            # Also save timestamped backup
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(MODEL_DIR, f"custom_lstm_backup_{timestamp}.pkl")
            self.lstm_model.save(backup_path, format='pickle')
        
        if self.rl_agent:
            self.rl_agent.save(RL_AGENT_PATH)
            
            # Backup
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(MODEL_DIR, f"custom_rl_agent_backup_{timestamp}.pkl")
            self.rl_agent.save(backup_path)
        
        print("✅ Models saved with backups")
        self.publish_status()
    
    def fetch_training_data(self, hours: int = 24):
        """
        Fetch labeled training data from InfluxDB
        Returns: list of (sensor_sequence, label) tuples
        """
        try:
            print(f"📊 Fetching training data from last {hours} hours...")
            
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
                        'label_fall_occurred': int(record.values.get('label_fall_occurred', 0))
                    }
                    data_points.append(point)
            
            if len(data_points) < SEQUENCE_LENGTH:
                print(f"⚠️  Not enough training data: {len(data_points)} points")
                return []
            
            print(f"✅ Fetched {len(data_points)} training data points")
            
            # Create sequences
            training_samples = self.create_training_sequences(data_points)
            
            return training_samples
            
        except Exception as e:
            print(f"❌ Error fetching training data: {e}")
            return []
    
    def create_training_sequences(self, data_points):
        """Create sequences from data points"""
        # Sort by time
        data_points.sort(key=lambda x: x['time'])
        
        training_samples = []
        
        for i in range(len(data_points) - SEQUENCE_LENGTH + 1):
            sequence_data = []
            
            for j in range(SEQUENCE_LENGTH):
                point = data_points[i + j]
                feature_vector = [
                    point['sensor_x'],
                    point['sensor_y'],
                    point['sensor_z'],
                    point['sensor_gyro_roll'],
                    point['sensor_gyro_pitch'],
                    point['sensor_gyro_yaw']
                ]
                sequence_data.append(feature_vector)
            
            # Label is from the last point in sequence
            label = data_points[i + SEQUENCE_LENGTH - 1]['label_fall_occurred']
            
            sequence_array = np.array(sequence_data)
            training_samples.append((sequence_array, label))
        
        return training_samples
    
    def start_training_mode(self, episodes=1000, hours=24, **kwargs):
        """Start training mode with Custom LSTM"""
        print("\n" + "=" * 70)
        print("🎓 Starting Training Mode (Custom LSTM)")
        print("=" * 70)
        
        self.operational_mode = OperationalMode.TRAINING
        self.is_training = True
        
        # Fetch training data
        training_samples = self.fetch_training_data(hours=hours)
        
        if len(training_samples) == 0:
            print("❌ No training data available. Using synthetic data...")
            training_samples = self.generate_synthetic_data(episodes)
        
        print(f"📊 Training samples: {len(training_samples)}")
        
        # Train Custom LSTM
        print(f"🏋️  Training Custom LSTM for {episodes} episodes...")
        
        fall_count = sum(1 for _, label in training_samples if label == 1)
        normal_count = len(training_samples) - fall_count
        
        print(f"   Falls: {fall_count}, Normal: {normal_count}")
        
        # Training loop
        for episode in range(min(episodes, len(training_samples))):
            sensor_sequence, true_fall = training_samples[episode % len(training_samples)]
            
            # Train LSTM
            self.lstm_model.simple_train(sensor_sequence, true_fall)
            
            # Get prediction for RL training
            prediction = self.lstm_model.predict(sensor_sequence)
            
            # Train RL agent
            action, state = self.rl_agent.choose_action(prediction, list(self.recent_predictions))
            
            # Simulate outcome
            false_alarm = (prediction > 50 and not true_fall)
            reward = self.rl_agent.get_reward(action, prediction, false_alarm, true_fall)
            
            self.rl_agent.update_q_table(state, action, reward)
            self.rl_agent.decay_epsilon()
            
            self.recent_predictions.append(prediction)
            
            if (episode + 1) % 200 == 0:
                print(f"  Episode {episode + 1}/{episodes} - "
                      f"LSTM samples: {self.lstm_model.trained_samples}, "
                      f"RL episodes: {self.rl_agent.total_episodes}")
        
        print("✅ Training completed")
        
        # Auto-save models
        self.save_models()
        
        self.is_training = False
        self.publish_status()
    
    def generate_synthetic_data(self, num_samples):
        """Generate synthetic training data for testing"""
        print("🎲 Generating synthetic training data...")
        
        samples = []
        
        for i in range(num_samples):
            # 5% falls, 95% normal
            is_fall = (i % 20 == 0)
            
            if is_fall:
                # Fall pattern: high acceleration, high rotation
                sequence = np.random.randn(SEQUENCE_LENGTH, INPUT_SIZE) * 5
                sequence[:, :3] += np.array([0, 0, -20])  # Strong downward
            else:
                # Normal pattern: low values
                sequence = np.random.randn(SEQUENCE_LENGTH, INPUT_SIZE) * 0.5
                sequence[:, 2] += -9.81  # Gravity
            
            samples.append((sequence, is_fall))
        
        return samples
    
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
        self.publish_status()
    
    def fetch_recent_production_data(self, minutes=1):
        """Fetch recent production data"""
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
                        'x': record.values.get('x', 0),
                        'y': record.values.get('y', 0),
                        'z': record.values.get('z', 0),
                        'gyro_roll': record.values.get('gyro_roll', 0),
                        'gyro_pitch': record.values.get('gyro_pitch', 0),
                        'gyro_yaw': record.values.get('gyro_yaw', 0)
                    }
                    data_points.append(point)
            
            return data_points
            
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
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
        Real-time inference loop using Custom LSTM
        """
        print("🔄 Starting real-time inference (Custom LSTM)...")
        print(f"⏱️  Checking every {interval} seconds")
        
        if not self.lstm_model:
            print("⚠️  No trained model. Initializing...")
            self.initialize_models()
        
        self.operational_mode = OperationalMode.INFERENCE
        self.publish_status()
        
        import time
        
        try:
            while True:
                if self.operational_mode != OperationalMode.INFERENCE:
                    print("⏸️  Paused (not in inference mode)", end='\r', flush=True)
                    time.sleep(interval)
                    continue
                
                # Fetch recent data
                data_points = self.fetch_recent_production_data(minutes=1)
                
                if len(data_points) < SEQUENCE_LENGTH:
                    print(f"⚠️  Insufficient data ({len(data_points)} points)", end='\r', flush=True)
                    time.sleep(interval)
                    continue
                
                # Prepare sequence
                sequence = self.prepare_sequence(data_points)
                if sequence is None:
                    time.sleep(interval)
                    continue
                
                # Custom LSTM prediction
                start_time = time.time()
                fall_probability = self.lstm_model.predict(sequence)
                inference_time = (time.time() - start_time) * 1000
                
                # RL decision
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
                    'confidence': round(fall_probability / 100 if fall_probability > 50 else 1 - fall_probability / 100, 4),
                    'rl_action': action,
                    'timestamp': datetime.now().isoformat(),
                    'model': 'Custom_LSTM',
                    'model_version': self.model_version,
                    'inference_time_ms': round(inference_time, 2)
                }
                
                self.mqtt_client.publish(
                    MQTT_TOPIC_PREDICTION,
                    json.dumps(prediction_data),
                    qos=1
                )
                
                # Display status
                status = "🚨 FALL RISK!" if fall_probability > 50 else "✅ Normal"
                print(f"{status} | Risk: {fall_probability:.1f}% | "
                      f"Action: {action:20s} | "
                      f"Time: {inference_time:.1f}ms",
                      end='\r', flush=True)
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n⏹️  Stopping inference...")
        finally:
            self.cleanup()
    
    def publish_status(self):
        """Publish model status"""
        lstm_info = self.lstm_model.get_info() if self.lstm_model else {}
        
        status = {
            "application": "fall_detection",
            "mode": self.operational_mode,
            "model_type": "Custom_LSTM",
            "model_loaded": self.lstm_model is not None,
            "model_version": self.model_version,
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
        
        print(f"\n📊 Status published: {self.operational_mode}")
    
    def cleanup(self):
        """Cleanup resources"""
        self.mqtt_client.loop_stop()
        self.mqtt_client.disconnect()
        self.influx_client.close()
        print("👋 Cleanup complete")


if __name__ == "__main__":
    service = CustomLSTMFallDetectionService()
    service.real_time_inference(interval=3)
