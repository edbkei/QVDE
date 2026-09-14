"""
Enhanced MQTT to InfluxDB Bridge
Handles training and inference modes separately
Supports model commands and status
"""

import paho.mqtt.client as mqtt
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
import json
from datetime import datetime
import time
import os

# Configuration from environment or defaults
MQTT_BROKER = os.getenv("MQTT_BROKER", "mosquitto")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
INFLUXDB_URL = os.getenv("INFLUXDB_URL", "http://localhost:8086")
INFLUXDB_TOKEN = os.getenv("INFLUXDB_TOKEN", "")
INFLUXDB_ORG = os.getenv("INFLUXDB_ORG", "")

# Separate buckets for training and production data
INFLUXDB_BUCKET_PRODUCTION = os.getenv("INFLUXDB_BUCKET_PRODUCTION", "sensors")
INFLUXDB_BUCKET_TRAINING = os.getenv("INFLUXDB_BUCKET_TRAINING", "training_data")


class EnhancedMQTTInfluxBridge:
    """
    Enhanced bridge with support for:
    - Training vs Inference mode
    - Separate storage for training data
    - Model command handling
    - Ground truth labels
    """
    
    def __init__(self):
        # MQTT Client
        self.mqtt_client = mqtt.Client(client_id="mqtt_influx_bridge_enhanced")
        self.mqtt_client.on_connect = self.on_mqtt_connect
        self.mqtt_client.on_message = self.on_mqtt_message
        
        # InfluxDB Client
        self.influx_client = InfluxDBClient(
            url=INFLUXDB_URL,
            token=INFLUXDB_TOKEN,
            org=INFLUXDB_ORG
        )
        self.write_api = self.influx_client.write_api(write_options=SYNCHRONOUS)
        
        # Statistics
        self.message_count = 0
        self.training_count = 0
        self.inference_count = 0
        self.error_count = 0
        
        # Create training bucket if not exists
        self._ensure_training_bucket()
    
    def _ensure_training_bucket(self):
        """Ensure training data bucket exists"""
        try:
            buckets_api = self.influx_client.buckets_api()
            bucket = buckets_api.find_bucket_by_name(INFLUXDB_BUCKET_TRAINING)
            
            if not bucket:
                print(f"📦 Creating training data bucket: {INFLUXDB_BUCKET_TRAINING}")
                buckets_api.create_bucket(
                    bucket_name=INFLUXDB_BUCKET_TRAINING,
                    org=INFLUXDB_ORG,
                    retention_rules=[{"everySeconds": 2592000}]  # 30 days
                )
        except Exception as e:
            print(f"⚠️  Could not create training bucket: {e}")
    
    def on_mqtt_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print(f"✅ Connected to MQTT Broker")
            
            # Subscribe to all IoT topics
            client.subscribe("iot/#")
            print(f"📡 Subscribed to: iot/#")
            
            # Subscribe to training topics
            client.subscribe("iot/training/#")
            print(f"📡 Subscribed to: iot/training/#")
            
            # Subscribe to model command topics
            client.subscribe("iot/model/+/command")
            print(f"📡 Subscribed to: iot/model/+/command")
        else:
            print(f"❌ MQTT connection failed with code {rc}")
    
    def on_mqtt_message(self, client, userdata, msg):
        """Handle incoming MQTT messages"""
        try:
            topic = msg.topic

            # DEBUG: Print everything we receive
            print(f"\n?? DEBUG: Received message")
            print(f"   Topic: {topic}")
            print(f"   Payload length: {len(msg.payload)} bytes")
            print(f"   First 200 chars: {msg.payload[:200]}")
            
            # Handle model commands separately
            if "/model/" in topic and "/command" in topic:
                self.handle_model_command(topic, msg.payload)
                return
            
            # Parse JSON payload
            payload = json.loads(msg.payload.decode())

            print(f"   Parsed JSON keys: {list(payload.keys())}")
            print(f"   Has metadata: {'metadata' in payload}")
            print(f"   Has device_info: {'device_info' in payload}")
            print(f"   Has sensor_data: {'sensor_data' in payload}")
            
            # Determine if training or inference data
            is_training = self._is_training_data(topic, payload)
            
            # Handle based on type
            if is_training:
                self.handle_training_data(payload, topic)
                self.training_count += 1
            else:
                self.handle_inference_data(payload, topic)
                self.inference_count += 1
            
            self.message_count += 1
            
            if self.message_count % 10 == 0:
                print(f"📊 Messages: {self.message_count} "
                      f"(Training: {self.training_count}, "
                      f"Inference: {self.inference_count}, "
                      f"Errors: {self.error_count})",
                      end='\r', flush=True)
                
        except json.JSONDecodeError as e:
            self.error_count += 1
            print(f"\n❌ JSON decode error: {e}")
        except Exception as e:
            self.error_count += 1
            print(f"\n❌ Error processing message: {e}")
    
    def _is_training_data(self, topic: str, payload: dict) -> bool:
        """Determine if data is for training"""
        # Check topic
        if "/training/" in topic:
            return True
        
        # Check metadata
        metadata = payload.get("metadata", {})
        operational_mode = metadata.get("operational_mode", "inference")
        data_purpose = metadata.get("data_purpose", "production")
        
        return (operational_mode == "training" or 
                data_purpose in ["training", "testing", "validation"])
    
    def handle_training_data(self, payload: dict, topic: str):
        """Handle training data with ground truth labels"""
        metadata = payload.get("metadata", {})
        device_info = payload.get("device_info", {})
        sensor_data = payload.get("sensor_data", {})
        ground_truth = payload.get("ground_truth", {})
        
        # Extract information
        house_id = metadata.get("house_id", "unknown")
        sensor_id = metadata.get("sensor_id", "unknown")
        application = metadata.get("application", "unknown")
        sensor_type = device_info.get("sensor_type", "unknown")
        data_purpose = metadata.get("data_purpose", "training")
        
        # Create measurement name
        measurement = f"training_{application}_{sensor_type}"
        
        # Create InfluxDB point
        point = Point(measurement)
        
        # Add tags
        point.tag("house_id", house_id)
        point.tag("sensor_id", sensor_id)
        point.tag("application", application)
        point.tag("sensor_type", sensor_type)
        point.tag("data_purpose", data_purpose)
        point.tag("operational_mode", "training")
        
        if "location" in device_info:
            point.tag("location", device_info["location"])
        
        # Add sensor data as fields
        for key, value in sensor_data.items():
            if isinstance(value, (int, float)):
                point.field(f"sensor_{key}", float(value))
        
        # Add ground truth labels as fields
        for key, value in ground_truth.items():
            if isinstance(value, (int, float)):
                point.field(f"label_{key}", float(value))
            elif isinstance(value, bool):
                point.field(f"label_{key}", int(value))
            elif isinstance(value, str):
                point.tag(f"label_{key}", value)
        
        # Add metadata
        if "labeled_by" in ground_truth:
            point.tag("labeled_by", ground_truth["labeled_by"])
        
        # Use timestamp
        if "timestamp" in metadata:
            point.time(metadata["timestamp"])
        
        # Write to training bucket
        self.write_to_influxdb(point, INFLUXDB_BUCKET_TRAINING)
    
    def handle_inference_data(self, payload: dict, topic: str):
        """Handle production inference data"""
        metadata = payload.get("metadata", {})
        device_info = payload.get("device_info", {})
        sensor_data = payload.get("sensor_data", {})
        derived_data = payload.get("derived_data", {})
        model_info = payload.get("model_info", {})
        
        # Extract information
        house_id = metadata.get("house_id", "unknown")
        sensor_id = metadata.get("sensor_id", "unknown")
        application = metadata.get("application", "unknown")
        sensor_type = device_info.get("sensor_type", "unknown")
        
        # Create measurement name
        measurement = f"{application}_{sensor_type}"
        
        # Create InfluxDB point
        point = Point(measurement)
        
        # Add tags
        point.tag("house_id", house_id)
        point.tag("sensor_id", sensor_id)
        point.tag("application", application)
        point.tag("sensor_type", sensor_type)
        point.tag("operational_mode", "inference")
        
        if "location" in device_info:
            point.tag("location", device_info["location"])
        if "sensor_status" in device_info:
            point.tag("sensor_status", device_info["sensor_status"])
        
        # Add device info fields
        if "power_level" in device_info:
            point.field("power_level", float(device_info["power_level"]))
        
        # Add sensor data
        for key, value in sensor_data.items():
            if isinstance(value, (int, float)):
                point.field(key, float(value))
            elif isinstance(value, bool):
                point.field(key, int(value))
        
        # Add derived data (predictions)
        for key, value in derived_data.items():
            if isinstance(value, (int, float)):
                point.field(f"derived_{key}", float(value))
            elif isinstance(value, bool):
                point.field(f"derived_{key}", int(value))
            elif isinstance(value, str):
                point.tag(f"derived_{key}", value)
        
        # Add model information
        if model_info:
            if "model_version" in model_info:
                point.tag("model_version", model_info["model_version"])
            if "confidence" in model_info:
                point.field("model_confidence", float(model_info["confidence"]))
            if "inference_time_ms" in model_info:
                point.field("inference_time_ms", float(model_info["inference_time_ms"]))
        
        # Use timestamp
        #if "timestamp" in metadata:
            #point.time(metadata["timestamp"])
        
        # Write to production bucket
        self.write_to_influxdb(point, INFLUXDB_BUCKET_PRODUCTION)
    
    def handle_model_command(self, topic: str, payload: bytes):
        """Handle model training/management commands"""
        try:
            command_data = json.loads(payload.decode())
            command = command_data.get("command")
            parameters = command_data.get("parameters", {})
            
            print(f"\n📋 Model Command: {command}")
            print(f"   Parameters: {parameters}")
            
            # Store command in InfluxDB for audit trail
            point = Point("model_commands")
            point.tag("command", command)
            point.field("parameters", json.dumps(parameters))
            
            # Extract application from topic: iot/model/{application}/command
            parts = topic.split('/')
            if len(parts) >= 3:
                application = parts[2]
                point.tag("application", application)
            
            self.write_to_influxdb(point, INFLUXDB_BUCKET_PRODUCTION)
            
            # Publish command to processing services
            # (LSTM service subscribes to model commands)
            
        except Exception as e:
            print(f"\n❌ Error handling model command: {e}")
    
    def write_to_influxdb(self, point: Point, bucket: str):
        """Write data point to specified InfluxDB bucket"""
        try:
            print(f"\n?? Attempting write to bucket: {bucket}")
            line_protocol = point.to_line_protocol()
            print(f"   Point (first 300 chars): {line_protocol[:300]}")
            self.write_api.write(bucket=bucket, record=point)
            print(f"? Write successful to {bucket}!")
        except Exception as e:
            self.error_count += 1
            print(f"\n? InfluxDB write error (bucket: {bucket}): {e}")
            import traceback
            print(traceback.format_exc())
    
    def start(self):
        """Start the bridge service"""
        print("🚀 Starting Enhanced MQTT to InfluxDB Bridge")
        print(f"📡 MQTT Broker: {MQTT_BROKER}:{MQTT_PORT}")
        print(f"💾 InfluxDB: {INFLUXDB_URL}")
        print(f"📦 Production Bucket: {INFLUXDB_BUCKET_PRODUCTION}")
        print(f"🎓 Training Bucket: {INFLUXDB_BUCKET_TRAINING}")
        print(f"🏠 Monitoring: All houses, applications, sensors")
        print(f"🔄 Modes: Training + Inference")
        print("-" * 70)
        
        try:
            # Connect to MQTT
            self.mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
            
            # Start MQTT loop
            print("⏳ Waiting for messages...")
            self.mqtt_client.loop_forever()
            
        except KeyboardInterrupt:
            print("\n\n⏹️  Stopping bridge...")
        except Exception as e:
            print(f"\n❌ Error: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the bridge service"""
        print(f"\n📊 Statistics:")
        print(f"   Total messages: {self.message_count}")
        print(f"   Training data: {self.training_count}")
        print(f"   Inference data: {self.inference_count}")
        print(f"   Errors: {self.error_count}")
        
        self.mqtt_client.disconnect()
        self.influx_client.close()
        print("👋 Disconnected")


if __name__ == "__main__":
    bridge = EnhancedMQTTInfluxBridge()
    bridge.start()
