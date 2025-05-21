import serial
import time
import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firebase Admin SDK
cred = credentials.Certificate("credentials.json")
firebase_admin.initialize_app(cred)

# Firestore client
db = firestore.client()

# Set up the serial communication with Arduino
ser = serial.Serial('COM5', 115200)  # Adjust to match your Arduino's port
time.sleep(2)  # Allow time for serial connection

# Reference to the Firestore collection
collection = db.collection("gates").document("sal")

# Initialize variables
largest_sensor_value = 0

def initialize_motor_control():
    """Initialize the motor control document in Firestore if it doesn't exist"""
    collection.set({
        'height_sensor': 0
    }, merge=True)
    print("Initialized Firestore document for gate1.")

def send_command_to_arduino(command):
    """Sends '1' or '2' to the Arduino to control the motor"""
    ser.write(command.encode())  # Send command
    print(f"Sent command to Arduino: {command}")
    time.sleep(0.5)  # Give Arduino time to process

    # Read and print the Arduino's response
    while ser.in_waiting:
        response = ser.readline().decode('utf-8').strip()
        print(f"Arduino Response: {response}")

def read_and_update_firestore():
    global largest_sensor_value, tilted
    
    while True:
        if ser.in_waiting > 0:
            sensor_data = ser.readline().decode(errors='ignore').strip()
            
            if not sensor_data:
                print("Empty data received, skipping...")
                continue
            
            try:
                sensor_value = int(sensor_data)
                print("Sensor Value: ", sensor_value)
                if sensor_value <= 0:
                    continue  # Ignore zero values
                
                # Convert sensor reading (assuming 2501 is max range)
                height_value = 2501 - sensor_value
                height_db_value = collection.get().to_dict().get('height_sensor', 0)
                print("Height Value:", height_value)
                if height_value > height_db_value:
                    largest_sensor_value = height_value
                    print(f"Updating Firestore: height = {largest_sensor_value}")
                    collection.set({
                        'height_sensor': largest_sensor_value
                    }, merge=True)
                    if largest_sensor_value > 1600:
                        collection.set({
                        'tilt_mode': "original",
                    }, merge=True)
                    elif largest_sensor_value > 1800:
                        collection.set({
                        'tilt_mode': "high",
                    }, merge=True)
                    print("Firestore update successful!")

            except ValueError:
                print(f"Invalid sensor data received: {sensor_data}. Skipping...")

try:
    initialize_motor_control()
    read_and_update_firestore()
except KeyboardInterrupt:
    print("Program interrupted by user")
finally:
    ser.close()
