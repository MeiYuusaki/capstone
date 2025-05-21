import serial
import time
import firebase_admin
from firebase_admin import credentials, firestore
import threading

cred = credentials.Certificate("credentials.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

doc_ref = db.collection("gates").document("sal")

SERIAL_PORT = 'COM8'
BAUD_RATE = 9600
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
time.sleep(2)

alpha = 1.05

processing_tilt = False

def serial_monitor():
    print("Starting Serial Monitor...")
    while True:
        try:
            arduino_output = ser.readline().decode().strip()
            if arduino_output:
                print("Arduino:", arduino_output)
        except Exception as e:
            print(f"Serial Monitor Error: {e}")
            break

serial_thread = threading.Thread(target=serial_monitor, daemon=True)
serial_thread.start()

def on_snapshot(doc_snapshot, changes, read_time):
    global processing_tilt
    if processing_tilt:
        return

    for doc in doc_snapshot:
        data = doc.to_dict()
        steps_per_revolution = data.get("steps_per_revolution", 0)
        mode = data.get("tilt_mode", "low")
        to_tilt = data.get("to_tilt", False)

        if not to_tilt:
            return

        processing_tilt = True
        direction = 0
        arduino_steps = 0

        print(f"New Command from Firestore → Steps: {steps_per_revolution}, Mode: {mode}")

        if mode == "low":
            print("Mode: Low")
            if steps_per_revolution == 400:
                doc_ref.update({"to_tilt": False})
                processing_tilt = False
                return
            elif steps_per_revolution == 1000:
                arduino_steps = 600
                direction = 1
                steps_per_revolution = 400
            elif steps_per_revolution == 2000:
                arduino_steps = int(1000 * alpha) + 600
                direction = 1
                steps_per_revolution = 400

        elif mode == "original":
            print("Mode: Original")
            if steps_per_revolution == 400:
                arduino_steps = int(600 * alpha)
                direction = 2
                steps_per_revolution = 1000
            elif steps_per_revolution == 1000:
                doc_ref.update({"to_tilt": False})
                processing_tilt = False
                return
            elif steps_per_revolution == 2000:
                arduino_steps = int(1000 * alpha)
                direction = 1
                steps_per_revolution = 1000

        elif mode == "high":
            print("Mode: High")
            if steps_per_revolution == 400:
                arduino_steps = int(600 * alpha) + 1000
                direction = 2
                steps_per_revolution = 2000
            elif steps_per_revolution == 1000:
                arduino_steps = 1000
                direction = 2
                steps_per_revolution = 2000
            elif steps_per_revolution == 2000:
                doc_ref.update({"to_tilt": False})
                processing_tilt = False
                return

        else:
            print("Error: Invalid mode received.")
            processing_tilt = False
            return

        command = f"{arduino_steps} {direction}\n"
        ser.write(command.encode())
        print("Command Sent to Arduino:", command.strip())

        doc_ref.update({
            "to_tilt": False,
            "steps_per_revolution": steps_per_revolution
        })
        print("Firestore Updated: to_tilt set to 0")

        processing_tilt = False

doc_watch = doc_ref.on_snapshot(on_snapshot)

print("Listening for Firestore updates...")

while True:
    time.sleep(1)