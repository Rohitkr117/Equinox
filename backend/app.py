import time
import serial
import threading
import math
from flask import Flask, jsonify, render_template

# --- Configuration ---
SERIAL_PORT = 'COM18'
BAUD_RATE = 115200

app = Flask(__name__)

# --- Thread-Safe Data Storage ---
sensor_data = {
    'pitch': 0.0, 'roll': 0.0, 'yaw': 0.0,
    'raw_ax': 0, 'raw_ay': 0, 'raw_az': 0,
    'raw_gx': 0, 'raw_gy': 0, 'raw_gz': 0
}
data_lock = threading.Lock()

# --- Complementary Filter Variables ---
# dt is roughly 0.02s since we read at 50Hz
dt = 0.02  
pitch = 0.0
roll = 0.0
yaw = 0.0

# Accelerometer sensitivity (16384 LSB/g for +/- 2g)
# Gyroscope sensitivity (131 LSB/deg/s for +/- 250 deg/s)
ACCEL_SENSITIVITY = 16384.0
GYRO_SENSITIVITY = 131.0

def serial_reader():
    global sensor_data, pitch, roll, yaw
    
    # We loop indefinitely in case the serial port disconnects and reconnects
    while True:
        try:
            ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
            print(f"Successfully connected to {SERIAL_PORT}.")
            
            while True:
                if ser.in_waiting > 0:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    
                    # Ignore initial log messages from the ESP32
                    if "Initialized" in line or not line:
                        continue
                    
                    try:
                        # Expected format: ax,ay,az,gx,gy,gz
                        values = [int(v) for v in line.split(',')]
                        
                        if len(values) == 6:
                            raw_ax, raw_ay, raw_az, raw_gx, raw_gy, raw_gz = values
                            
                            # Convert raw values to physical units
                            # Accel to 'g'
                            ax_g = raw_ax / ACCEL_SENSITIVITY
                            ay_g = raw_ay / ACCEL_SENSITIVITY
                            az_g = raw_az / ACCEL_SENSITIVITY
                            
                            # Gyro to deg/s
                            gx_dps = raw_gx / GYRO_SENSITIVITY
                            gy_dps = raw_gy / GYRO_SENSITIVITY
                            gz_dps = raw_gz / GYRO_SENSITIVITY

                            # Calculate pitch and roll from accelerometer
                            # Using math.atan2 for full quadrant coverage
                            accel_pitch = -math.degrees(math.atan2(ay_g, math.sqrt(ax_g**2 + az_g**2)))
                            accel_roll = math.degrees(math.atan2(-ax_g, az_g))

                            # Apply Complementary Filter
                            # 96% trust in gyro, 4% trust in accelerometer for drift correction
                            alpha = 0.96
                            pitch = alpha * (pitch + gx_dps * dt) + (1.0 - alpha) * accel_pitch
                            roll = alpha * (roll + gy_dps * dt) + (1.0 - alpha) * accel_roll
                            
                            # Yaw estimation from gyro only (will drift over time without magnetometer)
                            yaw = yaw + gz_dps * dt

                            with data_lock:
                                sensor_data['pitch'] = pitch
                                sensor_data['roll'] = roll
                                sensor_data['yaw'] = yaw
                                sensor_data['raw_ax'] = raw_ax
                                sensor_data['raw_ay'] = raw_ay
                                sensor_data['raw_az'] = raw_az
                                sensor_data['raw_gx'] = raw_gx
                                sensor_data['raw_gy'] = raw_gy
                                sensor_data['raw_gz'] = raw_gz
                        
                    except (ValueError, IndexError):
                        pass # Ignore malformed packets quietly to avoid flooding the console
                        
        except serial.SerialException as e:
            print(f"Waiting for {SERIAL_PORT}... is the ESP32 closed in the Arduino IDE? Retrying in 3s.")
            time.sleep(3)

# --- Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data')
def get_data():
    with data_lock:
        return jsonify(sensor_data)

if __name__ == '__main__':
    # Start the serial reading loop in a daemon thread
    thread = threading.Thread(target=serial_reader, daemon=True)
    thread.start()
    
    # Run the web server
    app.run(host='0.0.0.0', port=5000, debug=False)
