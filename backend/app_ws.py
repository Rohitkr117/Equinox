import os
import csv
import datetime
import time
import threading
import math
import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks
import torch
import json
from tcn_model import TremorClassifierTCN
from flask import Flask, jsonify, render_template, request
import websocket  # websocket-client library

# --- Configuration ---
# The ESP32 hotspot assigns itself 192.168.4.1 by default
ESP32_WS_URL = 'ws://192.168.4.1:81'

app = Flask(__name__)

# Create dataset directory if it doesn't exist
DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset')
os.makedirs(DATASET_DIR, exist_ok=True)

# --- Thread-Safe Data Storage ---
sensor_data = {
    'pitch': 0.0, 'roll': 0.0, 'yaw': 0.0,
    'raw_ax': 0, 'raw_ay': 0, 'raw_az': 0,
    'raw_gx': 0, 'raw_gy': 0, 'raw_gz': 0,
    'ax_g': 0.0, 'ay_g': 0.0, 'az_g': 0.0,
    'gx_dps': 0.0, 'gy_dps': 0.0, 'gz_dps': 0.0,
    'dsp_freq': 0.0, 'dsp_amp': 0.0, 'dsp_axis': 'None',
    'needs_label': False,
    'episode_active': False,
    'episode_duration': 0.0,
    'compound_signal': False,
    'xai_fft_amps': [],
    'xai_fft_freqs': [],
    'predicted_class': 'Unknown'
}
data_lock = threading.Lock()

# --- Active Learning / Episode Logic ---
inference_model = None
inference_classes = []

def load_inference_model():
    global inference_model, inference_classes
    try:
        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
        model_path = os.path.join(models_dir, 'tremor_model.pt')
        classes_path = os.path.join(models_dir, 'classes.json')
        if os.path.exists(model_path) and os.path.exists(classes_path):
            with open(classes_path, 'r') as f:
                inference_classes = json.load(f)
            # Load TCN with 2 channels
            model = TremorClassifierTCN(input_channels=2, num_classes=len(inference_classes))
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.eval()
            inference_model = model
            print(f"[{datetime.datetime.now()}] Successfully loaded 2-Channel TCN Model. Classes: {inference_classes}")
    except Exception as e:
        print(f"Error loading inference model: {e}")

load_inference_model()

pending_episode_snapshots = []
episode_active = False
episode_start_time = 0
episode_recording_in_progress = False
episode_full_recording = {'x': [], 'y': [], 'z': [], 'gx': [], 'gy': [], 'gz': []}

# Variables for Episode Averaged FFT
sum_fft_x = None
sum_fft_y = None
sum_fft_z = None
episode_fft_count = 0

# --- DSP Buffers ---
BUFFER_SIZE = 256
SAMPLING_RATE = 50.0 # 50 Hz
buffer_x = []
buffer_y = []
buffer_z = []
buffer_gx = []
buffer_gy = []
buffer_gz = []

# --- Temporal Consistency Filter Variables ---
last_dom_freq = 0.0
freq_streak_count = 0

# --- Complementary Filter Variables ---
dt = 0.02  
pitch = 0.0
roll = 0.0
yaw = 0.0

ACCEL_SENSITIVITY = 16384.0
GYRO_SENSITIVITY = 131.0


def process_sensor_line(line):
    """Processes a single CSV line of sensor data."""
    global sensor_data, pitch, roll, yaw

    try:
        values = [int(v) for v in line.split(',')]
        
        if len(values) == 6:
            raw_ax, raw_ay, raw_az, raw_gx, raw_gy, raw_gz = values
            
            ax_g = raw_ax / ACCEL_SENSITIVITY
            ay_g = raw_ay / ACCEL_SENSITIVITY
            az_g = raw_az / ACCEL_SENSITIVITY
            
            gx_dps = raw_gx / GYRO_SENSITIVITY
            gy_dps = raw_gy / GYRO_SENSITIVITY
            gz_dps = raw_gz / GYRO_SENSITIVITY

            accel_pitch = -math.degrees(math.atan2(ay_g, math.sqrt(ax_g**2 + az_g**2)))
            accel_roll = math.degrees(math.atan2(-ax_g, az_g))

            alpha = 0.96
            pitch = alpha * (pitch + gx_dps * dt) + (1.0 - alpha) * accel_pitch
            roll = alpha * (roll + gy_dps * dt) + (1.0 - alpha) * accel_roll
            yaw = yaw + gz_dps * dt

            # Update DSP buffers
            if len(buffer_x) >= BUFFER_SIZE:
                buffer_x.pop(0)
                buffer_y.pop(0)
                buffer_z.pop(0)
                buffer_gx.pop(0)
                buffer_gy.pop(0)
                buffer_gz.pop(0)
            buffer_x.append(ax_g)
            buffer_y.append(ay_g)
            buffer_z.append(az_g)
            buffer_gx.append(gx_dps)
            buffer_gy.append(gy_dps)
            buffer_gz.append(gz_dps)

            if episode_recording_in_progress:
                episode_full_recording['x'].append(ax_g)
                episode_full_recording['y'].append(ay_g)
                episode_full_recording['z'].append(az_g)
                episode_full_recording['gx'].append(gx_dps)
                episode_full_recording['gy'].append(gy_dps)
                episode_full_recording['gz'].append(gz_dps)

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
                sensor_data['ax_g'] = ax_g
                sensor_data['ay_g'] = ay_g
                sensor_data['az_g'] = az_g
                sensor_data['gx_dps'] = gx_dps
                sensor_data['gy_dps'] = gy_dps
                sensor_data['gz_dps'] = gz_dps
        
    except (ValueError, IndexError):
        pass 


def websocket_reader():
    """Connects to the ESP32 WebSocket server and reads sensor data."""
    def on_message(ws_conn, message):
        line = message.strip()
        if "Initialized" in line or not line:
            return
        process_sensor_line(line)

    def on_error(ws_conn, error):
        print(f"WebSocket error: {error}")

    def on_close(ws_conn, close_status_code, close_msg):
        print("WebSocket connection closed. Reconnecting in 3s...")

    def on_open(ws_conn):
        print(f"Successfully connected to ESP32 at {ESP32_WS_URL}")

    while True:
        try:
            ws_conn = websocket.WebSocketApp(
                ESP32_WS_URL,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
                on_open=on_open
            )
            ws_conn.run_forever()
        except Exception as e:
            print(f"WebSocket connection failed: {e}")
        
        print(f"Waiting for ESP32 hotspot... Retrying in 3s.")
        time.sleep(3)


def dsp_worker():
    """Runs FFT on the sliding window every 0.5s to find dominant tremor frequencies."""
    global sensor_data, buffer_x, buffer_y, buffer_z
    global episode_active, pending_episode_snapshots, episode_start_time
    global sum_fft_x, sum_fft_y, sum_fft_z, episode_fft_count
    global episode_full_recording, episode_recording_in_progress
    global last_dom_freq, freq_streak_count
    
    while True:
        time.sleep(0.5)
        
        if len(buffer_x) < BUFFER_SIZE:
            continue
            
        x_data = np.array(buffer_x)
        y_data = np.array(buffer_y)
        z_data = np.array(buffer_z)
        
        x_data = x_data - np.mean(x_data)
        y_data = y_data - np.mean(y_data)
        z_data = z_data - np.mean(z_data)

        freqs = rfftfreq(BUFFER_SIZE, d=1.0/SAMPLING_RATE)
        valid_idx = np.where((freqs >= 4.0) & (freqs <= 12.0))[0]
        
        if len(valid_idx) == 0:
            continue
            
        # Removed Hamming window (it was halving the energy, breaking the 1.5 amplitude threshold constraint!)
        fft_x = np.abs(rfft(x_data))
        fft_y = np.abs(rfft(y_data))
        fft_z = np.abs(rfft(z_data))
        
        band_freqs = freqs[valid_idx]
        band_x = fft_x[valid_idx]
        band_y = fft_y[valid_idx]
        band_z = fft_z[valid_idx]
        
        # Realtime continuous tracking (fast peak finding)
        highest_amp = 0
        dom_freq = 0
        dom_axis = 'None'
        for ax_name, band_data in [('X', band_x), ('Y', band_y), ('Z', band_z)]:
            max_idx = np.argmax(band_data)
            if band_data[max_idx] > highest_amp:
                highest_amp = band_data[max_idx]
                dom_freq = band_freqs[max_idx]
                dom_axis = ax_name
        
        STREAK_THRESHOLD = 11  # 11 * 0.5s = 5.5s streak. Ensures vibration outlasts the 5.12s window.
        
        activity_window = 25
        recent_x = x_data[-activity_window:]
        recent_y = y_data[-activity_window:]
        recent_z = z_data[-activity_window:]
        
        is_moving_now = False
        if dom_axis == 'X':
            is_moving_now = (np.max(recent_x) - np.min(recent_x)) > 0.1
        elif dom_axis == 'Y':
            is_moving_now = (np.max(recent_y) - np.min(recent_y)) > 0.1
        elif dom_axis == 'Z':
            is_moving_now = (np.max(recent_z) - np.min(recent_z)) > 0.1

        if highest_amp > 1.5 and is_moving_now: 
            
            # Immediately show the live frequency! Don't wait for the streak, 
            # otherwise it "disappears" if the streak resets mid-shake.
            with data_lock:
                sensor_data['dsp_freq'] = dom_freq
                sensor_data['dsp_amp'] = highest_amp / (BUFFER_SIZE/2)
                sensor_data['dsp_axis'] = 'Accel ' + dom_axis

            if abs(dom_freq - last_dom_freq) <= 0.6: 
                freq_streak_count += 1
            else:
                freq_streak_count = 1 
                
            last_dom_freq = dom_freq
            
            if freq_streak_count >= STREAK_THRESHOLD:
                # WE ARE IN AN EPISODE
                if not episode_active:
                    episode_active = True
                    # Backdate the start time so the "Duration" mathematically includes the time you spent building the streak!
                    episode_start_time = time.time() - (STREAK_THRESHOLD * 0.5)
                    pending_episode_snapshots = []
                    
                    # Initialize cumulative FFT sums for Episode Averaging
                    sum_fft_x = np.zeros_like(band_x)
                    sum_fft_y = np.zeros_like(band_y)
                    sum_fft_z = np.zeros_like(band_z)
                    episode_fft_count = 0
                    

                    episode_full_recording = {
                        'x': list(buffer_x), 'y': list(buffer_y), 'z': list(buffer_z),
                        'gx': list(buffer_gx), 'gy': list(buffer_gy), 'gz': list(buffer_gz)
                    }
                    episode_recording_in_progress = True
                    
                    print(f"--- EPISODE STARTED! Frequency: {dom_freq:.1f}Hz ---")
                with data_lock:
                    sensor_data['episode_active'] = True

                # Stack FFT arrays to average later
                sum_fft_x += band_x
                sum_fft_y += band_y
                sum_fft_z += band_z
                episode_fft_count += 1
                
                # EARLY INFERENCE TRIGGER (We don't wait for the episode to end!)
                INFERENCE_STREAK = 16
                if freq_streak_count == INFERENCE_STREAK and inference_model is not None and len(inference_classes) > 0:
                    try:
                        slice_x, slice_y, slice_z = list(buffer_x), list(buffer_y), list(buffer_z)
                        slice_gx, slice_gy, slice_gz = list(buffer_gx), list(buffer_gy), list(buffer_gz)
                        
                        acc_mag = np.sqrt(np.array(slice_x)**2 + np.array(slice_y)**2 + np.array(slice_z)**2)
                        gyr_mag = np.sqrt(np.array(slice_gx)**2 + np.array(slice_gy)**2 + np.array(slice_gz)**2)
                        
                        tensor_data = torch.FloatTensor(np.stack((acc_mag, gyr_mag), axis=0)).unsqueeze(0)
                        
                        inference_model.eval()
                        with torch.no_grad():
                            outputs = inference_model(tensor_data)
                            _, predicted = outputs.max(1)
                            predicted_label = inference_classes[predicted.item()]
                            print(f"--- EARLY LIVE INFERENCE (8s into episode): Identified as '{predicted_label}' ---")
                            
                        with data_lock:
                            sensor_data['predicted_class'] = predicted_label
                            
                    except Exception as e:
                        print(f"Early Inference error: {e}")
        else:
            # END OF EPISODE LOGIC
            if episode_active and freq_streak_count > 0:
                duration = time.time() - episode_start_time
                print(f"--- EPISODE ENDED! Duration: {duration:.1f}s. Calculating Averaged FFT ---")
                
                # Turn off infinite recording so we can safely process the buffer
                episode_recording_in_progress = False
                
                # Perform the EXACT CENTER Slice extraction
                L = len(episode_full_recording['x'])
                if L >= BUFFER_SIZE:
                    start_idx = (L - BUFFER_SIZE) // 2
                    end_idx = start_idx + BUFFER_SIZE
                    
                    slice_x = episode_full_recording['x'][start_idx:end_idx]
                    slice_y = episode_full_recording['y'][start_idx:end_idx]
                    slice_z = episode_full_recording['z'][start_idx:end_idx]
                    
                    pending_episode_snapshots = [{
                        'x': slice_x,
                        'y': slice_y,
                        'z': slice_z,
                        'gx': episode_full_recording['gx'][start_idx:end_idx],
                        'gy': episode_full_recording['gy'][start_idx:end_idx],
                        'gz': episode_full_recording['gz'][start_idx:end_idx]
                    }]
                    print(f"Dataset Slice: Extracted EXACT math center = [{start_idx}:{end_idx}] out of total length {L}")
                    
                    # Calculate the FFT fingerprint ON THIS SPECIFIC SLICE ONLY!
                    final_fft_x = np.abs(rfft(slice_x))
                    final_fft_y = np.abs(rfft(slice_y))
                    final_fft_z = np.abs(rfft(slice_z))
                else:
                    print(f"Episode too short (length {L}), failed to extract valid 256 slice")
                    final_fft_x, final_fft_y, final_fft_z = [], [], []
                
                highest_avg_amp = 0
                best_avg_band = []
                total_compound_peaks = 0
                
                # Dynamic prominence: A peak must stand out by at least 25% of the max peak, and clear a 1.0 absolute floor
                for ax_name, avg_band in [('X', final_fft_x), ('Y', final_fft_y), ('Z', final_fft_z)]:
                    if len(avg_band) == 0:
                        continue
                        
                    max_val = np.max(avg_band)
                    peaks, _ = find_peaks(avg_band, prominence=max(1.0, max_val * 0.25))
                    
                    if len(peaks) > 1:
                        total_compound_peaks += len(peaks)
                        
                    if max_val > highest_avg_amp:
                        highest_avg_amp = max_val
                        best_avg_band = list(avg_band)
                
                final_compound_flag = (total_compound_peaks > 1)

                predicted_label = "Unknown"
                if inference_model is not None and len(inference_classes) > 0 and len(slice_x) == 256:
                    try:
                        # Compute 2-channel input (Accel Mag, Gyro Mag)
                        acc_mag = np.sqrt(np.array(slice_x)**2 + np.array(slice_y)**2 + np.array(slice_z)**2)
                        slice_gx = pending_episode_snapshots[0]['gx']
                        slice_gy = pending_episode_snapshots[0]['gy']
                        slice_gz = pending_episode_snapshots[0]['gz']
                        gyr_mag = np.sqrt(np.array(slice_gx)**2 + np.array(slice_gy)**2 + np.array(slice_gz)**2)
                        
                        tensor_data = torch.FloatTensor(np.stack((acc_mag, gyr_mag), axis=0)).unsqueeze(0)
                        
                        inference_model.eval()
                        with torch.no_grad():
                            outputs = inference_model(tensor_data)
                            _, predicted = outputs.max(1)
                            predicted_label = inference_classes[predicted.item()]
                            print(f"LIVE INFERENCE: Identified as '{predicted_label}'")
                    except Exception as e:
                        print(f"Inference error: {e}")

                with data_lock:
                    sensor_data['predicted_class'] = predicted_label
                    sensor_data['needs_label'] = True
                    sensor_data['episode_duration'] = duration
                    sensor_data['compound_signal'] = final_compound_flag
                    sensor_data['xai_fft_amps'] = best_avg_band
                    sensor_data['xai_fft_freqs'] = list(band_freqs)
                    sensor_data['episode_active'] = False
                    
                episode_active = False

            freq_streak_count = 0
            last_dom_freq = 0.0
            with data_lock:
                sensor_data['dsp_freq'] = 0.0
                sensor_data['dsp_amp'] = 0.0
                sensor_data['dsp_axis'] = 'None'


# --- Routes ---
@app.route('/')
def index():
    return render_template('home.html')

@app.route('/calibration')
def calibration():
    return render_template('index.html')

@app.route('/inference')
def inference():
    return render_template('inference.html')

@app.route('/model_status')
def model_status():
    """HTMX endpoint polled by the Inference Dashboard to check PyTorch readiness"""
    import os, json
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'tremor_model.pt')
    classes_path = os.path.join(os.path.dirname(__file__), 'models', 'classes.json')
    if os.path.exists(model_path) and os.path.exists(classes_path):
        try:
            with open(classes_path, 'r') as f:
                classes = json.load(f)
            classes_str = ", ".join(classes)
            html = f'''
            <div class="px-4 py-3 bg-emerald-900/40 border border-emerald-500/50 rounded-2xl flex items-center gap-3 shadow-2xl glass-card">
                <div class="w-3 h-3 bg-emerald-500 rounded-full shadow-[0_0_10px_#10b981]"></div>
                <div class="flex flex-col">
                    <span class="text-sm font-bold text-emerald-400">TCN Model Active</span>
                    <span class="text-[0.65rem] text-emerald-200/70 pt-1">Detecting: {classes_str}</span>
                </div>
            </div>
            '''
            return html
        except Exception:
            pass

    return '''
    <div class="px-4 py-3 bg-rose-900/40 border border-rose-500/50 rounded-2xl flex flex-col gap-1 shadow-2xl glass-card">
        <div class="flex items-center gap-2">
            <div class="w-3 h-3 bg-rose-500 rounded-full animate-pulse shadow-[0_0_10px_#f43f5e]"></div>
            <span class="text-sm font-bold text-rose-400">Model Not Found</span>
        </div>
        <span class="text-xs text-rose-200/70">Visit Calibration Engine to train platform.</span>
    </div>
    '''

@app.route('/log_data', methods=['POST'])
def log_data():
    global pending_episode_snapshots
    data = request.json
    label = data.get('label', 'Unknown')
    
    with data_lock:
        sensor_data['needs_label'] = False # Reset the UI flag
        
    if not pending_episode_snapshots:
        return jsonify({"status": "error", "message": "No pending data"})
        
    if label == "REJECT":
        pending_episode_snapshots = []
        print("--- USER REJECTED DATA: Discarding all snapshots! ---")
        return jsonify({"status": "success", "message": "Discarded"})
        
    base_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for idx, window in enumerate(pending_episode_snapshots):
        filename = f"{label}_slice{idx+1}_{base_timestamp}.csv"
        filepath = os.path.join(DATASET_DIR, filename)
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['ax_g', 'ay_g', 'az_g', 'gx_dps', 'gy_dps', 'gz_dps'])
            for i in range(len(window['x'])):
                writer.writerow([
                    window['x'][i], window['y'][i], window['z'][i],
                    window['gx'][i], window['gy'][i], window['gz'][i]
                ])
                
    saved_count = len(pending_episode_snapshots)
    pending_episode_snapshots = []
    print(f"Data successfully saved {saved_count} slices to disk!")
    return jsonify({"status": "success", "message": f"Saved {saved_count} files"})

@app.route('/data')
def get_data():
    with data_lock:
        return jsonify(sensor_data)

if __name__ == '__main__':
    thread = threading.Thread(target=websocket_reader, daemon=True)
    thread.start()
    
    dsp_thread = threading.Thread(target=dsp_worker, daemon=True)
    dsp_thread.start()
    
    app.run(host='0.0.0.0', port=5000, debug=False)
