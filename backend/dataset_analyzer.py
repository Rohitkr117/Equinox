import pandas as pd
import matplotlib.pyplot as plt
import argparse
import sys

def analyze_dataset(csv_path):
    print(f"Loading '{csv_path}' for analysis...")
    try:
        df = pd.    read_csv(csv_path)
    except Exception as e:
        print(f"Error reading dataset: {e}")
        sys.exit(1)

    if len(df) != 256:
        print(f"WARNING: Expected exactly 256 samples (for the Center Slice), but found {len(df)} samples.")
    else:
        print("Dataset size validated: Exactly 256 samples.")

    # Create subplots for Accelerometer and Gyroscope
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Plot Accelerometer Data
    if 'ax_g' in df.columns:
        ax1.plot(df.index, df['ax_g'], label='Accel X (g)', color='r', alpha=0.8)
        ax1.plot(df.index, df['ay_g'], label='Accel Y (g)', color='g', alpha=0.8)
        ax1.plot(df.index, df['az_g'], label='Accel Z (g)', color='b', alpha=0.8)
        ax1.set_ylabel('Acceleration (g)')
        ax1.set_title(f'Accelerometer Data - {csv_path}')
        ax1.legend(loc='upper right')
        ax1.grid(True, linestyle='--', alpha=0.6)

    # Plot Gyroscope Data
    if 'gx_dps' in df.columns:
        ax2.plot(df.index, df['gx_dps'], label='Gyro X (dps)', color='r', alpha=0.8)
        ax2.plot(df.index, df['gy_dps'], label='Gyro Y (dps)', color='g', alpha=0.8)
        ax2.plot(df.index, df['gz_dps'], label='Gyro Z (dps)', color='b', alpha=0.8)
        ax2.set_xlabel('Sample Index (0-255)')
        ax2.set_ylabel('Angular Velocity (deg/s)')
        ax2.set_title('Gyroscope Data')
        ax2.legend(loc='upper right')
        ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Change this hardcoded path to point to the CSV you want to analyze, or use command line args
    DEFAULT_CSV_PATH = r"C:\\Users\\rohit\\Desktop\\Coding\\Equinox\\backend\\dataset\\Compound_Random_slice1_20260409_092936.csv"

    parser = argparse.ArgumentParser(description="Analyze 256-sample dataset chunks")
    parser.add_argument("csv_file", nargs='?', default=DEFAULT_CSV_PATH, help="Path to the dataset CSV file")
    
    args = parser.parse_args()
    analyze_dataset(args.csv_file)
