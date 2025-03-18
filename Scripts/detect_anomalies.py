import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
import os

# Creating an output folder
output_folder = "Visualizations"
os.makedirs(output_folder, exist_ok=True)

def detect_anomalies(df):
    """Detects anomalies using z-score and rolling average methods"""

    # Ensure correct timestamp format
    df['utc_time'] = pd.to_datetime(df['utc_time'])
    df.set_index('utc_time', inplace=True)

    # Z-score anomaly detection
    df['z_score'] = zscore(df['flowQuantity_delta'])
    df['z_anomaly'] = df['z_score'].abs() > 3 # Marking extreme deviations

    # Rolling Average Anomaly Detection
    df['rolling_mean'] = df['flowQuantity_delta'].rolling(window=10, min_periods=1).mean()
    df['rolling_std'] = df['flowQuantity_delta'].rolling(window=10, min_periods=1).std()
    threshold = df['rolling_mean'] + (1.5 * df['rolling_std'])
    df['rolling_anomaly'] = df['flowQuantity_delta'] > threshold

    # Combined anomaly flag
    df['anomaly'] = df['z_anomaly'] | df['rolling_anomaly']

    # Save the detected anomalies
    df[df['anomaly']].to_csv("Data/anomalies_detected.csv")

    # Plot and Save Anomalies
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['flowQuantity_delta'], color='blue', label="Water Usage")
    plt.scatter(df[df['anomaly']].index, df[df['anomaly']]['flowQuantity_delta'], color='red', label="Anomalies", marker='o')
    plt.xlabel("Time")
    plt.ylabel("Water Flow (in Liters)")
    plt.title("Anomalies in Water Flow")
    plt.legend()
    plt.savefig(f"{output_folder}/anomalies_detected.png")
    plt.close()

    print(f"Anomalies detected and saved in {output_folder}/")

    return df

if __name__ == "__main__":
    df = pd.read_csv("Data\cleaned_data.csv")
    df = detect_anomalies(df)