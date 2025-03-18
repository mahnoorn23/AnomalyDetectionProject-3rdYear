import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create folder for visualizations to be stored
output_folder = "Visualizations"
os.makedirs(output_folder, exist_ok=True) # Ensure folder exists

def explore_data(df):
    """Performs basic data visualizations for statistical EDA."""

    df['utc_time'] = pd.to_datetime(df['utc_time'])
    df.set_index('utc_time', inplace=True)

    ### 1. Time-Series Plot of Water Flow
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['flowQuantity_delta'], color='blue', label="Water Flow")
    plt.xlabel("Time")
    plt.ylabel("Water Consumption (in Liters)")
    plt.title("Water Consumption Over Time")
    plt.legend()
    plt.savefig(f"{output_folder}/time_series.png") # Save figure
    plt.close()

if __name__ == "__main__":
    df = pd.read_csv("Data\cleaned_data.csv")
    explore_data(df)