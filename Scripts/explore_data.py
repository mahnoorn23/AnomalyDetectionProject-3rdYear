import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def explore_data(df):
    """Performs basic data visualisation for the water usage trends."""
    df['utc_time'] = pd.to_datetime(df['utc_time'])
    df.set_index('utc_time', inplace=True)

    # Time-Series Plot
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['flowQuantity_delta'], color='blue', label="Water Flow")
    plt.xlabel("Time")
    plt.ylabel("Water Consumption (in Liters)")
    plt.title("Water Consumption Over Time")
    plt.legend()
    plt.show()

    # Flow Distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(df['flowQuantity_delta'], bins=30, kde=True)
    plt.xlabel("Water Flow (in Liters)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Water Flow")
    plt.show()

if __name__ == "__main__":
    df = pd.read_csv("Data\cleaned_data.csv")
    explore_data(df)