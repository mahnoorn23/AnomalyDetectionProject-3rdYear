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

    ### 2. Histogram of Water Usage
    plt.figure(figsize=(8, 5))
    sns.histplot(df['flowQuantity_delta'], bins=50, kde=True)
    plt.xlabel("Water Flow (in Liters)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Water Flow")
    plt.savefig(f"{output_folder}/histogram.png")
    plt.close()

    ### 3. Box Plot for Outliers
    plt.figure(figsize=(8, 5))
    sns.boxplot(y=df['flowQuantity_delta'])
    plt.title("Box Plot of Water Usage")
    plt.ylabel("Water Consumption (in Liters)")
    plt.savefig(f"{output_folder}/boxplot.png")
    plt.close()

    ### 4. Pair Plot (Relationships between the variables)
    numerical_features = ['flowQuantity_delta', 'stepIndex_delta', 'cycleIndex_delta']
    sns.pairplot(df[numerical_features])
    plt.savefig(f"{output_folder}/pairplot.png")
    plt.close()

    ### 5. Correlation Heatmap
    plt.figure(figsize=(16, 12))
    correlation = df.corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Feature Correlation Heatmap")
    plt.savefig(f"{output_folder}/correlation_heatmap.png")
    plt.close()

    ### 6. Histogram by Asset Type
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='flowQuantity_delta', hue='asset_type', bins=50, kde=True)
    plt.xlabel("Water Consumption (in Liters)")
    plt.ylabel("Frequency")
    plt.title("Water Usage Distribution by Asset Type")
    plt.legend(title="Asset Type")
    plt.savefig(f"{output_folder}/histogram_by_asset_type.png")
    plt.close()

    ### 7. Daily Average Water Usage
    df['day'] = df.index.date
    daily_avg = df.groupby('day')['flowQuantity_delta'].mean()
    plt.figure(figsize=(12, 6))
    plt.plot(daily_avg.index, daily_avg, color='green', marker='o')
    plt.xlabel("Date")
    plt.ylabel("Average Water Usage (in Liters)")
    plt.title("Daily Average Water Usage")
    plt.savefig(f"{output_folder}/daily_avg.png")
    plt.close()

    print(f"All visualizations saved in {output_folder}/")

if __name__ == "__main__":
    df = pd.read_csv("Data\cleaned_data.csv")
    explore_data(df)