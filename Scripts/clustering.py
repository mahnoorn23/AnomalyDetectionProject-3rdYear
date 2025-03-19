import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

# Create output folder
output_folder = "Visualizations"
os.makedirs(output_folder, exist_ok=True)

def apply_clustering(df):
    """Applies K-Means clustering to group water usage patterns."""
    
    # Ensure timestamp format
    df['utc_time'] = pd.to_datetime(df['utc_time'])
    df.set_index('utc_time', inplace=True)

    # Normalize flow quantity for clustering 
    scaler = StandardScaler()
    df['flowQuantity_scaled'] = scaler.fit_transform(df[['flowQuantity_delta']])

    # Find optimal clusters using the Elbow Method
    X = df[['flowQuantity_scaled']].values
    wcss = []
    for i in range(1, 6):
        kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    # Plot the Elbow curve
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 6), wcss, marker='o', linestyle='--')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS (Within-cluster sum of squares)')
    plt.title('Elbow Method for Optimal Clusters')
    plt.savefig(f"{output_folder}/elbow_method.png")
    plt.close()

    # Apply K-Means Clustering (Assume optimal clusters = 3)
    optimal_k = 3
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X)

    # Plot Clusters
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=df['cluster'], y=df['flowQuantity_delta'])
    plt.xlabel("Cluster")
    plt.ylabel("Water Consumption")
    plt.title("Cluster Analysis of Water Usage")
    plt.savefig(f"{output_folder}/clusters.png")
    plt.close()

    df.to_csv("Data/clustered_data.csv")

    print(f"Clustering completed and saved in {output_folder}/")

    return df

if __name__ == "__main__":
    df = pd.read_csv("Data\cleaned_data.csv")
    df = apply_clustering(df)