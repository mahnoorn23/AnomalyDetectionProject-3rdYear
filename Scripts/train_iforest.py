import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import os

# Output folder
output_folder = "Models"
os.makedirs(output_folder, exist_ok=True)

def train_isolation_forest():
    # Load cleaned data
    df = pd.read_csv("Data\cleaned_data.csv")

    # Select feature/s
    X = df[['flowQuantity_delta']]

    # Train Isolation Forest model
    model = IsolationForest(
        n_estimators=100,
        contamination=0.01, # Expected percentage of anomalies
        random_state=42
    )
    df['anomaly_iforest'] = model.fit_predict(X)

    # Map results: -1 = anomaly, 1 = normal
    df['anomaly_iforest'] = df['anomaly_iforest'].map({1:0, -1:1})

    # Save detected anomalies
    df[df['anomaly_iforest'] == 1].to_csv("Data\iforest_anomalies.csv", index=False)

    # Save model
    import joblib
    joblib.dump(model, f"{output_folder}/isolation_forest_model.pk1")

    # Visualize the results
    plt.figure(figsize=(12, 6))
    plt.plot(df['flowQuantity_delta'], label="Water Usage", color="blue")
    plt.scatter(
        df[df['anomaly_iforest'] == 1].index,
        df[df['anomaly_iforest'] == 1]['flowQuantity_delta'],
        color= 'red', label="Anomaly", marker='x'
    )
    plt.title("Isolation Forest - Anomaly Detection")
    plt.xlabel("Time Index")
    plt.ylabel("Water Usage (in Liters)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Visualizations\isolation_forest_results.png")
    plt.close()

    print("Isolation Forest training complete!")
    print("Anomalies saved!")
    print("Model saved!")
    print("Plot saved!")

if __name__ == "__main__":
    train_isolation_forest()