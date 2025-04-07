import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import os

# Create a folder to store the grid search results
output_folder = "Results"
os.makedirs(output_folder, exist_ok=True)

def run_gridsearch():
    # Loading the cleaned dataset
    df = pd.read_csv("Data\cleaned_data.csv")
    X = df[['flowQuantity_delta']].values

    # Defining the hyperparameter grid
    n_estimators_list = [50, 100, 200]
    contamination_list = [0.005, 0.01, 0.02]
    max_samples_list = ['auto', 0.5, 0.75]

    # A list to store the results
    results = []

    # Loop over all combinations using nested loops
    for n_estimators in n_estimators_list:
        for contamination in contamination_list:
            for max_samples in max_samples_list:
                model = IsolationForest(
                    n_estimators=n_estimators,
                    contamination=contamination,
                    max_samples=max_samples,
                    random_state=42
                )
                # Fitting the model and predicting anomalies (-1 = anomaly, 1 = normal)
                predictions = model.fit_predict(X)
                num_anomalies = np.sum(predictions == -1)
                fraction_anomalies = num_anomalies / len(predictions)

                # Get the decision function scores (higher means more normal)
                scores = model.decision_function(X)
                # Compute the mean score for points flagged as anomalies
                anomaly_scores = scores[predictions == -1]
                if len(anomaly_scores) > 0:
                    mean_anomaly_score = np.mean(anomaly_scores)
                else:
                    mean_anomaly_score = np.nan

                # Record the hyperparameters and their results
                results.append({
                    "n_estimators": n_estimators,
                    "contamination": contamination,
                    "max_samples": max_samples,
                    "num_anomalies": num_anomalies,
                    "fraction_anomalies": fraction_anomalies,
                    "mean_anomaly_score": mean_anomaly_score
                })

    # Creating a DataFrame from the results and saving them as a CSV
    results_df = pd.DataFrame(results)
    results_csv_path = os.path.join(output_folder, "unsupervised_iforest_gridsearch_results.csv")
    results_df.to_csv(results_csv_path, index=False)
    print("Grid search results have been saved to:", results_csv_path)

if __name__ == "__main__":
    run_gridsearch()