import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
import joblib

# Output folders
model_folder = "Models"
results_folder = "Results"
visualizations_folder = "Visualizations"
os.makedirs(model_folder, exist_ok=True)
os.makedirs(results_folder, exist_ok=True)
os.makedirs(visualizations_folder, exist_ok=True)

def load_and_scale_data():
    # Loading the clean data
    df = pd.read_csv("Data\cleaned_data.csv")
    # Select the feature for training the model
    X = df[['flowQuantity_delta']].values
    # Scale the data to [0, 1] range
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    # Save the scaler for future usage
    joblib.dump(scaler, os.path.join(model_folder, "scaler.pk1"))
    return X_scaled, df

def build_autoencoder(input_dim):
    # Define a simple autoencoder architecture
    model = Sequential([
        Dense(16, activation='relu', input_shape=(input_dim,)),
        Dense(8, activation='relu'),
        Dense(16, activation='relu'),
        Dense(input_dim, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_autoencoder():
    # Loading and scaling data
    X_scaled, df = load_and_scale_data()

    # Build the autoencoder
    autoencoder = build_autoencoder(X_scaled.shape[1])
    autoencoder.summary()

    # Train the autoencoder
    history = autoencoder.fit(X_scaled, X_scaled,
                              epochs=30,
                              batch_size=64,
                              validation_split=0.1,
                              verbose=1)
    
    # Save the model
    autoencoder.save(os.path.join(model_folder, "autoencoder_model.h5"))
    print("Autoencoder model has been saved.")

    # Calculating reconstruction error (mean squared error per sample)
    reconstructions = autoencoder.predict(X_scaled)
    mse = np.mean(np.square(X_scaled - reconstructions), axis=1)

    # Set a threshold based on the 99th percentile of reconstruction errors
    threshold = np.percentile(mse, 99)
    print("Reconstruction error threshold:", threshold)

    # Label anomalies: True if the error > threshold, else False
    df['reconstruction_error'] = mse
    df['anomaly_autoencoder'] = df['reconstruction_error'] > threshold

    # Save the predictions to a CSV file
    df.to_csv(os.path.join(results_folder, "autoencoder_anomalies.csv"), index=False)

    # Plot the reconstruction error distribution and threshold
    plt.figure(figsize=(10, 6))
    plt.hist(mse, bins=50, alpha=0.7, label="Reconstruction Error")
    plt.axvline(threshold, color='red', linestyle='--', label=f"Threshold ({threshold:.4f})")
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Frequency")
    plt.title("Reconstruction Error Distribution")
    plt.legend()
    plt.savefig(os.path.join(visualizations_folder, "autoencoder_error_distribution.png"))
    plt.close()

    # Plot the anomalies in water usage
    plt.figure(figsize=(12, 6))
    plt.plot(df['flowQuantity_delta'], label="Water Usage")
    anomaly_indices = df.index[df['anomaly_autoencoder']]
    plt.scatter(anomaly_indices, df.loc[anomaly_indices, 'flowQuantity_delta'], color='red', label="Anomaly")
    plt.xlabel("Index")
    plt.ylabel("Water Usage (in Liters)")
    plt.title("Autoencoder Detected Anomalies")
    plt.legend()
    plt.savefig(os.path.join(visualizations_folder, "autoencoder_anomalies.png"))
    plt.close()

    print("Autoencoder training is complete. Results have been saved in the Results and Visualizations folders.")

if __name__ == "__main__":
    train_autoencoder()