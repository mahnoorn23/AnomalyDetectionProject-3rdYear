import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler

# Output folders
model_folder = "Models"
results_folder = "Results"
visual_folder = "Visualizations"
os.makedirs(model_folder, exist_ok=True)
os.makedirs(results_folder, exist_ok=True)
os.makedirs(visual_folder, exist_ok=True)

# Defining the autoencoder model using PyTorch
class Autoencoder(nn.Module):
    def __init__(self, input_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_size) # Linear activation for regression output
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def load_and_scale_data():
    # Loading the cleaned dataset
    df = pd.read_csv("Data\cleaned_data.csv")
    # Selecting the feature for training
    X = df[['flowQuantity_delta']].values
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Save the scaler for future usage (new data!)
    joblib.dump(scaler, os.path.join(model_folder, "scaler_pytorch.pk1"))
    return X_scaled, df

def train_autoencoder():
    # Load and scale the data
    X_scaled, df = load_and_scale_data()
    # Convert the data into a PyTorch tensor
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    # Create a DataLoader for mini-batch training
    dataset = torch.utils.data.TensorDataset(X_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    # Initialize the autoencoder model
    model = Autoencoder(input_size=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 30
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            inputs = batch[0]
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * inputs.size(0)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataset):.6f}")

    # Saving the trained model
    torch.save(model.state_dict(), os.path.join(model_folder, "autoencoder_model_pytorch.pth"))
    print("Autoencoder model saved to", os.path.join(model_folder, "autoencoder_model_pytorch.pth"))

    # Computing the reconstruction errors for each sample
    model.eval()
    with torch.no_grad():
        reconstructions = model(X_tensor)
        errors = torch.mean((X_tensor - reconstructions) ** 2, dim=1).numpy()

    # Setting a threshold at the 99th percentile of reconstruction errors
    threshold = np.percentile(errors, 99)
    print("Recontruction error threshold:", threshold)

    # Label anomalies: True if error > threshold
    df['reconstruction_error'] = errors
    df['anomaly_autoencoder'] = df['reconstruction_error'] > threshold

    # Saving the predictions to a CSV file
    result_csv_path = os.path.join(results_folder, "autoencoder_anomalies_pytorch.csv")
    df.to_csv(result_csv_path, index=False)
    print("Autoencoder anomalies have been saved to:", result_csv_path)

    # Plotting the reconstruction error distribution and threshold
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, alpha=0.7)
    plt.axvline(threshold, color='red', linestyle='--', label=f"Threshold: {threshold:.4f}")
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Frequency")
    plt.title("Reconstruction Error Distribution (PyTorch Autoencoder)")
    plt.legend()
    plt.savefig(os.path.join(visual_folder, "autoencoder_error_distribution_pytorch.png"))
    plt.close()
    
    # Plot detected anomalies on water usage series
    plt.figure(figsize=(12, 6))
    plt.plot(df['flowQuantity_delta'], label="Water Usage")
    anomaly_indices = df.index[df['anomaly_autoencoder']]
    plt.scatter(anomaly_indices, df.loc[anomaly_indices, 'flowQuantity_delta'], color='red', label="Anomaly")
    plt.xlabel("Index")
    plt.ylabel("Water Usage (in Liters)")
    plt.title("Autoencoder Detected Anomalies (PyTorch)")
    plt.legend()
    plt.savefig(os.path.join(visual_folder, "autoencoder_anomalies_pytorch.png"))
    plt.close()
    
    print("Autoencoder training and anomaly detection complete.")

if __name__ == "__main__":
    train_autoencoder()