import os
import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import joblib

# Output folders for storage
results_folder = "Results"
os.makedirs(results_folder, exist_ok=True)

# Data Loading and Scaling
def load_data():
    """ This loads the cleaned dataset and scales the 'flowQuantity_delta' 
    feature to the [0, 1] range. The scaler is then saved for future usage"""
    df = pd.read_csv("Data\cleaned_data.csv")
    X = df[['flowQuantity_delta']].values
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, "Models\scaler.pk1")
    return X_scaled, df

# Defining the Autoencoder Model
class SimpleAutoencoder(nn.Module):
    """ A simple autoencoder with one hidden layer in the encoder and decoder.
    The parameter 'latent_dim' defines the size of the latent representation.
    """
    def __init__(self, input_size, latent_dim):
        super(SimpleAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, input_size)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
# Training and Evaluation function
def train_and_evaluate(latent_dim, learning_rate, batch_size, num_epochs=30):
    """ Trains the autoencoder with the given hyperparameters and evaluates their
    performance. Returns a dictionary of results including final loss, reconstruction
    error threshold, anomaly counts, and the average error."""
    # Load and Scale the data
    X_scaled, df = load_data()
    # Converting the data to a PyTorch tensor
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    # Creating a DataLoader for mini-batch training
    dataset = torch.utils.data.TensorDataset(X_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initializing the autoencoder model
    model = SimpleAutoencoder(input_size=1, latent_dim=latent_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training Loop
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        model.train()
        for batch in dataloader:
            inputs = batch[0]
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * inputs.size(0)
        epoch_loss /= len(dataset)
        # Print epoch loss
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.6f}")

    final_loss = epoch_loss

    # Evaluating the reconstruction error over the entire dataset
    model.eval()
    with torch.no_grad():
        reconstructions = model(X_tensor)
        errors = torch.mean((X_tensor - reconstructions) ** 2, dim=1).numpy()

    # Setting a threshold at the 99th percentile of reconstruction errors
    threshold = np.percentile(errors, 99)
    num_anomalies = np.sum(errors > threshold)
    fraction_anomalies = num_anomalies / len(errors)
    mean_error = np.mean(errors)

    # Returning the hyperparameter configuration and the performance metrics
    result = {
        "latent_dim": latent_dim,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "final_loss": final_loss,
        "threshold": threshold,
        "num_anomalies": int(num_anomalies),
        "fraction_anomalies": fraction_anomalies,
        "mean_error": mean_error
    }
    return result

# Hyperparameter Tuning Loop
def hyperparameter_tuning():
    # Defining the grid
    latent_dims = [4, 8, 16]            # Size of latent representation
    learning_rates = [0.001, 0.0005]    # Different learning rates to test
    batch_sizes = [64, 128]             # Different batch sizes

    results = []
    # Looping over each hyperparameter combination
    for latent_dim in latent_dims:
        for lr in learning_rates:
            for batch_size in batch_sizes:
                print(f"Training with latent_dim={latent_dim}, learning_rate={lr}, batch_size={batch_size}")
                result = train_and_evaluate(latent_dim, lr, batch_size)
                results.append(result)

    # Saving the results for later analysis
    results_df = pd.DataFrame(results)
    results_csv_path = os.path.join(results_folder, "autoencoder_tuning_results.csv")
    results_df.to_csv(results_csv_path, index=False)
    print("Hyperparameter tuning results have been saved to:", results_csv_path)

if __name__ == "__main__":
    hyperparameter_tuning()