import pandas as pd

def load_data(file_path):
    """Loads the dataset and processes data types."""
    df = pd.read_csv(file_path)

    # Converting timestamp
    df['utc_time'] = pd.to_datetime(df['utc_time'])

    # Convert to numerical columns with mixed types
    df['cycleIndex_delta'] = pd.to_numeric(df['cycleIndex_delta'], errors='coerce')
    df['flowTimeIndex_delta'] = pd.to_numeric(df['flowTimeIndex_delta'], errors='coerce')

    print("Dataset Loaded Successfully!")
    print(df.info())
    return df

if __name__ == "__main__":
    df = load_data("Data\dataset_filtered 1(in).csv")
    df.to_csv("Data\processed_data.csv", index=False)