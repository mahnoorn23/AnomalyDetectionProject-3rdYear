import pandas as pd

def clean_data(df):
    """Cleans the missing values and processes the categories"""
    df.fillna(method='ffill', inplace=True) # Forward-fill missing values

    # Convert 'stream_temparture' into numerical categories
    df['stream_temperature'] = df['stream_temperature'].map({'Cold': 0, 'Hot': 1})

    # Remove the incorrect values in 'msgcount'
    df = df[df['msgcount'] >= 0]

    print("Data Cleaning Completed!")
    return df

if __name__ == "__main__":
    df = pd.read_csv("Data\processed_data.csv")
    df_cleaned = clean_data(df)
    df_cleaned.to_csv("Data\cleaned_data.csv", index=False)