import pandas as pd
import os

def load_data(file_name='data.csv'):
    path = os.path.join('data', file_name)
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"Data loaded successfully with {len(df)} rows.")
        return df
    else:
        print("File not found!")
        return None

if __name__ == "__main__":
    load_data()