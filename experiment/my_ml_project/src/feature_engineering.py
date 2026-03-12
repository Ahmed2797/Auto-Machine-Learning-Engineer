import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self):
        self.scaling_params = {}

    def label_encode(self, df, columns):
        # Basic categorical to numeric conversion
        for col in columns:
            df[col] = df[col].astype('category').cat.codes
        return df

    def scale_features(self, df, columns):
        # Industry standard: (x - mean) / std
        for col in columns:
            mean = df[col].mean()
            std = df[col].std()
            df[col] = (df[col] - mean) / std
            self.scaling_params[col] = {'mean': mean, 'std': std}
        return df

    def run_feature_engineering(self, df):
        print("Engineering features...")
        # Automatically detect categorical and numerical columns
        cat_cols = df.select_dtypes(include=['object']).columns
        num_cols = df.select_dtypes(include=['number']).columns
        
        df = self.label_encode(df, cat_cols)
        df = self.scale_features(df, num_cols)
        return df

if __name__ == "__main__":
    print("Feature Engineering Module Ready")