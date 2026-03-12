import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

class ModelTrainer:
    def __init__(self, model_name="random_forest.pkl"):
        self.model_name = model_name
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def train(self, X, y):
        print(f"Training started for {self.model_name}...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model.fit(X_train, y_train)
        print("Training completed.")
        
        # Save the model
        model_path = os.path.join('models', self.model_name)
        joblib.dump(self.model, model_path)
        print(f"Model saved at {model_path}")
        
        return X_test, y_test

if __name__ == "__main__":
    print("Model Training Module Ready")