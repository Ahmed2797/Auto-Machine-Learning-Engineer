# I will create several essential Python pipeline code files for your ML project within the `src/pipeline` directory (creating the directory if it does not exist). The pipeline files will include:
# 
# - `data_ingestion.py` : Load raw data from `data/raw`
# - `data_processing.py` : Process raw data into processed form in `data/processed`
# - `feature_engineering.py` : Create and save features
# - `model_training.py` : Train and save models
# - `model_evaluation.py` : Evaluate model performance
# 
# Let's start by providing the full content for each of these files.

pipeline_files = {
    "src/pipeline/data_ingestion.py": '''"""
Data Ingestion Module

Loads raw data from the data/raw directory.
"""

import os
import pandas as pd

RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/raw")

def load_raw_data(filename: str) -> pd.DataFrame:
    """
    Load raw data from CSV file located in data/raw.

    Args:
        filename (str): The filename of the raw data CSV.

    Returns:
        pd.DataFrame: Loaded raw data.
    """
    file_path = os.path.join(RAW_DATA_DIR, filename)
    data = pd.read_csv(file_path)
    return data


if __name__ == "__main__":
    # Example usage
    df = load_raw_data("example_raw_data.csv")
    print(df.head())
''',

    "src/pipeline/data_processing.py": '''"""
Data Processing Module

Processes raw data and saves processed data in data/processed directory.
"""

import os
import pandas as pd

PROCESSED_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/processed")

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic example processing function, modify as needed.

    Args:
        df (pd.DataFrame): Raw data.

    Returns:
        pd.DataFrame: Processed data.
    """
    # Example processing: drop rows with missing values
    processed_df = df.dropna().reset_index(drop=True)
    return processed_df

def save_processed_data(df: pd.DataFrame, filename: str):
    """
    Save processed data as CSV in data/processed.

    Args:
        df (pd.DataFrame): Processed data.
        filename (str): Filename to save as.
    """
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    file_path = os.path.join(PROCESSED_DATA_DIR, filename)
    df.to_csv(file_path, index=False)


if __name__ == "__main__":
    # Example usage
    from data_ingestion import load_raw_data
    raw_df = load_raw_data("example_raw_data.csv")
    processed_df = process_data(raw_df)
    save_processed_data(processed_df, "example_processed_data.csv")
    print("Data processed and saved.")
''',

    "src/pipeline/feature_engineering.py": '''"""
Feature Engineering Module

Creates features from processed data and saves features.
"""

import os
import pandas as pd

FEATURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/processed")

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Example feature engineering.

    Args:
        df (pd.DataFrame): Processed data.

    Returns:
        pd.DataFrame: Feature dataframe.
    """
    # Dummy example: create new feature as sum of numeric columns
    numeric_cols = df.select_dtypes(include=["number"]).columns
    if len(numeric_cols) > 0:
        df["feature_sum"] = df[numeric_cols].sum(axis=1)
    return df

def save_features(df: pd.DataFrame, filename: str):
    """
    Save feature data CSV in data/processed.

    Args:
        df (pd.DataFrame): Feature dataframe.
        filename (str): Filename to save as.
    """
    os.makedirs(FEATURES_DIR, exist_ok=True)
    file_path = os.path.join(FEATURES_DIR, filename)
    df.to_csv(file_path, index=False)


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(FEATURES_DIR, "example_processed_data.csv"))
    feature_df = create_features(df)
    save_features(feature_df, "example_features.csv")
    print("Features created and saved.")
''',

    "src/pipeline/model_training.py": '''"""
Model Training Module

Trains machine learning model and saves it to models directory.
"""

import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../models")

def train_model(X: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
    """
    Train a random forest classifier on features.

    Args:
        X (pd.DataFrame): Feature data.
        y (pd.Series): Labels.

    Returns:
        RandomForestClassifier: Trained model.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def save_model(model, filename: str):
    """
    Save trained model using joblib.

    Args:
        model: Trained ML model.
        filename (str): Filename to save the model.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, filename)
    joblib.dump(model, path)

if __name__ == "__main__":
    # Example usage
    df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/processed/example_features.csv"))
    # Assume last column is label for demonstration (replace as per your dataset)
    X = df.drop(columns=["label"], errors="ignore")
    y = df.get("label")
    if y is None:
        print("No label column found for model training. Please add labels to the feature data.")
    else:
        model = train_model(X, y)
        save_model(model, "random_forest_model.joblib")
        print("Model trained and saved.")
''',

    "src/pipeline/model_evaluation.py": '''"""
Model Evaluation Module

Evaluates trained model performance on test data.
"""

import os
import joblib
import pandas as pd
from sklearn.metrics import classification_report

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../models")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/processed")

def load_model(filename: str):
    """
    Load trained model from file.
    """
    path = os.path.join(MODELS_DIR, filename)
    return joblib.load(path)

def evaluate_model(model, X: pd.DataFrame, y: pd.Series):
    """
    Evaluate model using classification report.

    Args:
        model: Trained ML model.
        X (pd.DataFrame): Feature data.
        y (pd.Series): True labels.

    Returns:
        str: Classification report.
    """
    y_pred = model.predict(X)
    report = classification_report(y, y_pred)
    return report

if __name__ == "__main__":
    # Example usage
    feature_file = os.path.join(DATA_DIR, "example_features.csv")
    df = pd.read_csv(feature_file)
    X = df.drop(columns=["label"], errors="ignore")
    y = df.get("label")

    if y is None:
        print("No label column found for evaluation.")
    else:
        model = load_model("random_forest_model.joblib")
        report = evaluate_model(model, X, y)
        print("Model Evaluation Report:\n", report)
'''
}

# Output all files with their complete content:
for filepath, content in pipeline_files.items():
    print(f"# File: {filepath}\n")
    print(content)
    print("\n\n")
# File: src/pipeline/data_ingestion.py

"""
Data Ingestion Module

Loads raw data from the data/raw directory.
"""

import os
import pandas as pd

RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/raw")

def load_raw_data(filename: str) -> pd.DataFrame:
    """
    Load raw data from CSV file located in data/raw.

    Args:
        filename (str): The filename of the raw data CSV.

    Returns:
        pd.DataFrame: Loaded raw data.
    """
    file_path = os.path.join(RAW_DATA_DIR, filename)
    data = pd.read_csv(file_path)
    return data


if __name__ == "__main__":
    # Example usage
    df = load_raw_data("example_raw_data.csv")
    print(df.head())




# File: src/pipeline/data_processing.py

"""
Data Processing Module

Processes raw data and saves processed data in data/processed directory.
"""

import os
import pandas as pd

PROCESSED_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/processed")

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic example processing function, modify as needed.

    Args:
        df (pd.DataFrame): Raw data.

    Returns:
        pd.DataFrame: Processed data.
    """
    # Example processing: drop rows with missing values
    processed_df = df.dropna().reset_index(drop=True)
    return processed_df

def save_processed_data(df: pd.DataFrame, filename: str):
    """
    Save processed data as CSV in data/processed.

    Args:
        df (pd.DataFrame): Processed data.
        filename (str): Filename to save as.
    """
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    file_path = os.path.join(PROCESSED_DATA_DIR, filename)
    df.to_csv(file_path, index=False)


if __name__ == "__main__":
    # Example usage
    from data_ingestion import load_raw_data
    raw_df = load_raw_data("example_raw_data.csv")
    processed_df = process_data(raw_df)
    save_processed_data(processed_df, "example_processed_data.csv")
    print("Data processed and saved.")




# File: src/pipeline/feature_engineering.py

"""
Feature Engineering Module

Creates features from processed data and saves features.
"""

import os
import pandas as pd

FEATURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/processed")

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Example feature engineering.

    Args:
        df (pd.DataFrame): Processed data.

    Returns:
        pd.DataFrame: Feature dataframe.
    """
    # Dummy example: create new feature as sum of numeric columns
    numeric_cols = df.select_dtypes(include=["number"]).columns
    if len(numeric_cols) > 0:
        df["feature_sum"] = df[numeric_cols].sum(axis=1)
    return df

def save_features(df: pd.DataFrame, filename: str):
    """
    Save feature data CSV in data/processed.

    Args:
        df (pd.DataFrame): Feature dataframe.
        filename (str): Filename to save as.
    """
    os.makedirs(FEATURES_DIR, exist_ok=True)
    file_path = os.path.join(FEATURES_DIR, filename)
    df.to_csv(file_path, index=False)


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(FEATURES_DIR, "example_processed_data.csv"))
    feature_df = create_features(df)
    save_features(feature_df, "example_features.csv")
    print("Features created and saved.")




# File: src/pipeline/model_training.py

"""
Model Training Module

Trains machine learning model and saves it to models directory.
"""

import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../models")

def train_model(X: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
    """
    Train a random forest classifier on features.

    Args:
        X (pd.DataFrame): Feature data.
        y (pd.Series): Labels.

    Returns:
        RandomForestClassifier: Trained model.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def save_model(model, filename: str):
    """
    Save trained model using joblib.

    Args:
        model: Trained ML model.
        filename (str): Filename to save the model.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, filename)
    joblib.dump(model, path)

if __name__ == "__main__":
    # Example usage
    df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/processed/example_features.csv"))
    # Assume last column is label for demonstration (replace as per your dataset)
    X = df.drop(columns=["label"], errors="ignore")
    y = df.get("label")
    if y is None:
        print("No label column found for model training. Please add labels to the feature data.")
    else:
        model = train_model(X, y)
        save_model(model, "random_forest_model.joblib")
        print("Model trained and saved.")




# File: src/pipeline/model_evaluation.py

"""
Model Evaluation Module

Evaluates trained model performance on test data.
"""

import os
import joblib
import pandas as pd
from sklearn.metrics import classification_report

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../models")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/processed")

def load_model(filename: str):
    """
    Load trained model from file.
    """
    path = os.path.join(MODELS_DIR, filename)
    return joblib.load(path)

def evaluate_model(model, X: pd.DataFrame, y: pd.Series):
    """
    Evaluate model using classification report.

    Args:
        model: Trained ML model.
        X (pd.DataFrame): Feature data.
        y (pd.Series): True labels.

    Returns:
        str: Classification report.
    """
    y_pred = model.predict(X)
    report = classification_report(y, y_pred)
    return report

if __name__ == "__main__":
    # Example usage
    feature_file = os.path.join(DATA_DIR, "example_features.csv")
    df = pd.read_csv(feature_file)
    X = df.drop(columns=["label"], errors="ignore")
    y = df.get("label")

    if y is None:
        print("No label column found for evaluation.")
    else:
        model = load_model("random_forest_model.joblib")
        report = evaluate_model(model, X, y)
        print("Model Evaluation Report:\n", report)
