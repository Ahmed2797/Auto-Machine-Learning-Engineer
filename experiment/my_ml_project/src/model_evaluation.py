import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, r2_score

class ModelEvaluator:
    def __init__(self, model_type="classification"):
        self.model_type = model_type

    def evaluate(self, y_true, y_pred):
        print(f"--- Model Evaluation Report ({self.model_type}) ---")
        
        if self.model_type == "classification":
            acc = accuracy_score(y_true, y_pred)
            report = classification_report(y_true, y_pred)
            matrix = confusion_matrix(y_true, y_pred)
            
            print(f"Accuracy: {acc:.4f}")
            print("\nClassification Report:\n", report)
            print("\nConfusion Matrix:\n", matrix)
            return {"accuracy": acc, "report": report}
            
        elif self.model_type == "regression":
            r2 = r2_score(y_true, y_pred)
            print(f"R2 Score: {r2:.4f}")
            return {"r2_score": r2}

if __name__ == "__main__":
    print("Model Evaluation Module Ready")