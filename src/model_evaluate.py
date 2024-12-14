import os
import pickle
import mlflow
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score
from dataclasses import dataclass
from mlflow import log_metric, log_param

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@dataclass
class DataPaths:
    Count_Vector = os.path.join(project_dir, "Data/Count_Vector.pickle")
    tfidf_Vector = os.path.join(project_dir, "Data/tfidf.pickle")
    tfidf_maxfeat_1000_Vector = os.path.join(project_dir, "Data/tfidf_maxfeat_1000.pickle")
    tfidf_maxfeat_2000_Vector = os.path.join(project_dir, "Data/tfidf_maxfeat_2000.pickle")
    tfidf_maxfeat_3000_Vector = os.path.join(project_dir, "Data/tfidf_maxfeat_3000.pickle")
    target_data_path: str = "/home/dev/sms_spam_classifier/Data/dependant_data_path.csv"
    artifact_path: str = os.path.join("artifacts", "models")
    results_path: str = os.path.join("artifacts", "evaluation_results.csv")

class ModelEvaluator:
    def __init__(self):
        self.data_paths = DataPaths()

        # Ensure results directory exists
        os.makedirs(os.path.dirname(self.data_paths.results_path), exist_ok=True)

    def load_test_data(self, X_path):
        """Load test data for evaluation."""
        # Load independent data
        with open(X_path, "rb") as file:
            X = pickle.load(file)

        # Load target data
        y = pd.read_csv(self.data_paths.target_data_path).values

        return X, y

    def evaluate_models(self):
        """Evaluate models and log metrics to MLflow and DagsHub."""
        evaluation_results = []

        # Start MLflow run
        mlflow.set_tracking_uri("https://dagshub.com/<username>/<repo_name>.mlflow")
        mlflow.set_experiment("Model Evaluation")

        # Iterate over feature file paths
        for feature_file in [self.data_paths.Count_Vector, self.data_paths.tfidf_Vector, 
                             self.data_paths.tfidf_maxfeat_1000_Vector, self.data_paths.tfidf_maxfeat_2000_Vector, 
                             self.data_paths.tfidf_maxfeat_3000_Vector]:
            feature_name = os.path.basename(feature_file).split('.')[0]  # Extract feature name
            print(f"Evaluating models for feature: {feature_name}")

            # Load test data
            X, y = self.load_test_data(feature_file)

            # Iterate over saved models
            for model_file in os.listdir(self.data_paths.artifact_path):
                if model_file.endswith(".pkl") and feature_name in model_file:
                    model_path = os.path.join(self.data_paths.artifact_path, model_file)

                    # Load model
                    with open(model_path, "rb") as file:
                        model = pickle.load(file)

                    # Predict and evaluate
                    y_pred = model.predict(X)
                    accuracy = accuracy_score(y, y_pred)
                    recall = recall_score(y, y_pred, zero_division=1)

                    # Log metrics to MLflow
                    with mlflow.start_run(run_name=f"{model_file}"):
                        log_param("feature", feature_name)
                        log_param("model", model_file)
                        log_metric("accuracy", accuracy)
                        log_metric("recall", recall)

                    # Append results
                    evaluation_results.append({
                        "model_name": model_file,
                        "data_name": feature_name,
                        "accuracy": accuracy,
                        "recall": recall
                    })

        # Save evaluation results to CSV
        results_df = pd.DataFrame(evaluation_results)
        results_df.to_csv(self.data_paths.results_path, index=False)
        print("Evaluation completed. Results saved to", self.data_paths.results_path)

if __name__ == "__main__":
    evaluator = ModelEvaluator()
    evaluator.evaluate_models()
