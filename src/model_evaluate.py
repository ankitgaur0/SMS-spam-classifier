import pandas as pd
import mlflow

# File paths
MODEL_PERFORMANCE_CSV = "/home/dev/sms_spam_classifier/artifacts/models/model_performance.csv"
#MLFLOW_TRACKING_URI = "https://dagshub.com/ankitgaur0/SMS-spam-classifier.mlflow"
MLFLOW_TRACKING_URI = "https://ankitgaur0:15e4b3184a1a65a54181295e2940be05847eed1f@dagshub.com/ankitgaur0/SMS-spam-classifier.mlflow"


# MLflow setup
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("SMS Spam Classifier")    

# Load the existing CSV file with model performance data
performance_df = pd.read_csv(MODEL_PERFORMANCE_CSV)

# Log each row of the DataFrame to MLflow
for index, row in performance_df.iterrows():
    with mlflow.start_run():
        mlflow.log_param("model_name", row["model_name"])
        mlflow.log_param("data_name", row["data_name"])
        mlflow.log_metric("accuracy", row["accuracy"])
        mlflow.log_metric("precision", row["precision"])
        mlflow.log_metric("recall", row["recall"])

# Log the entire CSV file as an artifact to MLflow
with mlflow.start_run():
    mlflow.log_artifact(MODEL_PERFORMANCE_CSV)

print("Model performance data tracked in MLflow and DagsHub.")
