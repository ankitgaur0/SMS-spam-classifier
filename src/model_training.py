import os
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from dataclasses import dataclass
import pandas as pd

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

class ModelTrainer:
    def __init__(self):
        self.data_paths = DataPaths()

        self.models = {
            "svc": SVC(kernel="sigmoid", gamma=1.0),
            "knc": KNeighborsClassifier(),
            "mnb": MultinomialNB(),
            "dtc": DecisionTreeClassifier(max_depth=5),
            "lrc": LogisticRegression(solver='liblinear', penalty='l1'),
            "rfc": RandomForestClassifier(n_estimators=50, random_state=2),
            "abc": AdaBoostClassifier(n_estimators=50, random_state=2),
            "bc": BaggingClassifier(n_estimators=50, random_state=2),
            "etc": ExtraTreesClassifier(n_estimators=50, random_state=2),
            "gbc": GradientBoostingClassifier(n_estimators=50, random_state=2),
            "xgb": XGBClassifier(n_estimators=50, random_state=2)
        }

        # Ensure artifact directory exists
        os.makedirs(self.data_paths.artifact_path, exist_ok=True)

    def load_and_split_data(self, X_path):
        # Load independent data
        with open(X_path, "rb") as file:
            X = pickle.load(file)

        # Load target data
        y = pd.read_csv(self.data_paths.target_data_path).values

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
        return X_train, X_test, y_train, y_test

    def train_and_save_models(self):
        overall_results = []

        # Iterate through feature file paths
        for feature_file in [self.data_paths.Count_Vector, self.data_paths.tfidf_Vector, 
                             self.data_paths.tfidf_maxfeat_1000_Vector, self.data_paths.tfidf_maxfeat_2000_Vector, 
                             self.data_paths.tfidf_maxfeat_3000_Vector]:
            
            feature_name = os.path.basename(feature_file).split('.')[0]  # Extract the name without the extension
            print(f"Training models using feature: {feature_name}")
            X_train, X_test, y_train, y_test = self.load_and_split_data(feature_file)
            results = []

            for model_name, model in self.models.items():
                print(f"Training {model_name}...")
                model.fit(X_train, y_train)

                # Save the model using the feature name
                model_path = os.path.join(self.data_paths.artifact_path, f"{model_name}_{feature_name}.pkl")
                with open(model_path, "wb") as model_file:
                    pickle.dump(model, model_file)

                # Evaluate on the test set
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=1)
                recall = recall_score(y_test, y_pred, zero_division=1)

                results.append({
                    "model_name": model_name,
                    "data_name": feature_name,
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall
                })

            overall_results.extend(results)

        # Save overall results as a DataFrame
        results_df = pd.DataFrame(overall_results)
        results_df_path = os.path.join(self.data_paths.artifact_path, "model_performance.csv")
        results_df.to_csv(results_df_path, index=False)

        print("Training completed. Results saved.")

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train_and_save_models()
