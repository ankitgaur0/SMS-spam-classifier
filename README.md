# SMS Classification(Spam-Ham)

[Ankit Gaur](https://github.com/ankitgaur0)

This project is an SMS spam classification task using machine learning. The dataset consists of labeled SMS messages categorized as either "ham" (non-spam) or "spam." The goal is to create a machine learning model that can accurately predict whether a given SMS is spam or not.

## Project Overview

The project uses a dataset containing SMS messages and their corresponding labels ("ham" or "spam"). The dataset is preprocessed to extract relevant features and then fed into a machine learning model for training and evaluation. The code is implemented using Python and various libraries such as `scikit-learn`, `pandas`, `seaborn`, and `xgboost`.

## Workflow

1. **Data Preprocessing**:
   - The dataset contains the following columns: `v1` (label), `v2` (message text), and several unnamed columns (which are not needed for classification).
   - The features used for classification are `v2` (message text) and `Unnamed: 2`, which is used to store additional derived features from the message text.
   
2. **Text Processing**:
   - The text data in `v2` is preprocessed by removing unwanted characters, performing text normalization, and vectorizing the text using techniques like `CountVectorizer` or `TF-IDF`.
   
3. **Feature Engineering**:
   - Additional features are created by extracting information such as message length, number of characters, etc.

4. **Model Training**:
   - The data is split into training and test sets.
   - Various machine learning models, including XGBoost and others, are trained and evaluated on the dataset.
   - Hyperparameter tuning is done using `GridSearchCV` or `RandomizedSearchCV` for optimal model performance.

5. **Evaluation**:
   - The performance of the model is evaluated using metrics such as accuracy, precision, recall, F1-score, and confusion matrix.

## Dependencies

The following Python packages are used in this project:

- `alembic==1.14.0`
- `altair==5.5.0`
- `Flask==3.1.0`
- `scikit-learn==1.6.0`
- `seaborn==0.13.0`
- `xgboost==2.1.3`
- `pandas==1.5.3`
- `mlflow==2.13.1`
- `joblib==1.4.2`
- `nltk==3.8`
- `numpy==1.24.4`
- `requests==2.32.3`

For the complete list of dependencies, check the `pip list` provided.

## Usage

To run this project locally:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/sms-spam-classification.git
   cd sms-spam-classification




### Key Points:
- This `README.md` provides an overview of the project and includes sections such as the workflow, dependencies, usage, data description, and license.
- You can customize the repository link and installation instructions to fit your specific setup.
- For preprocessing the data, `preprocess_data.py` should include the code for cleaning and preparing the data, while `train_model.py` should contain the model training script.

Let me know if you need help with any specific part of the code!
