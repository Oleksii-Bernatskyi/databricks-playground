import warnings

warnings.filterwarnings("ignore")

import pyspark.sql
from load_preprocessed_data import load_data
from sklearn.model_selection import train_test_split


# Read the data from delta table and put it in pandas
table_name = 'processed_df'
df = load_data(table_name)

# Prepare df's to use
X = df.drop("diabetes", axis=1)
y = df["diabetes"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


import mlflow
from mlflow.tracking import MlflowClient

# Initialize MLflow Client
client = MlflowClient()

# Define the model name (as registered in MLflow)
model_name = "xgboost_model"

import mlflow.pyfunc

model_version = 2

loaded_model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{model_name}/{model_version}"
)

preds = loaded_model.predict(X_test)
print(preds)
