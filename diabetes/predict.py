import warnings
warnings.filterwarnings('ignore')

import pyspark.sql

import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
import seaborn as sns
from scipy import stats
from hyperopt.pyll import scope
import matplotlib.pyplot as plt
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


# Initialize SparkSession 
spark = pyspark.sql.SparkSession.builder.getOrCreate()

# Read the data from hive and put it in pandas
spark_df = spark.sql("SELECT * FROM hive_metastore.default.processed_df")
df = spark_df.toPandas()

# Prepare df's to use
X = df.drop('diabetes', axis=1)
y = df['diabetes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Select experiment
client = MlflowClient()
experiment = client.get_experiment('b438e5228f2c42c0bf17bcc05011775d')

# Select run with the best accuracy
best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        order_by=["metrics.accuracy DESC"],
        max_results=1,
        run_view_type=ViewType.
    )
model_uri=f"runs:/{best_run[0].info.run_id}/model"

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(model_uri)

# Predict on a Pandas DataFrame.
preds = loaded_model.predict(pd.DataFrame(X_test))

print(preds[:100])

