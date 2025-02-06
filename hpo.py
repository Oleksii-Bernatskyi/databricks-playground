import warnings

warnings.filterwarnings("ignore")

import pyspark.sql

import mlflow
import xgboost as xgb
from hyperopt.pyll import scope
from sklearn.model_selection import train_test_split
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)


# Initialize SparkSession
spark = pyspark.sql.SparkSession.builder.getOrCreate()

# Read the data from hive and put it in pandas
spark_df = spark.sql("SELECT * FROM hive_metastore.default.processed_df")
df = spark_df.toPandas()

# Split the data into X and y
X = df.drop("diabetes", axis=1)
y = df["diabetes"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Put data into DMatrix format
train_xgb = xgb.DMatrix(X_train, label=y_train)
valid_xgb = xgb.DMatrix(X_test, label=y_test)


# Define the objective hyperopt function with mlflow logging on(metrics and model artifact)
def objective(params):
    with mlflow.start_run():
        mlflow.autolog()
        # mlflow.set_tag('model', 'xgboost')
        mlflow.log_params(params)
        booster = xgb.train(
            params=params,
            dtrain=train_xgb,
            num_boost_round=1000,
            evals=[(valid_xgb, "valid")],
            early_stopping_rounds=50,
        )
        y_pred = booster.predict(valid_xgb)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)

        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("accuracy", accuracy)

    return {"loss": -accuracy, "status": STATUS_OK}


# Define the search space for hyperopt
search_space = {
    "max_depth": scope.int(hp.quniform("max_depth", 4, 100, 1)),
    "learning_rate": hp.loguniform("learning_rate", -3, 0),
    "reg_alpha": hp.loguniform("reg_alpha", -5, -1),
    "reg_lambda": hp.loguniform("reg_lambda", -6, -1),
    "min_child_weight": hp.loguniform("min_child_weight", -1, 3),
    "n_estimators": scope.int(hp.quniform("n_estimators", 100, 1000, 1)),
    "objective": "binary:hinge",
    "seed": 42,
}

# Run the hyperopt search
best_result = fmin(
    fn=objective,
    space=search_space,
    algo=tpe.suggest,
    max_evals=10,
    trials=Trials(),
    verbose=True,
    show_progressbar=True,
)
