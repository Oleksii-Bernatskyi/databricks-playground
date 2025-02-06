import warnings

warnings.filterwarnings("ignore")

import mlflow
from mlflow.models.model import get_model_info
from mlflow.models import infer_signature, set_signature
import pyspark.sql
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split

# Initialize SparkSession
spark = pyspark.sql.SparkSession.builder.getOrCreate()

# Read the data from hive and put it in pandas
spark_df = spark.sql("SELECT * FROM hive_metastore.default.processed_df")
df = spark_df.toPandas()

# Prepare df's to use
X = df.drop("diabetes", axis=1)
y = df["diabetes"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Select experiment
client = MlflowClient()
experiment = client.get_experiment("b438e5228f2c42c0bf17bcc05011775d")

# Select run with the best accuracy
best_run = client.search_runs(
    experiment_ids=experiment.experiment_id,
    run_view_type=ViewType.ACTIVE_ONLY,
    order_by=["metrics.accuracy DESC"],
    max_results=1,
)
model_uri = f"runs:/{best_run[0].info.run_id}/model"

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(model_uri)

# Get the signature of the model
signature = infer_signature(X_test, loaded_model.predict(X_test))

# set the signature for the logged model
set_signature(model_uri, signature)

# now when you load the model again, it will have the desired signature
assert get_model_info(model_uri).signature == signature


# Register model
catalog = "llm_pj"
schema = "default"
model_name = "xgboost_model"
mlflow.set_registry_uri("databricks-uc")
mlflow.register_model(model_uri, f"{catalog}.{schema}.{model_name}")
