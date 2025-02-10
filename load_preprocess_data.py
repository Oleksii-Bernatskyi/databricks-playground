from databricks.sdk.runtime import *

import warnings

warnings.filterwarnings("ignore")

import pandas as pd
from typing import Tuple
from sklearn.preprocessing import StandardScaler


def load_data(table_name_: str) -> pd.DataFrame:
    # Read the data from hive and put it in pandas
    spark_df = spark.read.table(f"llm_pj.default.{table_name_}")
    df = spark_df.toPandas()
    return df


# Takes pd.Dataframe, transfroms categorical features into numerical features, return those new numerical features as well as target
def custom_preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    df["gender"] = df["Gender"].map({"Male": 1, "Female": 0, "Other": 2})
    df["family_history"] = df["Family_History"].map({"Yes": 1, "No": 0})
    df["physical_activity"] = df["Physical_Activity"].map(
        {"High": 2, "Medium": 1, "Low": 0}
    )
    df["diet_type"] = df["Diet_Type"].map(
        {"Non-Vegetarian": 0, "Vegetarian": 1, "Vegan": 2}
    )
    df["smoking_status"] = df["Smoking_Status"].map(
        {"Never": 0, "Former": 1, "Current": 2}
    )
    df["alcohol_intake"] = (
        df["Alcohol_Intake"].fillna(0).map({"Moderate": 1, "High": 2, 0: 0})
    )
    df["stress_level"] = df["Stress_Level"].map({"High": 2, "Medium": 1, "Low": 0})
    df["hypertension"] = df["Hypertension"].map({"Yes": 1, "No": 0})
    df["urban_rural"] = df["Urban_Rural"].map({"Urban": 1, "Rural": 0})
    df["health_insurance"] = df["Health_Insurance"].map({"Yes": 1, "No": 0})
    df["regural_checkups"] = df["Regular_Checkups"].map({"Yes": 1, "No": 0})
    df["chron_cond_med"] = df["Medication_For_Chronic_Conditions"].map(
        {"Yes": 1, "No": 0}
    )
    df["polycystic"] = df["Polycystic_Ovary_Syndrome"].map({"Yes": 1, "No": 0, "0": 0})
    df["thyroid"] = df["Thyroid_Condition"].map({"Yes": 1, "No": 0})

    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    list_of_features = list(df.select_dtypes(include=numerics).columns)

    df["diabetes"] = df["Diabetes_Status"].map({"Yes": 1, "No": 0})

    target_col = "diabetes"

    procesed_df = df[list_of_features + [target_col]]

    X_ = procesed_df.drop("diabetes", axis=1)
    y_ = procesed_df["diabetes"]

    return X_, y_


# Takes pd.DataFrame, apply kinda one hot encoding(get_dummies) to categorical features, then return numerical features and target
def naive_preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    X_ = df.drop("Diabetes_Status", axis=1)
    y_ = df["Diabetes_Status"].map({"Yes": 1, "No": 0})

    X_ = pd.get_dummies(X, drop_first=True)

    return X_, y_


# Takes features and target, applies scaler to features and does train-test split
def scale(X_: pd.DataFrame, y_: pd.Series) -> pd.DataFrame:
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(X_)
    df_scaled = pd.DataFrame(df_scaled, columns=X_.columns)
    df_scaled["diabetes"] = y_
    return df_scaled


table_name = "diabetes_prediction_india"
df_ = load_data(table_name)

# X, y = naive_preprocess(df_)
X, y = custom_preprocess(df_)

processed_df = scale(X, y)
print(processed_df.shape)
processed_df = spark.createDataFrame(processed_df)


processed_df.write.format("delta").mode("overwrite").saveAsTable("processed_df")
