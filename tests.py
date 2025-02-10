from databricks.sdk.runtime import *

import pytest
import pyspark
import pandas as pd
import load_preprocess_data

table_name = "diabetes_prediction_india"


@pytest.fixture(scope="function")
def df():
    return load_preprocess_data.load_data(table_name)


def preprocess(df):
    return load_preprocess_data.custom_preprocess(df)


def test_data_presence():
    # Check if the table exists
    assert spark.catalog.tableExists(f"llm_pj.default.{table_name}") is True


def test_data_load(df):
    # Check if df not empty
    assert df.shape[0] > 0
    # Check whether all features are there
    assert df.shape[1] == 27
    # Check if the data type is correct
    assert type(df) == type(pd.DataFrame())


def test_preprocess(df):
    X, y = preprocess(df)
    assert isinstance(df, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert X.shape[1] == 26
    assert y.shape[0] == X.shape[0]
    assert X.shape[0] > 0
    assert y.shape[0] > 0


def test_scale(df):
    X, y = preprocess(df)
    scaled_df = load_preprocess_data.scale(X, y)
    assert type(scaled_df) == type(pd.DataFrame())
    assert scaled_df.shape[1] == 27
    assert scaled_df.shape[0] == X.shape[0]
    assert scaled_df["BMI"].max() < X["BMI"].max()

