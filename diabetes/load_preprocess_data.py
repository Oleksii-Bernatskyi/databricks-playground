import warnings
warnings.filterwarnings('ignore')

import pyspark.sql
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Initialize SparkSession 
spark = pyspark.sql.SparkSession.builder.getOrCreate()

# Read the data from hive and put it in pandas
spark_df = spark.sql("SELECT * FROM hive_metastore.default.diabetes_prediction_india")
df = spark_df.toPandas()

# Takes features and target, applies scaler to features and does train-test split
def scale(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(X)
    df_scaled = pd.DataFrame(df_scaled, columns=X.columns)
    df_scaled['diabetes'] = y
    return df_scaled

# Takes pd.Dataframe, transfroms categorical features into numerical features, return those new numerical features as well as target
def custom_preprocess(df: pd.DataFrame) -> [pd.DataFrame, pd.Series]:
    df['gender'] = df['Gender'].map({'Male':1, 'Female':0, 'Other':2})
    df['family_history'] = df['Family_History'].map({'Yes':1, 'No':0})
    df['physical_activity'] = df['Physical_Activity'].map({'High':2, 'Medium':1, 'Low':0})
    df['diet_type'] = df['Diet_Type'].map({'Non-Vegetarian':0, 'Vegetarian':1, 'Vegan':2})
    df['smoking_status'] = df['Smoking_Status'].map({'Never':0, 'Former':1, 'Current':2})
    df['alcohol_intake'] = df['Alcohol_Intake'].fillna(0).map({'Moderate':1, 'High':2, 0:0})
    df['stress_level'] = df['Stress_Level'].map({'High':2, 'Medium':1, 'Low':0})
    df['hypertension'] = df['Hypertension'].map({'Yes':1, 'No':0})
    df['urban_rural'] = df['Urban_Rural'].map({'Urban':1, 'Rural':0})
    df['health_insurance'] = df['Health_Insurance'].map({'Yes':1, 'No':0})
    df['regural_checkups'] = df['Regular_Checkups'].map({'Yes':1, 'No':0})
    df['chron_cond_med'] = df['Medication_For_Chronic_Conditions'].map({'Yes':1, 'No':0})
    df['polycystic'] = df['Polycystic_Ovary_Syndrome'].map({'Yes':1, 'No':0, '0':0})
    df['thyroid'] = df['Thyroid_Condition'].map({'Yes':1, 'No':0})

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    list_of_features = list(df.select_dtypes(include=numerics).columns)

    df['diabetes'] = df['Diabetes_Status'].map({'Yes':1, 'No':0})

    target_col = 'diabetes'

    procesed_df = df[list_of_features + [target_col]]

    X = procesed_df.drop('diabetes', axis=1)
    y = procesed_df['diabetes']

    return X, y

# Takes pd.DataFrame, apply kinda one hot encoding(get_dummies) to categorical features, then return numerical features and target
def naive_preprocess(df: pd.DataFrame) -> [pd.DataFrame, pd.Series]:
    X = df.drop('Diabetes_Status', axis=1)
    y = df['Diabetes_Status'].map({'Yes':1, 'No':0})

    X = pd.get_dummies(X, drop_first=True)

    return X, y

# Apply preprocess and train-test split
# X, y = naive_preprocess(df)
X, y = custom_preprocess(df)
processed_df = scale(X, y)

display(processed_df.shape)


spark.createDataFrame(processed_df).write.mode("overwrite").saveAsTable("hive_metastore.default.processed_df2")

