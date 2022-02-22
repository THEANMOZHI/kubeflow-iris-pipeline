# kfp
import kfp
from kfp import dsl
from kfp.v2 import compiler
from kfp.v2.dsl import (Artifact, Dataset, Input, InputPath, Model, Output,
                        OutputPath, ClassificationMetrics, Metrics, component)
from kfp.v2.google.client import AIPlatformClient

# gcp
from google.cloud import aiplatform
from google_cloud_pipeline_components import aiplatform as gcc_aip
from google.cloud import storage
from googleapiclient import discovery
from oauth2client.client import GoogleCredentials

# i/o
from typing import NamedTuple
from io import StringIO
import os

# pandas & sklearn
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# definites
PROJECT_ID="kedro-kubeflow-334417"
BUCKET_NAME="gs://diab-gsbucket"
REGION="us-central1"
PIPELINE_NAME = "diabetes_pipeline"
PIPELINE_ROOT = f"{BUCKET_NAME}/pipeline_root/"
PIPELINE_ROOT


# data component
@component(base_image='python:3.7',
           packages_to_install=['numpy', 'pandas', 'google-cloud-storage==1.43.0', 'google-cloud-aiplatform==1.0.0',
                         'kubeflow-metadata', 'scikit-learn', 'gcsfs==2021.11.1'])
def data_component(bucket: str, value: float, marker: int) -> int:
    import kfp
    import pandas as pd
    import sklearn
    from sklearn.model_selection import train_test_split
    from kfp.v2.google.client import AIPlatformClient
    from google.cloud import storage
    
    # read data from gcs bucket
    data = pd.read_csv('gs://iris-kfp/iris.csv', index_col=False) 
    
    # data preprocessing
    # normalizing data
    df = data
    for column in df.columns:
        df[column] = (df[column] - df[column].mean()) / df[column].std() 
    data = df
    # dependent and independent data sets
    train_data = data.drop('class',axis=1)
    test_data = data['class']    
    
    # test-train  data split
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(train_data, test_data, test_size = value, random_state=42)
    X_train = X_train.to_csv()
    X_test = X_test.to_csv()
    y_train = y_train.to_csv()
    y_test = y_test.to_csv()
    
    # storage client
    storage_client = storage.Client()
    bucket = storage_client.get_bucket('iris-kfp')
    # blobs
    d1 = bucket.blob('X_train.csv')
    d2 = bucket.blob('X_test.csv')
    d3 = bucket.blob('y_train.csv')
    d4 = bucket.blob('y_test.csv')
    
    # uploading train-test datasets into gcs bucket
    d1.upload_from_string(f'{X_train}.csv', 'text/csv')
    d2.upload_from_string(f'{X_test}.csv', 'text/csv')
    d3.upload_from_string(f'{y_train}.csv', 'text/csv')
    d4.upload_from_string(f'{y_test}.csv', 'text/csv')
    
    # setting flag
    df1 = pd.read_csv("gs://iris-kfp/X_train.csv", index_col=0)
    df2 = pd.read_csv("gs://iris-kfp/X_test.csv", index_col=0)
    df3 = pd.read_csv("gs://iris-kfp/y_train.csv", index_col=0)
    df4 = pd.read_csv("gs://iris-kfp/y_test.csv", index_col=0)
    
    if(df1.empty == True and df2.empty == True and df3.empty == True and df4.empty == True):
        marker = 1
    else:
        marker = 0

    return marker



# model component
@component(base_image='python:3.7',
        packages_to_install=['numpy', 'pandas', 'google-cloud-storage==1.43.0', 'google-cloud-aiplatform==1.0.0',
                            'kubeflow-metadata', 'scikit-learn', 'gcsfs==2021.11.1'])
def model_component(bucket:str, xtrain:str, ytrain:str, xtest:str, ytest:str) -> float:
    import pandas as pd    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    
    # read test-train data sets from GCS bucket
    X_train = pd.read_csv(f'{bucket}/{xtrain}.csv', sep=",")
    y_train = pd.read_csv(f'{bucket}/{ytrain}.csv', sep=",")
    X_test = pd.read_csv(f'{bucket}/{xtest}.csv', sep=",")
    y_test = pd.read_csv(f'{bucket}/{ytest}.csv', sep=",")    
        
    # train model
    model = RandomForestClassifier(max_depth=2, random_state=3)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # find accuracy
    accuracy = accuracy_score(y_test, predictions)
    
    return accuracy


# when accuracy >= threshold
@component(base_image='python:3.7',
           packages_to_install=['numpy', 'pandas', 'google-cloud-storage==1.43.0', 'google-cloud-aiplatform==1.0.0',
                         'kubeflow-metadata', 'scikit-learn'])
def true_component(accuracy:float) -> None:
    print(f'Yes!! {accuracy} is the Accuracy and its greater than the threshold')


# when accuracy < thrershold
@component(base_image='python:3.7',
           packages_to_install=['numpy', 'pandas', 'google-cloud-storage==1.43.0', 'google-cloud-aiplatform==1.0.0',
                         'kubeflow-metadata', 'scikit-learn'])
def false_component(accuracy:float) -> None:
    print(f'No. {accuracy} is the Accuracy and its smaller than the threshold')


# pipeline
@kfp.dsl.pipeline(name = "iris-pipeline",
                  pipeline_root = PIPELINE_ROOT)
def iris_pipeline(
    display_name: str="iris-kfp",
    project: str = PROJECT_ID,
    gcp_region: str = "us-central1",
    api_endpoint: str = "us-central1-aiplatform.googleapis.com",
    marker: int = 0,
    test_train_split_ratio: float = 0.3,
    accuracy_threshold: float = 0.5,
    bucket: str = "gs://iris-kfp"
) -> None:
        
    # initiating data component
    data_op = data_component(bucket, test_train_split_ratio, marker)

    # initiating model component
    with dsl.Condition(data_op.output == 1):
        model_op = model_component(bucket, "X_train", "y_train", "X_test", "y_test")
    
        with dsl.Condition(model_op.output >= accuracy_threshold, name="accuracy>=50"):
            true_component(model_op.output)
        with dsl.Condition(model_op.output < accuracy_threshold, name="accuracy<50"):
            false_component(model_op.output)


compiler.Compiler().compile(
    pipeline_func=iris_pipeline, package_path="iris_kfp.json"
)

api_client = AIPlatformClient(
    project_id=PROJECT_ID,
    region=REGION,
)

response = api_client.create_run_from_job_spec(
    "iris_kfp.json", pipeline_root=PIPELINE_ROOT,
)