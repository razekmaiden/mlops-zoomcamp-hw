import pandas as pd
from prefect import flow, task, get_run_logger
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta
import pickle
import os

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

@task
def read_data(path):
    df = pd.read_parquet(path)
    return df

@task
def prepare_features(df, categorical, train=True):
    logger = get_run_logger()
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        #print(f"The mean duration of training is {mean_duration}")
        logger.info(f"The mean duration of training is {mean_duration}")
    else:
        #print(f"The mean duration of validation is {mean_duration}")
        logger.info(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

@task
def train_model(df, categorical):
    logger = get_run_logger()
    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts) 
    y_train = df.duration.values

    #print(f"The shape of X_train is {X_train.shape}")
    #print(f"The DictVectorizer has {len(dv.feature_names_)} features")
    logger.info(f"The shape of X_train is {X_train.shape}")
    logger.info(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    #print(f"The MSE of training is: {mse}")
    logger.info(f"The MSE of training is: {mse}")
    return lr, dv

@task
def run_model(df, categorical, dv, lr):
    logger = get_run_logger()
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    #print(f"The MSE of validation is: {mse}")
    logger.info(f"The MSE of validation is: {mse}")
    return



@task
def get_paths(date=None):
    logger = get_run_logger()
    train_gap = relativedelta(months=2)
    validation_gap = relativedelta(months=1)
    
    dt = datetime.utcnow()
    if date is not None:
        dt = datetime.strptime(date, '%Y-%m-%d')
    train_date = datetime.strftime(dt - train_gap, '%Y-%m')
    validation_date = datetime.strftime(dt - validation_gap, '%Y-%m')
    
    train_path = f"data/fhv_tripdata_{train_date}.parquet"
    validation_path = f"data/fhv_tripdata_{validation_date}.parquet"
    
    logger.info(f"Train Date: {train_path}")
    logger.info(f"Validation Date: {validation_path}")
    
    return train_path, validation_path, dt.strftime('%Y-%m-%d')

@flow
def main(date=None):
    logger = get_run_logger()
    train_path, val_path, to_save_date = get_paths(date).result()

    categorical = ['PUlocationID', 'DOlocationID']

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical).result()

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False).result()

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()
    run_model(df_val_processed, categorical, dv, lr)
    model_name = f"models/model-{to_save_date}.bin"
    dv_name = f"models/dv-{to_save_date}.bin"

    if not os.path.exists("models"):
        os.makedirs("models")
    logger.info(f"Saving Model Files: {model_name}, {dv_name}")
    with open(model_name, "wb") as f_out:
            pickle.dump(lr, f_out)
    with open(dv_name, "wb") as f_out:
            pickle.dump(dv, f_out)


#main(date="2021-08-15")

from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule
from prefect.flow_runners import SubprocessFlowRunner

DeploymentSpec(
    flow=main,
    name="cron-schedule-deployment",
    flow_runner=SubprocessFlowRunner(),
    tags=["ml"],
    schedule=CronSchedule(
        cron="0 9 15 * *",
        timezone="America/Santiago"),
)