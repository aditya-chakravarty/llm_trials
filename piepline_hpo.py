import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import TuningStep
from sagemaker.tuner import HyperparameterTuner, ContinuousParameter, CategoricalParameter, IntegerParameter
from sagemaker.sklearn.estimator import SKLearn
import mlflow
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Set up SageMaker session
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()
region = sagemaker_session.boto_region_name
bucket = sagemaker_session.default_bucket()

# Set up MLflow
mlflow.set_tracking_uri("your_mlflow_tracking_server_uri")
mlflow.set_experiment("Breast_Cancer_Classification_HPO")

# Load and split the data
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Create DataFrames
train_data = pd.DataFrame(X_train, columns=data.feature_names)
train_data['target'] = y_train
test_data = pd.DataFrame(X_test, columns=data.feature_names)
test_data['target'] = y_test

# Save data to S3
train_data.to_csv(f"s3://{bucket}/breast_cancer/train.csv", index=False)
test_data.to_csv(f"s3://{bucket}/breast_cancer/test.csv", index=False)

import argparse
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import joblib

def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--min-samples-split", type=int, default=2)
    args, _ = parser.parse_known_args()

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(os.environ["MLFLOW_EXPERIMENT_NAME"])

    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("min_samples_split", args.min_samples_split)

        # Read data
        train_data = pd.read_csv("/opt/ml/input/data/train/train.csv")
        test_data = pd.read_csv("/opt/ml/input/data/test/test.csv")

        X_train = train_data.drop("target", axis=1)
        y_train = train_data["target"]
        X_test = test_data.drop("target", axis=1)
        y_test = test_data["target"]

        # Train model
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            random_state=42
        )
        model.fit(X_train, y_train)

        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Save model
        joblib.dump(model, os.path.join("/opt/ml/model", "model.joblib"))
        mlflow.sklearn.log_model(model, "model")

    print("Training completed.")


# Create SKLearn estimator
sklearn_estimator = SKLearn(
    entry_point="train.py",
    role=role,
    instance_count=1,
    instance_type="ml.m5.xlarge",
    framework_version="0.23-1",
    py_version="py3",
    hyperparameters={
        "MLFLOW_TRACKING_URI": mlflow.get_tracking_uri(),
        "MLFLOW_EXPERIMENT_NAME": mlflow.get_experiment_by_name("Breast_Cancer_Classification_HPO").experiment_id
    }
)

# Define hyperparameter ranges
hyperparameter_ranges = {
    "n-estimators": IntegerParameter(50, 300),
    "max-depth": IntegerParameter(3, 15),
    "min-samples-split": CategoricalParameter([2, 5, 10])
}

# Create HyperparameterTuner
tuner = HyperparameterTuner(
    sklearn_estimator,
    "accuracy",
    hyperparameter_ranges,
    max_jobs=10,
    max_parallel_jobs=3,
    objective_type="Maximize"
)

# Create TuningStep
tuning_step = TuningStep(
    name="BreastCancerHPO",
    tuner=tuner,
    inputs={
        "train": sagemaker.inputs.TrainingInput(s3_data=f"s3://{bucket}/breast_cancer/train.csv"),
        "test": sagemaker.inputs.TrainingInput(s3_data=f"s3://{bucket}/breast_cancer/test.csv")
    }
)

# Create and run the pipeline
pipeline = Pipeline(
    name="BreastCancerClassificationHPO",
    steps=[tuning_step]
)

pipeline.upsert(role_arn=role)
execution = pipeline.start()

# Wait for the pipeline to complete
execution.wait()

# Get the best performing job
best_job = tuner.best_training_job()
best_job_name = best_job()

# Log the best hyperparameters and metrics to MLflow
with mlflow.start_run(run_name="HPO_Results"):
    best_hyperparameters = sagemaker_session.sagemaker_client.describe_training_job(
        TrainingJobName=best_job_name
    )["HyperParameters"]
    
    mlflow.log_params(best_hyperparameters)
    
    best_model_artifacts = sagemaker_session.sagemaker_client.describe_training_job(
        TrainingJobName=best_job_name
    )["ModelArtifacts"]["S3ModelArtifacts"]
    
    mlflow.log_artifact(best_model_artifacts, "best_model")
    
    best_metrics = sagemaker_session.sagemaker_client.describe_training_job(
        TrainingJobName=best_job_name
    )["FinalMetricDataList"]
    
    for metric in best_metrics:
        mlflow.log_metric(metric["MetricName"], metric["Value"])

print("Pipeline execution completed. Best hyperparameters and metrics logged to MLflow.")



































































