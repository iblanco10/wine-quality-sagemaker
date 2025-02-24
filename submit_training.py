import sagemaker
from sagemaker.sklearn.estimator import SKLearn

# Initialize SageMaker session
sagemaker_session = sagemaker.Session()

# Define IAM role for SageMaker execution
role = "arn:aws:iam::977099010778:role/service-role/AmazonSageMaker-ExecutionRole-20250223T173179"

# Define S3 bucket and script paths
bucket_name = "wine-quality-bucket1234"
script_path = "train_model.py"  # Use local file instead of S3
s3_data_path = f"s3://{bucket_name}/"  # Location of training data
output_path = f"s3://{bucket_name}/output/"  # Where model artifacts will be stored

# Define the Scikit-Learn Estimator
sklearn_estimator = SKLearn(
    entry_point=script_path,  # Path to the training script in S3
    role=role,
    instance_type="ml.m5.large",  # Adjust instance type as needed
    instance_count=1,
    framework_version="0.23-1",
    py_version="py3",
    output_path=output_path,  # Store model artifacts in S3
)

# Start the training job
sklearn_estimator.fit({"train": s3_data_path})

print("Training job submitted successfully!")

