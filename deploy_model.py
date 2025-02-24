import sagemaker
from sagemaker.sklearn.model import SKLearnModel

# Initialize SageMaker session
sagemaker_session = sagemaker.Session()

# Define IAM role for SageMaker execution
role = "arn:aws:iam::977099010778:role/service-role/AmazonSageMaker-ExecutionRole-20250223173179"

# Define S3 bucket and correct model path
bucket_name = "wine-quality-bucket1234"
model_path = f"s3://{bucket_name}/output/sagemaker-scikit-learn-2025-02-24-07-01-56-985/output/model.tar.gz"

# Create model object
sklearn_model = SKLearnModel(
    model_data=model_path,  # Use full path to model.tar.gz
    role=role,
    entry_point="train_model.py",
    framework_version="0.23-1",
    py_version="py3"
)

# Deploy model to an endpoint
predictor = sklearn_model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",
)

