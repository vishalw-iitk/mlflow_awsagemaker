import mlflow.sagemaker as mfs
from config import mlflow_config, aws_config
from config import sagemaker_deployment_config as sgmdply_config

model_uri_params = {
    'experiment_id' : sgmdply_config.experiemnt_id,
    'run_id' : sgmdply_config.run_id
}

image_url_params = {
    'aws_id' : sgmdply_config.aws_id,
    'tag_id' : sgmdply_config.tag_id,
    'region' : aws_config.region_name
}

deployment_params = {
    'app_name' : aws_config.app_name,
    'model_uri' : f"mlruns/{model_uri_params['experiment_id']}/{model_uri_params['run_id']}/artifacts/{mlflow_config.mlflow_model_name}",
    'region_name' : aws_config.region_name,
    'mode' : sgmdply_config.mode,
    'execution_role_arn' : sgmdply_config.execution_role_arn,
    'image_url' : image_url_params['aws_id'] + '.dkr.ecr.' + \
                image_url_params['region'] + '.amazonaws.com/mlflow-pyfunc:' + \
                image_url_params['tag_id']
}

mfs.deploy(**deployment_params)
