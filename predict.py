import pandas as pd
import boto3
import json
from utils.main_utils import get_preprocessed_data, get_matrices
from config import model_config, aws_config

def check_status(app_name, region_name):
    sage_client = boto3.client('sagemaker', region_name=region_name)
    endpoint_description = sage_client.describe_endpoint(EndpointName=app_name)
    endpoint_status = endpoint_description['EndpointStatus']
    return endpoint_status

def query_endpoint(input_json, app_name, region_name):
    client = boto3.session.Session().client('sagemaker-runtime', region_name)

    response = client.invoke_endpoint(
        EndpointName = app_name,
        Body = input_json,
        ContentType = 'application/json'
        # ContentType = 'application/json; format=pandas-split'
    )

    preds = response['Body'].read().decode('ascii')
    preds = json.loads(preds)
    print(f'Received response : {preds}')
    return preds

if __name__ == "__main__":
    model_file_name = model_config.model_file_name
    region_name = aws_config.region_name
    app_name = aws_config.app_name
    test_file_name = aws_config.test_file_name
    required_num_of_data_points = aws_config.required_num_of_data_points

    print(f'Application status is : {check_status(app_name, region_name)}')

    test_file_path = f'../{test_file_name}'
    df_test = pd.read_csv(test_file_path)
    print("Total datapoints(shape) available in the test set : ", df_test.shape)
    print(f"Considering the first {required_num_of_data_points} datapoints from {test_file_name}.csv")
    df_test = df_test.iloc[0:required_num_of_data_points]
    print(df_test.shape)

    preprocessed_test_file_name = model_config.preprocessed_test_file_name
    df_test = get_preprocessed_data(df_test, preprocessed_test_file_name)

    documenttype_section = model_config.documenttype_section
    df_test = df_test.loc[df_test['TRANSACTIONDOCUMENTTYPE']==documenttype_section]

    tfidfsparse_matirx_test = get_matrices(df_test, model_file_name)
    print(type(tfidfsparse_matirx_test.toarray()))

    input_json = json.dumps(tfidfsparse_matirx_test.toarray().tolist())
    
    predictions = query_endpoint(input_json, app_name, region_name)
