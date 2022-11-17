class mlflow_config:
    experiment_name = 'Endo_doctype_clssfify_with_xgb' # edit
    run_name = 'XGB model exp with predict and deploy code just added' # edit
    mlflow_model_name = 'model_file'

class model_config:
    model_file_name = 'model_nov22_5' # edit
    n_estimators = 100 # edit
    data_input_file_name = 'df_cleaned_combined.csv'
    preprocessed_file_name = 'df_cleaned_combined_new.csv'
    documenttype_section = 'ENDORSEMENT'
        
    preprocessed_test_file_name = 'df_test_cleaned_combined_new.csv'

class aws_config:
    region_name = 'ap-south-1' # edit
    app_name = 'email-classification-model'
    test_file_name = 'test_set.csv'
    required_num_of_data_points = 20

class sagemaker_deployment_config:
    experiemnt_id = '1' # edit
    run_id = '5a841869486b463cbbcc7f7b18639c18' # edit
    aws_id = '451632245592' # edit : cmd : aws sts get-caller-identity --query Account --output text
    execution_role_arn = 'arn:aws:iam::451632245592:role/aws-sagemaker-for-model-deployment' # edit
    tag_id = '1.30.0'
    mode = 'create'
