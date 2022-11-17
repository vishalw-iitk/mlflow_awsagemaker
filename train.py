import pandas as pd
from utils.main_utils import *
from config import model_config, mlflow_config
import mlflow

mlflow.set_experiment(mlflow_config.experiment_name)

with mlflow.start_run(run_name = mlflow_config.run_name) as run:
    '''Load the full dataset'''
    data_input_file_name = model_config.data_input_file_name
    model_file_name = model_config.model_file_name
    mlflow.log_param('data_input_file_name', data_input_file_name) # log model_input_file_name
    df = pd.read_csv(f'../{data_input_file_name}')
    print("Column headers : ", df.columns)

    '''Perform data preprocessing'''
    preprocessed_file_name = model_config.preprocessed_file_name
    mlflow.log_param('preprocessed_file_name', preprocessed_file_name) # log preprocess_file_name
    df = get_preprocessed_data(df, preprocessed_file_name)

    '''Train-Test split'''
    documenttype_section = model_config.documenttype_section
    mlflow.log_param('documenttype_section',documenttype_section) # log documenttype_section
    X_train, y_train, X_test, y_test = df_spliter_into_train_test(df, documenttype_section)

    '''Get tf-idf vectorizer and train matrix'''
    tfidf_vectorizer, tfidfsparse_matirx_train = get_tfidf_vectorizer(X_train, model_file_name)

    '''Test matrix for evaluation'''
    tfidfsparse_matirx_test = get_matrices(X_test, model_file_name)

    '''Resolve class imbalance'''
    tfidfsparse_matirx_train, y_train = resolve_class_imbalance(tfidfsparse_matirx_train, y_train)

    '''Model training'''
    mlflow.log_param('model_file_name', model_file_name) # log model name
    model_params = {
        'n_estimators' : model_config.n_estimators
    }
    mlflow.log_param('num_estimator', model_params['n_estimators']) # log n_estimators
    model = model_training(tfidfsparse_matirx_train, y_train, tfidfsparse_matirx_test, y_test, model_file_name, model_params)
    mlflow.sklearn.log_model(model, model_config.mlflow_model_name) # log model in mlflow

    '''Get model predictions on train and test set'''
    y_pred_train = get_model_predictions(model, tfidfsparse_matirx_train)
    y_pred_test = get_model_predictions(model, tfidfsparse_matirx_test)

    '''Get model evaluation loss'''
    model_train_loss = get_model_loss(y_train, y_pred_train)
    mlflow.log_param('model_train_loss', model_train_loss) # log model training loss in mlflow
    model_eval_loss = get_model_loss(y_test, y_pred_test)
    mlflow.log_param('model_eval_loss', model_eval_loss) # log model testing loss in mlflow

    '''Get performance metric for train set'''
    tr_precision, tr_recall, tr_fscore, tr_accuracy = model_performance(model, y_train, y_pred_train)
    ts_precision, ts_recall, ts_fscore, ts_accuracy = model_performance(model, y_test, y_pred_test)
    # log precision, recall, fscore and accuracy for train and test set in mlflow
    mlflow.log_metric('train_precision', tr_precision)
    mlflow.log_metric('train_recall', tr_recall)
    mlflow.log_metric('train_fscore', tr_fscore)
    mlflow.log_metric('train_accuracy', tr_accuracy)

    mlflow.log_metric('test_precision', ts_precision)
    mlflow.log_metric('test_recall', ts_recall)
    mlflow.log_metric('test_fscore', ts_fscore)
    mlflow.log_metric('test_accuracy', ts_accuracy)

    run_id = run.info.run_uuid
    experiment_id = run.info.experiment_id
    artifact_uri = mlflow.get_artifact_uri()
    
    print('run_id', run_id)
    print('experiment_id', experiment_id)
    print('artifact_uri : ', artifact_uri)

    mlflow.end_run()
