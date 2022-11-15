from utils.preprocess_utils import *
import pandas as pd
import os
import pickle
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def get_preprocessed_data(df, preprocessed_file_name):
    if os.path.exists(f'../{preprocessed_file_name}'):
        df = pd.read_csv(f'../{preprocessed_file_name}')
        print("Complete processed data : shape : ", df.shape)
        return df
    df['new_cleaned_data'] = df['text_content'].apply(lambda x: x.lower())
    df['new_cleaned_data'] = df['new_cleaned_data'].apply(remove_specific_set)
    df['new_cleaned_data'] = df['new_cleaned_data'].apply(replace_emails)
    df['new_cleaned_data'] = df['new_cleaned_data'].apply(replace_https)
    df['new_cleaned_data'] = df['new_cleaned_data'].apply(replace_only_numbers)
    df['new_cleaned_data'] = df['new_cleaned_data'].apply(retain_chars)
    df['new_cleaned_data'] = df['new_cleaned_data'].apply(remove_stopwords)
    df['new_cleaned_data'] = df['new_cleaned_data'].apply(lambda x: lemmatizer.lemmatize(x))
    
    lemmatized_relevant_english_words = set(lemmatized_english_words).intersection(' '.join(df['new_cleaned_data'].values).split())

    df['new_cleaned_data'] = df['new_cleaned_data'].apply(lambda x:remove_crap_text(x, lemmatized_relevant_english_words))
    df['new_cleaned_data'] = df['new_cleaned_data'].apply(retain_non_nums)
    df['new_cleaned_data'] = df['new_cleaned_data'].apply(retain_more_than_two_chars)
    df['new_cleaned_data'] = df['new_cleaned_data'].apply(merge_repeated_words)
    df['new_cleaned_data'] = df['new_cleaned_data'].apply(retain_pos_ents)
    df['new_cleaned_data_word_count'] = df['new_cleaned_data'].apply(lambda x:len(set(x.split())))

    df = df.loc[(df['new_cleaned_data_word_count']>=5)]

    df.to_csv(f'../{preprocessed_file_name}', index=False)

    print("Complete processed data : shape : ", df.shape)
    return df

def df_spliter_into_train_test(df, documenttype_section):
    '''Train-Test split'''
    df_train, df_test = train_test_split(df.loc[df['TRANSACTIONDOCUMENTTYPE']==documenttype_section], test_size=0.15, stratify = df.loc[df['TRANSACTIONDOCUMENTTYPE']==documenttype_section]['Updated_TRANSACTIONREQUESTTYPE'], random_state=42)
    y_train = df_train['Updated_TRANSACTIONREQUESTTYPE']
    y_test = df_test['Updated_TRANSACTIONREQUESTTYPE']
    X_train = df_train.drop('Updated_TRANSACTIONREQUESTTYPE', axis=1)
    X_test = df_test.drop('Updated_TRANSACTIONREQUESTTYPE', axis=1)
    print("train - test split completed")
    return X_train, y_train, X_test, y_test

def get_tfidf_vectorizer(X_train):
    '''Get tf-idf vectorizer and train matrix'''
    tfidf_vectorizer = TfidfVectorizer()
    print(X_train.shape)
    tfidfsparse_matirx_train = tfidf_vectorizer.fit_transform(X_train['new_cleaned_data'])
    print("Tfidf vecotrized generated", tfidfsparse_matirx_train.shape)
    return tfidf_vectorizer, tfidfsparse_matirx_train

def get_matrices(tfidf_vectorizer, X_test):
    '''Test matrix for evaluation'''
    print(X_test.shape)
    tfidfsparse_matirx = tfidf_vectorizer.transform(X_test['new_cleaned_data'])
    print(tfidfsparse_matirx.shape)
    print("tdidf matrix created")
    return tfidfsparse_matirx

def resolve_class_imbalance(tfidfsparse_matirx_train, y_train):
    '''Resolve class imbalance'''
    cls_imb = SMOTE()
    print("before class imbalance : ", tfidfsparse_matirx_train.shape, y_train.shape)
    tfidfsparse_matirx_train, y_train_new = cls_imb.fit_resample(tfidfsparse_matirx_train, y_train)
    print("after class imbalance : ", tfidfsparse_matirx_train.shape, y_train_new.shape)
    print('Class imbalancement worked upon')
    return tfidfsparse_matirx_train, y_train_new

def model_training(tfidfsparse_matirx_train, y_train_new, tfidfsparse_matirx_test, y_test, model_file_name, model_params):
    '''Model training'''
    print(tfidfsparse_matirx_train.shape, y_train_new.shape, tfidfsparse_matirx_test.shape, y_test.shape)
    model_path = f'xgbmodels/{model_file_name}'
    n_estimators = model_params['n_estimators']
    clf = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                    colsample_bynode=1, colsample_bytree=0.9, gamma=0, gpu_id=-1,
                    importance_type='gain', interaction_constraints='',
                    learning_rate=0.07, max_delta_step=0, max_depth=5,
                    min_child_weight=1, monotone_constraints='()',
                    n_estimators=n_estimators, n_jobs=6, nthread=1, num_leaves=54,
                    num_parallel_tree=1, random_state=0,
                    reg_alpha=0, reg_lambda=1, scale_pos_weight=1, silent=True,
                    subsample=0.9, tree_method='exact', validate_parameters=1,
                    verbosity=1, use_label_encoder=True)
    if os.path.exists(model_path):
        clf.load_model(model_path)
        return clf
    early_stopping_rounds = 30
    callbacks = [xgboost.callback.EarlyStopping(rounds=early_stopping_rounds, save_best=True, metric_name='merror', maximize=False)]
    booster = clf.fit(tfidfsparse_matirx_train, y_train_new,
    #         xgb_model = model_path,
            eval_set = [
                (tfidfsparse_matirx_train, y_train_new),
                (tfidfsparse_matirx_test, y_test)
            ],
            callbacks=callbacks,
            verbose = 2,
            eval_metric=['mlogloss', 'merror'],
    )
    booster.save_model(model_path)
    print("model trained")
    return booster

def get_model_predictions(model, tfidfsparse_matirx):
    '''Get model predictions on train and test set'''
    y_pred = model.predict(tfidfsparse_matirx)
    print('predictions : ', y_pred)
    return y_pred

def get_model_loss(y, y_pred):
    '''Get model evaluation loss'''
    le = LabelEncoder()
    y = le.fit_transform(y)
    y_pred = le.transform(y_pred)
    model_loss = mse(y_pred, y)
    print('model loss : mse : ', model_loss)
    return model_loss

def model_performance(model, y, y_pred):
    '''Get performance metric for train set'''
    # y = y.values
    # print(y)
    # print(y_pred)
    accuracy = accuracy_score(y_pred, y)
    print(accuracy)
    precision = precision_score(y_pred, y, average='weighted')
    recall = recall_score(y_pred, y, average='weighted')
    fscore = f1_score(y_pred, y, average='weighted')
    print(f'accuracy : {accuracy}\n\
precision : {precision}\n\
recall : {recall}\n\
fscore : {fscore}\
')
    return precision, recall, fscore, accuracy
