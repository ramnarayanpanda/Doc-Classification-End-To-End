import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from matplotlib import pyplot as plt
import pickle
import os
import argparse 
import logging
from tqdm import tqdm
import seaborn as sns
from utils.common import create_directory, read_yaml, write_yaml, create_text_file, save_pickle_file, \
                         read_pickle_file, read_txt_file
from utils.data_loader import doc_classifier_dataloader
from utils.metrics_plots import get_metrics, save_graphs_ML, save_graphs_DL
import mlflow 
import shutil


    

def naive_bayes_model(config_path, param_path, X_train, y_train, X_test, y_test):
    config = read_yaml(config_path)
    params = read_yaml(param_path)
        
    model_params = params['models']['ML']['naive_bayes']['params']
    mnb = MultinomialNB(**model_params)
    # pipe = Pipeline(steps=[('mnb', mnb)])
    # mnb_grid = GridSearchCV(pipe, scoring='f1_weighted', refit=True, cv=cv, verbose=5, n_jobs=-1)
    
    mnb.fit(X_train, y_train) 
    y_pred = mnb.predict(X_test) 
        
    unique_categories = [i for i in range(len(config['artifacts']['INPUT_CLASSES']))]
    metric_dct = get_metrics(y_test, y_pred, unique_categories)
    metric_dct['model_name'] = 'Naive Bayes'
        
    whole_model_name = metric_dct['model_name'] 
    test_accuracy = metric_dct['accuracy']
    graphs = save_graphs_ML(metric_dct, whole_model_name, unique_categories, test_accuracy)
    
    return {'params': mnb.get_params(), 
            'metrics': metric_dct, 
            'artifacts':{'model': mnb}.update(graphs)}
    


def rf_model(config_path, param_path, X_train, y_train, X_test, y_test):
    config = read_yaml(config_path)
    params = read_yaml(param_path)    

    model_params = params['models']['ML']['random_forest']['params']
    rf = RandomForestClassifier(**model_params)
    
    rf.fit(X_train, y_train) 
    y_pred = rf.predict(X_test) 
        
    unique_categories = [i for i in range(len(config['artifacts']['INPUT_CLASSES']))]
    metric_dct = get_metrics(y_test, y_pred, unique_categories)
    metric_dct['model_name'] = 'Random_forest'
    
    whole_model_name = metric_dct['model_name'] 
    test_accuracy = metric_dct['accuracy']
    graphs = save_graphs_ML(metric_dct, whole_model_name, unique_categories, test_accuracy)
    
    return {'params': rf.get_params(), 
            'metrics': metric_dct, 
            'artifacts':{'model': rf}.update(graphs)}



def stacking_classifier(config_path, param_path, X_train, y_train, X_test, y_test):
    config = read_yaml(config_path)
    params = read_yaml(param_path)
    
    mnb_model_params = params['models']['ML']['naive_bayes']['params']
    rf_model_params = params['models']['ML']['random_forest']['params']
    svc_model_params = params['models']['ML']['svc']['params']
    
    mnb = MultinomialNB(**mnb_model_params)
    rf = RandomForestClassifier(**rf_model_params)
    
    estimators = [('mnb', mnb), ('rf', rf)]
    clf = StackingClassifier(estimators=estimators, final_estimator=SVC(**svc_model_params))
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test) 
        
    unique_categories = [i for i in range(len(config['artifacts']['INPUT_CLASSES']))]
    metric_dct = get_metrics(y_test, y_pred, unique_categories)
    metric_dct['model_name'] = 'Random_forest + Naive_bayes Stacking'
    
    whole_model_name = metric_dct['model_name'] 
    test_accuracy = metric_dct['accuracy']
    graphs = save_graphs_ML(metric_dct, whole_model_name, unique_categories, test_accuracy)
    
    return {'params': clf.get_params(), 
            'metrics': metric_dct, 
            'artifacts':{'model': clf}.update(graphs)}




def train(config_path, param_path):
    config = read_yaml(config_path)
    params = read_yaml(param_path)
    
    train_batch_size = pd.read_csv(os.path.join(config['artifacts']['PREPROCESSED_DATA_DIR'], 
                                                config['artifacts']['TRAIN_FILE_NAMES_CSV'])).shape[0]
    test_batch_size = pd.read_csv(os.path.join(config['artifacts']['PREPROCESSED_DATA_DIR'], 
                                                config['artifacts']['TEST_FILE_NAMES_CSV'])).shape[0]
    
    train_dataloader = doc_classifier_dataloader(config_path, param_path,  
                                                 config['artifacts']['TRAIN_FILE_NAMES_CSV'],
                                                 config['artifacts']['PREPROCESSED_DATA_DIR'],
                                                 data_type='train', model_type='ML', batch_size=train_batch_size)
    test_dataloader = doc_classifier_dataloader(config_path, param_path,  
                                                 config['artifacts']['TEST_FILE_NAMES_CSV'],
                                                 config['artifacts']['PREPROCESSED_DATA_DIR'],
                                                 data_type='test', model_type='ML', batch_size=test_batch_size)
    
        
    train_df = pd.DataFrame(next(iter(train_dataloader)))
    X_train = train_df['text']
    y_train = train_df['class']
    test_df = pd.DataFrame(next(iter(test_dataloader)))
    X_test = test_df['text']
    y_test = test_df['class']
    
    
    # theres prbably better ways of doing below
    lst = [] 
    for row in X_train: 
        lst.append(np.reshape(row, (1,-1))) 
    X_train = np.array(lst).squeeze(1)
    
    lst = [] 
    for row in X_test: 
        lst.append(np.reshape(row, (1,-1))) 
    X_test = np.array(lst).squeeze(1)
    
    
    naive_bayes_outputs = naive_bayes_model(config_path, param_path, X_train, y_train, X_test, y_test) 
    rf_outputs = rf_model(config_path, param_path, X_train, y_train, X_test, y_test)
    stacking_outputs = stacking_classifier(config_path, param_path, X_train, y_train, X_test, y_test)
    
    temp_dir = config['artifacts']['TEMP_MLFLOW_ARTIFACTS_DIR']
        
    experiment_name = "MLModels"
    experiment_id = mlflow.create_experiment(experiment_name)
    for model_outputs in [naive_bayes_outputs, rf_outputs, stacking_outputs]:
        create_directory(dirs=[temp_dir])
        model_outputs['artifacts']['fig'].savefig(os.path.join(temp_dir + model_outputs['name']))
        
        with mlflow.start_run(experiment_id=experiment_id):
            mlflow.log_metrics(naive_bayes_outputs['metrics'])
            mlflow.log_params(naive_bayes_outputs['params'])
            mlflow.sklearn.log_model(naive_bayes_outputs['artifacts']['model'], 'model')
            mlflow.log_artifact(os.path.join(temp_dir + model_outputs['name']), 'graphs')
            
        shutil.rmtree(os.path.join(temp_dir))
        
    
    print('\n\n0.5 done\n\n')
    
    
    
    


if __name__ == '__main__':
    args = argparse.ArgumentParser()

    args.add_argument("--config", "-c", default="config/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")

    parsed_args = args.parse_args()

    try:
        logging.info("\n>>>>> stage one started")
        train(config_path=parsed_args.config, param_path=parsed_args.params)
        logging.info("stage one completed! all the data are saved in local >>>>>\n\n")
    except Exception as e:
        logging.exception(e)
        raise e
    