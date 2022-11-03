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
import itertools
import warnings
warnings.filterwarnings('ignore')




def get_data(config_path, param_path):
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

    return X_train, X_test, y_train, y_test 




def test_pred_save_mlflow(config_path, param_path, model, y_test, y_pred, model_name, track_params):
    config = read_yaml(config_path)
    params = read_yaml(param_path)

    unique_categories = config['artifacts']['INPUT_CLASSES']
    metric_dct = get_metrics(y_test, y_pred, [i for i in range(len(unique_categories))])

    param_dct = model.get_params()
    param_dct.update({'model_name': model_name})
    # track_params = ['model_name', 'alpha', 'fit_prior']
        
    whole_model_name = param_dct['model_name'] 
    graphs = save_graphs_ML(metric_dct, whole_model_name, unique_categories)
    
    graphs.update({'model': model})
    
    del metric_dct['confusion_matrix']
    
    model_output = {'params':{key:param_dct[key] for key in track_params}, 
            'metrics': metric_dct, 
            'artifacts':graphs}

    img_file_path = os.path.join('.', model_output['artifacts']['name'])
    model_output['artifacts']['fig'].savefig(img_file_path) 
    
    with mlflow.start_run():
        mlflow.log_metrics(model_output['metrics'])
        mlflow.log_params(model_output['params'])
        mlflow.sklearn.log_model(model_output['artifacts']['model'], 'model')
        mlflow.log_artifact(img_file_path, 'graphs')
    os.remove(img_file_path)




def train(config_path, param_path):
    config = read_yaml(config_path)
    params = read_yaml(param_path)

    mnb_params = params['models']['ML']['naive_bayes']['params'] 
    rf_params = params['models']['ML']['random_forest']['params']
    svc_params = params['models']['ML']['svc']['params']
    
    X_train, X_test, y_train, y_test = get_data(config_path, param_path)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


    mnb_params_lst = list(itertools.product(*list(mnb_params.values())))
    rf_params_lst = list(itertools.product(*list(rf_params.values())))
    svc_params_lst = list(itertools.product(*list(svc_params.values())))

    cnt = 0
    for param in mnb_params_lst:
        cnt+=1
        print(f"/n training for {cnt}")
        model_params = dict(zip(list(mnb_params.keys()), param))
        model_outputs = naive_bayes_model(config_path, param_path, X_train, X_test, y_train, y_test, model_params)
        # test_pred_save_mlflow(config_path, param_path, model_outputs)

    for param in rf_params_lst:
        cnt+=1
        if cnt%2!=0:
            continue 
        print(f"/n training for {cnt}")
        model_params = dict(zip(list(rf_params.keys()), param))
        model_outputs = rf_model(config_path, param_path, X_train, X_test, y_train, y_test, model_params)
        # test_pred_save_mlflow(config_path, param_path, model_outputs)

    for param in list(itertools.product(mnb_params_lst, rf_params_lst, svc_params_lst)):
        cnt+=1
        if cnt%37!=0:
            continue 
        print(f"/n training for {cnt}")
        model_params = {'mnb_model_params':dict(zip(list(mnb_params.keys()), param[0])), 
                        'rf_model_params':dict(zip(list(rf_params.keys()), param[1])), 
                        'svc_model_params':dict(zip(list(svc_params.keys()), param[2]))}
        model_outputs = stacking_classifier(config_path, param_path, X_train, X_test, y_train, y_test, model_params)
        # test_pred_save_mlflow(config_path, param_path, model_outputs)




    

def naive_bayes_model(config_path, param_path, X_train, X_test, y_train, y_test, model_params):
    config = read_yaml(config_path)
    params = read_yaml(param_path)

    mnb = MultinomialNB(**model_params)    
    mnb.fit(X_train, y_train) 
    y_pred = mnb.predict(X_test) 

    track_params = ['model_name', 'alpha', 'fit_prior']
    model_name = 'Naive Bayes'

    test_pred_save_mlflow(config_path, param_path, mnb, y_test, y_pred, model_name, track_params) 
    


def rf_model(config_path, param_path, X_train, X_test, y_train, y_test, model_params=None):
    config = read_yaml(config_path)
    params = read_yaml(param_path)    

    rf = RandomForestClassifier(**model_params)
    rf.fit(X_train, y_train) 
    y_pred = rf.predict(X_test) 
    
    track_params = ['model_name', 'n_estimators', 'max_depth', 'min_samples_split', 
                    'max_leaf_nodes']
    model_name = 'Random Forest'

    test_pred_save_mlflow(config_path, param_path, rf, y_test, y_pred, model_name, track_params)  



def stacking_classifier(config_path, param_path, X_train, X_test, y_train, y_test, model_params=None):
    config = read_yaml(config_path)
    params = read_yaml(param_path)

    mnb_model_params =  model_params['mnb_model_params']
    rf_model_params = model_params['rf_model_params']
    svc_model_params = model_params['svc_model_params']
    
    mnb = MultinomialNB(**mnb_model_params)
    rf = RandomForestClassifier(**rf_model_params)
    estimators = [('mnb', mnb), ('rf', rf)]
    stack_clf = StackingClassifier(estimators=estimators, final_estimator=SVC(**svc_model_params))
    
    stack_clf.fit(X_train, y_train)
    y_pred = stack_clf.predict(X_test) 

    track_params = ['model_name', 'rf__n_estimators', 'rf__max_depth', 'rf__min_samples_split', 'rf__max_leaf_nodes', 
                    'rf__max_samples', 'mnb__alpha', 'mnb__fit_prior']
    model_name = 'RF+MNB Stack'

    test_pred_save_mlflow(config_path, param_path, stack_clf, y_test, y_pred, model_name, track_params) 
        



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
    