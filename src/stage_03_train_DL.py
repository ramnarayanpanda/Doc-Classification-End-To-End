import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import optim
import torch.nn.functional as F
import torch.nn as nn

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import seaborn as sns

from utils.data_loader import doc_classifier_dataloader
from utils.metrics_plots import get_metrics, save_graphs_ML, save_graphs_DL
from utils.DL_Model_util import DLModel
from utils.common import create_directory, read_yaml, write_yaml, create_text_file, save_pickle_file, \
                         read_pickle_file, read_txt_file

import tqdm
import time
import argparse 
import logging
import os 
import mlflow 
import shutil
import itertools

import warnings
warnings.filterwarnings("ignore")

torch.cuda.empty_cache()






def test_metrics(model, device, valid_loader):
    model.eval()
    correct = 0
    total = 0
    sum_loss = 0.0

    with torch.no_grad():
        for batch_idx, sample in enumerate(valid_loader):
            data = torch.Tensor([i['text'] for i in sample])
            targets = torch.Tensor([i['class'].tolist() for i in sample])
            data = data.long().to(device)
            targets = targets.long().to(device)
            preds = model(data)
            loss = F.cross_entropy(preds, targets)
            correct += (torch.max(preds, 1)[1] == targets).float().sum()
            total += targets.shape[0]
            sum_loss += loss.item() * targets.shape[0]
    return sum_loss / total, correct / total






def train(config_path, param_path):
    config = read_yaml(config_path)
    params = read_yaml(param_path)

    train_hyper_params_keys = list(params['models']['DL']['params'].keys())
    train_hyper_params_values = list(params['models']['DL']['params'].values())

    cnt = 0
    for train_hyper_params_value in list(itertools.product(*train_hyper_params_values)):
        # cnt+=1 
        # if cnt%3!=0:
        #     continue 
        print(f"Count is {cnt}")
        t1 = time.time()
        train_hyper_params = dict(zip(train_hyper_params_keys, train_hyper_params_value))
        train_each_hyperparam(config_path, param_path, train_hyper_params)
        print("\n\n", train_hyper_params['model_name'], train_hyper_params['epochs'], time.time() - t1)

    print("DL training is done")




def train_each_hyperparam(config_path, param_path, train_hyper_params):
    config = read_yaml(config_path)
    params = read_yaml(param_path)
    
    
    count_encoder = read_pickle_file(os.path.join(config['artifacts']['ENCODERS']['encoder_dir_name'], 
                                                  config['artifacts']['ENCODERS']['count_encoder']))
    vocab_size = len(count_encoder) + 2 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f">>>>>>>>>>>>>>>>>>>>>{device}<<<<<<<<<<<<<<<<<<<<<<<<")
    
    embedding_dim = train_hyper_params['embedding_dim'] 
    hidden_dim = train_hyper_params['hidden_dim']
    n_layers = train_hyper_params['n_layers']
    drop_prob = train_hyper_params['drop_proba']
    model_name = train_hyper_params['model_name']
    bidirectional = train_hyper_params['bidirectional']
    seq_length = train_hyper_params['seq_length']
    batch_size = train_hyper_params['batch_size']
    take_all_layers_output = train_hyper_params['take_all_layers_output']
    batch_first = train_hyper_params['batch_first']
    epochs = train_hyper_params['epochs']
    lr = train_hyper_params['lr']
    whole_model_name = ('' if bidirectional == False else 'B') + model_name + '_' + str(epochs) + 'Epochs_' + str(n_layers) + 'Layers_' + str(embedding_dim) + 'Embed_' + str(seq_length) + 'SeqLength_' + str(drop_prob) + 'DropProb_' + str(take_all_layers_output) + 'TAKE_OUTPUT_OF_ALL_LAYERS_OF_MODEL_' 
    unique_categories = config['artifacts']['INPUT_CLASSES']    
    output_size = len(unique_categories)

    train_loader = doc_classifier_dataloader(config_path, param_path, 
                                             csv_file_name = config['artifacts']['TRAIN_FILE_NAMES_CSV'],
                                             file_path = config['artifacts']['PREPROCESSED_DATA_DIR'], 
                                             data_type='train', model_type='DL', 
                                             seq_length=seq_length, 
                                             batch_size=batch_size)
    test_loader = doc_classifier_dataloader(config_path, param_path, 
                                             csv_file_name = config['artifacts']['TEST_FILE_NAMES_CSV'],
                                             file_path = config['artifacts']['PREPROCESSED_DATA_DIR'], 
                                             data_type='test', model_type='DL', 
                                             seq_length=seq_length, 
                                             batch_size=batch_size)
    
    # model, criterion, optimizer load 
    model = DLModel(vocab_size, output_size, embedding_dim, hidden_dim,
                   n_layers, drop_prob, batch_first=batch_first, 
                   model_name=model_name, device=device, bidirectional=bidirectional,
                   word_embedding='', seq_length=seq_length, 
                   take_all_layers_output=take_all_layers_output)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_loss_perEpoch, train_accuracy_perEpoch, test_loss_perEpoch, test_accuracy_perEpoch = [], [], [], []
    
    for e in range(epochs):
        print('training epoch', e)

        t1 = time.time()
        model.train()
        sum_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, sample in enumerate(train_loader):
            data = torch.Tensor([i['text'] for i in sample])
            targets = torch.Tensor([i['class'].tolist() for i in sample])

            data = data.long().to(device)
            targets = targets.long().to(device)

            # print(data.shape, targets.shape) 

            preds = model(data)
            loss = criterion(preds, targets)

            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent or adam step
            optimizer.step()

            sum_loss += loss.item() * targets.shape[0]
            correct += (torch.max(preds, 1)[1] == targets).float().sum()
            total += targets.shape[0]

        test_results = test_metrics(model, device, test_loader)
        test_loss, test_acc = test_results[:2]

        train_loss_perEpoch.append(sum_loss / total)
        train_accuracy_perEpoch.append((correct / total).item())
        test_loss_perEpoch.append(test_loss)
        test_accuracy_perEpoch.append((test_acc).item())

    
    # Once more find the preds after the model got trained 
    model.eval()  
    pred_labels = []  
    true_labels = []
    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):
            data = torch.Tensor([i['text'] for i in sample])
            targets = torch.Tensor([i['class'].tolist() for i in sample])
            data = data.long().to(device)
            
            targets = targets.long().to(device)
            for i in targets:
                true_labels.append(i.cpu())    

            preds = model(data)
            for i in torch.max(preds, 1)[1]:
                pred_labels.append(i.cpu())
        
    metric_dct = get_metrics(true_labels, pred_labels, [i for i in range(len(unique_categories))])
    conf_mat_plot, acc_loss_plot = save_graphs_DL(metric_dct, whole_model_name, unique_categories,
                                                  train_loss_perEpoch, test_loss_perEpoch, 
                                                  train_accuracy_perEpoch, test_accuracy_perEpoch)    
    
    # model_params = params['models']['DL']['params']
    train_hyper_params['model_name'] = ('' if bidirectional == False else 'B') + model_name
    
    del metric_dct['confusion_matrix']
    
    conf_mat_plot['fig'].savefig(conf_mat_plot['name'])
    acc_loss_plot['fig'].savefig(acc_loss_plot['name'])
    
    with mlflow.start_run():
            mlflow.log_metrics(metric_dct)
            mlflow.log_params(train_hyper_params)
            mlflow.pytorch.log_model(model, 'model')
            mlflow.log_artifact(conf_mat_plot['name'], 'graphs')
            mlflow.log_artifact(acc_loss_plot['name'], 'graphs')
            
    os.remove(conf_mat_plot['name'])
    os.remove(acc_loss_plot['name'])



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