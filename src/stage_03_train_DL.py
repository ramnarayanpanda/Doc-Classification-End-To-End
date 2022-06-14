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
    
    
    count_encoder = read_pickle_file(os.path.join(config['artifacts']['ENCODERS']['encoder_dir_name'], 
                                                  config['artifacts']['ENCODERS']['count_encoder']))
    vocab_size = len(count_encoder) + 2 
    
    embedding_dim = params['models']['DL']['params']['embedding_dim']
    hidden_dim = params['models']['DL']['params']['hidden_dim']
    n_layers = params['models']['DL']['params']['n_layers']
    drop_prob = params['models']['DL']['params']['drop_proba']
    model_name = params['models']['DL']['params']['model_name']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bidirectional = params['models']['DL']['params']['bidirectional']
    seq_length = params['models']['DL']['params']['seq_length']
    batch_size = params['models']['DL']['params']['batch_size']
    take_all_layers_output = params['models']['DL']['params']['take_all_layers_output']
    batch_first = params['models']['DL']['params']['batch_first']
    epochs = params['models']['DL']['params']['epochs']
    lr = params['models']['DL']['params']['lr']
    output_size = params['models']['DL']['params']['output_size']
    
    whole_model_name = ('' if bidirectional == False else 'B') + model_name + '_' + str(epochs) + 'Epochs_' + str(n_layers) + 'Layers_' + str(embedding_dim) + 'Embed_' + str(seq_length) + 'SeqLength_' + str(drop_prob) + 'DropProb_' + str(take_all_layers_output) + 'TAKE_OUTPUT_OF_ALL_LAYERS_OF_MODEL_' 
    unique_categories = config['artifacts']['INPUT_CLASSES']
    
    
    print(f"\n>>>>>>>>>>>>>>>>{device}<<<<<<<<<<<<<<<<<<<<<\n")
    
        
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
    
    model_params = params['models']['DL']['params']
    model_params['model_name'] = ('' if bidirectional == False else 'B') + model_name
    
    del metric_dct['confusion_matrix']
    
    
    temp_dir = config['artifacts']['TEMP_MLFLOW_ARTIFACTS_DIR']
    create_directory(dirs=[temp_dir])
    
    conf_mat_plot['fig'].savefig(os.path.join(temp_dir, conf_mat_plot['name']))
    acc_loss_plot['fig'].savefig(os.path.join(temp_dir, acc_loss_plot['name']))
    
    with mlflow.start_run():
            mlflow.log_metrics(metric_dct)
            mlflow.log_params(model_params)
            mlflow.pytorch.log_model(model, 'model')
            mlflow.log_artifact(os.path.join(temp_dir, conf_mat_plot['name']), 'graphs')
            mlflow.log_artifact(os.path.join(temp_dir, acc_loss_plot['name']), 'graphs')
            
    shutil.rmtree(os.path.join(temp_dir))
    
    
    
    
    
    
    
def check(config_path, param_path):
    config = read_yaml(config_path)
    params = read_yaml(param_path)
    
    seq_length = params['models']['DL']['params']['seq_length']
    batch_size = 3
    
    train_loader = doc_classifier_dataloader(config_path, param_path, 
                                             csv_file_name = config['artifacts']['TRAIN_FILE_NAMES_CSV'],
                                             file_path = config['artifacts']['PREPROCESSED_DATA_DIR'], 
                                             data_type='train', model_type='DL', 
                                             seq_length=seq_length, 
                                             batch_size=batch_size)
    
    # data = next(iter(train_loader))
    
    # print(data[0]['class'].shape)
    
    # for batch_idx, (data, targets) in enumerate(train_loader):
    #     print(batch_idx, data.shape, targets.shape)
    
    # for batch_idx, sample in enumerate(train_loader):
    #     print(batch_idx, sample['text'], sample['class'])
    #     break
    
    # for batch_idx, row in enumerate(train_loader):
    #     try:
    #         print('done/n')
    #         data = torch.Tensor([i['text'] for i in row])
    #         targets = torch.Tensor([i['class'] for i in row])
    #         data = data.long()
    #         targets = targets.long()
    #     except: 
    #         print(list(row[0].keys()))
        
    #     break
    
    for batch_idx, row in enumerate(train_loader):
        
        data = torch.Tensor([i['class'].tolist() for i in row])
        break 
    print('Okay to go') 
    



if __name__ == '__main__':
    args = argparse.ArgumentParser()

    args.add_argument("--config", "-c", default="config/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")

    parsed_args = args.parse_args()

    try:
        logging.info("\n>>>>> stage one started")
        train(config_path=parsed_args.config, param_path=parsed_args.params)
        # check(config_path=parsed_args.config, param_path=parsed_args.params)
        logging.info("stage one completed! all the data are saved in local >>>>>\n\n")
    except Exception as e:
        logging.exception(e)
        raise e