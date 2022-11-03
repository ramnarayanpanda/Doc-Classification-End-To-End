from unittest.util import _MAX_LENGTH
import tqdm
import time
import argparse 
import logging
import os 
import mlflow 
import shutil
import pandas as pd 

from transformers import DistilBertForSequenceClassification, AdamW, DistilBertTokenizer
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
import itertools 
from utils.metrics_plots import get_metrics, save_graphs_ML, save_graphs_DL
from utils.common import create_directory, read_yaml, write_yaml, create_text_file, save_pickle_file, \
                         read_pickle_file, read_txt_file
from utils.BERT_preprocess import DocClassificationDataset
import warnings
warnings.filterwarnings('ignore')


from pynvml import *
nvmlInit()
h = nvmlDeviceGetHandleByIndex(0)
info = nvmlDeviceGetMemoryInfo(h)
print('\n', f'total : {info.total},   free : {info.free},   used : {info.used}', '\n')





def train(config_path, param_path):
    config = read_yaml(config_path)
    params = read_yaml(param_path)

    train_hyper_params_keys = list(params['models']['DistilBert']['params'].keys())
    train_hyper_params_values = list(params['models']['DistilBert']['params'].values())

    for train_hyper_params_value in list(itertools.product(*train_hyper_params_values)):
        t1 = time.time()
        train_hyper_params = dict(zip(train_hyper_params_keys, train_hyper_params_value))
        print("\n\n", train_hyper_params['model_name'], train_hyper_params['epochs'], time.time() - t1)
        train_each_hyperparam(config_path, param_path, train_hyper_params)

    print("DL training is done")






def train_each_hyperparam(config_path, param_path, train_hyper_params):
    torch.cuda.empty_cache()
    config = read_yaml(config_path)
    params = read_yaml(param_path)


    train_file_path = os.path.join(config['artifacts']['TRAIN_FILE_NAMES_CSV'])
    train_df = pd.read_csv(os.path.join(config['artifacts']['PREPROCESSED_DATA_DIR'], 
                                        train_file_path))

    test_file_path = config['artifacts']['TEST_FILE_NAMES_CSV']
    test_df = pd.read_csv(os.path.join(config['artifacts']['PREPROCESSED_DATA_DIR'], 
                                       test_file_path))

    unique_categories = list(train_df['class'].unique())
    model_name = train_hyper_params['model_name']

    train_dataset = DocClassificationDataset(config_path, param_path, model_name, train_df, data_type='train')
    test_dataset = DocClassificationDataset(config_path, param_path, model_name, test_df, data_type='test')

    epochs = train_hyper_params['epochs']
    lr = train_hyper_params['lr']
    batch_size = train_hyper_params['batch_size']

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')

    model_weights_dir = os.path.join(config['artifacts']['ENCODERS']['encoder_dir_name'], 
                                     config['artifacts']['ENCODERS']['bert_encoder'], 
                                     config['artifacts']['ENCODERS']['bert_model'])
    model = DistilBertForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_weights_dir, 
                                                                num_labels=len(unique_categories))
    model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    optim = AdamW(model.parameters(), lr=lr)

    model_tokenizer_dir = os.path.join(config['artifacts']['ENCODERS']['encoder_dir_name'], 
                                       config['artifacts']['ENCODERS']['bert_encoder'], 
                                       config['artifacts']['ENCODERS']['bert_tokenizer'])   
    tokenizer = DistilBertTokenizer.from_pretrained(model_tokenizer_dir, model_max_length=300) 

    model.train()
    for epoch in range(epochs):
        print(f"epoch {epoch}")
        for batch in train_loader:
            items = tokenizer(batch['text'], truncation=True, padding=True, return_tensors="pt")

            optim.zero_grad()
            input_ids = items['input_ids'].to(device)
            attention_mask = items['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            loss.backward()
            optim.step()

    print("\n>>>>>>>>> Started Evaluation <<<<<<<<\n")


    model.eval()
    pred_labels = []  
    true_labels = []
    with torch.no_grad():
        for batch in test_loader:
            # print("\n\n", type(batch['text']))

            items = tokenizer(batch['text'], truncation=True, padding=True, return_tensors="pt")

            optim.zero_grad()
            input_ids = items['input_ids'].to(device)
            attention_mask = items['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

            # print(torch.argmax((outputs['logits']).to('cpu'), dim=1).tolist(), labels.tolist())

            pred_labels += torch.argmax((outputs['logits']).to('cpu'), dim=1).tolist()
            true_labels += labels.tolist()

    # print(f"\n\nValidation is done {len(pred_labels), len(true_labels)}\n")


    metric_dct = get_metrics(true_labels, pred_labels, [i for i in range(len(unique_categories))])

    ####################################
    # change this hard coded name 
    whole_model_name = 'distill-bert-uncased'   
    train_hyper_params['model_name'] = whole_model_name
    graphs = save_graphs_ML(metric_dct, whole_model_name, unique_categories)

    graphs.update({'model': model})
    
    del metric_dct['confusion_matrix']

    img_file_path = os.path.join('.', graphs['name'])
    graphs['fig'].savefig(img_file_path)
        
    with mlflow.start_run():
        mlflow.log_metrics(metric_dct)
        mlflow.log_params(train_hyper_params)
        mlflow.pytorch.log_model(model, 'model')
        mlflow.log_artifact(img_file_path, 'graphs')
            
    os.remove(img_file_path)






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