from utils.preprocess import Preprocessor
from utils.common import create_directory, read_yaml, read_pickle_file
import os
from utils.data_loader import TransformData
import numpy as np
import copy
import torch
from transformers import DistilBertForSequenceClassification, AdamW, DistilBertTokenizer



config_path = "./config/config.yaml"
param_path = "./params.yaml"
config = read_yaml(config_path)
params = read_yaml(param_path)
params_copy = copy.copy(params)
params_copy['preprocess'].pop('test_size')
params_copy['preprocess'].pop('valid_size')


preproc = Preprocessor(**params_copy['preprocess'])
label_encoder_path = os.path.join(config['artifacts']['ENCODERS']['encoder_dir_name'],

config['artifacts']['ENCODERS']['label_encoder'])
label_encoder = read_pickle_file(label_encoder_path)



def predict(text, model, model_type, seq_length):
    preproc_text = preproc.transform(text)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'

    if model_type == 'ML':
        tfidf_encoder_path = os.path.join(config['artifacts']['ENCODERS']['encoder_dir_name'],
                                          config['artifacts']['ENCODERS']['tfidf_encoder'])
        tfidf_encoder = read_pickle_file(tfidf_encoder_path)
        transformed_text = tfidf_encoder.transform([preproc_text]).todense()
        output = label_encoder.inverse_transform(model.predict(transformed_text))
        return output

    if model_type == 'DL':
        transform_DL_data = TransformData(config_path, param_path, model_type='DL', seq_length=seq_length)
        # sample = {'text':text, 'class':'None'}
        sample = {'text':preproc_text, 'class':'None'}
        transformed_text = transform_DL_data(sample)
        data = torch.Tensor(transformed_text['text']).long().to(device)

        is_batch_first = params['models']['DL']['params']['batch_first']
        if is_batch_first:
            data = data.reshape(1, data.shape[0])

        with torch.no_grad():
            preds = model(data)
        return label_encoder.inverse_transform(torch.argmax(preds, 1).to('cpu'))


    if model_type == 'BERT':
        model_tokenizer_dir = os.path.join(config['artifacts']['ENCODERS']['encoder_dir_name'],
                                           config['artifacts']['ENCODERS']['bert_encoder'],
                                           config['artifacts']['ENCODERS']['bert_tokenizer'])
        tokenizer = DistilBertTokenizer.from_pretrained(model_tokenizer_dir, model_max_length=300)
        # items = tokenizer(text, truncation=True, padding=True, return_tensors="pt")
        items = tokenizer(preproc_text, truncation=True, padding=True, return_tensors="pt")

        with torch.no_grad():
            input_ids = items['input_ids'].to(device)
            attention_mask = items['attention_mask'].to(device)
            preds = model(input_ids, attention_mask=attention_mask)

            return label_encoder.inverse_transform([torch.argmax(preds['logits']).to('cpu')])