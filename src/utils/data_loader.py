import torch 
from torch.utils.data import Dataset, DataLoader
import pandas as pd  
import os 
from utils.common import read_txt_file, read_yaml, read_pickle_file
import numpy as np 
from nltk.tokenize import word_tokenize, RegexpTokenizer




class TransformData(object):
    
    def __init__(self, config_path, param_path, model_type='DL', seq_length=350):
        config = read_yaml(config_path)
        params = read_yaml(param_path)
        
        self.model_type = model_type
        self.seq_length = seq_length
        
        self.encoders_dir = config['artifacts']['ENCODERS']['encoder_dir_name']
        self.label_encoder = read_pickle_file(os.path.join(self.encoders_dir, config['artifacts']['ENCODERS']['label_encoder']))
        self.count_encoder = read_pickle_file(os.path.join(self.encoders_dir, config['artifacts']['ENCODERS']['count_encoder']))
        self.tfidf_encoder = read_pickle_file(os.path.join(self.encoders_dir, config['artifacts']['ENCODERS']['tfidf_encoder']))
        self.one_hot_encoder = read_pickle_file(os.path.join(self.encoders_dir, config['artifacts']['ENCODERS']['one_hot_encoder']))
        
    def apply_feature_extraction(self, sample):
        # print("\n\nentered feature extractor successfully")
        
        if self.model_type=='DL':
            tokenizer = RegexpTokenizer(r'\w+')
            sample['text'] = [self.count_encoder.get(word, 0) for word in tokenizer.tokenize(sample['text'])]
            # print("\n\n", sample['class'], "\n\n")
            
            # sample['class'] = self.one_hot_encoder.transform([[sample['class']]]).todense()
            
        elif self.model_type=='ML':
            sample['text'] = self.tfidf_encoder.transform([sample['text']]).todense()
        sample['class'] = self.label_encoder.transform([sample['class']])[0]
        
        
        # print(type(sample['text']), sample['text'].shape, type(sample['class']))
        
        # print("\n\napplied feature extractor successfully")
        
        return  sample
            
    def apply_padding(self, sample):
        text_len = len(sample['text'])
        if text_len <= self.seq_length:
            zeroes = list(np.zeros(self.seq_length - text_len))
            new = zeroes + sample['text']
        elif text_len > self.seq_length:
            new = sample['text'][0:self.seq_length]
        sample['text'] = new
        
        # print(f"applied padding successfully{features.shape}\n\n")
        # return (sample['text'], sample['class']) 
        return sample 
        
        
    def __call__(self, sample):
        sample = self.apply_feature_extraction(sample)
        
        if self.model_type=='DL':
            sample = self.apply_padding(sample) 
        
        # print("\n\n__call__ successfully\n\n")
        # print(type(sample['text']), type(sample['class']))
        
        return sample 






class DocClassificationDataset(Dataset):
    
    # this file path is processed_data/train(test)
    def __init__(self, config_path, param_path, csv_file_name, file_path, 
                 data_type='train', model_type='DL', seq_length=350):
        self.file_path = file_path
        self.file_df = pd.read_csv(os.path.join(file_path, csv_file_name))
        self.data_type = data_type
        self.model_type = model_type
        self.seq_length = seq_length
        
        self.transform = TransformData(config_path, param_path, model_type=model_type, seq_length=seq_length)
                
        
    def __len__(self):
        return len(self.file_df)

    def __getitem__(self, idx):        
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # text_features = [] 
        # class_features = [] 
        
        text_features = read_txt_file(
                    os.path.join(
                        self.file_path, self.data_type,
                        self.file_df.iloc[idx, 1], 
                        self.file_df.iloc[idx, 0]
                    )
                )
            
        class_features = self.file_df.iloc[idx, 1]
            
        sample = {'text':text_features, 'class':class_features}
        
        sample = self.transform(sample)
        
        # print('ttttttttttttttttttttttttttttttttttttttttt', type(sample))
        return sample
    
    
     
    
    

def doc_classifier_dataloader(config_path, param_path, csv_file_name, file_path, 
                              data_type='train', model_type='DL', seq_length=350, batch_size=32):
    
    classification_dataset = DocClassificationDataset(config_path, param_path, 
                                                      csv_file_name, file_path,
                                                      data_type=data_type, model_type=model_type, 
                                                      seq_length=seq_length) 
    
    dataloader = DataLoader(classification_dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=0, collate_fn=lambda x: x, drop_last=True)
    return dataloader