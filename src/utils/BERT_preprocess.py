import torch
from torch.utils.data import Dataset 
from utils.common import read_yaml, read_txt_file, read_pickle_file, create_directory
import os 
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification



def download_model(config_path, param_path):
    # implement this function to download weights and automatically rename some of the files and delete other files

    config = read_yaml(config_path)
    params = read_yaml(param_path)

    model_dir = os.path.join(config['artifacts']['ENCODERS']['encoder_dir_name'], 
                             config['artifacts']['ENCODERS']['bert_encoder'])
    model_weights_dir = os.path.join(config['artifacts']['ENCODERS']['encoder_dir_name'], 
                                     config['artifacts']['ENCODERS']['bert_encoder'], 
                                     config['artifacts']['ENCODERS']['bert_model'])
    model_tokenizer_dir = os.path.join(config['artifacts']['ENCODERS']['encoder_dir_name'], 
                                       config['artifacts']['ENCODERS']['bert_encoder'], 
                                       config['artifacts']['ENCODERS']['bert_tokenizer'])

    create_directory(dirs = [model_dir])
    create_directory(dirs = [model_weights_dir])    
    create_directory(dirs = [model_tokenizer_dir])  

    model = DistilBertForSequenceClassification.from_pretrained(model_dir)




class DocClassificationDataset(Dataset):
    def __init__(self, config_path, param_path, model_name, txt_file_names_df, data_type='train'):
        self.config = read_yaml(config_path)
        self.params = read_yaml(param_path)
        # self.tokenizer = self.load_tokenizer(model_name)
        self.txt_file_names_df = txt_file_names_df   # ['file_name', 'directory'] directory is nothing but class name

        # while passing labels inside __getitem__ we will do this label encoding 
        self.label_encoder = read_pickle_file(os.path.join(
            self.config['artifacts']['ENCODERS']['encoder_dir_name'], 
            self.config['artifacts']['ENCODERS']['label_encoder']
        ))
        self.data_type = data_type

    def __getitem__(self, idx):
        text = read_txt_file(
            os.path.join(
                self.config['artifacts']['PREPROCESSED_DATA_DIR'],
                self.data_type,
                str(self.txt_file_names_df.iloc[idx, 1]),
                str(self.txt_file_names_df.iloc[idx, 0])
            )
        )

        # adding this to remove first 20 words from each doc before using tokenizer, 
        # cause we are only taking 120 words in tokenizer
        # text = ' '.join(text.split(' ')[20:])

        labels = torch.tensor(self.label_encoder.transform([self.txt_file_names_df.iloc[idx, 1]])[0])
        item = {'text':text, 'labels':labels}
        # text_encoding = self.tokenizer(text, truncation=True, padding=True, )

        # print("\n\n", len(text_encoding['input_ids']), len(text_encoding['attention_mask']))

        # item = {key: torch.tensor(val) for key, val in text_encoding.items()}
        # # print("\n", self.txt_file_names_df.iloc[idx, 1], "\n")
        # # print("\n\n", self.label_encoder.transform([self.txt_file_names_df.iloc[idx, 1]])[0], "\n\n")
        # item['labels'] = torch.tensor(self.label_encoder.transform([self.txt_file_names_df.iloc[idx, 1]])[0])
        # return item
        return item  

    def __len__(self):
        return len(self.txt_file_names_df)


    def load_tokenizer(self, model_name):
        # check for the tokenizer for the type of BERT, of not present then download otherwise use the same
        # we will check this inside encoder/bert_model_name/

        model_tokenizer_dir = os.path.join(self.config['artifacts']['ENCODERS']['encoder_dir_name'], 
                                           self.config['artifacts']['ENCODERS']['bert_encoder'], 
                                           self.config['artifacts']['ENCODERS']['bert_tokenizer'])      

        return DistilBertTokenizer.from_pretrained(model_tokenizer_dir, max_length=300)


