from utils.featurization import FeatureExtractor
import os 
import argparse 
from tqdm import tqdm
import logging
from utils.common import create_directory, read_yaml, write_yaml, create_text_file, save_pickle_file, read_txt_file
from os.path import isfile, isdir
from os import listdir
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder



config_path='config/config.yaml' 
param_path='params.yaml'


config = read_yaml(config_path)
params = read_yaml(param_path)


preprocessed_train_data_dir = os.path.join(config['artifacts']['PREPROCESSED_DATA_DIR'], 'train') 

# create artifacts directory where all the feature extractors will be saved 
encoders_dir = config['artifacts']['ENCODERS']['encoder_dir_name']
create_directory(dirs=[encoders_dir])

data_list = []  
for dir in listdir(preprocessed_train_data_dir):
    for file in listdir(os.path.join(preprocessed_train_data_dir, dir)):
        current_file = os.path.join(preprocessed_train_data_dir, dir, file)
        text = read_txt_file(path_to_file_name=current_file)
        data_list.append([dir, text])  
train_df = pd.DataFrame(data_list, columns=['class', 'text']) 


print(train_df['class'].unique())

encoder = OneHotEncoder()
encoder.fit(train_df[['class']])  

encoder.transform([['politics']])