from utils.featurization import FeatureExtractor
import os 
import argparse 
from tqdm import tqdm
import logging
from utils.common import create_directory, read_yaml, write_yaml, create_text_file, save_pickle_file, read_txt_file
from os.path import isfile, isdir
from os import listdir
import pandas as pd



def save_feature_encoders_to_artifacts(config_path, param_path): 
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
    
    
    # print(f"\n\n>>>>>>>>>>>here we are: {params['feature_extraction']['params']['ngram_range']}\n\n")
    # print(type(params['feature_extraction']['params']['ngram_range']))
    feature_extractor = FeatureExtractor(ngram_range=eval(params['feature_extraction']['params']['ngram_range']))
    feature_extractor.transform(train_df)
    label_encoder = feature_extractor.label_encoder
    count_encoder = feature_extractor.count_encoding
    tfidf_encoder = feature_extractor.tf_idf_encoding
    
    # saving 3 feature ecoders 
    save_pickle_file(os.path.join(encoders_dir, config['artifacts']['ENCODERS']['label_encoder']), label_encoder)
    save_pickle_file(os.path.join(encoders_dir, config['artifacts']['ENCODERS']['count_encoder']), count_encoder)
    save_pickle_file(os.path.join(encoders_dir, config['artifacts']['ENCODERS']['tfidf_encoder']), tfidf_encoder)



if __name__ == '__main__':
    args = argparse.ArgumentParser()

    args.add_argument("--config", "-c", default="config/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")

    parsed_args = args.parse_args()

    try:
        logging.info("\n>>>>> stage one started")
        save_feature_encoders_to_artifacts(config_path=parsed_args.config, param_path=parsed_args.params)
        logging.info("stage one completed! all the data are saved in local >>>>>\n\n")
    except Exception as e:
        logging.exception(e)
        raise e