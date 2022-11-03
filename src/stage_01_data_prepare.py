# this file will create converted data for each model, ML/FastText, DL, BERT
# and update the config.yaml for data directories for each of the models 

# TO DO: While iterating thorugh source_data to preprocess check if it is a dir, 
# for now assuming everything inside source_data is a dir 


from utils.preprocess import Preprocessor
import os 
import argparse 
import shutil
from tqdm import tqdm
import logging
from utils.common import create_directory, read_yaml, write_yaml, create_text_file, save_local_df, read_txt_file
from os.path import isfile, isdir
from os import listdir
import pandas as pd 
from sklearn.model_selection import train_test_split



# The below func will update the config.yaml file with the source data classes / folders inside source_data dir  
def update_config(config_path): 
    config = read_yaml(config_path)
    source_data_dir = config['artifacts']['SOURCE_DATA_DIR']
        
    classes = []
    for dir in listdir(source_data_dir):
        if isdir(os.path.join(source_data_dir, dir)):
           classes.append(dir) 
    
    print(f"\nWe have total of {len(classes)} different classes\n")  
    config['artifacts']['INPUT_CLASSES'] = classes
    
    write_yaml(path_to_yaml=config_path, config=config)
    
    print('\nupdate config done\n')
    
     


def get_data(config_path, param_path):
    config = read_yaml(config_path)
    params = read_yaml(param_path)
    
    # create preprocessed data dir
    source_data_dir = config['artifacts']['SOURCE_DATA_DIR']
    preprocessed_data_dir = config['artifacts']['PREPROCESSED_DATA_DIR']
    create_directory(dirs=[preprocessed_data_dir])   
    
    # create preprocessed train dir
    create_directory(dirs=[os.path.join(preprocessed_data_dir, 'train')])
    
    # create preprocessed test dir 
    create_directory(dirs=[os.path.join(preprocessed_data_dir, 'test')])
    
    
    # create each class of dataset as dir for train and test 
    classes_dirs = [os.path.join(preprocessed_data_dir, 'train', dir) for dir in config['artifacts']['INPUT_CLASSES']]
    create_directory(dirs=classes_dirs)
    
    classes_dirs = [os.path.join(preprocessed_data_dir, 'test', dir) for dir in config['artifacts']['INPUT_CLASSES']]
    create_directory(dirs=classes_dirs)
    
    
    file_names_list = []
    for dir in listdir(source_data_dir):
        for file in listdir(os.path.join(source_data_dir, dir)):
            file_names_list.append([dir, file])  
    file_names_df = pd.DataFrame(file_names_list, columns=['class', 'file_name'])
    
    X_train, X_test, y_train, y_test = train_test_split(file_names_df['file_name'], file_names_df['class'], 
                                                        test_size=params['preprocess']['test_size'], 
                                                        stratify=file_names_df['class'], random_state=42)
    train_data = pd.concat((X_train, y_train), axis=1).sample(frac=1).reset_index(drop=True).astype(str)
    test_data = pd.concat((X_test, y_test), axis=1).sample(frac=1).reset_index(drop=True).astype(str)
    
    
    # saving the file names as df as we will utilize them data loader
    save_local_df(train_data, os.path.join(preprocessed_data_dir, config['artifacts']['TRAIN_FILE_NAMES_CSV']))  # df: ['file_name', 'class']
    save_local_df(test_data, os.path.join(preprocessed_data_dir, config['artifacts']['TEST_FILE_NAMES_CSV']))
    
    # initialize preprocessing object  
    preproc = Preprocessor(remove_special=params['preprocess']['remove_special'], 
                           remove_stop=params['preprocess']['remove_stop'], 
                           summy=params['preprocess']['summy'],
                           stem=params['preprocess']['stem'], stem_type=params['preprocess']['stem_type'], 
                           lemm=params['preprocess']['lemm'], lemm_type=params['preprocess']['lemm_type'])
    
    no_of_files_cnt = 0
    def write_file(row):
        nonlocal no_of_files_cnt
        no_of_files_cnt+=1
        if no_of_files_cnt%1000==0:
            print(f"No of files done {no_of_files_cnt}")
        current_file = os.path.join(source_data_dir, row['class'], row['file_name'])
        text = read_txt_file(path_to_file_name=current_file)
        text = preproc.transform(text) 
        create_text_file(path_to_file_name=os.path.join(preprocessed_data_dir, train_or_test,  
                                                        row['class'], row['file_name']), text=text)
        
    train_or_test = 'train'
    train_data.apply(write_file, axis=1)  
    train_or_test = 'test'
    test_data.apply(write_file, axis=1)         
    print('\nget data done\n')

  


if __name__ == '__main__':
    args = argparse.ArgumentParser()

    args.add_argument("--config", "-c", default="config/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")

    parsed_args = args.parse_args()

    try:
        logging.info("\n>>>>> stage one started")
        update_config(config_path=parsed_args.config)
        get_data(config_path=parsed_args.config, param_path=parsed_args.params)
        logging.info("stage one completed! all the data are saved in local >>>>>\n\n")
    except Exception as e:
        logging.exception(e)
        raise e