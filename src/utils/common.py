import yaml 
import os 
import pandas as pd  
import json 
import pickle 


def read_yaml(path_to_yaml: str) -> dict: 
    with open(path_to_yaml) as yaml_file: 
        content = yaml.safe_load(yaml_file) 
    return content  

def write_yaml(path_to_yaml: str, config: dict): 
    with open(path_to_yaml, 'w') as yaml_file:
        yaml.dump(config, yaml_file)
    print(f">>>>>>>>>Updated config.yaml file with classes")



def create_directory(dirs: list): 
    for dir_path in dirs: 
        os.makedirs(dir_path, exist_ok=True) 
        print(f"directory is created at {dir_path}")        
        
def create_text_file(path_to_file_name: str, text: str): 
    with open(path_to_file_name, 'w') as txt_file: 
        txt_file.write(text) 
        
        
        
def save_pickle_file(path_to_file_name: str, data):
    with open(path_to_file_name, 'wb') as f:
        pickle.dump(data, f)
    print(f"saved pickle file {path_to_file_name}")
    
def read_pickle_file(path_to_file_name: str):
    with open(path_to_file_name, 'rb') as f:
        encoder = pickle.load(f)
    return encoder 
    
    
def read_txt_file(path_to_file_name: str): 
    with open(path_to_file_name, 'r') as read_file:
        text = read_file.read().replace('\n', ' ')
    return text 
    


def save_local_df(data, data_path: str):
    data.to_csv(data_path, index=False) 
    print(f"data is saved at {data_path}")
    
def save_reports(report:dict, report_path:str):
    with open(report_path, 'w') as f: 
        json.dump(report, f, indent=4)
    print(f"reports are saved at {report_path}")