import streamlit as st
import streamlit.components.v1 as stc
from io import StringIO
import mlflow 
import os 
import shutil 

path = 'D:/MLRUNS/news_group_data/mlruns/0'
cnt = 0
for run in os.listdir(path):
    if os.path.isdir(os.path.join(path, run)):
        # if 'data' in os.listdir(os.path.join(path, run, 'artifacts', 'model')) and run not in ['f372e94b4e0642be860a8e3a4cd42487', '7c2fa4a8d6fd486a8b3efbdaada52406', 
        # 'cea1ea6367e94284abe80f9523de13b8', '399075862d0b465385f7abf27dcaaead']:
        if 'data' in os.listdir(os.path.join(path, run, 'artifacts', 'model')) and \
        run not in ['4e59d90fbe8d4923877292a42cff95a2', '0150166fbb8e4c43ae8663f02e973425', '1f67b5f9da4d4a819d0cad6f02b641bd', 'da228d7471624af9a0d3f75f80f21ca6', 'c93b1811d39344b29edaf448d955c299']:
            print(run)
            shutil.rmtree(os.path.join(path, run, 'artifacts', 'model'))
            os.mkdir(os.path.join(path, run, 'artifacts', 'model'))
            print(run)
            cnt+=1

print(cnt)



# from utils.common import create_directory, read_yaml, write_yaml, create_text_file, save_pickle_file, \
#                          read_pickle_file, read_txt_file
# import itertools 

# config_path = 'config/config.yaml'
# param_path = 'params.yaml'

# config = read_yaml(config_path)
# params = read_yaml(param_path)

# train_hyper_params_keys = list(params['models']['DL']['params'].keys())
# train_hyper_params_values = list(params['models']['DL']['params'].values())

# cnt = 0
# s = ""
# for train_hyper_params_value in list(itertools.product(*train_hyper_params_values)):
#     cnt+=1
#     if cnt%3!=0:
#         continue 
#     s += str(dict(zip(train_hyper_params_keys, train_hyper_params_value)))


# create_text_file('./dct_text.txt', s)