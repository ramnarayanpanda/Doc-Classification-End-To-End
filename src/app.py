import streamlit as st
import streamlit.components.v1 as stc

# File Processing Pkgs
import pandas as pd
import mlflow
import predict
from io import StringIO
import warnings
warnings.filterwarnings('ignore')
import torch
import logging
import argparse
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def show_output(text, df, text_file, model_type):
    if model_type == 'ML':
        run_id = df.loc[df['metrics.accuracy'].idxmax()]['run_id']
        model = mlflow.sklearn.load_model("runs:/" + run_id + "/model")
        seq_length = None

    elif model_type == 'DL':
        run_id = df.loc[df['metrics.accuracy'].idxmin()]['run_id']
        # model = mlflow.pytorch.load_model("runs:/" + run_id + "/model")
        model_path = os.path.join('mlruns/0/', run_id, 'artifacts/model/data/model.pth')
        model = torch.load(model_path, map_location =device)
        seq_length = int(df.loc[df['metrics.accuracy'].idxmin()]['params.seq_length'])

    elif model_type == 'BERT':
        run_id = df.loc[df['metrics.accuracy'].idxmax()]['run_id']
        model_path = os.path.join('mlruns/0/', run_id, 'artifacts/model/data/model.pth')
        model = torch.load(model_path, map_location =device)
        # seq_length = int(df.loc[df['metrics.accuracy'].idxmax()]['params.seq_length'])
        seq_length = 300

    print('\n', run_id, '\n')

    output = predict.predict(text, model, model_type, seq_length)
    print(output)

    flag = 0
    text_to_print = ""
    for line in text.split('\n'):
        if line.startswith('Lines'):
            flag = 1
        elif flag==1:
            text_to_print += line

    if text_to_print=="":
        text_to_print = text

    html_str = '''<p style="font-family:Courier; color:LightGreen;font-size: 20px;">Doc Name</p>'''
    st.markdown(html_str, unsafe_allow_html=True)
    st.write(text_file.name )

    html_str = '''<p style="font-family:Courier; color:LightGreen;font-size: 20px;">Doc Content</p>'''
    st.markdown(html_str, unsafe_allow_html=True)
    st.write(text_to_print)

    html_str = '''<p style="font-family:Courier; color:LightGreen;font-size: 20px;">Doc Type</p>'''
    st.markdown(html_str, unsafe_allow_html=True)
    st.write(output[0])
    st.write(" ")

    # font_size = 20
    # variable_output = 'a varaible'
    # html_str = f"""<style> p.a {{font: bold color:Blue {font_size}px Courier;}} </style>
    #                <p class="a">{variable_output}</p>"""
    # st.markdown(html_str, unsafe_allow_html=True)

    # line = "-"*50
    # s = 'Here you are'
    # html_str = '''<p style="font-family:Courier; color:Blue; font-size: 40px;">==========================</p>
    #               <p class="a">{{variable_output}}</p>'''
    # st.markdown(html_str, unsafe_allow_html=True)

    # html_str = """<html>
    # <body>
    #     <p>Here is my variable: {{ variable }}</p>
    # </body>
    # </html>"""
    # st.markdown(html_str, unsafe_allow_html=True)

    st.write(" ")





def main():
    st.title("Document Classification")
    model = ["Home", "Random Forest", "Naive Bayes", "Stacking RF+NB",
             "Bidirectional GRU", "GRU", "Bidirectional LSTM", "LSTM", "BERT"]
    choice = st.sidebar.selectbox("Models",model)
    text_files = st.file_uploader("Upload Text File",type=["txt"], accept_multiple_files=True)

    if len(text_files)>0:
        text_files = text_files[::-1]

        for text_file in text_files:
            stringio = StringIO(text_file.getvalue().decode("utf-8"))
            text = stringio.read()

            print(choice)

            if choice == "Random Forest":
                html_str = '''<p style="font-family:Courier; color:Red; font-size: 30px;">Random Forest</p>'''
                st.markdown(html_str, unsafe_allow_html=True)
                # st.subheader("Random Forest")
                df = mlflow.search_runs(filter_string="params.model_name='Random Forest'")
                # print(df)
                show_output(text, df, text_file, model_type='ML')

            elif choice == "Naive Bayes":
                html_str = '''<p style="font-family:Courier; color:Red; font-size: 30px;">Naive Bayes</p>'''
                st.markdown(html_str, unsafe_allow_html=True)
                # st.subheader("Naive Bayes")
                df = mlflow.search_runs(filter_string="params.model_name='Naive Bayes'")
                show_output(text, df, text_file, model_type='ML')

            elif choice == "Stacking RF+NB":
                html_str = '''<p style="font-family:Courier; color:Red; font-size: 30px;">Stacking RF+NB</p>'''
                st.markdown(html_str, unsafe_allow_html=True)
                # st.subheader("Stacking RF+NB")
                df = mlflow.search_runs(filter_string="params.model_name='RF+MNB Stack'")
                show_output(text, df, text_file, model_type='ML')

            elif choice == "Bidirectional GRU":
                html_str = '''<p style="font-family:Courier; color:Red; font-size: 30px;">Bidirectional GRU</p>'''
                st.markdown(html_str, unsafe_allow_html=True)
                # st.subheader("Bidirectional GRU")
                df = mlflow.search_runs(filter_string="params.model_name='BGRU'")
                show_output(text, df, text_file, model_type='DL')

            elif choice == "GRU":
                html_str = '''<p style="font-family:Courier; color:Red; font-size: 30px;">GRU</p>'''
                st.markdown(html_str, unsafe_allow_html=True)
                # st.subheader("GRU")
                df = mlflow.search_runs(filter_string="params.model_name='GRU'")
                show_output(text, df, text_file, model_type='DL')

            elif choice == "Bidirectional LSTM":
                html_str = '''<p style="font-family:Courier; color:Red; font-size: 30px;">Bidirectional LSTM</p>'''
                st.markdown(html_str, unsafe_allow_html=True)
                # st.subheader("Bidirectional LSTM")
                df = mlflow.search_runs(filter_string="params.model_name='BLSTM'")
                show_output(text, df, text_file, model_type='DL')

            elif choice == "LSTM":
                html_str = '''<p style="font-family:Courier; color:Red; font-size: 30px;">LSTM</p>'''
                st.markdown(html_str, unsafe_allow_html=True)
                # st.subheader("LSTM")
                df = mlflow.search_runs(filter_string="params.model_name='LSTM'")
                show_output(text, df, text_file, model_type='DL')

            elif choice == "BERT":
                html_str = '''<p style="font-family:Courier; color:Red; font-size: 30px;">BERT</p>'''
                st.markdown(html_str, unsafe_allow_html=True)
                # st.subheader("BERT")
                df = mlflow.search_runs(filter_string="params.model_name='distill-bert-uncased'")
                show_output(text, df, text_file, model_type='BERT')



if __name__ == '__main__':
    main()
