import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import optim
import torch.nn.functional as F
import torch.nn as nn

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import seaborn as sns

from utils.data_loader import doc_classifier_dataloader
from utils.metrics_plots import get_metrics, save_graphs_ML, save_graphs_DL
from utils.DL_Model_util import DLModel

import tqdm
import time

torch.cuda.empty_cache()


# Hyperparams
seq_length = 300
BATCH_SIZE = 32
output_size = 20
HIDDEN_DIM = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 0.001

# Send these as lists
N_LAYERSS = [8]  # [2, 4, 10]
MODEL_TYPES = ['LSTM']  # , 'GRU', 'RNN']
EMBEDDING_DIMS = [50, 100]
DROP_PROBS = [0.6, 0.8]
EPOCHSS = [25]  # , 10, 20, 50]
BIDIRECTIONALS = [True]
TAKE_OUTPUT_OF_ALL_LAYERS_OF_MODELS = [True, False]












train = pd.read_csv(data_source + '/' + data_source + '_train.csv')
test = pd.read_csv(data_source + '/' + data_source + '_test.csv')

train = data_preprocessor.transform(train)
test = data_preprocessor.transform(test)

unique_categories = list(train['category'].unique())

# To find the avg amount of words in a sequence for the dataset
X_train, y_train = feature_extraction.transform(train, seq_length=SEQ_LENGTH, model_type='DL', data_type='train')
X_test, y_test = feature_extraction.transform(test, seq_length=SEQ_LENGTH, model_type='DL', data_type='test')
print('Train & Test shapes', X_train.shape, y_train.shape, X_test.shape, y_test.shape)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42, stratify=y_train)

# This onehot is used just for calculating roc_auc score
category_onehot = OneHotEncoder()
category_onehot.fit_transform(pd.DataFrame(y_train))


train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
valid_data = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
test_data = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

train_loader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE, drop_last=True)
valid_loader = DataLoader(valid_data, shuffle=True, batch_size=BATCH_SIZE, drop_last=True)
test_loader = DataLoader(test_data, shuffle=True, batch_size=BATCH_SIZE, drop_last=True)

# Instantiate the model w/ hyperparams
vocab_to_int = feature_extraction.count_encoding
vocab_size = len(vocab_to_int) + 2  # +1 for the 0 padding, +1 for unknowns
print('The whole vocab size is', vocab_size)




train_grid = [('LSTM', 2, True, 25, False, 0.8, 100, ''),
              ('LSTM', 4, True, 25, False, 0.8, 100, ''),
              ('LSTM', 4, False, 25, False, 0.8, 100, ''),
              ('GRU', 4, True, 25, False, 0.6, 100, ''),
              ('GRU', 4, True, 25, False, 0.8, 100, '')]



def train(model):
    global train_loss_perEpoch, train_accuracy_perEpoch, valid_loss_perEpoch,  valid_accuracy_perEpoch, max_acc, df_with_acc_loss

    print(MODEL_TYPE, N_LAYERS, BIDIRECTIONAL, EPOCHS, TAKE_OUTPUT_OF_ALL_LAYERS_OF_MODEL, DROP_PROB, EMBEDDING_DIM,
          WORD_EMBEDDING)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for e in range(EPOCHS):
        print('training epoch', e)
        if e >= 10 and valid_loss_perEpoch[-1] > 1:
            print('Stopped training as valid loss is bad by the end of 10th epoch')
            break

        t1 = time.time()
        model.train()
        sum_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, (data, targets) in enumerate(train_loader):
            data = data.long().to(DEVICE)
            targets = targets.long().to(DEVICE)

            # print('data shape', data.shape, torch.max(data), torch.min(data))
            preds = model(data)
            # print(scores.shape, targets.shape)
            # print(scores[0], targets[0])
            # scores = scores.type(torch.LongTensor)
            # targets = targets.type(torch.LongTensor)
            # print(preds.shape, targets.shape)
            loss = criterion(preds, targets)

            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent or adam step
            optimizer.step()

            sum_loss += loss.item() * targets.shape[0]
            correct += (torch.max(preds, 1)[1] == targets).float().sum()
            total += targets.shape[0]

            # print(batch_idx)

        val_loss, val_acc = validation_metrics(model)
        test_results = test_metrics(model)
        test_loss, test_acc = test_results[:2]

        train_loss_perEpoch.append(sum_loss / total)
        train_accuracy_perEpoch.append((correct / total).item())
        valid_loss_perEpoch.append(val_loss)
        valid_accuracy_perEpoch.append((val_acc).item())
        test_loss_perEpoch.append(test_loss)
        test_accuracy_perEpoch.append((test_acc).item())

        # if e%1==0:
        print("train loss %.3f, train accuracy %.3f, val loss %.3f, val accuracy %.3f, test loss %.3f, test accuracy %.3f, \
              test precision %.3f, test recall %.3f, test f1-score %.3f, test roc_auc_score %.3f" % (
            sum_loss / total, correct / total, val_loss, val_acc, test_loss, test_acc, test_results[2]['precision'],
            test_results[2]['recall'], test_results[2]['f1-score'], test_results[2]['roc_auc_score']))
        print(time.time() - t1)

        if (('B' if BIDIRECTIONAL else '') + MODEL_TYPE + str(N_LAYERS) not in model_layer_dct):
            model_layer_dct[('B' if BIDIRECTIONAL else '') + MODEL_TYPE + str(N_LAYERS)] = val_loss
            checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(checkpoint, cur_dir + '/Weights/' + \
                       ('B' if BIDIRECTIONAL else '') + MODEL_TYPE + '_' + str(N_LAYERS) + 'Layers')
        else:
            if val_loss < model_layer_dct[('B' if BIDIRECTIONAL else '') + MODEL_TYPE + str(N_LAYERS)]:
                model_layer_dct[('B' if BIDIRECTIONAL else '') + MODEL_TYPE + str(N_LAYERS)] = val_loss
                checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
                torch.save(checkpoint, cur_dir + '/Weights/' + \
                           ('B' if BIDIRECTIONAL else '') + MODEL_TYPE + '_' + str(N_LAYERS) + 'Layers')

    df_with_acc_loss.append(
        [('' if BIDIRECTIONAL == False else 'B') + MODEL_TYPE, EPOCHS, e, N_LAYERS, EMBEDDING_DIM,
         SEQ_LENGTH, DROP_PROB, WORD_EMBEDDING, TAKE_OUTPUT_OF_ALL_LAYERS_OF_MODEL,
         max(train_accuracy_perEpoch), max(valid_accuracy_perEpoch), max(test_accuracy_perEpoch),
         min(train_loss_perEpoch), min(valid_loss_perEpoch), min(test_loss_perEpoch),
         train_accuracy_perEpoch.index(max(train_accuracy_perEpoch)),
         valid_accuracy_perEpoch.index(max(valid_accuracy_perEpoch)),
         test_accuracy_perEpoch.index(max(test_accuracy_perEpoch)),
         train_loss_perEpoch.index(min(train_loss_perEpoch)),
         valid_loss_perEpoch.index(min(valid_loss_perEpoch)),
         test_loss_perEpoch.index(min(test_loss_perEpoch))])




def validation_metrics(model):
    model.eval()
    correct = 0
    total = 0
    sum_loss = 0.0

    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(valid_loader):
            data = data.long().to(DEVICE)
            targets = targets.long().to(DEVICE)
            preds = model(data)
            loss = F.cross_entropy(preds, targets)
            correct += (torch.max(preds, 1)[1] == targets).float().sum()
            total += targets.shape[0]
            sum_loss += loss.item() * targets.shape[0]
    return sum_loss / total, correct / total




def test_metrics(model):
    model.eval()
    correct = 0
    total = 0
    sum_loss = 0.0
    confusion_matrix = np.zeros((len(unique_categories), len(unique_categories)))

    pred_labels, true_labels = [], []

    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(test_loader):
            data = data.long().to(DEVICE)
            targets = targets.long().to(DEVICE)
            preds = model(data)
            loss = F.cross_entropy(preds, targets)

            for i in torch.max(preds, 1)[1]:
                pred_labels.append(i.cpu())

            for i in targets:
                true_labels.append(i.cpu())

            correct += (torch.max(preds, 1)[1] == targets).float().sum()
            total += targets.shape[0]
            sum_loss += loss.item() * targets.shape[0]

            # for confusion matrix
            try:
                for t, p in zip(targets.view(-1), torch.max(preds, 1)[1].view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1

            except:
                print('here we got error while calculating confusion matrix', targets.shape)
                print(torch.max(targets.view(-1)), torch.max(torch.max(preds, 1)[1]), torch.min(targets.view(-1)),
                      torch.min(preds.view(-1)))

    pred_onehot = category_onehot.transform(pd.DataFrame(pred_labels)).todense()
    true_onehot = category_onehot.transform(pd.DataFrame(true_labels)).todense()
    metric_dct = {'confusion_matrix': confusion_matrix,
                  'precision': precision_score(true_labels, pred_labels, average='weighted'),
                  'recall': recall_score(true_labels, pred_labels, average='weighted'),
                  'f1-score': f1_score(true_labels, pred_labels, average='weighted'),
                  'roc_auc_score': roc_auc_score(true_onehot, pred_onehot, average='weighted', multi_class='ovr')}
    return sum_loss / total, correct / total, metric_dct




if __name__=='__main__':

    if TRAIN_TYPE == 'train':
        model_layer_dct = {}

        # This is a nested list which contains the Model, Expected Epochs, Run Epoch,  Layers, Embed Dim,
        # Seq Length, Dropout, Embed Weights, Fully connected, Train Accuracy, Valid accuracy, Test accuracy, Train Loss, Valid Loss, Test Loss
        df_with_acc_loss = []

        for (MODEL_TYPE, N_LAYERS, BIDIRECTIONAL, EPOCHS, TAKE_OUTPUT_OF_ALL_LAYERS_OF_MODEL, DROP_PROB, EMBEDDING_DIM,
             WORD_EMBEDDING) in train_grid:
            print(MODEL_TYPE, N_LAYERS, BIDIRECTIONAL, EPOCHS, TAKE_OUTPUT_OF_ALL_LAYERS_OF_MODEL, DROP_PROB, EMBEDDING_DIM, WORD_EMBEDDING)

            if WORD_EMBEDDING == 'GloVe':
                word_vecs = GloVe.load_glove_vectors(EMBEDDING_DIM)
                pretrained_embed_weights, vocab, vocab2index = GloVe.get_emb_matrix(word_vecs, vocab_to_int, emb_size=EMBEDDING_DIM)
            else:
                pretrained_embed_weights = ''

            model = LSTM.DocLSTM(vocab_size, output_size, EMBEDDING_DIM, HIDDEN_DIM,
                                 N_LAYERS, drop_prob=DROP_PROB, batch_first=True, model_type=MODEL_TYPE, device=DEVICE,
                                 bidirectional=BIDIRECTIONAL, word_embedding=WORD_EMBEDDING,
                                 glove_weights=pretrained_embed_weights, seq_length=SEQ_LENGTH,
                                 take_all_layers_output=TAKE_OUTPUT_OF_ALL_LAYERS_OF_MODEL)
            model.to(DEVICE)
            print(model)

            train_loss_perEpoch, train_accuracy_perEpoch, valid_loss_perEpoch, valid_accuracy_perEpoch, \
            test_loss_perEpoch, test_accuracy_perEpoch = [], [], [], [], [], []

            # Call train function on models
            train(model)

            model_weights_file = cur_dir + '/Weights/' + \
                                 ('B' if BIDIRECTIONAL else '') + MODEL_TYPE + '_' + str(N_LAYERS) + 'Layers'
            test_results = test_metrics(model)
            print('Test results: ', test_results[0], test_results[1], max(test_accuracy_perEpoch))
            print(train_loss_perEpoch, valid_loss_perEpoch)
            print(train_accuracy_perEpoch, valid_accuracy_perEpoch)

            # Plot the graphs
            whole_model_name = ('' if BIDIRECTIONAL == False else 'B') + \
                               MODEL_TYPE + '_' + str(EPOCHS) + 'Epochs_' + str(N_LAYERS) + 'Layers_' + str(EMBEDDING_DIM) + 'Embed_' + \
                               str(SEQ_LENGTH) + 'SeqLength_' + str(DROP_PROB) + 'DropProb_' + \
                               str(TAKE_OUTPUT_OF_ALL_LAYERS_OF_MODEL) + 'TAKE_OUTPUT_OF_ALL_LAYERS_OF_MODEL_' + WORD_EMBEDDING
            PlotGraphs.save_graphs(test_results[2], whole_model_name, unique_categories,
                                   train_loss_perEpoch, valid_loss_perEpoch, test_loss_perEpoch,
                                   train_accuracy_perEpoch, valid_accuracy_perEpoch, test_accuracy_perEpoch)

        # Saving the model info into excel
        print('Model dictionary: ', model_layer_dct)
        pd.DataFrame(df_with_acc_loss, columns=['Model', 'Expected Epochs', 'Run Epoch', 'Layers', 'Embed Dim',
                                                'Seq Length', 'Dropout', 'WORD_EMBEDDING', 'Is all the layers',
                                                'Train Accuracy', 'Valid accuracy', 'Test accuracy',
                                                'Train Loss', 'Valid Loss', 'Test Loss',
                                                'Train Accuracy Index', 'Valid accuracy Index', 'Test accuracy Index',
                                                'Train Loss Index', 'Valid Loss Index', 'Test Loss Index']). \
            to_csv('Execution Results1.csv', index=False)
        print('Training done no errors', '\n\n\n')



    elif TRAIN_TYPE == 'predict':
        MODEL_TYPE = 'GRU'
        EMBEDDING_DIM = 100
        N_LAYERS = 4
        DROP_PROB = 0.6
        TAKE_OUTPUT_OF_ALL_LAYERS_OF_MODEL = False
        BIDIRECTIONAL = True
        criterion = nn.CrossEntropyLoss()
        model = LSTM.DocLSTM(vocab_size, output_size, EMBEDDING_DIM, HIDDEN_DIM,
                             N_LAYERS, drop_prob=DROP_PROB, batch_first=True, model_type=MODEL_TYPE, device=DEVICE,
                             bidirectional=BIDIRECTIONAL, word_embedding='',
                             glove_weights='', seq_length=SEQ_LENGTH,
                             take_all_layers_output=TAKE_OUTPUT_OF_ALL_LAYERS_OF_MODEL)
        model.to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        model_weights_file = cur_dir + '/Weights/' + \
                             ('B' if BIDIRECTIONAL else '') + MODEL_TYPE + '_' + str(N_LAYERS) + 'Layers'
        load_check_point = torch.load(model_weights_file, map_location=torch.device('cpu'))
        model.load_state_dict(load_check_point['state_dict'])
        optimizer.load_state_dict(load_check_point['optimizer'])

        pred_df = pd.DataFrame(columns=['text', 'category'])
        pred_dir = cur_dir + '/Predict/'

        remove_lines_starting = ('message-id', 'newsgroups', 'path', 'from', 'references', 'sender', 'lines', 'date', 'xref')
        for file in listdir(pred_dir):
            cur_file = os.path.join(pred_dir, file)
            if isfile(cur_file):
                with open(cur_file, 'r', encoding="utf8") as read_file:
                    data = read_file.read().replace('\n', ' ')
                    #data = ' '.join([line.replace('\n', '') for line in read_file.readlines() if not line.lower().startswith(remove_lines_starting)])
                pred_df = pd.concat([pred_df, pd.DataFrame([[data, file.replace('.txt','')]], columns=['text', 'category'])],
                                    ignore_index=True, axis=0)

        print('Prediction size', pred_df.shape)
        # print(pred_df)

        pred_df = data_preprocessor.transform(pred_df)
        X_pred, y_pred = feature_extraction.transform(pred_df, seq_length=SEQ_LENGTH, model_type='DL', data_type='predict')

        X_pred = list(X_pred)
        y_pred = list(pred_df['category'])

        model.eval()
        with torch.no_grad():
            for i, doc in enumerate(X_pred):
                # print(torch.Tensor(doc).unsqueeze(0).shape)
                data = torch.Tensor(doc).unsqueeze(0).long().to(DEVICE)
                preds = model(data)
                #print(torch.max(preds, 1)[1].item())
                print(#torch.max(preds, 1)[1],
                      feature_extraction.label_encoder.inverse_transform(torch.max(preds, 1)[1].cpu().detach().numpy()),
                      y_pred[i])

        # test_results = test_metrics(model)
        # print('Test results: ', test_results[0], test_results[1])





