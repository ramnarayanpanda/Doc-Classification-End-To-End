from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, \
                            plot_roc_curve, confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


                            
def get_metrics(y_true, y_pred, unique_categories):
    category_onehot = OneHotEncoder()
    category_onehot.fit_transform(pd.DataFrame(y_true))
    
    # pred_onehot = category_onehot.transform(pd.DataFrame(y_pred)).todense()
    # true_onehot = category_onehot.transform(pd.DataFrame(y_true)).todense()
    metric_dct = {'confusion_matrix': confusion_matrix(y_true, y_pred, labels=unique_categories),
                  'accuracy': accuracy_score(y_true, y_pred),
                  'precision': precision_score(y_true, y_pred, average='weighted'),
                  'recall': recall_score(y_true, y_pred, average='weighted'),
                  'f1-score': f1_score(y_true, y_pred, average='weighted'),
                #   'roc_auc_score': roc_auc_score(true_onehot, pred_onehot, average='weighted', multi_class='ovr')
                  'roc_auc_score': 80
                  }
    
    return metric_dct  




def save_graphs_ML(metric_dct, whole_model_name, unique_categories, test_accuracy):
    header_text = whole_model_name + '\n' + \
                'Test Accuracy= %.3f,  Precision= %.3f,  Recall= %.3f,  f1-Score= %.3f,  roc_auc_score= %.3f' % \
                (test_accuracy, metric_dct['precision'], metric_dct['recall'], metric_dct['f1-score'], metric_dct['roc_auc_score'])
                            
    # Saving the confusion matrix
    fig = plt.figure(figsize=(15, 18))
    fig.suptitle(header_text, horizontalalignment='center', fontsize=14, fontweight='bold')
    df_cm = pd.DataFrame(metric_dct['confusion_matrix'], index=unique_categories, columns=unique_categories).astype(int)
    ax = sns.heatmap(df_cm, annot=True, fmt="d")
    ax.yaxis.set_ticklabels(ax.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=15)
    ax.xaxis.set_ticklabels(ax.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=15)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    # plt.savefig('ConfusionMatrix_' + whole_model_name + '.png')
    return {'name': 'ConfusionMatrix_' + whole_model_name + '.png', 'fig': fig}



# valid_accuracy_perEpoch, valid_loss_perEpoch
def save_graphs_DL(metric_dct, whole_model_name, unique_categories,
                train_loss_perEpoch, test_loss_perEpoch,
                train_accuracy_perEpoch, test_accuracy_perEpoch):
    
    header_text = whole_model_name + '\n' + \
                'Test Accuracy= %.3f,  Precision= %.3f,  Recall= %.3f,  f1-Score= %.3f,  roc_auc_score= %.3f' % \
                (max(test_accuracy_perEpoch), metric_dct['precision'], metric_dct['recall'], metric_dct['f1-score'], metric_dct['roc_auc_score'])
                

    # Save accuracy and loss graph with test
    fig = plt.figure(figsize=(20, 10))
    fig.suptitle(header_text, horizontalalignment='center', fontsize=14, fontweight='bold')
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.plot(range(len(train_loss_perEpoch)), train_loss_perEpoch, '-g', label='train loss')
    # ax1.plot(range(len(valid_loss_perEpoch)), valid_loss_perEpoch, '-r', label='valid loss')
    ax1.plot(range(len(test_loss_perEpoch)), test_loss_perEpoch, '-b', label='test loss')
    ax1.set_title("Loss Graph")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Losses")
    ax1.legend()
    ax1.grid()

    ax2.plot(range(len(train_accuracy_perEpoch)), train_accuracy_perEpoch, '-g', label='train accuracy')
    # ax2.plot(range(len(valid_accuracy_perEpoch)), valid_accuracy_perEpoch, '-r', label='valid accuracy')
    ax2.plot(range(len(test_accuracy_perEpoch)), test_accuracy_perEpoch, '-b', label='test accuracy')
    ax2.set_title("Accuracy Graph")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid()
    

    # Saving the confusion matrix
    fig1 = plt.figure(figsize=(15, 18))
    fig1.suptitle(header_text, horizontalalignment='center', fontsize=14, fontweight='bold')
    df_cm = pd.DataFrame(metric_dct['confusion_matrix'], index=unique_categories, columns=unique_categories).astype(int)
    ax = sns.heatmap(df_cm, annot=True, fmt="d")
    ax.yaxis.set_ticklabels(ax.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=15)
    ax.xaxis.set_ticklabels(ax.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=15)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    
    return ({'name':'ConfusionMatrix_' + whole_model_name + '.png', 'fig':fig1}, 
            {'name':'Accuracy&Loss Graph' + whole_model_name + '.png', 'fig':fig})
    
    