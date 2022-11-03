import torch.nn as nn
import torch
import numpy as np


class DLModel(nn.Module):
    """The RNN model that will be used to perform Document classification
    """
    # removed glove_weights=None, for now 
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim,
                 n_layers, drop_prob=0.3, batch_first=True, model_name='LSTM', device='cuda', bidirectional=False,
                 word_embedding='', seq_length=400, take_all_layers_output=False):
        super(DLModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.drop_prob = drop_prob
        self.device = device
        self.model_name = model_name
        self.take_output_of_all_lstm = take_all_layers_output

        if bidirectional==True:
            self.n_layers = n_layers*2
        else: self.n_layers = n_layers

        # Boy, Girl -> 0.9
        # King, Queen -> 0.85
        # sorority, fraternity -> 0.05
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        # if word_embedding=='GloVe':
        #     self.embed.weight.data.copy_(torch.from_numpy(glove_weights))
            # self.embed.weight.requires_grad = False  ## freeze embeddings

        self.rnn = nn.RNN(embedding_dim, hidden_dim, n_layers, batch_first=batch_first, bidirectional=bidirectional)
        self.gru = nn.GRU(embedding_dim, hidden_dim, n_layers, batch_first=batch_first, bidirectional=bidirectional)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=batch_first, bidirectional=bidirectional)
        self.dropout = nn.Dropout(drop_prob)
        if bidirectional==True:
            if take_all_layers_output:  self.fc = nn.Linear(hidden_dim * seq_length * 2, output_size)
            else:  self.fc = nn.Linear(hidden_dim * 2, output_size)
        else:
            if take_all_layers_output:  self.fc = nn.Linear(hidden_dim * seq_length, output_size)
            else:  self.fc = nn.Linear(hidden_dim, output_size)



    def forward(self, x):
        batch_size = x.size(0)
        # h0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device)
        # c0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device)

        #print('x shape', x.shape)
        x = self.embed(x)
        #print('here1', x.shape)

        if self.model_name=='LSTM':
            # out, (h1, c1) = self.lstm(x, (h0,c0))  # if we dont porovide h0, c0 by default it will be zeros
            out, (h1, c1) = self.lstm(x)
        elif self.model_name=='RNN':
            # out, h1 = self.rnn(x, h0)
            out, h1 = self.rnn(x)
        elif self.model_name=='GRU':
            # out, h1 = self.gru(x, h0)  #h1 is the size of units in the output layer of rnn'
            out, h1 = self.gru(x)

        # Do this if you want to take all the LSTM layers output and feed it to the fc layer
        if self.take_output_of_all_lstm:
            out = out.reshape(out.shape[0], -1)
            out = self.dropout(out)
            out = self.fc(out)
            # print(out.shape)

        # If you want to take the output of the last layer only then do this
        else:
            out = self.dropout(out)
            out = self.fc(out[:, -1, :])

        return out
