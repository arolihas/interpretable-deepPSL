# Input layer: (128, 400, 20)
# Mask layer: (128, 400)
# DimshuffleLayer layer: (128, 20, 400) #use permute
# Convolutional layer size 3: (128, 10, 400)
# Convolutional layer size 5: (128, 10, 400)
# Concatenated convolutional layers: (128, 20, 400)
# Final convolutional layer: (128, 20, 400)
# Second DimshuffleLayer layer: (128, 400, 20) #use permute
# Forward LSTM layer: (128, 400, 15)
# Backward LSTM layer: (128, 400, 15)# Concatenated hidden states: (128, 400, 30)
# Attention layer: (128, 2, 30)
# Last decoding step: (128, 30) slice 
# Dense layer: (128, 30)
# Output layer: (128, 10)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DeepLoc(nn.Module):
    def __init__(self, batch_size=128, seq_len=1000, n_feat=20, n_hid=15, n_class=10, learning_rate=0.0025, n_filters=10, dropout=0.5):
        super(DeepLoc, self).__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.n_feat = n_feat
        self.n_hid = n_hid
        self.n_class = n_class
        self.lr = learning_rate
        self.n_filt = n_filters
        self.drop_prob = dropout

        self.conv3 = nn.Conv1d(in_channels=self.n_feat, out_channels=self.n_filt, kernel_size=3, padding=1, stride=1)
        self.conv5 = nn.Conv1d(in_channels=self.n_feat, out_channels=self.n_filt, kernel_size=5, padding=2, stride=1)
        self.convF = nn.Conv1d(in_channels=self.n_feat, out_channels=self.n_filt*2, kernel_size=3, padding=1, stride=1)
        self.bilstm = nn.LSTM(input_size=self.seq_len, num_layers=2, hidden_size=self.n_hid, bidirectional=True)
        self.att = nn.MultiheadAttention(embed_dim=self.n_hid*2, num_heads=2)
        self.dense = nn.Linear(in_features=self.n_hid*2, out_features=self.n_hid*2)
        self.dropout = nn.Dropout(p=self.drop_prob)
        self.out = nn.Linear(in_features=self.n_hid*2, out_features=self.n_class)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(0)

    def forward(self, x, mask):
        x = x.permute([0,2,1])
        conv_3 = self.relu(self.conv3(x))
        conv_5 = self.relu(self.conv5(x))
        conv_cat = torch.cat((conv_3, conv_5), dim=1)
        conv_f = self.relu(self.convF(conv_cat))
        # masked = conv_f[mask.long()]
        # print(masked.size())
        lstm, _ = self.bilstm(conv_f)
        seq = torch.tanh(lstm)
        att_w, _ = self.att(seq, seq, seq)
        sliced = att_w[:,-1,:]
        ff_layer = self.relu(self.dropout(self.dense(sliced)))
        logits = self.out(ff_layer)
        out = self.softmax(logits)
        return out
        