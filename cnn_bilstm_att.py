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

class DeepLoc(nn.Module):
    def __init__(self):
        self.batch_size = 128
        self.seq_len = 1000
        self.n_feat = 20
        self.n_hid = 15
        self.n_class = 10
        self.lr = 0.0025
        self.n_filt = 10
        self.drop_prob = 0.5
        self.conv3 = nn.Conv1d(in_channels=self.n_feat, out_channels=self.n_filt, kernel_size=3, padding='same', stride=1)
        self.conv5 = nn.Conv1d(in_channels=self.n_feat, out_channels=self.n_filt, kernel_size=5, padding='same', stride=1)
        self.convF = nn.Conv1d(in_channels=self.n_feat, out_channels=self.n_filt*2, kernel_size=3, padding='same', stride=1)
        self.bilstm = nn.LSTM(num_layers=2, hidden_size=self.n_hid, bidirectional=True)
        self.att = nn.MultiheadAttention(embed_dim=self.n_hid*2, num_heads=2, kdim=self.n_hid)
        self.dense = nn.Linear(in_features=self.n_hid*2, out_features=self.n_hid*2)
        self.out = nn.Linear(in_features=self.n_hid*2, out_features=self.n_class)

    def forward(self, x, mask):
        conv_3 = nn.ReLU(self.conv3(x))
        conv_5 = nn.ReLU(self.conv5(x))
        conv_cat = torch.cat((conv_3, conv_5), dim=-1)
        conv_f = nn.ReLU(self.convF(conv_cat))
        masked = conv_f[mask]
        lstm = F.tanh(self.bilstm(masked))
        att_w = self.att(lstm)
        sliced = att_w[:,-1,:]
        ff_layer = nn.ReLU(self.dense(sliced))
        logits = self.out(ff_layer)
        out = nn.Softmax(logits)
        return out
        