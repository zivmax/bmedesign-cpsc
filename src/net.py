import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NetCore(nn.Module):
    def __init__(self, input_length, embedding_dim, kernel_sizes, num_filters, drop_out):
        super(NetCore, self).__init__()

        self.conv_layers = nn.ModuleList([
            nn.Conv1d(in_channels=1, out_channels=num_filters, kernel_size=k)
            for k in kernel_sizes
        ])

        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)

        reduced_size = input_length // 16

        self.fc1 = nn.Linear(len(kernel_sizes) * num_filters * reduced_size, 256)
        self.fc_embedding = nn.Linear(256, embedding_dim)

        self.dropout = nn.Dropout(drop_out)
        self.batch_norm1 = nn.BatchNorm1d(len(kernel_sizes) * num_filters)
        self.batch_norm2 = nn.BatchNorm1d(256)

    def forward(self, x):
        conv_results = []
        for conv in self.conv_layers:
            conv_results.append(F.relu(conv(x)))

        x = torch.cat(conv_results, dim=1)
        x = self.batch_norm1(x)
        x = self.pool1(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.batch_norm2(x)
        x = self.dropout(x)

        embedding = self.fc_embedding(x)
        return embedding
    

class CNN:
    def __init__(self, input_length, 
                 embedding_dim=64, 
                 kernel_sizes=[3, 5, 7], 
                 num_filters=6, 
                 drop_out=0.5,
                 optimizer_lr=0.001,
                 device='cuda'if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.embedding_dim = embedding_dim
        self.model = NetCore(input_length, embedding_dim, kernel_sizes, num_filters, drop_out).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), optimizer_lr)

    def train(self, df, 
              target,
              batch_size=64,
              num_epochs=50,
              learning_rate=0.001,
              weight_decay=1e-5,
              train_size=0.8,
              early_stopping=10):
        pass
        