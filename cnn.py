import torch
import torch.nn as nn
import numpy as np

import data

class CnnModel(nn.Module):
    def __init__(self, in_channels):
        super(CnnModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=100, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Dropout(0.5),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.cnn_output = None

    def train(self, input_data, train_info):
        ...

    def forward(self, samples, samples_info):
        cnn_output = self.cnn(samples)
        print(cnn_output.shape)
        pairs, labels, cl = self.generate_pairs(cnn_output, samples_info)
        print(labels[0], cl[0])
        min_idx = cl.index(min(cl))
        max_idx = cl.index(max(cl))
        print('CL min:', min(cl), labels[min_idx])
        print('CL max:', max(cl), labels[max_idx])

    def cl_backward(self):
        ...

    def generate_pairs(self, samples, samples_info):
        pairs = []
        labels = []
        cl = []
        for i, seq1 in enumerate(samples):
            for j, seq2 in enumerate(samples):
                if i != j:
                    pairs.append((seq1, seq2))
                    if (samples_info[j, 0] == samples_info[i, 0]
                            and samples_info[j, 1] == 1
                            and samples_info[i, 1] == 1):
                        y = 1
                    else:
                        y = 0
                    labels.append(y)
                    cl.append(CnnModel.contrastive_loss(seq1, seq2, y))

        return pairs, labels, cl


    @staticmethod
    def contrastive_loss(seq1, seq2, y):
        distance = (seq1 - seq2).pow(2).sum().sqrt()
        return y*distance + (1 - y)*torch.relu(1 - distance)

