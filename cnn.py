import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np

import data

# DATAFILE_DIR = 'Task1/U1S1.TXT'

class CNNmodel():
    def __init__(self):
        self.cnn = nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.AdaptiveMaxPool1d(1)
        )
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
        )

    def forward(self, x):
        """

        :param x: shape (batch_size, seq_len, 4)
        :return:
        """
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = self.embedding(x)
        return x

if __name__ == "__main__":
    signatures_data = data.load_data()
    np.random.shuffle(signatures_data)
    sequence = signatures_data[0]

    x = sequence[:, 0]
    y = sequence[:, 1]
    t = sequence[:, 2]
    pen = sequence[:, 3]

    # Normalizacja
    x = (x - np.min(x)) / (np.ptp(x) + 1e-8)
    y = (y - np.min(y)) / (np.ptp(y) + 1e-8)
    dt = np.diff(t, prepend=t[0]) / (np.ptp(t) + 1e-8)

    sequence = np.stack([x, y, dt, pen], axis=1)  # shape (seq_len, 4)
    sequence = torch.tensor(sequence, dtype=torch.float32)
    sequence = sequence.unsqueeze(0)

    # training_dataset = signatures_data[0:int(len(signatures_data) * 0.8)]
    # testing_dataset = signatures_data[int(len(signatures_data) * 0.8):]

    # test
    cnn = CNNmodel()
    embedding = cnn.forward(sequence)
    print(embedding)
