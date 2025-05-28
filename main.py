import numpy as np
import torch

import data
import cnn

if __name__ == "__main__":
    signatures_data, data_info = data.load_svc2004()
    data.show_sample(signatures_data[0])
    padded = data.pad_sequences(signatures_data)
    normalized = data.normalize(padded)
    train_data, test_data, train_info, test_info = data.train_test_split(normalized, data_info)

    # normalized_data = []
    # for sequence in signatures_data:
    #     x = sequence[:, 0]
    #     y = sequence[:, 1]
    #     t = sequence[:, 2]
    #     pen = sequence[:, 3]
    #
    #     # Normalizacja
    #     x = (x - np.min(x)) / (np.ptp(x) + 1e-8)
    #     y = (y - np.min(y)) / (np.ptp(y) + 1e-8)
    #     dt = np.diff(t, prepend=t[0]) / (np.ptp(t) + 1e-8)
    #
    #     sequence = np.stack([x, y, dt, pen], axis=0)
    #     print(sequence.shape)
    #     normalized_data.append(sequence)

    input_data = torch.tensor(train_data, dtype=torch.float32)
    print(input_data.shape)
    ...
    # sequence = sequence.unsqueeze(0)
    #
    # cnn = nn.Sequential(
    #     nn.Conv1d(4, 64, kernel_size=3, padding=1),
    #     nn.ReLU(),
    #     nn.BatchNorm1d(64),
    #     nn.MaxPool1d(2),
    # )
    # print(cnn)

    cnn = cnn.CnnModel(7)
    output = cnn.forward(input_data, train_info)
    # print(output.shape)