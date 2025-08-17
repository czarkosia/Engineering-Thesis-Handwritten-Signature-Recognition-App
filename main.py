from models import cae
from preprocessing import load_data, resample, normalize

import numpy as np
import torch

MAX_SEQUENCE_LENGTH = 128

if __name__ == "__main__":
    print("Loading data...")
    signatures_data, data_info = load_data.load_svc2004()
    print("Data loaded.")
    print("Resampling...")
    signatures_data = resample.resample_all(signatures_data, MAX_SEQUENCE_LENGTH)


    padded = load_data.pad_sequences(signatures_data)
    normalized = normalize.normalize(padded, 2, 7)
    train_data, test_data, train_info, test_info = load_data.train_test_split(normalized, data_info)

    input_data = torch.tensor(train_data, dtype=torch.float32)

    print("Building model...")
    print(input_data.size())
    cae = cae.CaeModel()
    output = cae.forward(input_data)
    print(output.size())
    print("Model built.")
    # true_mean = []
    # false_mean = []
    # for i in range(test_data.shape[0]):
    #     for j in range(test_data.shape[0]):
    #         if j != i:
    #             test_samples = torch.tensor(np.array([test_data[i],test_data[j]]), dtype=torch.float32)
    #             if_same = test_info[i][0] == test_info[j][0] and test_info[i][1] == test_info[j][1] == 1
    #             prediction = cnn.predict(test_samples)
    #             if if_same: true_mean.append(prediction)
    #             else: false_mean.append(prediction)
    # print("Distance for similar samples:", np.mean(true_mean), "std: ", np.std(true_mean))
    # print("Distance for different samples:", np.mean(false_mean), "std: ", np.std(false_mean))
