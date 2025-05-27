import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

DATABASE_NUMBER = 2 # 1 lub 2
FILE_DIR = ['Task1/', 'Task2/']
COLUMN_NAMES = [['x-coord', 'y-coord', 'time stamp', 'button status'],
                ['x-coord', 'y-coord', 'time stamp', 'button status', 'azimuth', 'altitude', 'pressure']]
TRAIN_TEST_SPLIT = 0.8

def load_svc2004(file_directory: str = FILE_DIR[DATABASE_NUMBER - 1],
                 column_names: list = None):
    if not os.path.exists(file_directory):
        raise NotADirectoryError('Directory not found')
    if column_names is None:
        column_names = COLUMN_NAMES[DATABASE_NUMBER - 1]

    dataset = []
    dataset_info = []
    for filename in os.listdir(file_directory):
        single_data = pd.read_csv(file_directory + filename, names=column_names, sep=' ').dropna()

        u_idx = filename.find('U')
        s_idx = filename.find('S')
        user_id = int(filename[u_idx+1:s_idx])
        sign_id = int(filename[s_idx+1:].replace('.TXT', ''))

        assert 0 < sign_id <= 40
        if sign_id <= 20:
            is_genuine = 1
        else:
            is_genuine = 0

        dataset_info.append(np.array([user_id, is_genuine]))
        dataset.append(single_data.to_numpy(dtype=int).transpose())

    return dataset, np.array(dataset_info)

def pad_sequences(dataset: list[np.ndarray]) -> np.ndarray:
    assert [seq.shape[0] == len(COLUMN_NAMES[DATABASE_NUMBER - 1]) for seq in dataset]
    max_length = np.max([seq.shape[1] for seq in dataset])
    print(max_length)
    new_dataset = np.empty((len(dataset), dataset[0].shape[0], max_length))
    print(new_dataset.shape)
    for i, seq in enumerate(dataset):
        new_dataset[i, :, :seq.shape[1]] = seq
    print(new_dataset.shape)
    return new_dataset

def normalize(dataset: np.ndarray):
    assert dataset.shape[1] == len(COLUMN_NAMES[DATABASE_NUMBER - 1])
    normalized_dataset = np.empty_like(dataset)
    for j, seq in enumerate(dataset):
        norm_seq = []
        t_idx = COLUMN_NAMES[DATABASE_NUMBER - 1].index('time stamp')
        t = seq[t_idx,:]
        dt = np.diff(t, prepend=t[0]) / (np.ptp(t) + 1e-8)
        for i in range(seq.shape[0]):
            if i == t_idx: continue
            x = seq[i,:]
            x = (x - np.min(x)) / (np.ptp(x) + 1e-8)
            norm_seq.append(x)
        norm_seq.insert(t_idx, t)
        normalized_dataset[j] = np.array(norm_seq)
    print(normalized_dataset.shape)
    return normalized_dataset

def train_test_split(dataset: np.ndarray, data_info: np.ndarray,
                     train_test_ratio: float = TRAIN_TEST_SPLIT):
    train_dataset, test_dataset = None, None
    train_info, test_info = None, None
    for user_id in set(data_info[:,0]):
        user_data = dataset[data_info[:,0] == user_id,:,:]
        user_data_info = data_info[data_info[:,0] == user_id,:]

        user_genuine = user_data[user_data_info[:,1] == 1,:,:]
        user_forgery = user_data[user_data_info[:,1] == 0,:,:]
        user_g_info = user_data_info[user_data_info[:,1] == 1,:]
        user_f_info = user_data_info[user_data_info[:, 1] == 0, :]

        user_g_train = user_genuine[:int(len(user_genuine) * train_test_ratio)]
        user_f_train = user_forgery[:int(len(user_forgery) * train_test_ratio)]
        user_g_test = user_genuine[int(len(user_genuine) * train_test_ratio):]
        user_f_test = user_forgery[int(len(user_forgery) * train_test_ratio):]

        user_g_train_info = user_g_info[:int(len(user_g_info) * train_test_ratio)]
        user_f_train_info = user_f_info[:int(len(user_g_info) * train_test_ratio)]
        user_g_test_info = user_g_info[int(len(user_g_info) * train_test_ratio):]
        user_f_test_info = user_f_info[int(len(user_g_info) * train_test_ratio):]

        user_train = np.concat([user_g_train, user_f_train])
        user_train_info = np.concat([user_g_train_info, user_f_train_info])
        user_test = np.concat([user_g_test, user_f_test])
        user_test_info = np.concat([user_g_test_info, user_f_test_info])

        if train_dataset is None or test_dataset is None:
            train_dataset = user_train
            test_dataset = user_test
            train_info = user_train_info
            test_info = user_test_info
        else:
            train_dataset = np.concatenate((train_dataset, user_train))
            test_dataset = np.concatenate((test_dataset, user_test))
            train_info = np.concatenate((train_info, user_train_info))
            test_info = np.concatenate((test_info, user_test_info))
    return train_dataset, test_dataset, train_info, test_info

def show_data(sample):
    fig, ax = plt.subplots()
    x_coord = [sample[i][0] for i in range(1,len(sample))]
    y_coord = [sample[i][1] for i in range(1,len(sample))]
    if len(x_coord) != len(y_coord):
        raise ValueError('x_coord and y_coord must have same length')
    ax.plot(x_coord, y_coord, marker='o')
    ax.xaxis.set_ticks([np.min(x_coord), np.max(x_coord)])
    ax.yaxis.set_ticks([np.min(y_coord), np.max(y_coord)])
    fig.show()

if __name__ == '__main__':
    samples = load_svc2004()
    show_data(samples[0])
