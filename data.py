import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


FILE_DIR = 'Task1/'
COLUMN_NAMES = ['x coord', 'y coord', 'time stamp', 'button status']

def load_data(file_directory: str = FILE_DIR) -> list[np.ndarray]:
    if not os.path.exists(FILE_DIR):
        raise NotADirectoryError('Directory not found')

    dataset = []
    for filename in os.listdir(file_directory):
        single_data = pd.read_csv(FILE_DIR + filename,names=COLUMN_NAMES, sep=' ').dropna()
        dataset.append(single_data.to_numpy(dtype=int))

    return dataset

def show_data(sample: np.ndarray):
    fig, ax = plt.subplots()
    x_coord = [sample[i][0] for i in range(1,len(sample))]
    y_coord = [sample[i][1] for i in range(1,len(sample))]
    if len(x_coord) != len(y_coord):
        raise ValueError('x_coord and y_coord must have same length')
    ax.plot(x_coord, y_coord, marker='o')
    ax.xaxis.set_ticks([np.min(x_coord), np.max(x_coord)])
    ax.yaxis.set_ticks([np.min(y_coord), np.max(y_coord)])
    fig.show()

def pad_sequences(dataset: list[np.ndarray]) -> np.ndarray:
    max_length = np.max([seq.shape[1] for seq in dataset])
    print(max_length)
    new_dataset = np.empty((len(dataset), dataset[0].shape[0], max_length))
    print(new_dataset.shape)
    for i, seq in enumerate(dataset):
        new_dataset[i, :, :seq.shape[1]] = seq
    print(new_dataset.shape)
    return new_dataset

if __name__ == '__main__':
    samples = load_data()
    show_data(samples[0])
