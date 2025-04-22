import numpy as np
import matplotlib.pyplot as plt
import os
# from tensorflow.python import keras

FILE_DIR = 'Task1/U1S1.TXT'

def read_file():
    if not os.path.exists(FILE_DIR):
        raise FileNotFoundError('File not found')
    data = []
    with open(FILE_DIR, 'r') as file:
        for line in file.readlines():
            row = []
            line = line.replace('\n', '').split(' ')
            for element in line:
                row.append(int(element))
            data.append(row)
    return data

def show_data():
    data = read_file()
    print(type(data[0][0]))
    fig, ax = plt.subplots()
    x_coord = [data[i][0] for i in range(1,len(data))]
    y_coord = [data[i][1] for i in range(1,len(data))]
    if len(x_coord) != len(y_coord):
        raise ValueError('x_coord and y_coord must have same length')
    ax.plot(x_coord, y_coord, marker='o')
    ax.xaxis.set_ticks([min(x_coord), max(x_coord)])
    ax.yaxis.set_ticks([min(y_coord), max(y_coord)])
    fig.show()

if __name__ == '__main__':
    show_data()
    plt.show()
