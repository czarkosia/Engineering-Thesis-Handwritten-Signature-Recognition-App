import numpy as np
import matplotlib.pyplot as plt

DATABASE_NUMBER = 2 # 1 lub 2
COLUMN_NAMES = [['x-coord', 'y-coord', 'time stamp', 'button status'],
                ['x-coord', 'y-coord', 'time stamp', 'button status', 'azimuth', 'altitude', 'pressure']]

def show_sample(sample: np.ndarray):
    sample = np.transpose(sample)
    fig, ax = plt.subplots()
    x_coord = [sample[i][0] for i in range(1,len(sample))]
    y_coord = [sample[i][1] for i in range(1,len(sample))]
    if len(x_coord) != len(y_coord):
        raise ValueError('x_coord and y_coord must have same length')
    ax.plot(x_coord, y_coord, marker='o')
    ax.xaxis.set_ticks([np.min(x_coord), np.max(x_coord)])
    ax.yaxis.set_ticks([np.min(y_coord), np.max(y_coord)])
    ax.set_xlabel('x coordinate')
    ax.set_ylabel('y coordinate')
    fig.show()
    fig.savefig('sample_coords.png')

    t_idx = COLUMN_NAMES[DATABASE_NUMBER - 1].index('time stamp')
    t = sample[:,t_idx] - np.min(sample[:,t_idx])
    plotted = np.array([sample[:,i] for i in range(sample.shape[1]) if i != t_idx]).transpose()
    labels = [COLUMN_NAMES[DATABASE_NUMBER - 1][i] for i in range(sample.shape[1]) if i != t_idx]

    fig2 = plt.figure()
    axes = fig2.subplots(len(COLUMN_NAMES[DATABASE_NUMBER - 1]) - 1, 1)
    fig2.subplots_adjust(hspace=1)
    for i in range(len(axes)):
        axes[i].plot(t, plotted[:, i])
        axes[i].set_title(labels[i])
        axes[i].set_xticks(np.arange(np.min(t), np.max(t), 500))
    axes[-1].set_xlabel('time')
    fig2.show()
    fig2.savefig('sample_plots.png')