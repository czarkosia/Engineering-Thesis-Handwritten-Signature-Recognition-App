import numpy as np


def normalize(dataset: np.ndarray, timestamp_col_idx: int, no_features: int):
    normalized_dataset = np.empty_like(dataset)
    for j, seq in enumerate(dataset):
        norm_seq = []
        t_idx = timestamp_col_idx
        t = seq[t_idx,:]
        dt = np.diff(t, prepend=t[0]) / (np.ptp(t) + 1e-8)
        for i in range(seq.shape[0]):
            if i == t_idx: continue
            x = seq[i,:]
            x = (x - np.min(x)) / (np.ptp(x) + 1e-8)
            norm_seq.append(x)
        norm_seq.insert(t_idx, t)
        normalized_dataset[j] = np.array(norm_seq)
    return normalized_dataset