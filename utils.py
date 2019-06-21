import pickle
import numpy as np


def save_pickle(obj, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def save_numpy(arr, file_path):
    np.save(file_path,arr)