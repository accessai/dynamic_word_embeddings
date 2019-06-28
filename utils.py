import os

import pickle
import numpy as np
import shutil
import gzip

def save_pickle(obj, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def save_numpy(arr, file_path):
    np.save(file_path,arr)


def extract_gzip(in_file, out_file):
    with gzip.open(in_file, 'rb') as ifile:
        with open(out_file, 'wb') as ofile:
            shutil.copyfileobj(ifile, ofile)

def delete_file(file_path):
    os.remove(file_path)