import numpy as np
import torch.utils.data
import torch

import utils


class SkipGramDataSet(torch.utils.data.Dataset):
    def __init__(self, data_path, vocab_path):
        self.file_path = data_path
        self.data = np.load(data_path)
        self.length = len(self.data)
        self.vocab_length = len(utils.load_pickle(vocab_path))+1

    def __getitem__(self, index):
        return torch.from_numpy(self.data[index][:-1]).type(torch.LongTensor), torch.from_numpy(self.data[index][-1:]).type(torch.LongTensor)

    def __len__(self):
        return self.length