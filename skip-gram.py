import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipGram(nn.Module):

    def __init__(self, vocab_size, embed_dim):
        super(SkipGram,self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embed_u = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
        self.embed_v = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)

        # insitalisation

