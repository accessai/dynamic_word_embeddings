import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipGram(nn.Module):

    def __init__(self, l=30000, m=40):
        """Skip Gram model

        :param l: vocabulary size
        :param m: embedding dimension
        """

        super(SkipGram,self).__init__()
        self.vocab_size = l
        self.embed_dim = m
        self.embed_u = nn.Embedding(num_embeddings=l, embedding_dim=m)
        self.embed_v = nn.Embedding(num_embeddings=l, embedding_dim=m)
        self.output = nn.Linear(in_features=m, out_features=l)

        # initialization

    def forward(self, X, N, neg_samples, batch_size):
        u_embed = self.embed_u(X)
        v_embed = self.embed_v(X)

        sim_vec = torch.mul(u_embed, v_embed)
        sim_vec= torch.sum(sim_vec, dim=(1,))
        sum_log_target = F.logsigmoid(sim_vec).squeeze()

        neg_vec = torch.bmm(self.embed_u(neg_samples), self.embed_v(neg_samples)).squeeze()
        neg_vec = torch.sum(neg_vec, dim=(1,))
        sum_log_neg_sampled = F.logsigmoid(-1 * neg_vec).squeeze()

        loss = sum_log_target + sum_log_neg_sampled

        return -1 * loss.sum() / batch_size


