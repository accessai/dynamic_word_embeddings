import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipGramModel(nn.Module):

    def __init__(self, l=30000, m=40):
        """Skip Gram model

        :param l: vocabulary size
        :param m: embedding dimension
        """

        super(SkipGramModel,self).__init__()
        self.vocab_size = l
        self.embed_dim = m
        self.embed_u = nn.Embedding(num_embeddings=l, embedding_dim=m)
        self.embed_v = nn.Embedding(num_embeddings=l, embedding_dim=m)
        self.fc1 = nn.Linear(in_features=m, out_features=1)

        # initialization

    def forward(self, X):
        """

        :param X: 2D tensor batch * vocab size, positive and negative samples
        :return:
        """

        u_embed = self.embed_u(X)
        v_embed = self.embed_v(X)

        out = torch.mul(u_embed, v_embed)
        out = self.fc1(out)
        # sim_vec = torch.sum(out, dim=(1,))
        # sum_log_target = F.logsigmoid(sim_vec)

        # neg_vec = torch.bmm(self.embed_u(N), self.embed_v(N)).squeeze()
        # neg_vec = torch.sum(neg_vec, dim=(1,))
        # sum_log_neg_sampled = F.logsigmoid(-1 * neg_vec).squeeze()
        #
        # loss = sum_log_target + sum_log_neg_sampled

        # return -1 * loss.sum() / (len(X) + len(N))
        return out


