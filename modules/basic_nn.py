import math
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttend(nn.Module):
    def __init__(self, embedding_size: int) -> None:
        super(SelfAttend, self).__init__()

        self.h1 = nn.Sequential(
            nn.Linear(embedding_size, 30),
            nn.Tanh()
        )
        
        self.gate_layer = nn.Linear(30, 1)

    def forward(self, seqs, seq_masks=None):
        """
        :param seqs: shape [batch_size, seq_length, embedding_size]
        :param seq_lens: shape [batch_size, seq_length]
        :return: shape [batch_size, seq_length, embedding_size]
        """
        gates = self.gate_layer(self.h1(seqs)).squeeze(-1)
        if seq_masks is not None:
            gates = gates.masked_fill(seq_masks == 0, -1e9)
        p_attn = F.softmax(gates, dim=-1)
        p_attn = p_attn.unsqueeze(-1)
        h = seqs * p_attn
        output = torch.sum(h, dim=1)
        return output

class BasicRS(nn.Module):
    def __init__(self, cfg):
        super(BasicRS, self).__init__()

        self.sa = SelfAttend(cfg.hidden)
        self.mh_self_attn = nn.MultiheadAttention(
            cfg.hidden, num_heads=cfg.head_num
        )
        self.ln = nn.LayerNorm(cfg.hidden)
        self.movie_embedding = nn.Embedding(cfg.mnum, cfg.mhidden)
        self.rate_embedding = nn.Embedding(cfg.rnum, cfg.rate_hidden)
        self.cfg = cfg
        self.mlp = nn.Sequential(
            nn.Linear(cfg.hidden + cfg.mhidden, cfg.hidden),
            nn.Tanh(),
            nn.Linear(cfg.hidden, cfg.rnum),
        )
    
    def forward(self, data):
        target_id = data[:, 0]
        his_id = data[:, 1: 1 + self.cfg.his]
        rate_id = data[:, 1 + self.cfg.his:]

        target = self.movie_embedding(target_id)
        his = self.movie_embedding(his_id)
        rate = self.rate_embedding(rate_id)

        hiddens = torch.cat([his, rate], dim=-1)
        x = hiddens.permute(1, 0, 2)
        output, _ = self.mh_self_attn(x, x, x)
        output = output.permute(1, 0, 2)
        residual_sum = self.ln(hiddens + output)
        agg = self.sa(residual_sum)
        
        score = self.mlp(torch.cat([agg, target], dim=-1))
        return score