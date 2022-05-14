import math
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

class MF(nn.Module):
    def __init__(self, cfg):
        super(MF, self).__init__()
        hidden = cfg.mf_hidden
        self.movie_embedding = nn.Embedding(cfg.mnum, hidden)
        self.user_embedding = nn.Embedding(cfg.unum, hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, cfg.rnum),
        )
    
    def forward(self, data):
        user_id = data[:, 0]
        movie_id = data[:, 1]

        user = self.movie_embedding(user_id)
        movie = self.movie_embedding(movie_id)

        score = self.mlp(torch.cat([user, movie], dim=-1))
        return score