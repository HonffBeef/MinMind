import torch
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import torch.nn as nn
from typing import Optional, Tuple
from torch.nn import functional as F
import math
from model.model import MokioMindConfig
from rope import *
from transformers.activations import ACT2FN

class FeedForward(nn.Module):
    #初始化
    #升维
    #降维
    #门控
    #dropout
    #激活函数
    def __init__(self, args:MokioMindConfig):
        super().__init__()
        if args.intermediate_size is None:
            intermediate_size = int(args.hidden_size * 8 / 3)
            args.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)

        self.up_project = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_project = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)
        self.gate_project = nn.Linear(args.hidden_size, args.intermediate_size, bias=False) 
        self.dropout = nn.Dropout(args.dropout)
        self.act_fn = ACT2FN[args.hidden_act]

    def forward(self, x):
        gated = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        return self.dropout(self.down_proj(gated))