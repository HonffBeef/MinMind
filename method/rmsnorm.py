import torch
import torch.nn as nn

#继承nn.Module类
class RMSNorm(nn.Module):
    # __init__初始化
    def __init__(self, dim:int, eps:float=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return torch.rsqrt(x.pow(2).mean(-1,keepdim=True) + self.eps) * x
    
    def forward(self, x):
        return self.weight * self.norm(x)

