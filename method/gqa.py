import torch
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import torch.nn as nn
from typing import Optional, Tuple
from torch.nn import functional as F
import math
from model.model import MokioMindConfig
from rope import *


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    重复key-value张量以匹配query头数 (用于分组查询注意力GQA)
    x: [bs, seq_len, num_kv_heads, head_dim] -> [bs, seq_len, num_kv_heads*n_rep, head_dim]
    """
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, num_key_value_heads, n_rep, head_dim)
        .reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, args: MokioMindConfig):
        super().__init__()

        self.num_key_value_heads = (
            args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads
        )
        assert args.num_attention_heads % self.num_key_value_heads == 0
        assert args.hidden_size % args.num_attention_heads == 0

        self.n_local_heads = args.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.hidden_size // args.num_attention_heads

        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)

        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout

        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention") and args.flash_attn

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],  # (cos, sin)
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        attention_mask: Optional[torch.Tensor] = None,  # 约定: 1=keep, 0=pad
    ):
        bsz, seq_len, _ = x.shape

        # ---- QKV ----
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        # ---- RoPE (FIX: 处理 KV cache 时的 position 偏移) ----
        cos, sin = position_embeddings
        past_len = past_key_value[0].size(1) if past_key_value is not None else 0
        cos_cur = cos[past_len : past_len + seq_len]
        sin_cur = sin[past_len : past_len + seq_len]
        xq, xk = apply_rotary_pos_emb(xq, xk, cos_cur, sin_cur)

        # ---- KV cache concat ----
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)

        past_kv = (xk, xv) if use_cache else None

        # ---- GQA repeat + transpose ----
        xq = xq.transpose(1, 2)  # [bs, heads, q_len, head_dim]
        xk = repeat_kv(xk, self.n_rep).transpose(1, 2)  # [bs, heads, k_len, head_dim]
        xv = repeat_kv(xv, self.n_rep).transpose(1, 2)  # [bs, heads, k_len, head_dim]

        q_len = xq.size(-2)
        k_len = xk.size(-2)

        # ---- 对齐 attention_mask 到 k_len（可选但更稳） ----
        if attention_mask is not None and attention_mask.size(-1) != k_len:
            # 常见情况：只传了当前 step 的 mask（seq_len），这里补上 past 的全 1
            if attention_mask.size(-1) == seq_len and past_len > 0:
                pad = torch.ones((bsz, past_len), device=attention_mask.device, dtype=attention_mask.dtype)
                attention_mask = torch.cat([pad, attention_mask], dim=-1)
            # 仍不匹配就直接报错，避免静默算错
            if attention_mask.size(-1) != k_len:
                raise ValueError(f"attention_mask last dim ({attention_mask.size(-1)}) != k_len ({k_len})")

        # ---- Attention ----
        if self.flash and q_len > 1 and (attention_mask is None or torch.all(attention_mask == 1)):
            # (FIX: bool mask 语义：True 表示被 mask 掉)
            attn_mask = None
            if attention_mask is not None:
                key_pad = (attention_mask == 0)  # True=pad -> mask out
                attn_mask = key_pad.view(bsz, 1, 1, k_len).expand(bsz, self.n_local_heads, q_len, k_len)

            output = F.scaled_dot_product_attention(
                xq,
                xk,
                xv,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [bs, heads, q_len, k_len]

            # (FIX: causal mask 必须是 [q_len, k_len]，支持 KV cache)
            causal_mask = torch.triu(
                torch.full((q_len, k_len), float("-inf"), device=scores.device),
                diagonal=1 + (k_len - q_len),
            )
            scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)

            # (FIX: padding mask 要匹配 k_len)
            if attention_mask is not None:
                extended_attention_mask = attention_mask.view(bsz, 1, 1, k_len)  # 1=keep, 0=pad
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask

            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ xv  # [bs, heads, q_len, head_dim]

        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv
