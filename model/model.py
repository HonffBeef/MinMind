from transformers import PretrainedConfig
import torch
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import torch.nn as nn
from typing import Optional, Tuple, List, Union
from torch.nn import functional as F
import math
from method.rope import *
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast


# huggingface的类
class MokioMindConfig(PretrainedConfig):
    model_type = "mokiomind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        
        ############ MoE ############
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = 'softmax',
        aux_loss_alpha: float = 0.1,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention

        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 4,
                "beta_slow": 1,
                "factor": 4,
                "original_max_position_embeddings": 2048,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )


class RMSNorm(nn.Module):
    # __init__初始化
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # FIX: 正确的RMSNorm，不要重复乘x
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x):
        # FIX: 调用 _norm 而不是不存在的 self.norm
        return self.weight * self._norm(x)



def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6,
                         rope_scaling: Optional[dict] = None):
 
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    
    # YARN缩放：用于外推到更长序列
    if rope_scaling is not None:
        # 使用dict.get()方法获取值，提供默认值
        # 元组解包：同时赋值多个变量
        orig_max, factor, beta_fast, beta_slow = (
            rope_scaling.get("original_max_position_embeddings", 2048), 
            rope_scaling.get("factor", 4),
            rope_scaling.get("beta_fast", 4.0), 
            rope_scaling.get("beta_slow", 1.0)
        )
        
        # 只有当序列长度超过原始长度时才应用缩放
        if end / orig_max > 1.0:
            # next()函数：找到第一个满足条件的元素
            # 生成器表达式：(i for i in ... if condition)
            # 找到需要校正的维度边界
            corr_dim = next((i for i in range(dim // 2) if 2 * math.pi / freqs[i] > orig_max), dim // 2)
            
            # 计算每个维度的插值权重
            # max(..., 1): 防止除零
            power = torch.arange(0, dim // 2, device=freqs.device).float() / max(dim // 2 - 1, 1)
            # 线性插值计算beta值
            beta = beta_slow + (beta_fast - beta_slow) * power
            
            # YaRN缩放公式：λ = (β·α - β + 1)/(β·α)
            # torch.where(): 条件选择，相当于 condition ? value1 : value2
            scale = torch.where(
                torch.arange(dim // 2, device=freqs.device) < corr_dim, 
                (beta * factor - beta + 1) / (beta * factor),  # 高频部分使用复杂缩放
                1.0 / factor                                    # 低频部分简单缩放
            )
            freqs = freqs * scale

    t = torch.arange(end, device=freqs.device)

    freqs = torch.outer(t, freqs).float()
    
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    def rotate_half(x):

        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
      
    # RoPE旋转公式：x_rotated = x*cos + rotate_half(x)*sin
    # .unsqueeze(unsqueeze_dim): 在指定维度增加一个维度，用于广播
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed



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

        # FIX: config里字段叫 flash_attention
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention") and args.flash_attention

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],  # (cos, sin) 这里约定已经按 start_pos 切好了
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

        # ---- RoPE (FIX: 避免双重position偏移；这里直接用传进来的cos/sin) ----
        cos, sin = position_embeddings
        past_len = past_key_value[0].size(1) if past_key_value is not None else 0  # 仍保留给mask对齐使用

        # FIX: 半精度时避免被 cos/sin(float32) 把 q/k upcast，导致后续 Linear dtype 不匹配
        cos = cos.to(dtype=xq.dtype)
        sin = sin.to(dtype=xq.dtype)

        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

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
        # FIX: flash + is_causal=True 在 k_len!=q_len（有cache且一次喂多token）时会产生错误mask，这里只在 past_len==0 时启用
        if self.flash and past_len == 0 and q_len > 1 and (attention_mask is None or torch.all(attention_mask == 1)):
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


class FeedForward(nn.Module):
    def __init__(self, config: MokioMindConfig):
        super().__init__()
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        # ACT2FN是transformers里激活函数的映射表，支持'silu','gelu'等
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        gated = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        return self.dropout(self.down_proj(gated))


class MokioMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MokioMindConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self_hidden_size = config.hidden_size
        self.head_dim = self_hidden_size // self.num_attention_heads

        # FIX: 命名一致
        self.self_attn = Attention(config)

        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # FIX: 必须加 self.
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.mlp = FeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None,
                use_cache=False, attention_mask=None):
        residual = hidden_states

        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states),  # pre-norm
            position_embeddings,
            past_key_value,
            use_cache,
            attention_mask
        )

        hidden_states = hidden_states + residual
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value


class MokioMindModel(nn.Module):
    def __init__(self, config: MokioMindConfig):
        super().__init__()
        self.config = config

        # FIX: 统一字段名
        self.vocab_size, self.num_hidden_layers = (
            config.vocab_size,
            config.num_hidden_layers,
        )

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

        # FIX: 统一为 self.layers
        self.layers = nn.ModuleList(
            [MokioMindBlock(i, config) for i in range(self.num_hidden_layers)]
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        #RoPE预计算
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.hidden_size // config.num_attention_heads,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        **kwargs
    ):
        batch_size, seq_len = input_ids.shape

        # 兼容性检查：某些框架会传入包含.layers属性的对象，视为不携带past信息
        if hasattr(past_key_values, 'layers'):
            past_key_values = None

        # FIX: 统一用 self.layers
        past_key_values = past_key_values or [None] * len(self.layers)

        # FIX: past_key_values 是 list，没有 .shape；start_pos 应从第一层K的长度取
        start_pos = 0
        if past_key_values is not None and len(past_key_values) > 0 and past_key_values[0] is not None:
            start_pos = past_key_values[0][0].size(1)

        hidden_states = self.dropout(self.embed_tokens(input_ids))

        # 从注册的buffer中取出对应位置范围的cos/sin作为position_embeddings
        # self.freqs_cos/freqs_sin的shape为 [max_pos, head_dim]
        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_len],
            self.freqs_sin[start_pos:start_pos + seq_len]
        )

        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask,
            )
            presents.append(present)

        hidden_states = self.norm(hidden_states)
        return hidden_states, presents


class MokioMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class=MokioMindConfig

    def __init__(self, config: MokioMindConfig):
        self.config=config

        super().__init__(config)

        self.model = MokioMindModel(config)

        self.lm_head = nn.Linear(
            self.config.hidden_size,
            self.config.vocab_size,
            bias=False,
        )

        #权重共享
        self.lm_head.weight = self.model.embed_tokens.weight



    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                **args):
        # FIX: MokioMindModel 只返回 (hidden_states, presents)
        h, past_kvs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args
        )
         
        # logits_to_keep用于在序列末尾保留一部分logits（用于截断或微调策略）
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep

        # 通过lm_head将hidden states投影到词表logits
        # h: [bsz, seq_len, hidden]
        logits = self.lm_head(h[:, slice_indices, :])

        # 使用PretrainedModel约定的输出容器CausalLMOutputWithPast
        # 通过__setitem__方式填充键值，保持与Hugging Face接口兼容
        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=past_kvs,   # FIX: 返回本轮更新后的cache
            hidden_states=h
        )
