import torch
from typing import Optional, Tuple


def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6,
                         rope_scaling: Optional[dict] = None):
    """
    预计算RoPE(Rotary Position Embedding)的频率
    RoPE通过旋转矩阵为query和key注入位置信息，替代传统的位置编码
    
    Args:
        dim: attention head的维度
        end: 最大序列长度
        rope_base: RoPE的基础频率，默认1e6
        rope_scaling: 可选的频率缩放配置(YARN方法)
    
    Returns:
        freqs_cos, freqs_sin: 预计算的cos和sin值，shape为[seq_len, dim]
    """
    # 计算频率：freqs[i] = 1 / (base^(2i/dim)) for i in [0, 2, 4, ..., dim-2]
    # torch.arange(0, dim, 2): 生成[0, 2, 4, ..., dim-2]
    # [: (dim // 2)]: 切片，确保长度为dim//2
    # .float() / dim: 归一化到[0,1)
    # rope_base ** (...): 计算base的幂次
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

    # 生成位置索引：[0, 1, 2, ..., end-1]
    t = torch.arange(end, device=freqs.device)
    # torch.outer(): 计算外积，生成 [seq_len, dim//2] 的频率矩阵
    # freqs[i,j] = t[i] * freqs[j]
    freqs = torch.outer(t, freqs).float()
    
    # 计算cos和sin值，并复制一遍以匹配完整的head维度
    # torch.cat(..., dim=-1): 在最后一维拼接
    # [torch.cos(freqs), torch.cos(freqs)]: 复制cos值
    """
    问题：RoPE的最后

		freqs_cos = torch.cat(【torch.cos(freqs), torch.cos(freqs)】, dim=-1)
		freqs_sin = torch.cat(【torch.sin(freqs), torch.sin(freqs)】, dim=-1)
		
		和我讲解的成对旋转有矛盾，这里解释一下
		
		minimind是这样配对的，旋转之后为【-q3, -q4, -q5, q0, q1, q2】，
		让q0和-q3配对进行旋转，
		我视频里理论讲解的是相邻配对，
		实际代码是头尾配对，所以会有冲突，
		但是两种旋转位置对于一个从头开始训练的模型来说都是可以的
		
		这里建议大家采用
		freqs_cos = torch.cos(freqs).repeat_interleave(2,dim=-1) # 【seq_len,dim】
		freqs_sin = torch.sin(freqs).repeat_interleave(2,dim=-1) # 【seq_len,dim】
		的标准llama做法
		"""
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """
    应用旋转位置编码到query和key
    RoPE通过复数旋转为每个位置的query和key添加位置信息
    
    Args:
        q, k: query和key张量 [batch, seq_len, num_heads, head_dim]
        cos, sin: 预计算的cos和sin值
        position_ids: 位置ID（未使用）
        unsqueeze_dim: 在哪个维度增加维度
    """
    """  
     若使用的是上面推荐的相邻旋转，这里的代码要改成
		 使用我原来的写法，一个频率序列是[c0,c1,c2,c0,c1,c2],[s0,s1,s2,s0,s1,s2]这样
		 然后用我原来的rotate_half，应用的q是[-q3,-q4,-q5, q0,q1,q2]
		 那么相同频率配对的其实是q3配c0，q0配c0 
		 而用改之后的标准写法，频率序列是[c0,c0,c1,c1,c2,c2],[s0,s0,s1,s1,s2,s2]，
		 然后用修改之后的rotate_half，q序列会变为[-q0,q1,-q2,q3,-q4,q5]，
		 这样的话就是q0和q1配c0
		 正负号不会出现问题
	   def rotate_half(x: torch.Tensor):
        return torch.cat([-x[..., 1::2], x[..., ::2]], dim=-1)
      """
    def rotate_half(x):
        """
        将向量的前半部分和后半部分交换并取负号
        这是复数旋转在实数域的实现：[a,b] -> [-b,a]
        """
        # x.shape[-1] // 2: 获取最后一个维度的中点
        # x[..., x.shape[-1] // 2:]: 取后半部分
        # x[..., : x.shape[-1] // 2]: 取前半部分  
        # -x[..., x.shape[-1] // 2:]: 后半部分取负号
        # torch.cat(..., dim=-1): 在最后一维拼接
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
      
    # RoPE旋转公式：x_rotated = x*cos + rotate_half(x)*sin
    # .unsqueeze(unsqueeze_dim): 在指定维度增加一个维度，用于广播
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed