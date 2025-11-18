"""
GPT 模型核心定义
包含现代架构特性：
- Rotary Embeddings (RoPE)
- QK Norm (LayerNorm on Queries/Keys)
- 无 Bias 的 Linear 层
- SwiGLU 变体 (这里用的是 ReLU^2)
- RMSNorm
- Group-Query Attention (GQA)
"""

import math
from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from minigpt.common import get_dist_info, print0
from minigpt.muon import Muon, DistMuon
from minigpt.adamw import DistAdamW

@dataclass
class GPTConfig:
    sequence_len: int = 1024  # 上下文窗口大小
    vocab_size: int = 50304   # 词表大小 (50257 + padding)
    n_layer: int = 12         # 层数
    n_head: int = 6           # Query 头数
    n_kv_head: int = 6        # Key/Value 头数 (如果 < n_head 则开启 GQA)
    n_embd: int = 768         # 嵌入维度

def norm(x):
    # 纯函数式 RMSNorm，没有可学习参数 (alpha/gamma)，简单且稳定
    return F.rms_norm(x, (x.size(-1),))

def apply_rotary_emb(x, cos, sin):
    # 应用旋转位置编码 (RoPE)
    assert x.ndim == 4  # (B, H, T, D)
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    out = torch.cat([y1, y2], 3)
    return out.to(x.dtype)

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        
        # 确保维度匹配
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head
        assert self.n_head % self.n_kv_head == 0
        
        # 线性投影层 (无 Bias，节省显存)
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, cos_sin, kv_cache):
        B, T, C = x.size()

        # 1. 生成 Q, K, V
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # 2. 应用 RoPE 和 QK Norm
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k) # QK Norm：这是大模型训练稳定的关键

        # 转置为 (B, H, T, D) 以便进行 Attention 计算
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 3. 处理 KV Cache (推理时使用)
        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)
        Tq = q.size(2) # 当前查询的长度
        Tk = k.size(2) # 总 Key 的长度 (Cache + Current)

        # 4. 计算 Attention
        # GQA 支持：如果 n_kv_head != n_head，PyTorch 的 sdp_attention 会自动处理广播
        enable_gqa = self.n_head != self.n_kv_head
        
        if kv_cache is None or Tq == Tk:
            # 训练模式 或 推理时的预填充阶段 (Prefill)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)
        elif Tq == 1:
            # 推理模式：生成下一个 token
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)
        else:
            # 混合情况：处理分块推理
            attn_mask = torch.zeros((Tq, Tk), dtype=torch.bool, device=q.device)
            prefix_len = Tk - Tq
            if prefix_len > 0:
                attn_mask[:, :prefix_len] = True
            attn_mask[:, prefix_len:] = torch.tril(torch.ones((Tq, Tq), dtype=torch.bool, device=q.device))
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=enable_gqa)

        # 5. 输出投影
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 4倍宽度的前馈网络
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        # ReLU^2 激活函数：也就是 (ReLU(x))^2
        # 这种激活函数比普通的 ReLU 或 GELU 更有利于产生稀疏特征，且计算廉价
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin, kv_cache):
        # Pre-Norm 结构：先 Norm 再进层
        x = x + self.attn(norm(x), cos_sin, kv_cache)
        x = x + self.mlp(norm(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 模型主体
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # 预计算 RoPE 的旋转矩阵 (Cos/Sin)
        # 这里预分配了 10 倍序列长度的 buffer，防止推理时溢出
        self.rotary_seq_len = config.sequence_len * 10
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def init_weights(self):
        self.apply(self._init_weights)
        # 零初始化特定的投影层，这有助于深层网络训练的初始稳定性
        torch.nn.init.zeros_(self.lm_head.weight)
        for block in self.transformer.h:
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
        
        # 重新初始化 rotary (因为它们是 buffer)
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        
        # 将 Embedding 转为 BF16 以节省显存 (Embedding 更新非常稀疏，BF16 足够)
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # 根据 GPT-NeoX / LLaMA 的经验初始化标准差
            fan_out = module.weight.size(0)
            fan_in = module.weight.size(1)
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        # 调整形状以便广播: (1, T, 1, D/2)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        # 估算每 token 的 FLOPs，用于统计训练效率
        nparams = sum(p.numel() for p in self.parameters())
        nparams_embedding = self.transformer.wte.weight.numel()
        l, h, q, t = self.config.n_layer, self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        # 粗略计算公式：6N + 12LHQT
        return 6 * (nparams - nparams_embedding) + 12 * l * h * q * t

    def setup_optimizers(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0):
        """
        配置优化器：这是 Muon + AdamW 混合优化的关键逻辑
        """
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()
        
        matrix_params = list(self.transformer.h.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        
        # 1. AdamW 组：用于 Embedding 和 LM Head
        # 学习率根据模型维度进行缩放 (1/sqrt(d_model))
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        if rank == 0:
            print(f"Scaling LR for AdamW params by {dmodel_lr_scale:.6f}")
            
        adam_groups = [
            dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
            dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
        ]
        
        AdamWFactory = DistAdamW if ddp else partial(torch.optim.AdamW, fused=True)
        adamw_optimizer = AdamWFactory(adam_groups, betas=(0.8, 0.95), eps=1e-10, weight_decay=weight_decay)
        
        # 2. Muon 组：用于所有 Transformer 内部的矩阵 (Attention + MLP)
        MuonFactory = DistMuon if ddp else Muon
        muon_optimizer = MuonFactory(matrix_params, lr=matrix_lr, momentum=0.95)
        
        optimizers = [adamw_optimizer, muon_optimizer]
        return optimizers

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        B, T = idx.size()

        # 获取当前长度的 RoPE 编码
        assert T <= self.cos.size(1), f"Sequence length {T} exceeds cache {self.cos.size(1)}"
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]

        # 前向传播 Transformer 主干
        x = self.transformer.wte(idx)
        x = norm(x)
        for block in self.transformer.h:
            x = block(x, cos_sin, kv_cache)
        x = norm(x)

        # 计算 Logits 或 Loss
        softcap = 15.0 # Logits Soft-capping，防止 logits 过大导致数值不稳定
        if targets is not None:  # 训练模式
            logits = self.lm_head(x)
            logits = softcap * torch.tanh(logits / softcap)
            logits = logits.float() # Loss 计算必须用 FP32
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            return loss
        else:                    # 推理模式
            logits = self.lm_head(x)
            logits = softcap * torch.tanh(logits / softcap)
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        简单的自回归生成函数 (用于测试)
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
            
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        
        # 简单的逐 token 生成循环
        for _ in range(max_tokens):
            logits = self.forward(ids)
            logits = logits[:, -1, :]
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
                
            ids = torch.cat((ids, next_ids), dim=1)
            yield next_ids.item()