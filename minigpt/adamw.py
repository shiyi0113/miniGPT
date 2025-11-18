"""
AdamW优化器 ZeRO-2
"""
import torch
import torch.distributed as dist
from torch import Tensor

class DistAdamW(torch.optim.Optimizer):

    def __init__(self, param_groups, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(param_groups, defaults)

    @torch.compile
    @torch.no_grad()
    def step(self):
        # 检查分布式环境是否初始化，如果没有（单卡且未使用 torchrun），则需要伪造环境或报错
        # 为了稳健性，建议始终使用 torchrun --nproc_per_node=1 运行单卡
        if not dist.is_initialized():
             raise RuntimeError("DistAdamW requires distributed environment. Please use 'torchrun'.")

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        
        reduce_scatter_futures: list[torch.Future] = []
        all_reduce_futures: list[torch.Future] = []
        grad_slices = []

        # 第一阶段：Reduce-Scatter
        # 将所有 GPU 上的梯度平均，并切分，每个 Rank 只拿自己负责的那一份
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            for base_i in range(len(params)):
                grad = params[base_i].grad
                # 简单的切分策略：假设参数维度能被 world_size 整除
                # 在实际生产代码中可能需要处理 padding，但这里保持极简
                rank_size = grad.shape[0] // world_size
                grad_slice = torch.empty_like(grad[:rank_size])
                
                # 异步启动 Reduce-Scatter
                reduce_scatter_futures.append(dist.reduce_scatter_tensor(grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True).get_future())
                grad_slices.append(grad_slice)

        # 第二阶段：参数更新与 All-Gather
        idx = 0
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']
            params = group['params']
            
            for base in range(len(params)):
                # 等待当前参数的梯度聚合完成
                reduce_scatter_futures[idx].wait()
                
                p = params[base]
                rank_size = p.shape[0] // world_size
                
                # 获取当前 Rank 负责的参数切片
                p_slice = p[rank * rank_size:(rank + 1) * rank_size]
                
                # 支持逐参数的学习率缩放 (lr_mul)
                lr = group['lr'] * getattr(p, "lr_mul", 1.0)
                
                state = self.state[p]
                g_slice = grad_slices[idx]
                
                # 状态初始化 (Lazy init)
                if not state:
                    state['step'] = torch.tensor(0, dtype=torch.int64, device=p.device)
                    state['exp_avg'] = torch.zeros_like(p_slice)
                    state['exp_avg_sq'] = torch.zeros_like(p_slice)
                
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                state['step'] += 1
                t = state['step']
                
                # 权重衰减 (Weight Decay)
                if wd != 0:
                    eff_weight_decay = lr * wd * getattr(p, "wd_mul", 1.0)
                    p_slice.mul_(1 - eff_weight_decay)
                
                # 更新动量 (Momentum)
                exp_avg.mul_(beta1).add_(g_slice, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g_slice, g_slice, value=1 - beta2)
                
                # 偏差修正 (Bias Correction)
                bias1 = 1 - beta1 ** t
                bias2 = 1 - beta2 ** t
                
                # 计算更新步长
                denom = exp_avg_sq.sqrt().add_(eps)
                step_size = lr * (torch.sqrt(bias2) / bias1)
                
                # 应用更新到切片
                update = exp_avg.div(denom).mul_(step_size)
                p_slice.add_(other=update, alpha=-1.0)
                
                idx += 1
                
                # 异步启动 All-Gather：将更新后的切片广播回完整参数
                all_reduce_futures.append(dist.all_gather_into_tensor(p, p_slice, async_op=True).get_future())
        
        # 等待所有通信完成，确保参数一致性
        torch.futures.collect_all(all_reduce_futures).wait()