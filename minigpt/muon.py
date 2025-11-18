"""
Muon 优化器
特别适合在有限显存下加速 Transformer 训练。
"""
import torch
from torch import Tensor
import torch.distributed as dist

@torch.compile
def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    """
    使用 Newton-Schulz 迭代计算 G 的零次幂（正交化）。
    采用 5 次迭代的变体，系数经过特殊优化以最大化收敛速度。
    注意：这对 BFloat16 非常友好，完美适配 RTX 30/40/50 系列。
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # 归一化谱范数，确保数值稳定性
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    
    # 执行 NS 迭代
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz
    Muon 内部运行标准 SGD-Momentum，然后对更新量进行正交化后处理。
    
    警告：
    - 不要用于 Embedding 或 LayerNorm (1D 参数)，只用于 2D 权重矩阵。
    - 那些 1D 参数应该交给 AdamW 处理。
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        params = list(params)
        # 再次检查：Muon 只能处理 >=2D 的参数
        assert all(p.ndim >= 2 for p in params)
        
        # 按参数形状分组，以支持未来可能的批处理优化
        param_groups = []
        for size in {p.numel() for p in params}:
            group = dict(params=[p for p in params if p.numel() == size])
            param_groups.append(group)
            
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            for p in params:
                g = p.grad
                assert g is not None
                state = self.state[p]
                
                # 初始化动量 buffer
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf: Tensor = state["momentum_buffer"]
                
                # 动量更新 (Momentum)
                buf.lerp_(g, 1 - group["momentum"])
                g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                
                # Newton-Schulz 正交化 (Orthogonalization)
                g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                
                # 根据矩阵形状缩放学习率，并更新权重
                scale = max(1, p.size(-2) / p.size(-1))**0.5
                p.add_(g, alpha=-group["lr"] * scale)

class DistMuon(torch.optim.Optimizer):
    """
    分布式版本的 Muon。
    逻辑与 DistAdamW 类似：
    1. Reduce-Scatter: 梯度平均并切分。
    2. 局部 Muon 更新。
    3. All-Gather: 同步权重。
    """
    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int = 5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        params = list(params)
        assert all(p.ndim == 2 for p in params), "Muon expects 2D parameters only"
        
        rank = dist.get_rank()
        
        # 按形状分组参数
        shapes = sorted({p.shape for p in params})
        param_groups = []
        for shape in shapes:
            group_params = [p for p in params if p.shape == shape]
            # 简单检查同一组内的设备和类型是否一致
            device, dtype = group_params[0].device, group_params[0].dtype
            assert all(p.device == device for p in group_params)
            assert all(p.dtype == dtype for p in group_params)
            
            # 这里的 zero_buffer 用于填充分布式通信时的空缺
            param_groups.append(dict(params=group_params, zero_buffer=torch.zeros_like(group_params[0])))
            
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        assert all(p.grad is not None for group in self.param_groups for p in group["params"])

        all_reduce_futures = []
        for group in self.param_groups:
            params = group["params"]
            zero_buffer = group["zero_buffer"]
            
            # 每次处理 world_size 个参数，确保每个 Rank 分到一个
            for base_i in range(0, len(params), world_size):
                owner_idx = base_i + rank
                
                # 准备 Reduce-Scatter 的输入列表
                rs_input = [p.grad for p in params[base_i:base_i + world_size]]
                rs_input.extend([zero_buffer] * (world_size - len(rs_input)))
                
                # 准备输出 buffer
                rs_output = params[owner_idx].grad if owner_idx < len(params) else torch.empty_like(zero_buffer)
                
                # 启动 Reduce-Scatter
                work = dist.reduce_scatter(rs_output, rs_input, op=dist.ReduceOp.AVG, async_op=True).get_future()
                all_reduce_futures.append(work)

        future_idx = 0
        all_gather_futures = []
        
        for group in self.param_groups:
            params = group["params"]
            zero_buffer = group["zero_buffer"]
            
            for base_i in range(0, len(params), world_size):
                owner_idx = base_i + rank
                
                # 等待梯度聚合完成
                all_reduce_futures[future_idx].wait()
                future_idx += 1
                
                # 只有当前 Rank 负责的那个参数才执行 Muon 更新
                if owner_idx < len(params):
                    p = params[owner_idx]
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf: Tensor = state["momentum_buffer"]
                    
                    buf.lerp_(g, 1.0 - group["momentum"])
                    g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                    g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                    
                    scale = (max(1.0, p.size(-2) / p.size(-1)) ** 0.5)
                    p.add_(g, alpha=-group["lr"] * scale)
                
                # 准备 All-Gather 的输入和输出
                ag_input = params[owner_idx] if owner_idx < len(params) else zero_buffer
                ag_output = params[base_i:base_i + world_size]
                ag_output.extend([torch.empty_like(zero_buffer) for _ in range(world_size - len(ag_output))])
                
                # 启动 All-Gather 同步权重
                work = dist.all_gather(ag_output, ag_input, async_op=True).get_future()
                all_gather_futures.append(work)

        # 等待所有权重同步完成
        torch.futures.collect_all(all_gather_futures).wait()