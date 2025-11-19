import torch
import pyarrow.parquet as pq
from collections import deque

from minigpt.common import get_dist_info
from minigpt.dataset import list_parquet_files
from minigpt.tokenizer import get_tokenizer

def tokenizing_distributed_data_loader(B, T, split, tokenizer_threads=4, tokenizer_batch_size=128, device="cuda", resume_state_dict=None):
    """
    数据加载器生成器。
    流式读取 Parquet 文件 -> Tokenize -> 组装成 (B, T) 的 Tensor -> 搬运到 GPU。
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"

    # 获取分布式环境信息 (DDP)  
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()

    # --- 内部函数：文档流生成器 ---
    def document_batches():
        # 获取文件列表
        parquet_paths = list_parquet_files()
        # 简单的训练/验证集划分
        if len(parquet_paths) > 0:
            parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
        else:
            # 如果没有文件（例如刚开始还没下载），这就没法工作了，但在测试时我们假设已有文件
            print(f"Warning: No parquet files found in base_data for split {split}")
            return

        # 恢复状态逻辑
        resume_pq_idx = resume_state_dict["pq_idx"] if resume_state_dict is not None else 0
        resume_rg_idx = resume_state_dict["rg_idx"] if resume_state_dict is not None else None
        
        pq_idx = resume_pq_idx 
        
        while True: # 无限循环 (Epochs)
            if pq_idx >= len(parquet_paths):
                pq_idx = 0 # 读完一轮，从头开始

            filepath = parquet_paths[pq_idx]
            try:
                pf = pq.ParquetFile(filepath)
            except Exception:
                # 文件损坏或读取失败，跳过
                pq_idx += 1
                continue
                
            # 确定 Row Group 的起始位置
            # 如果是恢复训练，从记录的 rg_idx 开始；否则根据 DDP Rank 决定
            if resume_rg_idx is not None:
                base_idx = resume_rg_idx // ddp_world_size
                base_idx += 1 
                rg_idx = base_idx * ddp_world_size + ddp_rank
                resume_rg_idx = None 
            else:
                rg_idx = ddp_rank

            # 遍历当前文件中的 Row Groups
            while rg_idx < pf.num_row_groups:
                rg = pf.read_row_group(rg_idx)
                batch = rg.column('text').to_pylist() # 获取文本列表
                
                # 分批 yield 出去，避免一次性 tokenize 太多
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i+tokenizer_batch_size], (pq_idx, rg_idx)
                
                # DDP 步进：如果有 8 张卡，当前卡读第 0 个，下一轮读第 8 个...
                rg_idx += ddp_world_size
            
            pq_idx += 1 # 下一个文件

    # --- 主逻辑：Token 流水线 ---
    
    # 初始化文档生成器
    batches = document_batches()
    
    # 计算每个 Batch 需要多少个 Token
    # B*T 是输入，+1 是因为我们需要错一位作为 Target (Label)
    needed_tokens = B * T + 1 
    
    tokenizer = get_tokenizer()
    bos_token = tokenizer.get_bos_token_id()
    
    token_buffer = deque() # 双端队列作为缓冲区
    
    while True:
        # 1. 填充缓冲区：直到有足够的 Token 凑够一个 Batch   因为一个行组中的Token不等于一次batch所需的 B*T+1 个Token
        while len(token_buffer) < needed_tokens:
            try:
                doc_batch, (pq_idx, rg_idx) = next(batches)
            except StopIteration:
                # 理论上 document_batches 是无限循环，不应停止，除非没文件
                break
                
            # 并行 Tokenize
            token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
            for tokens in token_lists:
                token_buffer.extend(tokens)
        
        if len(token_buffer) < needed_tokens:
            break # 数据耗尽

        # 2. 取出一个 Batch 的 Token
        tokens = [token_buffer.popleft() for _ in range(needed_tokens)]
        
        # 3. 转换为 Tensor 并搬运到 GPU
        # pin_memory=True 可以加速 CPU 到 GPU 的传输
        use_cuda_optimizations = device == "cuda"
        scratch = torch.tensor(tokens, dtype=torch.long, pin_memory=use_cuda_optimizations)
        
        # x: 输入 [0, 1, ... N-1]
        # y: 目标 [1, 2, ... N] (下一个词预测)
        inputs_cpu = scratch[:-1]
        targets_cpu = scratch[1:]
        
        inputs = inputs_cpu.view(B, T).to(device=device, non_blocking=use_cuda_optimizations)
        targets = targets_cpu.view(B, T).to(device=device, non_blocking=use_cuda_optimizations)
        
        # 记录当前状态，方便中断恢复
        state_dict = {"pq_idx": pq_idx, "rg_idx": rg_idx}
        
        yield inputs, targets, state_dict