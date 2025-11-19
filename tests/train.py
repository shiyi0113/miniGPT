"""
MiniGPT 训练脚本 (单卡/多卡通用)
"""
import os
import time
import math
import argparse
from dataclasses import asdict

import torch
import torch.nn.functional as F

from minigpt.common import compute_init, compute_cleanup, print0
from minigpt.gpt import GPT, GPTConfig
from minigpt.dataloader import tokenizing_distributed_data_loader
from minigpt.report import get_report

# -----------------------------------------------------------------------------
# 学习率调度器 (Cosine Decay with Warmup)
def get_lr(it, total_iters, warmup_iters, max_lr, min_lr):
    # 1. 预热阶段 (Linear Warmup)
    if it < warmup_iters:
        return max_lr * (it + 1) / warmup_iters
    # 2. 训练结束 (Min LR)
    if it > total_iters:
        return min_lr
    # 3. 余弦衰减 (Cosine Decay)
    decay_ratio = (it - warmup_iters) / (total_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

# -----------------------------------------------------------------------------
# 主训练逻辑
def main():
    parser = argparse.ArgumentParser()
    # 模型配置
    parser.add_argument("--n_layer", type=int, default=12, help="层数")
    parser.add_argument("--n_head", type=int, default=6, help="头数")
    parser.add_argument("--n_embd", type=int, default=768, help="嵌入维度")
    parser.add_argument("--sequence_len", type=int, default=1024, help="上下文长度")
    # 训练配置
    parser.add_argument("--batch_size", type=int, default=8, help="Micro Batch Size (每张卡的批次大小)")
    parser.add_argument("--total_steps", type=int, default=1000, help="总训练步数")
    parser.add_argument("--warmup_steps", type=int, default=100, help="预热步数")
    parser.add_argument("--learning_rate", type=float, default=6e-4, help="最大学习率")
    parser.add_argument("--output_dir", type=str, default="output", help="模型保存路径")
    # 运行配置
    parser.add_argument("--val_every", type=int, default=200, help="每隔多少步验证一次")
    parser.add_argument("--save_every", type=int, default=500, help="每隔多少步保存一次")
    args = parser.parse_args()

    # 1. 环境初始化
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 2. 初始化模型
    print0(f"Initializing GPT model...")
    config = GPTConfig(
        n_layer=args.n_layer, 
        n_head=args.n_head, 
        n_embd=args.n_embd, 
        sequence_len=args.sequence_len,
        vocab_size=50304 # 与 Tokenizer 训练时一致
    )
    model = GPT(config)
    model.init_weights()
    model.to(device)
    
    # DDP 包装
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[ddp_local_rank])
        raw_model = model.module
    else:
        raw_model = model

    # 3. 优化器
    optimizers = raw_model.setup_optimizers(
        weight_decay=0.1, 
        unembedding_lr=10 * args.learning_rate, # 最后一层通常用更大的学习率
        embedding_lr=10 * args.learning_rate, 
        matrix_lr=args.learning_rate
    )

    # 4. 数据加载器
    # 注意：tokenizer_batch_size 是 CPU 端并行处理的大小
    train_loader = tokenizing_distributed_data_loader(
        args.batch_size, args.sequence_len, split="train", device="cuda" if torch.cuda.is_available() else "cpu"
    )
    val_loader = tokenizing_distributed_data_loader(
        args.batch_size, args.sequence_len, split="val", device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # 5. 训练循环
    print0("Starting training loop...")
    report = get_report()
    total_time = 0
    
    for step in range(args.total_steps):
        t0 = time.time()
        
        # --- 学习率调度 ---
        lr = get_lr(step, args.total_steps, args.warmup_steps, args.learning_rate, args.learning_rate * 0.1)
        for opt in optimizers:
            for param_group in opt.param_groups:
                param_group['lr'] = lr
        
        # --- 前向传播 & 反向传播 ---
        # 获取一个 Batch
        inputs, targets, _ = next(train_loader)
        
        # 前向计算 Loss
        # 使用 BFloat16 混合精度 (RTX 30/40/50 必备)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            loss = model(inputs, targets)
            
        # 反向传播
        loss.backward()
        
        # --- 梯度更新 ---
        # 梯度裁剪 (防止爆炸)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # 优化器步进
        for opt in optimizers:
            opt.step()
            opt.zero_grad(set_to_none=True) # 清空梯度
            
        # 等待 GPU 完成 (用于计时准确)
        torch.cuda.synchronize()
        t1 = time.time()
        dt = t1 - t0
        total_time += dt
        
        # --- 日志打印 ---
        if step % 10 == 0:
            tokens_per_sec = (args.batch_size * args.sequence_len * ddp_world_size) / dt
            print0(f"step {step:4d}/{args.total_steps} | loss {loss.item():.4f} | lr {lr:.2e} | {dt*1000:.1f}ms | {tokens_per_sec:.0f} tok/s")
            
            report.log("Training", {
                "step": step, "loss": loss.item(), "lr": lr, "dt": dt
            })

        # --- 验证循环 (Validation) ---
        if step > 0 and (step % args.val_every == 0 or step == args.total_steps - 1):
            print0(f"Running validation...")
            model.eval()
            val_loss = 0
            val_steps = 20 # 验证 20 个 batch
            with torch.no_grad():
                for _ in range(val_steps):
                    inputs_v, targets_v, _ = next(val_loader)
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                        loss_v = model(inputs_v, targets_v)
                    val_loss += loss_v.item()
            val_loss /= val_steps
            print0(f"✅ Validation loss: {val_loss:.4f}")
            report.log("Validation", {"step": step, "val_loss": val_loss})
            model.train() # 切回训练模式

        # --- 模型保存 (Checkpoint) ---
        if step > 0 and (step % args.save_every == 0 or step == args.total_steps - 1):
            if ddp_rank == 0:
                ckpt_path = os.path.join(args.output_dir, f"ckpt_{step:05d}.pt")
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "config": asdict(config),
                    "step": step,
                    "val_loss": val_loss if 'val_loss' in locals() else None
                }
                torch.save(checkpoint, ckpt_path)
                print0(f"💾 Saved checkpoint to {ckpt_path}")

    compute_cleanup()
    print0("Training finished!")

if __name__ == "__main__":
    main()