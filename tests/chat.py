"""
MiniGPT 交互式聊天脚本 (CLI)
"""
import os
import time
import torch
import argparse
from minigpt.gpt import GPT, GPTConfig
from minigpt.tokenizer import get_tokenizer
from minigpt.common import autodetect_device_type

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="output/ckpt_00099.pt", help="模型检查点路径")
    parser.add_argument("--temperature", type=float, default=0.8, help="生成温度 (越高越随机)")
    parser.add_argument("--top_k", type=int, default=200, help="Top-K 采样")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="最大生成长度")
    args = parser.parse_args()

    # 1. 准备设备
    device_type = autodetect_device_type()
    device = torch.device(device_type)
    print(f"✨ Using device: {device}")

    # 2. 加载 Tokenizer
    print("📚 Loading tokenizer...")
    tokenizer = get_tokenizer()

    # 3. 加载模型 Checkpoint
    if not os.path.exists(args.ckpt):
        print(f"❌ Error: Checkpoint not found at {args.ckpt}")
        return

    print(f"📦 Loading model from {args.ckpt}...")
    checkpoint = torch.load(args.ckpt, map_location=device)
    
    # 从 checkpoint 中恢复配置
    # 这一点很关键：必须用训练时的同样配置来实例化模型
    gpt_conf = GPTConfig(**checkpoint["config"])
    model = GPT(gpt_conf)
    
    # 加载权重
    state_dict = checkpoint["model"]
    # 处理可能的 DDP 前缀 (如果是用多卡训练的，key 可能会有 "_orig_mod." 或 "module.")
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval() # 切换到评估模式 (关闭 Dropout 等)
    print("✅ Model loaded successfully!")

    # 4. 进入聊天循环
    print("\n💬 开始对话 (输入 'exit' 或 'quit' 退出)")
    print("-" * 50)

    while True:
        try:
            prompt = input("User: ")
        except EOFError:
            break
            
        if prompt.lower() in ["exit", "quit"]:
            break
        
        if not prompt.strip():
            continue

        # 编码输入
        # 这里我们要手动构造对话格式吗？
        # 为了简单演示，我们先直接把用户输入当成 prompt (续写模式)，不加特殊的 Chat 模板
        # 如果模型足够强，它会学会续写；如果模型很弱，它会乱说。
        
        # 简单的 Encode
        input_ids = tokenizer.encode(prompt, prepend=tokenizer.get_bos_token_id())
        
        # 生成
        print("Assistant: ", end="", flush=True)
        
        # 记录开始时间
        t0 = time.time()
        
        # 生成循环
        gen_tokens = []
        # generate 返回的是 generator，我们可以迭代它来实现流式打印
        # 但要注意我们的 generate 实现目前是 yield 每一个 token
        for token_id in model.generate(input_ids, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k):
            # 解码当前 token
            # 注意：单个 token 解码可能会乱码（对于多字节字符），但这在英文语境下通常没问题
            # 严谨做法是积攒 bytes 再解码，这里简化处理
            word = tokenizer.decode([token_id])
            print(word, end="", flush=True)
            gen_tokens.append(token_id)
            
        print("\n")
        t1 = time.time()
        tokens_sec = len(gen_tokens) / (t1 - t0)
        print(f"--- (Speed: {tokens_sec:.2f} tok/s) ---")
        print("-" * 50)

if __name__ == "__main__":
    main()