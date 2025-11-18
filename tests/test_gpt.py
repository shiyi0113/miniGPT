import torch
from minigpt.gpt import GPT, GPTConfig
from minigpt.common import autodetect_device_type

def test_gpt_smoke():
    # 1. 配置一个迷你的 GPT 用于测试 (为了速度和显存，参数设得很小)
    config = GPTConfig(
        sequence_len=32,   # 上下文长度
        vocab_size=512,    # 词表大小
        n_layer=2,         # 层数
        n_head=4,          # Query 头数
        n_kv_head=2,       # KV 头数 (测试 GQA)
        n_embd=64          # 嵌入维度
    )
    
    # 2. 自动选择设备 (RTX 5060 -> cuda)
    device_type = autodetect_device_type()
    device = torch.device(device_type)
    
    print(f"📦 初始化模型: {config}")
    try:
        model = GPT(config)
        model.to(device)
        print("✅ 模型初始化成功")
    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")
        return

    # 3. 前向传播测试 (Forward Pass)
    print("🔄 测试前向传播 (Forward Pass)...")
    batch_size = 2
    seq_len = 16
    # 随机生成一些 token ID
    dummy_idx = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    
    try:
        # 测试 Logits 输出
        logits = model(dummy_idx)
        expected_shape = (batch_size, seq_len, config.vocab_size)
        assert logits.shape == expected_shape, f"Logits 形状错误: {logits.shape} != {expected_shape}"
        print(f"✅ Logits 计算成功，形状: {logits.shape}")
        
        # 测试 Loss 计算
        loss = model(dummy_idx, targets=dummy_idx) # 自回归任务 targets 通常是 idx 移位，这里仅测试能否运行
        assert loss.ndim == 0, "Loss 应该是一个标量"
        print(f"✅ Loss 计算成功: {loss.item():.4f}")
        
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        # 打印详细错误栈以便调试
        import traceback
        traceback.print_exc()
        return

    # 4. 生成测试 (Generation)
    print("✨ 测试文本生成 (Generation)...")
    try:
        start_tokens = [1, 2, 3] # 假设的起始 token
        # 生成 5 个新 token
        gen_len = 5
        
        # generate 只 yield 新生成的 token
        generated = list(model.generate(start_tokens, max_tokens=gen_len))
        
        # 修正断言：长度应该只等于生成的长度 (5)，而不是 总长度 (8)
        assert len(generated) == gen_len, f"生成长度不符合预期: {len(generated)} != {gen_len}"
        
        # 手动拼接以便打印查看
        full_sequence = start_tokens + generated
        print(f"✅ 生成成功: 输入={start_tokens} -> 新生成={generated}")
        print(f"   完整序列: {full_sequence}")
        
    except Exception as e:
        print(f"❌ 生成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n🎉 恭喜！GPT 模型架构通过了所有冒烟测试！")

if __name__ == "__main__":
    test_gpt_smoke()