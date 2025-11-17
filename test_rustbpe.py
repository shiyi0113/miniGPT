import rustbpe
import time

def test_rustbpe():
    print("🔍 开始测试 rustbpe 模块...")
    
    # 1. 实例化
    try:
        tokenizer = rustbpe.Tokenizer()
        print("✅ Tokenizer 实例化成功")
    except Exception as e:
        print(f"❌ 实例化失败: {e}")
        return

    # 2. 准备测试数据
    # 制造一些重复模式以便 BPE 能够学习到合并规则
    data = [
        "hello world " * 20,
        "hello python " * 20,
        "rust is fast " * 20,
        "learning ai infra is cool " * 20
    ]
    
    # 3. 训练
    # 词表大小设为 300 (基础 256 字节 + 44 个合并规则)
    print("⏳ 开始训练 (vocab_size=300)...")
    t0 = time.time()
    try:
        # train_from_iterator(iterator, vocab_size, buffer_size, pattern)
        tokenizer.train_from_iterator(data, 300)
        dt = time.time() - t0
        print(f"✅ 训练完成，耗时: {dt:.4f}s")
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        return

    # 4. 编码测试
    test_str = "hello world rust is cool"
    try:
        ids = tokenizer.encode(test_str)
        print(f"✅ 编码结果: '{test_str}' -> {ids}")
        
        # 简单验证：如果有合并发生，token 数量应该少于字符数量
        # (注意：这里不一定严格小于，取决于训练数据是否覆盖了测试句子的模式，但大概率会变短)
        print(f"   原始长度: {len(test_str)}, Token 数量: {len(ids)}")
    except Exception as e:
        print(f"❌ 编码失败: {e}")
        return

    print("\n🎉 恭喜！Rust Tokenizer 模块运行正常！")

if __name__ == "__main__":
    test_rustbpe()