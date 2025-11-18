import rustbpe
import time
import os
from pathlib import Path

def test_rustbpe():
    # 1. 实例化
    try:
        tokenizer = rustbpe.Tokenizer()
        print("✅ RUSTBPE Tokenizer 实例化成功")
    except Exception as e:
        print(f"❌ 实例化失败: {e}")
        return

    # 2. 准备测试数据
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir,'taylorswift.txt')
    data = Path(data_path).read_text(encoding='utf-8')
    
    # 3. 训练
    # 词表大小设为 300 (基础 256 字节 + 44 个合并规则)
    print("⏳ 开始训练 (vocab_size=300)...")
    t0 = time.time()
    try:
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
        print(f"   原始长度: {len(test_str)}, Token 数量: {len(ids)}")
    except Exception as e:
        print(f"❌ 编码失败: {e}")
        return

    print("\n🎉 恭喜！Rust Tokenizer 模块运行正常！")

if __name__ == "__main__":
    test_rustbpe()