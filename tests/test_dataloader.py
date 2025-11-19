import torch
from minigpt.dataloader import tokenizing_distributed_data_loader
from minigpt.common import autodetect_device_type
from minigpt.tokenizer import get_tokenizer

def test_dataloader_smoke():
    print("🚚 开始 DataLoader 冒烟测试...")
    
    # 1. 配置参数
    B = 4   # Batch Size
    T = 32  # Sequence Length (时间步)
    device_type = autodetect_device_type()
    
    # 2. 初始化加载器
    print(f"📦 初始化加载器 (B={B}, T={T}, Device={device_type})...")
    try:
        train_loader = tokenizing_distributed_data_loader(
            B=B, 
            T=T, 
            split="train", 
            tokenizer_batch_size=256, 
            device=device_type
        )
    except Exception as e:
        print(f"❌ 加载器初始化失败: {e}")
        return

    # 3. 尝试获取第一个 Batch
    print("🔄 正在读取并处理第一个 Batch (可能需要几秒钟进行 Tokenize)...")
    try:
        # next() 会触发：读取Parquet -> Tokenize -> Tensor转换 -> GPU传输
        inputs, targets, state = next(train_loader)
        
        # 验证形状
        assert inputs.shape == (B, T), f"Inputs 形状错误: {inputs.shape}"
        assert targets.shape == (B, T), f"Targets 形状错误: {targets.shape}"
        print(f"✅ Batch 读取成功! Shape: {inputs.shape}")
        print(f"   State: {state}")
        
    except Exception as e:
        print(f"❌ 读取 Batch 失败: {e}")
        # 常见错误提示：如果是 FileNotFoundError，说明 dataset 下载路径不对
        import traceback
        traceback.print_exc()
        return

    # 4. 可视化验证 (Visual Inspection)
    # 把 Tensor 变回文本，看看是不是人话
    print("\n👀 数据可视化检查 (解码第一条数据):")
    tokenizer = get_tokenizer()
    
    # 取第一条数据的 token id 列表
    input_ids = inputs[0].tolist()
    target_ids = targets[0].tolist()
    
    text_in = tokenizer.decode(input_ids)
    text_target = tokenizer.decode(target_ids)
    
    print("-" * 40)
    print(f"【输入 Inputs】:\n{text_in}")
    print("-" * 40)
    print(f"【目标 Targets】(应该是输入的向左移一位):\n{text_target}")
    print("-" * 40)
    
    # 简单验证 shift 逻辑
    # input: A B C D
    # target: B C D E
    if input_ids[1:] == target_ids[:-1]:
        print("✅ 移位逻辑验证正确 (Inputs[1:] == Targets[:-1])")
    else:
        print("⚠️ 警告: 移位逻辑似乎不对，请检查打印出的文本")

    print("\n🎉 恭喜！数据流水线畅通无阻！")

if __name__ == "__main__":
    test_dataloader_smoke()