"""
训练 BPE Tokenizer 并保存到磁盘。
"""
import os
import time
import argparse
import torch
from minigpt.tokenizer import RustBPETokenizer
from minigpt.common import get_base_dir
from minigpt.dataset import parquets_iter_batched
from minigpt.report import get_report

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Train a BPE tokenizer')
parser.add_argument('--vocab_size', type=int, default=50304, help='词表大小 (默认 50304，即 GPT-2 50257 + padding)')
args = parser.parse_args()

# -----------------------------------------------------------------------------
# 文本迭代器：从 parquet 文件中流式读取文本
def text_iterator():
    # 为了演示速度，我们只训练前 100M 字符，这对于小词表足够了
    max_chars = 100_000_000 
    nchars = 0
    for batch in parquets_iter_batched(split="train"):
        for doc in batch:
            nchars += len(doc)
            yield doc
            if nchars > max_chars:
                return
text_iter = text_iterator()

# -----------------------------------------------------------------------------
# 开始训练
print(f"Training tokenizer with vocab_size={args.vocab_size}...")
t0 = time.time()
# 调用 Rust 核心进行并行训练
tokenizer = RustBPETokenizer.train_from_iterator(text_iter, args.vocab_size)
t1 = time.time()
print(f"Training completed in {t1 - t0:.2f}s")

# -----------------------------------------------------------------------------
# 保存到 ~/.cache/minigpt/tokenizer
base_dir = get_base_dir()
tokenizer_dir = os.path.join(base_dir, "tokenizer")
tokenizer.save(tokenizer_dir)

# -----------------------------------------------------------------------------
# 简单的自检
test_text = "Hello world! This is miniGPT."
encoded = tokenizer.encode(test_text)
decoded = tokenizer.decode(encoded)
assert decoded == test_text, "编解码一致性测试失败"
print(f"Sanity check passed: '{test_text}' -> {encoded} -> '{decoded}'")

# -----------------------------------------------------------------------------
# 生成 token_bytes 映射表 (用于计算 bits per byte 指标)
# 这是一个高级评估指标，如果不做评估可以跳过，但为了完整复刻我们加上
vocab_size = tokenizer.get_vocab_size()
special_set = set(tokenizer.get_special_tokens())
token_bytes = []

for token_id in range(vocab_size):
    try:
        token_str = tokenizer.id_to_token(token_id)
        if token_str in special_set:
            token_bytes.append(0)
        else:
            token_bytes.append(len(token_str.encode("utf-8")))
    except:
        token_bytes.append(0) # 防御性编程

token_bytes = torch.tensor(token_bytes, dtype=torch.int32, device='cpu')
token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
torch.save(token_bytes, token_bytes_path)
print(f"Saved token_bytes map to {token_bytes_path}")

# 记录报告
get_report().log(section="Tokenizer Training", data=[
    {"vocab_size": args.vocab_size},
    {"train_time": t1 - t0},
    {"save_path": tokenizer_dir}
])