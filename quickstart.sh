#!/bin/bash
set -e 

echo "欢迎来到MiniGPT"

echo "[1/5] 检查基础环境..."
# 检查/安装 Rust
if ! command -v cargo &> /dev/null; then
    echo "   -> 安装 Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
else
    echo "   -> Rust 已就绪"
fi

# 检查/安装 uv
if ! command -v uv &> /dev/null; then
    echo "   -> 安装 uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.cargo/env" 2>/dev/null || true
else
    echo "   -> uv 已就绪"
fi

# 检查 Python 开发头文件 (针对 Linux 编译 PyTorch/Triton)
if [ -f /etc/debian_version ]; then
    echo "   -> 安装 python3.12-dev (需要 sudo 密码)..."
    sudo apt-get update && sudo apt-get install -y python3.12-dev || echo "自动安装失败，请确保已安装 python3.12-dev"
fi


echo "[2/5] 构建项目环境..."
# 同步python环境
uv sync
# 编译 Rust 扩展 (rustbpe)
echo "   -> 编译 rustbpe 扩展..."
uv run maturin develop --release


echo "[3/5] 准备数据..."
echo "   -> 下载 FineWeb-Edu 数据集..."
# 前5个分片
uv run -m minigpt.dataset -n 5  

echo "   -> 训练 Tokenizer..."
# 读取下载的数据，训练词表，并保存到 ~/.cache/minigpt/tokenizer
uv run -m tests.train_tokenizer


echo "[4/5] 开始训练模型..."

# 清理旧的输出目录 (可选)
rm -rf output

# 启动训练
# 参数说明：
# --batch_size 4: 适配 8GB 显存
# --sequence_len 512: 减小上下文长度以节省显存
# --total_steps 200: 跑 200 步用于演示 (你可以改成 5000 或更多)
# --save_every 200: 训练结束时保存一次
uv run -m tests.train \
    --batch_size 4 \
    --sequence_len 512 \
    --total_steps 200 \
    --save_every 200 \
    --output_dir output


echo "[5/5] 启动对话..."

CHECKPOINT="output/ckpt_00199.pt" # 注意：这里的步数要和 total_steps 对应

if [ -f "$CHECKPOINT" ]; then
    echo "训练完成！正在加载模型: $CHECKPOINT"
    echo "---------------------------------------------------"
    # 启动聊天脚本
    uv run -m tests.chat --ckpt "$CHECKPOINT"
else
    echo "未找到模型权重文件，训练可能失败。"
    exit 1
fi