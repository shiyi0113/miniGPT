# MiniGPT: 从零复现的极简 ChatGPT

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)
![Rust](https://img.shields.io/badge/rust-enabled-orange.svg)

**MiniGPT** 是一个全栈式的生成式语言模型（LLM）学习项目。它旨在用最精简、可读性最高的代码，从零复现类似 ChatGPT 的全流程，涵盖了数据清洗、Tokenizer 训练（Rust 加速）、预训练流水线到最终的对话推理。

## 🛠️ 环境与硬件要求

* **操作系统**: Linux
* **软件依赖**:
    * Python >= 3.12
    * Rust (用于编译扩展)
    * uv (Python 包管理器，**强烈推荐**)
* **硬件参考**:
    * 本项目代码编写与测试于 **NVIDIA RTX 5060** 环境。
    * 默认配置 (`batch_size=4`, `seq_len=512`) 经过优化，适配 8GB 显存的消费级显卡。

## 🚀 快速开始

我们提供了一个全流程脚本 `quickstart.sh`，能够自动完成环境配置、Rust 编译、数据下载、模型训练和对话测试。

```bash
git clone https://github.com/shiyi0113/miniGPT.git
cd minigpt

chmod +x quickstart.sh
./quickstart.sh
```
## 📂 项目结构 (Project Structure)

```text
minigpt/
├── minigpt/            # 核心 Python 源码包
│   ├── gpt.py          # GPT 模型架构定义
│   ├── tokenizer.py    # RustBPE 的 Python 包装器
│   ├── dataset.py      # 数据集下载与分片管理
│   ├── dataloader.py   # 分布式流式数据加载器
│   ├── muon.py         # Muon 优化器实现
│   ├── adamw.py        # AdamW 优化器实现
│   └── report.py       # 训练日志记录工具
├── rustbpe/            # Rust 实现的高性能 BPE
│   ├── src/lib.rs      
│   └── Cargo.toml      
├── tests/              # 测试脚本
│   ├── train.py        
│   ├── chat.py         
│   ├── tok_train.py    
│   └── ...
├── pybpe/              # Python 版 BPE 实现
├── quickstart.sh       # 一键启动脚本
└── pyproject.toml      # Python 项目依赖与构建配置 (uv/maturin)
```

## 🗓️ 路线图 (Roadmap)

* [x] **基础预训练 (Pre-training)**: 实现完整的数据流、模型定义与训练循环。
* [x] **混合精度训练 (BF16)**: 全程支持 BFloat16，适配现代 GPU 架构。
* [x] **Rust 加速 Tokenizer**: 完成 RustBPE 的实现与 Python 绑定。
* [ ] **Web UI 界面**
* [ ] **指令微调 (SFT)**
* [ ] **RLHF 流程**

## 🤝 致谢 (Acknowledgements)

本项目在开发过程中深入参考了 **Andrej Karpathy** 的 **nanochat** 项目及相关教学代码。特别感谢 Karpathy 在 LLM 教育领域的杰出贡献，其开源精神极大地降低了我们理解现代大模型底层原理的门槛。

## 📄 开源协议 (License)

本项目遵循 **MIT License** 开源协议。