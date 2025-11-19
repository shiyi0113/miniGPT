"""
基础/预训练 数据集管理 (FineWeb-Edu).
包含用于下载和遍历 Parquet 文件的工具。
"""

import os
import argparse
import time
import requests
import pyarrow.parquet as pq
from multiprocessing import Pool

from minigpt.common import get_base_dir

# -----------------------------------------------------------------------------
# 预训练数据集配置

# 数据托管在 HuggingFace，按需下载  总数据集大小171GB，单个分片大小为95MB
BASE_URL = "https://hf-mirror.com/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main"
MAX_SHARD = 1822 # 最后一个分片是 shard_01822.parquet
index_to_filename = lambda index: f"shard_{index:05d}.parquet"

# 数据存储路径：~/.cache/minigpt/base_data
base_dir = get_base_dir()
DATA_DIR = os.path.join(base_dir, "base_data")
os.makedirs(DATA_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# 数据集工具

def list_parquet_files(data_dir=None):
    """列出目录下所有的 parquet 文件路径"""
    data_dir = DATA_DIR if data_dir is None else data_dir
    if not os.path.exists(data_dir):
        return []
    parquet_files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith('.parquet') and not f.endswith('.tmp')
    ])
    parquet_paths = [os.path.join(data_dir, f) for f in parquet_files]
    return parquet_paths

def parquets_iter_batched(split, start=0, step=1):
    """
    批量迭代数据集中的文档。
    - split: 'train' 或 'val' (最后一个文件作为验证集)
    - start/step: 用于 DDP 分布式训练时的分片读取 (Rank/World_Size)
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    parquet_paths = list_parquet_files()
    
    # 简单的划分：最后一个文件做验证集，其余做训练集
    # 如果只有一个文件，训练集和验证集可能会重叠或报错，建议至少下载2个文件
    if len(parquet_paths) == 0:
        return # 没数据就什么都不产出

    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
    
    for filepath in parquet_paths:
        try:
            pf = pq.ParquetFile(filepath)
            # 按 row_group 读取，避免一次性加载整个文件爆内存
            for rg_idx in range(start, pf.num_row_groups, step):
                rg = pf.read_row_group(rg_idx)
                texts = rg.column('text').to_pylist()
                yield texts
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            continue

# -----------------------------------------------------------------------------
# 下载数据集
def download_single_file(index):
    """下载单个文件，带有重试机制"""
    filename = index_to_filename(index)
    filepath = os.path.join(DATA_DIR, filename)
    
    if os.path.exists(filepath):
        print(f"Skipping {filepath} (already exists)")
        return True

    url = f"{BASE_URL}/{filename}"
    print(f"Downloading {filename}...")

    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            # 先写入临时文件，下载完成后重命名，防止断网导致文件损坏
            temp_path = filepath + f".tmp"
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024): # 1MB chunks
                    if chunk:
                        f.write(chunk)
            
            os.rename(temp_path, filepath)
            print(f"Successfully downloaded {filename}")
            return True

        except (requests.RequestException, IOError) as e:
            print(f"Attempt {attempt}/{max_attempts} failed for {filename}: {e}")
            # 清理临时文件
            if os.path.exists(filepath + ".tmp"):
                try: os.remove(filepath + ".tmp")
                except: pass
            
            if attempt < max_attempts:
                wait_time = 2 ** attempt
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"Failed to download {filename} after {max_attempts} attempts")
                return False

    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download FineWeb-Edu dataset shards")
    parser.add_argument("-n", "--num-files", type=int, default=5, help="下载的分片数量 (默认: 5)")
    parser.add_argument("-w", "--num-workers", type=int, default=4, help="并行下载的线程数")
    args = parser.parse_args()

    # 限制下载数量
    num = min(args.num_files, MAX_SHARD + 1)
    ids_to_download = list(range(num))
    print(ids_to_download)

    print(f"Downloading {len(ids_to_download)} shards to {DATA_DIR}...")
    
    # 并行下载
    with Pool(processes=args.num_workers) as pool:
        results = pool.map(download_single_file, ids_to_download)

    successful = sum(1 for success in results if success)
    print(f"Done! Downloaded: {successful}/{len(ids_to_download)} shards.")
