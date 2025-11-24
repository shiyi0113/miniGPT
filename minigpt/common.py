"""
常用工具
"""
import os
import re
import logging
import torch
import torch.distributed as dist

class ColoredFormatter(logging.Formatter):
    """自定义日志格式，添加颜色高亮"""
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'
    def format(self, record):
        # 给日志级别加颜色
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{self.BOLD}{levelname}{self.RESET}"
        # 格式化消息
        message = super().format(record)
        # 高亮消息中的特定部分（如数字、百分比）
        if levelname == 'INFO':
            message = re.sub(r'(\d+\.?\d*\s*(?:GB|MB|%|docs))', rf'{self.BOLD}\1{self.RESET}', message)
            message = re.sub(r'(Shard \d+)', rf'{self.COLORS["INFO"]}{self.BOLD}\1{self.RESET}', message)
        return message

def setup_default_logging():
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.basicConfig(
        level=logging.INFO,
        handlers=[handler]
    )

setup_default_logging()
logger = logging.getLogger(__name__)

def get_base_dir():
    # 将中间文件缓存到 ~/.cache/minigpt
    if os.environ.get("MINIGPT_BASE_DIR"):
        minigpt_dir = os.environ.get("MINIGPT_BASE_DIR")
    else:
        home_dir = os.path.expanduser("~")
        cache_dir = os.path.join(home_dir, ".cache")
        minigpt_dir = os.path.join(cache_dir, "minigpt")
    os.makedirs(minigpt_dir, exist_ok=True)
    return minigpt_dir

def print0(s="", **kwargs):
    """分布式训练时，只在主进程打印"""
    # 主进程 RANK=0 非分布式时没有RANK，也是0
    ddp_rank = int(os.environ.get('RANK', 0))
    if ddp_rank == 0:
        print(s, **kwargs)

def get_dist_info():
    """获取分布式训练的环境信息"""
    if int(os.environ.get('RANK', -1)) != -1:
        assert all(var in os.environ for var in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE'])
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        return True, ddp_rank, ddp_local_rank, ddp_world_size
    else:
        return False, 0, 0, 1
    
def autodetect_device_type():
    """自动检测设备：优先 CUDA,其次 MPS,最后 CPU"""
    if torch.cuda.is_available():
        device_type = "cuda"
    elif torch.backends.mps.is_available():
        device_type = "mps"
    else:
        device_type = "cpu"
    print0(f"Autodetected device type: {device_type}")
    return device_type

def compute_init(device_type="cuda"): 
    """
    初始化计算环境：随机种子、精度设置、分布式组网 (DDP)
    """
    assert device_type in ["cuda", "mps", "cpu"], "Invalid device type atm"
    if device_type == "cuda":
        assert torch.cuda.is_available(), "CUDA requested but not available"
    if device_type == "mps":
        assert torch.backends.mps.is_available(), "MPS requested but not available"

    # 复现性设置
    torch.manual_seed(42)
    if device_type == "cuda":
        torch.cuda.manual_seed(42)

    # 精度设置：RTX 30/40/50 系列开启 TF32
    if device_type == "cuda":
        torch.set_float32_matmul_precision("high") 

    # 分布式环境初始化 (DDP)
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    if ddp and device_type == "cuda":
        device = torch.device("cuda", ddp_local_rank)
        torch.cuda.set_device(device)
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    else:
        device = torch.device(device_type)

    if ddp_rank == 0:
        logger.info(f"Distributed world size: {ddp_world_size}")

    return ddp, ddp_rank, ddp_local_rank, ddp_world_size, device

def compute_cleanup():
    """清理分布式进程组"""
    if int(os.environ.get('RANK', -1)) != -1:
        dist.destroy_process_group()

class DummyWandb:
    """当不使用 wandb 时提供的哑类，防止报错"""
    def __init__(self):
        pass
    def log(self, *args, **kwargs):
        pass
    def finish(self):
        pass