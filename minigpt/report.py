"""
训练报告生成工具。
"""
import os
import datetime
from minigpt.common import get_base_dir, get_dist_info

class Report:
    def __init__(self, report_dir):
        os.makedirs(report_dir, exist_ok=True)
        self.report_dir = report_dir

    def log(self, section, data):
        """将一段数据记录到报告中"""
        slug = section.lower().replace(" ", "-")
        file_name = f"{slug}.md"
        file_path = os.path.join(self.report_dir, file_name)
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"## {section}\n")
            f.write(f"timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            for item in data:
                if not item: continue
                if isinstance(item, str):
                    f.write(item)
                else:
                    for k, v in item.items():
                        if isinstance(v, float): vstr = f"{v:.4f}"
                        else: vstr = str(v)
                        f.write(f"- {k}: {vstr}\n")
            f.write("\n")
        return file_path

class DummyReport:
    def log(self, *args, **kwargs): pass

def get_report():
    # 只有 Rank 0 负责写报告
    ddp, ddp_rank, _, _ = get_dist_info()
    if ddp_rank == 0:
        report_dir = os.path.join(get_base_dir(), "report")
        return Report(report_dir)
    else:
        return DummyReport()