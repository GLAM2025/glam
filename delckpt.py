import os
import re

def clean_ckpt_files(base_path):
    """
    遍历 base_path 下的所有文件夹，只保留编号是 2500 的整数倍的模型文件。
    """
    # 遍历 base_path 下的所有子目录
    for root, dirs, files in os.walk(base_path):
        # 筛选出以 `world_model_` 或 `agent_` 开头的文件
        model_files = [f for f in files if f.startswith("world_model_") or f.startswith("agent_")]
        
        # 提取文件编号并筛选出 2500 的整数倍
        valid_files = set()
        for file in model_files:
            match = re.search(r"_(\d+)\.pth$", file)  # 匹配文件名中的编号
            if match:
                step = int(match.group(1))
                if step % 2500 == 0:  # 只保留编号是 2500 的整数倍的文件
                    valid_files.add(file)
        
        # 删除不符合条件的文件
        for file in model_files:
            if file not in valid_files:
                file_path = os.path.join(root, file)
                print(f"Deleting: {file_path}")
                os.remove(file_path)

if __name__ == "__main__":
    base_path = "/home/hq/LSTW/GLAM/data/ckpt"  # 修改为你的 ckpt 文件夹路径
    clean_ckpt_files(base_path)