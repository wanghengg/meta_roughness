import pandas as pd
import torch
import random

def load_dataset(path):
    return pd.read_csv(path)

# 计算全局统计信息
def compute_global_stats(df):
    """计算数据集的全局均值和标准差"""
    features = df.iloc[:, :-1]
    mean = features.mean().values
    std = features.std().values + 1e-8  # 添加小值避免除零
    return mean, std

def build_tasks(df, tasks_per_epoch, support_k, query_k, global_mean=None, global_std=None):
    tasks = []

    # 如果没有提供全局统计信息，则计算
    if global_mean is None or global_std is None:
        global_mean, global_std = compute_global_stats(df)
    
    # 转换为tensor
    global_mean_tensor = torch.tensor(global_mean, dtype=torch.float32).unsqueeze(0)
    global_std_tensor = torch.tensor(global_std, dtype=torch.float32).unsqueeze(0)

    for _ in range(tasks_per_epoch):
        sampled = df.sample(support_k + query_k)

        support = sampled[:support_k]
        query = sampled[support_k:]

        # 提取数据并转换为tensor
        support_x = torch.tensor(support.iloc[:, :-1].values, dtype=torch.float32)
        support_y = torch.tensor(support.iloc[:, -1].values, dtype=torch.float32).unsqueeze(1)
        query_x = torch.tensor(query.iloc[:, :-1].values, dtype=torch.float32)
        query_y = torch.tensor(query.iloc[:, -1].values, dtype=torch.float32).unsqueeze(1)

        # 使用全局统计信息进行标准化
        support_x = (support_x - global_mean_tensor) / global_std_tensor
        query_x = (query_x - global_mean_tensor) / global_std_tensor

        tasks.append((support_x, support_y, query_x, query_y))

    return tasks