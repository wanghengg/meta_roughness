import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from models.maml_model import RoughnessNet
from utils.task_sampler import load_dataset, build_tasks, compute_global_stats
from scripts.inner_loop import inner_update
from config import *

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = [
    'PingFang SC',      # 苹方简体（macOS默认中文字体）
    'Heiti SC',         # 黑体简体
    'Hiragino Sans GB', # 冬青黑体简体
    'STHeiti',          # 华文黑体
    'Microsoft YaHei',  # 微软雅黑
    'SimHei',           # 黑体
    'Arial Unicode MS', # Arial Unicode
    'DejaVu Sans'       # DejaVu字体
]
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

try:
    import torch
    print(f"成功导入PyTorch，版本: {torch.__version__}")
except ImportError as e:
    print(f"错误: 无法导入PyTorch - {e}")
    exit(1)

try:
    from models.maml_model import RoughnessNet
    from utils.task_sampler import load_dataset, build_tasks
    from inner_loop import inner_update
    from config import *
    print("成功导入所有项目模块")
except ImportError as e:
    print(f"错误: 无法导入项目模块 - {e}")
    exit(1)

import os
data_file = "data/scicp_al_sample.csv"
if not os.path.exists(data_file):
    print(f"错误: 数据文件 {data_file} 不存在")
    exit(1)

try:
    df = load_dataset(data_file)
    print(f"成功加载数据，共 {len(df)} 条记录")
except Exception as e:
    print(f"错误: 加载数据失败 - {e}")
    exit(1)

try:
    model = RoughnessNet(INPUT_DIM)
    print(f"成功创建模型，输入维度: {INPUT_DIM}")
    
    model_file = "maml_model.pth"
    if os.path.exists(model_file):
        model.load_state_dict(torch.load(model_file))
        print("成功加载预训练模型")
    else:
        print(f"警告: 模型文件 {model_file} 不存在，使用随机初始化权重")
        
except Exception as e:
    print(f"错误: 模型创建或加载失败 - {e}")
    exit(1)

try:
    # 计算全局统计信息用于标准化
    global_mean, global_std = compute_global_stats(df)
    tasks = build_tasks(df, 5, SUPPORT_K, QUERY_K, global_mean, global_std)
    print(f"成功构建 {len(tasks)} 个任务")
except Exception as e:
    print(f"错误: 构建任务失败 - {e}")
    exit(1)

print("\n开始执行测试...")

# 存储所有任务的预测结果和真实值
all_predictions = []
all_targets = []
task_mses = []

for i, (support_x, support_y, query_x, query_y) in enumerate(tasks):
    try:
        adapted = inner_update(model, support_x, support_y, INNER_LR)
        pred = model(query_x, adapted)
        
        # 计算MSE
        mse = torch.mean((pred - query_y) ** 2)
        task_mses.append(mse.item())
        
        # 存储预测值和真实值
        all_predictions.extend(pred.detach().cpu().numpy())
        all_targets.extend(query_y.detach().cpu().numpy())
        
        print(f"Task {i}, MSE: {mse.item():.6f}")
    except Exception as e:
        print(f"Task {i} 执行失败: {e}")

# 绘制预测值与真实值的对比图
if all_predictions and all_targets:
    # 转换为numpy数组
    predictions_np = np.array(all_predictions)
    targets_np = np.array(all_targets)
    residuals = predictions_np - targets_np
    
    plt.figure(figsize=(12, 10))
    
    # 1. 散点图：预测值 vs 真实值
    plt.subplot(2, 2, 1)
    plt.scatter(targets_np, predictions_np, alpha=0.6, color='blue', s=30)
    # 添加理想预测线
    min_val = min(min(targets_np), min(predictions_np))
    max_val = max(max(targets_np), max(predictions_np))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='理想预测线')
    plt.xlabel('真实值 (Ra)')
    plt.ylabel('预测值 (Ra)')
    plt.title('预测值 vs 真实值散点图')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. 残差图
    plt.subplot(2, 2, 2)
    plt.scatter(targets_np, residuals, alpha=0.6, color='green', s=30)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('真实值 (Ra)')
    plt.ylabel('残差 (预测值 - 真实值)')
    plt.title('残差分布图')
    plt.grid(True, alpha=0.3)
    
    # 3. 误差直方图
    plt.subplot(2, 2, 3)
    plt.hist(residuals, bins=30, alpha=0.7, color='orange', edgecolor='black')
    plt.xlabel('残差值')
    plt.ylabel('频次')
    plt.title('预测误差分布直方图')
    plt.grid(True, alpha=0.3)
    
    # 4. 各任务MSE对比
    plt.subplot(2, 2, 4)
    bars = plt.bar(range(len(task_mses)), task_mses, alpha=0.7, color='purple')
    plt.xlabel('任务编号')
    plt.ylabel('MSE')
    plt.title('各任务预测误差(MSE)')
    plt.grid(True, alpha=0.3)
    # 在柱状图上添加数值标签
    for i, (bar, mse) in enumerate(zip(bars, task_mses)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001, 
                f'{mse:.5f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('prediction_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 输出统计信息
    overall_mse = np.mean(task_mses)
    overall_rmse = np.sqrt(overall_mse)
    
    # 安全计算R²
    try:
        correlation_matrix = np.corrcoef(targets_np, predictions_np)
        if not np.isnan(correlation_matrix).any() and len(targets_np) > 1:
            r_squared = correlation_matrix[0, 1] ** 2
        else:
            r_squared = 0.0
    except:
        r_squared = 0.0
    
    mae = np.mean(np.abs(residuals))
    max_error = np.max(np.abs(residuals))
    min_error = np.min(np.abs(residuals))
    
    print(f"\n=== 预测性能统计 ===")
    print(f"整体MSE: {overall_mse:.6f}")
    print(f"整体RMSE: {overall_rmse:.6f}")
    print(f"R²决定系数: {r_squared:.6f}")
    print(f"平均绝对误差(MAE): {mae:.6f}")
    print(f"最大误差: {max_error:.6f}")
    print(f"最小误差: {min_error:.6f}")
    print(f"标准差: {np.std(residuals):.6f}")
    
    # 添加一些额外的分析指标
    print(f"\n=== 误差分析 ===")
    print(f"误差小于0.01的比例: {(np.abs(residuals) < 0.01).mean()*100:.2f}%")
    print(f"误差小于0.05的比例: {(np.abs(residuals) < 0.05).mean()*100:.2f}%")
    print(f"误差大于0.1的比例: {(np.abs(residuals) > 0.1).mean()*100:.2f}%")
    
else:
    print("警告: 没有有效的预测结果用于绘图")

print("测试完成!")