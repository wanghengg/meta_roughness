import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any

# 设置matplotlib支持中文显示 - 根据macOS系统优化
plt.rcParams['font.sans-serif'] = [
    'PingFang SC',      # 苹方简体（macOS默认中文字体，显示效果最佳）
    'Heiti SC',         # 黑体简体
    'Hiragino Sans GB', # 冬青黑体简体
    'STHeiti',          # 华文黑体
    'Microsoft YaHei',  # 微软雅黑
    'SimHei',           # 黑体
    'Arial Unicode MS', # Arial Unicode
    'DejaVu Sans'       # DejaVu字体
]
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
plt.rcParams['font.size'] = 12              # 设置默认字体大小
plt.rcParams['axes.titlesize'] = 14         # 设置标题字体大小
plt.rcParams['axes.labelsize'] = 12         # 设置坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 10        # 设置x轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 10        # 设置y轴刻度字体大小

# 加载数据
df = pd.read_csv('data/scicp_al_sample.csv')

# 基本统计信息
print("数据基本信息：")
print(df.info())
print("\n数据统计描述：")
print(df.describe())

# 检查缺失值
print("\n缺失值情况：")
print(df.isnull().sum())

# 绘制特征与标签的相关性
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('特征相关性矩阵')
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')

# 绘制各特征的分布
plt.figure(figsize=(15, 10))
bins_value: Any = 30  # 使用Any类型避免类型检查错误
for i, col in enumerate(df.columns):
    plt.subplot(3, 3, i+1)
    sns.histplot(df[col], bins=bins_value, kde=True)  # type: ignore
    plt.title(f'{col} 分布')
plt.tight_layout()
plt.savefig('feature_distributions.png', dpi=300, bbox_inches='tight')

# 绘制标签的分布
plt.figure(figsize=(8, 5))
sns.histplot(df['Ra'], bins=bins_value, kde=True)  # type: ignore
plt.title('Ra (粗糙度) 分布')
plt.xlabel('Ra 值')
plt.ylabel('频率')
plt.tight_layout()
plt.savefig('ra_distribution.png', dpi=300, bbox_inches='tight')

print("\n数据分布分析完成，图表已保存。")