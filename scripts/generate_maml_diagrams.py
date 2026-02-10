#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MAML原理图生成脚本
用于生成MAML元学习框架的原理示意图
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

# 设置matplotlib支持中文显示和数学符号
plt.rcParams['font.sans-serif'] = [
    'STIXGeneral',          # STIX字体，专门支持数学符号
    'DejaVu Sans',          # DejaVu字体，良好的Unicode支持
    'PingFang SC',          # 苹方简体
    'Heiti SC',             # 黑体简体
    'Arial Unicode MS',     # Arial Unicode
]
plt.rcParams['mathtext.fontset'] = 'stix'  # 使用STIX数学字体集
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

def generate_maml_framework_diagram():
    """生成MAML框架原理图"""
    print("正在生成MAML框架原理图...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 第一个子图：传统学习 vs 元学习
    ax1 = axes[0]
    ax1.text(0.1, 0.8, '传统机器学习', fontsize=14, fontweight='bold')
    ax1.text(0.1, 0.6, '• 大量数据训练', fontsize=12)
    ax1.text(0.1, 0.5, '• 特定任务优化', fontsize=12)
    ax1.text(0.1, 0.4, '• 泛化能力有限', fontsize=12)
    ax1.text(0.1, 0.2, '• 新任务需重训练', fontsize=12)
    
    ax1.text(0.1, -0.2, '元学习(MAML)', fontsize=14, fontweight='bold', color='blue')
    ax1.text(0.1, -0.4, '• 学习初始化参数', fontsize=12, color='blue')
    ax1.text(0.1, -0.5, '• 快速适应新任务', fontsize=12, color='blue')
    ax1.text(0.1, -0.6, '• 少样本学习能力', fontsize=12, color='blue')
    ax1.text(0.1, -0.8, '• 通用学习策略', fontsize=12, color='blue')
    
    ax1.set_xlim(0, 1)
    ax1.set_ylim(-1, 1)
    ax1.axis('off')
    
    # 第二个子图：MAML双循环结构
    ax2 = axes[1]
    ax2.text(0.5, 0.9, 'MAML双循环优化', fontsize=14, fontweight='bold', ha='center')
    
    # 外循环
    ax2.add_patch(patches.Rectangle((0.1, 0.6), 0.8, 0.25, fill=False, edgecolor='blue', linewidth=2))
    ax2.text(0.5, 0.75, '外循环: 元参数优化', fontsize=12, ha='center', color='blue')
    
    # 内循环
    ax2.add_patch(patches.Rectangle((0.2, 0.3), 0.6, 0.2, fill=False, edgecolor='green', linewidth=2))
    ax2.text(0.5, 0.42, '内循环: 任务适应', fontsize=11, ha='center', color='green')
    
    # 连接箭头
    ax2.annotate('', xy=(0.5, 0.6), xytext=(0.5, 0.5), arrowprops=dict(arrowstyle='->', color='red'))
    
    ax2.text(0.1, 0.1, 'θ → θ\' → Loss', fontsize=12, ha='left')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    
    # 第三个子图：梯度流示意
    ax3 = axes[2]
    ax3.text(0.5, 0.9, '梯度计算流程', fontsize=14, fontweight='bold', ha='center')
    
    # 参数节点
    ax3.plot([0.2, 0.5, 0.8], [0.7, 0.7, 0.7], 'bo-', markersize=8)
    ax3.text(0.2, 0.6, 'θ', ha='center', fontsize=12)
    ax3.text(0.5, 0.6, 'θ\'', ha='center', fontsize=12)
    ax3.text(0.8, 0.6, 'Loss', ha='center', fontsize=12)
    
    # 梯度箭头
    ax3.annotate('', xy=(0.35, 0.7), xytext=(0.25, 0.7), arrowprops=dict(arrowstyle='<-', color='red'))
    ax3.annotate('', xy=(0.65, 0.7), xytext=(0.55, 0.7), arrowprops=dict(arrowstyle='<-', color='red'))
    
    ax3.text(0.3, 0.8, r'$\nabla L_{support}$', ha='center', fontsize=10, color='red')
    ax3.text(0.6, 0.8, r'$\nabla L_{query}$', ha='center', fontsize=10, color='red')
    
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    
    plt.tight_layout()
    plt.savefig('maml_framework.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ MAML框架原理图已生成: maml_framework.png")

def test_math_symbols():
    """测试数学符号显示"""
    print("正在测试数学符号显示...")
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # 测试各种数学符号
    symbols = [
        r'$\nabla$ (nabla)',           # ∇
        r'$\mathcal{L}$ (script L)',   # ℒ
        r'$\partial$ (partial)',       # ∂
        r'$\alpha$ (alpha)',           # α
        r'$\beta$ (beta)',             # β
        r'$\theta$ (theta)',           # θ
        r'$\lambda$ (lambda)',         # λ
        r'$\infty$ (infinity)',        # ∞
    ]
    
    ax.text(0.5, 0.9, '数学符号测试', fontsize=16, fontweight='bold', ha='center')
    
    for i, symbol in enumerate(symbols):
        y_pos = 0.8 - i * 0.1
        ax.text(0.3, y_pos, symbol, fontsize=14, ha='center')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('math_symbols_test.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 数学符号测试图已生成: math_symbols_test.png")

def generate_maml_gradients_diagram():
    """生成MAML梯度计算示意图"""
    print("正在生成MAML梯度计算示意图...")
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # 绘制计算图标题
    ax.text(0.5, 0.9, 'MAML梯度计算示意图', fontsize=16, fontweight='bold', ha='center')
    
    # 初始参数
    ax.plot(0.2, 0.8, 'bo', markersize=12, label=r'初始参数 $\theta$')
    ax.text(0.2, 0.75, r'$\theta$', ha='center', fontsize=14)
    
    # 支持集前向传播
    ax.plot(0.2, 0.6, 'go', markersize=10)
    ax.text(0.2, 0.55, 'Support Forward', ha='center', fontsize=10)
    
    # 损失计算
    ax.plot(0.2, 0.4, 'ro', markersize=10)
    ax.text(0.2, 0.35, r'$\mathcal{L}_{support}$', ha='center', fontsize=12)
    
    # 梯度计算
    ax.plot(0.4, 0.4, 'mo', markersize=10)
    ax.text(0.4, 0.35, r'$\nabla \mathcal{L}_{support}$', ha='center', fontsize=12)
    
    # 参数更新
    ax.plot(0.6, 0.6, 'co', markersize=12)
    ax.text(0.6, 0.55, r"$\theta' = \theta - \alpha \nabla \mathcal{L}$", ha='center', fontsize=14)
    
    # 查询集前向传播
    ax.plot(0.6, 0.4, 'go', markersize=10)
    ax.text(0.6, 0.35, 'Query Forward', ha='center', fontsize=10)
    
    # 最终损失
    ax.plot(0.6, 0.2, 'ro', markersize=10)
    ax.text(0.6, 0.15, r'$\mathcal{L}_{query}$', ha='center', fontsize=12)
    
    # 外循环梯度
    ax.plot(0.8, 0.2, 'ko', markersize=12)
    ax.text(0.8, 0.15, r'$\nabla \mathcal{L}_{meta}$', ha='center', fontsize=14)
    
    # 连接箭头
    ax.annotate('', xy=(0.2, 0.7), xytext=(0.2, 0.65), arrowprops=dict(arrowstyle='->'))
    ax.annotate('', xy=(0.2, 0.5), xytext=(0.2, 0.45), arrowprops=dict(arrowstyle='->'))
    ax.annotate('', xy=(0.3, 0.4), xytext=(0.25, 0.4), arrowprops=dict(arrowstyle='->'))
    ax.annotate('', xy=(0.5, 0.6), xytext=(0.45, 0.4), arrowprops=dict(arrowstyle='->'))
    ax.annotate('', xy=(0.6, 0.5), xytext=(0.6, 0.45), arrowprops=dict(arrowstyle='->'))
    ax.annotate('', xy=(0.6, 0.3), xytext=(0.6, 0.25), arrowprops=dict(arrowstyle='->'))
    ax.annotate('', xy=(0.75, 0.2), xytext=(0.65, 0.2), arrowprops=dict(arrowstyle='->'))
    
    # 二阶梯度标注
    ax.text(0.4, 0.6, '二阶梯度计算', fontsize=12, color='red', fontweight='bold', 
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('maml_gradients.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ MAML梯度计算示意图已生成: maml_gradients.png")

def generate_maml_application_diagram():
    """生成MAML在加工领域的应用场景图"""
    print("正在生成MAML应用场景图...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('MAML在金属加工领域的应用场景', fontsize=16, fontweight='bold')
    
    # 场景1: 不同材料适配
    ax1 = axes[0, 0]
    ax1.set_title('材料类型适配')
    materials = ['铝合金', '钛合金', '不锈钢', '镁合金']
    adaptation_speed = [0.85, 0.78, 0.92, 0.81]
    bars1 = ax1.bar(materials, adaptation_speed, color=['skyblue', 'lightgreen', 'salmon', 'gold'])
    ax1.set_ylabel('适应速度指数')
    ax1.set_ylim(0, 1)
    for bar, speed in zip(bars1, adaptation_speed):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                 f'{speed:.2f}', ha='center', va='bottom')
    
    # 场景2: 工艺参数优化
    ax2 = axes[0, 1]
    ax2.set_title('工艺参数寻优')
    params = ['切削速度', '进给量', '切削深度']
    improvement = [[0.75, 0.82, 0.68], [0.81, 0.76, 0.85], [0.69, 0.88, 0.79]]
    im = ax2.imshow(improvement, cmap='YlOrRd', aspect='auto')
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(params, rotation=45)
    ax2.set_yticks(range(3))
    ax2.set_yticklabels(['任务1', '任务2', '任务3'])
    plt.colorbar(im, ax=ax2, label='优化效果')
    
    # 场景3: 设备状态监测
    ax3 = axes[1, 0]
    ax3.set_title('设备状态适应')
    states = ['新刀具', '正常使用', '轻微磨损', '严重磨损']
    accuracy = [0.94, 0.89, 0.85, 0.78]
    bars3 = ax3.bar(states, accuracy, color=['green', 'blue', 'orange', 'red'])
    ax3.set_ylabel('预测准确率')
    ax3.set_ylim(0.7, 1.0)
    for bar, acc in zip(bars3, accuracy):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                 f'{acc:.2f}', ha='center', va='bottom')
    
    # 场景4: 批次间一致性
    ax4 = axes[1, 1]
    ax4.set_title('批次一致性保持')
    batches = ['批次A', '批次B', '批次C', '批次D', '批次E']
    consistency = [0.88, 0.91, 0.86, 0.89, 0.92]
    line4 = ax4.plot(batches, consistency, 'bo-', linewidth=2, markersize=8)
    ax4.fill_between(batches, consistency, alpha=0.3)
    ax4.set_ylabel('一致性指数')
    ax4.set_ylim(0.8, 1.0)
    for i, cons in enumerate(consistency):
        ax4.text(i, cons + 0.01, f'{cons:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('maml_application.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ MAML应用场景图已生成: maml_application.png")

def main():
    """主函数"""
    print("=" * 50)
    print("MAML原理图生成工具 - 符号测试版")
    print("=" * 50)
    
    # 检查当前目录
    current_dir = os.getcwd()
    print(f"当前工作目录: {current_dir}")
    
    try:
        # 测试数学符号显示
        test_math_symbols()
        
        # 生成带符号的框架图
        generate_maml_framework_diagram()
        
        # 生成其他图表
        generate_maml_gradients_diagram()
        generate_maml_application_diagram()
        
        print("\n" + "=" * 50)
        print("所有MAML原理图生成完成！")
        print("=" * 50)
        print("生成的文件:")
        print("- math_symbols_test.png: 数学符号兼容性测试")
        print("- maml_framework_test_symbols.png: 带数学符号的框架图")
        print("- maml_gradients.png: MAML梯度计算示意图")
        print("- maml_application.png: MAML应用场景图")
        
    except Exception as e:
        print(f"生成图表时出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()