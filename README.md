# 金属基复合材料表面粗糙度预测

基于Model-Agnostic Meta-Learning (MAML)的金属基复合材料表面粗糙度预测系统，专为数控加工场景设计的小样本学习解决方案。

## 🎯 项目简介

本项目利用元学习技术，通过MAML算法实现对SiC颗粒增强铝基复合材料在数控加工过程中表面粗糙度(Ra)的精确预测。系统能够在仅有少量样本的情况下快速适应新的加工条件，为智能制造提供可靠的质量预测能力。

## 🚀 技术特点

- **元学习框架**: 采用MAML算法，具备快速适应新任务的能力
- **小样本学习**: 仅需10个支持样本即可实现良好预测性能
- **高精度预测**: RMSE达到0.0497μm，满足工程应用精度要求
- **多任务适应**: 支持不同加工条件下的泛化预测

## 📊 性能指标

| 指标 | 数值 | 说明 |
|------|------|------|
| 整体RMSE | 0.0497 μm | 均方根误差 |
| 整体MSE | 0.002469 | 均方误差 |
| MAE | 0.0408 μm | 平均绝对误差 |
| 误差<0.05μm比例 | 64% | 高精度预测占比 |

## 🛠️ 技术栈

- **深度学习框架**: PyTorch 2.2.2
- **编程语言**: Python 3.11
- **核心算法**: Model-Agnostic Meta-Learning (MAML)
- **网络结构**: 多层感知机 (MLP)
- **优化器**: Adam
- **数据处理**: Pandas, NumPy
- **可视化**: Matplotlib, Seaborn

## 📁 项目结构

```
meta_roughness/
├── data/                    # 数据文件目录
│   └── raw_data.csv        # 原始加工数据
├── models/                  # 模型定义文件
│   └── maml_model.py       # MAML模型实现
├── utils/                   # 工具函数
│   └── task_sampler.py     # 任务采样器
├── scripts/                 # 执行脚本目录
│   ├── train.py            # 模型训练脚本
│   ├── test.py             # 模型测试脚本
│   ├── analyze_data.py     # 数据分析脚本
│   ├── generate_maml_diagrams.py # MAML原理图生成脚本
│   └── inner_loop.py       # 内循环更新逻辑
├── outputs/                 # 输出文件目录
│   ├── charts/             # 可视化图表
│   │   ├── prediction_analysis.png     # 预测分析图
│   │   ├── correlation_matrix.png      # 相关性矩阵图
│   │   ├── feature_distributions.png   # 特征分布图
│   │   ├── ra_distribution.png         # Ra分布图
│   │   ├── training_loss_curve.png     # 训练损失曲线
│   │   ├── maml_framework.png          # MAML框架图
│   │   ├── maml_gradients.png          # MAML梯度图
│   │   ├── maml_application.png        # MAML应用场景图
│   │   └── math_symbols_test.png       # 数学符号测试图
│   └── models/             # 模型文件
│       └── maml_model.pth  # 训练好的模型权重
├── docs/                    # 文档目录
│   └── 金属基复合材料表面粗糙度预测研究报告.md  # 技术研究报告
├── config.py               # 配置参数文件
├── README.md               # 项目说明文档
└── .gitignore              # Git忽略文件配置
```

## 🚀 快速开始

### 环境准备

```bash
# 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# 或 .venv\Scripts\activate  # Windows

# 安装依赖
pip install torch==2.2.2 pandas numpy matplotlib seaborn scikit-learn
```

### 数据准备

将您的加工数据保存为 `data/raw_data.csv`，包含以下列：
- `SiCp_fraction`: SiC颗粒体积分数 (%)
- `spindle_speed`: 主轴转速 (rpm)
- `feed_rate`: 进给速度 (mm/rev)
- `depth_of_cut`: 切削深度 (mm)
- `tool_wear`: 刀具磨损 (mm)
- `vibration_rms`: 振动RMS值 (mm/s)
- `Ra`: 表面粗糙度目标值 (μm)

### 模型训练

```bash
python scripts/train.py
```

### 模型测试

```bash
python scripts/test.py
```

### 数据分析

```bash
python scripts/analyze_data.py
```

## 📈 可视化分析

项目提供多种可视化功能：

- **预测性能分析**: 预测值vs真实值散点图、残差分析
- **数据特征分析**: 相关性矩阵、特征分布图
- **MAML原理图**: 框架结构、梯度计算流程、应用场景

```bash
python scripts/generate_maml_diagrams.py
```

## 📖 技术文档

详细的项目技术文档请参考：[docs/金属基复合材料表面粗糙度预测研究报告.md](docs/金属基复合材料表面粗糙度预测研究报告.md)

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进项目：

1. Fork本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 📄 许可证

本项目采用MIT许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- 提交GitHub Issue
- 发送邮件至项目维护者

## 🙏 致谢

- 感谢Chelsea Finn等人的MAML算法研究
- 感谢PyTorch开源社区的支持
- 感谢所有为项目贡献的开发者