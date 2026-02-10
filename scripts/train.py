import sys
import os
import torch
import matplotlib.pyplot as plt

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from models.maml_model import RoughnessNet
from utils.task_sampler import load_dataset, build_tasks, compute_global_stats
from config import *
from scripts.inner_loop import inner_update

df = load_dataset("data/scicp_al_sample.csv")

# 计算全局统计信息用于标准化
global_mean, global_std = compute_global_stats(df)
print(f"全局统计信息 - 均值: {global_mean}, 标准差: {global_std}")

model = RoughnessNet(INPUT_DIM)
optimizer = torch.optim.Adam(model.parameters(), lr=META_LR)

# 记录训练loss
loss_history = []

for epoch in range(EPOCHS):
    tasks = build_tasks(df, TASKS_PER_EPOCH, SUPPORT_K, QUERY_K, global_mean, global_std)
    meta_loss = torch.tensor(0.0, requires_grad=True)

    for support_x, support_y, query_x, query_y in tasks:
        adapted_params = inner_update(model, support_x, support_y, INNER_LR)
        query_preds = model(query_x, adapted_params)
        loss_q = torch.nn.functional.mse_loss(query_preds, query_y)
        meta_loss = meta_loss + loss_q

    meta_loss = meta_loss / TASKS_PER_EPOCH

    optimizer.zero_grad()
    meta_loss.backward()
    optimizer.step()

    # 记录当前epoch的loss
    loss_history.append(meta_loss.item())

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Meta Loss: {meta_loss.item():.6f}")
        # 每50个epoch保存一次模型
        torch.save(model.state_dict(), 'maml_model.pth')
        print(f"模型已保存到 maml_model.pth")

# 训练结束后保存最终模型
torch.save(model.state_dict(), 'maml_model.pth')
print(f"训练完成，最终模型已保存到 maml_model.pth")

# 绘制训练loss曲线
plt.figure(figsize=(10, 6))
plt.plot(range(EPOCHS), loss_history, label='Meta Loss')
plt.title('Training Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Meta Loss (MSE)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# 保存图像
plt.savefig('training_loss_curve.png', dpi=300, bbox_inches='tight')
print(f"训练loss曲线已保存到 training_loss_curve.png")

# 显示图像（可选，如果在非交互式环境中运行可能需要注释掉）
# plt.show()