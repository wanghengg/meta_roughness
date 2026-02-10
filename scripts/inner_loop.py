import torch
import torch.nn.functional as F


def inner_update(model, support_x, support_y, lr):
    """
    MAML 内层更新：根据支持集计算梯度并更新参数
    
    Args:
        model: 模型实例
        support_x: 支持集特征
        support_y: 支持集标签
        lr: 学习率
    
    Returns:
        updated: 更新后的参数字典
    """
    params = {n: p for n, p in model.named_parameters()}

    preds = model(support_x, params)
    loss = F.mse_loss(preds, support_y)

    # 将 params.values() 转换为列表以匹配 grad 函数的要求
    param_tensors = list(params.values())
    grads = torch.autograd.grad(loss, param_tensors, create_graph=True)

    updated = {
        name: param - lr * grad
        for (name, param), grad in zip(params.items(), grads)
    }
    return updated