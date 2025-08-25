import torch
from tensordict import TensorDict

def make_batch(tensordict: TensorDict, num_minibatches: int):
    tensordict = tensordict.reshape(-1) # 把所有的前两个dimension合并成为batch
    perm = torch.randperm(
        (tensordict.shape[0] // num_minibatches) * num_minibatches,
        device=tensordict.device,
    ).reshape(num_minibatches, -1)
    for indices in perm:
        yield tensordict[indices]

def tensordict_next_hierarchical_control(tensordict: TensorDict):
    pos_control = tensordict["agents", "pos_control_input"]
    atti_control = tensordict["agents", "atti_control_input"]
    
    # 创建目标空间（保持相同形状）
    next_pos = torch.zeros_like(pos_control)
    next_atti = torch.zeros_like(atti_control)
    
    # 核心修改：正确对齐时间步维度
    # 将当前步[1:N]的数据赋值给下一步[0:N-1]
    next_pos[:, :-1] = pos_control[:, 1:]  # 修改切片维度为时间步维度
    next_atti[:, :-1] = atti_control[:, 1:]
    
    # 处理最后一个时间步（复制当前最后一步）
    next_pos[:, -1] = pos_control[:, -1]
    next_atti[:, -1] = atti_control[:, -1]
    
    # 更新tensordict
    tensordict.set(("next", "agents", "pos_control_input"), next_pos)
    tensordict.set(("next", "agents", "atti_control_input"), next_atti)
    
    return tensordict