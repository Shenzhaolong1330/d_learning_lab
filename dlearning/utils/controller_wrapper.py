import torch
import os
from tensordict import TensorDict
from omni_drones.controllers import DSLPIDController,LeePositionController,AttitudeController

FILE_PATH = os.path.dirname(__file__)


class DSLPIDControllerWrapper(torch.nn.Module):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller

    def forward(self, tensordict: TensorDict) -> TensorDict:
        # 获取当前状态
        # DONE 完善控制器的输入
        #   状态的格式 done
        #   控制目标 在控制器中设为默认 done
        #   控制器状态 done
        # print(f"TensorDict keys before setting action: {tensordict.keys(True, True)}")
        state = tensordict.get(("agents", "observation"))
        # 获取上一时刻控制器状态，如果没有则初始化为 None
        prev_controller_state = tensordict.get("controller_state", None)
        if prev_controller_state is None:
            # 初始化控制器状态，这里需要根据 Controller 的需求调整
            prev_controller_state = TensorDict({}, batch_size=tensordict.batch_size)

        # 调用控制器获取动作和新的控制器状态
        action, new_controller_state = self.controller(state = state, control_target = None, controller_state = prev_controller_state)

        # 更新 TensorDict
        tensordict.set(("agents","action"), action)
        tensordict.set("controller_state", new_controller_state)

        # print(f"TensorDict keys after setting action: {tensordict.keys(True, True)}")

        return tensordict


class ControllerWrapper(torch.nn.Module):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller

    def forward(self, tensordict: TensorDict) -> TensorDict:
        # print(f"TensorDict keys before setting action: {tensordict.keys(True, True)}")
        state = tensordict.get(("agents", "observation"))[...,:13]
        # action = self.controller(root_state = state, target_pos = torch.ones_like(state[..., :3]))
        action = self.controller(root_state = state)

        # forward(
        #     self, 
        #     root_state: torch.Tensor, 
        #     target_pos: torch.Tensor=None,
        #     target_vel: torch.Tensor=None,
        #     target_acc: torch.Tensor=None,
        #     target_yaw: torch.Tensor=None,
        #     body_rate: bool=False
        # ):

        # 更新 TensorDict
        tensordict.set(("agents","action"), action)
        # print(f"TensorDict keys after setting action: {tensordict.keys(True, True)}")

        return tensordict

