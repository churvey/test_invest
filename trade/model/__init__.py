import torch
import torch.nn as nn

class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
       
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(self, y_pred, y_true):
      
        error = y_true - y_pred
        abs_error = torch.abs(error)
        
        # 计算二次损失和线性损失的权重
        quadratic = torch.clamp(abs_error, max=self.delta)
        linear = (abs_error - quadratic)
        
        # 组合损失
        loss = 0.5 * quadratic ** 2 + self.delta * linear
        return torch.mean(loss)  # 返回批次平均损失