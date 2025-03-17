import numpy as np
import torch
import trade_cpp

# 创建测试数据
data = {
    "features": np.random.rand(1000, 256).astype(np.float32),
    "labels": np.random.rand(1000, 10).astype(np.float32)
}

# 初始化采样器
sampler = trade_cpp.NumpyDictSampler(data, 64)

# iterator = iter(sampler)

# # 迭代获取批次
for batch in sampler:
    print(batch["features"].device)  # 应显示为CPU
    print(batch["features"].is_pinned())  # 应返回True
    print(batch["features"].to("cuda"))  # 应返回True
    break

# 66 ~73