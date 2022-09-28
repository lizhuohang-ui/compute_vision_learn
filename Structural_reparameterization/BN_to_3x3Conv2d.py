import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np


# BN -> 3x3 Conv2d
def main():
    torch.random.manual_seed(42)

    tensor_test = torch.randn(1, 2, 3, 3)

    BN = nn.BatchNorm2d(num_features=2)

    BN.eval()

    with torch.no_grad():
        output1 = BN(tensor_test)
        print(output1)

    kernel1_1 = nn.ZeroPad2d(1)(torch.ones(1, 1, 1, 1))
    kernel1_2 = torch.zeros(1, 1, 3, 3)
    kernel2_1 = torch.zeros(1, 1, 3, 3)
    kernel2_2 = nn.ZeroPad2d(1)(torch.ones(1, 1, 1, 1))

    kernel1 = torch.cat([kernel1_1, kernel1_2], 1)
    kernel2 = torch.cat([kernel2_1, kernel2_2], 1)

    kernel = torch.cat([kernel1, kernel2], 0)

    running_mean = BN.running_mean
    running_var = BN.running_var
    gamma = BN.weight
    beta = BN.bias
    eps = BN.eps
    std = torch.sqrt(running_var + eps)
    conv_weight_fuse = kernel * (gamma / std).reshape(-1, 1, 1, 1)
    bias_fuse = beta - running_mean * gamma / std

    Conv_fuse = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3, stride=1, padding=1, bias=True)
    Conv_fuse.load_state_dict(OrderedDict(weight=conv_weight_fuse, bias=bias_fuse))

    with torch.no_grad():
        output2 = Conv_fuse(tensor_test)
        print(output2)

    print("output1和output2对应数值的最大误差为:", (output1 - output2).max().numpy())
    # np.testing.assert_allclose 用于确定精度
    np.testing.assert_allclose(output1.numpy(), output2.numpy(), rtol=1e-3, atol=1e-5)


if __name__ == "__main__":
    main()

