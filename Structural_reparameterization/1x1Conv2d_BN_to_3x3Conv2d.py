import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np

# 1x1 Conv2d + BN -> 3x3 Conv2d
def main():
    torch.manual_seed(42)

    tensor_test = torch.rand(1, 2, 3, 3)

    Conv_Bn = nn.Sequential(OrderedDict(conv=nn.Conv2d(in_channels=2, out_channels=2, kernel_size=1, stride=1, bias=False),
                                        bn=nn.BatchNorm2d(num_features=2)))

    Conv_Bn.eval()
    with torch.no_grad():
        output1 = Conv_Bn(tensor_test)
        print(output1)

    conv_weight = Conv_Bn.conv.weight
    running_mean = Conv_Bn.bn.running_mean
    running_var = Conv_Bn.bn.running_var
    gamma = Conv_Bn.bn.weight
    beta = Conv_Bn.bn.bias
    eps = Conv_Bn.bn.eps
    std = torch.sqrt(running_var + eps)
    # 1x1 Conv2d -> 3x3 Conv2d
    conv_weight_fuse = nn.ZeroPad2d(1)(conv_weight)
    conv_weight_fuse = conv_weight_fuse * (gamma / std).reshape(-1, 1, 1, 1)
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
