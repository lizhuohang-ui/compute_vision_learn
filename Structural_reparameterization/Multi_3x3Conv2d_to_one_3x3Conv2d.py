import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np

def main():
    torch.random.manual_seed(42)

    tensor_test = torch.randn(1, 2, 3, 3)

    class Multi_Conv2d(nn.Module):
        def __init__(self, c1, c2, k, s, p, bias):
            super(Multi_Conv2d, self).__init__()
            self.conv1 = nn.Conv2d(c1, c2, k, s, p, bias=bias)
            self.conv2 = nn.Conv2d(c1, c2, k, s, p, bias=bias)
            self.conv3 = nn.Conv2d(c1, c2, k, s, p, bias=bias)

        def forward(self, x):
            x1 = self.conv1(x)
            x2 = self.conv2(x)
            x3 = self.conv3(x)
            return x1 + x2 + x3

    Multi_Conv = Multi_Conv2d(2, 2, 3, 1, 1, bias=True)

    with torch.no_grad():
        output1 = Multi_Conv(tensor_test)
        print(output1)

    conv1_wight = Multi_Conv.conv1.weight
    conv2_wight = Multi_Conv.conv2.weight
    conv3_wight = Multi_Conv.conv3.weight

    conv1_bias = Multi_Conv.conv1.bias
    conv2_bias = Multi_Conv.conv2.bias
    conv3_bias = Multi_Conv.conv3.bias

    conv_wight_fuse = conv1_wight + conv2_wight + conv3_wight
    conv_bias_fuse = conv1_bias + conv2_bias + conv3_bias

    Conv_fuse = nn.Conv2d(2, 2, 3, 1, 1, bias=True)
    Conv_fuse.load_state_dict(OrderedDict(weight=conv_wight_fuse, bias=conv_bias_fuse))

    with torch.no_grad():
        output2 = Conv_fuse(tensor_test)
        print(output2)

    print("output1和output2对应数值的最大误差为:", (output1 - output2).max().numpy())
    # np.testing.assert_allclose 用于确定精度
    np.testing.assert_allclose(output1.numpy(), output2.numpy(), rtol=1e-3, atol=1e-5)


if __name__ == "__main__":
    main()
