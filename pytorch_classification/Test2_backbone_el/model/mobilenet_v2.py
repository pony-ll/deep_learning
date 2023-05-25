from torch import nn
import torch


def _make_divisible(ch, divisor=8, min_ch=None):
    """
    作用：将通道数调整为输入通道数的整数倍
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


class ConvBNReLU(nn.Sequential):
    """
    这段代码定义了一个继承自 nn.Sequential 的类 ConvBNReLU，用于构建一个卷积层、批归一化层和 ReLU6 激活函数层的序列模块。

    具体实现过程如下：

    1. 接收输入通道数 in_channel、输出通道数 out_channel、卷积核大小 kernel_size、步长 stride 和分组数 groups 等参数。
    2. 计算填充量 padding，采用 "大小不变" 的填充方式，即对于奇数大小的卷积核，向两边各填充 (kernel_size - 1) // 2 个像素，
    对于偶数大小的卷积核，则填充 (kernel_size // 2, kernel_size // 2 - 1) 个像素。
    3. 调用父类 nn.Sequential 的构造函数，传入以下三个子模块：
    一个卷积层 nn.Conv2d，输入通道数为 in_channel，输出通道数为 out_channel，卷积核大小为 kernel_size，步长为 stride，
    填充量为 padding，分组数为 groups，且不包含偏置项（bias=False）。
    一个批归一化层 nn.BatchNorm2d，输入通道数为 out_channel。
    一个使用 ReLU6 激活函数的层 nn.ReLU6，将其 inplace 参数设置为 True，表示直接在原张量上修改，避免额外的内存开销。
    4. 返回构造好的序列模块。
    """
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2  # 大小不变填充
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    """
    这段代码定义了一个名为 InvertedResidual 的类，该类继承了 nn.Module 类，用于构建深度神经网络的组成部分。
    InvertedResidual 类的初始化函数 __init__() 接收四个参数：in_channel，输入张量的通道数；out_channel，输出张量的通道数；
    stride，深度可分离卷积层的步幅；expand_ratio，通道数扩展比例，是一个介于 0 和 1 之间的浮点数。

    该类主要的操作在 __init__() 函数中完成，首先根据 expand_ratio 扩展通道数，然后根据 stride 来决定是否需要添加一个恒等变换，
    接下来将恒等变换和深度可分离卷积操作组合成一个序列，最后赋值给 self.conv。

    在类的 forward() 函数中，如果需要使用恒等变换，就将输入张量和经过 self.conv 序列操作后的张量相加，否则仅返回 self.conv 操作后的张量。
    """
    def __init__(self, in_channel, out_channel, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        hidden_channel = in_channel * expand_ratio
        self.use_shortcut = stride == 1 and in_channel == out_channel

        layers = []
        if expand_ratio != 1:
            # 1x1 pointwise conv
            layers.append(ConvBNReLU(in_channel, hidden_channel, kernel_size=1))
        layers.extend([
            # 3x3 depthwise conv
            ConvBNReLU(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel),
            # 1x1 pointwise conv(linear)
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_shortcut:  # 如果满足条件，则使用shortcut连接
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    """
    这段代码是实现了一个基于 MobileNetV2 的卷积神经网络模型，用于分类任务。
    其中，InvertedResidual 是 MobileNetV2 的基本组成单元。
    MobileNetV2 中的 __init__ 方法初始化了网络的各个组件，包括特征提取层 features、
    全局平均池化层 avgpool 和分类层 classifier，并对这些组件进行了合适的初始化操作。
    features 由一系列的卷积层、BN层和ReLU激活函数组成，其中的 InvertedResidual 模块由 block 指定。
    通过循环生成一系列的 InvertedResidual 模块，来构建网络的特征提取层。
    最后，通过平均池化和全连接层得到网络的分类结果。
    """
    def __init__(self, num_classes=1000, alpha=1.0, round_nearest=8):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = _make_divisible(32 * alpha, round_nearest)  # 把通道数调整为输入通道的整数倍，为了更好地调用设备
        last_channel = _make_divisible(1280 * alpha, round_nearest)

        inverted_residual_setting = [
            # t, c, n, s     t:扩展因子  c：通道数  n：倒残差结构重复的次数  s：每一个块的第一层的步距
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        features = []
        # conv1 layer
        features.append(ConvBNReLU(3, input_channel, stride=2))
        # 构建倒残差块
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * alpha, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, last_channel, 1))
        # combine feature layers  合并特征层
        self.features = nn.Sequential(*features)

        # building classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes)
        )

        # 权重初始化 遍历所有子模块，如果是卷积的话，就对权重进行初始化；如果存在偏置的话，就把偏置设置为零；
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):  # 如果子模块是BN层的话，就把方差设置为1，均值设置为0
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):  # 如果是全连接层的话，权重初始化均值为0，方差为0.01，偏置为0
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
