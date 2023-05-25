import torch.nn as nn
import torch


class BasicBlock(nn.Module):  # 18层和34层所使用的残差结构
    expansion = 1  # 表示输入通道与输出通道相同，没有扩展

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        """
        定义残差模块
        标准3x3卷积
        BN
        relu
        标准3x3卷积
        BN
        ds
        """
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        # 使用identity算是记录本层的输入
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)  # 如果下采样不为None则对x进行下采样

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity  # F(X)+X  残差连接
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):  # 50层、101层和152层网络所使用的残差结构
    """
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch

    三层结构：1. 减少输入的通道数
            2. 窄而深的卷积层
            3. 用于增加通道数

    首先定义了一个Bottleneck类，继承自nn.Module。该类有一个类变量expansion=4，
    表示瓶颈块中第二个卷积层输出通道数是第一个卷积层输出通道数的4倍，即进行了通道扩展。
    __init__方法用于初始化瓶颈块的参数。
    其中，in_channel和out_channel分别表示输入特征图和输出特征图的通道数；stride表示第二个卷积层的步距；downsample用于降采样，
    将输入特征图的大小减半。
    在__init__方法中，首先根据out_channel和width_per_group计算出每个分组的通道数width。
    然后定义了三个卷积层conv1、conv2和conv3，分别对应瓶颈块的三个结构：减少输入的通道数、窄而深的卷积层和用于增加通道数。
    这三个卷积层都用nn.Conv2d实现，其中第二个卷积层中设置了分组卷积的参数groups，可以有效降低计算复杂度。
    在__init__方法中，还定义了三个批归一化层bn1、bn2和bn3，用于归一化每个卷积层的输出特征图；
    定义了一个ReLU激活函数，并将downsample赋值给self.downsample。
    forward方法表示前向传播的过程。首先将输入x保存在变量identity中，如果存在downsample则将x通过downsample进行降采样后再将结果保存在identity中。
    然后对x依次进行卷积、批归一化和ReLU激活，得到out1、out2和out3。最后将out3与identity相加后再通过ReLU激活函数得到输出结果out，返回out。

    """
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):  # groups和width_per_group用于分组卷积，从而降低计算复杂度
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel * self.expansion,  # 第三层的通道数扩大四倍
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x  # 记录当前层的输入值
        if self.downsample is not None:
            identity = self.downsample(x)  # 如果下采样不为None则对x进行下采样

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity  # 残差连接
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, blocks_num, num_classes=1000, include_top=True, groups=1, width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.groups = groups
        self.width_per_group = width_per_group

        # 定义第一层
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 使用残差块构建网络
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 521, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 模型初始化
        for m in self.modeles():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    # 构建残差块网络
    def _make_layer(self, block, channel, block_num, stride=1):
        downasmple = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downasmple = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(
            block(self.in_channel, channel, downasmple=downasmple, stride=stride,
                  groups=self.groups, with_per_group=self.width_per_group)
        )

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                with_per_group=self.width_per_group
                                ))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)


def resnet34(num_classes=1000, include_top=True):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet50(num_classes=1000, include_top=True):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)
