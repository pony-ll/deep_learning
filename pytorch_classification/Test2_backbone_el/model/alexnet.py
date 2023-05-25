import torch.nn as nn
import torch


class AlexNet(nn.Module):

    # 定义网络结构
    def __init__(self, num_classes=1000, init_weights=False):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),  # input[3, 224, 224]  output[48, 55, 55]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[48, 27, 27]
            nn.Conv2d(48, 128, kernel_size=5, padding=2),           # output[128, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 13, 13]
            nn.Conv2d(128, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),          # output[128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 6, 6]
        )

        #定义分类器层
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )
        if init_weights:
            self._initialize_weights()
    # 定义前向传播
    def forward(self, x):
        x = self.features(x)  # 网络的前向传播
        x = torch.flatten(x, start_dim=1)  # 分类个数x
        x = self.classifier(x)  # 分类
        return x

    # 初始化权重
    def _initialize_weights(self):
        """
            这段代码是一个类方法 _initialize_weights 的定义，它用于初始化神经网络模型的权重。

            在这个方法中，通过遍历神经网络的所有模块 self.modules()，对于每个模块 m，进行如下操作：

            如果 m 是一个二维卷积层 (nn.Conv2d)，则使用 Kaiming 正态分布初始化方法 (nn.init.kaiming_normal_) 对该层的权重进行初始化。
            Kaiming 初始化方法根据该层输入和非线性激活函数的特性，初始化权重。同时，如果该层有偏置项 (bias)，则将偏置项初始化为常数0 (nn.init.constant_)。

            如果 m 是一个线性层 (nn.Linear)，则使用正态分布初始化方法 (nn.init.normal_) 对该层的权重进行初始化，均值为0，标准差为0.01。
            同时，将偏置项初始化为常数0 (nn.init.constant_)。

            这段代码的作用是为神经网络的卷积层和线性层初始化权重，以帮助网络在训练过程中更好地学习数据的表示。
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
