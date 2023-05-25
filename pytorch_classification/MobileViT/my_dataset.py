import os

from PIL import Image
import torch
from torch.utils.data import Dataset


class MyDataSet(Dataset):
    """
    这段代码定义了一个自定义数据集类 MyDataSet，用于读取图像数据并进行预处理。

        该数据集类继承自 PyTorch 的 Dataset 类，实现了 __init__、__len__ 和 __getitem__ 方法，以及一个静态方法 collate_fn。其中：

        __init__ 方法接收两个列表参数，分别是训练集/测试集的图像路径和对应的标签列表，以及可选的预处理操作 transform。
        __len__ 方法返回数据集中图像的数量。
        __getitem__ 方法根据索引获取图像数据和标签信息，并返回经过转换后的数据和标签。
        collate_fn 是一个静态方法，用于将多个样本合并成一个 batch。
        在这里，它将图像数据按照 batch 维度进行堆叠，并将标签转换为 tensor 格式。

        这个自定义数据集类可以方便地适配各种类型的图像数据集，同时也可以进行灵活的数据预处理和扩充。
        我们可以通过创建 MyDataSet 对象来读取指定目录下的图像数据，然后将其传递给 DataLoader 进行迭代训练或测试。
    """

    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        # RGB为彩色图片，L为灰度图片
        if img.mode != 'RGB':
            # os.remove(str(self.images_path))
            # print("已删除{}".format(self.images_path))
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels
