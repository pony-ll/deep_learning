# 可视化可行方案一
from torchstat import stat
from model import LeNet

model = LeNet()
# stat(model, (3, 32, 32))


# 可视化可行方案二
from torchsummary import summary

summary(model=model, input_size=(3, 32, 32), batch_size=-1, device="cpu")


