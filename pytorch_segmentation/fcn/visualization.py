# from torchsummary import summary
# import torch
# from src import fcn_resnet50
#
# device = torch.device('cpu')
# model = fcn_resnet50(aux=False, num_classes=21, pretrain_backbone=False).to(device)
#
# # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # input_tensor = torch.randn(1, 3, 224, 224).to(device)
#
#
# summary(model, input_size=(3, 224, 224))
