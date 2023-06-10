import torch
from torch.nn import CrossEntropyLoss
from torch.nn import Module
from torchvision import models


class ResNet18(Module):
    def __init__(self, num_classes=2):
        super(self.__class__, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = torch.nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, images, targets=None):
        print(type(targets))
        x = self.resnet(images)

        if targets is not None:
            loss_func = CrossEntropyLoss()
            loss = loss_func(x, targets.view(-1))

            return x, loss

        return x, None
