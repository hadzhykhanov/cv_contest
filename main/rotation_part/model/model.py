from torch.nn import CrossEntropyLoss, Linear
from torch.nn import Module
from torchvision import models


class ResNet18(Module):
    def __init__(self, num_classes=4):
        super(self.__class__, self).__init__()
        self.model = models.efficientnet_b4(weights=True)

        self.model.classifier[1] = Linear(in_features=1792, out_features=num_classes)
        # self.resnet = models.resnet18(pretrained=True)
        # self.resnet.fc = torch.nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, images, targets=None):
        x = self.model(images)

        if targets is not None:
            loss_func = CrossEntropyLoss()
            loss = loss_func(x, targets)

            return x, loss

        return x, None
