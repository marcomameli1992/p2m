import torchvision
import torch
import torch.nn as nn
import torchvision


class VGG(nn.Module):
    '''
        Pretrained VGG for image feature extraction
    '''

    def __init__(self) -> None:
        super(VGG, self).__init__()

        model_conv = torchvision.models.vgg16(pretrained=True).features

        self.layers = list(model_conv.children())
        self.conv3 = nn.Sequential(*self.layers[:-14])
        self.conv4 = nn.Sequential(*self.layers[:-7])
        self.conv5 = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.conv3(x), self.conv4(x), self.conv5(x)

