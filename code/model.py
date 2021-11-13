import torch 
import torch.nn as nn
import pdb
from torchvision.models.resnet import ResNet, BasicBlock
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self, num_channels, num_layers):
        super(ConvNet, self).__init__()
        self.num_channels = num_channels
        self.num_layers = num_layers
        in_channels = [self.num_channels, 16, 32]
        out_channels = [16, 32, 64]
        conv_modules = []
        for idx in range(self.num_layers):
            conv_modules.append(nn.Sequential(
                nn.Conv2d(in_channels[idx], out_channels[idx], kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
                #nn.BatchNorm2d(out_channels[idx]),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=2, stride=2)))
        self.encoder = nn.Sequential(*conv_modules)
        self.fc = torch.nn.Linear(out_channels[self.num_layers-1], out_channels[self.num_layers-1], bias=True) 
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.encoder(x)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        return out

class MnistResNet(ResNet):
    def __init__(self):
        super(MnistResNet, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=2)
        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
    def forward(self, x):
        return super(MnistResNet, self).forward(x)

if __name__=='__main__':
    model = MnistResNet()
    img = torch.zeros(16, 1, 28, 28)
    xx = model(img)
    pdb.set_trace() 
    model = ConvNet(num_channels=3, num_layers=3)
    img = torch.zeros(16, 3, 64, 64)
    xx = model(img)
    model = ConvNet(num_channels=1, num_layers=2)
    img = torch.zeros(16, 1, 28, 28)
    xx = model(img)
