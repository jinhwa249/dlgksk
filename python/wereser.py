# import torchvision.transforms as transforms

# mnist_transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5),(1.0,))])

# from torchvision.datasets import MNIST
# import requests
# download_root = 'pythorch/data/MNIST_DATASET'

# train_dataset = MNIST(download_root, transform=mnist_transform,
#                       train=True, download= True)
# vaild_dataset = MNIST(download_root, transform=mnist_transform,
#                       train=True, download= True)
# test_dataset = MNIST(download_root, transform=mnist_transform,
#                       train=True, download= True)

import torch.nn as nn
import torch
# model = nn.Linear(in_features=1, out_features=1, bias=True)
model = nn.Linear(1, 1)

class SingleLayer(nn.Module):
    def __init__(self, inputs):
        super().__init__()
        self.layer = nn.Linear(1, 2)
        self.layer = nn.Linear(inputs, 1)
        self.activation = nn.Sigmoid()
        
    def forward(self, x):
        X = self.layer(X)
        X = self.activation(X)
        return X
print(model.bias)    
    
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels==64, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels==30, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=30*5*5, out_channels==10, bias=True),
            nn.ReLU(inplace=True))
    def forward(self, x):
        X = self.layer1(x)
        X = self.layer2(x)
        X = x.view(X.shape[0], -1)
        X = self.layer3(x) 
        return X
def main():
    # model = nn.Linear(in_features=1, out_channels=)
    midel = MLP()
    print(list(model.children()))


if __name__=='__name__':
    main()           

