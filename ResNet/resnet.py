import torch
from torch import nn
from torch.nn import functional as F

class Residual(nn.Module):
    """一个残差块"""
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        """
        kernel_size=3, padding=1
        strides = 2时, 尺寸减半

        use_1x1conv用于需要调整通道数时时
        """
        super().__init__() 
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                                kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                                kernel_size=3, padding=1, stride=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels,num_channels,
                                   kernel_size=1, stride=strides)

        else:
            self.conv3 = None

        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        
    def forward(self, x):
        Y = F.relu(self.bn1(self.conv1(x)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3: # 恒等映射
            x = self.conv3(x)
        Y += x  # in-place
        return F.relu(Y)

def resnet_block(input_channels, num_channels, num_residuals,
                first_block=False):
    """
    返回一个包含若干个残差块的列表
    """
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            # 第一个模块不需要调整通道数和尺寸减半
            blk.append(Residual(num_channels, num_channels))

    return blk

b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
b2 = nn.Sequential(*resnet_block(64, 64, num_residuals=2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, num_residuals=2))
b4 = nn.Sequential(*resnet_block(128, 256,num_residuals=2))
b5 = nn.Sequential(*resnet_block(256, 512, num_residuals=2))

net = nn.Sequential(b1, b2, b3, b4, b5,
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(), nn.Linear(512, 10))



if __name__ == "__main__":

    # 一个残差块, 观察输出形状
    blk = Residual(input_channels=3, num_channels=3, use_1x1conv=True, strides=2)
    X = torch.rand(4, 3, 6, 6)
    Y = blk(X)
    print(Y.shape)

    print("\n=========== ResNet shape of each layer ===========")
    X = torch.rand(size=(1, 1, 224, 224))
    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape:\t', X.shape)
