from torch import nn

class ConvClassifier(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super(ConvClassifier, self).__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, 1, bias=True)

    def forward(self, x):
        x = self.conv(x)
        x = x.mean((-1, -2), keepdim=False)
        return x


class SingleLinearClassifier(nn.Module):
    def __init__(self, in_features: int, num_classes: int):
        super(SingleLinearClassifier, self).__init__()
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.linear(x)
        x = x.mean(dim=1)
        return x


class MultiLayerPerception(nn.Module):
    def __init__(self, in_features: int, hidden_units: int, num_classes: int, dropout: float):
        super(MultiLayerPerception, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_units)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_units, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x

class LightweightConvClassifier(nn.Module):
    """瓶颈结构卷积分类器，通过两次降维减少参数"""
    def __init__(self, in_channels: int, num_classes: int, reduction_ratio: int = 16):
        super().__init__()
        # 第一次降维：压缩通道至1/reduction_ratio
        self.reduce1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False)
        # 第二次降维：进一步压缩至更低保真度特征
        self.reduce2 = nn.Conv2d(in_channels // reduction_ratio, in_channels // (reduction_ratio * 2), kernel_size=1, bias=False)
        # 最终分类层
        self.classifier = nn.Conv2d(in_channels // (reduction_ratio * 2), num_classes, kernel_size=1, bias=True)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.reduce1(x)  # (B, C, F, T) → (B, C/r, F, T)
        x = self.act(x)
        x = self.reduce2(x)  # → (B, C/(2r), F, T)
        x = self.act(x)
        x = self.classifier(x)  # → (B, num_classes, F, T)
        x = x.mean((-1, -2), keepdim=False)  # 全局池化
        return x

