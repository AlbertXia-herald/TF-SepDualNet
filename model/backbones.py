import torch
import torch.nn as nn
from model.beats.BEATs import BEATsConfig, BEATs
import torch.nn.functional as F
from model.classifiers import SingleLinearClassifier, ConvClassifier, LightweightConvClassifier
from model.function import ReverseLayerF
from model.shared import ConvBnRelu, ResNorm, AdaResNorm, BroadcastBlock, TimeFreqSepConvolutions, LowRankLinear


class _BaseBackbone(nn.Module):
    """ Base Module for backbones. """

class DCASEBaselineCnn3(_BaseBackbone):
    """
    Previous baseline system of the task 1 of DCASE Challenge.
    A simple CNN consists of 3 conv layers and 2 linear layers.

    Note: Kernel size of the Max-pooling layers need to be changed for adapting to different size of inputs.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of output classes.
        base_channels (int): Number of base channels.
        kernel_size (int): Kernel size of convolution layers.
        dropout (float): Dropout rate.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10, base_channels: int = 16, kernel_size: int = 7,
                 dropout: float = 0.3):
        super(DCASEBaselineCnn3, self).__init__()
        self.conv1 = ConvBnRelu(in_channels, base_channels, kernel_size, padding=(kernel_size - 1) // 2, bias=True)
        self.conv2 = ConvBnRelu(base_channels, base_channels, kernel_size, padding=(kernel_size - 1) // 2, bias=True)
        self.conv3 = ConvBnRelu(base_channels, base_channels * 2, kernel_size, padding=(kernel_size - 1) // 2,
                                bias=True)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        # Adjust the kernel of max_pooling util the output shape is (F=2, T=1)
        self.max_pooling1 = nn.MaxPool2d((5, 5))
        self.max_pooling2 = nn.MaxPool2d((4, 10))
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(base_channels * 4, 100)
        self.relu = nn.ReLU()
        self.dropout3 = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(100, num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        # print('INPUT SHAPE:', x.shape)
        x = self.conv1(x)

        # print('CONV2 INPUT SHAPE:', x.shape)
        x = self.conv2(x)
        x = self.max_pooling1(x)
        x = self.dropout1(x)

        # print('CONV3 INPUT SHAPE:', x.shape)
        x = self.conv3(x)
        x = self.max_pooling2(x)
        x = self.dropout2(x)

        # print('FLATTEN INPUT SHAPE:', x.shape)
        x = self.flatten(x)

        # print('LINEAR INPUT SHAPE:', x.shape)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout3(x)
        x = self.linear2(x)
        # print('OUTPUT SHAPE:', x.shape)
        return x


class BCResNet(_BaseBackbone):
    """
    Implementation of BC-ResNet, based on Broadcasted Residual Learning.
    Check more details at: https://arxiv.org/abs/2106.04140

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of output classes.
        base_channels (int): Number of base channels that controls the complexity of model.
        depth (int): Network depth with single option: 15.
        kernel_size (int): Kernel size of each convolutional layer in BC blocks.
        dropout (float): Dropout rate.
        sub_bands (int): Number of sub-bands for SubSpectralNorm (SSN). ``1`` indicates SSN is not applied.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10, base_channels: int = 40, depth: int = 15,
                 kernel_size: int = 3, dropout: float = 0.1, sub_bands: int = 1):
        super(BCResNet, self).__init__()
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.sub_bands = sub_bands

        cfg = {
            15: ['N', 1, 1, 'N', 'M', 1.5, 1.5, 'N', 'M', 2, 2, 'N', 2.5, 2.5, 2.5, 'N'],
        }

        self.conv_layers = nn.Conv2d(in_channels, 2 * base_channels, 5, stride=2, bias=False, padding=2)
        # Compute the number of channels for each layer.
        layer_config = [int(i * base_channels) if not isinstance(i, str) else i for i in cfg[depth]]
        self.middle_layers = self._make_layers(base_channels, layer_config)
        # Get the index of channel number for the cla_layer.
        last_num_index = -1 if not isinstance(layer_config[-1], str) else -2
        # 1x1 convolution layer as the cla_layer.
        self.classifier = ConvClassifier(layer_config[last_num_index], num_classes)

    def _make_layers(self, width: int, layer_config: list):
        layers = []
        vt = 2 * width
        for v in layer_config:
            if v == 'N':
                layers += [ResNorm(channels=vt)]
            elif v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif v != vt:
                layers += [BroadcastBlock(vt, v, self.kernel_size, self.dropout, self.sub_bands)]
                vt = v
            else:
                layers += [BroadcastBlock(vt, vt, self.kernel_size, self.dropout, self.sub_bands)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.middle_layers(x)
        x = self.classifier(x)
        return x


class TFSepNet(_BaseBackbone):
    """
    Implementation of TF-SepNet-64, based on Time-Frequency Separate Convolutions. Check more details at:
    https://ieeexplore.ieee.org/abstract/document/10447999 and
    https://dcase.community/documents/challenge2024/technical_reports/DCASE2024_Cai_61_t1.pdf

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of output classes.
        base_channels (int): Number of base channels that controls the complexity of model.
        depth (int): Network depth with two options: 16 or 17. When depth = 17, an additional Max-pooling layer is inserted before the last TF-SepConvs black.
        kernel_size (int): Kernel size of each convolutional layer in TF-SepConvs blocks.
        dropout (float): Dropout rate.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10, base_channels: int = 64, depth: int = 17,
                 kernel_size: int = 3, dropout: float = 0.1):
        super(TFSepNet, self).__init__()
        assert base_channels % 2 == 0, "Base_channels should be divisible by 2."
        self.dropout = dropout
        self.kernel_size = kernel_size

        # Two settings of the depth. ``17`` have an additional Max-pooling layer before the final block of TF-SepConvs.
        cfg = {
            16: ['N', 1, 1, 'N', 'M', 1.5, 1.5, 'N', 'M', 2, 2, 'N', 2.5, 2.5, 2.5, 'N'],
            17: ['N', 1, 1, 'N', 'M', 1.5, 1.5, 'N', 'M', 2, 2, 'N', 'M', 2.5, 2.5, 2.5, 'N'],
        }

        self.conv_layers = nn.Sequential(ConvBnRelu(in_channels, base_channels // 2, 3, stride=2, padding=1),
                                         ConvBnRelu(base_channels // 2, 2 * base_channels, 3, stride=2, padding=1,
                                                    groups=base_channels // 2))
        # Compute the number of channels for each layer.
        layer_config = [int(i * base_channels) if not isinstance(i, str) else i for i in cfg[depth]]
        self.middle_layers = self._make_layers(base_channels, layer_config)
        # Get the index of channel number for the cla_layer.
        last_num_index = -1 if not isinstance(layer_config[-1], str) else -2
        # 1x1 convolution layer as the cla_layer.
        self.classifier = ConvClassifier(layer_config[last_num_index], num_classes)

    def _make_layers(self, width: int, layer_config: list):
        layers = []
        vt = width * 2
        for v in layer_config:
            if v == 'N':
                layers += [ResNorm(channels=vt)]
            elif v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif v != vt:
                layers += [TimeFreqSepConvolutions(vt, v, self.kernel_size, self.dropout)]
                vt = v
            else:
                layers += [TimeFreqSepConvolutions(vt, vt, self.kernel_size, self.dropout)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.middle_layers(x)
        x = self.classifier(x)
        return x


class PretrainedBEATs(_BaseBackbone):
    """
    Module wrapping a BEATs encoder with pretrained weights and a new linear classifier.
    Check more details at: https://arxiv.org/abs/2212.09058

    Args:
        pretrained (str): path to the pretrained checkpoint. Leave ``None`` when no need pretrained.
    """

    def __init__(self, pretrained=None, num_classes=10, **kwargs):
        super(PretrainedBEATs, self).__init__()
        # Load model config and weights from checkpoints when use pretrained, otherwise use default settings
        ckpt = torch.load(pretrained) if pretrained else None
        hyperparams = ckpt['cfg'] if pretrained else kwargs
        cfg = BEATsConfig(hyperparams)
        self.encoder = BEATs(cfg)
        if pretrained:
            self.encoder.load_state_dict(ckpt['model'], strict=False)
        # Create a new linear classifier
        self.classifier = SingleLinearClassifier(in_features=cfg.encoder_embed_dim, num_classes=num_classes)

    def forward(self, x):
        x = self.encoder.extract_features(x)[0]
        return self.classifier(x)

#调整后的TFSepNet,替换CNN14.AudioNet的作用
class TFSep2Net(_BaseBackbone):
    def __init__(self, in_channels: int = 1, base_channels: int = 64, depth: int = 17,
                 kernel_size: int = 3, dropout: float = 0.1, dim: int = 1024):  # 增加dim参数
        super(TFSep2Net, self).__init__()
        assert base_channels % 2 == 0, "Base_channels should be divisible by 2."
        self.dropout = dropout
        self.kernel_size = kernel_size
        self.dim = dim  # 输出特征维度

        # 控制网络深度和通道数
        cfg = {
            16: ['N', 1, 1, 'N', 'M', 1.5, 1.5, 'N', 'M', 2, 2, 'N', 2.5, 2.5, 2.5, 'N'],
            17: ['N', 1, 1, 'N', 'M', 1.5, 1.5, 'N', 'M', 2, 2, 'N', 'M', 2.5, 2.5, 2.5, 'N'],
        }

        # 前端卷积层
        self.conv_layers = nn.Sequential(
            ConvBnRelu(in_channels, base_channels // 2, 3, stride=2, padding=1),
            ConvBnRelu(base_channels // 2, 2 * base_channels, 3, stride=2, padding=1, groups=base_channels // 2)
        )

        # 计算中间层通道配置
        layer_config = [int(i * base_channels) if not isinstance(i, str) else i for i in cfg[depth]]
        self.middle_layers = self._make_layers(base_channels, layer_config)

        # 获取分类头前的特征通道数（用于映射到dim）
        last_num_index = -1 if not isinstance(layer_config[-1], str) else -2
        self.feat_channels = layer_config[last_num_index]

        # 全局池化（将2D特征转为1D）和特征映射层（对齐到dim维）
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化
        self.feature_proj = nn.Linear(self.feat_channels, dim)  # 映射到目标维度
        self.bn = nn.BatchNorm1d(dim)  # 保持与AudioNet一致的BN层
        self.out_channels = layer_config[last_num_index]

    def _make_layers(self, width: int, layer_config: list):
        layers = []
        vt = width * 2
        for v in layer_config:
            if v == 'N':
                layers += [ResNorm(channels=vt)]
            elif v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif v != vt:
                layers += [TimeFreqSepConvolutions(vt, v, self.kernel_size, self.dropout)]
                vt = v
            else:
                layers += [TimeFreqSepConvolutions(vt, vt, self.kernel_size, self.dropout)]
        return nn.Sequential(*layers)

    def forward(self, x):
        # 前向传播到中间特征
        x = self.conv_layers(x)
        x = self.middle_layers(x)  # 输出形状：(batch, feat_channels, freq, time)

        # 全局池化：(batch, feat_channels, freq, time) → (batch, feat_channels, 1, 1)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # 展平为(batch, feat_channels)

        # 映射到dim维并标准化（匹配AudioNet的输出格式）
        x = self.feature_proj(x)
        x = self.bn(x)
        return x

# 定义适配TFSepNet的编码器（复用原encoder逻辑，替换audio_net）
class TFSepEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, depth: int = 17, dim=1024, city_class_num=10, device_class_num=9):
        super(TFSepEncoder, self).__init__()
        # 音频特征提取器替换为调整后的TFSepNet
        self.audio_net = TFSep2Net(in_channels=in_channels,depth=depth,dim=dim)  # 使用修改后的TFSep2Net
        # 以下复用原encoder的城市/设备分类分支
        self.city = nn.Sequential(
            nn.Linear(dim, int(dim * 0.5)),
            nn.ReLU(),
            nn.Linear(int(dim * 0.5), int(dim * 0.5)),
            nn.ReLU(),
            nn.Linear(int(dim * 0.5), city_class_num),
        )
        self.device = nn.Sequential(
            nn.Linear(dim, int(dim * 0.5)),
            nn.ReLU(),
            nn.Linear(int(dim * 0.5), int(dim * 0.5)),
            nn.ReLU(),
            nn.Linear(int(dim * 0.5), device_class_num),
        )
        self.shared_layer = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
        )

    def forward(self, x, x_t, type, alpha):
        # 复用原encoder的forward逻辑
        if type == "train":
            batch = x.size(0)
            all_x = torch.cat([x, x_t], dim=0)
            all_x_feature = self.audio_net(all_x)  # 特征来自TFSepNet
            reverse_feature = ReverseLayerF.apply(all_x_feature, alpha)
            pre_city = self.city(self.shared_layer(reverse_feature))
            pre_device = self.device(self.shared_layer(reverse_feature))
            return pre_device, pre_city, all_x_feature #返还预测的device，city和拼接后提取出的音频特征（all）
        else:
            x_t = self.audio_net(x_t)
            return x_t

class LightweightTFSepEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 32, num_domains: int = 4):
        super().__init__()
        # 特征解耦模块：低秩分解线性层替代传统线性层
        self.feature_decouple = LowRankLinear(in_dim, hidden_dim, rank_ratio=0.25)  # 秩=1/4原维度
        # 域分类器：共享低维特征减少参数
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_domains)
        )
        # 轻量投影头：用于特征对齐
        self.projection = nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=1)

    def forward(self, x):
        # 特征解耦
        x = self.feature_decouple(x)
        # 域分类输出
        domain_logits = self.domain_classifier(x.mean((-1, -2)))  # 全局池化后分类
        # 投影特征用于对齐
        projected = self.projection(x)
        return domain_logits, projected


# 结合TFSepNet和DualAlignment的域适应模型
class TFSepDualNet(_BaseBackbone):
    """
    双通道TFSepNet，支持域适应训练，输出场景分类和域分类结果
    - 初始通道数：64
    - 域数量：9（设备域）
    """
    def __init__(self,
                 in_channels: int = 1,
                 base_channels: int = 64,  # 初始通道数64
                 depth: int = 17,
                 kernel_size: int = 3,
                 dropout: float = 0.1,
                 num_classes: int = 10,  # 场景类别数
                 num_device_domains: int = 9,
                 num_city_domains=10):  # 域数量
        super(TFSepDualNet, self).__init__()
        assert base_channels % 2 == 0, "base_channels必须为偶数"
        self.num_classes = num_classes
        self.num_domains = num_device_domains

        # 基础配置（深度17的网络结构）
        cfg = {
            17: ['N', 1, 1, 'N', 'M', 1.5, 1.5, 'N', 'M', 2, 2, 'N', 'M', 2.5, 2.5, 2.5, 'N'],
        }

        # 1. 共享编码器（通道变化：1→32→128）
        self.shared_encoder = nn.Sequential(
            ConvBnRelu(in_channels, base_channels//2, 3, stride=2, padding=1),  # 64/2=32
            ConvBnRelu(base_channels//2, 2*base_channels, 3, stride=2, padding=1, groups=base_channels//2)  # 2*64=128
        )

        # 2. 中间层（基于TF分离卷积，通道数按配置变化）
        layer_config = [int(i * base_channels) if not isinstance(i, str) else i for i in cfg[depth]]
        self.middle_layers = self._make_layers(2*base_channels, layer_config, kernel_size, dropout)

        # 3. 场景分类分支（双分支结构）
        # 分支1：通道160 → 64 → 10
        self.scene_branch1 = nn.Sequential(
            ConvBnRelu(160, base_channels, 1),  # 160→64
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(base_channels, num_classes)
        )
        # 分支2：通道160 → 64 → 10
        self.scene_branch2 = nn.Sequential(
            ConvBnRelu(160, base_channels, 1),  # 160→64
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(base_channels, num_classes)
        )

        # 4. 域分类分支（设备+城市，共享特征）
        # 设备域分类：160→64→9
        self.domain_device = nn.Sequential(
            ConvBnRelu(160, base_channels, 1),  # 160→64
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(base_channels, num_device_domains)
        )
        # 城市域分类：160→64→9（假设城市域数量也为9）
        self.domain_city = nn.Sequential(
            ConvBnRelu(160, base_channels, 1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(base_channels, num_city_domains)  # 关键修复：9→10
        )

    def _make_layers(self, init_channels, layer_config, kernel_size, dropout):
        """构建中间TF分离卷积层，确保通道匹配"""
        layers = []
        current_channels = init_channels  # 初始128
        for v in layer_config:
            if v == 'N':
                layers.append(ResNorm(current_channels))
            elif v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                # 通道变化：128→96→96→128→...→160
                layers.append(TimeFreqSepConvolutions(current_channels, v, kernel_size, dropout))
                current_channels = v
        return nn.Sequential(*layers)

    def forward(self, x=None, x_t=None, type="train", alpha=0.0):
        """
        - 训练模式：返回(设备预测, 城市预测, 源域分支1, 源域分支2, 目标域分支1, 目标域分支2)
        - 验证/测试模式：返回融合后的场景预测
        """
        if type == "train":
            # 源域前向
            x = self.shared_encoder(x)
            x = self.middle_layers(x)  # 输出通道160
            f1_src = self.scene_branch1(x)
            f2_src = self.scene_branch2(x)

            # 目标域前向
            x_t = self.shared_encoder(x_t)
            x_t = self.middle_layers(x_t)  # 输出通道160
            t1_tgt = self.scene_branch1(x_t)
            t2_tgt = self.scene_branch2(x_t)

            # 域分类（梯度反转）
            x_feat = ReverseLayerF.apply(torch.cat([x, x_t]), alpha)
            pre_device = self.domain_device(x_feat)
            pre_city = self.domain_city(x_feat)

            return pre_device, pre_city, f1_src, f2_src, t1_tgt, t2_tgt

        else:  # 验证/测试模式
            x = self.shared_encoder(x_t)
            x = self.middle_layers(x)
            y1 = self.scene_branch1(x)
            y2 = self.scene_branch2(x)
            return (y1 + y2) / 2  # 融合双分支





