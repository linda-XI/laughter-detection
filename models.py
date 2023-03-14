import numpy as np
from torch import nn
import torch.nn.functional as F
import torch
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
from torch import Tensor
import math
import copy
from functools import partial
from model_utils import MBConvConfig, Conv2dNormActivation, SqueezeExcitation


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class MLPModel(nn.Module):
    def __init__(self, linear_layer_size=101 * 40, hid_dim1=600, hid_dim2=100, dropout_rate=0.5, filter_sizes=None):
        super().__init__()
        print(f"training with dropout={dropout_rate}")
        self.input_dim = linear_layer_size
        self.hid_dim1 = hid_dim1
        self.hid_dim2 = hid_dim2
        self.dropout = nn.Dropout(dropout_rate)
        self.linear1 = nn.Linear(self.input_dim, hid_dim1)
        self.linear2 = nn.Linear(hid_dim1, hid_dim2)
        self.linear3 = nn.Linear(hid_dim2, 1)
        self.bn1 = nn.BatchNorm1d(num_features=hid_dim1)
        self.bn2 = nn.BatchNorm1d(num_features=hid_dim2)

        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = np.inf

    def forward(self, src):
        src = src.view((-1, self.input_dim))
        hidden1 = self.linear1(src)
        hidden1 = self.bn1(hidden1)
        hidden1 = self.dropout(hidden1)
        hidden1 = F.relu(hidden1)

        hidden2 = self.linear2(hidden1)
        hidden2 = self.bn2(hidden2)
        hidden2 = self.dropout(hidden2)
        hidden2 = F.relu(hidden2)
        output = self.linear3(hidden2)
        output = torch.sigmoid(output)
        return output

    def set_device(self, device):
        self.to(device)





class InvertedResidual(nn.Module):
    def __init__(
            self, inp: int, oup: int, stride: int, expand_ratio: int,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()
        self.stride = stride
        if stride not in [1, 2]:
            raise ValueError(f"stride should be 1 or 2 instead of {stride}")

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # pw
            layers.append(
                Conv2dNormActivation(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.ReLU6)
            )
        layers.extend(
            [
                # dw
                Conv2dNormActivation(
                    hidden_dim,
                    hidden_dim,
                    stride=stride,
                    groups=hidden_dim,
                    norm_layer=norm_layer,
                    activation_layer=nn.ReLU6,
                ),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                norm_layer(oup),
            ]
        )
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class DW(nn.Module):
    def __init__(
            self, inp: int, oup: int, stride: int, norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()

        self.stride = stride
        if stride not in [1, 2]:
            raise ValueError(f"stride should be 1 or 2 instead of {stride}")

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        #         hidden_dim = int(round(inp * expand_ratio))
        #         self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[nn.Module] = []
        #         if expand_ratio != 1:
        #             # pw
        #             layers.append(
        #                 Conv2dNormActivation(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.ReLU6)
        #             )
        layers.extend(
            [
                # dw
                Conv2dNormActivation(
                    inp,
                    inp,
                    stride=stride,
                    groups=inp,
                    # groups=hidden_dim,
                    norm_layer=norm_layer,
                    activation_layer=nn.ReLU6,
                ),
                # pw-linear
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                norm_layer(oup),
            ]
        )
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x: Tensor) -> Tensor:
        # if self.use_res_connect:
        #   return x + self.conv(x)
        # else:
        return self.conv(x)


class ResidualBlockNoBN(nn.Module):
    '''
    ResNet without Batch Normalisation
    '''

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlockNoBN, self).__init__()

        # Conv Layer 1
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=(3, 3), stride=stride, padding=1, bias=True
        )

        # Conv Layer 2
        self.conv2 = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels,
            kernel_size=(3, 3), stride=1, padding=1, bias=True
        )
        # self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels, out_channels=out_channels,
                    kernel_size=(1, 1), stride=stride, bias=False
                )  # ,
            )  # nn.BatchNorm2d(out_channels)
            # )

    def forward(self, x):
        out = nn.ReLU()(self.conv1(x))  # out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = self.conv2(out)  # out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = nn.ReLU()(out)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        # Conv Layer 1
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=(3, 3), stride=stride, padding=1, bias=True
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Conv Layer 2
        self.conv2 = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels,
            kernel_size=(3, 3), stride=1, padding=1, bias=True
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels, out_channels=out_channels,
                    kernel_size=(1, 1), stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = nn.ReLU()(out)
        return out


class ResidualBlock_DW(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock_DW, self).__init__()

        norm_layer = nn.BatchNorm2d
        block = DW
        # Conv Layer 1
        #         self.conv1 = nn.Conv2d(
        #             in_channels=in_channels, out_channels=out_channels,
        #             kernel_size=(3, 3), stride=stride, padding=1, bias=True
        #         )
        self.conv1 = block(in_channels, out_channels, stride, norm_layer=norm_layer)

        # Conv Layer 2
        #         self.conv2 = nn.Conv2d(
        #             in_channels=out_channels, out_channels=out_channels,
        #             kernel_size=(3, 3), stride=1, padding=1, bias=True
        #         )
        self.conv2 = block(out_channels, out_channels, 1, norm_layer=norm_layer)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels, out_channels=out_channels,
                    kernel_size=(1, 1), stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out += self.shortcut(x)
        out = nn.ReLU()(out)
        return out


class MobileNetV2(nn.Module):
    def __init__(
            self,
            num_classes: int = 1,
            width_mult: float = 1.0,
            inverted_residual_setting: Optional[List[List[int]]] = None,
            round_nearest: int = 8,
            block: Optional[Callable[..., nn.Module]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            dropout_rate: float = 0.2,
            # useless param
            linear_layer_size=None,
            filter_sizes=None

    ) -> None:
        """
        MobileNet V2 main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use
            dropout (float): The droupout probability
        """
        super(MobileNetV2, self).__init__()

        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = np.inf

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 64
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError(
                f"inverted_residual_setting should be non-empty or a 4-element list, got {inverted_residual_setting}"
            )

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features: List[nn.Module] = [
            Conv2dNormActivation(1, input_channel, stride=2, norm_layer=norm_layer, activation_layer=nn.ReLU6)
        ]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                input_channel = output_channel
        # building last several layers
        features.append(
            Conv2dNormActivation(
                input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.ReLU6
            )
        )
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        # x = torch.sigmoid(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    def set_device(self, device):
        for b in [self.features]:
            b.to(device)
        self.to(device)


class ResNet(nn.Module):
    '''
    Resnet model that is smaller than 'ResNetBigger' and 'ResNetNoBN'
    '''

    def __init__(self, num_classes=1, dropout_rate=0.5):
        super(ResNet, self).__init__()
        print(f"training with dropout={dropout_rate}")
        # Initial input conv
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=(3, 3),
            stride=1, padding=1, bias=False
        )

        self.bn1 = nn.BatchNorm2d(32)

        self.block1 = self._create_block(32, 32, stride=1)
        self.block2 = self._create_block(32, 16, stride=2)
        self.block3 = self._create_block(16, 16, stride=2)
        self.block4 = self._create_block(16, 16, stride=2)
        self.bn2 = nn.BatchNorm1d(192)
        self.bn3 = nn.BatchNorm1d(32)
        self.linear1 = nn.Linear(192, 32)
        self.linear2 = nn.Linear(32, num_classes)

        self.dropout = nn.Dropout(dropout_rate)

        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = np.inf

    # A block is just two residual blocks for ResNet18
    def _create_block(self, in_channels, out_channels, stride):
        return nn.Sequential(
            ResidualBlock(in_channels, out_channels, stride),
            ResidualBlock(out_channels, out_channels, 1)
        )

    def forward(self, x):
        # Output of one layer becomes input to the next
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = nn.AvgPool2d(4)(out)
        out = out.view(out.size(0), -1)
        out = self.bn2(out)
        out = self.dropout(out)
        out = self.linear1(out)
        out = self.bn3(out)
        out = self.dropout(out)
        out = F.relu(out)
        out = self.linear2(out)
        out = torch.sigmoid(out)
        return out

    def set_device(self, device):
        for b in [self.block1, self.block2, self.block3, self.block4]:
            b.to(device)
        self.to(device)


class ResNetBigger(nn.Module):
    def __init__(self, num_classes=1, dropout_rate=0.5, linear_layer_size=192, filter_sizes=[64, 32, 16, 16]):
        super(ResNetBigger, self).__init__()
        print(f"training with dropout={dropout_rate}")
        # Initial input conv
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=64, kernel_size=(3, 3),
            stride=1, padding=1, bias=False
        )

        self.bn1 = nn.BatchNorm2d(64)

        self.linear_layer_size = linear_layer_size

        self.filter_sizes = filter_sizes

        self.block1 = self._create_block(64, filter_sizes[0], stride=1)
        self.block2 = self._create_block(
            filter_sizes[0], filter_sizes[1], stride=2)
        self.block3 = self._create_block(
            filter_sizes[1], filter_sizes[2], stride=2)
        self.block4 = self._create_block(
            filter_sizes[2], filter_sizes[3], stride=2)
        self.bn2 = nn.BatchNorm1d(linear_layer_size)
        self.bn3 = nn.BatchNorm1d(32)
        self.linear1 = nn.Linear(linear_layer_size, 32)
        self.linear2 = nn.Linear(32, num_classes)

        self.dropout = nn.Dropout(dropout_rate)

        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = np.inf

    # A block is just two residual blocks for ResNet18
    def _create_block(self, in_channels, out_channels, stride):
        return nn.Sequential(
            ResidualBlock(in_channels, out_channels, stride),
            ResidualBlock(out_channels, out_channels, 1)
        )

    def forward(self, x):
        # Output of one layer becomes input to the next
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = nn.AvgPool2d(4)(out)
        out = out.view(out.size(0), -1)
        out = self.bn2(out)
        out = self.dropout(out)
        out = self.linear1(out)
        out = self.bn3(out)
        out = self.dropout(out)
        out = F.relu(out)
        out = self.linear2(out)
        out = torch.sigmoid(out)
        return out

    def set_device(self, device):
        for b in [self.block1, self.block2, self.block3, self.block4]:
            b.to(device)
        self.to(device)


class ResNetBigger_DW(nn.Module):
    def __init__(self, num_classes=1, dropout_rate=0.5, linear_layer_size=192, filter_sizes=[64, 32, 16, 16]):
        super(ResNetBigger_DW, self).__init__()
        print(f"training with dropout={dropout_rate}")
        # Initial input conv
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=64, kernel_size=(3, 3),
            stride=1, padding=1, bias=False
        )

        self.bn1 = nn.BatchNorm2d(64)

        self.linear_layer_size = linear_layer_size

        self.filter_sizes = filter_sizes

        self.block1 = self._create_block(64, filter_sizes[0], stride=1)
        self.block2 = self._create_block(
            filter_sizes[0], filter_sizes[1], stride=2)
        self.block3 = self._create_block(
            filter_sizes[1], filter_sizes[2], stride=2)
        self.block4 = self._create_block(
            filter_sizes[2], filter_sizes[3], stride=2)
        self.bn2 = nn.BatchNorm1d(linear_layer_size)
        self.bn3 = nn.BatchNorm1d(32)
        self.linear1 = nn.Linear(linear_layer_size, 32)
        self.linear2 = nn.Linear(32, num_classes)

        self.dropout = nn.Dropout(dropout_rate)

        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = np.inf

    # A block is just two residual blocks for ResNet18
    def _create_block(self, in_channels, out_channels, stride):
        return nn.Sequential(
            ResidualBlock_DW(in_channels, out_channels, stride),
            ResidualBlock_DW(out_channels, out_channels, 1)
        )

    def forward(self, x):
        # Output of one layer becomes input to the next
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = nn.AvgPool2d(4)(out)
        out = out.view(out.size(0), -1)
        out = self.bn2(out)
        out = self.dropout(out)
        out = self.linear1(out)
        out = self.bn3(out)
        out = self.dropout(out)
        out = F.relu(out)
        out = self.linear2(out)
        out = torch.sigmoid(out)
        return out

    def set_device(self, device):
        for b in [self.block1, self.block2, self.block3, self.block4]:
            b.to(device)
        self.to(device)



class ResNetNoBN(nn.Module):
    '''
    Like ResNetBigger but without Batch Normalisation
    '''

    def __init__(self, num_classes=1, dropout_rate=0.5, linear_layer_size=192):
        super(ResNetNoBN, self).__init__()
        print(f"training with dropout={dropout_rate}")
        # Initial input conv
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=64, kernel_size=(3, 3),
            stride=1, padding=1, bias=False
        )

        # self.bn1 = nn.BatchNorm2d(64)

        self.linear_layer_size = linear_layer_size

        # Create blocks
        self.block1 = self._create_block(64, 64, stride=1)
        self.block2 = self._create_block(64, 32, stride=2)
        self.block3 = self._create_block(32, 16, stride=2)
        self.block4 = self._create_block(16, 16, stride=2)
        self.linear1 = nn.Linear(linear_layer_size, 32)
        self.linear2 = nn.Linear(32, num_classes)

        self.dropout = nn.Dropout(dropout_rate)

        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = np.inf

    # A block is just two residual blocks for ResNet18
    def _create_block(self, in_channels, out_channels, stride):
        return nn.Sequential(
            ResidualBlockNoBN(in_channels, out_channels, stride),
            ResidualBlockNoBN(out_channels, out_channels, 1)
        )

    def forward(self, x):
        # Output of one layer becomes input to the next
        out = nn.ReLU()(self.conv1(x))  # out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = nn.AvgPool2d(4)(out)
        out = out.view(out.size(0), -1)
        # out = self.bn2(out)
        out = self.dropout(out)
        out = self.linear1(out)
        # out = self.bn3(out)
        out = self.dropout(out)
        out = F.relu(out)
        out = self.linear2(out)
        out = torch.sigmoid(out)
        return out

    def set_device(self, device):
        for b in [self.block1, self.block2, self.block3, self.block4]:
            b.to(device)
        self.to(device)


class EfficientNet_B0(nn.Module):
    def __init__(
            self,
            dropout: float,
            stochastic_depth_prob: float = 0.2,
            num_classes: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            #last channel for B0 is none
            #last_channel: Optional[int] = None,
    ) -> None:
        """
        EfficientNet V1 and V2 main class

        Args:
            inverted_residual_setting (Sequence[Union[MBConvConfig, FusedMBConvConfig]]): Network structure
            dropout (float): The droupout probability
            stochastic_depth_prob (float): The stochastic depth probability
            num_classes (int): Number of classes
            norm_layer (Optional[Callable[..., nn.Module]]): Module specifying the normalization layer to use
            last_channel (int): The number of channels on the penultimate layer
        """
        super(EfficientNet_B0, self).__init__()

        bneck_conf = partial(MBConvConfig, width_mult=1.0, depth_mult=1.0)
        # inverted_residual_setting = Sequence[Union[MBConvConfig, FusedMBConvConfig]]
        #last param control num of layers
        inverted_residual_setting = [
            bneck_conf(1, 3, 1, 32, 16, 1),
            bneck_conf(6, 3, 2, 16, 24, 2),
            bneck_conf(6, 5, 2, 24, 40, 2),
            bneck_conf(6, 3, 2, 40, 80, 3),
            bneck_conf(6, 5, 1, 80, 112, 3),
            bneck_conf(6, 5, 2, 112, 192, 4),
            bneck_conf(6, 3, 1, 192, 320, 1),
        ]
        last_channel = None

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        layers: List[nn.Module] = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(
            Conv2dNormActivation(
                3, firstconv_output_channels, kernel_size=3, stride=2, norm_layer=norm_layer, activation_layer=nn.SiLU
            )
        )

        # building inverted residual blocks
        total_stage_blocks = sum(cnf.num_layers for cnf in inverted_residual_setting)
        stage_block_id = 0
        for cnf in inverted_residual_setting:
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                # copy to avoid modifications. shallow copy is enough
                block_cnf = copy.copy(cnf)

                # overwrite info if not the first conv in the stage
                if stage:
                    block_cnf.input_channels = block_cnf.out_channels
                    block_cnf.stride = 1

                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / total_stage_blocks

                stage.append(block_cnf.block(block_cnf, sd_prob, norm_layer))
                stage_block_id += 1

            layers.append(nn.Sequential(*stage))

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = last_channel if last_channel is not None else 4 * lastconv_input_channels
        layers.append(
            Conv2dNormActivation(
                lastconv_input_channels,
                lastconv_output_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.SiLU,
            )
        )

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(lastconv_output_channels, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.out_features)
                nn.init.uniform_(m.weight, -init_range, init_range)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.classifier(x)
        x = torch.sigmoid(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
