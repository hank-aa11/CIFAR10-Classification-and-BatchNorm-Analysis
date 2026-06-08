"""Models for Part 1: BasicCNN (for ablations) + PreAct ResNet-18 (for best run)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------- activations -----------------------------------
def get_activation(name):
    name = name.lower()
    return {
        'relu': nn.ReLU(inplace=True),
        'gelu': nn.GELU(),
        'silu': nn.SiLU(inplace=True),
        'mish': nn.Mish(inplace=True),
        'leaky_relu': nn.LeakyReLU(0.1, inplace=True),
    }[name]


# ----------------------------- BasicCNN --------------------------------------
class BasicCNN(nn.Module):
    """Configurable CNN used for ablation studies.

    Contains all required components: Conv2d, MaxPool2d, Linear, activations.
    Optional components: BatchNorm, Dropout, Residual (toggle with use_bn,
    dropout, residual).
    """

    def __init__(self, channels=(64, 128, 256), num_classes=10,
                 activation='relu', use_bn=True, dropout=0.3,
                 residual=False):
        super().__init__()
        self.activation_name = activation
        self.use_bn = use_bn
        self.residual = residual

        c1, c2, c3 = channels
        act = lambda: get_activation(activation)

        def conv_block(in_c, out_c):
            layers = [nn.Conv2d(in_c, out_c, 3, padding=1, bias=not use_bn)]
            if use_bn:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(act())
            layers += [nn.Conv2d(out_c, out_c, 3, padding=1, bias=not use_bn)]
            if use_bn:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(act())
            return nn.Sequential(*layers)

        self.block1 = conv_block(3, c1)
        self.block2 = conv_block(c1, c2)
        self.block3 = conv_block(c2, c3)
        self.pool = nn.MaxPool2d(2, 2)

        # 1x1 projections for residual paths (in case residual=True)
        self.proj1 = nn.Conv2d(3, c1, 1, bias=False)
        self.proj2 = nn.Conv2d(c1, c2, 1, bias=False)
        self.proj3 = nn.Conv2d(c2, c3, 1, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c3, 256),
            act(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(256, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.block1(x)
        if self.residual:
            h = h + self.proj1(x)
        h = self.pool(h)

        h2 = self.block2(h)
        if self.residual:
            h2 = h2 + self.proj2(h)
        h = self.pool(h2)

        h3 = self.block3(h)
        if self.residual:
            h3 = h3 + self.proj3(h)
        h = self.pool(h3)

        h = self.dropout(h)
        return self.head(h)


# ----------------------------- PreAct ResNet ---------------------------------
class PreActBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1,
                               padding=1, bias=False)
        self.shortcut = (nn.Sequential() if stride == 1 and in_planes == planes
                         else nn.Conv2d(in_planes, planes, 1,
                                        stride=stride, bias=False))

    def forward(self, x):
        out = F.relu(self.bn1(x), inplace=True)
        shortcut = (self.shortcut(out) if not isinstance(self.shortcut, nn.Sequential)
                    else x)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out), inplace=True))
        return out + shortcut


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, base=64):
        super().__init__()
        self.in_planes = base
        self.conv1 = nn.Conv2d(3, base, 3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, base, num_blocks[0], 1)
        self.layer2 = self._make_layer(block, base * 2, num_blocks[1], 2)
        self.layer3 = self._make_layer(block, base * 4, num_blocks[2], 2)
        self.layer4 = self._make_layer(block, base * 8, num_blocks[3], 2)
        self.bn = nn.BatchNorm2d(base * 8 * block.expansion)
        self.linear = nn.Linear(base * 8 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.relu(self.bn(out), inplace=True)
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.linear(out)


def preact_resnet18(num_classes=10):
    return PreActResNet(PreActBlock, [2, 2, 2, 2], num_classes=num_classes)