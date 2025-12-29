from torch import nn


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(BasicBlock, self).__init__()
        self.expansion = 1
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )
        self.identity_downsample = identity_downsample
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.layers(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(Bottleneck, self).__init__()

        self.expansion = 4
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(
                out_channels,
                out_channels * self.expansion,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels * self.expansion),
        )
        self.identity_downsample = identity_downsample
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x

        out = self.layers(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layer_config: list[int], image_channels, num_classes):
        super(ResNet, self).__init__()

        self.in_channels = 64
        self.block = block
        self.init_layers = nn.Sequential(
            nn.Conv2d(
                image_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.layers = self._make_layers(layer_config)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(
            (
                512 * block.expansion_factor
                if hasattr(block, "expansion_factor")
                else 512 * self._get_expansion(block)
            ),
            num_classes,
        )

    def _get_expansion(self, block):
        if block == Bottleneck:
            return 4
        return 1

    def _make_layers(self, layer_config):
        layers = []
        layers.append(self._make_layer(self.block, layer_config[0], 64, stride=1))
        layers.append(self._make_layer(self.block, layer_config[1], 128, stride=2))
        layers.append(self._make_layer(self.block, layer_config[2], 256, stride=2))
        layers.append(self._make_layer(self.block, layer_config[3], 512, stride=2))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.init_layers(x)
        x = self.layers(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

    def _make_layer(self, block, num_residual_blocks, out_channels, stride):
        identity_downsample = None
        layers = []
        expansion = self._get_expansion(block)

        if stride != 1 or self.in_channels != out_channels * expansion:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels * expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * expansion),
            )

        layers.append(
            block(self.in_channels, out_channels, identity_downsample, stride)
        )
        self.in_channels = out_channels * expansion

        for _ in range(1, num_residual_blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)


def resnet18(img_channels=3, num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], img_channels, num_classes)


def plain18(img_channels=3, num_classes=10):
    model = ResNet(BasicBlock, [2, 2, 2, 2], img_channels, num_classes)
    for layer in model.layers.children():
        for block in layer.children():
            if hasattr(block, "identity_downsample"):
                object.__setattr__(block, "identity_downsample", None)
            original_layers = block.layers
            original_relu = block.relu

            def make_forward(layers: nn.Sequential, relu: nn.ReLU):
                def forward(x):
                    return relu(layers(x))

                return forward

            block.forward = make_forward(original_layers, original_relu)  # type: ignore
    return model


def resnet34(img_channels=3, num_classes=10):
    return ResNet(BasicBlock, [3, 4, 6, 3], img_channels, num_classes)


def resnet50(img_channels=3, num_classes=10):
    return ResNet(Bottleneck, [3, 4, 6, 3], img_channels, num_classes)


def resnet101(img_channels=3, num_classes=10):
    return ResNet(Bottleneck, [3, 4, 23, 3], img_channels, num_classes)
