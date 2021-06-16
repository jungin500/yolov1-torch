import torch
from torch.nn import *


class YoloLayer(Module):
    def __init__(self, layer_params, pool=True):
        super().__init__()
        layers = []
        for in_channels, out_channels, kernel_size, stride, activation in layer_params:
            layers.append(
                YoloConv(in_channels=in_channels, out_channels=out_channels,
                         kernel_size=kernel_size, stride=stride, activation=activation)
            )

        self.in_layers = Sequential(*layers)
        if pool:
            self.pool = MaxPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.pool = Identity()

    def forward(self, x):
        x = self.in_layers(x)
        x = self.pool(x)
        return x


class YoloConv(Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=1, activation='relu'):
        super().__init__()
        self.conv = Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels,
                           stride=stride, padding=kernel_size[0] // 2)
        if activation.lower() == 'relu':
            self.activation = LeakyReLU(negative_slope=0.1, inplace=True)
        else:
            self.activation = Identity

    def forward(self, x):
        x = self.conv(x)
        self.activation(x)  # inplace
        return x


class YoloFinalLayer(Module):
    def __init__(self, activation='relu'):
        super().__init__()
        self.classifier0 = Linear(in_features=50176, out_features=4096)
        if activation.lower() == 'relu':
            self.act = LeakyReLU(negative_slope=0.1, inplace=True)
        else:
            self.act = Identity()
        self.classifier1 = Linear(in_features=4096, out_features=1470)

    def forward(self, x):
        x = x.view(-1, 50176)
        x = self.classifier0(x)
        self.act(x)
        x = self.classifier1(x)
        x = x.view(-1, 30, 7, 7)
        return x


class YoloFeatureExtractor(Module):
    def __init__(self, activation='relu'):
        super().__init__()
        layers = [
            YoloLayer([
                # in, out, kernel, stride, apoolct
                (3, 64, (7, 7), 2, activation),
            ]),
            YoloLayer([
                (64, 192, (3, 3), 1, activation),
            ]),
            YoloLayer([
                (192, 128, (1, 1), 1, activation),
                (128, 256, (3, 3), 1, activation),
                (256, 256, (1, 1), 1, activation),
                (256, 512, (3, 3), 1, activation),
            ]),
            YoloLayer([
                (512, 256, (1, 1), 1, activation),
                (256, 512, (3, 3), 1, activation),
                (512, 256, (1, 1), 1, activation),
                (256, 512, (3, 3), 1, activation),
                (512, 256, (1, 1), 1, activation),
                (256, 512, (3, 3), 1, activation),
                (512, 256, (1, 1), 1, activation),
                (256, 512, (3, 3), 1, activation),
                (512, 512, (1, 1), 1, activation),
                (512, 1024, (3, 3), 1, activation),
            ]),
            YoloLayer([
                (1024, 512, (1, 1), 1, activation),
                (512, 1024, (3, 3), 1, activation),
                (1024, 512, (1, 1), 1, activation),
                (512, 1024, (3, 3), 1, activation),
            ], pool=False)
        ]

        self.layers = Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x


class YoloClassifier(Module):
    def __init__(self, activation='relu'):
        super().__init__()
        layers = [
            YoloLayer([
                (1024, 1024, (3, 3), 1, 'relu'),
                (1024, 1024, (3, 3), 2, 'relu'),
            ], pool=False),
            YoloLayer([
                (1024, 1024, (3, 3), 1, 'relu'),
                (1024, 1024, (3, 3), 1, 'relu'),
            ], pool=False),
            YoloFinalLayer('relu')
        ]

        self.layers = Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x


class YOLOv1(Module):
    def __init__(self):
        super().__init__()

        self.features = YoloFeatureExtractor('relu')
        self.classifiers = YoloClassifier('relu')

    def forward(self, x):
        x = self.features(x)
        x = self.classifiers(x)
        return x


class YOLOv1Pretrainer(Module):
    def __init__(self, classes):
        super().__init__()

        self.features = YoloFeatureExtractor('relu')
        self.classifiers = Sequential(
            AvgPool2d(kernel_size=(7, 7)),
            Flatten(),
            Linear(in_features=1024, out_features=classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifiers(x)
        return x
