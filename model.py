import torch
import torch.nn as nn
import torchvision.models as models


class SimpleEncoder(nn.Module):
    '''
    Input: (Height, Weight, 3)
    Output: (Height/8, Weight/8, 256)
    '''
    def __init__(self):
        super(SimpleEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.LeakyReLU(),

            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(num_features=128, affine=True),  # num_features=batch_size x num_features [x width]
            nn.LeakyReLU(),

            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(num_features=256, affine=True),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        encode = self.encoder(x)
        return encode

    def get_out_channel(self):
        return 256


class SimpleDecoder(nn.Module):
    '''
    Input: (Height/8, Weight/8, 256)
    Output: (Height, Weight, 3)
    '''
    def __init__(self):
        super(SimpleDecoder, self).__init__()

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=128,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(num_features=128, affine=True),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(num_features=64, affine=True),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=3,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        decode = self.decoder(x)
        return decode


class VGGEncoder(nn.Module):
    def __init__(self, model='vgg16', out_channels=None, batch_norm=True):
        super(VGGEncoder, self).__init__()

        cfg = {
            'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512,
                  512, 512, 'M'],
        }

        model_list = {'vgg11': 'A', 'vgg13': 'B', 'vgg16': 'D', 'vgg19': 'E'}
        self.out_channels = out_channels
        self.encoder = self._make_vgg_layers(cfg[model_list[model]], batch_norm)

    def forward(self, x):
        encode = self.encoder(x)
        return encode

    def get_out_channel(self):
        return 512 if self.out_channels is None else self.out_channels

    def _make_vgg_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        if self.out_channels is not None:
            layers += [nn.Conv2d(512, self.out_channels, kernel_size=1)]
        return nn.Sequential(*layers)


class VGG16Feature(nn.Module):
    def __init__(self):
        super(VGG16Feature, self).__init__()
        self.vgg = models.vgg16_bn(pretrained=True).features
        # for param in self.vgg.parameters():
        #     param.requires_grad = False

    def forward(self, x):
        x = self.vgg(x)
        return x


class VGG16Classifier(nn.Module):
    def __init__(self):
        super(VGG16Classifier, self).__init__()
        self.vgg = models.vgg16_bn(pretrained=True).classifier
        # for param in self.vgg.parameters():
        #     param.requires_grad = False

    def forward(self, x):
        x = self.vgg(x)
        return x

    def get_feature(self, x):
        for name, layer in self.vgg._modules.items():
            if int(name) <= 3:
                x = layer(x)
            else:
                break
        return x


class VGGDecoder(nn.Module):
    def __init__(self, model='vgg16', out_channels=None,  batch_norm=True):
        super(VGGDecoder, self).__init__()

        cfg = {
            'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512,
                  512, 512, 'M'],
            'S': [64, 'M', 128, 'M', 256, 'M', 512, 'M', 512, 'M']
        }

        model_list = {'vgg11': 'A', 'vgg13': 'B', 'vgg16': 'D', 'vgg19': 'E', 'Simple': 'S'}
        self.out_channels = out_channels
        self.decoder = self._make_vgg_layers(cfg[model_list[model]], batch_norm)

    def forward(self, x):
        decode = self.decoder(x)
        return decode

    def _make_vgg_layers(self, cfg, batch_norm=False):
        in_channels = int(512/8)
        layers = [nn.Conv2d(512, in_channels, kernel_size=1)] if self.out_channels is None else [nn.Conv2d(32, in_channels, kernel_size=1)]
        for v in cfg[::-1]:
            if v == 'M':
                layers += [nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels,
                                              kernel_size=4, stride=2, padding=1)]
            else:
                vv = int(v/8)
                conv2d = nn.Conv2d(in_channels, vv, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(vv), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = vv
        conv2d = nn.Conv2d(in_channels, 3, kernel_size=3, padding=1)
        layers += [conv2d, nn.Sigmoid()]
        # layers += [conv2d, nn.LeakyReLU()]
        return nn.Sequential(*layers)