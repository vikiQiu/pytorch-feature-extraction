import torch
import torch.nn as nn


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
            ),  # 32*32*3->16*16*64
            nn.LeakyReLU(),

            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=2,
                padding=1,
            ),  # 16*16*64->8*8*128
            nn.BatchNorm2d(num_features=128, affine=True),  # num_features=batch_size x num_features [x width]
            nn.LeakyReLU(),

            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=2,
                padding=1,
            ),  # 8*8*128->4*4*256 Total 4096 feature
            nn.BatchNorm2d(num_features=256, affine=True),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        encode = self.encoder(x)
        return encode


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
    def __init__(self, model='vgg16', batch_norm=True):
        super(VGGEncoder, self).__init__()

        cfg = {
            'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512,
                  512, 512, 'M'],
        }

        model_list = {'vgg11': 'A', 'vgg13': 'B', 'vgg16': 'D', 'vgg19': 'E'}
        self.encoder = self._make_vgg_layers(cfg[model_list[model]], batch_norm)

    def forward(self, x):
        encode = self.encoder(x)
        print('Encoder size:')
        print(encode.shape)
        return encode

    @staticmethod
    def _make_vgg_layers(cfg, batch_norm=False):
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
        return nn.Sequential(*layers)


class VGGDecoder(nn.Module):
    def __init__(self, model='vgg16', batch_norm=True):
        super(VGGDecoder, self).__init__()

        cfg = {
            'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512,
                  512, 512, 'M'],
        }

        model_list = {'vgg11': 'A', 'vgg13': 'B', 'vgg16': 'D', 'vgg19': 'E'}
        self.decoder = self._make_vgg_layers(cfg[model_list[model]], batch_norm)

    def forward(self, x):
        decode = self.decoder(x)
        print('Decoder size:')
        print(decode.shape)
        return decode

    @staticmethod
    def _make_vgg_layers(cfg, batch_norm=False):
        layers = []
        in_channels = 512
        for v in cfg[::-1]:
            if v == 'M':
                layers += [nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels,
                                              kernel_size=4, stride=2, padding=1)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        conv2d = nn.Conv2d(in_channels, 3, kernel_size=3, padding=1)
        layers += [conv2d, nn.LeakyReLU(inplace=True)]
        return nn.Sequential(*layers)