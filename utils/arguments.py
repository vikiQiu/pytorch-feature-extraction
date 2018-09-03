import argparse


def train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='ImageNet1000-val',
                        help="Dataset to train. Now support ['ImageNet1000-val'].")
    parser.add_argument("--dataset-dir", type=str, default='..\data\ILSVRC2012\\', help="Dataset Directory")
    parser.add_argument("--model", type=str, default='conv',
                        help="Dataset to train. Now support ['conv', 'vgg11', 'vgg13', 'vgg16', 'vgg19']."
                             "'conv' is a simple network with 3 convolution layer."
                             "Decoder have the similar architecture as the encoder.")
    # parser.add_argument("--img-transform", type=str, default='default', help="Image Transformer")
    parser.add_argument("--img-loader", type=str, default='default', help="Image Loader")
    parser.add_argument("--epoch", type=int, default=100, help="Epoch number.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size.")
    parser.add_argument("--lr", type=int, default=1e-4, help="Learning rate.")
    parser.add_argument("--img-size", type=int, default=224,
                        help="Height Weight of the training images after transform.")
    parser.add_argument('--load-model', action="store_true", default=False)
    parser.add_argument('--use-gpus', action='store_false', dest='cuda', default=True)
    parser.add_argument("--feature-channel", type=int, default=None, help="The output channels of encoder.",
                        dest='fea_c')

    args = parser.parse_args()

    return args


def evaluate_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='ImageNet1000-val',
                        help="Dataset to train. Now support ['ImageNet1000-val'].")
    parser.add_argument("--dataset-dir", type=str, default='..\data\ILSVRC2012\\', help="Dataset Directory")
    parser.add_argument("--model", type=str, default='conv',
                        help="The architecture of encoder. Now support ['conv', 'vgg11', 'vgg13', 'vgg16', 'vgg19']."
                             "'conv' is a simple network with 3 convolution layer."
                             "Decoder have the similar architecture as the encoder.")
    parser.add_argument("--img-loader", type=str, default='default', help="Image Loader")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size.")
    # parser.add_argument("--img-transform", type=str, default='default', help="Image Transformer")
    parser.add_argument("--img-size", type=int, default=224,
                        help="Height Weight of the training images after transform.")
    parser.add_argument('--use-gpus', action='store_false', dest='cuda', default=True)
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument("--feature-channel", type=int, default=None, help="The output channels of encoder.",
                        dest='fea_c')

    args = parser.parse_args()

    return args