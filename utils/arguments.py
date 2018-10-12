import argparse


def train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='ImageNet1000-val',
                        help="Dataset to train. Now support ['ImageNet1000-val'].")
    parser.add_argument("--dataset-dir", type=str, default='..\data\ILSVRC2012\\', help="Dataset Directory")
    parser.add_argument("--test-dir", type=str, default='..\data\ILSVRC2012\\', help="Test dataset Directory")
    parser.add_argument("--cover-dir", type=str, default='..\data\ILSVRC2012\\', help="Cover dataset Directory")
    parser.add_argument("--model", type=str, default='conv',
                        help="Dataset to train. Now support ['conv', 'vgg11', 'vgg13', 'vgg16', 'vgg19']."
                             "'conv' is a simple network with 3 convolution layer."
                             "Decoder have the similar architecture as the encoder.")
    # parser.add_argument("--img-transform", type=str, default='default', help="Image Transformer")
    parser.add_argument("--img-loader", type=str, default='default', help="Image Loader")
    parser.add_argument("--epoch", type=int, default=16, help="Epoch number.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Learning rate.")
    parser.add_argument("--img-size", type=int, default=224,
                        help="Height Weight of the training images after transform.")
    parser.add_argument('--load-model', action="store_true", default=False)
    parser.add_argument('--use-gpus', action='store_false', dest='cuda', default=True)
    parser.add_argument("--feature-channel", type=int, default=None, help="The output channels of encoder.",
                        dest='fea_c')
    parser.add_argument("--imgnet-p", type=float, default=0,
                        help="The percentage of ImageNet train dataset to train the classifier."
                             "If 0, only train the decoder.")
    parser.add_argument("--decoder", type=str, default='vgg',
                        help="Decoder structure. Now support ['vgg', 'simple'].")

    args = parser.parse_args()

    return args


def feature_classifier_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='ImageNet1000-val',
                        help="Dataset to train. Now support ['ImageNet1000-val'].")
    parser.add_argument("--dataset-dir", type=str, default='..\data\ILSVRC2012\ILSVRC2012_img_train_subset\\',
                        help="Train dataset Directory")
    parser.add_argument("--model", type=str, default='conv',
                        help="Dataset to train. Now support ['conv', 'vgg11', 'vgg13', 'vgg16', 'vgg19']."
                             "'conv' is a simple network with 3 convolution layer."
                             "Decoder have the similar architecture as the encoder.")
    # parser.add_argument("--img-transform", type=str, default='default', help="Image Transformer")
    parser.add_argument("--img-loader", type=str, default='default', help="Image Loader")
    parser.add_argument("--epoch", type=int, default=100, help="Epoch number.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--img-size", type=int, default=224,
                        help="Height Weight of the training images after transform.")
    parser.add_argument('--load-model', action="store_true", default=False)
    parser.add_argument('--use-gpus', action='store_false', dest='cuda', default=True)
    parser.add_argument("--feature-channel", type=int, default=None, help="The output channels of encoder.",
                        dest='fea_c')
    parser.add_argument("--main-model", type=str, default='VAE',
                        help="Main model. Now support ['AE', 'VAE', AEClass, vgg_classifier].")

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
    parser.add_argument("--main-model", type=str, default='VAE',
                        help="Main model. Now support ['AE', 'VAE', AEClass, vgg_classifier].")
    parser.add_argument("--img-loader", type=str, default='default', help="Image Loader")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size.")
    # parser.add_argument("--img-transform", type=str, default='default', help="Image Transformer")
    parser.add_argument("--img-size", type=int, default=224,
                        help="Height Weight of the training images after transform.")
    parser.add_argument('--use-gpus', action='store_false', dest='cuda', default=True)
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument("--feature-channel", type=int, default=None, help="The output channels of encoder.",
                        dest='fea_c')
    parser.add_argument('--load-feature', action="store_true", default=False)

    args = parser.parse_args()

    return args