import os
import argparse
import torch
import time
import torch.utils.data as Data
from torch.autograd import Variable
import torchvision
import torch.nn as nn
from torchvision.utils import save_image
from data_process import ImageNetDataset, transformers, loaders
from model import SimpleEncoder, SimpleDecoder, VGGEncoder, VGGDecoder


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='ImageNet1000-val',
                    help="Dataset to train. Now support ['ImageNet1000-val'].")
parser.add_argument("--dataset-dir", type=str, default='..\data\ILSVRC2012\\', help="Dataset Directory")
parser.add_argument("--model", type=str, default='conv',
                    help="Dataset to train. Now support ['conv', 'vgg11', 'vgg13', 'vgg16', 'vgg19'].")
# parser.add_argument("--img-transform", type=str, default='default', help="Image Transformer")
parser.add_argument("--img-loader", type=str, default='default', help="Image Loader")
parser.add_argument("--epoch", type=int, default=100, help="Epoch number.")
parser.add_argument("--batch-size", type=int, default=256, help="Batch size.")
parser.add_argument("--lr", type=int, default=1e-4, help="Learning rate.")
parser.add_argument("--img-size", type=int, default=224, help="Height Weight of the training images after transform.")
parser.add_argument('--load-model', action="store_true", default=False)
parser.add_argument('--use-gpus', action='store_false', dest='cuda', default=True)
parser.add_argument("--feature-channel", type=int, default=None, help="The output channels of encoder.", dest='fea_c')

args = parser.parse_args()
cuda = args.cuda and torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

HEIGHT = args.img_size
WEIGHT = args.img_size


def check_dir_exists(dirs):
    for d in dirs:
        if not os.path.exists(d):
            os.mkdir(d)


def getDataset(args):
    '''
        Now support ['ImageNet1000-val']。
        Add more dataset in future.
        '''
    if args.dataset == 'ImageNet1000-val':
        label_dir = os.path.join(args.dataset_dir, 'ILSVRC2012_bbox_val_v3')
        img_dir = os.path.join(args.dataset_dir, 'ILSVRC2012_img_val')
        dataset = ImageNetDataset(img_dir, label_dir,
                                  img_transform=transformers['crop' + str(args.img_size)],
                                  loader=loaders[args.img_loader])
        return dataset
    pass


def init_model(model):
    if model == 'conv':
        encoder, decoder = SimpleEncoder(), SimpleDecoder()
    elif 'vgg' in model:
        encoder, decoder = VGGEncoder(model, args.fea_c), VGGDecoder(model, args.fea_c)
    else:
        print('Model not found! Use "conv" instead.')
        encoder, decoder = SimpleEncoder(), SimpleDecoder()
    ae = AutoEncoder(encoder, decoder)
    return ae


def getDataLoader(args):
    dataset = getDataset(args)
    return Data.DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, **kwargs)


class AutoEncoder(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super(AutoEncoder, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return encode, decode


def train():
    start_time = time.time()
    model_name = 'model/AE_%s%s_model-%s.pkl' % (args.model, '' if args.fea_c is None else args.fea_c, args.dataset)
    pic_dir = 'res/AE_%s%s-%s/' % (args.model, '' if args.fea_c is None else args.fea_c, args.dataset)
    if os.path.exists(model_name) and args.load_model:
        print('Loading model ...')
        autoencoder = torch.load(model_name).to(device)
    else:
        autoencoder = init_model(args.model).to(device)

    train_loader = getDataLoader(args)
    optimizer = torch.optim.Adam(list(autoencoder.parameters()), lr=args.lr)
    loss_func = nn.MSELoss()

    check_dir_exists(['res/', 'model', pic_dir])

    for epoch in range(args.epoch):
        step_time = time.time()
        for step, (x, y) in enumerate(train_loader):
            # b_x = Variable(x.view(-1, 3, HEIGHT, WEIGHT))  # batch x, shape (batch, 32*32*3)
            b_x = Variable(x).cuda() if cuda else Variable(x)
            b_y = b_x.detach().cuda() if cuda else b_x.detach()  # batch y, shape (batch, 32*32*3)

            encoded, decoded = autoencoder(b_x)

            if step % 100 == 0:
                img_to_save = decoded.data
                save_image(img_to_save, '%s/%s-%s.jpg' % (pic_dir, epoch, step))
            # io.imsave('.xxx.jpg',img_to_save[0])

            loss = loss_func(decoded, b_y)  # mean square error
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()

            if step % 10 == 0:
                torch.save(autoencoder, model_name)
                print('Epoch:', epoch, 'Step:', step, '|', 'train loss %.6f; Time cost %.2f s'
                      % (loss.data[0], time.time() - step_time))
                step_time = time.time()
    print('Finished. Totally cost %.2f' % (time.time() - start_time))


def evaluate():
    model_name = 'model/AE_%s%s_model-%s.pkl' % (args.model, '' if args.fea_c is None else args.fea_c, args.dataset)
    assert os.path.exists(model_name)
    print('Loading model ...')
    autoencoder = torch.load(model_name).to(device)
    train_loader = getDataLoader(args)

    step_time = time.time()
    for step, (x, y) in enumerate(train_loader):
        b_x = Variable(x).cuda() if cuda else Variable(x)

        encoded, _ = autoencoder(b_x)

        if step % 1 == 0:
            img_to_save = encoded.data
            print(img_to_save)

    # print('Finished. Totally cost %.2f' % (time.time() - start_time))


if __name__ == '__main__':
    train()
    # evaluate()

