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


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='ImageNet1000-val',
                    help="Dataset to train. Now support ['ImageNet1000-val'].")
parser.add_argument("--dataset-dir", type=str, default='..\data\ILSVRC2012\\', help="Dataset Directory")
# parser.add_argument("--img-transform", type=str, default='default', help="Image Transformer")
parser.add_argument("--img-loader", type=str, default='default', help="Image Loader")
parser.add_argument("--epoch", type=int, default=600, help="Epoch number.")
parser.add_argument("--batch-size", type=int, default=256, help="Batch size.")
parser.add_argument("--lr", type=int, default=1e-4, help="Learning rate.")
parser.add_argument("--img-size", type=int, default=224, help="Height Weight of the training images after transform.")
parser.add_argument('--load-model', action="store_true", default=False)
parser.add_argument('--use-gpus', action='store_false', dest='cuda', default=True)

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


def getDataLoader(args):
    '''
    Now support ['ImageNet1000-val']。
    Add more dataset in future.
    '''
    if args.dataset == 'ImageNet1000-val':
        label_dir = os.path.join(args.dataset_dir, 'ILSVRC2012_bbox_val_v3')
        img_dir= os.path.join(args.dataset_dir, 'ILSVRC2012_img_val')
        dataset = ImageNetDataset(img_dir, label_dir,
                                  img_transform=transformers['crop'+str(args.img_size)],
                                  loader=loaders[args.img_loader])
        return Data.DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    pass


class AutoEncoder(torch.nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

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
            ), # 16*16*64->8*8*128
            nn.BatchNorm2d(num_features=128, affine=True),  # num_features=batch_size x num_features [x width]
            nn.LeakyReLU(),

            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=2,
                padding=1,
            ), # 8*8*128->4*4*256 Total 4096 feature
            nn.BatchNorm2d(num_features=256, affine=True),
            nn.LeakyReLU(),
        )

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
        # x = x.view(-1, 3, HEIGHT, WEIGHT)
        encode = self.encoder(x)

        decode = self.decoder(encode)
        return encode, decode


def train():
    start_time = time.time()
    model_name = 'model/AE_model-%s.pkl' % args.dataset
    if os.path.exists(model_name) and args.load_model:
        print('Loading model ...')
        autoencoder = torch.load(model_name).to(device)
    else:
        autoencoder = AutoEncoder().to(device)

    train_loader = getDataLoader(args)
    optimizer = torch.optim.Adam(list(autoencoder.parameters()), lr=args.lr)
    loss_func = nn.MSELoss()

    check_dir_exists(['res/', 'model', 'res/AE-'+args.dataset])

    for epoch in range(args.epoch):
        for step, (x, y) in enumerate(train_loader):
            step_time = time.time()
            # b_x = Variable(x.view(-1, 3, HEIGHT, WEIGHT))  # batch x, shape (batch, 32*32*3)
            b_x = Variable(x).cuda() if cuda else Variable(x)
            b_y = b_x.detach().cuda() if cuda else b_x.detach()  # batch y, shape (batch, 32*32*3)

            encoded, decoded = autoencoder(b_x)

            if step % 100 == 0:
                img_to_save = decoded.data
                save_image(img_to_save, 'res/AE-%s/%s-%s.jpg' % (args.dataset, epoch, step))
            # io.imsave('.xxx.jpg',img_to_save[0])

            # print('wwwwww')
            # print(type(decoded))
            # print(type(b_y))
            # b_y.type_as(decoded)
            loss = loss_func(decoded, b_y)  # mean square error
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()

            if step % 10 == 0:
                torch.save(autoencoder, model_name)
                print('Epoch:', epoch, 'Step:', step, '|', 'train loss %.6f; Time cost %.2f s'
                      % (loss.data[0], time.time() - step_time))
    print('Finished. Totally cost %.2f' % time.time() - start_time)


if __name__ == '__main__':
    train()

