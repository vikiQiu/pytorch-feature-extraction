import os
import torch
import time
import json
import numpy as np
import torch.utils.data as Data
from torch.autograd import Variable
# import torchsummary
import torch.nn as nn
from torchvision.utils import save_image
from data_process import getDataLoader
from utils.arguments import train_args
from utils.utils import check_dir_exists
from model import SimpleEncoder, SimpleDecoder, VGGEncoder, VGGDecoder


def init_model(model, args):
    if model == 'conv':
        encoder, decoder = SimpleEncoder(), SimpleDecoder()
    elif 'vgg' in model:
        encoder, decoder = VGGEncoder(model, args.fea_c), VGGDecoder(model, args.fea_c)
    else:
        print('Model not found! Use "conv" instead.')
        encoder, decoder = SimpleEncoder(), SimpleDecoder()
    ae = AutoEncoder(encoder, decoder)
    return ae


class AutoEncoder(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super(AutoEncoder, self).__init__()

        self.encoder, _ = encoder
        self.decoder = decoder

    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return encode, decode

    def encode(self, x):
        return self.encoder(x)


def train():
    ################################################################
    # Arguments
    ################################################################
    ae_args = train_args()
    cuda = ae_args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if ae_args.cuda else {}
    # global ae_args, cuda, device, kwargs

    start_time = time.time()
    args = ae_args
    model_name = 'model/AE_%s%s_model-%s.pkl' % (args.model, '' if args.fea_c is None else args.fea_c, args.dataset)
    pic_dir = 'res/AE_%s%s-%s/' % (args.model, '' if args.fea_c is None else args.fea_c, args.dataset)
    if os.path.exists(model_name) and args.load_model:
        print('Loading model ...')
        autoencoder = torch.load(model_name).to(device)
    else:
        autoencoder = init_model(args.model, args).to(device)

    train_loader = getDataLoader(args, kwargs)
    optimizer = torch.optim.Adam(list(autoencoder.parameters()), lr=args.lr)
    loss_func = nn.MSELoss()

    check_dir_exists(['res/', 'model', pic_dir])
    loss_val = None

    for epoch in range(args.epoch):
        step_time = time.time()
        for step, (x, y) in enumerate(train_loader):
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

            loss_val = 0.99*loss_val + 0.01*loss.data[0] if loss_val is not None else loss.data[0]

            if step % 10 == 0:
                torch.save(autoencoder, model_name)
                print('Epoch:', epoch, 'Step:', step, '|', 'train loss %.6f; Time cost %.2f s'
                      % (loss.data[0], time.time() - step_time))
                step_time = time.time()
    print('Finished. Totally cost %.2f' % (time.time() - start_time))


if __name__ == '__main__':
    train()
    # model_name = 'AE_%s%s_model-%s' % (args.model, '' if args.fea_c is None else args.fea_c, args.dataset)
    # evaluate_pic('similar_pic/%s/' % model_name, model_name)
