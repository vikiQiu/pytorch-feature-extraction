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
from torch.nn import functional as F
from model import SimpleEncoder, SimpleDecoder, VGGEncoder, VGGDecoder


def init_model(model, args):
    if model == 'conv':
        encoder, decoder = SimpleEncoder(), SimpleDecoder()
    elif 'vgg' in model:
        encoder, decoder = VGGEncoder(model, args.fea_c), VGGDecoder(model, args.fea_c)
    else:
        print('Model not found! Use "conv" instead.')
        encoder, decoder = SimpleEncoder(), SimpleDecoder()
    vae = VAE(encoder, decoder)
    return vae


class VAE(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()

        self.encoder = encoder
        self.encode_channel = encoder.get_out_channel()
        self.mu_layer = nn.Conv2d(self.encode_channel, self.encode_channel, kernel_size=1)
        self.std_layer = nn.Conv2d(self.encode_channel, self.encode_channel, kernel_size=1)
        self.decoder = decoder

    def forward(self, x):
        encoded = self.encoder(x)
        mu = self.mu_layer(encoded)
        std = self.std_layer(encoded)
        z = self.reparameterize(mu, std)
        decoded = self.decoder(z)
        return decoded, mu, std

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.sqrt(torch.exp(0.5 * logvar))
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu


def loss_function(recon_x, x, mu, std):
    BCE = F.mse_loss(recon_x, x)

    # KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    var_sum = torch.sum(std.pow(2), (1, 2, 3))
    KLD = 0.5 * (-torch.log(var_sum) + torch.sum(mu.pow(2), (1, 2, 3)) + var_sum)
    KLD = torch.mean(KLD)/(x.shape[1]*x.shape[2]*x.shape[3])

    return BCE + KLD, BCE, KLD


def train():
    ################################################################
    # Arguments
    ################################################################
    vae_args = train_args()
    cuda = vae_args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if vae_args.cuda else {}

    start_time = time.time()
    args = vae_args
    model_name = 'model/VAE_%s%s_model-%s.pkl' % (args.model, '' if args.fea_c is None else args.fea_c, args.dataset)
    pic_dir = 'res/VAE_%s%s-%s/' % (args.model, '' if args.fea_c is None else args.fea_c, args.dataset)
    if os.path.exists(model_name) and args.load_model:
        print('Loading model ...')
        if cuda:
            vae = torch.load(model_name).to(device)
        else:
            vae = torch.load(model_name, map_location='cpu')
    else:
        vae = init_model(args.model).to(device)

    train_loader = getDataLoader(args, kwargs)
    optimizer = torch.optim.Adam(list(vae.parameters()), lr=args.lr)

    check_dir_exists(['res/', 'model', pic_dir])
    loss_val = None

    for epoch in range(args.epoch):
        step_time = time.time()
        for step, (x, y) in enumerate(train_loader):
            b_x = Variable(x).cuda() if cuda else Variable(x)
            b_y = b_x.detach().cuda() if cuda else b_x.detach()  # batch y, shape (batch, 32*32*3)

            decoded, mu, std = vae(b_x)

            if step % 100 == 0:
                img_to_save = torch.cat([b_x.data, decoded.data])
                save_image(img_to_save, '%s/%s-%s.jpg' % (pic_dir, epoch, step))
            # io.imsave('.xxx.jpg',img_to_save[0])

            loss, bce, kld = loss_function(decoded, b_y, mu, std)  # mean square error
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()

            loss_val = 0.99*loss_val + 0.01*loss.data[0] if loss_val is not None else loss.data[0]

            if step % 10 == 0:
                torch.save(vae, model_name)
                print('Epoch:', epoch, 'Step:', step, '|', 'train loss %.6f; KLD %.6f; BCE %.6f; Time cost %.2f s'
                      % (loss_val, kld, bce, time.time() - step_time))
                step_time = time.time()
    print('Finished. Totally cost %.2f' % (time.time() - start_time))


if __name__ == '__main__':
    train()
    # model_name = 'AE_%s%s_model-%s' % (args.model, '' if args.fea_c is None else args.fea_c, args.dataset)
    # evaluate_pic('similar_pic/%s/' % model_name, model_name)
