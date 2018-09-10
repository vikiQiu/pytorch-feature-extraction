import os
import torch
import time
import json
import numpy as np
import torch.utils.data as Data
import torch.nn.functional as F
from torch.autograd import Variable
# import torchsummary
import torch.nn as nn
from torchvision.utils import save_image
from data_process import getDataLoader
from utils.arguments import train_args
from utils.utils import check_dir_exists
from model import SimpleEncoder, SimpleDecoder, VGGEncoder, VGGDecoder, VGG16Feature, VGG16Classifier


def init_model(model, args):
    if model == 'conv':
        encoder, decoder = SimpleEncoder(), SimpleDecoder()
    elif 'vgg' in model:
        encoder, decoder = VGGEncoder(model, args.fea_c), VGGDecoder(model, args.fea_c)
    else:
        print('Model not found! Use "conv" instead.')
        encoder, decoder = SimpleEncoder(), SimpleDecoder()
    ae = AEClass(encoder, decoder)
    return ae


class AEClass(torch.nn.Module):
    def __init__(self, encode_channels=32, num_class=1000):
        super(AEClass, self).__init__()

        self.encode_channels = encode_channels
        self.features = VGG16Feature()
        self.small_features = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, encode_channels, kernel_size=1),
            nn.BatchNorm2d(encode_channels),
            nn.ReLU(inplace=True)
        )
        self.decoder = VGGDecoder(model='vgg16', out_channels=encode_channels)
        self.classification = nn.Sequential(
            nn.Linear(encode_channels * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, num_class)
        )
        # self.classification = VGG16Classifier()

    def forward(self, x):
        fea = self.features(x)
        encode = self.small_features(fea)
        decode = self.decoder(encode)

        c = encode.view(x.size(0), -1)
        # c = fea.view(x.size(0), -1)
        c = self.classification(c)
        return encode, decode, c

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
    model_name = 'model/AEClass_%s%s_model-%s.pkl' % (args.model, '' if args.fea_c is None else args.fea_c, args.dataset)
    pic_dir = 'res/AEClass_%s%s-%s/' % (args.model, '' if args.fea_c is None else args.fea_c, args.dataset)
    if os.path.exists(model_name) and args.load_model:
        print('Loading model ...')
        autoencoder = torch.load(model_name).to(device)
    else:
        print('Init model ...')
        autoencoder = AEClass(args.fea_c).to(device)

    train_loader = getDataLoader(args, kwargs)
    optimizer1 = torch.optim.Adam(list(autoencoder.parameters()), lr=args.lr)
    optimizer2 = torch.optim.Adam(list(autoencoder.classification.parameters()), lr=args.lr)
    loss_decoder = nn.MSELoss()
    loss_class = nn.CrossEntropyLoss().cuda(cuda)

    check_dir_exists(['res/', 'model', pic_dir])
    loss_val = None

    total, correct, top5correct = 0, 0, 0
    for epoch in range(args.epoch):
        step_time = time.time()
        for step, (x, y) in enumerate(train_loader):
            b_x = Variable(x).cuda() if cuda else Variable(x)
            b_y = b_x.detach().cuda() if cuda else b_x.detach()  # batch y, shape (batch, 32*32*3)
            label = Variable(torch.Tensor([y[2][i] for i in range(len(y[0]))]).long())
            label = label.cuda() if cuda else label

            encoded, decoded, prob_class = autoencoder(b_x)

            if step % 100 == 0:
                img_to_save = decoded.data
                save_image(img_to_save, '%s/%s-%s.jpg' % (pic_dir, epoch, step))
            # io.imsave('.xxx.jpg',img_to_save[0])

            loss1 = loss_decoder(decoded, b_y)
            loss2 = loss_class(prob_class, label) # mean square error
            loss = loss2 + loss1

            if epoch % 4 != 0:
                optimizer1.zero_grad()
                loss1.backward()
                optimizer1.step()
            else:
                loss2.backward()
                optimizer2.step()

            _, predicted = torch.max(prob_class.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
            top5pre = prob_class.topk(5, 1, True, True)
            top5pre = top5pre[1].t()
            top5correct += top5pre.eq(label.view(1, -1).expand_as(top5pre)).sum().item()

            loss_val = 0.99*loss_val + 0.01*loss.data[0] if loss_val is not None else loss.data[0]

            if step % 10 == 0:
                torch.save(autoencoder, model_name)
                print('Epoch:', epoch, 'Step:', step, '|',
                      'train loss %.6f; Time cost %.2f s; Classification error %.6f; Decoder error %.6f; '
                      'Accuracy %.3f%%; Top5 Accuracy %.3f%%' %
                      (loss.data[0], time.time() - step_time, loss2, loss1, correct*100/total, top5correct*100/total))
                correct, total, top5correct = 0, 0, 0
                step_time = time.time()
    print('Finished. Totally cost %.2f' % (time.time() - start_time))


if __name__ == '__main__':
    train()
    # model_name = 'AE_%s%s_model-%s' % (args.model, '' if args.fea_c is None else args.fea_c, args.dataset)
    # evaluate_pic('similar_pic/%s/' % model_name, model_name)
