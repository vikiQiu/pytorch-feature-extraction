import os
import torch
import time
import json
import shutil
import numpy as np
import torch.utils.data as Data
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import torch.nn as nn
from torchvision.utils import save_image
from data_process import getDataLoader
from utils.arguments import train_args
from utils.utils import check_dir_exists, cal_cos, cal_accuracy
from model import SimpleEncoder, SimpleDecoder, VGGEncoder, VGGDecoder, VGG16Feature


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
    def __init__(self, encode_channels=32):
        super(AutoEncoder, self).__init__()

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

    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

    def encode(self, x):
        encoded = self.features(x)
        encoded = self.small_features(encoded)
        return encoded


def get_feature_loss(data_loader, mol, cuda):
    features = []
    labels = []
    loss = 0
    loss_decoder = nn.MSELoss()
    for step, (x, y) in enumerate(data_loader):
        b_x = Variable(x).cuda() if cuda else Variable(x)
        b_y = b_x.detach().cuda() if cuda else b_x.detach()
        label = [(y[0][i], y[1][i]) for i in range(len(y[0]))]
        labels.extend(label)

        feature, decoded = mol(b_x)
        loss += loss_decoder(decoded, b_y).data[0]

        f = feature.cpu() if cuda else feature
        f = f.data.view(b_x.shape[0], -1).numpy().tolist()
        features.extend(f)

        if step % 50 == 0:
            print('Step %d finished.' % step)
    return features, labels, loss


def test(test_loader, mol, cuda, name):
    test_time = time.time()
    print('#### Start %s testing with %d batches ####' % (name, len(test_loader)))

    feature, labels, loss = get_feature_loss(test_loader, mol, cuda)
    similar_mat = cal_cos(feature)
    accuracy, _ = cal_accuracy(similar_mat, labels, topk=1)
    top5accuracy, _ = cal_accuracy(similar_mat, labels, topk=5)
    print('[Testing] Feature accuracy = %.5f%%; top5 accuracy = %.5f%%; time cost %.2fs'
          % (accuracy, top5accuracy, time.time() - test_time))
    return accuracy, top5accuracy, loss


def train():
    ################################################################
    # Arguments
    ################################################################
    ae_args = train_args()
    cuda = ae_args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if ae_args.cuda else {}
    # global ae_args, cuda, device, kwargs
    args = ae_args

    log_dir = 'log/log_AE_%s%s_model-%s/' %\
              (args.model, '' if args.fea_c is None else args.fea_c, args.dataset)
    writer = SummaryWriter(log_dir)

    start_time = time.time()
    model_name = 'model/AE_%s%s_model-%s.pkl' % (args.model, '' if args.fea_c is None else args.fea_c, args.dataset)
    pic_dir = 'res/AE_%s%s-%s/' % (args.model, '' if args.fea_c is None else args.fea_c, args.dataset)
    if os.path.exists(model_name) and args.load_model:
        print('Loading model ...')
        mol = torch.load(model_name).to(device)
    else:
        print('Init model ...')
        mol = AutoEncoder(args.fea_c).to(device)

    print('Prepare data loader ...')
    train_loader = getDataLoader(args, kwargs, train=True)
    test_loader = getDataLoader(args, kwargs, train=False)

    optimizer1 = torch.optim.Adam(list(mol.small_features.parameters())+list(mol.decoder.parameters()), lr=args.lr)
    optimizer2 = torch.optim.Adam(list(mol.features.parameters()), lr=args.lr/10)
    loss_decoder = nn.MSELoss()

    check_dir_exists(['res/', 'model', pic_dir, log_dir])

    total, correct, top5correct, cnt = 0, 0, 0, 0
    print('Start training ...')
    for epoch in range(args.epoch):
        # Testing
        test_acc, test_top5acc, test_loss = test(test_loader, mol, cuda, 'Full')
        writer.add_scalar('test/accuracy', test_acc, epoch)
        writer.add_scalar('test/top5accuracy', test_top5acc, epoch)
        writer.add_scalar('test/loss_decoder', test_loss, epoch)

        step_time = time.time()
        for step, (x, y) in enumerate(train_loader):
            b_x = Variable(x).cuda() if cuda else Variable(x)
            b_y = b_x.detach().cuda() if cuda else b_x.detach()  # batch y, shape (batch, 32*32*3)

            encoded, decoded = mol(b_x)

            if step % 100 == 0:
                img_to_save = decoded.data
                save_image(img_to_save, '%s/%s-%s.jpg' % (pic_dir, epoch, step))

            loss = loss_decoder(decoded, b_y)
            writer.add_scalar('train/loss_decoder', loss, cnt)

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            loss.backward()
            optimizer1.step()
            optimizer2.step()

            if step % 50 == 0:
                if os.path.exists(model_name):
                    shutil.copy2(model_name, model_name.split('.pkl')[0]+'_back.pkl')
                torch.save(mol, model_name)
                print('[Training] Epoch:', epoch, 'Step:', step, '|',
                      'train loss %.6f; Time cost %.2f s; Decoder error %.6f'%
                      (loss.data[0], time.time() - step_time, loss))
                step_time = time.time()

            cnt += 1

    print('Finished. Totally cost %.2f' % (time.time() - start_time))
    writer.export_scalars_to_json(os.path.join(log_dir, 'all_scalars.json'))
    writer.close()


if __name__ == '__main__':
    train()
    # model_name = 'AE_%s%s_model-%s' % (args.model, '' if args.fea_c is None else args.fea_c, args.dataset)
    # evaluate_pic('similar_pic/%s/' % model_name, model_name)
