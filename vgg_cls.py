import os
import torch
import time
import json
import shutil
import numpy as np
import torch.utils.data as Data
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.autograd import Variable
import torch.nn as nn
from torchvision.utils import save_image
from data_process import getDataLoader
from utils.arguments import train_args
from utils.utils import check_dir_exists, evaluate_cover, test, evaluate_labeled_data
from model import VGG16Feature


class VGGClass(torch.nn.Module):
    def __init__(self, encode_channels=32, num_class=1000):
        super(VGGClass, self).__init__()

        self.encode_channels = encode_channels
        self.features = VGG16Feature()
        self.small_features = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, encode_channels, kernel_size=1),
            nn.BatchNorm2d(encode_channels),
        )
        self.classification = nn.Sequential(
            nn.Linear(encode_channels * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, num_class)
        )

    def forward(self, x):
        fea = self.get_encode_features(x)
        c = self.classification(fea)
        return c

    def get_encode_features(self, x):
        fea = self.features(x)
        fea = self.small_features(fea)
        fea = fea.view(x.size(0), -1)
        return fea

    def get_fc_features(self, x, return_both=False):
        fea = self.get_encode_features(x)
        c = fea
        for name, layer in self.classification._modules.items():
            if int(name) <= 3:
                c = layer(c)
        if return_both:
            return fea, c
        else:
            return fea

    def get_prob_class(self, x):
        return self.forward(x)


def train(mol_short='VGGClass', main_model=VGGClass):
    ################################################################
    # Arguments
    ################################################################
    ae_args = train_args()
    cuda = ae_args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if ae_args.cuda else {}
    # global ae_args, cuda, device, kwargs
    args = ae_args

    log_dir = 'log/log_%s_%s%s_model-%s/' %\
              (mol_short, args.model, '' if args.fea_c is None else args.fea_c, args.dataset)
    writer = SummaryWriter(log_dir)

    start_time = time.time()
    model_name = 'model/%s_%s%s_model-%s.pkl' % (mol_short, args.model, '' if args.fea_c is None else args.fea_c, args.dataset)
    pic_dir = 'res/%s_%s%s-%s/' % (mol_short, args.model, '' if args.fea_c is None else args.fea_c, args.dataset)
    evaluation_dir = 'res/evaluation_pic/%s_%s%s-%s' % (mol_short, args.model, '' if args.fea_c is None else args.fea_c, args.dataset)
    if os.path.exists(model_name) and args.load_model:
        print('Loading model ...')
        mol = torch.load(model_name).to(device)
    else:
        print('Init model ...')
        mol = main_model(args.fea_c).to(device)

    print('Prepare data loader ...')
    train_loader = getDataLoader(args, kwargs)
    test_loader = getDataLoader(args, kwargs, train='test')
    # small_test_loader = getDataLoader(args, kwargs, train=False, p=10)
    cover_loader = getDataLoader(args, kwargs, train='cover')
    cover_sample_loader = getDataLoader(args, kwargs, train='cover_sample')

    # Optimizer & Loss function
    optimizer1 = torch.optim.Adam(list(mol.classification.parameters())+list(mol.small_features.parameters()),
                                  lr=args.lr)
    loss_class = nn.CrossEntropyLoss().cuda(cuda)
    loss_val = None

    # Check directories
    check_dir_exists(['res/', 'model', pic_dir, log_dir, 'res/evaluation_pic', evaluation_dir])

    total, correct, top5correct, cnt = 0, 0, 0, 0
    print('Start training ...')
    for epoch in range(args.epoch):
        # Evaluation cover
        if (epoch % 5 == 0) and epoch != 0:
            eval_dir = os.path.join(evaluation_dir, 'epoch%d' % epoch)
            evaluate_cover(cover_loader, cover_sample_loader, mol, cuda, eval_dir)

        # Testing classifier
        encode_accuracy, encode_top5accuracy, fc_accuracy, fc_top5accuracy = evaluate_labeled_data(test_loader, mol, cuda)
        writer.add_scalar('test/encode_feature_accuracy', np.mean(encode_accuracy), epoch)
        writer.add_scalar('test/encode_feature_top5accuracy', np.mean(encode_top5accuracy), epoch)
        writer.add_scalar('test/fc_feature_accuracy', np.mean(fc_accuracy), epoch)
        writer.add_scalar('test/fc_feature_top5accuracy', np.mean(fc_top5accuracy), epoch)
        test_acc, test_top5acc = test(test_loader, mol, cuda, 'Full')
        writer.add_scalar('test/class_accuracy', test_acc, epoch)
        writer.add_scalar('test/class_top5accuracy', test_top5acc, epoch)

        step_time = time.time()
        for step, (x, y) in enumerate(train_loader):
            b_x = Variable(x).cuda() if cuda else Variable(x)
            b_y = b_x.detach().cuda() if cuda else b_x.detach()  # batch y, shape (batch, 32*32*3)
            label = Variable(torch.Tensor([y[2][i] for i in range(len(y[0]))]).long())
            label = label.cuda() if cuda else label

            encoded, decoded, prob_class = mol(b_x)

            if step % 100 == 0:
                img_to_save = decoded.data
                save_image(img_to_save, '%s/%s-%s.jpg' % (pic_dir, epoch, step))

            loss = loss_class(decoded, b_y)
            writer.add_scalar('train/loss_classifier', loss, cnt)

            optimizer1.zero_grad()
            loss.backward()
            optimizer1.step()

            _, predicted = torch.max(prob_class.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
            top5pre = prob_class.topk(5, 1, True, True)
            top5pre = top5pre[1].t()
            top5correct += top5pre.eq(label.view(1, -1).expand_as(top5pre)).sum().item()
            writer.add_scalar('train/accuracy', correct/total, cnt)
            writer.add_scalar('train/top5_accuracy', top5correct/total, cnt)

            loss_val = 0.99*loss_val + 0.01*loss.data[0] if loss_val is not None else loss.data[0]

            if step % 10 == 0:
                if os.path.exists(model_name):
                    shutil.copy2(model_name, model_name.split('.pkl')[0]+'_back.pkl')
                torch.save(mol, model_name)
                print('[Training] Epoch:', epoch, 'Step:', step, '|',
                      'train loss %.6f; Time cost %.2f s; Accuracy %.3f%%; Top5 Accuracy %.3f%%' %
                      (loss.data[0], time.time() - step_time, correct*100/total, top5correct*100/total))
                correct, total, top5correct = 0, 0, 0
                step_time = time.time()

            cnt += 1

    print('Finished. Totally cost %.2f' % (time.time() - start_time))
    writer.export_scalars_to_json(os.path.join(log_dir, 'all_scalars.json'))
    writer.close()


if __name__ == '__main__':
    train()
    pass
