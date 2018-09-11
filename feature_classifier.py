import os
import torch
import time
import shutil
from torch.autograd import Variable
import torch.nn as nn
from data_process import getDataLoader
from utils.utils import check_dir_exists
from utils.arguments import feature_classifier_args
from Autoencoder import AutoEncoder
from VAE import VAE
from vgg_classifier import VGGNet
from AE_class import AEClass


class FClassifier(torch.nn.Module):
    def __init__(self, fea_mol, main_model, encode_channels=32, num_class=1000):
        super(FClassifier, self).__init__()

        self.fea_mol = fea_mol
        self.main_model = main_model

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
        if self.main_model == 'AE':
            feature, _ = self.fea_mol(x)
        elif self.main_model == 'VAE':
            _, feature, _ = self.fea_mol(x)
            # feature = torch.cat([mu, std])
        elif self.main_model == 'AEClass':
            feature, _, _ = self.fea_mol(x)

        c = feature.view(x.size(0), -1)
        c = self.classification(c)
        return c

    def encode(self, x):
        return self.encoder(x)


def train():
    ################################################################
    # Arguments
    ################################################################
    ae_args = args = feature_classifier_args()
    cuda = ae_args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if ae_args.cuda else {}

    ################################################################
    # Load or Init model
    ################################################################
    start_time = time.time()
    model_name = 'model/FeaClass_%s_%s%s_model-%s.pkl' \
                 % (args.main_model, args.model, '' if args.fea_c is None else args.fea_c, args.dataset)
    if os.path.exists(model_name) and args.load_model:
        print('Loading model ...')
        mol = torch.load(model_name).to(device)
    else:
        print('Init model ...')

        feature_name = 'model/%s_%s%s_model-%s.pkl' % (
            args.main_model, args.model, '' if args.fea_c is None else args.fea_c, args.dataset)
        print('Loading feature model (%s)...' % feature_name)
        assert os.path.exists(feature_name)
        if cuda:
            fea_mol = torch.load(feature_name).to(device)
        else:
            fea_mol = torch.load(feature_name, map_location='cpu').to(device)
        mol = FClassifier(fea_mol, args.main_model, args.fea_c).to(device)

    train_loader = getDataLoader(args, kwargs)
    optimizer = torch.optim.Adam(list(mol.classification.parameters()), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss().cuda(cuda)

    check_dir_exists(['res/', 'model'])

    total, correct, top5correct = 0, 0, 0
    for epoch in range(args.epoch):
        step_time = time.time()
        for step, (x, y) in enumerate(train_loader):
            b_x = Variable(x).cuda() if cuda else Variable(x)
            label = Variable(torch.Tensor([y[2][i] for i in range(len(y[0]))]).long())
            label = label.cuda() if cuda else label

            prob_class = mol(b_x)

            loss = loss_fn(prob_class, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(prob_class.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
            top5pre = prob_class.topk(5, 1, True, True)
            top5pre = top5pre[1].t()
            top5correct += top5pre.eq(label.view(1, -1).expand_as(top5pre)).sum().item()

            if step % 10 == 0:
                shutil.copy2(model_name, model_name.split('.pkl')[0]+'_back.pkl')
                torch.save(mol, model_name)
                print('Epoch:', epoch, 'Step:', step, '|',
                      'train loss %.6f; Time cost %.2f s; Classification error %.6f'
                      'Accuracy %.3f%%; Top5 Accuracy %.3f%%' %
                      (loss.data[0], time.time() - step_time, loss, correct*100/total, top5correct*100/total))
                correct, total, top5correct = 0, 0, 0
                step_time = time.time()
    print('Finished. Totally cost %.2f' % (time.time() - start_time))


if __name__ == '__main__':
    train()
    pass
