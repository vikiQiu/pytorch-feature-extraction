import os
import torch
import time
import numpy as np
from torch.autograd import Variable
# import torchsummary
import torchvision.models as models
import torch.nn as nn
from data_process import getDataLoader
from utils.arguments import train_args
from utils.utils import check_dir_exists, evaluate_labeled_data


class ResNet(torch.nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        mol = models.resnet50(pretrained=True)

        self.features = nn.Sequential(*list(mol.children())[:-2])
        self.avg_pool = list(mol.children())[-2]
        self.fc = list(mol.children())[-1]

    def forward(self, x):
        fea = self.features(x)
        fea = self.avg_pool(fea)
        fea = fea.view(x.size(0), -1)
        c = self.fc(fea)
        return c

    def get_encode_features(self, x):
        fea = self.features(x)
        fea = fea.view(x.size(0), -1)
        return fea

    def get_fc_features(self, x, return_both=False):
        fea = self.features(x)
        fc = self.avg_pool(fea)
        fc = fc.view(fc.size(0), -1)
        if return_both:
            return fc, fc
        else:
            return fc


def train(mol_short='resnet50'):
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
    model_name = 'model/%s_%s%s_model-%s.pkl' % (mol_short, args.model, '' if args.fea_c is None else args.fea_c, args.dataset)
    evaluation_dir = 'res/evaluation_pic/%s_%s%s-%s' % (mol_short, args.model, '' if args.fea_c is None else args.fea_c, args.dataset)
    if os.path.exists(model_name) and args.load_model:
        print('Loading model ...')
        mol = torch.load(model_name).to(device)
    else:
        print('Init model ...')
        mol = ResNet().to(device)

    # train_loader = getDataLoader(args, kwargs)
    test_loader = getDataLoader(args, kwargs, train='test')
    # cover_loader = getDataLoader(args, kwargs, train='cover')
    cover_val_loader = getDataLoader(args, kwargs, train='cover_validation')
    cover_sample_loader = getDataLoader(args, kwargs, train='cover_sample')

    loss_class = nn.CrossEntropyLoss().cuda(cuda)

    check_dir_exists(['res/', 'model', 'res/evaluation_pic', evaluation_dir])

    # Evaluation
    check_dir_exists([os.path.join(evaluation_dir, 'cos'), os.path.join(evaluation_dir, 'distance')])
    # evaluate_cover(cover_val_loader, cover_sample_loader, mol, cuda, evaluation_dir)
    encode_accuracy, encode_top5accuracy, fc_accuracy, fc_top5accuracy = evaluate_labeled_data(test_loader, mol, cuda)
    print('Encode accuracy:', np.mean(encode_accuracy))
    print('Encode top5 accuracy:', np.mean(encode_top5accuracy))
    print('Fc accuracy:', np.mean(fc_accuracy))
    print('Fc top5 accuracy:', np.mean(fc_top5accuracy))

    total, correct, top5correct, loss_total = 0, 0, 0, 0
    for epoch in range(1):
        step_time = time.time()
        for step, (x, y) in enumerate(test_loader):
            b_x = Variable(x, volatile=True).cuda() if cuda else Variable(x)
            label = Variable(torch.Tensor([y[2][i] for i in range(len(y[0]))]).long())
            label = label.cuda() if cuda else label

            prob_class = mol(b_x)

            loss = loss_class(prob_class, label) # mean square error

            _, predicted = torch.max(prob_class.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
            top5pre = prob_class.topk(5, 1, True, True)
            top5pre = top5pre[1].t()
            top5correct += top5pre.eq(label.view(1, -1).expand_as(top5pre)).sum().item()

            loss_total += loss.data[0]*label.size(0)

            if step % 10 == 0:
                torch.save(mol, model_name)
                print('Epoch:', epoch, 'Step:', step, '|',
                      'test loss %.6f; Time cost %.2f s; Classification error %.6f; '
                      'Top1 Accuracy %.3f; Top5 Accuracy %.3f' %
                      (loss_total/total, time.time() - step_time, loss, correct*100/total, top5correct*100/total))
                step_time = time.time()
    print('Finished. Totally cost %.2f' % (time.time() - start_time))


if __name__ == '__main__':
    train()