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
from utils.utils import check_dir_exists, evaluate_cover, evaluate_labeled_data
from model import VGG16Feature, VGG16Classifier


class VGGNet(torch.nn.Module):
    def __init__(self, encode_channels=32, num_class=1000):
        super(VGGNet, self).__init__()

        self.encode_channels = encode_channels
        self.features = VGG16Feature()
        self.classification = VGG16Classifier()

    def forward(self, x):
        fea = self.features(x)
        c = fea.view(x.size(0), -1)
        c = self.classification(c)
        return c

    def get_encode_features(self, x):
        fea = self.features(x)
        fea = fea.view(x.size(0), -1)
        return fea

    def get_fc_features(self, x, return_both=False):
        fea = self.get_encode_features(x)
        fc = self.classification.get_feature(fea)
        if return_both:
            return fea, fc
        else:
            return fea


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
    model_name = 'model/vgg_classifier_%s%s_model-%s.pkl' % (args.model, '' if args.fea_c is None else args.fea_c, args.dataset)
    evaluation_dir = 'res/evaluation_pic/vgg_%s%s-%s' % (args.model, '' if args.fea_c is None else args.fea_c, args.dataset)
    if os.path.exists(model_name) and args.load_model:
        print('Loading model ...')
        vgg = torch.load(model_name).to(device)
    else:
        print('Init model ...')
        vgg = VGGNet(args.fea_c).to(device)

    # train_loader = getDataLoader(args, kwargs)
    test_loader = getDataLoader(args, kwargs, train='test')
    # cover_loader = getDataLoader(args, kwargs, train='cover')
    cover_val_loader = getDataLoader(args, kwargs, train='cover_validation')
    cover_sample_loader = getDataLoader(args, kwargs, train='cover_sample')

    optimizer = torch.optim.Adam(list(vgg.parameters()), lr=args.lr)
    loss_class = nn.CrossEntropyLoss().cuda(cuda)

    check_dir_exists(['res/', 'model', 'res/evaluation_pic', evaluation_dir])

    # Evaluation
    check_dir_exists([os.path.join(evaluation_dir, 'cos'), os.path.join(evaluation_dir, 'distance')])
    evaluate_cover(cover_val_loader, cover_sample_loader, vgg, cuda, evaluation_dir)
    encode_accuracy, encode_top5accuracy, fc_accuracy, fc_top5accuracy = evaluate_labeled_data(test_loader, vgg, cuda)
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

            prob_class = vgg(b_x)

            loss = loss_class(prob_class, label) # mean square error
            # optimizer.zero_grad()  # clear gradients for this training step
            # loss.backward()
            # optimizer.step()

            _, predicted = torch.max(prob_class.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
            top5pre = prob_class.topk(5, 1, True, True)
            top5pre = top5pre[1].t()
            top5correct += top5pre.eq(label.view(1, -1).expand_as(top5pre)).sum().item()

            loss_total += loss.data[0]*label.size(0)

            if step % 10 == 0:
                torch.save(vgg, model_name)
                print('Epoch:', epoch, 'Step:', step, '|',
                      'test loss %.6f; Time cost %.2f s; Classification error %.6f; '
                      'Top1 Accuracy %.3f; Top5 Accuracy %.3f' %
                      (loss_total/total, time.time() - step_time, loss, correct*100/total, top5correct*100/total))
                step_time = time.time()
    print('Finished. Totally cost %.2f' % (time.time() - start_time))


if __name__ == '__main__':
    train()