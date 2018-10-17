import os
import torch
import time
import shutil
import numpy as np
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import torch.nn as nn
from data_process import getDataLoader
from utils.arguments import train_args
from utils.utils import check_dir_exists, evaluate_cover, evaluate_labeled_data, remove_dir_exists
from base_model.model_inception import inception_v3_features, Inception3Class


def get_main_function(main_fn):
    if main_fn == 'train_cls':
        return train_cls
    elif main_fn == 'train_decoder':
        return train_decoder
    elif main_fn == 'train':
        return train
    else:
        return


def get_mol_short(model):
    if model == 'inception':
        return 'IcpV3'


class ModelCls(torch.nn.Module):
    def __init__(self, args):
        super(ModelCls, self).__init__()
        if args.model == 'inception':
            self.features = inception_v3_features(pretrained=True)
            self.classifier = Inception3Class()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def test_cls(test_loader, mol, cuda, name):
    # TODO:
    total, correct, top5correct = 0, 0, 0
    loss_class_fn = nn.CrossEntropyLoss().cuda(cuda)
    step_time = time.time()
    loss_cls = []
    print('#### Start %s testing with %d batches ####' % (name, len(test_loader)))

    for step, (x, y) in enumerate(test_loader):
        b_x = Variable(x, volatile=True).cuda() if cuda else Variable(x)
        label = Variable(torch.Tensor([y[2][i] for i in range(len(y[0]))]).long())
        label = label.cuda() if cuda else label

        prob_class = mol(b_x)
        loss_c = loss_class_fn(prob_class, label)
        loss_c.backward()
        loss_cls.append(loss_c.item())

        _, predicted = torch.max(prob_class.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()
        top5pre = prob_class.topk(5, 1, True, True)
        top5pre = top5pre[1].t()
        top5correct += top5pre.eq(label.view(1, -1).expand_as(top5pre)).sum().item()

        if step % 20 == 0:
            print('[%s Testing] Step: %d | Classification error %.6f; '
                  'Accuracy %.3f%%; Top5 Accuracy %.3f%%; Time cost %.2f s' %
                  (name, step, np.mean(loss_cls), correct * 100 / total, top5correct * 100 / total, time.time() - step_time))
            step_time = time.time()

    print('[%s Testing] #### Final Score ####: Test size %d; Accuracy %.3f%%; Top5 Accuracy %.3f%%; Time cost %.2f s' %
          (name, total, correct * 100 / total, top5correct * 100 / total, time.time() - step_time))
    return np.mean(loss_cls), correct/total, top5correct/total


def train_cls(args):
    ################################################################
    # Arguments
    ################################################################
    cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    ################################################################
    # Model name & Directory
    ################################################################
    mol_short = get_mol_short(args.model) + '(cls)'
    model_name = 'model/%s_%s_model.pkl' % (mol_short, args.model, )
    print('[Model] model name is', model_name)
    pic_dir = 'res/%s_%s%s-%s/' % (mol_short, args.model, '' if args.fea_c is None else args.fea_c, args.dataset)
    evaluation_dir = 'res/evaluation_pic/%s_%s%s-%s' % (
        mol_short, args.model, '' if args.fea_c is None else args.fea_c, args.dataset)
    log_dir = 'log/log_%s_%s%s_model-%s/' % \
              (mol_short, args.model, '' if args.fea_c is None else args.fea_c, args.dataset)
    remove_dir_exists([log_dir])
    check_dir_exists(['res/', 'model', pic_dir, log_dir, 'res/evaluation_pic', evaluation_dir])
    writer = SummaryWriter(log_dir)

    if os.path.exists(model_name) and args.load_model:
        print('Loading model ...')
        mol = torch.load(model_name).to(device)
    else:
        print('Init model ...')
        mol = ModelCls(args)

    ################################################################
    # Prepare Data & Optimizer
    ################################################################
    print('Prepare data loader ...')
    test_loader = getDataLoader(args, kwargs, train='test')
    train_loader = getDataLoader(args, kwargs, train='train')
    cover_val_loader = getDataLoader(args, kwargs, train='cover_validation')
    cover_sample_loader = getDataLoader(args, kwargs, train='cover_sample')

    optimizer = torch.optim.Adam(mol.classifier.parameters(), lr=args.lr)
    loss_class = nn.CrossEntropyLoss(reduction='none').cuda(cuda)

    total, correct, top5correct, cnt = 0, 0, 0, 0
    print('Start training ...')
    for epoch in range(args.epoch):
        ################################################################
        # Testing
        ################################################################
        if epoch > 0:
            mol.eval()
            # Testing on ImageNet val
            print('######### Testing on ImageNet val Dataset ###########')
            test_loss_cls, test_acc, test_top5acc = test_cls(test_loader, mol, cuda, 'Full')
            writer.add_scalar('test_imagenet/loss_classifier', test_loss_cls, epoch)
            writer.add_scalar('test_imagenet/accuracy', test_acc, epoch)
            writer.add_scalar('test_imagenet/top5accuracy', test_top5acc, epoch)

        if epoch % 5 == 4 or epoch == args.epoch-1:
            mol.eval()
            evaluate_labeled_data(test_loader, mol, cuda)

        ################################################################
        # Training
        ################################################################
        step_time = time.time()
        mol.train()
        for step, (x, y) in enumerate(train_loader):
            b_x = Variable(x).cuda() if cuda else Variable(x)
            label = Variable(torch.Tensor([y[2][i] for i in range(len(y[0]))]).long())
            label = label.cuda() if cuda else label

            prob_class = mol(b_x)
            loss = loss_class(prob_class, label)
            writer.add_scalar('train/loss_classifier', torch.mean(loss.data), cnt)

            optimizer.zero_grad()
            torch.mean(loss).backward()
            optimizer.step()

            _, predicted = torch.max(prob_class.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
            top5pre = prob_class.topk(5, 1, True, True)
            top5pre = top5pre[1].t()
            top5correct += top5pre.eq(label.view(1, -1).expand_as(top5pre)).sum().item()

            writer.add_scalar('train/accuracy', correct / total, cnt)
            writer.add_scalar('train/top5_accuracy', top5correct / total, cnt)

            if step % 10 == 0:
                if os.path.exists(model_name):
                    shutil.copy2(model_name, model_name.split('.pkl')[0] + '_back.pkl')
                torch.save(mol, model_name)
                total_tmp = total if total != 0 else 1
                print('[Training] Epoch:', epoch, 'Step:', step, '|',
                      'Time cost %.2f s; Classification error %.6f'
                      'Accuracy %.3f%%; Top5 Accuracy %.3f%%' %
                      (time.time() - step_time, loss.data[0], correct * 100 / total_tmp,
                       top5correct * 100 / total_tmp))
                correct, total, top5correct = 0, 0, 0
                step_time = time.time()

            cnt += 1
    pass


def train_decoder(args):
    pass


def train(args):
    pass


if __name__ == '__main__':
    args = train_args()
    get_main_function(args.main_fn)(args)
