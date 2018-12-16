import os
import time
import torch
import logging
import torch.nn as nn
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from utils.arguments import train_args
from data_process import getDataLoader
from base_model.model_inception import inception_v3_features, inception_v3
from utils.utils import check_dir_exists, evaluate_cover, evaluate_labeled_data, GPU, log_print


class InceptionFinetuneModel:
    def __init__(self, name='inception_finetune_experiments'):
        self.args = train_args()

        self._get_gpu()
        self.cuda = torch.cuda.is_available() and self.args.gpu != -2
        self.device = torch.device("cuda" if self.cuda else "cpu")
        self.kwargs = {'num_workers': 6, 'pin_memory': True}

        self.mol_dir = 'model'

        self.log_dir = os.path.join('log')
        self.log_file = os.path.join(self.log_dir, 'log_%s.log' % name)
        self.summary_dir = os.path.join('summary', self.args.name)
        check_dir_exists([self.mol_dir, self.log_dir, os.path.dirname(self.summary_dir), self.summary_dir])
        self.writer = SummaryWriter(self.summary_dir)
        logging.basicConfig(filename=self.log_file, level=logging.INFO)

    def _get_gpu(self):
        if self.args.gpu==-1:
            gpu = GPU().choose_gpu()
            if int(gpu) >= 0:
                log_print('Using the automatically choosed GPU %s' % gpu)
                os.environ["CUDA_VISIBLE_DEVICES"] = gpu
            else:
                log_print('No free GPU, using CPU instead.')

    def _load_raw_model(self):
        mol = inception_v3(pretrained=True, training=False).to(self.device)
        return mol

    def _load_finetune_model(self, mol_path):
        if os.path.exists(mol_path):
            print('Loading model ...')
            mol = torch.load(mol_path).to(self.device)
        else:
            mol = inception_v3(pretrained=True, training=False).to(self.device)
        return mol

    def _transform_cuda(self, x):
        return x.cuda() if self.cuda else x

    def finetune_model(self, bn_train=False):
        finetune_model_name = 'incpetion_finetune_%s' % ('bn_train' if bn_train else 'bn_not_train')
        finetune_mol_path = os.path.join(self.mol_dir, finetune_model_name + '.pkl')
        mol = self._load_finetune_model(finetune_mol_path)

        test_loader = getDataLoader(self.args, self.kwargs, train='test')
        total, correct, top5correct, loss_total = 0, 0, 0, 0
        loss_class = nn.CrossEntropyLoss().cuda(self.cuda)
        return

    def raw_model(self, is_train=True):
        mol = self._load_raw_model()
        test_loader = getDataLoader(self.args, self.kwargs, train='test')
        total, correct, top5correct, loss_total = 0, 0, 0, 0
        loss_class = nn.CrossEntropyLoss().cuda(self.cuda)

        step_time = time.time()
        if is_train:
            mol.train()
        else:
            mol.eval()
        log_print('############## %s Mode #################' % ('Training' if is_train else 'Evaluating'))
        for step, (x, y) in enumerate(test_loader):
            b_x = self._transform_cuda(Variable(x, volatile=True))
            label = Variable(torch.Tensor([y[2][i] for i in range(len(y[0]))]).long())
            label = self._transform_cuda(label)

            prob_class = mol(b_x)[0] if is_train else mol(b_x)
            loss = loss_class(prob_class, label)  # mean square error

            _, predicted = torch.max(prob_class.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
            top5pre = prob_class.topk(5, 1, True, True)
            top5pre = top5pre[1].t()
            top5correct += top5pre.eq(label.view(1, -1).expand_as(top5pre)).sum().item()

            loss_total += loss.data[0] * label.size(0)

            if step % 10 == 0:
                log_print('%s mode Step: %d | test loss %.6f; Time cost %.2f s; Classification error %.6f; '
                          'Top1 Accuracy %.3f; Top5 Accuracy %.3f' %
                          ('train' if is_train else 'eval', step, loss_total / total, time.time() - step_time,
                           loss, correct * 100 / total, top5correct * 100 / total))
                step_time = time.time()

        log_print('%s mode final result: Loss %.3f; Top1 accuracy %.3f; Top5 Accuracy %.3f' %
                  ('Train' if is_train else 'Evaluation', loss_total / total,
                   correct * 100 / total, top5correct * 100 / total))


if __name__ == '__main__':
    icp = InceptionFinetuneModel()
    # icp.raw_model()
    icp.raw_model(False)
