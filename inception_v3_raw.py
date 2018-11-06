import os
import torch
import time
import json
import numpy as np
from torch.autograd import Variable
# import torchsummary
import torch.nn as nn
from data_process import getDataLoader
from utils.arguments import train_args
from utils.utils import check_dir_exists, evaluate_cover, evaluate_labeled_data, read_imagenet_label_name
from base_model.model_inception import inception_v3_features, inception_v3


def get_main_function(main_fn):
    if main_fn == 'train':
        return train
    elif main_fn == 'get_features':
        return get_features
    elif main_fn == 'cover_label':
        return sample_cover_label
    else:
        return train


def get_features(mol_short='inception_v3'):
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
    model_name = 'model/%s_%s%s_model-%s.pkl' % (
    mol_short, args.model, '' if args.fea_c is None else args.fea_c, args.dataset)
    evaluation_dir = 'res/evaluation_pic/%s_%s%s-%s' % (
    mol_short, args.model, '' if args.fea_c is None else args.fea_c, args.dataset)
    if os.path.exists(model_name) and args.load_model:
        print('Loading model ...')
        mol = torch.load(model_name).to(device)
    else:
        print('Init model ...')
        mol = inception_v3_features(pretrained=True, training=False).to(device)

    # train_loader = getDataLoader(args, kwargs)
    test_loader = getDataLoader(args, kwargs, train='test')

    check_dir_exists(['res/', 'model', 'res/evaluation_pic', evaluation_dir])
    t_per_img = []
    for epoch in range(1):
        step_time = time.time()
        for step, (x, y) in enumerate(test_loader):
            b_x = Variable(x, volatile=True).cuda() if cuda else Variable(x)

            t0 = time.time()
            features = mol(b_x)
            t_tmp = (time.time() - t0) / len(b_x) * 1000
            t_per_img.append(t_tmp)
            print('cost %.6fms per image this batch. cost %.6fms per image till now.' % (t_tmp, np.mean(sorted(t_per_img)[1:-1])))

    print('Finished. Totally cost %.2f' % (time.time() - start_time))


def train(mol_short='inception_v3'):
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
        mol = inception_v3(pretrained=True, training=False).to(device)
    mol.eval()

    # train_loader = getDataLoader(args, kwargs)
    test_loader = getDataLoader(args, kwargs, train='test')
    # cover_loader = getDataLoader(args, kwargs, train='cover')
    cover_val_loader = getDataLoader(args, kwargs, train='cover_validation')
    cover_sample_loader = getDataLoader(args, kwargs, train='cover_sample')

    loss_class = nn.CrossEntropyLoss().cuda(cuda)

    check_dir_exists(['res/', 'model', 'res/evaluation_pic', evaluation_dir])

    # Evaluation
    # check_dir_exists([os.path.join(evaluation_dir, 'cos'), os.path.join(evaluation_dir, 'distance')])
    # evaluate_cover(cover_sample_loader, cover_sample_loader, mol, cuda, evaluation_dir, args)
    # encode_accuracy, encode_top5accuracy, fc_accuracy, fc_top5accuracy = evaluate_labeled_data(test_loader, mol, cuda)
    # print('Encode accuracy:', np.mean(encode_accuracy))
    # print('Encode top5 accuracy:', np.mean(encode_top5accuracy))
    # print('Fc accuracy:', np.mean(fc_accuracy))
    # print('Fc top5 accuracy:', np.mean(fc_top5accuracy))

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


def sample_cover_label(mol_short='inception_v3'):

    ################################################################
    # Arguments
    ################################################################
    args = train_args()
    cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    start_time = time.time()
    out_path = 'res/evaluation_pic/sample_cover_label_%s.json' % mol_short
    print('Init model ...')
    mol = inception_v3(pretrained=True, training=False).to(device)
    cover_sample_loader = getDataLoader(args, kwargs, train='cover_sample')
    labels = read_imagenet_label_name(os.path.dirname(args.dataset_dir))

    mol.eval()
    out = {}
    for step, (x, y) in enumerate(cover_sample_loader):
        x = x.cuda() if cuda else x
        label = [y[1][i] for i in range(len(y[0]))]

        prob_class = mol(x)
        top5pre = prob_class.topk(10, 1, True, True)
        top5pre_label = top5pre[1].tolist()
        top5pre_prob = top5pre[0].tolist()
        for i in range(len(label)):
            out[label[i]] = [[labels[top5pre_label[i][j]], top5pre_label[i][j], top5pre_prob[i][j]] for j in range(len(top5pre_prob[i]))]
    out = {k: out[k] for k in sorted(out.keys())}
    print(out)
    with open(out_path, 'w') as f:
        json.dump(out, f)


if __name__ == '__main__':
    args = train_args()
    get_main_function(args.main_fn)()