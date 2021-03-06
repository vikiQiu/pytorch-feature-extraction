import os
import gc
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
from data_process import getDataLoader, RandomBatchSampler, getDataset
from utils.arguments import train_args
from utils.utils import check_dir_exists, evaluate_cover, remove_dir_exists, evaluate_labeled_data, GPU
from model import VGGDecoder, VGG16Feature, SimpleDecoder


class AEClass(torch.nn.Module):
    def __init__(self, encode_channels=32, num_class=1000, decoder='vgg', encoder='vgg'):
        super(AEClass, self).__init__()

        self.encode_channels = encode_channels
        self.features = VGG16Feature()
        if encode_channels != 512:
            self.small_features = nn.Sequential(
                nn.Conv2d(512, 128, kernel_size=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, encode_channels, kernel_size=1),
                nn.BatchNorm2d(encode_channels),
            )
        if decoder == 'vgg':
            self.decoder = VGGDecoder(model='vgg16', out_channels=encode_channels)
        elif decoder == 'vgg2':
            self.decoder = VGGDecoder(model='vgg16_2', out_channels=encode_channels)
        elif decoder == 'simple2':
            self.decoder = VGGDecoder(model='Simple_2', out_channels=encode_channels)
        else:
            self.decoder = VGGDecoder(model='Simple', out_channels=encode_channels)
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
        encode = self.small_features(fea) if self.encode_channels != 512 else fea
        decode = self.decoder(encode)

        c = encode.view(x.size(0), -1)
        c = self.classification(c)
        return None, decode, c

    def get_encode_features(self, x):
        fea = self.features(x)
        fea = self.small_features(fea) if self.encode_channels != 512 else fea
        fea = fea.view(x.size(0), -1)
        return fea

    def get_prob_class(self, x):
        fea = self.features(x)
        encode = self.small_features(fea) if self.encode_channels != 512 else fea
        c = encode.view(x.size(0), -1)
        c = self.classification(c)
        return c

    def get_fc_features(self, x, return_both=False):
        fea = self.get_encode_features(x)
        c = fea
        for name, layer in self.classification._modules.items():
            if int(name) <= 5:
                c = layer(c)
        if return_both:
            return fea, c
        else:
            return c


def get_optimized_params(mol, require, lr):
    if require == 'small_features':
        params = list(mol.small_features.parameters())
        return [{'params': x, 'lr': lr} for x in params]
    elif require == 'decoder':
        params = list(mol.decoder.parameters())
        return [{'params': x, 'lr': lr} for x in params]
    elif require == 'classifier':
        params = list(mol.classification.parameters())
        return [{'params': x, 'lr': lr} for x in params]
    elif 'vgg' in require:
        params = []
        require = int(require.split('.')[1])
        for name, param in mol.features.named_parameters():
            if int(name.split('.')[1]) >= require:
                params.append({'params': param, 'lr': lr/10})
        return params
    else:
        return []


def test_decoder(test_loader, mol, cuda, name):
    step_time = time.time()
    print('#### Start %s testing with %d batches ####' % (name, len(test_loader)))

    decoders = []
    for step, (x, y) in enumerate(test_loader):
        x = x.detach().cuda() if cuda else x.detach()
        b_y = x.detach().cuda() if cuda else x.detach()  # batch y, shape (batch, 32*32*3)

        _, decoded, _ = mol(x)

        loss_decoder = F.mse_loss(decoded, b_y)
        # loss_decoder.backward()
        decoders.append(loss_decoder.item())
        loss_decoder = None

        if step % 1 == 0:
            print('[%s Testing] Step: %d | Decoder error %.6f; Time cost %.2f s' %
                  (name, step, loss_decoder.item(), time.time() - step_time))
            step_time = time.time()

    print('[%s Testing] #### Final Score ####: Decoder error %.6f; Time cost %.2f s' %
          (name, np.mean(decoders), time.time() - step_time))

    return np.mean(decoders)


def test_cls_decoder(test_loader, mol, cuda, name):
    # TODO:
    total, correct, top5correct = 0, 0, 0
    loss_class_fn = nn.CrossEntropyLoss().cuda(cuda)
    loss_decoder_fn = nn.MSELoss()
    step_time = time.time()
    loss_decoder, loss_cls = [], []
    print('#### Start %s testing with %d batches ####' % (name, len(test_loader)))

    for step, (x, y) in enumerate(test_loader):
        b_x = Variable(x, volatile=True).cuda() if cuda else Variable(x)
        b_y = b_x.detach().cuda() if cuda else b_x.detach()
        label = Variable(torch.Tensor([y[2][i] for i in range(len(y[0]))]).long())
        label = label.cuda() if cuda else label

        _, decoded, prob_class = mol(b_x)
        loss_d = loss_decoder_fn(decoded, b_y)
        loss_c = loss_class_fn(prob_class, label)
        (loss_d + loss_c).backward()
        loss_decoder.append(loss_d.item())
        loss_cls.append(loss_c.item())
        loss_d = None
        loss_c = None

        _, predicted = torch.max(prob_class.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()
        top5pre = prob_class.topk(5, 1, True, True)
        top5pre = top5pre[1].t()
        top5correct += top5pre.eq(label.view(1, -1).expand_as(top5pre)).sum().item()

        if step % 20 == 0:
            print('[%s Testing] Step: %d | Classification error %.6f; Decoder error %.6f; '
                  'Accuracy %.3f%%; Top5 Accuracy %.3f%%; Time cost %.2f s' %
                  (name, step, np.mean(loss_cls), np.mean(loss_decoder), correct * 100 / total, top5correct * 100 / total, time.time() - step_time))
            step_time = time.time()

    print('[%s Testing] #### Final Score ####: Test size %d; Accuracy %.3f%%; Top5 Accuracy %.3f%%; Time cost %.2f s' %
          (name, total, correct * 100 / total, top5correct * 100 / total, time.time() - step_time))
    return np.mean(loss_cls), np.mean(loss_decoder), correct/total, top5correct/total


def train_decoder_only(args, mol_short='AEClass_d', main_model=AEClass):
    print('######### This is train decoder only function #############')
    ################################################################
    # Arguments
    ################################################################
    ae_args = args
    cuda = ae_args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if ae_args.cuda else {}
    # global ae_args, cuda, device, kwargs

    d_name = '' if args.decoder == 'vgg' else '_' + args.decoder + 'decoder'
    mol_short = mol_short + args.name + d_name
    log_dir = 'log/log_%s_%s%s_model-%s/' %\
              (mol_short, args.model, '' if args.fea_c is None else args.fea_c, args.dataset)
    remove_dir_exists([log_dir])
    writer = SummaryWriter(log_dir)

    start_time = time.time()
    model_name = 'model/%s_%s%s_model-%s.pkl' \
                 % (mol_short, args.model, '' if args.fea_c is None else args.fea_c, args.dataset)
    print('[Model] model name is', model_name)
    pic_dir = 'res/%s_%s%s-%s/' % (mol_short, args.model, '' if args.fea_c is None else args.fea_c, args.dataset)
    evaluation_dir = 'res/evaluation_pic/%s_%s%s-%s' % (mol_short, args.model, '' if args.fea_c is None else args.fea_c, args.dataset)
    if os.path.exists(model_name) and args.load_model:
        print('Loading model ...')
        mol = torch.load(model_name).to(device)
    else:
        print('Init model ...')
        mol = main_model(args.fea_c, decoder=args.decoder).to(device)

    print('Prepare data loader ...')
    test_loader = getDataLoader(args, kwargs, train='test')
    # small_test_loader = getDataLoader(args, kwargs, train=False, p=10)
    cover_loader = getDataLoader(args, kwargs, train='cover')
    cover_val_loader = getDataLoader(args, kwargs, train='cover_validation')
    cover_sample_loader = getDataLoader(args, kwargs, train='cover_sample')

    # optimizer1 = torch.optim.Adam(list(mol.classification.parameters())+list(mol.small_features.parameters())+
                                  # list(mol.decoder.parameters()), lr=args.lr)
    optimizer1 = torch.optim.Adam(([] if args.fea_c == 512 else get_optimized_params(mol, 'small_features', args.lr)) +
                                  get_optimized_params(mol, 'decoder', args.lr)+
                                  get_optimized_params(mol, 'vgg.38', args.lr/5),
                                  lr=args.lr)
    loss_decoder = nn.MSELoss()

    check_dir_exists(['res/', 'model', pic_dir, log_dir, 'res/evaluation_pic', evaluation_dir])
    loss_val = None

    print('Start training ...')
    cnt = 0
    for epoch in range(args.epoch):
        if epoch % 5 == (5-1) or epoch == (args.epoch - 1):
            # Evaluation on cover data
            eval_dir = os.path.join(evaluation_dir, 'epoch%d' % epoch)
            evaluate_cover(cover_val_loader, cover_sample_loader, mol, cuda, eval_dir)

        if epoch != 0:
            # Testing on Cover val
            print('######### Testing on Cover val Dataset ###########')
            test_loss_decoder = test_decoder(cover_val_loader, mol, cuda, 'Full')
            # test_loss = (1 - args.alpha) * test_loss_cls + args.alpha * test_loss_decoder / 0.001
            writer.add_scalar('test_cover/loss_decoder', test_loss_decoder, epoch)
            # writer.add_scalar('test_cover/loss_classifier', test_loss_cls, epoch)
            # writer.add_scalar('test_cover/loss', test_loss, epoch)
            # writer.add_scalar('test_cover/accuracy', test_acc, epoch)
            # writer.add_scalar('test_cover/top5accuracy', test_top5acc, epoch)

            # Testing on ImageNet val
            print('######### Testing on ImageNet val Dataset ###########')
            # test_loss_decoder, test_loss_cls, test_acc, test_top5acc = test_cls_decoder(test_loader, mol, cuda, 'Full')
            test_loss_decoder = test_decoder(test_loader, mol, cuda, 'Full')
            # test_loss = (1 - args.alpha) * test_loss_cls + args.alpha * test_loss_decoder / 0.001
            writer.add_scalar('test_imagenet/loss_decoder', test_loss_decoder, epoch)
            # writer.add_scalar('test_imagenet/loss_classifier', test_loss_cls, epoch)
            # writer.add_scalar('test_imagenet/loss', test_loss, epoch)
            # writer.add_scalar('test_imagenet/accuracy', test_acc, epoch)
            # writer.add_scalar('test_imagenet/top5accuracy', test_top5acc, epoch)

        step_time = time.time()
        print('######### Training with %d batches total ##########' % len(cover_loader))
        for step, (x, y) in enumerate(cover_loader):
            b_x = Variable(x).cuda() if cuda else Variable(x)
            b_y = b_x.detach().cuda() if cuda else b_x.detach()  # batch y, shape (batch, 32*32*3)

            _, decoded, prob_class = mol(b_x)

            if step % 500 == 0 or step == (len(cover_loader) - 1):
                img_to_save = decoded.data
                save_image(img_to_save, '%s/%s-%s.jpg' % (pic_dir, epoch, step))

            loss1 = loss_decoder(decoded, b_y)
            # loss = (1-args.alpha) * loss2 + args.alpha * loss1 / 0.001
            loss = loss1
            writer.add_scalar('train/loss_decoder', loss1, cnt)
            writer.add_scalar('train/loss', loss, cnt)

            optimizer1.zero_grad()
            loss.backward()
            optimizer1.step()

            loss_val = 0.99*loss_val + 0.01*loss.data[0] if loss_val is not None else loss.data[0]

            if step % 10 == 0 or step == (len(cover_loader) - 1):
                if os.path.exists(model_name):
                    shutil.copy2(model_name, model_name.split('.pkl')[0]+'_back.pkl')
                torch.save(mol, model_name)
                print('[Training] Epoch:', epoch, 'Step:', step, '|',
                      'train loss %.6f; Time cost %.2f s; Decoder error %.6f;' %
                      (loss.data[0], time.time() - step_time, loss1.data[0]))
                step_time = time.time()

            cnt += 1

    print('Finished. Totally cost %.2f' % (time.time() - start_time))
    writer.export_scalars_to_json(os.path.join(log_dir, 'all_scalars.json'))
    writer.close()


def train(args, mol_short='AEClass_both', main_model=AEClass):
    print('######### This is train function #############')
    ################################################################
    # Arguments
    ################################################################
    ae_args = args
    if args.gpu == -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = GPU().choose_gpu()
    elif args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    cuda = ae_args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if ae_args.cuda else {}

    d_name = '' if args.decoder == 'vgg' else '_' + args.decoder + 'decoder'
    mol_short = mol_short + args.name + d_name
    log_dir = 'log/log_%s_%s%s_model-%s/' %\
              (mol_short, args.model, '' if args.fea_c is None else args.fea_c, args.dataset)
    remove_dir_exists([log_dir])
    writer = SummaryWriter(log_dir)

    start_time = time.time()
    model_name = 'model/%s_%s%s_model-%s.pkl' % (mol_short, args.model, '' if args.fea_c is None else args.fea_c, args.dataset)
    print('[Model] model name is', model_name)
    pic_dir = 'res/%s_%s%s-%s/' % (mol_short, args.model, '' if args.fea_c is None else args.fea_c, args.dataset)
    evaluation_dir = 'res/evaluation_pic/%s_%s%s-%s' % (mol_short, args.model, '' if args.fea_c is None else args.fea_c, args.dataset)
    if os.path.exists(model_name) and args.load_model:
        print('Loading model ...')
        if cuda:
            mol = torch.load(model_name).to(device)
        else:
            mol = torch.load(model_name, map_location='cpu')
    else:
        print('Init model ...')
        mol = main_model(args.fea_c).to(device)

    print('Prepare data loader ...')
    # train_loader = getDataLoader(args, kwargs, train='train')
    fuse_loader = getDataLoader(args, kwargs, train='fuse', p=args.imgnet_p)
    test_dataset = RandomBatchSampler(getDataset(args, train='test'), args.batch_size)
    test_loader = getDataLoader(args, kwargs, train='test')
    # small_test_loader = getDataLoader(args, kwargs, train=False, p=10)
    # cover_loader = getDataLoader(args, kwargs, train='cover')
    cover_val_loader = getDataLoader(args, kwargs, train='cover_validation')
    cover_sample_loader = getDataLoader(args, kwargs, train='cover_sample')

    # optimizer_cls = torch.optim.Adam(get_optimized_params(mol, 'classifier', lr=args.lr) +
    #                                  get_optimized_params(mol, 'small_features', lr=args.lr) +
    #                                  get_optimized_params(mol, 'vgg.38', lr=args.lr/5), lr=args.lr)
    optimizer_d = torch.optim.Adam(([] if args.fea_c == 512 else get_optimized_params(mol, 'small_features', args.lr)) +
                                   get_optimized_params(mol, 'decoder', args.lr) +
                                   get_optimized_params(mol, 'vgg.38', args.lr / 5),
                                   lr=args.lr)
    optimizer = torch.optim.Adam(([] if args.fea_c == 512 else get_optimized_params(mol, 'small_features', args.lr)) +
                                 get_optimized_params(mol, 'classifier', lr=args.lr) +
                                 get_optimized_params(mol, 'decoder', args.lr) +
                                 get_optimized_params(mol, 'vgg.38', args.lr / 5),
                                 lr=args.lr)
    loss_decoder = nn.MSELoss()
    loss_class = nn.CrossEntropyLoss(reduction='none').cuda(cuda)

    def loss_cls_fn(prob_class, label, weight, beta=0.01):
        if torch.sum(weights) == 0:
            return 99999
        else:
            lss = loss_class(prob_class, label) * weight
            lss = torch.sum(lss)/torch.sum(weight)
            return lss * beta

    check_dir_exists(['res/', 'model', pic_dir, log_dir, 'res/evaluation_pic', evaluation_dir])

    total, correct, top5correct, cnt = 0, 0, 0, 0
    print('Start training ...')
    for epoch in range(args.epoch):
        if (epoch % 5 == 0) and epoch != 0:
            # Evaluation on cover data
            mol.eval()
            eval_dir = os.path.join(evaluation_dir, 'epoch%d' % epoch)
            evaluate_cover(cover_val_loader, cover_sample_loader, mol, cuda, eval_dir, args)

            fc_accuracy, fc_top5accuracy = evaluate_labeled_data(test_loader, mol, cuda, both=False)
            print('Fc accuracy:', np.mean(fc_accuracy))
            print('Fc top5 accuracy:', np.mean(fc_top5accuracy))

        if epoch >= 0:
            mol.eval()
            # Testing on ImageNet val
            print('######### Testing on ImageNet val Dataset ###########')
            test_loss_decoder, test_loss_cls, test_acc, test_top5acc = test_cls_decoder(test_loader, mol, cuda, 'Full')
            # test_loss = (1 - args.alpha) * test_loss_cls + args.alpha * test_loss_decoder / 0.001
            writer.add_scalar('test_imagenet/loss_decoder', test_loss_decoder, epoch)
            writer.add_scalar('test_imagenet/loss_classifier', test_loss_cls, epoch)
            # writer.add_scalar('test_imagenet/loss', test_loss, epoch)
            writer.add_scalar('test_imagenet/accuracy', test_acc, epoch)
            writer.add_scalar('test_imagenet/top5accuracy', test_top5acc, epoch)

            # Testing on Cover val
            print('######### Testing on Cover val Dataset ###########')
            # test_loss_decoder, test_loss_cls, test_acc, test_top5acc = test_cls_decoder(cover_val_loader, mol, cuda, 'Full')
            test_loss_decoder = test_decoder(cover_val_loader, mol, cuda, 'Full')
            # test_loss = (1 - args.alpha) * test_loss_cls + args.alpha * test_loss_decoder / 0.001
            writer.add_scalar('test_cover/loss_decoder', test_loss_decoder, epoch)
            # writer.add_scalar('test_cover/loss_classifier', test_loss_cls, epoch)
            # writer.add_scalar('test_cover/loss', test_loss, epoch)
            # writer.add_scalar('test_cover/accuracy', test_acc, epoch)
            # writer.add_scalar('test_cover/top5accuracy', test_top5acc, epoch)

        step_time = time.time()
        for step, (x, y) in enumerate(fuse_loader):
            mol.train()
            b_x = Variable(x).cuda() if cuda else Variable(x)
            b_y = b_x.detach().cuda() if cuda else b_x.detach()  # batch y, shape (batch, 32*32*3)
            # //TODO
            # train_cls =
            label = Variable(torch.Tensor([y[2][i] for i in range(len(y[0]))]).long())
            label = label.cuda() if cuda else label
            weights = Variable(torch.Tensor([y[3][i] for i in range(len(y[0]))]))
            weights = weights.cuda() if cuda else weights

            _, decoded, prob_class = mol(b_x)

            if step % 100 == 0:
                img_to_save = decoded.data
                save_image(img_to_save, '%s/%s-%s.jpg' % (pic_dir, epoch, step))

            loss1 = loss_decoder(decoded, b_y)
            loss2 = loss_cls_fn(prob_class, label, weights, beta=0.01)
            loss = 0
            writer.add_scalar('train/loss_decoder', loss1, cnt)
            writer.add_scalar('train/loss_classifier', loss2, cnt)
            # writer.add_scalar('train/loss', loss, cnt)

            if loss2 != 99999:
                loss = (1 - args.alpha) * loss2 + args.alpha * loss1
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(prob_class.data, 1)
                predicted = torch.Tensor([predicted[i] for i in range(len(weights)) if weights[i] == 1])
                label2 = torch.Tensor([label[i] for i in range(len(weights)) if weights[i] == 1])
                total += predicted.shape[0]
                correct += (predicted == label2).sum().item()

                prob_class2 = prob_class[[i for i in range(len(weights)) if weights[i] == 1]]
                top5pre = prob_class2.topk(5, 1, True, True)
                top5pre = top5pre[1].t()
                label2 = label2.long().cuda() if cuda else label2.long()
                top5correct += top5pre.eq(label2.view(1, -1).expand_as(top5pre)).sum().item()
                writer.add_scalar('train/accuracy', correct / total, cnt)
                writer.add_scalar('train/top5_accuracy', top5correct / total, cnt)

            else:
                optimizer_d.zero_grad()
                loss1.backward()
                optimizer_d.step()

            # loss_val = 0.99*loss_val + 0.01*loss.data[0] if loss_val is not None else loss.data[0]

            if step % 10 == 0:
                if os.path.exists(model_name):
                    shutil.copy2(model_name, model_name.split('.pkl')[0]+'_back.pkl')
                torch.save(mol, model_name)
                total_tmp = total if total != 0 else 1

                t_x, t_y = test_dataset.get_sample()
                test_acc, test_top5_acc = _test_sample_batch(mol, cuda, t_x, t_y)

                print('[Training] Epoch:', epoch, 'Step:', step, '|',
                      'Time cost %.2f s; Classification error %.6f; Decoder error %.6f; Loss %.6f; '
                      'Accuracy %.3f%%/%.3f%%; Top5 Accuracy %.3f%%/%.3f%% (%s)' %
                      (time.time() - step_time, 0.0888 if type(loss2)==int else loss2.data[0], loss1.data[0],
                       0.0888 if type(loss)==int else loss.data[0], correct*100/total_tmp, test_acc*100,
                       top5correct*100/total_tmp, test_top5_acc*100, time.ctime()))

                writer.add_scalar('test/accuracy', test_acc, cnt)
                writer.add_scalar('test/top5_accuracy', test_top5_acc, cnt)

                correct, total, top5correct = 0, 0, 0
                step_time = time.time()

            cnt += 1

    print('Finished. Totally cost %.2f' % (time.time() - start_time))
    writer.export_scalars_to_json(os.path.join(log_dir, 'all_scalars.json'))
    writer.close()


def _test_sample_batch(mol, cuda, x, y):
    mol.eval()
    b_x = Variable(x).cuda() if cuda else Variable(x)
    label = Variable(torch.Tensor([y[i][2] for i in range(len(y))]).long())
    label = label.cuda() if cuda else label

    prob_class = mol.get_prob_class(b_x)
    _, predicted = torch.max(prob_class.data, 1)
    total = label.size(0)
    correct = (predicted == label).sum().item()
    top5pre = prob_class.topk(5, 1, True, True)
    top5pre = top5pre[1].t()
    top5correct = top5pre.eq(label.view(1, -1).expand_as(top5pre)).sum().item()

    acc = correct / total
    top5_acc = top5correct / total
    return acc, top5_acc


if __name__ == '__main__':
    args = train_args()
    if args.imgnet_p == 0:
        train_decoder_only(args)
    else:
        train(args)

