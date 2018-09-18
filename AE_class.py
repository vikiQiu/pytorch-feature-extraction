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
# import torchsummary
import torch.nn as nn
from torchvision.utils import save_image
from data_process import getDataLoader
from utils.arguments import train_args
from utils.utils import check_dir_exists, generate_features, save_images
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
        c = self.classification(c)
        return encode, decode, c

    def get_features(self, x):
        fea = self.features(x)
        fea = self.small_features(fea)
        fea = fea.view(x.size(0), -1)
        return fea

    def get_prob_class(self, x):
        fea = self.features(x)
        encode = self.small_features(fea)
        c = encode.view(x.size(0), -1)
        c = self.classification(c)
        return c


def test(test_loader, mol, cuda, name):
    total, correct, top5correct = 0, 0, 0
    loss_class = nn.CrossEntropyLoss().cuda(cuda)
    step_time = time.time()
    print('#### Start %s testing with %d batches ####' % (name, len(test_loader)))

    for step, (x, y) in enumerate(test_loader):
        # if np.random.randn() > 0.1:
        #     continue

        b_x = Variable(x).cuda() if cuda else Variable(x)
        label = Variable(torch.Tensor([y[2][i] for i in range(len(y[0]))]).long())
        label = label.cuda() if cuda else label

        prob_class = mol.get_prob_class(b_x)
        loss = loss_class(prob_class, label)  # mean square error

        _, predicted = torch.max(prob_class.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()
        top5pre = prob_class.topk(5, 1, True, True)
        top5pre = top5pre[1].t()
        top5correct += top5pre.eq(label.view(1, -1).expand_as(top5pre)).sum().item()

        if step % 20 == 0:
            print('[%s Testing] Step: %d | '
                  'Classification error %.6f; Accuracy %.3f%%; Top5 Accuracy %.3f%%; Time cost %.2f s' %
                  (name, step, loss, correct * 100 / total, top5correct * 100 / total, time.time() - step_time))
            step_time = time.time()

    print('[%s Testing] #### Final Score ####: Test size %d; Accuracy %.3f%%; Top5 Accuracy %.3f%%; Time cost %.2f s' %
          (name, total, correct * 100 / total, top5correct * 100 / total, time.time() - step_time))
    return correct/total, top5correct/total


def evaluate_cover(cover_loader, cover_sample_loader, mol, cuda, save_dir, topk=20):
    print('####### Evaluating ##########')

    sample_features, features = {}, {}
    print('[Feature] Sample cover feature')
    fea, labels = generate_features(cover_sample_loader, mol, cuda)
    sample_features['features'] = np.array(fea)
    sample_features['labels'] = labels

    print('[Feature] Cover feature')
    fea, labels = generate_features(cover_loader, mol, cuda)
    features['features'] = np.array(fea)
    features['labels'] = labels

    cos_out = {}
    dist_out = {}
    for i in range(len(sample_features['features'])):
        fea_sample = np.array([sample_features['features'][i]])
        norm = np.dot(fea_sample, fea_sample.T)
        similarity_cos = []
        similarity_dist = []
        for j in range(len(features['features'])):
            fea = np.array([features['features'][j]])
            cos = np.dot(fea_sample, fea.T) / np.sqrt(np.dot(fea, fea.T)*norm)
            similarity_cos.append(cos[0][0])
            dist = np.mean(np.abs(fea - fea_sample))
            similarity_dist.append(dist)
        inds = np.argsort(similarity_cos)[::-1][:topk]
        labels = [[features['labels'][ind], similarity_cos[ind]] for ind in inds]
        cos_out[sample_features['labels'][i]] = labels
        imgs = [sample_features['labels'][i]]
        imgs.extend([x[0] for x in labels])
        save_images(imgs, os.path.join(save_dir, 'cos'))

        inds = np.argsort(similarity_dist)[:topk]
        labels = [[features['labels'][ind], similarity_dist[ind]] for ind in inds]
        dist_out[sample_features['labels'][i]] = labels
        imgs = [sample_features['labels'][i]]
        imgs.extend([x[0] for x in labels])
        save_images(imgs, os.path.join(save_dir, 'distance'))

        if i % 10 == 0:
            print('[Similar feature] output similar images.')

    pass


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

    log_dir = 'log/log_AEClass_%s%s_model-%s/' %\
              (args.model, '' if args.fea_c is None else args.fea_c, args.dataset)
    writer = SummaryWriter(log_dir)

    start_time = time.time()
    model_name = 'model/AEClass_%s%s_model-%s.pkl' % (args.model, '' if args.fea_c is None else args.fea_c, args.dataset)
    pic_dir = 'res/AEClass_%s%s-%s/' % (args.model, '' if args.fea_c is None else args.fea_c, args.dataset)
    evaluation_dir = 'res/evaluation_pic/AEClass_%s%s-%s' % (args.model, '' if args.fea_c is None else args.fea_c, args.dataset)
    if os.path.exists(model_name) and args.load_model:
        print('Loading model ...')
        mol = torch.load(model_name).to(device)
    else:
        print('Init model ...')
        mol = AEClass(args.fea_c).to(device)

    print('Prepare data loader ...')
    train_loader = getDataLoader(args, kwargs)
    test_loader = getDataLoader(args, kwargs, train='test')
    # small_test_loader = getDataLoader(args, kwargs, train=False, p=10)
    cover_loader = getDataLoader(args, kwargs, train='cover')
    cover_sample_loader = getDataLoader(args, kwargs, train='cover_sample')

    optimizer2 = torch.optim.Adam(list(mol.features.parameters()), lr=args.lr/5)
    optimizer1 = torch.optim.Adam(list(mol.classification.parameters())+list(mol.small_features.parameters())+
                                  list(mol.decoder.parameters()), lr=args.lr)
    loss_decoder = nn.MSELoss()
    loss_class = nn.CrossEntropyLoss().cuda(cuda)

    check_dir_exists(['res/', 'model', pic_dir, log_dir, 'res/evaluation_pic', evaluation_dir])
    loss_val = None

    total, correct, top5correct, cnt = 0, 0, 0, 0
    print('Start training ...')
    for epoch in range(args.epoch):
        # Evaluation
        eval_dir = os.path.join(evaluation_dir, 'epoch%d' % epoch)
        check_dir_exists([eval_dir, os.path.join(eval_dir, 'cos'), os.path.join(eval_dir, 'distance')])
        evaluate_cover(cover_loader, cover_sample_loader, mol, cuda, eval_dir)

        # Testing
        test_acc, test_top5acc = test(test_loader, mol, cuda, 'Full')
        writer.add_scalar('test/accuracy', test_acc, epoch)
        writer.add_scalar('test/top5accuracy', test_top5acc, epoch)

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
            # io.imsave('.xxx.jpg',img_to_save[0])

            loss1 = loss_decoder(decoded, b_y)
            loss2 = loss_class(prob_class, label) # mean square error
            loss = (1-args.alpha) * loss2 + args.alpha * loss1
            writer.add_scalar('train/loss_decoder', loss1, cnt)
            writer.add_scalar('train/loss_classifier', loss2, cnt)
            writer.add_scalar('train/loss', loss, cnt)

            # if epoch % 4 != 0:
            #     optimizer1.zero_grad()
            #     loss1.backward()
            #     optimizer1.step()
            # else:
            #     loss2.backward()
            #     optimizer2.step()

            optimizer1.zero_grad()
            # optimizer2.zero_grad()
            loss.backward()
            optimizer1.step()
            # optimizer2.step()

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
                      'train loss %.6f; Time cost %.2f s; Classification error %.6f; Decoder error %.6f; '
                      'Accuracy %.3f%%; Top5 Accuracy %.3f%%' %
                      (loss.data[0], time.time() - step_time, loss2, loss1, correct*100/total, top5correct*100/total))
                correct, total, top5correct = 0, 0, 0
                step_time = time.time()

            cnt += 1

    print('Finished. Totally cost %.2f' % (time.time() - start_time))
    writer.export_scalars_to_json(os.path.join(log_dir, 'all_scalars.json'))
    writer.close()


if __name__ == '__main__':
    train()
    pass
