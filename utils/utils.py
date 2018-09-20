import os
import json
import time
import shutil
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


def check_dir_exists(dirs):
    for d in dirs:
        if not os.path.exists(d):
            os.mkdir(d)


def prepare_train_data(pic_num):
    '''
    Prepare a subset of ILSVRC2012 ImageNet dataset.
    Data directory:
        - label1
            - pic1
            - pic2
            ...
    :param pic_num: Pick up a fix size train data from each label.
    '''
    data_dir = 'F:\ILSVRC2012_img_train\\'
    out_dir = 'E:\work\\feature generation\data\ILSVRC2012\ILSVRC2012_img_train_subset'
    labels = os.listdir(data_dir)
    assert len(labels) == 1000

    for i, label in enumerate(labels):
        print('Deal with label %d: %s' % (i, label))
        os.mkdir(os.path.join(out_dir, label))
        files = os.listdir(os.path.join(data_dir, label))
        files = [x for x in files if x.endswith('.JPEG')]
        if len(files) != 1300:
            print('The number of pictures in label %s is %d, not 1300' % (label, len(files)))

        np.random.seed(1)
        files = np.random.choice(files, pic_num+100, replace=False)
        cnt = 0
        for f in files:
            if cnt == pic_num:
                break
            try:
                im = Image.open(os.path.join(data_dir, label, f))
            except Exception as e:
                print('Bad file:', os.path.join(data_dir, label, f))
            else:
                shutil.copy(os.path.join(data_dir, label, f), os.path.join(out_dir, label, f))
                cnt += 1


def check_cover_data(data_dir='E:\work\image enhancement\data\cover0712\images'):
    files = os.listdir(data_dir)
    files = [x for x in files if x.endswith('.jpg')]
    print('Images number is', len(files))
    bad_img = []
    for i, f in enumerate(files):
        if i % 1000 == 0:
            print('[Checking] %d finished.' % i)
        try:
            im = Image.open(os.path.join(data_dir, f)).convert('RGB')
        except Exception as e:
            print('Bad file', i, ':', os.path.join(data_dir, f))
            bad_img.append(f)
    print(bad_img)
    print(im.convert('RGB'))
    for f in bad_img:
        os.remove(os.path.join(data_dir, f))


def choose_cover_train(data_dir='E:\work\image enhancement\data\cover0712'):
    files = os.listdir(os.path.join(data_dir, 'images'))
    files = [x for x in files if x.endswith('.jpg')]

    check_dir_exists([os.path.join(data_dir, 'samples'), os.path.join(data_dir, 'validation')])

    np.random.seed(123)
    inds = list(range(len(files)))
    np.random.shuffle(inds)
    val_inds = inds[200000:234000]
    sample_inds = inds[234000:]
    val = [files[i] for i in val_inds]
    sample = [files[i] for i in sample_inds]
    print('Validiation number = %d; Sample number = %d' % (len(val), len(sample)))

    print('Move validation images')
    for f in val:
        shutil.move(os.path.join(data_dir, 'images', f), os.path.join(data_dir, 'validation', f))
    print('Move sampled images')
    for f in sample:
        shutil.move(os.path.join(data_dir, 'images', f), os.path.join(data_dir, 'samples', f))
    return


def check_train_data(pic_num):
    '''
    Prepare a subset of ILSVRC2012 ImageNet dataset.
    Data directory:
        - label1
            - pic1
            - pic2
            ...
    :param pic_num: Pick up a fix size train data from each label.
    '''
    data_dir = 'F:\ILSVRC2012_img_train\\'
    out_dir = 'E:\work\\feature generation\data\ILSVRC2012\ILSVRC2012_img_train_subset'
    labels = os.listdir(out_dir)
    assert len(labels) == 1000
    bad_labels = []

    for i, label in enumerate(labels):
        # os.mkdir(os.path.join(out_dir, label))
        files = os.listdir(os.path.join(out_dir, label))
        print('Deal with label %d: %s. Len = %d' % (i, label, len(files)))

        try:
            for f in files:
                im = Image.open(os.path.join(out_dir, label, f))
        except Exception as e:
            print('Directory %s has bad images. Reload images ...' % label)
            bad_labels.append(label)
            print(bad_labels)

    for i, label in enumerate(bad_labels):
        print('Deal with label %d: %s. Len = %d' % (i, label, len(files)))
        shutil.rmtree(os.path.join(out_dir, label))
        # os.removedirs(os.path.join(out_dir, label))
        os.mkdir(os.path.join(out_dir, label))

        files = os.listdir(os.path.join(data_dir, label))
        files = [x for x in files if x.endswith('.JPEG')]
        np.random.seed(1)
        files = np.random.choice(files, pic_num + 100, replace=False)

        cnt = 0
        for f in files:
            if cnt == pic_num:
                break
            try:
                im = Image.open(os.path.join(data_dir, label, f))
            except Exception as e:
                print('Bad file:', os.path.join(data_dir, label, f))
            else:
                shutil.copy(os.path.join(data_dir, label, f), os.path.join(out_dir, label, f))
                cnt += 1


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


def evaluate_cover(cover_loader, cover_sample_loader, mol, cuda, save_dir, topk=23):
    print('####### Evaluating Cover Data (Choose top%d similar images in the samples) ##########' % topk)

    print('[Feature] Generating Encode and fc Sample cover feature')
    encode_fea, fc_fea, labels = generate_features(cover_sample_loader, mol, cuda)
    sample_encode_features = {'features': np.array(encode_fea), 'labels': labels}
    sample_fc_features = {'features': np.array(fc_fea), 'labels': labels}

    print('[Feature] Generating Encode and fc Cover feature')
    encode_fea, fc_fea, labels = generate_features(cover_loader, mol, cuda)
    encode_features = {'features': np.array(encode_fea), 'labels': labels}
    fc_features = {'features': np.array(fc_fea), 'labels': labels}

    evaluate_cover_by_features(sample_encode_features, encode_features, save_dir, topk, 'encode')
    evaluate_cover_by_features(sample_fc_features, fc_features, save_dir, topk, 'fc')

    pass


def evaluate_labeled_data(test_loader, mol, cuda):
    print('####### Evaluating Labeld Data ##########')

    print('[Feature] Generating Encode and fc ImageNet validation feature')
    encode_fea, fc_fea, labels = generate_features(test_loader, mol, cuda)
    labels = [(x[0], x[1]) for x in labels]

    test_time = time.time()
    similar_mat = cal_cos(encode_fea)
    encode_accuracy, _ = cal_accuracy(similar_mat, labels, topk=1)
    encode_top5accuracy, _ = cal_accuracy(similar_mat, labels, topk=5)
    print('[Encode Testing] Feature accuracy = %.5f%%; top5 accuracy = %.5f%%; time cost %.2fs'
          % (np.mean(encode_accuracy) * 100, np.mean(encode_top5accuracy) * 100, time.time() - test_time))

    test_time = time.time()
    similar_mat = cal_cos(fc_fea)
    fc_accuracy, _ = cal_accuracy(similar_mat, labels, topk=1)
    fc_top5accuracy, _ = cal_accuracy(similar_mat, labels, topk=5)
    print('[FC Testing] Feature accuracy = %.5f%%; top5 accuracy = %.5f%%; time cost %.2fs'
          % (np.mean(fc_accuracy) * 100, np.mean(fc_top5accuracy) * 100, time.time() - test_time))
    return encode_accuracy, encode_top5accuracy, fc_accuracy, fc_top5accuracy


def evaluate_cover_by_features(sample_features, features, save_dir, topk=20, name='encode'):
    cos_out = {}
    dist_out = {}
    check_dir_exists([save_dir, os.path.join(save_dir, 'cos_%s' % name), os.path.join(save_dir, 'distance_%s' % name)])
    for i in range(len(sample_features['features'])):
        fea_sample = np.array([sample_features['features'][i]])
        norm = np.dot(fea_sample, fea_sample.T)
        similarity_cos = []
        similarity_dist = []
        for j in range(len(features['features'])):
            fea = np.array([features['features'][j]])
            cos = np.dot(fea_sample, fea.T) / np.sqrt(np.dot(fea, fea.T) * norm)
            similarity_cos.append(cos[0][0])
            dist = np.mean(np.abs(fea - fea_sample))
            similarity_dist.append(dist)
        inds = np.argsort(similarity_cos)[::-1][:topk]
        labels = [[features['labels'][ind], similarity_cos[ind]] for ind in inds]
        cos_out[sample_features['labels'][i]] = labels
        imgs = [sample_features['labels'][i]]
        imgs.extend([x[0] for x in labels])
        save_images(imgs, os.path.join(save_dir, 'cos_%s' % name))

        inds = np.argsort(similarity_dist)[:topk]
        labels = [[features['labels'][ind], similarity_dist[ind]] for ind in inds]
        dist_out[sample_features['labels'][i]] = labels
        imgs = [sample_features['labels'][i]]
        imgs.extend([x[0] for x in labels])
        save_images(imgs, os.path.join(save_dir, 'distance_%s' % name))

        if i % 10 == 0:
            print('[Similar feature] output similar images.')

    out = {'cos': cos_out, 'distance': dist_out}
    with open(os.path.join(save_dir, 'similar_data.json'), 'w') as f:
        json.dump(out, f)


def cal_cos(fea):
    '''
    cos matrix
    :param fea: [[features]]
    :return:
    '''
    print('Calculating cos matrix')
    feature_mat = np.round(np.array(fea), 8).T
    d = np.dot(feature_mat.T, feature_mat)
    norm = (feature_mat * feature_mat).sum(0, keepdims=True) ** .5
    similar_mat = d / norm / norm.T

    # fea = np.array(fea)
    # s = np.sqrt(np.sum(fea**2, 1))

    return similar_mat


def cal_distance(X):
    X = np.array(X).T
    m,n = X.shape
    G = np.dot(X.T, X)
    H = np.tile(np.diag(G), (n, 1))
    return H + H.T - 2*G


def cal_accuracy(similar_mat, labels, model_name=None, topk=5, asscending=False, is_output=True):
    print('Calculating accuracy')
    accuracy = []
    similar_pic = {}

    for i, (label, img_name) in enumerate(labels):
        if asscending:
            inds = np.argsort(similar_mat[i])[1:(topk + 1)]
        else:
            inds = np.argsort(similar_mat[i])[::-1][1:(topk + 1)]
        similar_pic[img_name] = [[labels[ind][1], labels[ind][0] == label] for ind in inds]
        accu = np.mean([labels[ind][0] == label for ind in inds])
        accuracy.append(accu)
    print(np.mean(accuracy))
    out = {'similar_pic': similar_pic, 'accuracy': accuracy}

    if model_name is not None and is_output:
        similar_pic_dir = 'similar_pic/%s/' % model_name
        check_dir_exists(['similar_pic/', similar_pic_dir])
        with open(os.path.join(similar_pic_dir, 'similar_res_%s.json' % model_name), 'w') as f:
            json.dump(out, f)
    return accuracy, similar_pic


def generate_features(data_loader, mol, cuda):
    fc_features = []
    encode_features = []
    labels = []
    print('Total %d data batches' % len(data_loader))
    for step, (x, y) in enumerate(data_loader):
        b_x = Variable(x).cuda() if cuda else Variable(x)
        label = y
        encode_feature = mol.get_encode_features(b_x).data
        fc_feature = mol.get_fc_features(b_x).data
        encode_feature = encode_feature.cpu().numpy().tolist() if cuda else encode_feature.numpy().tolist()
        fc_feature = fc_feature.cpu().numpy().tolist() if cuda else fc_feature.numpy().tolist()
        labels.extend(label)
        encode_features.extend(encode_feature)
        fc_features.extend(fc_feature)

        if step % 10 == 0:
            print('Step %d finished!' % step)
    return encode_features, fc_features, labels


def center_fix_size_transform(size):
    trans = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop((size, size)),
        transforms.ToTensor() # range [0, 255] -> [0.0,1.0]
        ]
    )
    return trans


def save_images(files, pic_dir):
    imgs = []
    for f in files:
        img = Image.open(f).convert('RGB')
        img = center_fix_size_transform(224)(img).numpy().tolist()
        imgs.append(img)
    imgs = torch.Tensor(imgs)
    save_image(imgs, '%s/%s' % (pic_dir, os.path.basename(files[0])))


if __name__ == '__main__':
    # prepare_train_data(200)
    # check_cover_data('E:\work\\feature generation\data\cover\images')
    choose_cover_train('E:\work\\feature generation\data\cover')
