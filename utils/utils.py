import os
import json
import shutil
import torch
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


def cal_accuracy(similar_mat, labels, model_name=None, topk=5, asscending=False):
    print('Calculating accuracy')
    accuracy = []
    similar_pic = {}
    similar_pic_dir = 'similar_pic/%s/' % model_name
    check_dir_exists(['similar_pic/', similar_pic_dir])

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

    if model_name is None:
        with open(os.path.join(similar_pic_dir, 'similar_res_%s.json' % model_name), 'w') as f:
            json.dump(out, f)
    return accuracy, similar_pic


def generate_features(data_loader, mol, cuda):
    features = []
    labels = []
    print('Total %d data batches' % len(data_loader))
    for step, (x, y) in enumerate(data_loader):
        b_x = Variable(x).cuda() if cuda else Variable(x)
        label = y
        feature = mol.get_features(b_x).data
        feature = feature.cpu().numpy().tolist() if cuda else feature.numpy().tolist()
        labels.extend(label)
        features.extend(feature)

        if step % 10 == 0:
            print('Step %d finished!' % step)
    return features, labels


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
    check_cover_data()
