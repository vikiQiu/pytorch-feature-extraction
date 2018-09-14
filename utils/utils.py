import os
import shutil
import numpy as np
from PIL import Image


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


if __name__ == '__main__':
    prepare_train_data(200)

