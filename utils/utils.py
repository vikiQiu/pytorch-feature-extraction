import os
import shutil
import numpy as np


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
        files = np.random.choice(files, pic_num, replace=False)
        for f in files:
            shutil.copy(os.path.join(data_dir, label, f), os.path.join(out_dir, label, f))


if __name__ == '__main__':
    prepare_train_data(200)

