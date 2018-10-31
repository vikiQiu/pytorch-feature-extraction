import os
import sys
import argparse
import shutil
import pandas as pd


def check_dir_exists(dirs):
    for d in dirs:
        if not os.path.exists(d):
            os.mkdir(d)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", type=str, default='..\data\ILSVRC2012\\', help="Image directory to transfrom.")
    parser.add_argument("--output-dir", type=str, default='..\data\ILSVRC2012\\', help="Output directory to store prepared data.")
    parser.add_argument("--label-path", type=str, default='..\data\ILSVRC2012\\', help="Label path.")

    args = parser.parse_args()

    return args


def prepare_dataset():
    args = get_args()
    check_dir_exists([args.output_dir])

    # get labels
    print('Loading label file for ImageNet ...')
    df = pd.read_csv(args.label_path)
    img_list = list(df.img_list)
    label_list = list(df.label_list)

    # check for label dirctories
    check_dir_exists([os.path.join(args.output_dir, l) for l in list(set(label_list))])

    # copy the images into the correct label directory
    N = len(img_list)
    print('Totally %d images to copy ...' % N)
    for i in range(len(img_list)):
        img_name = os.path.basename(img_list[i])
        if i % 1000 == 0:
            print('%d/%d images have been done' % (i, N))
        shutil.copy(os.path.join(args.img_dir, img_name), os.path.join(args.output_dir, label_list[i], img_name))


if __name__ == '__main__':
    prepare_dataset()