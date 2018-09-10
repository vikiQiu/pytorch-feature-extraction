import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
import torch.utils.data as Data
import torchvision.transforms as transforms
from torchvision.utils import save_image
import xml.etree.ElementTree as ET


################################################################
# Data Loader
################################################################
def default_loader(img):
    img = Image.open(img).convert('RGB')
    return img


################################################################
# Data Transformer
################################################################
def fix_size_transform(size):
    trans = transforms.Compose([
        transforms.Resize(size),
        transforms.RandomCrop((size, size)),
        transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
        ]
    )
    return trans


# transforms.ToTensor()
default_transformer = transforms.Compose([
    transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
    ]
)

transformers = {'default': default_transformer, 'crop224': fix_size_transform(224)}
loaders = {'default': default_loader}


################################################################
# Dataset and Data Loader
################################################################

def getDataset(args):
    '''
        Now support ['ImageNet1000-val']。
        Add more dataset in future.
        '''
    if args.dataset == 'ImageNet1000-val':
        label_dir = os.path.join(args.dataset_dir, 'ILSVRC2012_bbox_val_v3')
        img_dir = os.path.join(args.dataset_dir, 'ILSVRC2012_img_val')
        dataset = ImageNetDataset(img_dir, label_dir,
                                  img_transform=transformers['crop' + str(args.img_size)],
                                  loader=loaders[args.img_loader])
        return dataset
    pass


def getDataLoader(args, kwargs):
    dataset = getDataset(args)
    return Data.DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, **kwargs)


################################################################
# Save similar pictures
################################################################
def save_similar_pic(similar_pic, k, similar_pic_dir, img_dir):
    for img_name in list(similar_pic.keys())[:k]:
        img = default_loader(os.path.join(img_dir, img_name))
        img = [fix_size_transform(224)(img).numpy()]
        for (i, sim) in similar_pic[img_name]:
            sim_img = default_loader(os.path.join(img_dir, i))
            img.append(fix_size_transform(224)(sim_img).numpy())

        img = torch.Tensor(img)
        save_image(img, '%s/%s' % (similar_pic_dir, img_name))
    return


################################################################
# Get ILSVRC2012 dataset label from XML
################################################################
def get_xml_label(dir_path, file_name):
    tree = ET.ElementTree(file=os.path.join(dir_path, file_name))
    label, pic_name = '', ''
    for elem in tree.iter(tag='name'):
        label = elem.text
        break

    for elem in tree.iter(tag='filename'):
        pic_name = elem.text
        break

    return pic_name, label


def filter_label(not_exists, label_list, label_ind, img_list, img_dir, Debug=False):
    for pic in not_exists:
        pic_name = os.path.join(img_dir, pic)
        if pic_name in img_list:
            ind = img_list.index(pic_name)
            img_list.remove(pic_name)
            del label_list[ind]
            del label_ind[ind]
            if Debug: print('%s is removed.' % pic_name)
    return img_list, label_list, label_ind


def get_all_xml_labels(dir_path, img_dir, output_file='labels.csv', Debug=False):
    files = [f for f in os.listdir(dir_path) if f.endswith('.xml')]
    if Debug: print('Totally {} xml files'.format(len(files)))
    img_list = []
    label_list = []
    for i, f in enumerate(files):
        if Debug and i % 1000 == 0: print('File {}: {}'.format(i, f))
        img, label = get_xml_label(dir_path, f)
        img += '.JPEG'
        img_list.append(os.path.join(img_dir, img))
        label_list.append(label)

    if Debug:
        print(img_list[:10])
        print(label_list[:10])

    classes = list(np.unique(label_list))
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}

    label_ind = [class_to_idx[x] for x in label_list]
    pd.DataFrame({'img_list': img_list, 'label_list': label_list, 'label_ind': label_ind}).\
        to_csv(os.path.join(dir_path, output_file), index=False)
    pd.DataFrame({'labels': classes, 'ind': list(range(len(classes)))}).\
        to_csv(os.path.join(dir_path, 'label_ind.csv'), index=False)
    return img_list, label_list, label_ind


def get_imagenet1000_val_labels(dir_path, img_dir, file_name='labels.csv', Debug=False):
    '''
    Get ImageNet 1000 dataset (ILSVRC2012) validation labels.
    :param file_name: A csv file with 'img_list' and 'label_list' columns
    '''
    filter_list = ['ILSVRC2012_val_00021280.JPEG']
    if not os.path.exists(os.path.join(dir_path, file_name)):
        print('Creating label file for ImageNet ...')
        img_list, label_list, label_ind = get_all_xml_labels(dir_path, img_dir, file_name)
    else:
        print('Loading label file for ImageNet ...')
        df = pd.read_csv(os.path.join(dir_path, file_name))
        img_list = list(df.img_list)
        label_list = list(df.label_list)
        label_ind = list(df.label_ind)

    img_list, label_list, label_ind = filter_label(filter_list, label_list, label_ind, img_list, img_dir, Debug)
    if Debug:
        print('Totally {} images.'.format(len(img_list)))
        print(img_list[:10])
        print(label_list[:10])
        print(default_loader(img_list[0]))
    return img_list, label_list, label_ind


################################################################
# Get ILSVRC2012 set
################################################################
class ImageNetDataset(Data.Dataset):
    def __init__(self,
                 img_dir,
                 label_dir,
                 label_file='label.csv',
                 img_transform=None,
                 loader=default_loader):
        self.img_list, self.label_list, self.label_ind = get_imagenet1000_val_labels(label_dir, img_dir, label_file)
        self.img_transform = img_transform
        self.loader = loader
        self.all_label = list(sorted(np.unique(self.label_list)))

    def __getitem__(self, index):
        img_path = self.img_list[index]
        img_name = os.path.basename(img_path)
        label = self.label_list[index]
        img = self.loader(img_path)
        # img = img_path
        if self.img_transform is not None:
            img = self.img_transform(img)
        return img, (label, img_name, self.all_label.index(label))

    def __len__(self):
        return len(self.label_list)


def testImageNetDataset(img_dir, label_dir, img_transform):
    dataset = ImageNetDataset(img_dir, label_dir, img_transform=img_transform)

    for index, (img, label) in enumerate(dataset):
        # img.show()
        print(img.shape)
        print('label', label)


if __name__ == '__main__':
    label_dir = '..\data\ILSVRC2012\\ILSVRC2012_bbox_val_v3\\'
    img_dir = '..\data\ILSVRC2012\\ILSVRC2012_img_val\\'

    '''TEST get_all_xml_labels()'''
    # get_all_xml_labels(dir_path, Debug=True)

    '''TEST get_imagenet1000_val_labels()'''
    # get_imagenet1000_val_labels(label_dir, img_dir, 'labels.csv', Debug=True)

    # TEST ImageNetDataset
    testImageNetDataset(img_dir, label_dir, fix_size_transform(256))