import os
import json
import time
import shutil
import torch
import torch.nn as nn
from PIL import Image
from torch.autograd import Variable
from torchvision.utils import save_image
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


def check_dir_exists(dirs):
    for d in dirs:
        if not os.path.exists(d):
            os.mkdir(d)


def remove_dir_exists(dirs):
    for d in dirs:
        if os.path.exists(d):
            shutil.rmtree(d)


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


def choose_cover_samples(data_dir='E:\work\image enhancement\data\cover0712', replace_file=None):
    '''
    Need 500 cover sample images, while only 141 samples.
    Move 351 images from validation to samples
    :param data_dir:
    :return:
    '''
    files = os.listdir(os.path.join(data_dir, 'validation'))
    files = [x for x in files if x.endswith('.jpg')]

    np.random.seed(123)
    inds = list(range(len(files)))
    np.random.shuffle(inds)
    sample_inds = inds[:351]
    sample = [files[i] for i in sample_inds]
    print('Sample number = %d' % (len(sample)))

    print('Move sampled images')
    for f in sample:
        shutil.move(os.path.join(data_dir, 'validation', f), os.path.join(data_dir, 'samples', f))
    return


def update_cover_labels(json_file, label_file):
    '''
    Update the judge json to the label json
    :param json_file: in the directory 'res/evaluation_pic/your_model_name/similar_fc_data.json'
    :param label_file: The label file stored in cover data directory
    :return:
    '''
    with open(json_file) as f:
        dat = json.load(f)['cos']

    samples = sorted(list(dat.keys()))

    if os.path.exists(label_file):
        with open(label_file) as f:
            labels = json.load(f)
    else:
        labels = {s: {} for s in samples}

    judge_file = os.path.join(os.path.dirname(json_file), 'judge_cos_fc_labels.json')

    if os.path.exists(judge_file):
        with open(judge_file) as f:
            judges = json.load(f)
        for sample in list(judges.keys()):
            good = judges[sample]['good']
            # bad = judges[sample]['bad']
            # ok = set(range(len(dat[sample]))) - set(good) - set(bad)
            ok = judges[sample]['ok']
            bad = set(range(len(dat[sample]))) - set(good) - set(ok)
            for ind in good:
                labels[sample][dat[sample][ind][0]] = 2
            for ind in ok:
                print(sample,ind)
                print(dat[sample])
                print(dat[sample][ind])
                print(dat[sample][ind][0])
                labels[sample][dat[sample][ind][0]] = 1
            for ind in bad:
                labels[sample][dat[sample][ind][0]] = -1

        with open(label_file, 'w') as f:
            json.dump(labels, f)

    else:
        judges = {s: {'good': [], 'bad': []} for s in samples}
        print(samples, len(samples))
        print(judges)
        with open(judge_file, 'w') as f:
            json.dump(judges, f)
        labels = judges

    return labels


def make_judge_file(json_files, label_file, cover_dir, save_dir, topk=23):
    '''
    make file and images to be judged.
    Compare the label file and
    :param json_files:
    :param label_file:
    :return:
    '''
    remove_dir_exists([save_dir])
    check_dir_exists([save_dir])
    dat = None
    for json_file in json_files:
        with open(json_file) as f:
            tmp = json.load(f)['distance']
            tmp = {l: set([ll[0] for ll in tmp[l][:topk]]) for l in tmp.keys()}
        if dat is None:
            dat = tmp
        else:
            dat = {l: dat[l] | tmp[l] for l in dat.keys()}
    with open(label_file) as f:
        labels = json.load(f)
        labels = {l: set(labels[l].keys()) for l in labels}

    not_judge = {l: sorted(list(dat[l] - (dat[l] & labels[l]))) for l in labels}
    for l in not_judge:
        files = [os.path.join(cover_dir, 'validation', os.path.basename(ll)) for ll in not_judge[l]]
        files = [os.path.join(cover_dir, 'samples', os.path.basename(l))]*10 + files
        save_images(files, save_dir, nrow=10)

    to_judge_json = {l: {'good': [], 'ok': []} for l in not_judge}
    not_judge = {l: [[x, 0, 0] for x in list(not_judge[l])] for l in not_judge.keys()}
    with open(os.path.join(save_dir, 'similar_fc.json'), 'w') as f:
        json.dump(not_judge, f)
    with open(os.path.join(save_dir, 'to_judge.json'), 'w') as f:
        json.dump(to_judge_json, f)


def generate_judge_criteria(cover_dir, save_dir):
    '''
    Generate the judge critieria json
    {sample cover name: {
        'good': what is good?
        'bad': what is bad?
        }
    }
    :param cover_dir: the sample cover directory
    :param save_dir: directory to save the json
    :return:
    '''
    files = os.listdir(cover_dir)
    out = {}
    for f in files:
        out[f] = {'good': '', 'ok': '', 'bad': ''}
    with open(os.path.join(save_dir, 'judge_criteria.json'), 'w') as f:
        json.dump(out, f)


def judge_cover_labels(json_file, label_file):
    '''
    Judge the cover json file by the label file.
    :param json_file: in the res/evaluation_pic/xxx/
    :param label_file: in the cover dicrectory
    :return: Get top1 accuracy and so on .
    '''
    with open(json_file) as f:
        dat = json.load(f)['cos']
    with open(label_file) as f:
        labels = json.load(f)

    cos_top1_accuracy = []
    cos_top5_accuracy = []
    # cos_top10_accuracy = []
    for s_img in dat.keys():
        judges = [labels[s_img][l[0]] for l in dat[s_img][:5]]
        cos_top1_accuracy.append(judges[0])
        cos_top5_accuracy.extend(judges[:5])
        # cos_top10_accuracy.extend(judges[:10])

    cos_top1_accuracy = np.array(cos_top1_accuracy)
    cos_top5_accuracy = np.array(cos_top5_accuracy)
    # cos_top10_accuracy = np.array(cos_top10_accuracy)
    print('[Result] Accuracy of cover features:')
    print('[Top1 cos] NOT BAD accuracy = %.4f; GOOD accuracy = %.4f; BAD accuracy = %.4f; NOT SURE rate = %.4f'
          % (np.mean(cos_top1_accuracy > 0), np.mean(cos_top1_accuracy == 2),
             np.mean(cos_top1_accuracy == -1), np.mean(cos_top1_accuracy == 0)))
    print('[Top5 cos] NOT BAD accuracy = %.4f; GOOD accuracy = %.4f; BAD accuracy = %.4f; NOT SURE rate = %.4f'
          % (np.mean(cos_top5_accuracy > 0), np.mean(cos_top5_accuracy == 2),
             np.mean(cos_top5_accuracy == -1), np.mean(cos_top5_accuracy == 0)))
    # print('[Top10 cos] NOT BAD accuracy = %.4f; GOOD accuracy = %.4f; BAD accuracy = %.4f; NOT SURE rate = %.4f'
    #       % (np.mean(cos_top10_accuracy > 0), np.mean(cos_top10_accuracy == 2),
    #          np.mean(cos_top10_accuracy == -1), np.mean(cos_top10_accuracy == 0)))


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


def evaluate_cover(cover_loader, cover_sample_loader, mol, cuda, save_dir, args, topk=23):
    print('####### Evaluating Cover Data (Choose top%d similar images in the samples) ##########' % topk)

    print('[Feature] Generating Encode and fc Sample cover feature')
    fc_fea, labels = generate_features(cover_sample_loader, mol, cuda, both=False)
    # sample_encode_features = {'features': np.array(encode_fea), 'labels': labels}
    sample_fc_features = {'features': np.array(fc_fea), 'labels': labels}

    label_file = os.path.join(args.cover_dir, 'val_labels.json')
    if os.path.exists(label_file):
        with open(label_file) as f:
            cover_label = json.load(f)
            cover_label = {os.path.basename(l): {os.path.basename(ll): cover_label[l][ll] for ll in cover_label[l]}
                           for l in cover_label}
    else:
        cover_label = {os.path.basename(x[1]): {} for x in labels}

    # print(cover_label, len(cover_label))

    # print('[Feature] Generating Encode and fc Cover feature')
    fc_fea, labels = generate_features(cover_loader, mol, cuda, both=False)
    # encode_features = {'features': np.array(encode_fea), 'labels': labels}
    fc_features = {'features': np.array(fc_fea), 'labels': labels}

    # evaluate_cover_by_features(sample_encode_features, encode_features, save_dir, topk, 'encode')
    evaluate_cover_by_features(sample_fc_features, fc_features, save_dir, topk, 'fc', cover_label)

    pass


def evaluate_labeled_data(test_loader, mol, cuda, both=True):
    print('####### Evaluating Labeld Data ##########')

    if both:
        print('[Feature] Generating Encode and fc ImageNet validation feature')
        encode_fea, fc_fea, labels = generate_features(test_loader, mol, cuda)
        labels = [(x[0], x[1]) for x in labels]
        test_time = time.time()
        similar_mat = cal_cos(encode_fea)
        encode_accuracy, _ = cal_accuracy(similar_mat, labels, topk=1)
        encode_top5accuracy, _ = cal_accuracy(similar_mat, labels, topk=5)
        print('[Encode Testing] Feature accuracy = %.5f%%; top5 accuracy = %.5f%%; time cost %.2fs'
              % (np.mean(encode_accuracy) * 100, np.mean(encode_top5accuracy) * 100, time.time() - test_time))
    else:
        print('[Feature] Generating fc ImageNet validation feature')
        fc_fea, labels = generate_features(test_loader, mol, cuda, both)
        labels = [(x[0], x[1]) for x in labels]

    test_time = time.time()
    similar_mat = cal_cos(fc_fea)
    fc_accuracy, _ = cal_accuracy(similar_mat, labels, topk=1)
    fc_top5accuracy, _ = cal_accuracy(similar_mat, labels, topk=5)
    print('[FC Testing] Feature accuracy = %.5f%%; top5 accuracy = %.5f%%; time cost %.2fs'
          % (np.mean(fc_accuracy) * 100, np.mean(fc_top5accuracy) * 100, time.time() - test_time))

    if both:
        return encode_accuracy, encode_top5accuracy, fc_accuracy, fc_top5accuracy
    else:
        return fc_accuracy, fc_top5accuracy


def evaluate_cover_by_features(sample_features, features, save_dir, topk=20, name='encode', cover_label=None):
    '''
    cos_out: {sample_img_name: [val_img_name, cos, distance, judged_label(-1, 0, 1, 2)]}
    judged_label:
        2: good
        1: ok
        0: not judge yet
        -1: bad
    :param sample_features: Sample cover features {'features': np.array, 'labels': ['0', img_name]}
    :param features: Training or validation cover features{'features': np.array, 'labels': ['0', img_name]}
    :param save_dir: The directory to save the json result and the images
    :param topk: choose topk similar images to generate the image
    :param name: ['encode', 'fc']
    :param cover_label: If cover_label=None, do not judge the label.
                        {sample_img_name: {val_img_name: label(-1, 0, 1, 2)}}
    :return:
    '''
    print('Evaluating cover by %s features ... ' % name)
    cos_out = {}
    dist_out = {}
    check_dir_exists([save_dir, os.path.join(save_dir, 'cos_%s' % name), os.path.join(save_dir, 'distance_%s' % name)])

    cos_top1_accuracy = []
    cos_top5_accuracy = []
    cos_top10_accuracy = []
    for i in range(len(sample_features['features'])):
        if cover_label is not None:
            judges = cover_label[os.path.basename(sample_features['labels'][i][1])]
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
        labels = [[features['labels'][ind][1], similarity_cos[ind], similarity_dist[ind]] for ind in inds]
        if cover_label is not None:
            for l in [os.path.basename(x[0]) for x in labels]:
                if l not in judges.keys():
                    judges[l] = 0
            labels = [[x[0], x[1], x[2], judges[os.path.basename(x[0])]] for x in labels]
            cos_top1_accuracy.append(labels[0][3])
            cos_top5_accuracy.extend([x[3] for x in labels[:5]])
            cos_top10_accuracy.extend([x[3] for x in labels[:10]])
        cos_out[sample_features['labels'][i][1]] = labels
        imgs = [sample_features['labels'][i][1]]
        imgs.extend([x[0] for x in labels])
        save_images(imgs, os.path.join(save_dir, 'cos_%s' % name))

        inds = np.argsort(similarity_dist)[:topk]
        labels = [[features['labels'][ind][1], similarity_cos[ind], similarity_dist[ind]] for ind in inds]
        dist_out[sample_features['labels'][i][1]] = labels
        imgs = [sample_features['labels'][i][1]]
        imgs.extend([x[0] for x in labels])
        save_images(imgs, os.path.join(save_dir, 'distance_%s' % name))

        if i % 10 == 0:
            print('[Similar feature] output similar images.')

    if cover_label is not None:
        cos_top1_accuracy = np.array(cos_top1_accuracy)
        cos_top5_accuracy = np.array(cos_top5_accuracy)
        cos_top10_accuracy = np.array(cos_top10_accuracy)
        print('[Result] Accuracy of cover features:')
        print('[Top1 cos] NOT BAD accuracy = %.4f; GOOD accuracy = %.4f; BAD accuracy = %.4f; NOT SURE rate = %.4f'
              % (np.mean(cos_top1_accuracy > 0), np.mean(cos_top1_accuracy == 2),
                 np.mean(cos_top1_accuracy == -1), np.mean(cos_top1_accuracy == 0)))
        print('[Top5 cos] NOT BAD accuracy = %.4f; GOOD accuracy = %.4f; BAD accuracy = %.4f; NOT SURE rate = %.4f'
              % (np.mean(cos_top5_accuracy > 0), np.mean(cos_top5_accuracy == 2),
                 np.mean(cos_top5_accuracy == -1), np.mean(cos_top5_accuracy == 0)))
        print('[Top10 cos] NOT BAD accuracy = %.4f; GOOD accuracy = %.4f; BAD accuracy = %.4f; NOT SURE rate = %.4f'
              % (np.mean(cos_top10_accuracy > 0), np.mean(cos_top10_accuracy == 2),
                 np.mean(cos_top10_accuracy == -1), np.mean(cos_top10_accuracy == 0)))
    out = {'cos': cos_out, 'distance': dist_out}
    with open(os.path.join(save_dir, 'similar_%s_data.json' % name), 'w') as f:
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


def generate_features(data_loader, mol, cuda, both=True):
    fc_features = []
    encode_features = []
    labels = []
    print('Total %d data batches' % len(data_loader))
    mol.eval()
    for step, (x, y) in enumerate(data_loader):
        b_x = Variable(x, volatile=True).cuda() if cuda else Variable(x)
        label = [(y[0][i], y[1][i]) for i in range(len(y[0]))]
        encode_feature, fc_feature = mol.get_fc_features(b_x, return_both=True)
        encode_feature = encode_feature.data.cpu().numpy().tolist() if cuda else encode_feature.data.numpy().tolist()
        fc_feature = fc_feature.data.cpu().numpy().tolist() if cuda else fc_feature.data.numpy().tolist()
        labels.extend(label)
        encode_features.extend(encode_feature)
        fc_features.extend(fc_feature)

        if step % 10 == 0:
            print('Step %d finished!' % step)
    if both:
        return encode_features, fc_features, labels
    else:
        return fc_features, labels


def center_fix_size_transform(size):
    trans = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop((size, size)),
        transforms.ToTensor() # range [0, 255] -> [0.0,1.0]
        ]
    )
    return trans


def read_imagenet_label_name(dir_name):
    '''
    Get <label_idx: label_name> directory
    :param dir_name: the ImageNet directory
    :return:
    '''
    json_name = os.path.join(dir_name, 'name_label.json')
    with open(json_name) as f:
        res = json.load(f)
    res = {int(x): res[x] for x in res.keys()}
    return res


def save_images(files, pic_dir, nrow=8):
    imgs = []
    for f in files:
        img = Image.open(f).convert('RGB')
        img = center_fix_size_transform(224)(img).numpy().tolist()
        imgs.append(img)
    imgs = torch.Tensor(imgs)
    save_image(imgs, os.path.join(pic_dir, os.path.basename(files[0])), nrow=nrow)


class GPU:
    def __init__(self, sleep=120):
        self.max_gpu_util = 100
        self.min_men_free = 8000
        self.sleep = sleep
        return

    def choose_gpu(self):
        while True:
            id = self.which_to_use()
            if id == -1:
                print('[%s]Waiting for free gpu... Sleep %ds. ' % (time.ctime(), self.sleep))
                time.sleep(self.sleep)
            else:
                # os.environ["CUDA_VISIBLE_DEVICES"] = str(id)
                print('Using GPU %d' % id)
                # return
                return str(id)

    def get_gpu_men(self):
        gpu = os.popen('nvidia-smi -q -d Utilization |grep Gpu').readlines()
        gpu = [float(x.strip().split(':')[1].split('%')[0]) for x in gpu]
        men = os.popen('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free').readlines()
        men = [float(x.strip().split(':')[1].split('MiB')[0]) for x in men]
        print(gpu, men)
        return gpu, men

    def which_to_use(self):
        gpu, men = self.get_gpu_men()
        idx = [x for x in range(len(gpu)) if (gpu[x] <= self.max_gpu_util) and (men[x] >= self.min_men_free)]
        print(idx)
        if len(idx) == 0:
            return -1
        else:
            return idx[np.argmax([men[i] for i in idx])]


if __name__ == '__main__':
    inception_json = '..\\res\evaluation_pic\Fin\inception_v3_conv-ImageNet1000-val\similar_fc_data.json'
    vgg_json = '..\\res\evaluation_pic\Fin\\vgg_conv-ImageNet1000-val\similar_fc_data.json'
    resnet_json = '..\\res\evaluation_pic\Fin\\resnet50_conv-ImageNet1000-val\similar_fc_data.json'
    vgg_cls_json = '..\\res\evaluation_pic\Fin\\VGGClass_conv32-ImageNet1000-train-sub\epoch20\similar_fc_data.json'
    label_file = 'E:\work\\feature generation\data\cover\\val_labels.json'
    cover_dir = 'E:\work\\feature generation\data\cover'
    eval_pic_dir = 'E:\work\\feature generation\pytorch-feature-extraction\\res\evaluation_pic\judges'
    aeclass1017_json = 'E:\work\\feature generation\pytorch-feature-extraction\\res\evaluation_pic\\20181018\AEClass_both_conv512-ImageNet1000-train-sub\epoch0\similar_fc_data.json'
    vggbase_all_json = 'E:\work\\feature generation\pytorch-feature-extraction\\res\evaluation_pic\\20181024\VGGClass_all_conv512-ImageNet1000-train-sub\epoch0\similar_fc_data.json'
    vggbase_json = 'E:\work\\feature generation\pytorch-feature-extraction\\res\evaluation_pic\\20181024\VGGClass_conv512-ImageNet1000-train-sub\epoch15\similar_fc_data.json'
    binary512_json = 'E:\work\\feature generation\\1109_pytorch_feature_extraction\\res\\1123\inception_512fea_binary_model\similar_fc_data.json'
    binary256_json = 'E:\work\\feature generation\\1109_pytorch_feature_extraction\\res\\1123\inception_256fea_binary_model\similar_fc_data.json'
    # prepare_train_data(200)
    # check_cover_data('E:\work\\feature generation\data\cover\images')
    # choose_cover_train('E:\work\\feature generation\data\cover')
    # labels = update_cover_labels('..\\res\evaluation_pic\Fin\inception_v3_conv-ImageNet1000-val\similar_fc_data.json',
    #                              label_file='E:\work\\feature generation\data\cover\\val_labels.json')
    # judge_cover_labels('..\\res\evaluation_pic\Fin\inception_v3_conv-ImageNet1000-val\similar_fc_data.json',
    #                              label_file='E:\work\\feature generation\data\cover\\val_labels.json')
    make_judge_file([binary512_json, binary256_json],
                    label_file=label_file, cover_dir=cover_dir, save_dir=eval_pic_dir, topk=5)
    # generate_judge_criteria(os.path.join(cover_dir, 'samples'), eval_pic_dir)

    # add new judged labels
    # labels = update_cover_labels('..\\res\evaluation_pic\judges\similar_fc.json',
    #                              label_file='E:\work\\feature generation\data\cover\\val_labels.json')
    # judge_cover_labels(vggbase_json,
    #                    label_file='E:\work\\feature generation\data\cover\\val_labels.json')