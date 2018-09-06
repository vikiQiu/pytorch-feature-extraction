import os
import torch
import json
import numpy as np
from torch.autograd import Variable
from data_process import getDataLoader
from utils.arguments import evaluate_args
from utils.utils import check_dir_exists
from Autoencoder import AutoEncoder
from VAE import VAE


################################################################
# Arguments
################################################################
args = evaluate_args()
cuda = args.cuda and torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}


def generate_feature(output_file):
    '''
    Generate feature of given dataset.
        1. laod model
        2. for each data in train_loader => (encoded, (label, img_name))
        3. output a dictionary {'features': [encoded], 'labels': [(label, img_name)]}
    :param output_file: output file name
    '''
    model_name = 'model/%s_%s%s_model-%s.pkl' % (args.main_model, args.model, '' if args.fea_c is None else args.fea_c, args.dataset)
    assert os.path.exists(model_name)
    print('Loading model ...')
    if cuda:
        mol = torch.load(model_name).to(device)
    else:
        mol = torch.load(model_name, map_location='cpu').to(device)
    # print(torchsummary.summary(autoencoder, input_size=(3, HEIGHT, WEIGHT)))
    train_loader = getDataLoader(args, kwargs)

    check_dir_exists(['feature'])

    features = []
    labels = []
    for step, (x, y) in enumerate(train_loader):
        b_x = Variable(x).cuda() if cuda else Variable(x)
        label = [(y[0][i], y[1][i]) for i in range(len(y[0]))]
        labels.extend(label)

        if args.main_model == 'AE':
            feature, _ = mol(b_x)
        elif args.main_model == 'VAE':
            _, mu, std = mol(b_x)
            feature = torch.cat([mu, std])

        f = feature.cpu() if cuda else feature
        f = f.data.view(b_x.shape[0], -1).numpy().tolist()
        features.extend(f)

        if step % 10 == 0:
            print('Step %d finished.' % step)

    out = {'features': features, 'labels': labels}
    with open(output_file, 'w') as f:
        json.dump(out, f)

    return out


def cal_cos(fea):
    '''
    cos matrix
    :param fea: [[features]]
    :return:
    '''
    print('Calculating cos matrix')
    feature_mat = np.round(np.array(fea), 8).T
    d = feature_mat.T @ feature_mat
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


def cal_accuracy(similar_mat, labels, model_name):
    print('Calculating accuracy')
    accuracy = []
    similar_pic = {}
    similar_pic_dir = 'similar_pic/%s/' % model_name
    check_dir_exists(['similar_pic/', similar_pic_dir])

    for i, (label, img_name) in enumerate(labels):
        inds = np.argsort(similar_mat[i])[::-1][1:(args.top_k + 1)]
        similar_pic[img_name] = [[labels[ind][1], labels[ind][0] == label] for ind in inds]
        accu = np.mean([labels[ind][0] == label for ind in inds])
        accuracy.append(accu)
    print(np.mean(accuracy))
    out = {'similar_pic': similar_pic, 'accuracy': accuracy}

    with open(os.path.join(similar_pic_dir, 'similar_res_%s.json' % model_name), 'w') as f:
        json.dump(out, f)
    return accuracy, similar_pic


# def evaluate_pic(similar_pic_dir, model_name):
#     with open(os.path.join(similar_pic_dir, 'similar_res_%s.json' % model_name)) as f:
#         out = json.load(f)
#     similar_pic = out['similar_pic']
#     save_similar_pic(similar_pic, 100, similar_pic_dir, os.path.join(args.dataset_dir, 'ILSVRC2012_img_val'))


def evaluate():
    '''
    Evaluate the AE model on feature extraction work:
        1. Extract features from encode result in AE.
        2. Find top k similar pictures and compare their labels
    '''
    model_name = '%s_%s%s_model-%s' % (args.main_model, args.model, '' if args.fea_c is None else args.fea_c, args.dataset)
    output_file = 'feature/%s.json' % model_name

    if os.path.exists(output_file):
        print('Loading features...')
        with open(output_file) as f:
            res = json.load(f)
    else:
        res = generate_feature(output_file)
    similar_mat = cal_distance(res['features'])
    accuracy, similar_pic = cal_accuracy(similar_mat, res['labels'], model_name)


if __name__ == '__main__':
    evaluate()