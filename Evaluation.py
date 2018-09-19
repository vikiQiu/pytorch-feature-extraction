import os
import torch
import json
import numpy as np
from torch.autograd import Variable
from data_process import getDataLoader
from utils.arguments import evaluate_args
from utils.utils import check_dir_exists, cal_cos, cal_distance, cal_accuracy
from Autoencoder import AutoEncoder
from VAE import VAE
from vgg_raw import VGGNet
from AE_class import AEClass


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
    print(model_name)
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
        elif args.main_model == 'AEClass':
            feature, _, _ = mol(b_x)
        elif args.main_model == 'vgg_classifier':
            feature = mol.get_encode_features(b_x)

        f = feature.cpu() if cuda else feature
        f = f.data.view(b_x.shape[0], -1).numpy().tolist()
        features.extend(f)

        if step % 10 == 0:
            print('Step %d finished.' % step)

    out = {'features': features, 'labels': labels}
    with open(output_file, 'w') as f:
        json.dump(out, f)

    return out


def normalize_feature(features):
    features = np.array(features)
    mu = features.mean(0)
    std = features.std(0)
    return (features - mu)/std


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

    if os.path.exists(output_file) and args.load_feature:
        print('Loading features...')
        with open(output_file) as f:
            res = json.load(f)
    else:
        print('Generating features ...')
        res = generate_feature(output_file)
    similar_mat = cal_distance(normalize_feature(res['features']))
    accuracy, similar_pic = cal_accuracy(similar_mat, res['labels'], model_name, args.top_k)


if __name__ == '__main__':
    evaluate()