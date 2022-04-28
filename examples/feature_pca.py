import torch
import os.path as osp
from ibl import models
from ibl.pca import PCA
from loguru import logger
# from ibl.evaluators import extract_features
from ibl.utils.data.preprocessor import Preprocessor
from ibl.utils.data import IterLoader, get_transformer_train, get_transformer_test
from ibl import datasets
from torch.utils.data import DataLoader
from ibl.utils.data.sampler import DistributedRandomTupleSampler, DistributedSliceSampler
import random
from ibl.evaluators_one import extract_cnn_feature, Evaluator


def get_data(data_dir, test_batch_size=32, scale='250k', height=480, width=640, workers=8):
    root = osp.join(data_dir, 'pitts')
    dataset = datasets.create('pitts', root, scale=scale)

    test_transformer = get_transformer_test(height, width)
    # test_loader = DataLoader(
    #     Preprocessor(sorted(list(set(dataset.q_test) | set(dataset.db_test))),
    #                  root=dataset.images_dir, transform=test_transformer),
    #     batch_size=test_batch_size, num_workers=workers,
    #     sampler=DistributedSliceSampler(
    #         sorted(list(set(dataset.q_test) | set(dataset.db_test)))),
    #     shuffle=False, pin_memory=True)

    # train_extract_loader = DataLoader(
    #     Preprocessor(sorted(list(set(dataset.q_train) | set(dataset.db_train))),
    #                  root=dataset.images_dir, transform=test_transformer),
    #     batch_size=test_batch_size, num_workers=workers,
    #     sampler=DistributedSliceSampler(
    #         sorted(list(set(dataset.q_train) | set(dataset.db_train)))),
    #     shuffle=False, pin_memory=True)

    test_loader = DataLoader(
        Preprocessor(sorted(list(set(dataset.q_test) | set(dataset.db_test))),
                     root=dataset.images_dir, transform=test_transformer),
        batch_size=test_batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    train_extract_loader = DataLoader(
        Preprocessor(sorted(list(set(dataset.q_train) | set(dataset.db_train))),
                     root=dataset.images_dir, transform=test_transformer),
        batch_size=test_batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return dataset, test_loader, train_extract_loader


def get_model(model_path):
    base_model = models.create(name="vgg16",
                               train_layers='conv5',
                               cut_at_pooling=True)
    pool_layer = models.create('netvlad', dim=base_model.feature_dim)
    model = models.create('embednet', base_model, pool_layer)
    pth_file = osp.join(model_path, 'model_best.pth.tar')
    state_dict = torch.load(pth_file, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict['state_dict'])

    return model


def get_pca(model_path, pca_whitening):

    pca_parameters_path = osp.join(model_path, 'pca_params_model_best.h5')
    pca = PCA(feat_num, pca_whitening, pca_parameters_path)
    return pca


def extract_features(model, data_loader, dataset, print_freq=100,
                     vlad=True, pca=None, gpu=None, sync_gather=False):
    model.eval()
    features_dict = {}
    if (pca is not None):
        pca.load(gpu=gpu)
    with torch.no_grad():
        for idx, (imgs, fnames, _, _, _) in enumerate(data_loader):
            outputs = extract_cnn_feature(model, imgs, vlad, gpu=gpu)
            if (pca is not None):
                outputs = pca.infer(outputs)
            outputs = outputs.data.cpu()
            for fname, feature in zip(fnames, outputs):
                features_dict[fname] = feature
            if idx % 100 == 0:
                print("extract_cnn_feature: {}/{}".format(idx, len(data_loader)))
    return features_dict


def pca_train(model_path, pca_whitening, data_dir):
    device = torch.device("cuda")
    model = get_model(model_path=model_path)
    model.to(device)
    pca = get_pca(model_path=model_path, pca_whitening=pca_whitening)
    dataset, test_loader, train_extract_loader = get_data(data_dir)

    logger.info("extrac_features")
    dict_f = extract_features(model, train_extract_loader, sorted(list(set(dataset.q_train) | set(dataset.db_train))),
                              vlad=True, gpu=device, sync_gather=False)
    logger.info("sample features")
    features = list(dict_f.values())
    if (len(features) > 10000):
        features = random.sample(features, 10000)
    features = torch.stack(features)
    logger.info("pca train features")
    pca.train(features)
    evaluator = Evaluator(model)
    evaluator.evaluate(test_loader, sorted(list(set(dataset.q_test) | set(dataset.db_test))),
                       dataset.q_test, dataset.db_test, dataset.test_pos,
                       vlad=True, pca=pca, gpu=device, sync_gather=args.sync_gather)
    synchronize()
    return


if __name__ == "__main__":
    data_dir = "/data/zebin/data/Pittsburgh"
    model_path = "/data/zebin/OpenIBL/logs/netVLADBaseline/pitts250k-vgg16/vgg16-sare_ind-lr0.001-tuple7-cd512-rd4096"
    pca_whitening = False
    feat_num = 4096
    pca_train(model_path=model_path,
              pca_whitening=pca_whitening, data_dir=data_dir)
