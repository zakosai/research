__author__ = 'linh'
import numpy as np
import tensorflow as tf
import scipy.io
import matplotlib.pyplot as plt
from cf_dae import cf_vae_extend, params
from scipy.sparse import load_npz
import  argparse
import os
import csv
params = params()


def load_cvae_data(args):
    data = {}
    variables = load_npz(os.path.join(args.data_dir, "mult_nor.npz"))
    if args.data_type == args.data_dir.split("_")[-1]:
      variables = variables[-args.item_no:]
    else:
      variables = variables[:args.item_no]
    data["content"] = variables.toarray()

    data_file = list(open("%s/%s_user_product.txt"%(args.data_dir, args.data_type)).readlines())
    data["train_users"], data["train_items"], data["test_users"], data["val_users"] = load_rating(data_file, args.item_no)
    return data


def load_rating(data, item_no):
    data = [i.split(" ") for i in data]
    data = [[int(j) for j in i] for i in data]
    data = [i[1:] for i in data]
    print(max([max(i) for i in data]))
    train_no = int(len(data) * 0.7)
    val_no = int(len(data) * 0.05)
    test_users = [i[-5:] for i in data[train_no + val_no:]]
    val_users = [i[-5:] for i in data[train_no: train_no + val_no]]
    train_users = data[:train_no] + [i[:-5] for i in data[train_no:]]

    train_items = [0] * item_no
    for i, user in enumerate(train_users):
        for item in user:
            if train_items[item] == 0:
                train_items[item] = [i]
            else:
                train_items[item].append(i)
    for i in range(item_no):
        if train_items[i] == 0:
            train_items[i] = []
    return train_users, train_items, test_users, val_users


def main(args):
    model_type = args.model
    ckpt = args.ckpt_folder
    initial = args.initial
    iter = args.iter
    data_dir = args.data_dir
    zdim = args.zdim
    gs = args.gridsearch

    params.lambda_u = 10
    params.lambda_v = 1
    params.lambda_r = 1
    params.C_a = 1
    params.C_b = 0.01
    params.max_iter_m = 1
    params.EM_iter = args.iter
    params.num_iter = 150

    C = [0.1, 1, 10]

    # # for updating W and b in vae
    # self.learning_rate = 0.001
    # self.batch_size = 500
    # self.num_iter = 3000
    # self.EM_iter = 100

    data = load_cvae_data(args)
    np.random.seed(0)
    tf.set_random_seed(0)

    # images = np.fromfile(os.path.join(data_dir,"images.bin"), dtype=np.uint8)
    # img = images.reshape((13791, 32, 32, 3))
    # img = img.astype(np.float32)/255
    num_factors = zdim
    best_recall = 0
    best_hyper = []
    dim = data['content'].shape[1]
    test_size = len(data["test_users"])

    if gs == 1:
        i = 0
        recalls = []
        for u in [0.1, 1, 10]:
            params.lambda_u = u
            for v in [1, 10, 100]:
                params.lambda_v = v
                for r in [0.1, 1, 10]:
                    params.lambda_r = r
                    if i > -1:
                        model = cf_vae_extend(num_users=args.user_no, num_items=args.item_no, num_factors=num_factors, params=params,
                                              input_dim=dim, encoding_dims=[400, 200], z_dim=zdim, decoding_dims=[200,400,dim],
                                              decoding_dims_str=[200, 4526], loss_type='cross_entropy',
                                              model=model_type, ckpt_folder=ckpt)
                        model.fit(data["train_users"], data["train_items"], data["content"], params, data["test_users"])
                        model.save_model(os.path.join(ckpt,"cf_dae_%d_%d.mat"%(model_type, i)))
                        # model.load_model("cf_vae.mat")
                        f = open(os.path.join(ckpt, "result_cdae_%d.txt"%model_type), 'a')
                        f.write("%d-----------%f----------%f----------%f\n"%(i,u,v,r))
                        recall = model.predict_val(data["train_users"][-test_size:], data["test_users"], f)
                        f.write("\n")
                        f.close()
                        if recall > best_recall:
                            best_recall = recall
                            best_hyper = [u,v, r, i]

                        print(u, v, r)
                    i += 1

        f = open(os.path.join(ckpt, "result_sum.txt"), "a")
        f.write("Best recall CDAE: %f at %d (%f, %f, %f)\n" % (best_recall, best_hyper[3], best_hyper[0], best_hyper[1],
                                                                                       best_hyper[2]))
        f.close()
    else:
        model = cf_vae_extend(num_users=args.user_no, num_items=args.item_no, num_factors=num_factors, params=params,
                              input_dim=8000, encoding_dims=[400, 200], z_dim=zdim, decoding_dims=[200, 400, 8000],
                              decoding_dims_str=[200, 4526], loss_type='cross_entropy',
                              model=model_type, ckpt_folder=ckpt)
        model.fit(data["train_users"], data["train_items"], data["content"], params, data["test_users"])
        model.save_model(os.path.join(ckpt,"cf_dae_%s_%d.mat"%(args.data_type, model_type)))
        model.predict_val(data["train_users"][-test_size:], data["test_users"])

if __name__ == '__main__':
    np.random.seed(0)
    tf.set_random_seed(0)

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model', type=int, default=0,
                        help='type of model: 0-only text, 1-text+image, 2-text+image+structure, 3-text+structure')
    parser.add_argument('--ckpt_folder', type=str, default='pre_model/exp1/')
    parser.add_argument('--initial', type=bool, default=True)
    parser.add_argument('--iter', type=int, default=30)
    parser.add_argument('--data_dir', type=str, default='data/amazon')
    parser.add_argument('--zdim', type=int, default=50)
    parser.add_argument('--gridsearch', type=int, default=0)
    parser.add_argument('--data_type', type=str, default='Health')
    parser.add_argument('--user_no', type=int, default=6040)
    parser.add_argument('--item_no', type=int, default=3883)
    args = parser.parse_args()
    main(args)

