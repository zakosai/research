__author__ = 'linh'

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import load_npz
from cvae_user import cf_vae_extend, params_class
import argparse
import os
import scipy

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--ckpt_folder',  type=str, default='pre_model/exp1/',
                   help='where model is stored')
parser.add_argument('--data_dir',  type=str, default='data/amazon',
                   help='where model is stored')
parser.add_argument('--mat_file',  type=str, default='cf_vae_1.mat',
                   help='where model is stored')
parser.add_argument('--type',  type=str, default=15,
                   help='where model is stored')

args = parser.parse_args()
ckpt = args.ckpt_folder
data_dir = args.data_dir
extend_file =args.mat_file

def load_cvae_data(data_dir, item_no):
  variables = load_npz(os.path.join(data_dir,"item.npz"))
  dataset = {}
  dataset["content"] = variables.toarray()


  dataset["train_users"], dataset["train_items"], dataset["test_users"], dataset["train_no"] = read_file(data_dir, item_no)
  user_info = list(open(os.path.join(data_dir, "user_info_train.txt"))) + \
              list(open(os.path.join(data_dir, "user_info_test.txt")))
  user_info = [u.strip() for u in user_info]
  user_info = [u.split(",") for u in user_info]
  user_info = [u[1:] for u in user_info]
  dataset['user_info'] = np.array(user_info).astype(np.float32)
  # col = [0] + list(range(6, dataset["user_info"].shape[1] - 1))
  # dataset["user_info"] = dataset["user_info"][:, col]

  return dataset

def read_file(dir, item_no):
    train = []
    infer = []
    item = [0] * item_no
    idx = 0
    for line in open(os.path.join(dir, "train.txt")):
        a = line.strip().split()
        if a == []:
            l = []
        else:
            l = [int(x) for x in a[1:-1]]
            for i in l:
                if item[i] == 0:
                    item[i] = [idx]
                else:
                    item[i].append(idx)
        train.append(l)
        infer.append([int(a[-1])])
        idx += 1
    train_no = idx
    for line in open(os.path.join(dir, "test.txt")):
        a = line.strip().split()
        if a == []:
            l = []
        else:
            l = [int(x) for x in a[1:-1]]
            for i in l:
                if item[i] == 0:
                    item[i] = [idx]
                else:
                    item[i].append(idx)
        train.append(l)
        infer.append([int(a[-1])])
        idx += 1

    for i in range(len(item)):
        if item[i] == 0:
            item[i] = []

    print(train_no)

    return train, item, infer, train_no

params = params_class()
params.lambda_u = 1
params.lambda_v = 10
params.lambda_r = 0.1
params.C_a = 1
params.C_b = 0.01
params.max_iter_m = 1

n_item = len(list(open(os.path.join(data_dir, "item_id.txt"))))

data = load_cvae_data(data_dir, n_item)
num_factors = 50
model = cf_vae_extend(num_users=6040, num_items=3706, num_factors=num_factors, params=params,
    input_dim=8000, encoding_dims=[200, 100], z_dim = 50, decoding_dims=[100, 200, 8000], decoding_dims_str=[100,200, 1863],
    loss_type='cross_entropy', encoding_dims_str=[200,100])


# d = os.path.join(ckpt, "vae.mat")
# print(d)
model.load_model(extend_file)
pred = model.predict_all()
model.predict_val(pred, data['train_users'], data['test_users'])