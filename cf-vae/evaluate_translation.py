__author__ = 'linh'

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import load_npz
from cf_dae import cf_vae_extend, params
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

def load_cvae_data(data_dir):
  data = {}
  # variables = scipy.io.loadmat(data_dir + "mult_nor.mat")
  # data["content"] = variables['X']
  variables = load_npz(os.path.join(data_dir, "mult_nor.npz"))
  data["content"] = variables.toarray()
  data["train_users"] = load_rating(os.path.join(data_dir,"cf-train-%s-users.dat"%args.type))
  data["train_items"] = load_rating(os.path.join(data_dir,"cf-train-%s-items.dat"%args.type))
  data["test_users"] = load_rating(os.path.join(data_dir,"cf-test-%s-users.dat"%args.type))

  return data

def load_rating(path):
  arr = []
  for line in open(path):
    a = line.strip().split()
    if a ==[]:
      l = []
    else:
      l = [int(x) for x in a[1:]]
    arr.append(l)
  return arr

params = params()
params.lambda_u = 1
params.lambda_v = 10
params.lambda_r = 0.1
params.C_a = 1
params.C_b = 0.01
params.max_iter_m = 1


data = load_cvae_data(data_dir)
num_factors = 50
model = cf_vae_extend(num_users=5584, num_items=13790, num_factors=num_factors, params=params,
    input_dim=8000, encoding_dims=[200, 100], z_dim = 50, decoding_dims=[100, 200, 8000], decoding_dims_str=[100,200, 1863],
    loss_type='cross_entropy', encoding_dims_str=[200,100])



model.load_model(extend_file)
model.predict_test(data['test_users'][6000:])


