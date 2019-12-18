__author__ = 'linh'

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import load_npz
from cf_dae import cf_vae_extend, params
import argparse
import os
import scipy
from cdae_new import load_cvae_data

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--thred',  type=int, default=15084,
                   help='where model is stored')
parser.add_argument('--data_dir',  type=str, default='data/amazon',
                   help='where model is stored')
parser.add_argument('--mat_file',  type=str, default='cf_vae_1.mat',
                   help='where model is stored')
parser.add_argument('--type',  type=str, default=15,
                   help='where model is stored')
parser.add_argument('--data_type', type=str, default='Health')
parser.add_argument('--item_no', type=int, default=3883)
parser.add_argument('--k',  type=int, default=50,
                   help='top-K')

args = parser.parse_args()
thred = args.thred
data_dir = args.data_dir
extend_file =args.mat_file


params = params()
params.lambda_u = 10
params.lambda_v = 1
params.lambda_r = 1
params.C_a = 1
params.C_b = 0.01
params.max_iter_m = 1


data = load_cvae_data(args)
num_factors = 50
model = cf_vae_extend(num_users=6556, num_items=34295, num_factors=num_factors, params=params,
    input_dim=8000, encoding_dims=[200, 100], z_dim = 50, decoding_dims=[100, 200, 8000], decoding_dims_str=[100,200, 1863],
    loss_type='cross_entropy')

test_size = len(data["test_users"])
model.load_model(extend_file)
if args.k == 50:
    k = [50, 100, 150, 200, 250, 300]
else:
    k = [10, 20, 30, 40, 50]

for m in k:
    model.predict_val(data["train_users"][-test_size:], data["test_users"])



