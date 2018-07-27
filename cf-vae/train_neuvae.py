import numpy as np
import tensorflow as tf
import scipy.io
import matplotlib.pyplot as plt
from neuvae import neuVAE, params
from scipy.sparse import load_npz
import argparse
import os
import csv


np.random.seed(0)
tf.set_random_seed(0)


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--model',  type=int, default=0,
                   help='type of model: 0-only text, 1-text+image, 2-text+image+structure, 3-text+structure')
parser.add_argument('--ckpt_folder',  type=str, default='pre_model/exp1/',
                   help='where model is stored')
parser.add_argument('--initial',  type=bool, default=True,
                   help='where model is stored')
parser.add_argument('--iter',  type=int, default=30,
                   help='where model is stored')
parser.add_argument('--data_dir',  type=str, default='data/amazon',
                   help='where model is stored')
parser.add_argument('--zdim',  type=int, default=50,
                   help='where model is stored')
parser.add_argument('--gridsearch',  type=int, default=0,
                   help='gridsearch or not')
parser.add_argument('--data_type',  type=str, default='5',
                   help='gridsearch or not')
parser.add_argument('--user_dim',  type=int, default=9975,
                   help='gridsearch or not')
parser.add_argument('--user_no',  type=int, default=6040,
                   help='gridsearch or not')
parser.add_argument('--item_no',  type=int, default=3883,
                   help='gridsearch or not')
args = parser.parse_args()
model_type = args.model
ckpt = args.ckpt_folder
initial = args.initial
iter = args.iter
data_dir = args.data_dir
zdim = args.zdim
gs = args.gridsearch
data_type = args.data_type
print(model_type)

def load_cvae_data(data_dir):
  data = {}
  variables = load_npz(os.path.join(data_dir,"mult_nor.npz"))
  data["content"] = variables.toarray()
  user = np.load(os.path.join(data_dir, "user_info_%s.npy"%data_type))
  data["user"] = user
  data["train_users"] = load_rating(data_dir + "cf-train-%sp-users.dat"%data_type)
  data["train_items"] = load_rating(data_dir + "cf-train-%sp-items.dat"%data_type)
  data["test_users"] = load_rating(data_dir + "cf-test-%sp-users.dat"%data_type)
  data["test_items"] = load_rating(data_dir + "cf-test-%sp-items.dat"%data_type)
  rating = list(open(data_dir + "train-%s.csv"%data_type))
  rating = [r.strip() for r in rating]
  rating = [r.split(",") for r in rating]
  rating = [[int(i) for i in r] for r in rating]
  data["rating"] = np.array(rating)
  return data

def load_rating(path):
  arr = []
  for line in open(path):
    a = line.strip().split()
    if a[0]==0:
      l = []
    else:
      l = [int(x) for x in a[1:]]
    arr.append(l)
  return arr



params = params()
params.lambda_u = 10
params.lambda_v = 10
params.lambda_r = 1
params.C_a = 1
params.C_b = 0.01
params.max_iter_m = 1
params.EM_iter = args.iter
params.num_iter = args.iter


data = load_cvae_data(data_dir)
np.random.seed(0)
tf.set_random_seed(0)

num_factors = zdim


model = neuVAE(num_users=args.user_no, num_items=args.item_no, num_factors=num_factors, params=params,
                      input_dim=8000, encoding_dims=[200, 100], z_dim=zdim, decoding_dims=[100, 200, 8000],
                      loss_type='cross_entropy',
                      model = model_type, ckpt_folder=ckpt, initial=initial, user_dim=args.user_dim)
model.fit(data["rating"], data["content"], data["user"])
model.predict(data["train_users"], data["test_users"], data["content"], data["user"])
