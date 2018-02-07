__author__ = 'linh'

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import load_npz
from cf_vae_cpmf import cf_vae, params


def load_cvae_data():
  data = {}
  data_dir = "data/amazon/"
  # variables = scipy.io.loadmat(data_dir + "mult_nor.mat")
  # data["content"] = variables['X']
  variables = load_npz("data/amazon/mult_nor-small.npz")
  data["content"] = variables.toarray()
  data["train_users"] = load_rating(data_dir + "cf-train-1-users-small.dat")
  data["train_items"] = load_rating(data_dir + "cf-train-1-items-small.dat")
  data["test_users"] = load_rating(data_dir + "cf-test-1-users-small.dat")
  data["test_items"] = load_rating(data_dir + "cf-test-1-items-small.dat")

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
params.lambda_u = 0.1
params.lambda_v = 10
params.lambda_r = 1
params.C_a = 1
params.C_b = 0.01
params.max_iter_m = 1

data = load_cvae_data()
num_factors = 50
model = cf_vae(num_users=8000, num_items=16000, num_factors=num_factors, params=params,
    input_dim=8000, encoding_dims=[200, 100], z_dim = 50, decoding_dims=[100, 200, 8000],
    loss_type='cross_entropy')
model.load_model("cf_vae.mat")
# model.load_model("cf_vae.mat")
pred = model.predict_all()
recalls = model.predict(pred, data['train_users'], data['test_users'], 30)

plt.figure()
plt.ylabel("Recall@M")
plt.xlabel("M")
plt.plot(np.arange(5, 30, 5),recalls)
plt.savefig("result/cf-vae-result.png")