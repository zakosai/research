__author__ = 'linh'

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import load_npz
from cf_vae_cpmf_extend import cf_vae_extend, params


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
model = cf_vae_extend(num_users=8000, num_items=16000, num_factors=num_factors, params=params,
    input_dim=8000, encoding_dims=[2000, 1000], z_dim = 500, decoding_dims=[1000, 2000, 8000], decoding_dims_str=[100,200, 1863],
    loss_type='cross_entropy')
model.load_model("pre_model/zdim2/cf_vae_0.mat")
# model.load_model("cf_vae.mat")
pred = model.predict_all()
recalls = model.predict(pred, data['train_users'], data['test_users'], 10)

images = np.fromfile("data/amazon/images.bin", dtype=np.uint8)
img = images.reshape((16000, 64, 64, 3))
img = img.astype(np.float32)/255
# num_factors = 50
model_im = cf_vae_extend(num_users=8000, num_items=16000, num_factors=num_factors, params=params,
    input_dim=8000, encoding_dims=[2000, 1000], z_dim = 500, decoding_dims=[1000, 2000, 8000], decoding_dims_str=[100,200, 1863],
    loss_type='cross_entropy')
model_im.load_model("pre_model/zdim2/cf_vae_1.mat")
# model.load_model("cf_vae.mat")
pred_im = model_im.predict_all()
recalls_im= model_im.predict(pred_im, data['train_users'], data['test_users'], 10)

plt.figure()
plt.ylabel("Recall@M")
plt.xlabel("M")
plt.plot(np.arange(1, 10, 1),recalls, '-b', label="cf-vae")
plt.plot(np.arange(1, 10, 1), recalls_im, '-r', label="img-extend")
plt.legend(loc='upper left')
plt.savefig("result/cf-vae-extend-result_zdim500_M10.png")