import numpy as np
import tensorflow as tf
import scipy.io
import matplotlib.pyplot as plt
from cf_vae_cpmf_extend import cf_vae_extend, params
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
args = parser.parse_args()
model_type = args.model
ckpt = args.ckpt_folder
initial = args.initial
iter = args.iter
data_dir = args.data_dir
print(model_type)

def load_cvae_data(data_dir):
  data = {}
  # variables = scipy.io.loadmat(data_dir + "mult_nor-small.mat")
  # data["content"] = variables['X']
  variables = load_npz(os.path.join(data_dir,"mult_nor-small.npz"))
  data["content"] = variables.toarray()
  # variables = load_npz("data/amazon/structure_mult_nor-small.npz")
  data["structure"] = variables.toarray()
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
params.C_a = 10
params.C_b = 0.01
params.max_iter_m = 1
params.EM_iter = args.iter

C = [0.1, 1]

# # for updating W and b in vae
# self.learning_rate = 0.001
# self.batch_size = 500
# self.num_iter = 3000
# self.EM_iter = 100

data = load_cvae_data(data_dir)
np.random.seed(0)
tf.set_random_seed(0)

images = np.fromfile(os.path.join(data_dir,"images.bin"), dtype=np.uint8)
img = images.reshape((16000, 64, 64, 3))
img = img.astype(np.float32)/255
num_factors = 50


# i = 0
# recalls = []
# for u in C:
#     params.lambda_u = u
#     for v in C:
#         params.lambda_v = v
#         for r in C:
#             params.lambda_r = r
#
#             model = cf_vae_extend(num_users=8000, num_items=16000, num_factors=num_factors, params=params,
#                                   input_dim=8000, encoding_dims=[1000, 200], z_dim = 50, decoding_dims=[200, 1000, 8000],
#                                   decoding_dims_str=[100,200, 1863], loss_type='cross_entropy', model = model_type, ckpt_folder=ckpt, initial=initial)
#             model.fit(data["train_users"], data["train_items"], data["content"],img, data["structure"], params)
#             model.save_model(os.path.join(ckpt,"cf_vae_%d_%d.mat"%(model_type, i)))
#             # model.load_model("cf_vae.mat")
#             pred = model.predict_all()
#             recall = model.predict(pred, data['train_users'], data['test_users'], 10)
#             recalls.append(recall)
#             plt.figure()
#             plt.ylabel("Recall@M")
#             plt.xlabel("M")
#             plt.plot(np.arange(1, 10, 1),recalls)
#             plt.savefig(os.path.join(ckpt, "cvae_%d_%d.png"%(model_type, i)))
#             plt.close()
#
# with open(os.path.join(ckpt, "result_%d.csv"%model_type), "w") as csvfile:
#     wr = csv.writer(csvfile)
#     wr.writerows(recalls)


model = cf_vae_extend(num_users=8000, num_items=16000, num_factors=num_factors, params=params,
                      input_dim=8000, encoding_dims=[1000, 200], z_dim = 50, decoding_dims=[200, 1000, 8000],
                      decoding_dims_str=[100,200, 1863], loss_type='cross_entropy', model = model_type, ckpt_folder=ckpt, initial=initial)
model.fit(data["train_users"], data["train_items"], data["content"],img, data["structure"], params)
model.save_model(os.path.join(ckpt,"cf_vae_%d.mat"%(model_type)))
# model.load_model("cf_vae.mat")
pred = model.predict_all()
recalls = model.predict(pred, data['train_users'], data['test_users'], 40)
# recalls.append(recall)
plt.figure()
plt.ylabel("Recall@M")
plt.xlabel("M")
plt.plot(np.arange(5, 40, 5),recalls)
plt.savefig(os.path.join(ckpt, "cvae_%d.png"%(model_type)))
plt.close()