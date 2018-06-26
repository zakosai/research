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
parser.add_argument('--data_type',  type=str, default='5',
                   help='where model is stored')
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
data_type = args.data_type
print(model_type)

def load_cvae_data(data_dir):
  data = {}
  # variables = scipy.io.loadmat(data_dir + "mult_nor.mat")
  # data["content"] = variables['X']
  variables = load_npz(os.path.join(data_dir,"mult_nor.npz"))
  data["content"] = variables.toarray()

  data["train_users"] = load_rating(data_dir + "cf-train-%sp-users.dat"%data_type)
  data["train_items"] = load_rating(data_dir + "cf-train-%sp-items.dat"%data_type)
  data["test_users"] = load_rating(data_dir + "cf-test-%sp-users.dat"%data_type)
  data["test_items"] = load_rating(data_dir + "cf-test-%sp-items.dat"%data_type)

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
params.lambda_u = 1
params.lambda_v = 10
params.lambda_r = 0.1
params.C_a = 1
params.C_b = 0.01
params.max_iter_m = 1
params.EM_iter = iter
params.num_iter = 150



# # for updating W and b in vae
# self.learning_rate = 0.001
# self.batch_size = 500
# self.num_iter = 3000
# self.EM_iter = 100


data = load_cvae_data(data_dir)
np.random.seed(0)
tf.set_random_seed(0)

num_factors = 50

i = 27
recalls = []
for u in [10]:
    params.lambda_u = u
    for v in [1, 10, 100]:
        params.lambda_v = v
        for r in [0.1, 1, 10]:
            params.lambda_r = r

            model = cf_vae_extend(num_users=args.user_no, num_items=args.item_no, num_factors=num_factors, params=params,
                input_dim=8000, encoding_dims=[200, 100], z_dim = 50, decoding_dims=[100, 200, 8000],
                decoding_dims_str=[100,200, 4526], loss_type='cross_entropy', ckpt_folder=ckpt, model=model_type)
            model.fit(data["train_users"], data["train_items"], data["content"], params, data["test_users"])
            model.save_model(os.path.join(ckpt,"cf_dae_%d_%d.mat"%(model_type, i)))
            # model.load_model("cf_vae.mat")
            f = open(os.path.join(ckpt, "result_dae_%d.txt"%model_type), 'a')
            f.write("-----------%f----------%f----------%f\n"%(u,v,r))
            pred_all = model.predict_all()
            model.predict_val(pred_all, data["train_users"], data["test_users"], f)
            f.write("\n")
            f.close()
            print(u, v, r)
            i += 1

#
# model = cf_vae_extend(num_users=5551, num_items=16980, num_factors=num_factors, params=params,
#                 input_dim=8000, encoding_dims=[200, 100], z_dim = 50, decoding_dims=[100, 200, 8000],
#                 decoding_dims_str=[100,200, 1863], loss_type='cross_entropy', ckpt_folder=ckpt, model=model_type)
# model.fit(data["train_users"], data["train_items"], data["content"],img, data["content"], params, data["test_users"])
# model.save_model(os.path.join(ckpt,"cf_dae_%d.mat"%(model_type)))

# plt.figure()
# plt.ylabel("Recall@M")
# plt.xlabel("M")
# plt.plot(np.arange(5, 40, 5),recalls)
# plt.show()
