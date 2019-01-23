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
import pickle

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
print(model_type)

def load_cvae_data(data_dir):


  f = open(os.path.join(data_dir,"dataset.pkl"), 'rb')
  dataset = pickle.load(f)
  variables = load_npz(os.path.join(data_dir, "mult_nor.npz"))
  dataset["content"] = variables.toarray()

  train_item = [0] * dataset['item_no']
  train_tag = [0] * dataset['tag_no']
  tag_list = np.where(dataset['tag_item_onehot'] == 1)
  for i in range(len(tag_list[0])):
      item_id = tag_list[0][i]
      tag_id = tag_list[1][i]
      if train_item[item_id] == 0:
          train_item[item_id] = [tag_id]
      else:
          train_item[item_id].append(tag_id)

      if train_tag[tag_id] == 0:
          train_tag[tag_id] = [item_id]
      else:
          train_tag[tag_id].append(item_id)
  for i in range(len(train_item)):
      if train_item[i] == 0:
          train_item[i] = []
  for i in range(len(train_tag)):
      if train_tag[i] == 0:
          train_tag[i] = []


  dataset["train_users"] = train_item
  dataset["train_items"] = train_tag
  dataset["test_users"] = dataset['user_item_test']
  return dataset


def load_rating(path):
  arr = []
  for line in open(path):
    a = line.strip().split()
    if a == []:
      l = []
    else:
      l = [int(x) for x in a[1:]]
    arr.append(l)
  return arr



params = params()
params.lambda_u = 10
params.lambda_v = 1
params.lambda_r = 10
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

data = load_cvae_data(data_dir)
np.random.seed(0)
tf.set_random_seed(0)

# images = np.fromfile(os.path.join(data_dir,"images.bin"), dtype=np.uint8)
# img = images.reshape((13791, 32, 32, 3))
# img = img.astype(np.float32)/255
num_factors = 50
best_recall = 0
best_hyper = []
dim = data['content'].shape[1]


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
                    model = cf_vae_extend(num_users=data['item_no'], num_items=data['tag_no'], num_factors=num_factors,
                                          params=params,input_dim=dim, encoding_dims=[400, 200], z_dim=zdim,
                                          decoding_dims=[200,400,dim], decoding_dims_str=[200, 4526], loss_type='cross_entropy',
                                          model = model_type, ckpt_folder=ckpt)
                    model.fit(data["train_users"], data["train_items"], data["content"], params, data["test_users"])
                    model.save_model(os.path.join(ckpt,"cf_dae_%d_%d.mat"%(model_type, i)))
                    # model.load_model("cf_vae.mat")
                    f = open(os.path.join(ckpt, "result_cdae_%d.txt"%model_type), 'a')
                    f.write("%d-----------%f----------%f----------%f\n"%(i,u,v,r))
                    pred_all = model.predict_all()
                    recall = model.predict_val(pred_all, data["train_users"], data["test_users"], f)
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
    model.fit(data["train_users"], data["train_items"], data["content"],params, data["test_users"])
    model.save_model(os.path.join(ckpt,"cf_dae_%d.mat"%(model_type)))
    #model.load_model(os.path.join(ckpt, "cf_dae_0.mat"))
    pred = model.predict_all()
    model.predict_val(pred, data["train_users"], data["test_users"])

