# pylint: skip-file
import mxnet as mx
import numpy as np
import logging
from math import sqrt
from scipy.sparse import load_npz
from autoencoder import AutoEncoderModel
import os

def load_cvae_data():
  data = {}
  data_dir = "data/citeulike-a/"
  variables = scipy.io.loadmat(data_dir + "mult_nor.mat")
  data["content"] = variables['X']
  variables = load_npz("data/amazon2/mult_nor-small.npz")
  data["content"] = variables.toarray()
  data["train_users"] = load_rating(data_dir + "cf-train-1-users.dat")
  data["train_items"] = load_rating(data_dir + "cf-train-1-items.dat")
  data["test_users"] = load_rating(data_dir + "cf-test-1-users.dat")
  data["test_items"] = load_rating(data_dir + "cf-test-1-items.dat")

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

def load_rating2(path, num_u=8000, num_v=16000):
  R = np.mat(np.zeros((num_u,num_v)))
  fp =open(path)
  for i,line in enumerate(fp):
    segs = line.strip().split(' ')[1:]
    for seg in segs:
        R[i,int(seg)] = 1
  return R

def predict_val(pred_all, train_users, test_users, file=None):
    user_all = test_users
    ground_tr_num = [len(user) for user in user_all]


    pred_all = list(pred_all)

    recall_avgs = []
    precision_avgs = []
    mapk_avgs = []
    for m in [5, 35]:
        print "m = " + "{:>10d}".format(m) + "done"
        recall_vals = []
        for i in range(len(user_all)):
            top_M = list(np.argsort(-pred_all[i])[0:(m +1)])
            if train_users[i] in top_M:
                top_M.remove(train_users[i])
            else:
                top_M = top_M[:-1]
            if len(top_M) != m:
                print(top_M, train_users[i])
            if len(train_users[i]) != 1:
                print(i)
            hits = set(top_M) & set(user_all[i])   # item idex from 0
            hits_num = len(hits)
            try:
                recall_val = float(hits_num) / float(ground_tr_num[i])
            except:
                recall_val = 1
            recall_vals.append(recall_val)
            # precision = float(hits_num) / float(m)
            # precision_vals.append(precision)

        recall_avg = np.mean(np.array(recall_vals))
        # precision_avg = np.mean(np.array(precision_vals))
        # # mapk = ml_metrics.mapk([list(np.argsort(-pred_all[k])) for k in range(len(pred_all)) if len(user_all[k])!= 0],
        # #                        [u for u in user_all if len(u)!=0], m)
        print recall_avg
        file.write("m = %d, recall = %f"%(m, recall_avg))
        # precision_avgs.append(precision_avg)

def predict_all(U, V):
    return np.dot(U, (V.T))

if __name__ == '__main__':
    lambda_u = .1 # lambda_u in CDL
    lambda_v = 10 # lambda_v in CDL
    K = 50  # no of latent vectors in the compact representation
    p = 10 # used for data-folder name
    data_dir ="data/citeulike-a/" # whether to use dummy data
    num_iter = 200
    batch_size = 512

    np.random.seed(1234) # set seed
    lv = 1e-2 # lambda_v/lambda_n in CDL
    dir_save = 'cdl%d' % p
    if not os.path.isdir(dir_save):
        os.system('mkdir %s' % dir_save)
    fp = open(dir_save+'/cdl.log','w')
    print 'p%d: lambda_v/lambda_u/ratio/K: %f/%f/%f/%d' % (p,lambda_v,lambda_u,lv,K)
    fp.write('p%d: lambda_v/lambda_u/ratio/K: %f/%f/%f/%d\n' % \
            (p,lambda_v,lambda_u,lv,K))
    fp.close()
    data = load_cvae_data()
    #
    # variables = load_npz("data/amazon2/mult_nor-small.npz")
    # X = variables.toarray()
    #
    R = load_rating2(data_dir + "cf-train-1-users.dat")

    X = data["content"]
    # R = data["train_users"]
    # set to INFO to see less information during training
    logging.basicConfig(level=logging.DEBUG)
    #ae_model = AutoEncoderModel(mx.gpu(0), [784,500,500,2000,10], pt_dropout=0.2,
    #    internal_act='relu', output_act='relu')

    #mx.cpu() no param needed for cpu.

    ae_model = AutoEncoderModel(mx.gpu(1), [X.shape[1],200,K],
        pt_dropout=0.2, internal_act='relu', output_act='relu')

    train_X = X

    #ae_model.layerwise_pretrain(train_X, 256, 50000, 'sgd', l_rate=0.1, decay=0.0,
    #                         lr_scheduler=mx.misc.FactorScheduler(20000,0.1))
    #V = np.zeros((train_X.shape[0],10))
    V = np.random.rand(train_X.shape[0],K)/10
    lambda_v_rt = np.ones((train_X.shape[0],K))*sqrt(lv)
    U, V, theta, BCD_loss = ae_model.finetune(train_X, R, V, lambda_v_rt, lambda_u,
            lambda_v, dir_save, batch_size,
            num_iter, 'sgd', l_rate=0.1, decay=0.0,
            lr_scheduler=mx.misc.FactorScheduler(20000,0.1))
    #ae_model.save('cdl_pt.arg')
    np.savetxt(dir_save+'/final-U.dat',U,fmt='%.5f',comments='')
    np.savetxt(dir_save+'/final-V.dat',V,fmt='%.5f',comments='')
    np.savetxt(dir_save+'/final-theta.dat',theta,fmt='%.5f',comments='')

    f = open("result_dae.txt", "a")
    f.write("-------%d-------%d--------\n"%(lambda_u, lambda_v))
    pred_all = predict_all(U, V)
    predict_val(pred_all, data["train_users"], data["test_users"], f)
    f.write("\n")
    f.close()

    #ae_model.load('cdl_pt.arg')
    Recon_loss = lambda_v/lv*ae_model.eval(train_X,V,lambda_v_rt)
    print "Training error: %.3f" % (BCD_loss+Recon_loss)
    fp = open(dir_save+'/cdl.log','a')
    fp.write("Training error: %.3f\n" % (BCD_loss+Recon_loss))
    fp.close()
    #print "Validation error:", ae_model.eval(val_X)

