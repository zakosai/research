import numpy as np
from numpy import linalg as LA
import scipy.io as sio

class PMF:
    def __init__(self, num_feat=10, epsilon=1, _lambda=0.1, momentum=0.8, maxepoch=20, num_batches=10, batch_size=1000):
        self.num_feat = num_feat
        self.epsilon = epsilon
        self._lambda = _lambda
        self.momentum = momentum
        self.maxepoch = maxepoch
        self.num_batches = num_batches
        self.batch_size = batch_size
        
        self.w_C = None
        self.w_I = None

        self.err_train = []
        self.err_val = []
        
    def fit(self, train_vec, val_vec, train_users, test_users):
        # mean subtraction
        m = 0
        e_m = 0
        self.mean_inv = np.mean(train_vec[:,2])
        
        pairs_tr = train_vec.shape[0]
        pairs_va = val_vec.shape[0]
        
        # 1-p-i, 2-m-c
        num_inv = int(max(np.amax(train_vec[:,0]), np.amax(val_vec[:,0]))) + 1
        num_com = int(max(np.amax(train_vec[:,1]), np.amax(val_vec[:,1]))) + 1

        incremental = False
        if ((not incremental) or (self.w_C is None)):
            # initialize
            self.epoch = 0
            self.w_C = 0.1 * np.random.randn(num_com, self.num_feat)
            self.w_I = 0.1 * np.random.randn(num_inv, self.num_feat)
            
            self.w_C_inc = np.zeros((num_com, self.num_feat))
            self.w_I_inc = np.zeros((num_inv, self.num_feat))

        while self.epoch < self.maxepoch:
            self.epoch += 1

            # Shuffle training truples
            shuffled_order = np.arange(train_vec.shape[0])
            np.random.shuffle(shuffled_order)

            # Batch update
            for batch in range(self.num_batches):
                # print "epoch %d batch %d" % (self.epoch, batch+1)

                batch_idx = np.mod(np.arange(self.batch_size * batch,
                                             self.batch_size * (batch+1)),
                                   shuffled_order.shape[0])

                batch_invID = np.array(train_vec[shuffled_order[batch_idx], 0], dtype='int32')
                batch_comID = np.array(train_vec[shuffled_order[batch_idx], 1], dtype='int32')

                # Compute Objective Function
                pred_out = np.sum(np.multiply(self.w_I[batch_invID,:], 
                                                self.w_C[batch_comID,:]),
                                axis=1) # mean_inv subtracted

                rawErr = pred_out - train_vec[shuffled_order[batch_idx], 2] + self.mean_inv

                # Compute gradients
                Ix_C = 2 * np.multiply(rawErr[:, np.newaxis], self.w_I[batch_invID,:]) \
                        + self._lambda * self.w_C[batch_comID,:]
                Ix_I = 2 * np.multiply(rawErr[:, np.newaxis], self.w_C[batch_comID,:]) \
                        + self._lambda * self.w_I[batch_invID,:]
            
                dw_C = np.zeros((num_com, self.num_feat))
                dw_I = np.zeros((num_inv, self.num_feat))

                # loop to aggreate the gradients of the same element
                for i in range(self.batch_size):
                    dw_C[batch_comID[i],:] += Ix_C[i,:]
                    dw_I[batch_invID[i],:] += Ix_I[i,:]


                # Update with momentum
                self.w_C_inc = self.momentum * self.w_C_inc + self.epsilon * dw_C / self.batch_size
                self.w_I_inc = self.momentum * self.w_I_inc + self.epsilon * dw_I / self.batch_size


                self.w_C = self.w_C - self.w_C_inc
                self.w_I = self.w_I - self.w_I_inc

                # Compute Objective Function after
                if batch == self.num_batches - 1:
                    pred_out = np.sum(np.multiply(self.w_I[np.array(train_vec[:,0], dtype='int32'),:],
                                                    self.w_C[np.array(train_vec[:,1], dtype='int32'),:]),
                                        axis=1) # mean_inv subtracted
                    rawErr = pred_out - train_vec[:, 2] + self.mean_inv
                    obj = LA.norm(rawErr) ** 2 \
                            + 0.5*self._lambda*(LA.norm(self.w_I) ** 2 + LA.norm(self.w_C) ** 2)

                    self.err_train.append(np.sqrt(obj/pairs_tr))

                # Compute validation error
                if batch == self.num_batches - 1:
                    pred_out = np.sum(np.multiply(self.w_I[np.array(val_vec[:,0], dtype='int32'),:],
                                                    self.w_C[np.array(val_vec[:,1], dtype='int32'),:]),
                                        axis=1) # mean_inv subtracted
                    rawErr = pred_out - val_vec[:, 2] + self.mean_inv
                    self.err_val.append(LA.norm(rawErr)/np.sqrt(pairs_va))

            # recall = self.predict_val(train_users, test_users)
            # if recall > m:
            #     m = recall
            #     e_m = self.epoch

        # print("max recall: %f in epoch %d"%(m, e_m))
                # Print info

                #if batch == self.num_batches - 1:
                #    print 'Training RMSE: %f, Test RMSE %f' % (self.err_train[-1], self.err_val[-1])
    def save_model(self, save_path_pmf):
        # self.saver.save(self.sess, save_path_weights)
        sio.savemat(save_path_pmf, {"U":self.w_I, "V":self.w_C, "mean":self.mean_inv})
        print "all parameters saved"
    def load_model(self, load_path_pmf):
        # self.saver.restore(self.sess, load_path_weights)
        data = sio.loadmat(load_path_pmf)
        self.w_I = data["U"]
        self.w_C = data["V"]
        self.mean_inv = data["mean"]
        print "model loaded"

    def predict(self, invID): 
        return np.dot(self.w_C, self.w_I[invID,:]) + self.mean_inv

    def predict_val(self, train_users, test_users, file=None):
        user_all = test_users
        ground_tr_num = [len(user) for user in user_all]


        recall_avgs = []
        precision_avgs = []
        mapk_avgs = []
        for m in [10]:
            print "m = " + "{:>10d}".format(m) + "done"
            recall_vals = []
            ndcg = []
            hit = 0
            for i in range(len(user_all)):
                train = train_users[i]
                pred = self.predict(i)
                top_M = list(np.argsort(-pred)[0:(m +len(train))])
                for u in train:
                    if u in top_M:
                        top_M.remove(u)
                top_M = top_M[:m]
                if len(top_M) != m:
                    print(top_M, train_users[i])
                hits = set(top_M) & set(user_all[i])   # item idex from 0
                hits_num = len(hits)
                if hits_num > 0:
                    hit += 1
                try:
                    recall_val = float(hits_num) / float(ground_tr_num[i])
                except:
                    recall_val = 1
                recall_vals.append(recall_val)
                score = []
                for k in range(m):
                    if top_M[k] in hits:
                        score.append(1)
                    else:
                        score.append(0)
                actual = self.dcg_score(score, pred[top_M], m)
                best = self.dcg_score(score, score, m)
                if best ==0:
                    ndcg.append(0)
                else:
                    ndcg.append(float(actual)/best)
                # precision = float(hits_num) / float(m)
                # precision_vals.append(precision)

            recall_avg = np.mean(np.array(recall_vals))
            # precision_avg = np.mean(np.array(precision_vals))
            # mapk = ml_metrics.mapk([list(np.argsort(-pred_all[k])) for k in range(len(pred_all)) if len(user_all[k])!= 0],
            #                        [u for u in user_all if len(u)!=0], m)

            print("recall %f, hit: %f, NDCG: %f"%(recall_avg, float(hit)/len(user_all), np.mean(ndcg)))
            #print recall_avg
            if file != None:
                file.write("m = %d, recall = %f\t"%(m, recall_avg))
    def dcg_score(self, y_true, y_score, k=5):
        """Discounted cumulative gain (DCG) at rank K.

        Parameters
        ----------
        y_true : array, shape = [n_samples]
            Ground truth (true relevance labels).
        y_score : array, shape = [n_samples, n_classes]
            Predicted scores.
        k : int
            Rank.

        Returns
        -------
        score : float
        """
        order = np.argsort(y_score)[::-1]
        y_true = np.take(y_true, order[:k])

        gain = 2 ** y_true - 1

        discounts = np.log2(np.arange(len(y_true)) + 2)
        return np.sum(gain / discounts)
        
    def set_params(self, parameters):
        if isinstance(parameters, dict):
            self.num_feat = parameters.get("num_feat", 10)
            self.epsilon = parameters.get("epsilon", 1)
            self._lambda = parameters.get("_lambda", 0.1)
            self.momentum = parameters.get("momentum", 0.8)
            self.maxepoch = parameters.get("maxepoch", 20)
            self.num_batches = parameters.get("num_batches", 10)
            self.batch_size = parameters.get("batch_size", 1000)