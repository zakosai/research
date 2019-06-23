import pickle
import numpy as np

class Dataset(object):
    def __init__(self, folder, seq_len):
        self.n_user = len(list(open("%s/user.txt"%folder)))
        self.n_item_A = len(list(open("%s/itemA.txt"%folder)))
        self.n_item_B = len(list(open("%s/itemB.txt"%folder)))

        self.dataset = pickle.load(open("%s/dataset.obj"%folder, "rb"))

        self.seq_len = seq_len
        self.max_target_sequence = max([len(i) for i in self.dataset['rating_B']])

        self.eos_A = self.n_item_A
        self.eos_B = self.n_item_B
        self.go = self.n_item_B
        self.n_item_B += 2
        self.n_item_A += 1

        self.emb_A = self.create_emb(self.n_item_A, self.dataset['rating_A'])
        self.emb_B = self.create_emb(self.n_item_B, self.dataset['rating_B'], 1)


    def create_emb(self, n_item, rating, type=0):
        emb = np.zeros((n_item, self.n_user))

        for i, r in enumerate(rating):
            for j in range(len(r)-type):
                emb[i, j] = 1
        return emb


    def create_batch(self, idx):
        target_batch = []
        input_emb_batch = []
        target_emb_batch = []
        target_sequence = []
        max_input_length = max([len(self.dataset['rating_A'][i]) for i in idx])
        max_target_legth = max([len(self.dataset['rating_B'][i]) for i in idx])

        for i in idx:
            tmp_input = self.dataset['rating_A'][i]
            tmp_input = tmp_input + [self.eos_A]*(max_input_length - len(tmp_input))
            input_emb_batch.append(self.emb_A[tmp_input])

            tmp_target = self.dataset['rating_B'][i]
            target_sequence.append(len(tmp_target))
            tmp_target = tmp_target + [self.emb_B]*(max_target_legth - len(tmp_target))
            target_batch.append(tmp_target)
            tmp_target = [self.go] + tmp_target[:-1]

            # for k, j in enumerate(tmp_target):
            #     try:
            #         tmp = self.emb_B[j]
            #     except:
            #         print(j, k)
            #         print(i)
            target_emb_batch.append(self.emb_B[tmp_target])


        return (np.array(input_emb_batch).reshape((len(idx), max_input_length, self.n_user)),
                np.array(target_batch).reshape((len(idx), max_target_legth)),
                np.array(target_emb_batch).reshape((len(idx), max_target_legth, self.n_user)),
                target_sequence)


# np.set_printoptions(threshold=np.inf)






