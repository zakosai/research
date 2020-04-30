'''
Created on Aug 8, 2016
Processing datasets. 

@author: Xiangnan He (xiangnanhe@gmail.com)
'''
import scipy.sparse as sp
import numpy as np
import os

class Dataset(object):
    '''
    classdocs
    '''

    def __init__(self, path):
        '''
        Constructor
        '''
        self.trainMatrix = self.load_rating_file_as_matrix(path + ".train.rating")
        self.testRatings = self.load_rating_file_as_list(path + ".test.rating")
        self.testNegatives = self.load_negative_file(path + ".test.negative")
        assert len(self.testRatings) == len(self.testNegatives)
        
        self.num_users, self.num_items = self.trainMatrix.shape
        
    def load_rating_file_as_list(self, filename):
        ratingList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item = int(arr[0]), int(arr[1])
                ratingList.append([user, item])
                line = f.readline()
        return ratingList
    
    def load_negative_file(self, filename):
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                negatives = []
                for x in arr[1: ]:
                    negatives.append(int(x))
                negativeList.append(negatives)
                line = f.readline()
        return negativeList
    
    def load_rating_file_as_matrix(self, filename):
        '''
        Read .rating file and Return dok matrix.
        The first line of .rating file is: num_users\t num_items
        '''
        # Get number of users and items
        num_users, num_items = 0, 0
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                line = f.readline()
        # Construct matrix
        mat = sp.dok_matrix((num_users+1, num_items+1), dtype=np.float32)
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                if (rating > 0):
                    mat[user, item] = 1.0
                line = f.readline()    
        return mat


class Dataset2(object):
    def __init__(self, data_dir, data_type):
        data = self.load_cvae_data(data_dir, data_type)
        self.train = data['train_users']
        self.test = data['test_users']

    def load_cvae_data(self, data_dir, data_type):
        data = {}
        data["train_users"] = self.load_rating(data_dir + "cf-train-%s-users.dat" % data_type)
        data["train_items"] = self.load_rating(data_dir + "cf-train-%s-items.dat" % data_type)
        data["test_users"] = self.load_rating(data_dir + "cf-test-%s-users.dat" % data_type)
        data["test_items"] = self.load_rating(data_dir + "cf-test-%s-items.dat" % data_type)
        return data

    def load_rating(self, path):
        arr = []
        for line in open(path):
            a = line.strip().split()
            if a[0] == 0:
                l = []
            else:
                l = [int(x) for x in a[1:]]
            arr.append(l)
        return arr


def recallK(train, test, predict, k=50):
    recall = []
    ndcg = []
    mAP = []
    for i in range(len(train)):
        pred = np.argsort(predict[i])[::-1][:k+len(train[i])]
        pred = [item for item in pred if item not in train[i]]
        pred = pred[:k]
        hits = [p for p in pred if p in test[i]]
        recall.append(float(len(hits))/len(test[i]))

        # ncdg
        score = [1 if p in hits else 0 for p in pred]
        actual = dcg_score(score, predict[i, pred], k)
        best = dcg_score(score, score, k)
        if best == 0:
            ndcg.append(0)
        else:
            ndcg.append(float(actual) / best)

        # mAP
        mAP.append(mAP_score(test[i], pred))

    return np.mean(recall), np.mean(ndcg), np.mean(mAP)


def dcg_score(y_true, y_score, k=50):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])

    gain = 2 ** y_true - 1

    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gain / discounts)


def mAP_score(test, pred):
    AP = 0
    j = 1
    for i in range(len(pred)):
        if pred[i] in test:
            AP += float(j)/(i+1)
            j += 1
    return AP
