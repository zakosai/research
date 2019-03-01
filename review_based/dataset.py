import pandas as pd
import gzip
import argparse
import sys
import os
import pickle
import re
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

class Dataset(object):
    def __init__(self, data, cnn=False, max_sequence_length=1000, max_nb_words=20000, folder="data/",
                 embedding_dim=300):
        f = open(os.path.join(data, "dataset.pkl"), "rb")
        self.data = pickle.load(f)
        f.close()
        if cnn:
            self.max_sequence_length = max_sequence_length
            self.max_nb_words = max_nb_words
            texts = list(self.data['train'][2]) + list(self.data['test'][2])
            self.tokenizer = Tokenizer(num_words=self.max_nb_words)
            self.tokenizer.fit_on_texts(texts)
            self.embedding_dim = embedding_dim

            self.embeddings_index = {}
            f = open(folder + 'glove.6B.300d.txt', encoding='utf8')
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                self.embeddings_index[word] = coefs
            f.close()

            word_index = self.tokenizer.word_index
            self.vocab_size = len(word_index) + 1
            self.embedding_matrix = np.random.random((len(word_index) + 1, embedding_dim))
            for word, i in word_index.items():
                embedding_vector = self.embeddings_index.get(word)
                if embedding_vector is not None:
                    # words not found in embedding index will be all-zeros.
                    self.embedding_matrix[i] = embedding_vector

        # tf-idf
        else:
            text = self.data['train'][2]
            tfidf = TfidfVectorizer(stop_words='english', max_features=20000)
            self.y_review = tfidf.fit_transform(text.astype('U'), self.data['train'][3])
            text = []
            for u in range(self.data['user_no']):
                user = self.data['train_user'][u]
                text.append(' '.join(user[1]))
            self.X_user = tfidf.transform(text).toarray()
            text = []
            for i in range(self.data['item_no']):
                try:
                    item = self.data['train_item'][i]
                    text.append(' '.join(item[1]))
                except:
                    text.append(' ')
            self.X_item = tfidf.transform(text).toarray()

            self.user_onehot = np.zeros((self.data['user_no'], self.data['item_no']))
            for user in self.data['train_user']:
                d = self.data['train_user'][user]
                for i, j in enumerate(d[0]):
                    self.user_onehot[user, j] = d[2][i]
            self.description = np.load(os.path.join(data, "description.npy"))
            print(self.description.shape)


    def create_batch(self, idx, k=2, type='train'):
        data_b = self.data[type].iloc[idx]
        y_rating = np.array(data_b[3])
        y_review = self.tokenizer.texts_to_sequences(list(data_b[2]))
        y_review = pad_sequences(y_review, maxlen=self.max_sequence_length)

        #Create X
        sequences_user = []
        sequences_item = []
        for i, d in data_b.iterrows():
            user = self.data['train_user'][d[0]]
            if type == "train":
                ids = list(set(range(len(user[0]))) - set([user[0].index(d[1])]))
            else:
                ids = list(range(len(user[0])))
            ids = np.random.permutation(ids)
            u_ids = ids[:min(len(user[0]), k)]
            seq = [user[1][j] for j in u_ids]
            sequences_user.append(' '.join(seq))

            try:
                item = self.data['train_item'][d[1]]
                if type == "train":
                    ids = list(set(range(len(item[0]))) - set([item[0].index(d[0])]))
                else:
                    ids = list(range(len(item[0])))
                ids = np.random.permutation(ids)
                i_ids = ids[:min(len(user[0]), k)]
                seq = [item[1][j] for j in i_ids]
                sequences_item.append(' '.join(seq))
            except:
                sequences_item.append(' ')
        X_user = self.tokenizer.texts_to_sequences(sequences_user)
        X_user = pad_sequences(X_user, maxlen=self.max_sequence_length)
        X_item = self.tokenizer.texts_to_sequences(sequences_item)
        X_item = pad_sequences(X_item, maxlen=self.max_sequence_length)

        return X_user, X_item, y_review, y_rating

    def create_implicit_batch(self, idx, type="train"):
        data_b = self.data[type].iloc[idx]
        y_rating = np.array(data_b[3])

        sequences_user = []
        sequences_item = []
        for i, d in data_b.iterrows():
            user = self.data['train_user'][d[0]]
            seq = np.zeros(self.data['item_no'])
            for k, j in enumerate(user[0]):
                if j != d[1]:
                    seq[j] = user[2][k]
            sequences_user.append(seq)


            seq = np.zeros(self.data['user_no'])
            try:
                item = self.data['train_item'][d[0]]
                for k, j in enumerate(item[0]):
                    if j != d[0]:
                        seq[j] = item[2][k]
            except:
                pass
            sequences_item.append(seq)

        return sequences_user, sequences_item, y_rating

    def create_tfidf(self, idx, k=2, type='train'):
        data_b = self.data[type].iloc[idx]
        y_rating = np.array(data_b[3])


        # Create X
        sequences_user = []
        sequences_item = []
        for i, d in data_b.iterrows():
            user = self.data['train_user'][d[0]]
            if type == "train":
                ids = list(set(range(len(user[0]))) - set([user[0].index(d[1])]))
            else:
                ids = list(range(len(user[0])))
            ids = np.random.permutation(ids)
            u_ids = ids[:min(len(user[0]), k)]
            seq = [user[1][j] for j in u_ids]
            sequences_user.append(' '.join(seq))

            try:
                item = self.data['train_item'][d[1]]
                if type == "train":
                    ids = list(set(range(len(item[0]))) - set([item[0].index(d[0])]))
                else:
                    ids = list(range(len(item[0])))
                ids = np.random.permutation(ids)
                i_ids = ids[:min(len(user[0]), k)]
                seq = [item[1][j] for j in i_ids]
                sequences_item.append(' '.join(seq))
            except:
                sequences_item.append(' ')
        X_user = self.tfidf.transform(sequences_user).toarray()
        X_item = self.tfidf.transform(sequences_item).toarray()

        return X_user, X_item, y_rating

    def create_tfidf_full(self, idx, k=2, type='train'):
        data_b = self.data[type].iloc[idx]
        y_rating = np.array(data_b[3])
        X_user = self.X_user[data_b[0]]
        X_item = self.X_item[data_b[1]]
        user_onehot = self.user_onehot[data_b[0]]
        item_onehot = self.user_onehot.T[data_b[1]]
        description = self.description[data_b[1], :]
        # user_onehot = np.concatenate((X_user, user_onehot), axis=1)
        # item_onehot = np.concatenate((X_item, item_onehot), axis=1)
        if type=='train':
            y_review = self.y_review[idx].toarray()
        else:
            y_review = data_b[2]
        # X_user = np.concatenate((X_user, user_onehot), axis=1)
        # X_item = np.concatenate((X_item, item_onehot), axis=1)

        # Create X
        # sequences_user = np.zeros((len(idx), self.data['item_no']))
        # sequences_item = np.zeros((len(idx)))
        # for i, d in data_b.iterrows():
        #     user = self.data['train_user'][d[0]]
        #     sequences_user.append(' '.join(user[1]))
        #
        #     try:
        #         item = self.data['train_item'][d[1]]
        #         sequences_item.append(' '.join(item[1]))
        #     except:
        #         sequences_item.append(' ')
        # X_user = self.tfidf.transform(sequences_user).toarray()
        # X_item = self.tfidf.transform(sequences_item).toarray()

        return X_user, X_item, user_onehot, item_onehot, y_rating, y_review, description

    def create_tfidf_neg(self, idx, k=2, type='train'):
        data_b = self.data[type].iloc[idx]
        y_rating = np.array(data_b[3])
        X_user = self.X_user[data_b[0]]
        X_item = self.X_item[data_b[1]]

        # Create X
        # sequences_user = []
        # sequences_item = []
        neg_items = []
        for i, d in data_b.iterrows():
            user = self.data['train_user'][d[0]]
            r = np.random.randint(self.data['item_no'])
            while r in user[0]:
                r = np.random.randint(self.data['item_no'])
            neg_items.append(r)

        y_rating = np.concatenate((y_rating, np.zeros(len(neg_items))))
        X_user = np.concatenate((X_user, X_user), axis=0)
        X_item = np.concatenate((X_item, self.X_item[neg_items]), axis=0)

        #
        #     try:
        #         item = self.data['train_item'][d[1]]
        #         sequences_item.append(' '.join(item[1]))
        #     except:
        #         sequences_item.append(' ')
        # X_user = self.tfidf.transform(sequences_user).toarray()
        # X_item = self.tfidf.transform(sequences_item).toarray()

        return X_user, X_item, y_rating


def parse_args():
    parser = argparse.ArgumentParser(
        description='Prepare dataset'
    )
    parser.add_argument(
        '--data',
        default='data/review.json',
        dest='data',
        help='data file',
        type=str
    )
    parser.add_argument(
        '--output_folder',
        default='',
        dest='folder',
        help='where to store dataset.pkl',
        type=str
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)

def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')

def clean_str(string):
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()

def amz_to_pkl(args):
    data = getDF(args.data)

    # drop non review
    drop = []
    for i, r in enumerate(data['reviewText']):
        if r == "":
            drop.append(i)
    data.drop(data.index[drop], inplace=True)
    data = data.reset_index(drop=True)
    if args.folder == '':
        folder = args.data.split("/")[:-1]
        folder = "/".join(folder)
    else:
        folder = args.folder

    # sort data
    data.sort_values(by='unixReviewTime', inplace=True, ascending=False)

    users = []
    items = []
    test = []
    train = []
    train_user = {}
    train_item = {}
    for i, d in data.iterrows():
        try:
            uid = users.index(d.reviewerID)
            flag_test = False
        except:
            uid = len(users)
            users.append(d.reviewerID)
            flag_test = True

        try:
            iid = items.index(d.asin)
        except:
            iid = len(items)
            items.append(d.asin)

        if flag_test:
            test.append([uid, iid, clean_str(d.reviewText), int(d.overall)])
        else:
            train.append([uid, iid, clean_str(d.reviewText), int(d.overall)])
            if uid in train_user:
                train_user[uid][0].append(iid)
                train_user[uid][1].append(clean_str(d.reviewText))
                train_user[uid][2].append(int(d.overall))

            else:
                train_user[uid] = [[iid], [clean_str(d.reviewText)], [int(d.overall)]]
            if iid not in train_item:
                train_item[iid] = [[uid], [clean_str(d.reviewText)], [int(d.overall)]]
            else:
                train_item[iid][0].append(uid)
                train_item[iid][1].append(clean_str(d.reviewText))
                train_item[iid][2].append(int(d.overall))

    # Write user, item id
    f = open(os.path.join(folder, "user_id.txt"), "w")
    f.write('\n'.join(users))
    f.close()

    f = open(os.path.join(folder, "item_id.txt"), "w")
    f.write('\n'.join(items))
    f.close()

    # write preprocessed data
    dataset = {'user_no': len(users),
               'item_no': len(items),
               'train': pd.DataFrame(train),
               'test': pd.DataFrame(test),
               'train_user': train_user,
               'train_item': train_item}
    f = open(os.path.join(folder, "dataset.pkl"), "wb")
    pickle.dump(dataset, f)
    print("Finish writing data to %s"%(os.path.join(folder, "dataset.pkl")))

def description(args):
    item = list(open(os.path.join("data", args.data, "item_id.txt")))
    item = [i.strip() for i in item]

    data = getDF(os.path.join("../cf-vae/data", args.data, "meta.json.gz"))
    text = [0] * len(item)
    count = 0
    for _, i in data.iterrows():
        try:
            idx = item.index(i.asin)
            text[idx] = str(i.description) + str(i.title)
            count += 1
        except:
            continue

    for i in range(len(item)):
        if text[i] == 0:
            text[i] = ""

    tfidf = TfidfVectorizer(stop_words='english', max_features=8000)
    description = tfidf.fit_transform(text).toarray()
    np.save(os.path.join("data", args.data, "description"), description)

def main():
    args = parse_args()
    description(args)

if __name__ == '__main__':
    main()