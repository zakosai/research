import pandas as pd
import gzip
import argparse
import sys
import os
import pickle
import re
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
import numpy as np




class Dataset(object):
    def __init__(self, data, max_sequence_length=1000, max_nb_words=20000, folder="data/", embedding_dim=100):
        self.data = data
        self.max_sequence_length = max_sequence_length
        self.max_nb_words = max_nb_words
        texts = [i[2] for i in self.data['train']] + [i[2] for i in self.data['test']]
        self.tokenizer = Tokenizer(num_words=self.max_nb_words)
        self.tokenizer.fit_on_texts(texts)
        self.embedding_dim = embedding_dim

        self.embeddings_index = {}
        f = open(folder + 'glove.6B.100d.txt', encoding='utf8')
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


    def create_batch(self, idx, k=2, type='train'):
        data_b = self.data[type].iloc[idx]
        y_rating = list(data_b[3]) - 1
        y_review = self.tokenizer.texts_to_sequences(list(data_b[2]))
        y_review = pad_sequences(y_review, maxlen=self.max_sequence_length)

        #Create X
        sequences_user = []
        sequences_item = []
        for i, d in data_b.iterrows():
            user = self.data['train_user'][d[0]]
            u_ids = list(np.random.randint(0, len(user[0]), min(k, len(user[0]))))
            seq = [user[1][j] for j in u_ids]
            sequences_user.append(' '.join(seq))

            item = self.data['train_item'][d[1]]
            i_ids = list(np.random.randint(0, len(item[0]), k))
            seq = [item[1][j] for j in i_ids]
            sequences_item.append(' '.join(seq))
        X_user = self.tokenizer.texts_to_sequences(sequences_user)
        X_user = pad_sequences(X_user, maxlen=self.max_sequence_length)
        X_item = self.tokenizer.texts_to_sequences(sequences_item)
        X_item = pad_sequences(X_item, maxlen=self.max_sequence_length)

        return X_user, X_item, y_review, y_rating

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
            if uid in train:
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

def main():
    args = parse_args()
    amz_to_pkl(args)

if __name__ == '__main__':
    main()