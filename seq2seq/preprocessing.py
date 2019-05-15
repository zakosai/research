import pandas as pd
import numpy as np
import gzip
from gensim.parsing.preprocessing import remove_stopwords
import gensim
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz

def parse(path):
    g = gzip.open(path, "rb")
    for l in g:
        yield eval(l)


def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')

def create_amazon(dir_r, type, fsum):
    fsum.write("-----------------------------------------------\n %s - %s\n"%(type, dir_r))
    data = getDF("%s/reviews.json.gz"%dir_r) #read review
    os.makedirs("data/%s"%type)

    item_unique = sorted(data.asin.unique())
    user_unique = sorted(data.reviewerID.unique())
    print("# num of user: %d \n# num of item: %d" % (len(user_unique), len(item_unique)))
    fsum.write("# num of user: %d \n# num of item: %d\n" % (len(user_unique), len(item_unique)))

    # Group data following user

    n_user = len(user_unique)
    n_item = len(item_unique)

    ratings = [0] * n_user
    f = open("data/%s/ratings.txt" % type, "w")
    for _, r in data.iterrows():
        uid = user_unique.index(r.reviewerID)
        iid = item_unique.index(r.asin)
        if ratings[uid] == 0:
            ratings[uid] = [[iid, r.overall, r.reviewTime]]
        else:
            ratings[uid].append([iid, r.overall, r.reviewTime])
        f.write("%d::%i::%d::%s\n" % (uid, iid, int(r.overall), r.reviewTime))
    f.close()

    print("Max item user rated: %d" % max([len(i) for i in ratings]))
    print("Min item user rated: %d" % min([len(i) for i in ratings]))
    print("Mean item user rated: %d" % np.mean([len(i) for i in ratings]))

    fsum.write("Max item user rated: %d\n" % max([len(i) for i in ratings]))
    fsum.write("Min item user rated: %d\n" % min([len(i) for i in ratings]))
    fsum.write("Mean item user rated: %d\n" % np.mean([len(i) for i in ratings]))

    # write new id of items into file
    f = open("data/%s/item_id.txt" % type, "w")
    f.write("\n".join(item_unique))
    f.close()

    f = open("data/%s/user_id.txt" % type, "w")
    f.write("\n".join(user_unique))
    f.close()


    # Prepare train, test
    shuffle_id = np.random.permutation(n_user)
    train_len = int(0.7 * n_user)
    train_id = shuffle_id[:train_len]
    test_id = shuffle_id[train_len:]

    os.mkdir("data/%s/implicit"%type)
    ftrain = open("data/%s/implicit/train.txt" % type, "w")
    for idx in train_id:
        user = np.array(ratings[idx]).reshape((len(ratings[idx]), 3))
        user = user[np.argsort(user[:, 2])]
        item = list(user[:, 0])
        item = [str(i) for i in item]
        ftrain.write("%d %s\n" % (idx, " ".join(item)))
    ftrain.close()

    ftest = open("data/%s/implicit/test.txt" % type, "w")
    for idx in test_id:
        user = np.array(ratings[idx]).reshape((len(ratings[idx]), 3))
        user = user[np.argsort(user[:, 2])]
        item = list(user[:, 0])
        item = [str(i) for i in item]
        ftest.write("%d %s\n" % (idx, " ".join(item)))
    ftest.close()


    # Hybrid
    data_item = getDF("%s/meta.json.gz" % dir_r)
    data_item = data_item[data_item.asin.isin(item_unique)]
    text = []
    list_cat = []
    for i in item_unique:
        d = data_item[data_item.asin == i]
        text += (d.title + d.description).tolist()
        list_cat.append(d.categories.tolist())

    # write file
    text = [gensim.utils.simple_preprocess(str(t)) for t in text]
    text = [' '.join(t) for t in text]
    f = open("data/%s/description_fix.txt" % type, "w")
    f.write("\n".join(text))
    f.close()

    # tf-idf

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(text)
    save_npz("data/%s/item.npz" % type, X)

    categories = data_item.categories.tolist()
    categories = [i for cat in categories for c in cat for i in c]
    categories = list(set(categories))
    len(categories)

    f = open("data/%s/categories.txt" % type, "w")
    for c in list_cat:
        arr = ['0'] * len(categories)
        for i in c[0][0]:
            arr[categories.index(i)] = '1'
        f.write(",".join(arr))
    f.close()

if __name__ == '__main__':
    dataset = ["CD", "Grocery", "Kitchen", "Office", "Outdoor"]
    fsum = open("data/amazon/summary.txt", "w")
    for type in dataset:
        dir_r = "../cf-vae/data/%s/"%type
        create_amazon(dir_r, type, fsum)



