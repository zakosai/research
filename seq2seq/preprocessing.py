import pandas as pd
import numpy as np
import gzip
from gensim.parsing.preprocessing import remove_stopwords
import gensim
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz
from datetime import datetime
import argparse

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
    fsum.write("-----------------------------------------------\n %s - %s\n" % (type, dir_r))
    data = getDF("%s/reviews.json.gz" % dir_r) #read review
    if not os.path.exists("data/%s" % type):
        os.makedirs("data/%s" % type)
    data.sort_values(by=['reviewerID', 'unixReviewTime'], inplace=True)

    group = data.groupby(['reviewerID'])
    len_group = group.apply(lambda x: len(x))
    user_unique = len_group[len_group >= 15].keys().tolist()
    item_unique = sorted(data.asin.unique())
    n_user = len(user_unique)
    n_item = len(item_unique)
    print("# num of user: %d \n# num of item: %d" % (n_user, n_item))
    fsum.write("# num of user: %d \n# num of item: %d\n" % (n_user, n_item))

    # Group data following user
    ratings = [0] * n_user
    f = open("data/%s/ratings.txt" % type, "w")
    for _, r in data.iterrows():
        if r.reviewerID in user_unique:
            uid = user_unique.index(r.reviewerID)
            iid = item_unique.index(r.asin)
            if ratings[uid] == 0:
                ratings[uid] = [[iid, r.overall, r.unixReviewTime]]
            else:
                ratings[uid].append([iid, r.overall, r.unixReviewTime])
            f.write("%d,%i,%d,%s\n" % (uid, iid, int(r.overall), r.unixReviewTime))
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

    # os.mkdir("data/%s/implicit"%type)
    ftrain = open("data/%s/train.txt" % type, "w")
    for idx in train_id:
        user = np.array(ratings[idx]).reshape((len(ratings[idx]), 3)).astype(np.int32)
        user = user[np.argsort(user[:, 2])]
        item = list(user[:, 0])
        item = [str(i) for i in item]
        ftrain.write("%d %s\n" % (idx, " ".join(item)))
    ftrain.close()
    ftest = open("data/%s/test.txt" % type, "w")
    for idx in test_id:
        user = np.array(ratings[idx]).reshape((len(ratings[idx]), 3)).astype(np.int32)
        user = user[np.argsort(user[:, 2])]
        item = list(user[:, 0])
        item = [str(i) for i in item]
        ftest.write("%d %s\n" % (idx, " ".join(item)))
    ftest.close()

    fsum.write("Train num: %d, test num: %d\n"%(len(train_id), len(test_id)))

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

    # Category
    categories = data_item.categories.tolist()
    categories = [i for cat in categories for c in cat for i in c]
    categories = list(set(categories))
    f = open("data/%s/categories.txt" % type, "w")
    for c in list_cat:
        arr = ['0'] * len(categories)
        for i in c[0][0]:
            arr[categories.index(i)] = '1'
        f.write(",".join(arr))
        f.write("\n")
    f.close()
    f = open("data/%s/list_categories.txt"%type, "w")
    f.write("\n".join(categories))
    f.close()

    fsum.write("text length: %d - cat length: %d\n"%(X.shape[1], len(categories)))


def create_amazon_based_on_ratings(dir_r, type, fsum):
    fsum.write("-----------------------------------------------\n %s - %s for rating \n" % (type, dir_r))
    dir_w = "data/%s/ratings"%type
    if not os.path.exists(dir_w):
        os.makedirs(dir_w)

    ratings = pd.read_csv("%s/ratings.csv"%dir_r, header=None)
    ratings.rename(index=str, columns={0: "user", 1: "item", 2: "rating", 3: "timestamp"}, inplace=True)

    # group user, pick ones who rated more than 10 items
    group = ratings.groupby(['user'])
    len_group = group.apply(lambda x: len(x))

    user_unique = len_group[len_group > 10].keys().tolist()
    ratings = ratings[ratings.user.isin(user_unique)]
    ratings.sort_values(['user', 'timestamp'], ascending=[1, 1], inplace=True)
    item_unique = pd.unique(ratings.item).tolist()
    n_user = len(user_unique)
    n_item = len(item_unique)
    print("# num of user: %d \n# num of item: %d" % (n_user, n_item))
    fsum.write("# num of user: %d \n# num of item: %d\n" % (n_user, n_item))

    rat = [0] * n_user
    f = open("%s/ratings.txt" % dir_w, "w")
    for i in range(len(ratings)):
        uid = user_unique.index(ratings.user[i])
        iid = item_unique.index(ratings.item[i])
        if rat[uid] == 0:
            rat[uid] = [[iid, ratings.rating[i], ratings.timestamp[i]]]
        else:
            rat[uid].append([iid, ratings.rating[i], ratings.timestamp[i]])
        f.write("%d::%i::%d::%s\n" % (uid, iid, int(ratings.rating[i]), ratings.timestamp[i]))
    f.close()

    ratings = rat
    print("Max item user rated: %d" % max([len(i) for i in ratings]))
    print("Min item user rated: %d" % min([len(i) for i in ratings]))
    print("Mean item user rated: %d" % np.mean([len(i) for i in ratings]))

    fsum.write("Max item user rated: %d\n" % max([len(i) for i in ratings]))
    fsum.write("Min item user rated: %d\n" % min([len(i) for i in ratings]))
    fsum.write("Mean item user rated: %d\n" % np.mean([len(i) for i in ratings]))

    # write user_id, item_id to file
    f = open("%s/user_id.txt" % dir_w, "w")
    f.write("\n".join(user_unique))
    f.close()

    f = open("%s/item_id.txt" % dir_w, "w")
    f.write("\n".join(item_unique))
    f.close()

    # Divide train, test

    shuffle_id = np.random.permutation(n_user)
    train_len = int(0.7 * n_user)
    train_id = shuffle_id[:train_len]
    test_id = shuffle_id[train_len:]

    ftrain = open("%s/train.txt" % dir_w, "w")
    for idx in train_id:
        user = np.array(ratings[idx]).reshape((len(ratings[idx]), 3))
        user = user[np.argsort(user[:, 2])]
        item = [int(i) for i in list(user[:, 0])]
        item = [str(i) for i in item]
        ftrain.write("%d %s\n" % (idx, " ".join(item)))
    ftrain.close()

    ftest = open("%s/test.txt" % dir_w, "w")
    for idx in test_id:
        user = np.array(ratings[idx]).reshape((len(ratings[idx]), 3))
        user = user[np.argsort(user[:, 2])]
        item = [int(i) for i in list(user[:, 0])]
        item = [str(i) for i in item]
        ftest.write("%d %s\n" % (idx, " ".join(item)))
    ftest.close()
    fsum.write("Train num: %d, test num: %d\n" % (len(train_id), len(test_id)))



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
    f = open("%s/description_fix.txt" % dir_w, "w")
    f.write("\n".join(text))
    f.close()

    # tf-idf

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(text)
    save_npz("%s/item.npz" % dir_w, X)

    categories = data_item.categories.tolist()
    categories = [i for cat in categories for c in cat for i in c]
    categories = list(set(categories))

    f = open("%s/categories.txt" % dir_w, "w")
    for c in list_cat:
        arr = ['0'] * len(categories)
        for i in c[0][0]:
            arr[categories.index(i)] = '1'
        f.write(",".join(arr))
        f.write("\n")
    f.close()

    f = open("%s/list_categories.txt" % dir_w, "w")
    f.write("\n".join(categories))
    f.close()

    fsum.write("text length: %d - cat length: %d\n" % (X.shape[1], len(categories)))


def create_user_info(data_dir):
    if data_dir == "data/ml-1m":
        ratings = np.genfromtxt("%s/ratings.txt" % data_dir, np.int32, delimiter=" ", )
    else:
        ratings = np.genfromtxt("%s/ratings.txt" % data_dir, np.int32, delimiter=",", )
    user_info = []
    time_info = []
    fuser = open("%s/user_info_train.txt" % data_dir, "w")
    ftime = open("%s/time_train.txt" % data_dir, "w")
    for line in open("%s/train.txt" % data_dir):
        # read line
        list_p = line.strip().split()
        list_p = [int(p) for p in list_p]
        u = list_p[0]
        list_p = list_p[1:]

        # create arr
        no_item = len(list_p)
        r = [0] * 5
        weekdays = [0] * 7
        # cat = np.zeros(categories.shape[1])
        time = []
        tmp_rating = ratings[np.where(ratings[:, 0] == u)]
        line_no = 0

        for p in list_p:
            # rating
            rat = tmp_rating[line_no]
            if p == rat[1] or u == rat[0]:
                r[rat[2] - 1] += 1
                # cat += categories[rat[1]]
                t = datetime.utcfromtimestamp(int(rat[3]))
                weekdays[t.weekday()] += 1
                time.append(rat[3])
                line_no += 1
            else:
                print(rat, line_no)
        #     r = np.array(r)/sum(r)
        #     weekdays = np.array(weekdays)/sum(weekdays)
        #     cat = cat/sum(cat)

        user_info.append([no_item] + r + weekdays)
        fuser.write("%d,%s\n" % (u, ",".join([str(i) for i in user_info[-1]])))
        ftime.write("%d,%s\n" % (u, ",".join([str(i) for i in time])))
        time_info.append(time)
    fuser.close()
    ftime.close()

    # for test
    user_info = []
    time_info = []
    fuser = open("%s/user_info_test.txt" % data_dir, "w")
    ftime = open("%s/time_test.txt" % data_dir, "w")
    for line in open("%s/test.txt" % data_dir):
        # read line
        list_p = line.strip().split()
        list_p = [int(p) for p in list_p]
        u = list_p[0]
        list_p = list_p[1:]

        # create arr
        no_item = len(list_p)
        r = [0] * 5
        hour = [0]*4
        weekdays = [0] * 7
        # cat = np.zeros(categories.shape[1])
        time = []
        tmp_rating = ratings[np.where(ratings[:, 0] == u)]
        line_no = 0

        for p in list_p:
            # rating
            rat = tmp_rating[line_no]
            if p == rat[1] or u == rat[0]:
                r[rat[2] - 1] += 1
                # cat += categories[rat[1]]
                t = datetime.utcfromtimestamp(int(rat[3]))
                hour[int(t.hour/6)] += 1
                weekdays[t.weekday()] += 1
                time.append(rat[3])
                line_no += 1
            else:
                print(rat, line_no)
        #     r = np.array(r)/sum(r)
        #     weekdays = np.array(weekdays)/sum(weekdays)
        #     cat = cat/sum(cat)

        user_info.append([no_item] + r + weekdays)
        fuser.write("%d,%s\n" % (u, ",".join([str(i) for i in user_info[-1]])))
        ftime.write("%d,%s\n" % (u, ",".join([str(i) for i in time])))
        time_info.append(time)
    fuser.close()
    ftime.close()


def create_gru4rec(dataset):
    data = pd.read_csv("data/%s/ratings.txt"%dataset, sep=",", header=None)
    data.columns = ["user_id", "item_id", "rating", "date"]
    index = pd.DatetimeIndex(data.date)
    data.date = index.astype(np.int64) // 10 ** 9
    data = data.sort_values(by=["user_id", "date"])

    test_id = list(open("data/%s/test.txt"%dataset))
    test_id = [int(t.strip().split(" ")[0]) for t in test_id]

    train_session = []
    train_time = []
    test_session = []
    test_time = []
    user_id = 0
    date = -1
    for _, row in data.iterrows():
        if row.user_id == user_id and row.date == date:
            item_list.append(row.item_id)
        else:
            if date != -1 and len(item_list) > 0:
                if user_id in test_id and len(item_list) > 1:
                    test_session.append(item_list)
                    test_time.append(date)
                else:
                    train_session.append(item_list)
                    train_time.append(date)
            item_list = []
            user_id = row.user_id
            date = row.date

    def write_file(sess, time, type="train"):
        f = open("GRU4Rec_TensorFlow/data/%s/%s.txt"%(dataset, type), "w")
        f.write("SessionId,ItemId,Timestamps\n")
        for j, s in enumerate(sess):
            for i in s:
                f.write("%d,%d,%d\n"%(j, i, time[j]))
        f.close()

    write_file(train_session, train_time, "train")
    write_file(test_session, test_time, "test")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--data', type=str, default="Tool", help='dataset name')
    args = parser.parse_args()
    type = args.data
    # fsum = open("data/summary.txt", "a")
    # create_amazon("../cf-vae/data/%s" % type, type, fsum)
    create_user_info(args.data)
    # fsum.close()
    # create_gru4rec(type)
    # dataset = ["book", "Garden", "Automotive", "Beauty", "Grocery", "Outdoor", "Office"]
    # # fsum = open("data/summary.txt", "a")
    # for type in dataset:
    #     # dir_r = "../cf-vae/data/%s"%type
    #     # create_amazon(dir_r, type, fsum)
    #     # # create_amazon_based_on_ratings(dir_r, type, fsum)
    #     # create_user_info("data/%s"%type)
    # create_gru4rec(type)


