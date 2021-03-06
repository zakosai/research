import gzip
import pandas as pd
import numpy as np
import argparse
import os


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


def get_categories(folder):
    product_ids = list(open("/media/linh/DATA/research/cf-vae/data2/%s/product_ids.txt" % folder))
    product_ids = [i.strip() for i in product_ids]

    product = getDF("/media/linh/DATA/research/cf-vae/data/%s/meta.json.gz" % folder)
    product = product[product.asin.isin(product_ids)].categories
    product.reset_index(drop=True, inplace=True)
    product = product.apply(lambda x: x[0])
    return product


def convert_categories(data):
    categories = data.tolist()
    categories = [i for s in categories for i in s]
    categories = list(set(categories))
    output = np.zeros((len(data), len(categories)))
    data = data.tolist()
    for i, d in enumerate(data):
        if len(d) != 0:
            near_list = [categories.index(s) for s in d]
            output[i, near_list] = 1
    return output


def load_rating(path):
    arr = []
    for line in open(path):
        a = line.strip().split()
        if a[0] == 0:
            l = []
        else:
            l = [int(x) for x in a[1:]]
        arr.append(l)
    return arr


def preprocess(folder, f):
    data = pd.read_csv("data/%s/review_info.txt" % folder, index_col=0)
    data.columns = ['u_id', 'p_id', 'rating', 'unixTime', 'brand', 'categories']
    categories = get_categories(folder)
    categories = convert_categories(categories)

    data['time'] = pd.to_datetime(data.unixTime, unit='s')
    data['time'] = data.time.apply(lambda x: x.weekday())
    weekday = pd.get_dummies(data.time)
    rating_score = pd.get_dummies(data.rating)
    data = pd.concat([data.u_id, data.p_id, weekday, rating_score], axis=1)
    columns = data.columns.tolist()[2:]

    if not os.path.exists("data/%s" % folder):
        os.makedirs("data/%s" % folder)

    f.write(folder)
    for type in [1, 8]:
        train = load_rating("/media/linh/DATA/research/cf-vae/data2/%s/cf-train-%dp-users.dat"%(folder, type))
        grouped = data.groupby('u_id')
        user_info = [0] * len(train)
        for name, group in grouped:
            p_lists = train[name]
            d = group[group.p_id.isin(p_lists)][columns].as_matrix().mean(axis=0)
            cat = categories[p_lists].mean(axis=0)
            user_info[name] = d.tolist() + cat.tolist()
        np.save("data/%s/user_info_%s.npy"%(folder, type), np.array(user_info))
        f.write("%d, %d" % np.array(user_info).shape)
        print("Finish %s %d"%(folder, type))

    print("---------------------------------")


def gen_neucf(folder):
    print(folder)
    for type in [1, 8]:
        train = load_rating("/media/linh/DATA/research/cf-vae/data2/%s/cf-train-%dp-users.dat"%(folder, type))
        test = load_rating("/media/linh/DATA/research/cf-vae/data2/%s/cf-train-%dp-users.dat"%(folder, type))
        n_item = len(list(open("/media/linh/DATA/research/cf-vae/data2/%s/cf-train-%dp-items.dat"%(folder, type))))
        transaction = list(open("/media/linh/DATA/research/cf-vae/data2/%s/review_info.txt" % (folder)))
        transaction = [t.strip().split(', ')[:3] for t in transaction]
        columns = transaction[0]
        transaction = pd.DataFrame(transaction[1:], columns=columns, dtype='int')

        train_file = open("../data/%s/%s%d.train.rating"%(folder, folder, type), 'w')
        test_file = open("../data/%s/%s%d.test.rating" % (folder, folder, type), 'w')
        test_negative = open("../data/%s/%s%d.test.negative" % (folder, folder, type), 'w')


        train_neucf = []
        for i in range(len(train)):
            tmp_trans = transaction[transaction.u_id == i]
            neg = list(set(range(n_item)) - set(train[i] + test[i]))
            for j in train[i]:
                r = tmp_trans[tmp_trans.p_id == j].rating.values[0]
                train_neucf.append([i, j, r])

            for j in test[i]:
                r = tmp_trans[tmp_trans.p_id == j].rating.values[0]
                test_file.write('%d\t%d\t%d\n'%(i, j, r))
                n = np.random.permutation(neg)[:99].tolist()
                n = '\t'.join([str(ne) for ne in n])
                test_negative.write('(%d,%d)\t%s\n'%(i, j, n))

        rand = np.random.permutation(range(len(train_neucf)))
        for i in rand:
            train_file.write('%d\t%d\t%d\n'%(train_neucf[i][0], train_neucf[i][1], train_neucf[i][2]))

        train_file.close()
        test_file.close()
        test_negative.close()


def re_cal_review_info(folder):
    product_id = list(open("/media/linh/DATA/research/cf-vae/data2/%s/product_ids.txt" % folder))
    product_id = [u.strip() for u in product_id]

    # review_info = pd.read_csv("/media/linh/DATA/research/cf-vae/data2/%s/review_info.txt" % folder, delimiter=', ')
    review_info = list(open("/media/linh/DATA/research/cf-vae/data2/%s/review_info.txt" % folder))
    review_info = [r.split(', ')[:4] for r in review_info]
    cols = review_info[0]
    review_info = pd.DataFrame(review_info, columns=cols, dtype='int')
    review_info['brand'] = np.nan
    review_info['categories'] = np.nan
    reviews = getDF("/media/linh/DATA/research/cf-vae/data/%s/meta.json.gz" % folder)

    def convert(x):
        re = reviews[reviews.asin == product_id[int(x.p_id)]]
        x.brand = re.brand.values[0]
        x.categories = ','.join(re.categories.values[0][0])
        return x

    review_info = review_info.apply(lambda x: convert(x), axis=1)
    review_info = review_info.astype({'u_id': int, 'p_id': int, 'rating': int, 'unixTime': int})
    review_info.to_csv("data/%s/review_info.txt" % folder)

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--data', type=str, default='Baby')
    # args = parser.parse_args()
    # # folders = ['Instrument', 'Kindle', 'Music', 'Office', 'Pet', 'Phone', 'Video']
    # # folders = ['Kitchen', 'TV', 'Beauty', 'Toy', 'Health', 'Clothing']
    summary = open("data/summary.txt", "a")
    # preprocess(args.data, summary)
    # summary.close()
    #
    # folders = ["TV", "Toy", "Tool"]
    # for f in folders:
    #     print(f)
    #     re_cal_review_info(f)

    # folders = ['Instrument', 'Kindle', 'Music', 'Office', 'Pet', 'Phone', 'Video', 'Garden', 'Beauty', 'Health',
    #            'Kitchen', 'TV', 'Toy', 'Tool']
    folders = ['Music', 'Office', 'Pet', 'Phone', 'Video', 'Garden', 'Beauty', 'Kindle']
    for f in folders:
        print(f)
        preprocess(f, summary)






