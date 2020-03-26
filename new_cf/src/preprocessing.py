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
    data = pd.read_csv("/media/linh/DATA/research/cf-vae/data2/%s/review_info.txt" % folder, delimiter=', ')
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='Baby')
    args = parser.parse_args()
    # folders = ['Instrument', 'Kindle', 'Music', 'Office', 'Pet', 'Phone', 'Video']
    # folders = ['Kitchen', 'TV', 'Beauty', 'Toy', 'Health', 'Clothing']
    summary = open("data/summary.txt", "a")
    preprocess(args.data, summary)
    summary.close()






