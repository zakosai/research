from PIL import Image
import numpy as np
import os
import pandas as pd
import gzip

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

def

if __name__ == '__main__':
    products = getDF("data/meta_Health_and_Personal_Care.json")
    reviews = getDF("data/reviews_Health_and_Personal_Care_5.json")
