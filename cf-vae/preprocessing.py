from PIL import Image
import numpy as np
import os
dir = "data/movie/images"
im_list = []
for i in range(16000):
    try:
        im = Image.open(os.path.join(dir, str(i)+".jpg"))
        im = im.resize((64, 64), Image.ANTIALIAS)
        im = np.asarray(im).astype(np.uint8)
        im = im.flatten()
        if len(im) != 12288:
            im = np.zeros(12288).astype(np.uint8)
        im_list.append(im.tolist())
    except:
        print(i)
        im = np.zeros(12288).astype(np.uint8)
        im_list.append(im.tolist())

    if i%1000 == 0:
        print(i)

im_list = np.asarray(im_list).astype(np.uint8)
im_list = im_list.flatten()
im_list.tofile("data/movie/images.bin")