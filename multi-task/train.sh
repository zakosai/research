#!/usr/bin/env bash

mkdir experiment/delicious
python vae.py --ckpt_folder=experiment/delicious/ --data_dir=data/delicious/
python train_cvae_extend.py --ckpt_folder=experiment/delicious/ --data_dir=data/delicious/

mkdir experiment/tool
python multi-VAE.py --data_dir=data/tool/dataset.pkl --ckpt_folder=experiment/tool/
python vae_item.py --data_dir=data/tool/dataset.pkl --ckpt_folder=experiment/tool/
python vae-unet.py --data_dir=data/tool/dataset.pkl --ckpt_folder=experiment/tool/
python vae.py --ckpt_folder=experiment/tool/ --data_dir=data/tool/
python train_cvae_extend.py --ckpt_folder=experiment/tool/ --data_dir=data/tool/

mkdir experiment/outdoor
python multi-VAE.py --data_dir=data/outdoor/dataset.pkl --ckpt_folder=experiment/outdoor/
python vae_item.py --data_dir=data/outdoor/dataset.pkl --ckpt_folder=experiment/outdoor/
python vae-unet.py --data_dir=data/outdoor/dataset.pkl --ckpt_folder=experiment/outdoor/
python vae.py --ckpt_folder=experiment/outdoor/ --data_dir=data/outdoor/
python train_cvae_extend.py --ckpt_folder=experiment/outdoor/ --data_dir=data/outdoor/

mkdir experiment/grocery
python multi-VAE.py --data_dir=data/grocery/dataset.pkl --ckpt_folder=experiment/grocery/
python vae_item.py --data_dir=data/grocery/dataset.pkl --ckpt_folder=experiment/grocery/
python vae-unet.py --data_dir=data/grocery/dataset.pkl --ckpt_folder=experiment/grocery/
python vae.py --ckpt_folder=experiment/grocery/ --data_dir=data/grocery/
python train_cvae_extend.py --ckpt_folder=experiment/grocery/ --data_dir=data/grocery/