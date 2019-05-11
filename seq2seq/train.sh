#!/usr/bin/env bash

python cf-vae.py --ckpt_folder=experiment/ml-1m/cvae --data_dir=data/ml-1m
python vae.py --ckpt_folder=experiment/ml-1m/cvae_100 --data_dir=data/ml-1m --zdim=100
python cf-vae.py --ckpt_folder=experiment/ml-1m/cvae_100 --data_dir=data/ml-1m --zdim=100
