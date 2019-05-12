#!/usr/bin/env bash

python vae.py --ckpt_folder=experiment/ml-1m/cvae_100 --data_dir=data/ml-1m --zdim=100
python cf_vae.py --ckpt_folder=experiment/ml-1m/cvae_100 --data_dir=data/ml-1m --zdim=100
source ../.env3/bin/activate
python seq2seq.py --data=ml-1m --num_p=3706
