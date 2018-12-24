#!/usr/bin/env bash




dir='tool outdoor grocery'
for d in $dir
do
#python vae.py  --ckpt_folder=experiment/$d/ --data_dir=data/$d/
#python train_cvae_extend.py --ckpt_folder=experiment/$d/ --data_dir=data/$d/
#python NeuMF.py --data=data/$d/dataset.pkl --ckpt=experiment/$d/
#python FM.py --data=data/$d/dataset.pkl --ckpt=experiment/$d/
#python MLP.py --data=data/$d/dataset.pkl --ckpt=experiment/$d/
#python GMF.py --data=data/$d/dataset.pkl --ckpt=experiment/$d/
#python NeuMF.py --data=data/$d/dataset.pkl --ckpt=experiment/$d/ --mf_pretrain=experiment/$d/GMF.h5 \
#--mlp_pretrain=experiment/$d/MLP.h5
#python multi-VAE.py --data=data/$d/dataset.pkl --ckpt=experiment/$d/
python vae_item.py --data=data/$d/dataset.pkl --ckpt=experiment/$d/
python vae-unet.py --data=data/$d/dataset.pkl --ckpt=experiment/$d/

done

