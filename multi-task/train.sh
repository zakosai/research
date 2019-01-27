#!/usr/bin/env bash
#python vae_item.py --data=data/lastfm/dataset.pkl --ckpt=experiment/lastfm/
#python vae-unet.py --data=data/lastfm/dataset.pkl --ckpt=experiment/lastfm/
#python vae-unet.py --data=data/delicious/dataset.pkl --ckpt=experiment/delicious/

python vae_unet.py --data=data/delicious/dataset.pkl --ckpt=experiment/delicious/

dir='lastfm delicious'
for d in $dir
do
#python vae_item.py --data=data/$d/dataset.pkl --ckpt=experiment/$d/
python vae.py  --ckpt_folder=experiment/$d/ --data_dir=data/$d/
python train_cvae_extend.py --ckpt_folder=experiment/$d/ --data_dir=data/$d/
python vae_unet.py --data=data/$d/dataset.pkl --ckpt=experiment/$d/
#python dae.py  --ckpt_folder=experiment/$d/ --data_dir=data/$d/
#python train_cf_dae.py --ckpt_folder=experiment/$d/ --data_dir=data/$d/ --gridsearch=1
python NeuMF.py --data=data/$d/dataset.pkl --ckpt=experiment/$d/
python FM.py --data=data/$d/dataset.pkl --ckpt=experiment/$d/
python MLP.py --data=data/$d/dataset.pkl --ckpt=experiment/$d/
python GMF.py --data=data/$d/dataset.pkl --ckpt=experiment/$d/
python NeuMF.py --data=data/$d/dataset.pkl --ckpt=experiment/$d/ --mf_pretrain=experiment/$d/GMF.h5 \
--mlp_pretrain=experiment/$d/MLP.h5
python multi-VAE.py --data=data/$d/dataset.pkl --ckpt=experiment/$d/
python multi-vae-item.py --data=data/$d/dataset.pkl --ckpt=experiment/$d/

#
##
##
done

python vae.py  --ckpt_folder=experiment/delicious/ --data_dir=data/delicious/
python train_cf_dae.py --ckpt_folder=experiment/delicious/ --data_dir=data/delicious/ --gridsearch=1
#python vae_item.py --data=data/grocery/dataset.pkl --ckpt=experiment/grocery/
#python vae-unet.py --data=data/grocery/dataset.pkl --ckpt=experiment/grocery/
#python vae.py  --ckpt_folder=experiment/outdoor/ --data_dir=data/outdoor/
#python train_cvae_extend.py --ckpt_folder=experiment/outdoor/ --data_dir=data/outdoor/
