mkdir -p pre2/zdim250

python train_vae.py --ckpt_folder=pre2/zdim250
python train_img_vae --ckpt_folder=pre2/zdim250
python train_cvae_extend --model=0 --ckpt_folder=pre2/zdim250 --data_dir=data/movie/
python train_cvae_extend --model=1 --ckpt_folder=pre2/zdim250 --data_dir=data/movie/