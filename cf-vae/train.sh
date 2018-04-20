python train_cvae_extend.py --model=1 --ckpt_folder=pre2/zdim500 --data_dir=data/movie/ --iter=15

mkdir -p pre_model/new_z
python train_vae.py --ckpt_folder=pre_model/new_z --data_dir=data/amazon
python train_img_vae.py --ckpt_folder=pre_model/new_z --data_dir=data/amazon
python train_cvae_extend.py --model=0 --ckpt_folder=pre_model/new_z --data_dir=data/amazon/ --iter=15
python train_cvae_extend.py --model=1 --ckpt_folder=pre_model/new_z --data_dir=data/amazon/ --iter=15
#python train_cvae_extend.py --model=0 --ckpt_folder=pre2/v1 --data_dir=data/movie/ --iter=15
#python train_cvae_extend.py --model=1 --ckpt_folder=pre2/v1 --data_dir=data/movie/ --iter=15