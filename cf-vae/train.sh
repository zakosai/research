python train_vae.py --ckpt_folder=kitchen/user5 --data_dir=data/kitchen/ --zdim=50
#python train_img_vae.py --ckpt_folder=sport/vae --data_dir=data/sport/ --zdim=50

python train_cvae_user.py --model=0 --ckpt_folder=kitchen/user5 --data_dir=data/kitchen/ --iter=50 --zdim=50 --gridsearch=1

python train_cvae_extend.py --model=0 --ckpt_folder=kitchen/user5 --data_dir=data/kitchen/ --iter=50 --zdim=50 --gridsearch=1





#python train_dae.py --ckpt_folder=kitchen/user1 --data_dir=data/kitchen/
#python train_cf_dae.py --model=0 --ckpt_folder=sport/dae --data_dir=data/sport/ --iter=50
#python train_im_dae.py --ckpt_folder=sport/dae --data_dir=data/sport/
python train_cf_dae.py --model=0 --ckpt_folder=kitchen/user5 --data_dir=data/kitchen/ --iter=50

#python train_cvae_user.py --model=0 --ckpt_folder=sport/user_8 --data_dir=data/sport/ --iter=50 --zdim=50
