#python train_vae.py --ckpt_folder=grocery2/vae --data_dir=data/amazon2/ --zdim=50
#python train_cvae_extend.py --model=0 --ckpt_folder=grocery2/vae --data_dir=data/amazon2/ --iter=100 --zdim=50
#cp -a grocery2/vae/vae* grocery2/vae_im
#python train_img_vae.py --ckpt_folder=grocery2/vae_im --data_dir=data/amazon2/ --zdim=50
#python train_cvae_extend.py --model=1 --ckpt_folder=grocery2/vae_im --data_dir=data/amazon2/ --iter=100 --zdim=50


python train_cf_dae.py --model=1 --ckpt_folder=grocery2/dae_im --data_dir=data/amazon2/ --iter=50

#python train_dae.py --ckpt_folder=grocery2/dae --data_dir=data/amazon2/ --zdim=50
python train_cf_dae.py --model=0 --ckpt_folder=grocery2/dae --data_dir=data/amazon2/ --iter=50
#cp -a grocery2/dae/dae* grocery2/dae_im
#python train_im_dae.py --ckpt_folder=grocery2/dae_im --data_dir=data/amazon2/ --zdim=50
#    python train_img_vae.py --ckpt_folder=pre2/z3_$i --data_dir=data/movie/ --zdim=$i
#    python train_cvae_extend.py --model=1 --ckpt_folder=pre2/z3_$i --data_dir=data/movie/ --iter=100 --zdim=$i
