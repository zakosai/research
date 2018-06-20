python train_vae.py --ckpt_folder=kitchen/user8_n --data_dir=data/kitchen/ --zdim=50 --data_type=8 --user_dim=896
python train_cvae_user.py --model=0 --ckpt_folder=kitchen/user8_n --data_dir=data/kitchen/ --iter=50 --zdim=50 --gridsearch=1 --data_type=8 --user_dim=896 --user_no=7981 --item_no=19184

python train_vae.py --ckpt_folder=sport/8n --data_dir=data/sport/ --zdim=50 --data_type=8 --user_dim=742
#python train_vae.py --ckpt_folder=sport/8n --data_dir=data/sport/ --zdim=50 --data_type=8
python train_cvae_user.py --model=0 --ckpt_folder=sport/8n --data_dir=data/sport/ --iter=50 --zdim=50 --gridsearch=1 --data_type=8 --user_dim=742 --user_no=5584 --item_no=13790
#python train_cvae_user.py --model=0 --ckpt_folder=sport/8n --data_dir=data/sport/ --iter=50 --zdim=50 --gridsearch=1 --data_type=8
#python train_cf_dae.py --model=0 --ckpt_folder=sport/1n --data_dir=data/sport/ --iter=50 --data_type=1
#python train_cf_dae.py --model=0 --ckpt_folder=sport/5n --data_dir=data/sport/ --iter=50 --data_type=5
#python train_cf_dae.py --model=0 --ckpt_folder=sport/8n --data_dir=data/sport/ --iter=50 --data_type=8
#
#python train_vae.py --ckpt_folder=ml/70p --data_dir=data/ml-1m/ --zdim=50 --data_type=70p
#python train_cvae_user.py --model=0 --ckpt_folder=ml/70p --data_dir=data/ml-1m/ --iter=50 --zdim=50 --gridsearch=1 --data_type=70p
