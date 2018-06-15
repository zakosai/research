python train_vae.py --ckpt_folder=ml/70p --data_dir=data/ml-1m/ --zdim=50 --data_type=70p
python train_dae.py --ckpt_folder=ml/70p --data_dir=data/ml-1m/


python train_cvae_user.py --model=0 --ckpt_folder=ml/70p --data_dir=data/ml-1m/ --iter=50 --zdim=50 --gridsearch=1 --data_type=70p
python train_cvae_extend.py --model=0 --ckpt_folder=ml/70p --data_dir=data/ml-1m/ --iter=50 --zdim=50 --gridsearch=1 --data_type=70p
python train_cf_dae.py --model=0 --ckpt_folder=ml/70p --data_dir=data/ml-1m/ --iter=50 --data_type=70p


python train_vae.py --ckpt_folder=ml/70p --data_dir=data/ml-1m/ --zdim=50 --data_type=15

python train_cvae_user.py --model=0 --ckpt_folder=ml/70p --data_dir=data/ml-1m/ --iter=50 --zdim=50 --gridsearch=1 --data_type=15
python train_cvae_extend.py --model=0 --ckpt_folder=ml/70p --data_dir=data/ml-1m/ --iter=50 --zdim=50 --gridsearch=1 --data_type=15
python train_cf_dae.py --model=0 --ckpt_folder=ml/70p --data_dir=data/ml-1m/ --iter=50 --data_type=15

python train_vae.py --ckpt_folder=ml/70p --data_dir=data/ml-1m/ --zdim=50 --data_type=5

python train_cvae_user.py --model=0 --ckpt_folder=ml/70p --data_dir=data/ml-1m/ --iter=50 --zdim=50 --gridsearch=1 --data_type=5
python train_cvae_extend.py --model=0 --ckpt_folder=ml/70p --data_dir=data/ml-1m/ --iter=50 --zdim=50 --gridsearch=1 --data_type=5
python train_cf_dae.py --model=0 --ckpt_folder=ml/70p --data_dir=data/ml-1m/ --iter=50 --data_type=5