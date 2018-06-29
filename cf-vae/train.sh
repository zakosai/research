folders='Toy'
rate='1 8 80'
python train_dae.py --ckpt_folder=sport/1nn --data_dir=data/sport/ --zdim=50 --data_type=1 --user_dim=742
python train_vae.py --ckpt_folder=sport/1nn --data_dir=data/sport/ --zdim=50 --data_type=1 --user_dim=742
python train_cf_dae.py --model=0 --ckpt_folder=sport/1nn --data_dir=data/sport/ --iter=50 --data_type=1 --user_no=5584 --item_no=13790

python train_cvae_extend.py --model=0 --ckpt_folder=sport/1nn --data_dir=data/sport/ --iter=50 --zdim=50 --gridsearch=1 --data_type=1 --user_no=5584 --item_no=13790
#
python train_cvae_user.py --model=0 --ckpt_folder=sport/1nn --data_dir=data/sport/ --iter=50 --zdim=50 --gridsearch=1 --data_type=1 --user_dim=742 --user_no=5584 --item_no=13790
python train_cdae_user.py --model=0 --ckpt_folder=sport/1nn --data_dir=data/sport/ --iter=50 --zdim=50 --gridsearch=1 --data_type=1 --user_dim=742 --user_no=5584 --item_no=13790

#


#for f in $folders
#do
#    python train_dae.py --ckpt_folder=$f/20 --data_dir=data/$f/ --zdim=50 --data_type=8 --user_dim=742
#    cp $f/20/dae* $f/80/
#
#done

#for f in $folders
#do
#    for r in $rate
#    do
#        dim="$(sed -n '3p' data/$f/info.txt)"
#        user_no="$(sed -n '1p' data/$f/info.txt)"
#        item_no="$(sed -n '2p' data/$f/info.txt)"
#
#        python train_cvae_extend.py --model=0 --ckpt_folder=$f/$r --data_dir=data/$f/ --iter=50 --zdim=50 --gridsearch=1 --data_type=$r --user_no=$user_no --item_no=$item_no
#
#        python train_vae.py --ckpt_folder=$f/$r --data_dir=data/$f/ --zdim=50 --data_type=$r --user_dim=$dim
#        python train_cvae_user.py --model=0 --ckpt_folder=$f/$r --data_dir=data/$f/ --iter=50 --zdim=50 --gridsearch=1 --data_type=$r --user_dim=$dim --user_no=$user_no --item_no=$item_no
#
#        python train_cf_dae.py --model=0 --ckpt_folder=$f/$r --data_dir=data/$f/ --iter=50 --data_type=$r --user_no=$user_no --item_no=$item_no
#
#        python train_dae.py --ckpt_folder=$f/$r --data_dir=data/$f/ --zdim=50 --data_type=$r --user_dim=$dim
#        python train_cdae_user.py --model=0 --ckpt_folder=$f/$r --data_dir=data/$f/ --iter=50 --zdim=50 --gridsearch=1 --data_type=$r --user_dim=$dim --user_no=$user_no --item_no=$item_no
#
#
#
#    done
#
#done

#python train_vae.py --ckpt_folder=sport/8n --data_dir=data/sport/ --zdim=50 --data_type=8 --user_dim=742
#python train_cvae_user.py --model=0 --ckpt_folder=sport/8n --data_dir=data/sport/ --iter=50 --zdim=50 --gridsearch=1 --data_type=5 --user_dim=742 --user_no=5584 --item_no=13790
#
#python train_vae.py --ckpt_folder=ml/15 --data_dir=data/ml-1m/ --zdim=50 --data_type=15 --user_dim=9975
#python train_cvae_user.py --model=0 --ckpt_folder=ml/15 --data_dir=data/ml-1m/ --iter=50 --zdim=50 --gridsearch=1 --data_type=15 --user_dim=9975
#
#python train_vae.py --ckpt_folder=ml/70p --data_dir=data/ml-1m/ --zdim=50 --data_type=70p --user_dim=30
#python train_cvae_user.py --model=0 --ckpt_folder=ml/70p --data_dir=data/ml-1m/ --iter=50 --zdim=50 --data_type=70p --user_dim=30

#python train_vae.py --ckpt_folder=sport/8n --data_dir=data/sport/ --zdim=50 --data_type=8 --user_dim=742
#python train_vae.py --ckpt_folder=sport/8n --data_dir=data/sport/ --zdim=50 --data_type=8
#python train_cvae_user.py --model=0 --ckpt_folder=sport/8n --data_dir=data/sport/ --iter=50 --zdim=50 --gridsearch=1 --data_type=8 --user_dim=742 --user_no=5584 --item_no=13790
#python train_cvae_user.py --model=0 --ckpt_folder=sport/8n --data_dir=data/sport/ --iter=50 --zdim=50 --gridsearch=1 --data_type=8
#python train_cf_dae.py --model=0 --ckpt_folder=sport/1n --data_dir=data/sport/ --iter=50 --data_type=1
#python train_cf_dae.py --model=0 --ckpt_folder=sport/5n --data_dir=data/sport/ --iter=50 --data_type=5
#python train_cf_dae.py --model=0 --ckpt_folder=sport/8n --data_dir=data/sport/ --iter=50 --data_type=8
#
#python train_vae.py --ckpt_folder=ml/70p --data_dir=data/ml-1m/ --zdim=50 --data_type=70p
#python train_cvae_user.py --model=0 --ckpt_folder=ml/70p --data_dir=data/ml-1m/ --iter=50 --zdim=50 --gridsearch=1 --data_type=70p
#python train_cdae_user.py --model=0 --ckpt_folder=Toy/1 --data_dir=data/Toy/ --iter=50 --zdim=50  --data_type=1 --user_no=2611 --item_no=10074 --user_dim=400

#python train_dae.py --ckpt_folder=Toy/1 --data_dir=data/Toy/ --zdim=50 --data_type=1 --user_dim=979