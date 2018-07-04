folders='Toy Tool Beauty Electronics TV'
rate='1 8'


for f in $folders
do
    for r in $rate
    do
        dim="$(sed -n '3p' data2/$f/info.txt)"
        user_no="$(sed -n '1p' data2/$f/info.txt)"
        item_no="$(sed -n '2p' data2/$f/info.txt)"
        python train_vae.py --ckpt_folder=$f/$r --data_dir=data2/$f/ --zdim=50 --data_type=$r --user_dim=$dim --type=text
        python train_vae.py --ckpt_folder=$f/$r --data_dir=data2/$f/ --zdim=50 --data_type=$r --user_dim=$dim --type=user


        #python train_vae.py --ckpt_folder=$f/$r2 --data_dir=data2/$f/ --zdim=50 --data_type=$r --user_dim=$dim

        python train_dae.py --ckpt_folder=$f/$r --data_dir=data2/$f/ --zdim=50 --data_type=$r --user_dim=$dim --type=text

        python train_cf_dae.py --model=0 --ckpt_folder=$f/$r --data_dir=data2/$f/ --iter=50 --data_type=$r --user_no=$user_no --item_no=$item_no --gridsearch=1 --zdim=50
        python train_cvae_user.py --model=0 --ckpt_folder=$f/$r --data_dir=data2/$f/ --iter=50 --zdim=50 --gridsearch=1 --data_type=$r --user_dim=$dim --user_no=$user_no --item_no=$item_no
        python train_cvae_extend.py --model=0 --ckpt_folder=$f/$r --data_dir=data2/$f/ --iter=50 --zdim=50 --gridsearch=1 --data_type=$r --user_no=$user_no --item_no=$item_no

        #python train_dae.py --ckpt_folder=$f/$r --data_dir=data2/$f/ --zdim=50 --data_type=$r --user_dim=$dim --type=user
        #python train_cdae_user.py --model=0 --ckpt_folder=$f/$r --data_dir=data2/$f/ --iter=50 --zdim=50 --gridsearch=1 --data_type=$r --user_dim=$dim --user_no=$user_no --item_no=$item_no



    done

done

