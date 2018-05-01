#python train_img_vae.py --ckpt_folder=grocery/z20 --data_dir=data/amazon2/ --zdim=20
#python train_cvae_extend.py --model=1 --ckpt_folder=grocery/z20 --data_dir=data/amazon2/ --iter=15 --zdim=20

for i in 20 50 100
do
    #mkdir grocery/z$i
    #python train_vae.py --ckpt_folder=grocery/z$i --data_dir=data/amazon2/ --zdim=$i
    python train_cvae_extend.py --model=0 --ckpt_folder=grocery/zdim$i --data_dir=data/amazon2/ --iter=15 --zdim=$i
    #python train_img_vae.py --ckpt_folder=grocery/z$i --data_dir=data/amazon2/ --zdim=$i
    python train_cvae_extend.py --model=1 --ckpt_folder=grocery/zdim$i --data_dir=data/amazon2/ --iter=15 --zdim=$i
done

#
#python train_cvae_extend.py --model=1 --ckpt_folder=pre2/zdim500 --data_dir=data/movie/ --iter=15
#
#mkdir -p pre2/zdim50
#python train_vae.py --ckpt_folder=pre2/zdim50 --data_dir=data/movie
#python train_img_vae.py --ckpt_folder=pre3/zdim50_resnet --data_dir=data/amazon2/
#python train_cvae_extend.py --model=0 --ckpt_folder=pre_model/new_z --data_dir=data/amazon/ --iter=15
#python train_cvae_extend.py --model=1 --ckpt_folder=pre_model/new_z --data_dir=data/amazon/ --iter=15
#python train_cvae_extend.py --model=0 --ckpt_folder=pre2/v1 --data_dir=data/movie/ --iter=15
#python train_cvae_extend.py --model=1 --ckpt_folder=pre2/v1 --data_dir=data/movie/ --iter=15

