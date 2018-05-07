#python train_cvae_extend.py --model=6 --ckpt_folder=grocery/z4_50 --data_dir=data/amazon2/ --iter=70 --zdim=50
#python train_cvae_extend.py --model=0 --ckpt_folder=grocery/z4_50 --data_dir=data/amazon2/ --iter=150 --zdim=50
#python train_cvae_extend.py --model=1 --ckpt_folder=grocery/z6_50 --data_dir=data/amazon2/ --iter=150 --zdim=50
#
#python train_vae.py --ckpt_folder=pre2/z6_50 --data_dir=data/movie/ --zdim=50
#python train_img_vae.py --ckpt_folder=pre2/z6_50 --data_dir=data/movie/ --zdim=50
#python train_cvae_extend.py --model=0 --ckpt_folder=pre2/z6_50 --data_dir=data/movie/ --iter=100 --zdim=50
##python train_cvae_extend.py --model=6 --ckpt_folder=pre2/z6_50 --data_dir=data/movie/ --iter=100 --zdim=50
#python train_cvae_extend.py --model=1 --ckpt_folder=pre2/z6_50 --data_dir=data/movie/ --iter=100 --zdim=50

#python train_ave.py --ckpt_folder=citeu/z1_50 --data_dir=data/citeulike-a/
python train_cave.py --model=0 --ckpt_folder=citeu/z1_50 --data_dir=data/citeulike-a/ --iter=200 --zdim=50
mkdir -p citeut/z1_50
python train_ave.py --ckpt_folder=citeut/z1_50 --data_dir=data/citeulike-t/
python train_cave.py --model=0 --ckpt_folder=citeut/z1_50 --data_dir=data/citeulike-t/ --iter=200 --zdim=50
git pull origin resnet
mkdir citeu/z2_50

python train_ave.py --ckpt_folder=citeu/z1_50 --data_dir=data/citeulike-a/
python train_cave.py --model=0 --ckpt_folder=citeu/z1_50 --data_dir=data/citeulike-a/ --iter=200 --zdim=50
#mkdir pre2/z6_100
#python train_vae.py --ckpt_folder=pre2/z6_100 --data_dir=data/movie/ --zdim=100
#python train_img_vae.py --ckpt_folder=pre2/z6_100 --data_dir=data/movie/ --zdim=100
#python train_cvae_extend.py --model=1 --ckpt_folder=pre2/z6_100 --data_dir=data/movie/ --iter=100 --zdim=100

#for i in 50 100 150
#do
#    mkdir grocery/z3_$i
#    python train_vae.py --ckpt_folder=grocery/z3_$i --data_dir=data/amazon2/ --zdim=$i
#    python train_cvae_extend.py --model=0 --ckpt_folder=grocery/z3_$i --data_dir=data/amazon2/ --iter=100 --zdim=$i
#    python train_img_vae.py --ckpt_folder=grocery/z3_$i --data_dir=data/amazon2/ --zdim=$i
#    python train_cvae_extend.py --model=1 --ckpt_folder=grocery/z3_$i --data_dir=data/amazon2/ --iter=100 --zdim=$i
#done
#
#
#for i in 50 100 150
#do
#    mkdir pre2/z3_$i
#    python train_vae.py --ckpt_folder=pre2/z3_$i --data_dir=data/movie/ --zdim=$i
#    python train_cvae_extend.py --model=0 --ckpt_folder=pre2/z3_$i --data_dir=data/movie/ --iter=100 --zdim=$i
#    python train_img_vae.py --ckpt_folder=pre2/z3_$i --data_dir=data/movie/ --zdim=$i
#    python train_cvae_extend.py --model=1 --ckpt_folder=pre2/z3_$i --data_dir=data/movie/ --iter=100 --zdim=$i
#done
