mkdir citeu/z4_50
cp citeu/z2_50/vae_* citeu/z4_50/.
python train_cave.py --ckpt_folder=citeu/z4_50 --data_dir=data/citeulike-a/ --iter=15 --zdim=50 --model=0

python train_cf_dae.py --ckpt_folder=pre3/dae --data_dir=data/amazon2/ --iter=15  --zdim=50 --model=1




#for i in 50 100 150
#do
#    mkdir pre2/z3_$i
#    python train_vae.py --ckpt_folder=pre2/z3_$i --data_dir=data/movie/ --zdim=$i
#    python train_cvae_extend.py --model=0 --ckpt_folder=pre2/z3_$i --data_dir=data/movie/ --iter=100 --zdim=$i
#    python train_img_vae.py --ckpt_folder=pre2/z3_$i --data_dir=data/movie/ --zdim=$i
#    python train_cvae_extend.py --model=1 --ckpt_folder=pre2/z3_$i --data_dir=data/movie/ --iter=100 --zdim=$i
#done
