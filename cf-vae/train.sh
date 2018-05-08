cd CollaborativeVAE
python test_cvae.py


cd ..
cd research/cf-vae
python train_cf_dae.py --model=0 --iter=15 --ckpt_folder=pre3/dae/

python train_cf_dae.py --model=1 --iter=15 --ckpt_folder=pre3/dae/

cd ..
cd cdl
python cdl.py



#for i in 50 100 150
#do
#    mkdir pre2/z3_$i
#    python train_vae.py --ckpt_folder=pre2/z3_$i --data_dir=data/movie/ --zdim=$i
#    python train_cvae_extend.py --model=0 --ckpt_folder=pre2/z3_$i --data_dir=data/movie/ --iter=100 --zdim=$i
#    python train_img_vae.py --ckpt_folder=pre2/z3_$i --data_dir=data/movie/ --zdim=$i
#    python train_cvae_extend.py --model=1 --ckpt_folder=pre2/z3_$i --data_dir=data/movie/ --iter=100 --zdim=$i
#done
