folders="Tool Outdoor Kitchen TV Beauty Toy Health Automotive Baby CD Clothing Garden Grocery Kindle Music Office Pet Phone Video"
for f in $folders; do
    echo ${f}
#    python multiVAE.py --data_dir=/media/linh/DATA/research/cf-vae/data2/${f}/ --data_type=1p --iter=200 --learning_rate=0.001 >> /media/linh/DATA/research/new_cf/result5/multi_${f}_1.txt
#    python triple_vae.py --data_dir=/media/linh/DATA/research/cf-vae/data2/${f}/ --data_type=1p  --iter=200 --learning_rate=0.001 >> /media/linh/DATA/research/new_cf/result5/${f}_1.txt
#    python multiVAE.py --data_dir=/media/linh/DATA/research/cf-vae/data2/${f}/ --data_type=8p --iter=200 --learning_rate=0.001 >> /media/linh/DATA/research/new_cf/result5/multi_${f}_8.txt
    python triple_vae.py --data_dir=../cf-vae/data2/${f}/ --data_type=8p  --iter=200 --learning_rate=0.001 >> /media/linh/DATA/research/new_cf/result6/${f}_8.txt
#    mkdir experiment/${f}/
#    mkdir experiment/${f}/1
#    python train_vae.py --ckpt_folder=experiment/${f}/1 --data_dir=../cf-vae/data2/${f}/ --data_type=1
#    python train_vae.py --ckpt_folder=experiment/${f}/1 --data_dir=data/${f}/ --data_type=1 --type=user
#     python train_cvae_user.py --data_dir=../cf-vae/data2/${f}/ --ckpt_folder=experiment/${f}/1 --data_type=1 --gridsearch=1
#
#     mkdir experiment/${f}/8
#     python train_vae.py --ckpt_folder=experiment/${f}/8 --data_dir=../cf-vae/data2/${f}/ --data_type=8
#    python train_vae.py --ckpt_folder=experiment/${f}/8 --data_dir=data/${f}/ --data_type=8 --type=user
#     python train_cvae_user.py --data_dir=../cf-vae/data2/${f}/ --ckpt_folder=experiment/${f}/8 --data_type=8 --gridsearch=1


done


