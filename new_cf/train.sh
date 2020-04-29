#folders="Automotive Baby CD Clothing Garden Grocery Kindle Music Office Pet Phone Video"
#folders="Tool Outdoor Kitchen TV Beauty Toy Automotive Baby CD Garden Grocery Kindle Music Office Pet Phone Video"
folders="Tool Kitchen Beauty TV Toy Garden Office Kindle"
for f in $folders; do
    echo ${f}
#    python cvae_cf.py --data_dir=/media/linh/DATA/research/cf-vae/data2/${f}/ --data_type=1p --iter=400 --learning_rate=0.001 >> /media/linh/DATA/research/new_cf/result9/cvaecf_${f}_1.txt
#    python cvae_cf.py --data_dir=/media/linh/DATA/research/cf-vae/data2/${f}/ --data_type=8p --iter=400 --learning_rate=0.001 >> /media/linh/DATA/research/new_cf/result9/cvaecf_${f}_8.txt
#    python multiVAE_old.py --data_dir=/media/linh/DATA/research/cf-vae/data2/${f}/ --data_type=1p --iter=200 --learning_rate=0.001 >> /media/linh/DATA/research/new_cf/result8/multi_${f}_1.txt
#    python multiVAE_old.py --data_dir=/media/linh/DATA/research/cf-vae/data2/${f}/ --data_type=8p --iter=200 --learning_rate=0.001 >> /media/linh/DATA/research/new_cf/result8/multi_${f}_8.txt
#    python triple_vae.py --data_dir=/media/linh/DATA/research/cf-vae/data2/${f}/ --data_type=1p  --iter=200 --learning_rate=0.001 >> /media/linh/DATA/research/new_cf/result7/${f}_1.txt
#    python new_model.py --data_dir=/media/linh/DATA/research/cf-vae/data2/${f}/ --data_type=8p --iter=200 --learning_rate=0.001 >> /media/linh/DATA/research/new_cf/result8/new_${f}_8.txt
    python triple_vae.py --data_dir=../cf-vae/data2/${f}/ --data_type=8p  --iter=400 --learning_rate=0.0001 >> /media/linh/DATA/research/new_cf/result8/${f}_8.txt
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

#python triple_vae.py --data_dir=../cf-vae/data2/Instrument/ --data_type=1p  --iter=200 --learning_rate=0.001
