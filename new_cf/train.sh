#folders="Automotive Baby CD Clothing Garden Grocery Kindle Music Office Pet Phone Video"
#python src/preprocessing.py
#folders="Tool Outdoor Kitchen TV Beauty Toy Automotive Baby CD Garden Grocery Kindle Music Office Pet Phone Video"
#folders="Tool Beauty TV Toy Garden Office Kindle"
python src/preprocessing.py
folders="Music Office Pet Phone Video Garden Beauty Kindle"
for f in $folders; do
    echo ${f}

#    python neuCF/GMF.py --path data/${f}/ --dataset ${f}1 --epochs 20 --batch_size 512 --num_factors 50 --regs [0,0] --num_neg 10 --lr 0.001 --learner adam --verbose 1 --out 1
#    python neuCF/MLP.py --path data/${f}/ --dataset ${f}1 --epochs 20 --batch_size 512 --layers [1000,200,50] --reg_layers [0,0,0] --num_neg 10 --lr 0.001 --learner adam --verbose 1 --out 1
#     python neuCF/NeuMF.py --path data/${f}/ --dataset ${f}1 --type 1p --epochs 40 --batch_size 512 --num_factors 50 --layers [1000,200,50] --reg_layers [0,0,0] --num_neg 10 --lr 0.001 --learner adam --verbose 1 --out 1 --mf_pretrain Pretrain/${f}1_GMF.h5 --mlp_pretrain Pretrain/${f}1_MLP.h5 >>/media/linh/DATA/research/new_cf/result8/neuCF_${f}_1.txt
#
#    python neuCF/GMF.py --path data/${f}/ --dataset ${f}8 --epochs 20 --batch_size 512 --num_factors 50 --regs [0,0] --num_neg 10 --lr 0.001 --learner adam --verbose 1 --out 1
#    python neuCF/MLP.py --path data/${f}/ --dataset ${f}8 --epochs 20 --batch_size 512 --layers [1000,200,50] --reg_layers [0,0,0] --num_neg 10 --lr 0.001 --learner adam --verbose 1 --out 1
#python neuCF/NeuMF.py --path data/Kindle/ --dataset Kindle8 --type 8p --epochs 40 --batch_size 512 --num_factors 50 --layers [1000,200,50] --reg_layers [0,0,0] --num_neg 10 --lr 0.001 --learner adam --verbose 1 --out 1 --mf_pretrain Pretrain/Kindle8_GMF.h5 --mlp_pretrain Pretrain/Kindle8_MLP.h5 >>/media/linh/DATA/research/new_cf/result8/neuCF_Kindle_8.txt

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

#python triple_vae.py --data_dir=../cf-vae/data2/Garden/ --data_type=8p  --iter=400 --learning_rate=0.0001

#python neuCF/NeuMF.py --path data/${f}/ --dataset ${f}1 --type 1p --epochs 2 --batch_size 512 --num_factors 8 --layers [64,32,16,8] --num_neg 10 --lr 0.001 --learner adam --verbose 1 --out 1 --mf_pretrain Pretrain/${f}1_GMF.h5 --mlp_pretrain Pretrain/${f}1_MLP.h5