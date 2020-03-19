folders="Automotive Baby CD Clothing Garden Grocery Instrument Kindle Music Office Pet Phone Video"
for f in $folders; do
    echo ${f}
    python multiVAE.py --data_dir=/media/linh/DATA/research/cf-vae/data2/${f}/ --data_type=1p --iter=200  >> /media/linh/DATA/research/new_cf/result/multi_${f}_1.txt
    python triple_vae.py --data_dir=/media/linh/DATA/research/cf-vae/data2/${f}/ --data_type=1p  --iter=200  >> /media/linh/DATA/research/new_cf/result/${f}_1.txt
    python multiVAE.py --data_dir=/media/linh/DATA/research/cf-vae/data2/${f}/ --data_type=8p --iter=200 >> /media/linh/DATA/research/new_cf/result/multi_${f}_8.txt
    python triple_vae.py --data_dir=../cf-vae/data2/${f}/ --data_type=8p  --iter=200 --learning_rate=0.001 >> /media/linh/DATA/research/new_cf/result/${f}_8.txt
done