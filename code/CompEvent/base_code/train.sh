
# cd /code/base_code; ./train.sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 
# python train.py -opt /code/video_deblur/base_code/options/train_newdata/train_efnet.yml
# python train.py -opt /code/video_deblur/base_code/options/train_newdata/train_evs.yml
python train.py -opt /code/video_deblur/base_code/options/train_newdata/train_eifnet.yml