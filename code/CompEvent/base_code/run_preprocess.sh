#!/bin/bash

root_path="/gdata/linrj/EF-SAI-Dataset/sparse/4/indoor/train"
# save_path="/gdata/linrj/EF-SAI-Dataset_Results/dense_1_outdoor_train"

python ./preprocess/npy2txt.py --root_path ${root_path}"/"
./preprocess/do_E2VID.sh ${root_path}
python ./preprocess/res2npy.py --root_path ${root_path}"/"
python ./preprocess/data_refocus.py --root_path ${root_path}"/"

# python test.py \
#         --root_path ${root_path}"/" \
#         --re_save_path ${save_path}"/"
# python ./metrics/calculate_PSNR_SSIM.py --root_path ${save_path}