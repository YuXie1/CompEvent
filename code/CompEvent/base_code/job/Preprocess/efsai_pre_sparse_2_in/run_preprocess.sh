#!/bin/bash

root_path="/gdata/linrj/EF-SAI-Dataset/sparse/2/indoor/train"

cd ../../../
python ./preprocess/npy2txt.py --root_path ${root_path}"/"
./preprocess/do_E2VID.sh ${root_path}
python ./preprocess/res2npy.py --root_path ${root_path}"/"
python ./preprocess/data_refocus.py --root_path ${root_path}"/"
