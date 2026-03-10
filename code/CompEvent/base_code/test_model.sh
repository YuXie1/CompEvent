#!/bin/bash

# root_path="/gdata/linrj/Val_data"
# root_path="/gdata/linrj/EF-SAI-Dataset/sparse/1/indoor/test"
# save_path="/gdata/linrj/EF-SAI-Dataset_Results/sparse_1_indoor_test_40000pth"

# root_path="/gdata/linrj/EF-SAI-Dataset/total/256/test"
root_path="/gdata/linrj/EF-SAI-Dataset/total/128/test"
# root_path="/gdata/linrj/Val_data"

# save_path="/gdata/linrj/EF-SAI-Dataset_Results/totalval_pretrainedpth"
# model_path="/ghome/linrj/XSY_NM/EF-SAI/PreTrained/EF_SAI_Net.pth"

# save_path="/gdata/linrj/EF-SAI-Dataset_Results/total256_lr7e-4_is256_10000Gpth"

# save_path="/gdata/linrj/EF-SAI-Dataset_Results/total256_lr7e-4_is256_10000Gpth"
# model_path="/ghome/linrj/XSY_NM/experiments/EFSAI_alldata128_lr7e-4_is256/models/10000_G.pth"

save_path="/gdata/linrj/EF-SAI-Dataset_Results/total128_lr1e-4_is128_latestGpth"
model_path="/ghome/linrj/XSY_NM/experiments/EFSAI_alldata128_lr7e-4_is256/models/latest_G.pth"

# model_path="/ghome/linrj/XSY_NM/experiments/DCSR_Eventx2_EvInt/models/latest_G.pth"

# python test.py \
        # --root_path ${root_path}"/" \
        # --model_path ${model_path}\
        # --re_save_path ${save_path}"/"
python ./metrics/calculate_PSNR_SSIM.py \
        --root_path ${save_path} \
        --cal_opt efsai_test