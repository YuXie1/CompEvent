import h5py
import os
org_path = "/data/queahren/Low_level_datasets/GOPRO/GOPRO_SCER/test"
# tar_path = "/data/queahren/Low_level_datasets/GOPRO/GOPRO_new/train"

# 指定原始HDF5文件路径和新的HDF5文件路径
h5_file_path = [os.path.join(org_path, s) for s in os.listdir(org_path)]
for h5_file in h5_file_path:

    original_h5_path = h5_file
    out_path = '/data/queahren/Low_level_datasets/GOPRO/GOPRO_SS/test/'

    video_name = original_h5_path.split('/')[-1].split('.')[0]

    with h5py.File(original_h5_path, 'r') as original_h5:
        dataset_len = len(original_h5['images'].keys())

        for index in range(dataset_len):

            image_data = original_h5['images']['image{:09d}'.format(index)][:]
            sharp_image_data = original_h5['sharp_images']['image{:09d}'.format(index)][:]
            voxel_data = original_h5['voxels']['voxel{:09d}'.format(index)][:]
            mask = original_h5['masks']['mask{:09d}'.format(index)][:]
            new_h5_path = out_path + video_name + '_{:09d}'.format(index)
            with h5py.File(new_h5_path, 'w') as new_h5:
                # 创建数据集并写入数据
                new_h5.create_dataset('image'.format(index), data=image_data)
                new_h5.create_dataset('sharp'.format(index), data=sharp_image_data)
                new_h5.create_dataset('voxel'.format(index), data=voxel_data)
                new_h5.create_dataset('mask'.format(index), data=mask)