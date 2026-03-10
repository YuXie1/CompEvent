import os, shutil


def copy_files(i, j):

    files = os.listdir(path_test + '/eframe_re')
    files.sort()

    print('Test:')
    for file in files:
        filename = file.split('.')[0]
        new_filename = str(i + int(filename)).zfill(4)
        print(filename, new_filename)
        # shutil.copy(path_test + '/eframe_re/' + filename + '.npy',   save_path_test + '/eframe_re/' + new_filename + '.npy')
        # shutil.copy(path_test + '/frame_re/' + filename + '.npy',    save_path_test + '/frame_re/' +  new_filename + '.npy')
        # shutil.copy(path_test + '/event_re/' + filename + '.npy',    save_path_test + '/event_re/' +  new_filename + '.npy')
        # shutil.copy(path_test + '/gt_re/'    + filename + '.png',    save_path_test + '/gt_re/' +  new_filename + '.png')
        shutil.copy(path_test + '/event/'    + filename + '.npy',    save_path_test + '/event/' +  new_filename + '.npy')

    files = os.listdir(path_train + '/eframe_re')
    files.sort()

    # print('Train:')
    # for file in files:
    #     filename = file.split('.')[0]
    #     new_filename = str(j + int(filename)).zfill(4)
    #     print(filename, new_filename)
    #     # shutil.copy(path_train + '/eframe_re/' + filename + '.npy',   save_path_train + '/eframe_re/' + new_filename + '.npy')
    #     # shutil.copy(path_train + '/frame_re/' + filename + '.npy',    save_path_train + '/frame_re/' +  new_filename + '.npy')
    #     # shutil.copy(path_train + '/event_re/' + filename + '.npy',    save_path_train + '/event_re/' +  new_filename + '.npy')
    #     # shutil.copy(path_train + '/gt_re/'    + filename + '.png',    save_path_train + '/gt_re/' +  new_filename + '.png')
    #     shutil.copy(path_test + '/event/'    + filename + '.npy',    save_path_test + '/event/' +  new_filename + '.npy')
    
         

path = '/gdata/linrj/EF-SAI-Dataset/sparse/4/indoor'
path_test = path + '/test'
path_train = path + '/train'

save_path = '/gdata/linrj/EF-SAI-Dataset/total/256'
save_path_test = save_path + '/test'
save_path_train = save_path + '/train'

copy_files(98, 0)



