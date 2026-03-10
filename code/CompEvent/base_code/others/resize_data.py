import numpy as np
import os
from PIL import Image

# 缩放0.5倍

def main():

    # root_path = '/gdata/linrj/EF-SAI-Dataset/total/256/test/'
    # save_path = '/gdata/linrj/EF-SAI-Dataset/total/128/test/'
    
    root_path = '/gdata/linrj/EF-SAI-Dataset/total/256/train/'
    save_path = '/gdata/linrj/EF-SAI-Dataset/total/128/train/'

    eframe_path = root_path + 'eframe_re/'
    event_path = root_path + 'event_re/'
    frame_path = root_path + 'frame_re/'
    gt_path = root_path + 'gt_re/'

    event_resize(event_path, save_path + 'event_re/')
    frame_resize(eframe_path, save_path + 'eframe_re/')
    frame_resize(frame_path, save_path + 'frame_re/')
    gt_resize(gt_path, save_path + 'gt_re/')



def event_resize(event_path, save_path):

    event_files = os.listdir(event_path)
    event_files.sort()

    if not(os.path.exists(save_path)):
        os.makedirs(save_path)

    for filename in event_files:
        print('Processing ',filename)
        ev_data = np.load(event_path+filename, allow_pickle=True).item()
        event_pos = ev_data['Pos']
        event_neg = ev_data['Neg']

        new_event_pos = np.zeros((30, 128, 128))
        new_event_neg = np.zeros((30, 128, 128))
        
        for i in range(128):
            for j in range(128):
                for k in range(30):
                    new_event_pos[k, i, j] = (event_pos[k, i*2, j*2] + event_pos[k, i*2+1, j*2] + event_pos[k, i*2, j*2+1] + event_pos[k, i*2+1, j*2+1])/4
                    new_event_neg[k, i, j] = (event_neg[k, i*2, j*2] + event_neg[k, i*2+1, j*2] + event_neg[k, i*2, j*2+1] + event_neg[k, i*2+1, j*2+1])/4
                    
        event_pos = new_event_pos
        event_neg = new_event_neg

        out = {}
        out['Pos'] = event_pos
        out['Neg'] = event_neg

        np.save(save_path+filename, out)


def frame_resize(frame_path, save_path):

    frame_files = os.listdir(frame_path)
    frame_files.sort()

    if not(os.path.exists(save_path)):
        os.makedirs(save_path)

    for filename in frame_files:
        print('Processing ',filename)
        frame = np.load(frame_path + filename)
        new_frame_data = np.zeros((30, 128, 128))
        for i in range(128):
            for j in range(128):
                for k in range(30):
                    new_frame_data[k, i, j] = (frame[k, i*2, j*2] + frame[k, i*2+1, j*2] + frame[k, i*2, j*2+1] + frame[k, i*2+1, j*2+1])/4
        np.save(save_path+filename, new_frame_data)


        

def gt_resize(gt_path, save_path):

    gt_files = os.listdir(gt_path)
    gt_files.sort()

    if not(os.path.exists(save_path)):
        os.makedirs(save_path)

    for filename in gt_files:
        print('Processing ',filename)
        gt= Image.open(gt_path+filename)
        # 将图片转换为数组
        gt_data = np.array(gt)
        new_gt_data = np.zeros((128, 128, 3))
        for i in range(128):
            for j in range(128):
                for k in range(3):
                    new_gt_data[i, j, k] = (int(gt_data[i*2, j*2]) + int(gt_data[i*2+1, j*2]) + int(gt_data[i*2, j*2+1]) + int(gt_data[i*2+1, j*2+1]))/4           
        gt = Image.fromarray(np.uint8(new_gt_data))
        gt = gt.convert("L")
        gt.save(save_path+filename)


if __name__ == '__main__':
    main()    