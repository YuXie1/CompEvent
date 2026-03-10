import numpy as np
import os
from PIL import Image
from PIL import ImageDraw 

# 缩放0.5倍

def main():
    
    root_path = '/gdata/linrj/EF-SAI-Dataset/total/256/test/'

    event_path = root_path + 'event/'

    event_visualize(event_path, root_path + 'event_vis_20ms/')

def event_visualize(event_path, save_path):
    event_files = os.listdir(event_path)
    event_files.sort()
    img_size = (260,346) #Resolution of Davis346
    if not(os.path.exists(save_path)):
        os.makedirs(save_path)
    
    for filename in event_files:
        print('Processing ',filename)
        ev_data = np.load(event_path+filename, allow_pickle=True).item()
        events = ev_data['events']
        t = events['t'] #timestamp
        x = events['x']
        y = events['y']
        p = events['p'] #plority
        reference_time = ev_data['ref_t']
        v = ev_data['v']         #Speed of slide motion
        fx = ev_data['fx']       #Camera intrinsics
        depth = ev_data['depth'] #Depth of the target
        x = x + np.round(fx*v*(t-reference_time)/depth) #Refocusing events to reference time
        useless_idx = np.where((x<0)|(x>=img_size[1]))  #Find and remove the events that are outside the image boundaries
        t = np.delete(t,useless_idx)
        x = np.delete(x,useless_idx)
        y = np.delete(y,useless_idx)
        p = np.delete(p,useless_idx)
        refocused_events = dict()
        refocused_events['x'] = x.astype(np.uint16)
        refocused_events['y'] = y
        refocused_events['t'] = t
        refocused_events['p'] = p
        ev_data['events'] = refocused_events

        event_pos = np.zeros((img_size[0], img_size[1]))
        event_neg = np.zeros((img_size[0], img_size[1]))
        deltaT = 0.02

        for i in range(0, len(t)):
            if t[i] > t[0] + deltaT:
                break
            if p[i] == 1:
                event_pos[int(y[i]), int(x[i])] += 1        
            else:
                event_neg[int(y[i]), int(x[i])] += 1  
            
        img = Image.new('RGB', (img_size[1], img_size[0]), (255, 255, 255))
        draw = ImageDraw.Draw(img) 
        
        for i in range(img_size[0]):
            for j in range(img_size[1]):
                if event_pos[i, j] > event_neg[i, j]:
                    draw.point((j, i) , 'red')
                elif event_pos[i, j] < event_neg[i, j]:
                    draw.point((j, i) , 'blue')

        picname = filename.split('.')[0] + '.png'
        img.save(save_path + picname)
        

def event_visualize1(event_path, save_path, imgsize):

    event_files = os.listdir(event_path)
    event_files.sort()
    
    if not(os.path.exists(save_path)):
        os.makedirs(save_path)

    for filename in event_files:
        print('Processing ',filename)
        ev_data = np.load(event_path+filename, allow_pickle=True).item()
        event_pos = ev_data['Pos']
        event_neg = ev_data['Neg']
        event_pos = np.sum(event_pos, axis=0)
        event_neg = np.sum(event_neg, axis=0)

        img = Image.new('RGB', (imgsize, imgsize), (255, 255, 255))
        draw = ImageDraw.Draw(img) 
        
        for i in range(imgsize):
            for j in range(imgsize):
                if event_pos[i, j] > event_neg[i, j]:
                    draw.point((j, i) , 'red')
                elif event_pos[i, j] < event_neg[i, j]:
                    draw.point((j, i) , 'blue')

        picname = filename.split('.')[0] + '.png'
        img.save(save_path + picname)




if __name__ == '__main__':
    main()    