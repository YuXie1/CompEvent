import os
import time
import torch
from torch.utils import data as data
from torch.utils.data import ConcatDataset
from torchvision import transforms
import numpy as np
import random
from basicsr.utils.utils import randomCrop
from PIL import Image
from torchvision import transforms as TF
from torchvision.transforms import functional as F

class Train_Video_Dataset(data.Dataset):
    def __init__(self, opt, data_path):
        super(Train_Video_Dataset, self).__init__()

        self.rank = int(os.environ.get('RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))

        self.num_frames_seq = opt['num_train_video_frames']
        self.middle_frame_id = self.num_frames_seq // 2
        self.event_vox_prefix = 'event_voxel_parsed'
        self.blur_image_prefix = 'blur_processed_parsed'
        self.sharp_image_prefix = 'gt_processed_parsed'
        self.transform = transforms.ToTensor()
        self.get_filetaxnomy(data_path)

        self.use_flip = opt.get('use_flip', True)
        self.crop_height = opt['crop_size']
        self.crop_width = opt['crop_size']

        self.mean = opt.get('mean', [0.5, 0.5, 0.5])
        self.std = opt.get('std', [0.5, 0.5, 0.5])
        self.norm_voxel = opt.get('norm_voxel', True)

        self.val_num_images = opt.get('val_num_images', None)
        self.is_debug = 'debug' in opt.get('name', '')
    
    def get_filetaxnomy(self, data_dir):
        self.input_dict = {}
        self.input_dict['blur_images'] = {}
        self.input_dict['blur_images']['0'] = []
        self.input_dict['blur_images']['1'] = []
        self.input_dict['blur_images']['2'] = []
        self.input_dict['blur_images']['3'] = []
        self.input_dict['sharp_images'] = {}
        self.input_dict['sharp_images']['0'] = []
        self.input_dict['sharp_images']['1'] = []
        self.input_dict['sharp_images']['2'] = []
        self.input_dict['sharp_images']['3'] = []
        self.input_dict['event_voxel'] = {}
        self.input_dict['event_voxel']['0'] = []
        self.input_dict['event_voxel']['1'] = []
        self.input_dict['event_voxel']['2'] = []
        self.input_dict['event_voxel']['3'] = []
        
        print(f"Scanning directory: {data_dir}")

        all_subfolders = os.listdir(data_dir)
        print(f"Subfolders found in {data_dir}: {all_subfolders}")

        for scene in all_subfolders:
            scene_path = os.path.join(data_dir, scene)
            
            if os.path.isdir(scene_path):
                print(f"Checking scene folder: {scene_path}")

                event_voxel_dir = os.path.join(scene_path, self.event_vox_prefix)
                blur_image_dir = os.path.join(scene_path, self.blur_image_prefix)
                sharp_image_dir = os.path.join(scene_path, self.sharp_image_prefix)

                if os.path.exists(event_voxel_dir) and os.path.exists(blur_image_dir) and os.path.exists(sharp_image_dir):
                    print(f"Found valid subfolders in {scene_path}")
                    
                    for patch_idx in range(4):
                        event_vox_patch_dir = os.path.join(event_voxel_dir, str(patch_idx).zfill(5))
                        blur_patch_dir = os.path.join(blur_image_dir, str(patch_idx).zfill(5))
                        sharp_patch_dir = os.path.join(sharp_image_dir, str(patch_idx).zfill(5))

                        num_blur_images = len(os.listdir(blur_patch_dir))
                        print(f"Number of blur images in patch {patch_idx}: {num_blur_images}")

                        for image_idx in range(num_blur_images):
                            blur_name = os.path.join(blur_patch_dir, str(image_idx).zfill(5) + '.png')
                            sharp_name = os.path.join(sharp_patch_dir, str(image_idx).zfill(5) + '.png')
                            left_voxel_name = os.path.join(event_vox_patch_dir, str(image_idx).zfill(5) + '.npz')
                            self.input_dict['blur_images'][str(patch_idx)].append(blur_name)
                            self.input_dict['sharp_images'][str(patch_idx)].append(sharp_name)
                            self.input_dict['event_voxel'][str(patch_idx)].append(left_voxel_name)
                else:
                    print(f"Required subfolders not found in {scene_path}")
                    continue
            else:
                print(f"Skipping non-directory: {scene_path}")

        if not self.input_dict['blur_images']['0']:
            raise FileNotFoundError(f"Missing required subfolders in {data_dir}")

    def transform_frame(self, frame_tensor):
        if self.mean is not None and self.std is not None and self.mean != '~' and self.std != '~':
            frame_tensor = TF.normalize(frame_tensor, self.mean, self.std)
        return frame_tensor

    def transform_voxel(self, voxel_tensor):
        if self.norm_voxel:

            max_abs = max(abs(voxel_tensor.min().item()), abs(voxel_tensor.max().item()))
            if max_abs > 0:
                voxel_tensor = voxel_tensor / max_abs
        return voxel_tensor

    def random_transform(self, img, voxel, seed=None):

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        if self.use_flip and random.random() > 0.5:
            img = F.hflip(img)
            voxel = torch.flip(voxel, dims=[-1])

        _, h, w = img.shape
        if self.crop_height is not None and self.crop_width is not None and h > self.crop_height and w > self.crop_width:
            top = random.randint(0, h - self.crop_height)
            left = random.randint(0, w - self.crop_width)
            img = img[:, top:top+self.crop_height, left:left+self.crop_width]
            voxel = voxel[:, top:top+self.crop_height, left:left+self.crop_width]
        
        return img, voxel

    def __getitem__(self, index):
        rand_patch_idx = np.random.randint(0, 4)
        event_vox_list = []
        blur_list, gt_list = [], []
        seed = random.randint(0, 2 ** 32)

        half = self.num_frames_seq // 2
        total_frames = len(self.input_dict['blur_images'][str(rand_patch_idx)])
        indices = []
        for offset in range(-half, half+1):
            idx = index + offset
            if idx < 0:
                idx = 0
            elif idx >= total_frames:
                idx = total_frames - 1
            indices.append(idx)
        for video_num_idx in indices:
            left_event_vox = np.load(self.input_dict['event_voxel'][str(rand_patch_idx)][video_num_idx])["data"]
            left_event_vox_tensor = torch.from_numpy(left_event_vox).float()
            left_event_vox_tensor = self.transform_voxel(left_event_vox_tensor)
            blur_image = Image.open(self.input_dict['blur_images'][str(rand_patch_idx)][video_num_idx])
            gt_image = Image.open(self.input_dict['sharp_images'][str(rand_patch_idx)][video_num_idx])
            blur_image_tensor = self.transform(blur_image)
            gt_image_tensor = self.transform(gt_image)

            blur_image_tensor = self.transform_frame(blur_image_tensor)
            gt_image_tensor = self.transform_frame(gt_image_tensor)

            blur_image_tensor, left_event_vox_tensor = self.random_transform(blur_image_tensor, left_event_vox_tensor, seed)
            gt_image_tensor, _ = self.random_transform(gt_image_tensor, left_event_vox_tensor, seed)
            event_vox_list.append(left_event_vox_tensor[None, ...])
            blur_list.append(blur_image_tensor[None, ...])
            gt_list.append(gt_image_tensor[None, ...])
        
        blur_input_clip = torch.cat(blur_list)
        gt_clip = torch.cat(gt_list)
        gt_clip_middle = gt_clip[self.middle_frame_id]
        event_vox_tensor = torch.cat(event_vox_list)
        
        sample = {
            'clean_gt_clip': gt_clip,
            'clean_middle': gt_clip_middle,
            'blur_input_clip': blur_input_clip,
            'event_vox_clip': event_vox_tensor
        }
        return sample

    def __len__(self):

        if len(self.input_dict['blur_images']['0']) == 0:
            raise ValueError("Data is empty. Check the input data.")
        base_len = len(self.input_dict['blur_images']['0'])

        if self.val_num_images is not None:
            return min(base_len, self.val_num_images)
        return base_len

class Test_Video_Dataset(data.Dataset):
    def __init__(self, opt, data_path):
        super(Test_Video_Dataset, self).__init__()

        self.num_frames_seq = opt['num_test_video_frames']
        self.middle_frame_id = self.num_frames_seq // 2
        self.event_vox_prefix = 'event_voxel'
        self.blur_image_prefix = 'blur_processed'
        self.sharp_image_prefix = 'gt_processed'
        self.transform = transforms.ToTensor()
        self.get_filetaxnomy(data_path)

        self.mean = opt.get('mean', [0.5, 0.5, 0.5])
        self.std = opt.get('std', [0.5, 0.5, 0.5])
        self.norm_voxel = opt.get('norm_voxel', True)

    def get_filetaxnomy(self, data_dir):
        self.input_dict = {}
        self.input_dict['blur_images'] = []
        self.input_dict['sharp_images'] = []
        self.input_dict['event_voxel'] = []
        
        print(f"Scanning directory: {data_dir}")

        all_subfolders = os.listdir(data_dir)
        print(f"Subfolders found in {data_dir}: {all_subfolders}")

        for scene in all_subfolders:
            scene_path = os.path.join(data_dir, scene)
            
            if os.path.isdir(scene_path):
                print(f"Checking scene folder: {scene_path}")

                event_voxel_dir = os.path.join(scene_path, self.event_vox_prefix)
                blur_image_dir = os.path.join(scene_path, self.blur_image_prefix)
                sharp_image_dir = os.path.join(scene_path, self.sharp_image_prefix)

                if os.path.exists(event_voxel_dir) and os.path.exists(blur_image_dir) and os.path.exists(sharp_image_dir):
                    print(f"Found valid subfolders in {scene_path}")

                    num_blur_images = len(os.listdir(blur_image_dir))
                    print(f"Number of blur images in scene {scene}: {num_blur_images}")

                    for image_idx in range(num_blur_images):
                        blur_name = os.path.join(blur_image_dir, str(image_idx).zfill(5) + '.png')
                        sharp_name = os.path.join(sharp_image_dir, str(image_idx).zfill(5) + '.png')
                        left_voxel_name = os.path.join(event_voxel_dir, str(image_idx).zfill(5) + '.npz')
                        self.input_dict['blur_images'].append(blur_name)
                        self.input_dict['sharp_images'].append(sharp_name)
                        self.input_dict['event_voxel'].append(left_voxel_name)
                else:
                    print(f"Required subfolders not found in {scene_path}, skipping...")
                    continue
            else:
                print(f"Skipping non-directory: {scene_path}")

        if not self.input_dict['blur_images']:
            raise FileNotFoundError(f"Missing required subfolders in {data_dir}")
        
        print(f"Total images loaded: {len(self.input_dict['blur_images'])}")

    def transform_frame(self, frame_tensor):
        if self.mean is not None and self.std is not None and self.mean != '~' and self.std != '~':
            frame_tensor = TF.normalize(frame_tensor, self.mean, self.std)
        return frame_tensor

    def transform_voxel(self, voxel_tensor):
        if self.norm_voxel:
            max_abs = max(abs(voxel_tensor.min().item()), abs(voxel_tensor.max().item()))
            if max_abs > 0:
                voxel_tensor = voxel_tensor / max_abs
        return voxel_tensor

    def __getitem__(self, index):
        event_vox_list = []
        blur_list, gt_list = [], []
        half = self.num_frames_seq // 2
        total_frames = len(self.input_dict['blur_images'])
        indices = []
        for offset in range(-half, half+1):
            idx = index + offset
            if idx < 0:
                idx = 0
            elif idx >= total_frames:
                idx = total_frames - 1
            indices.append(idx)
        for video_num_idx in indices:
            left_event_vox = np.load(self.input_dict['event_voxel'][video_num_idx])["data"]
            left_event_vox_tensor = torch.from_numpy(left_event_vox)
            blur_image = Image.open(self.input_dict['blur_images'][video_num_idx])
            gt_image = Image.open(self.input_dict['sharp_images'][video_num_idx])
            blur_image_tensor = self.transform(blur_image)
            gt_image_tensor = self.transform(gt_image)
            event_vox_list.append(left_event_vox_tensor[None, ...])
            blur_list.append(blur_image_tensor[None, ...])
            gt_list.append(gt_image_tensor[None, ...])
        blur_input_clip = torch.cat(blur_list)
        gt_clip = torch.cat(gt_list)
        gt_clip_middle = gt_clip[self.middle_frame_id]
        event_vox_tensor = torch.cat(event_vox_list)
        sample = {}
        sample['clean_gt_clip'] = gt_clip
        sample['clean_middle'] = gt_clip_middle
        sample['blur_input_clip'] = blur_input_clip
        sample['event_vox_clip'] = event_vox_tensor
        return sample

    def __len__(self):
        return len(self.input_dict['blur_images'])

def get_train_dataset(args):
    data_with_mode = os.path.join(args.data_dir, 'train')
    scene_list = os.listdir(data_with_mode)
    dataset_list = []
    for scene in scene_list:
        data_path = os.path.join(data_with_mode, scene)
        dset = Train_Video_Dataset(args, data_path)
        dataset_list.append(dset)
    dataset_train_concat = ConcatDataset(dataset_list)
    return dataset_train_concat

def get_test_dataset(args):
    data_with_mode = os.path.join(args.data_dir, 'test')
    scene_list = os.listdir(data_with_mode)
    dataset_list = []
    for scene in scene_list:
        data_path = os.path.join(data_with_mode, scene)
        dsets = Test_Video_Dataset(args, data_path)
        dataset_list.append(dsets)
    dataset_test_concat = ConcatDataset(dataset_list)
    return dataset_test_concat
