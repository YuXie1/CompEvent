import os
import torch
from torch.utils import data as data
from torchvision import transforms as TF
import numpy as np
from PIL import Image
from torchvision.transforms import functional as F
import re

def group_by_scene(file_list):
    scene_dict = {}
    pattern = re.compile(r'^(\d{4})_')
    for f in file_list:
        m = pattern.match(os.path.basename(f))
        if m:
            scene = m.group(1)
            scene_dict.setdefault(scene, []).append(f)
    return scene_dict

class LOLBlurTrainVideoDataset(data.Dataset):
    def __init__(self, opt, data_path):
        super(LOLBlurTrainVideoDataset, self).__init__()
        self.num_frames_seq = opt['num_train_video_frames']
        self.middle_frame_id = self.num_frames_seq // 2
        self.event_vox_prefix = 'event_voxel_16'
        self.blur_image_prefix = 'blur'
        self.sharp_image_prefix = 'sharp'
        self.transform = TF.ToTensor()
        self.use_flip = opt.get('use_flip', True)
        self.crop_height = opt.get('crop_size', None)
        self.crop_width = opt.get('crop_size', None)
        self.mean = opt.get('mean', [0.5, 0.5, 0.5])
        self.std = opt.get('std', [0.5, 0.5, 0.5])
        self.norm_voxel = opt.get('norm_voxel', True)
        self.val_num_images = opt.get('val_num_images', None)
        self.is_debug = 'debug' in opt.get('name', '')
        self._build_clips(data_path)

    def _build_clips(self, data_dir):
        blur_dir = os.path.join(data_dir, self.blur_image_prefix)
        sharp_dir = os.path.join(data_dir, self.sharp_image_prefix)
        voxel_dir = os.path.join(data_dir, self.event_vox_prefix)
        blur_files = sorted([os.path.join(blur_dir, f) for f in os.listdir(blur_dir) if f.endswith('.png')])
        sharp_files = sorted([os.path.join(sharp_dir, f) for f in os.listdir(sharp_dir) if f.endswith('.png')])
        voxel_files = sorted([os.path.join(voxel_dir, f.replace('.png', '.npz')) for f in os.listdir(blur_dir) if f.endswith('.png')])
        assert len(blur_files) == len(sharp_files) == len(voxel_files), f"数据数量不一致: blur({len(blur_files)}), sharp({len(sharp_files)}), voxel({len(voxel_files)})"

        blur_scene = group_by_scene(blur_files)
        sharp_scene = group_by_scene(sharp_files)
        voxel_scene = group_by_scene(voxel_files)
        self.clips = []
        self.scene_blur = {}
        self.scene_sharp = {}
        self.scene_voxel = {}
        for scene in blur_scene:
            b_list = sorted(blur_scene[scene])
            s_list = sorted(sharp_scene[scene])
            v_list = sorted(voxel_scene[scene])
            assert len(b_list) == len(s_list) == len(v_list), f"场景{scene}数量不一致"
            self.scene_blur[scene] = b_list
            self.scene_sharp[scene] = s_list
            self.scene_voxel[scene] = v_list
            n = len(b_list)
            for center_idx in range(n):
                self.clips.append((scene, center_idx))
        if self.val_num_images is not None:
            self.clips = self.clips[:self.val_num_images]

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
            np.random.seed(seed)
        if self.use_flip and np.random.rand() > 0.5:
            img = F.hflip(img)
            voxel = torch.flip(voxel, dims=[-1])
        _, h, w = img.shape
        if self.crop_height is not None and self.crop_width is not None and h > self.crop_height and w > self.crop_width:
            top = np.random.randint(0, h - self.crop_height)
            left = np.random.randint(0, w - self.crop_width)
            img = img[:, top:top+self.crop_height, left:left+self.crop_width]
            voxel = voxel[:, top:top+self.crop_height, left:left+self.crop_width]
        return img, voxel

    def __getitem__(self, index):
        scene, center_idx = self.clips[index]
        b_list = self.scene_blur[scene]
        s_list = self.scene_sharp[scene]
        v_list = self.scene_voxel[scene]
        n = len(b_list)
        half = self.num_frames_seq // 2
        indices = []
        for offset in range(-half, half+1):
            idx = center_idx + offset
            if idx < 0:
                idx = 0
            elif idx >= n:
                idx = n - 1
            indices.append(idx)
        event_vox_list = []
        blur_list, gt_list = [], []
        seed = np.random.randint(0, 2 ** 32)
        for idx in indices:
            left_event_vox = np.load(v_list[idx])["voxel"]
            left_event_vox_tensor = torch.from_numpy(left_event_vox).float()
            left_event_vox_tensor = self.transform_voxel(left_event_vox_tensor)
            blur_image = Image.open(b_list[idx])
            gt_image = Image.open(s_list[idx])
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
        return len(self.clips)

class LOLBlurTestVideoDataset(data.Dataset):
    def __init__(self, opt, data_path):
        super(LOLBlurTestVideoDataset, self).__init__()
        self.num_frames_seq = opt['num_test_video_frames']
        self.middle_frame_id = self.num_frames_seq // 2
        self.event_vox_prefix = 'event_voxel_16'
        self.blur_image_prefix = 'blur'
        self.sharp_image_prefix = 'sharp'
        self.transform = TF.ToTensor()
        self.mean = opt.get('mean', [0.5, 0.5, 0.5])
        self.std = opt.get('std', [0.5, 0.5, 0.5])
        self.norm_voxel = opt.get('norm_voxel', True)
        self.val_num_images = opt.get('val_num_images', None)
        self.is_debug = 'debug' in opt.get('name', '')
        self._build_clips(data_path)

    def _build_clips(self, data_dir):
        blur_dir = os.path.join(data_dir, self.blur_image_prefix)
        sharp_dir = os.path.join(data_dir, self.sharp_image_prefix)
        voxel_dir = os.path.join(data_dir, self.event_vox_prefix)
        blur_files = sorted([os.path.join(blur_dir, f) for f in os.listdir(blur_dir) if f.endswith('.png')])
        sharp_files = sorted([os.path.join(sharp_dir, f) for f in os.listdir(sharp_dir) if f.endswith('.png')])
        voxel_files = sorted([os.path.join(voxel_dir, f.replace('.png', '.npz')) for f in os.listdir(blur_dir) if f.endswith('.png')])
        assert len(blur_files) == len(sharp_files) == len(voxel_files), f"数据数量不一致: blur({len(blur_files)}), sharp({len(sharp_files)}), voxel({len(voxel_files)})"

        blur_scene = group_by_scene(blur_files)
        sharp_scene = group_by_scene(sharp_files)
        voxel_scene = group_by_scene(voxel_files)
        self.clips = []
        self.scene_blur = {}
        self.scene_sharp = {}
        self.scene_voxel = {}
        for scene in blur_scene:
            b_list = sorted(blur_scene[scene])
            s_list = sorted(sharp_scene[scene])
            v_list = sorted(voxel_scene[scene])
            assert len(b_list) == len(s_list) == len(v_list), f"场景{scene}数量不一致"
            self.scene_blur[scene] = b_list
            self.scene_sharp[scene] = s_list
            self.scene_voxel[scene] = v_list
            n = len(b_list)
            for center_idx in range(n):
                self.clips.append((scene, center_idx))
        if self.val_num_images is not None:
            self.clips = self.clips[:self.val_num_images]

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
        scene, center_idx = self.clips[index]
        b_list = self.scene_blur[scene]
        s_list = self.scene_sharp[scene]
        v_list = self.scene_voxel[scene]
        n = len(b_list)
        half = self.num_frames_seq // 2
        indices = []
        for offset in range(-half, half+1):
            idx = center_idx + offset
            if idx < 0:
                idx = 0
            elif idx >= n:
                idx = n - 1
            indices.append(idx)
        event_vox_list = []
        blur_list, gt_list = [], []
        for idx in indices:
            left_event_vox = np.load(v_list[idx])["voxel"]
            left_event_vox_tensor = torch.from_numpy(left_event_vox).float()
            left_event_vox_tensor = self.transform_voxel(left_event_vox_tensor)
            blur_image = Image.open(b_list[idx])
            gt_image = Image.open(s_list[idx])
            blur_image_tensor = self.transform(blur_image)
            gt_image_tensor = self.transform(gt_image)
            blur_image_tensor = self.transform_frame(blur_image_tensor)
            gt_image_tensor = self.transform_frame(gt_image_tensor)
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
        return len(self.clips)
