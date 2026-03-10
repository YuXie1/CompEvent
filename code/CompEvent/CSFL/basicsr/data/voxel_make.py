import os
import numpy as np
import scipy.io as sio
import yaml

with open('/code/CompEvent/CSFL/options/train/LOL_Blur/CompEvent-LOL_Blur.yml', 'r') as f:
    config = yaml.safe_load(f)
bins = config['datasets']['train']['bins']
print('bins:', bins)

def binary_events_to_voxel_grid(events, num_bins, width, height):
    assert(events.shape[1] == 4)
    assert(num_bins > 0)
    assert(width > 0)
    assert(height > 0)
    voxel_grid = np.zeros((num_bins, height, width), np.float32).ravel()
    last_stamp = events[-1, 0]
    first_stamp = events[0, 0]
    deltaT = last_stamp - first_stamp
    if deltaT == 0:
        deltaT = 1.0
    events[:, 0] = (num_bins - 1) * (events[:, 0] - first_stamp) / deltaT
    ts = events[:, 0]
    xs = events[:, 1].astype(np.int32)
    ys = events[:, 2].astype(np.int32)
    pols = events[:, 3]
    pols[pols == 0] = -1
    tis = ts.astype(np.int32)
    dts = ts - tis
    vals_left = pols * (1.0 - dts)
    vals_right = pols * dts
    valid_indices = tis < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width
              + tis[valid_indices] * width * height, vals_left[valid_indices])
    valid_indices = (tis + 1) < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width
              + (tis[valid_indices] + 1) * width * height, vals_right[valid_indices])
    voxel_grid = np.reshape(voxel_grid, (num_bins, height, width))
    return voxel_grid

mat_dir = '/data/Clivia/LOL_Blur/test/mat'
save_dir = '/data/Clivia/LOL_Blur/test/event_voxel_16'
os.makedirs(save_dir, exist_ok=True)
H, W = 240, 320

mat_files = [f for f in os.listdir(mat_dir) if f.endswith('.mat')]
for fname in mat_files:
    fpath = os.path.join(mat_dir, fname)
    mat = sio.loadmat(fpath)
    x = mat['section_event_x'].flatten()
    y = mat['section_event_y'].flatten()
    t = mat['section_event_timestamp'].flatten()
    p = mat['section_event_polarity'].flatten()
    events = np.stack([t, x, y, p], axis=1)
    if len(events) == 0:
        voxel = np.zeros((bins, H, W), dtype=np.float32)
    else:
        voxel = binary_events_to_voxel_grid(events, bins, W, H)

    np.savez_compressed(
        os.path.join(save_dir, fname.replace('.mat', '.npz')),
        voxel=voxel
    )
    print(f'Saved: {fname} -> {fname.replace(".mat", ".npz")}')