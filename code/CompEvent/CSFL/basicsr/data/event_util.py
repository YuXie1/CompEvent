import numpy as np
import torch

def events_to_voxel_grid(events, num_bins, width, height, return_format='CHW'):

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
    xs = events[:, 1].astype(int)
    ys = events[:, 2].astype(int)
    pols = events[:, 3]
    pols[pols == 0] = -1

    tis = ts.astype(int)
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

    if return_format == 'CHW':
        return voxel_grid
    elif return_format == 'HWC':
        return voxel_grid.transpose(1,2,0)

def voxel_norm(voxel):
    nonzero_ev = (voxel != 0)
    num_nonzeros = nonzero_ev.sum()

    if num_nonzeros > 0:

        mean = voxel.sum() / num_nonzeros
        stddev = torch.sqrt((voxel ** 2).sum() / num_nonzeros - mean ** 2)
        mask = nonzero_ev.float()
        voxel = mask * (voxel - mean) / stddev

    return voxel

def filter_event(x,y,p,t, s_e_index=[0,6]):
    t_1=t.squeeze(1)
    uniqw, inverse = np.unique(t_1, return_inverse=True)
    discretized_ts = np.bincount(inverse)
    index_exposure_start = np.sum(discretized_ts[0:s_e_index[0]])
    index_exposure_end = np.sum(discretized_ts[0:s_e_index[1]+1])
    x_1 = x[index_exposure_start:index_exposure_end]
    y_1 = y[index_exposure_start:index_exposure_end]
    p_1 = p[index_exposure_start:index_exposure_end]
    t_1 = t[index_exposure_start:index_exposure_end]
    
    return x_1, y_1, p_1, t_1

