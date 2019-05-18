"""
Given a depth map and RGB image, construct a pointcloud/mesh
This script is specified for KITTI dataset.
"""
import numpy as np
import matplotlib.pyplot as plt
import os, sys
import cv2
from viz_utils import write_ply, depth2cloud

data_dir = '/media/visionlab/My Passport/kitti_raw_data'
project_dir = '/home/visionlab/workspace/GeoSup_internal/'
mode = 'invdepth' # or 'depth', 'invdepth'
prediction_file = 'predictions/Godard_disparities.npy'
frame_id = 247
K0 = np.array([[9.597910e+02, 0.000000e+00, 6.960217e+02],
    [0.000000e+00, 9.569251e+02, 2.241806e+02],
    [0.000000e+00, 0.000000e+00, 1.000000e+00]])

sys.path.insert(0, project_dir)
from common.validator import Validator

match_scale = True
shuffle = False

vmin = 0.01
vmax = 0.2
err_vmin = 0.01
err_vmax = 0.2
EPS = 1e-2
BAR_WIDTH = 20

if __name__ == '__main__':
    val_dir = data_dir   # root directory of kitti raw data
    val_list = os.path.join(project_dir, 'monodepth/filenames/eigen_test_files.txt')

    # initialize validator
    validator = Validator(val_dir=val_dir, val_list=val_list, match_scale=match_scale, use_interp=False)
    # get rgb image
    rgb_file = validator.filenames[frame_id]
    rgb = cv2.imread(rgb_file)
    rgb = rgb[..., ::-1]

    # get depth prediction
    pred = np.squeeze(np.load(prediction_file))
    pred = validator.prepare(pred, mode=mode, verbose=True)[0]
    depth = pred[frame_id]

    # get gt depth
    interp_gt_depths = list(np.load('kitti_test_depths_interp.npy'))
    gt_depth = interp_gt_depths[frame_id]

    height, width = gt_depth.shape
    crop = validator.get_crop(height, width, 'eigen')

    # crop everything
    rgb = rgb[crop[0]:crop[1], crop[2]:crop[3]]
    depth = depth[crop[0]:crop[1], crop[2]:crop[3]]
    # gt_depth = gt_depth[crop[0]:crop[1], crop[2]:crop[3]]

    # adjust intrinsics accordingly
    K = K0.copy()
    K[0, 2] -= crop[2]
    K[1, 2] -= crop[0]


    plt.subplot(211)
    plt.imshow(rgb)
    plt.subplot(212)
    plt.imshow(1.0/(depth+0.001), cmap='plasma')
    plt.show()

    xyzrgb = depth2cloud(depth, rgb[..., ::-1] / 255.0, K, trim_margin=4)
    write_ply(xyzrgb, 'kitti.ply')
