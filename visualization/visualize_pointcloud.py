"""
Given a depth map and RGB image, construct a pointcloud/mesh
This script is specified for VOID dataset.
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys, os
from viz_utils import write_ply, depth2cloud

project_dir = '/home/visionlab/workspace/GeoSup_internal'
validation_dir = '/local/Data/VOID'
validation_list = os.path.join(project_dir, 'GeoNet/data/void/test.txt')
prediction_file = 'predictions/VOID_GeoNet_depth.npy'
frame_id = 136

K480x640 = np.array([[514.638, 0, 315.267],
    [0, 518.858, 247.358],
    [0, 0, 1]], dtype=np.float32)
W0, H0 = 640, 480


if __name__ == '__main__':
    depth = np.load(prediction_file)[frame_id]
    with open(validation_list, 'r') as fid:
        rgb_file = os.path.join(validation_dir, fid.readlines()[frame_id].strip())
    _, rgb, _ = np.split(cv2.imread(rgb_file), 3, axis=1)
    rows, cols = depth.shape[:2]
    assert rgb.shape[0] == H0 and rgb.shape[1] == W0
    if rgb.shape[0] != rows or rgb.shape[1] != cols:
        # resize rgb image to match the size of the depth image
        rgb = cv2.resize(rgb, (cols, rows))
        # also scale the intrinsics accordingly
        K0 = K480x640
        K = np.stack((K0[0, :] * cols / W0,
            K0[1, :] * rows / H0,
            np.array([0.0, 0.0, 1.0])))

    # minz = np.min(depth)
    # maxz = np.max(depth)
    # rgb = rgb * (depth[..., np.newaxis] - 1.1 * minz) / (0.9 * maxz - 1.1 * minz)
    # rgb = rgb.astype(np.uint8)

    plt.subplot(121)
    plt.imshow(rgb)
    plt.subplot(122)
    plt.imshow(depth, cmap='jet')
    plt.show()

    xyzrgb = depth2cloud(depth, rgb[..., ::-1] / 255.0, K, trim_margin=4)
    write_ply(xyzrgb, 'void.ply')

