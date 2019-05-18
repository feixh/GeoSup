# Visualization utilities.
# Author: Xiaohan Fei
import numpy as np

def write_ply(X, filename):
  """ Save an colored pointcloud to a .PLY file.
  Args
      X: Nx3 numpy array as a colored pointcloud in XYZRGB format.
      filename: the .PLY filename
  """
  with open(filename, 'w') as fid:
    fid.write('''ply
format ascii 1.0
comment written by xfei
element vertex {}
property float32 x
property float32 y
property float32 z
property uchar red
property uchar green
property uchar blue
end_header\n'''.format(X.shape[0]))
    for pt in X:
      fid.write('{} {} {} {} {} {}\n'.format(pt[0], pt[1], pt[2],
	int(255*pt[5]), int(255*pt[4]), int(255*pt[3])))


def depth2cloud(depth, rgb, K, trim_margin=20):
  """ Convert a depth frame and an rgb frame to a colored pointcloud.
  Args
      depth: (1)HW(1) depth image
      rgb: (1)HW3 3-channel color image.
      K: (1)3x3 camera intrinsics.
  Returns: Nx6 colored pointcloud in XYZRGB form
  """
  rgb, depth, K = [np.squeeze(item) for item in (rgb, depth, K)]
  rows, cols = rgb.shape[:2]
  Y, X = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
  X = np.stack([X, Y, np.ones(X.shape)], axis=-1)
  X = depth[..., np.newaxis] * X
  if trim_margin > 0:
      X = X[trim_margin:-trim_margin, trim_margin:-trim_margin, :]
  X = X.reshape([-1, 3])
  X = X.dot(np.linalg.inv(K).T)   # Nx3
  if trim_margin > 0:
      rgb = rgb[trim_margin:-trim_margin, trim_margin:-trim_margin, :]
  color = rgb.reshape([-1, 3])
  X = np.concatenate([X, color], axis=-1)
  return X
