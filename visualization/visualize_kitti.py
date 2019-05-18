""" Head-to-head comparison of monocular depth prediction trained on KITTI.
"""
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

data_dir = '/media/visionlab/My Passport/kitti_raw_data'
project_dir = '/home/visionlab/workspace/GeoSup_internal/'
mode = 'depth' # or 'depth', 'invdepth'

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

def configure_plt():
    plt.box(False)
    plt.axis('off')

if __name__ == '__main__':
    val_dir = data_dir   # root directory of kitti raw data
    val_list = os.path.join(project_dir, 'monodepth/filenames/eigen_test_files.txt')

    # prepare

    if sys.argv[1] == 'interp':
      validator = Validator(val_dir=val_dir, val_list=val_list, match_scale=match_scale, use_interp=True)
      # test
      np.save('kitti_test_depths_interp.npy', validator.interp_gt_depths)
      exit(0)
    elif sys.argv[1] == 'bar':
      err_bar = np.tile(np.linspace(err_vmin, err_vmax, 100), (BAR_WIDTH, 1))
      err_bar = np.flipud(err_bar.T)
      invdepth_bar = np.tile(np.linspace(vmin, vmax, 100), (BAR_WIDTH, 1))
      invdepth_bar = np.flipud(invdepth_bar.T)
      plt.figure(0)
      plt.subplot(211)
      plt.imshow(invdepth_bar, cmap='plasma')
      configure_plt()

      plt.subplot(212)
      plt.imshow(err_bar, cmap='hot')
      configure_plt()

      plt.show()
      exit(0)
    else:
      validator = Validator(val_dir=val_dir, val_list=val_list, match_scale=match_scale, use_interp=False)


    preds = [np.squeeze(np.load(filename)) for filename in sys.argv[1::2]]
    tags = [tag.strip() for tag in sys.argv[2::2]]
    preds = [validator.prepare(pred, mode=mode, verbose=True)[0] for pred in preds]
    num_models = len(preds)

    interp_gt_depths = list(np.load('kitti_test_depths_interp.npy'))

    indices = range(len(interp_gt_depths))
    if shuffle: np.random.shuffle(indices)

    for index in indices:
        print(index)
        filename = validator.filenames[index]
        gt_depth = interp_gt_depths[index]

        height, width = gt_depth.shape
        crop = validator.get_crop(height, width, 'eigen')
        image = plt.imread(filename)
        image = image[crop[0]:crop[1], crop[2]:crop[3]]
        gt_depth = gt_depth[crop[0]:crop[1], crop[2]:crop[3]]

        plt.subplot(2*num_models+2, 1, 1)
        plt.imshow(image)
        configure_plt()

        plt.subplot(2*num_models+2, 1, 2)
        plt.imshow(1./(gt_depth+EPS), cmap='plasma', vmin=vmin, vmax=vmax)
        configure_plt()

        for j in range(num_models):
            pred = preds[j][index][crop[0]:crop[1], crop[2]:crop[3]]
            plt.subplot(2*num_models+2, 1, 3+j)
            plt.imshow(1./(pred+EPS), cmap='plasma', vmin=vmin, vmax=vmax)
            configure_plt()

            plt.subplot(2*num_models+2, 1, 3+j+num_models)
            err_map = np.abs(pred-gt_depth)/(gt_depth+EPS)
            plt.imshow(err_map, cmap='hot', vmin=err_vmin, vmax=err_vmax)
            configure_plt()


        plt.subplots_adjust(hspace=0.05)
        plt.show()
