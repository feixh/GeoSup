""" Visualize input image, ground truth depth, prediction from baseline and prediction from our model.
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys, os


project_dir = '/local2/workspace/GeoSup'
sys.path.insert(0, project_dir)
from GeoNet.visma_validator import Validator

validation_dir = '/local2/Data/VOID'
validation_list = os.path.join(project_dir, 'GeoNet/data/void/test.txt')
EPS = 0.0001
BAR_WIDTH = 20
vmin = 0.7
vmax = 1.2
err_vmin = 0.02
err_vmax = 0.7
# save_list = [14, 26, 27, 51, 75, 98, 136, 164, 467, 496]
# save_list = [136]   # the one used in slides

save_list = []

def configure_plt():
    plt.box(False)
    plt.axis('off')

def draw_colorbar():
    """ Draw the colorbar for range maps.
    """
    print('draw colorbar')
    depth_bar = np.tile(np.linspace(vmin, vmax, 100), (BAR_WIDTH, 1))
    depth_bar = np.flipud(depth_bar.T)
    plt.imshow(depth_bar, cmap='jet')
    plt.box(False)
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    if sys.argv[1] == 'colorbar':
        draw_colorbar()
        exit(0)

    preds = [np.load(filename) for filename in sys.argv[1:]]
    tags = ['GeoNet', 'GeoNet+SIGL']
    num_models = len(preds)
    if num_models == 0:
        raise ValueError('no model to visualize')
    validator = Validator(val_dir=validation_dir,
            val_list=validation_list,
            match_scale=True,
            which_camera='realsense')

    for i, filename in enumerate(validator.filenames):
        if len(save_list) > 0 and i not in save_list: continue
        mask = validator.masks[i]
        depth = validator.gt[i]
        median_depth = np.median(depth[mask])
        image = plt.imread(filename)
        height, width = image.shape[0], image.shape[1] // 3
        image = image[:, width:width*2, :]
        image = image[:,:,::-1]
        if len(save_list) > 0:
            plt.imsave('{:04}_image.jpg'.format(i), image)

        # total_plots = 2*num_models + 2
        plt.subplot(2, num_models+1, 1)
        plt.imshow(image)
        configure_plt()

        plt.subplot(2, num_models+1, num_models+2)
        plt.imshow(1.0/(depth+EPS), cmap='jet', vmin=vmin, vmax=vmax)
        configure_plt()
        if len(save_list) > 0:
            plt.imsave('{:04}_gt.jpg'.format(i), 1.0/(depth+EPS), cmap='jet', vmin=vmin, vmax=vmax)

        for j in range(num_models):
            pred = preds[j][i, ...]
            pred = cv2.resize(pred, (width, height))
            # median matching
            pred *= median_depth / np.median(pred[mask])
            plt.subplot(2, num_models+1, 2+j)
            inv_depth = 1.0 / (pred+EPS)
            plt.imshow(inv_depth, cmap='jet', vmin=vmin, vmax=vmax)
            configure_plt()
            if len(save_list) > 0:
                plt.imsave('{:04}_pred_{}.jpg'.format(i, tags[j]), inv_depth, cmap='jet', vmin=vmin, vmax=vmax)

            plt.subplot(2, num_models+1, num_models + 3 + j)
            absrel = np.abs(depth - pred) / (depth + EPS)
            plt.imshow(absrel, cmap='hot', vmin=err_vmin, vmax=err_vmax)
            configure_plt()
            if len(save_list) > 0:
                plt.imsave('{:04}_absrel_{}.jpg'.format(i, tags[j]), absrel, cmap='hot', vmin=err_vmin, vmax=err_vmax)

        print(i)
        plt.subplots_adjust(hspace=0.05)
        plt.show()

