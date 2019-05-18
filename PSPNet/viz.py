import os
import glob

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tools import decode_labels
import cv2

from inference import grab_all_entries

# def grab_all_entries(dataroot, filetype='.png'):
#     """ Grab absolute path of all images.
#     Args:
#         dataroot: Root directory of raw kitti dataset.
#     Returns:
#         a list of absolute paths
#     """
#
#     def expand_date_folder(date_folder_path):
#         entries = []
#         for seq_folder in os.listdir(date_folder_path):
#             seq_folder_path = os.path.join(date_folder_path, seq_folder)
#             if os.path.isdir(seq_folder_path):
#                 entries += glob.glob(
#                     os.path.join(seq_folder_path, 'image_02', 'data',
#                                  '*' + filetype))
#                 entries += glob.glob(
#                     os.path.join(seq_folder_path, 'image_03', 'data',
#                                  '*' + filetype))
#         return entries
#
#     entries = []
#     for date_folder in os.listdir(dataroot):
#         date_folder_path = os.path.join(dataroot, date_folder)
#         if os.path.isdir(date_folder_path):
#             entries += expand_date_folder(date_folder_path)
#     entries = np.array(entries)
#     entries.sort()
#     return entries


def load_func(imgpath, segpath):
    """ Load image and segmentation mask given paths.
    Args:
        imgpath, segpath: string
    Returns:
        randk-3 image and segmentation mask tensors of type float32
    """

    def _py_load_func(segpath):
        return np.load(segpath).astype(np.float32)

    # need this tuple somehow
    segmask = tuple(tf.py_func(_py_load_func, [segpath], [tf.float32]))
    image = tf.image.decode_png(tf.read_file(imgpath))
    image = tf.image.resize_bilinear(
        tf.expand_dims(image, dim=0), size=[256, 512])
    image = tf.squeeze(image, axis=0)
    return image, segmask[0]


if __name__ == '__main__':
    # load data
    img_list = grab_all_entries('/local/Data/NIPS18/kitti_raw_data', '.png')
    img_list.sort()
    seg_list = grab_all_entries('/local/Data/NIPS18/kitti_raw_data_seg',
                                '_compact.npy')
    seg_list.sort()
    assert img_list.size == seg_list.size

    # create dataset loader
    dataloader = tf.data.Dataset.from_tensor_slices((img_list, seg_list))
    dataloader = dataloader.map(load_func)
    iterator = dataloader.make_initializable_iterator()
    next_element = iterator.get_next()

    sess = tf.Session()
    sess.run(iterator.initializer)
    label_map = decode_labels(
        tf.argmax(next_element[1], axis=2),
        img_shape=[256, 512],
        num_classes=19)

    while True:
        try:
            (image, prob_mask), mask = sess.run([next_element, label_map])
            plt.clf()
            plt.subplot(131)
            plt.imshow(image.astype(np.uint8))
            plt.subplot(132)
            plt.imshow(mask[0, ...].astype(np.uint8))
            plt.subplot(133)
            blend = cv2.addWeighted(image, 0.7, mask[0, ...], 0.3, 0)
            plt.imshow(blend.astype(np.uint8))
            plt.show()
        except tf.errors.OutOfRangeError:
            break

    # dataloader = tf.data.Dataset.from_tensor_slices(seg_list)
    # dataloader = dataloader.map(lambda segpath:
    #         tf.py_func(load_func1, [segpath], [tf.float32]))
    # iterator = dataloader.make_initializable_iterator()
    # next_element = iterator.get_next()
    # sess = tf.Session()
    # sess.run(iterator.initializer)
    # mask = sess.run(next_element)
