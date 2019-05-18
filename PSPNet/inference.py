from __future__ import print_function

import argparse
import os
import sys
import time
import tensorflow as tf
import numpy as np
from scipy import misc
import glob

from model import PSPNet101, PSPNet50
from tools import *
import matplotlib.pyplot as plt
from my_utils import grab_all_entries, group_log_prob_map

ADE20k_param = {'crop_size': [473, 473],
                'num_classes': 150,
                'model': PSPNet50}
cityscapes_param = {'crop_size': [720, 720],
                    'num_classes': 19,
                    'model': PSPNet101}

kitti_param = {'crop_size': [720, 720],
        'num_classes': 19,
        'model': PSPNet101,
        'img_shape': [512, 1024]}

SAVE_DIR = './output/'
SNAPSHOT_DIR = './model/cityscapes_model/pspnet101'

def get_arguments():
    parser = argparse.ArgumentParser(description="Reproduced PSPNet")
    parser.add_argument("--img-path", type=str, default='',
                        help="Path to the RGB image file.")
    parser.add_argument("--checkpoints", type=str, default=SNAPSHOT_DIR,
                        help="Path to restore weights.")
    parser.add_argument("--save-dir", type=str, default=SAVE_DIR,
                        help="Path to save output.")
    parser.add_argument("--flipped-eval", action="store_true",
                        help="whether to evaluate with flipped img.")
    parser.add_argument("--dataset", type=str, default='',
                        choices=['ade20k', 'cityscapes', 'kitti'],
                        required=True)
    parser.add_argument("--dataroot", type=str,
            default='/local/Data/NIPS18/kitti_raw_data')
    parser.add_argument("--batch-size", type=int, default=1)

    return parser.parse_args()

def save(saver, sess, logdir, step):
   model_name = 'model.ckpt'
   checkpoint_path = os.path.join(logdir, model_name)

   if not os.path.exists(logdir):
      os.makedirs(logdir)
   saver.save(sess, checkpoint_path, global_step=step)
   print('The checkpoint has been created.')

def load(saver, sess, ckpt_path):
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))



def load_func(path):
    """ Load the image given its path.
    Args:
        path: absolute path of the image file.
    Returns:
        image as a rank-3 tensor
    """
    return tf.squeeze(
        tf.image.resize_bilinear(
            tf.expand_dims(tf.image.decode_png(tf.read_file(path)), dim=0),
            kitti_param['img_shape']), 0)


def augment_func(img):
    # h = 720
    # w = 720
    # tools.preprocess with some modification
    # Convert RGB to BGR
    crop_size = kitti_param['crop_size']
    img_shape = kitti_param['img_shape']
    h, w = (tf.maximum(crop_size[0], img_shape[0]), tf.maximum(crop_size[1], img_shape[1]))
    # h = img.shape[0]
    # w = img.shape[1]
    img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
    img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)
    # Extract mean.
    img -= IMG_MEAN

    pad_img = tf.image.pad_to_bounding_box(img, 0, 0, h, w)
    # pad_img = tf.expand_dims(pad_img, dim=0)
    return pad_img

if __name__ == '__main__':
    tf.reset_default_graph()
    args = get_arguments()

    entries = grab_all_entries(args.dataroot)
    # entries = tf.expand_dims(tf.convert_to_tensor(entries), dim=-1)

    entry_placeholder = tf.placeholder(tf.string, shape=[None])
    dataset = tf.data.Dataset.from_tensor_slices(entry_placeholder)
    # we can do shuffle to entry list and then pass them via placeholder
    dataset = dataset.map(load_func, num_parallel_calls=8) \
            .map(augment_func, num_parallel_calls=8) \
            .batch(args.batch_size)

    # use one-shot iterator, since
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    img = next_element

    # load parameters
    # if args.dataset == 'ade20k':
    #     param = ADE20k_param
    # elif args.dataset == 'cityscapes':
    #     param = cityscapes_param
    # elif args.dataset == 'kitti':
    #     param = kitti_param
    param = kitti_param

    img_shape = param['img_shape']
    crop_size = param['crop_size']
    num_classes = param['num_classes']
    PSPNet = param['model']

    # # preprocess images
    # # img, filename = load_img(args.img_path)
    h, w = tf.maximum(crop_size[0], img_shape[0]), tf.maximum(crop_size[1], img_shape[1])
    # img = preprocess(img, h, w)

    # Create network.
    net = PSPNet({'data': img}, is_training=False, num_classes=num_classes)
    with tf.variable_scope('', reuse=True):
        flipped_img = tf.image.flip_left_right(tf.squeeze(img))
        flipped_img = tf.expand_dims(flipped_img, dim=0)
        net2 = PSPNet({'data': flipped_img}, is_training=False, num_classes=num_classes)

    raw_output = net.layers['conv6']

    # Do flipped eval or not
    if args.flipped_eval:
        flipped_output = tf.image.flip_left_right(tf.squeeze(net2.layers['conv6']))
        flipped_output = tf.expand_dims(flipped_output, dim=0)
        raw_output = tf.add_n([raw_output, flipped_output])

    # Predictions.
    raw_output_up = tf.image.resize_bilinear(raw_output, size=[h, w], align_corners=True)
    raw_output_up = tf.image.crop_to_bounding_box(
            raw_output_up, 0, 0, img_shape[0], img_shape[1])
    raw_output_up = tf.image.resize_bilinear(raw_output_up, size=[256, 512], align_corners=False)
    # we want the raw output which is the probability distribution over different
    # categories
    # raw_output_up = tf.argmax(raw_output_up, axis=3)
    # pred = decode_labels(raw_output_up, img_shape, num_classes)
    # FIXME: resize to 256 x 512

    # Init tf Session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()

    sess.run(init)
    sess.run(iterator.initializer, feed_dict={entry_placeholder: entries})

    restore_var = tf.global_variables()

    ckpt = tf.train.get_checkpoint_state(args.checkpoints)
    if ckpt and ckpt.model_checkpoint_path:
        loader = tf.train.Saver(var_list=restore_var)
        load_step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
        load(loader, sess, ckpt.model_checkpoint_path)
    else:
        print('No checkpoint file found.')

    def run_and_save(imgpath):
        sess.run(iterator.initializer, feed_dict={entry_placeholder: [imgpath]})
        prob = sess.run(raw_output_up)
        lp = group_log_prob_map(prob[0, ...])
        np.save('logprob.npy', lp)
        plt.imshow(lp.argmax(axis=2))
        plt.show()

    raise Exception

    i = 0
    while True:
        print('{:6d}/{:6d}'.format(i, len(entries)))
        prob = sess.run(raw_output_up)
        curpath = entries[i]
        segpath = '/'.join([x if x != 'kitti_raw_data' else 'kitti_raw_data_seg' for x in curpath.split('/')])
        segdir = os.path.dirname(segpath)
        if not os.path.exists(segdir):
            os.makedirs(segdir)
        np.save(segpath + '.npy', prob[0, ...].astype(np.float16))
        i += 1


    # prob = sess.run(raw_output_up)
    # preds = sess.run(pred)

    # if not os.path.exists(args.save_dir):
    #     os.makedirs(args.save_dir)
    # misc.imsave(args.save_dir + 'kitti_test.png', preds[0])
