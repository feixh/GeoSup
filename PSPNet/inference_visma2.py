from __future__ import print_function

import argparse
import os
import sys
import time
import tensorflow as tf
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt

from model import PSPNet101, PSPNet50
from tools import *
from my_utils import group_log_prob_map

ADE20k_param = {'crop_size': [473, 473],
                'num_classes': 150,
                'model': PSPNet50}
cityscapes_param = {'crop_size': [720, 720],
                    'num_classes': 19,
                    'model': PSPNet101}

SNAPSHOT_DIR = './model/ade20k_model/pspnet50'

BATCH_SIZE = 4

def get_arguments():
    parser = argparse.ArgumentParser(description="Reproduced PSPNet")
    parser.add_argument("--checkpoints", type=str, default=SNAPSHOT_DIR,
                        help="Path to restore weights.")
    parser.add_argument("--flipped-eval", default=False, action="store_true",
                        help="whether to evaluate with flipped img.")
    parser.add_argument("--dataset", type=str, default='',
                        choices=['ade20k', 'cityscapes'],
                        required=True)
    parser.add_argument("--dataroot", default='/local/Data/VOID',
            help='root directory of data')

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


if __name__ == '__main__':
    tf.reset_default_graph()
    args = get_arguments()

    # load parameters
    if args.dataset == 'ade20k':
        param = ADE20k_param
    elif args.dataset == 'cityscapes':
        param = cityscapes_param

    crop_size = param['crop_size']
    num_classes = param['num_classes']
    PSPNet = param['model']

    # define the loading function within context
    def load_func(img_path):
        img = tf.image.decode_jpeg(tf.read_file(img_path), channels=3)
        _, target, _ = tf.split(img, 3, axis=1)
        return target, img_path

    def augment_func(img, img_path):
        img_shape = tf.shape(img)
        h, w = (tf.maximum(crop_size[0], img_shape[0]), tf.maximum(crop_size[1], img_shape[1]))
        img = tf.squeeze(preprocess(img, h, w), axis=0)
        return img, img_path

    # preprocess images
    entry_placeholder = tf.placeholder(tf.string, shape=[None])
    dataset = tf.data.Dataset.from_tensor_slices(entry_placeholder)
    dataset = dataset.map(load_func, num_parallel_calls=8) \
            .map(augment_func, num_parallel_calls=8) \
            .batch(BATCH_SIZE)

    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    img_op, filename_op = next_element

    # Create network.
    net = PSPNet({'data': img_op}, is_training=False, num_classes=num_classes)
    with tf.variable_scope('', reuse=True):
        flipped_img = tf.image.flip_left_right(tf.squeeze(img_op))
        flipped_img = tf.expand_dims(flipped_img, dim=0)
        net2 = PSPNet({'data': flipped_img}, is_training=False, num_classes=num_classes)

    raw_output = net.layers['conv6']

    # Do flipped eval or not
    if args.flipped_eval:
        flipped_output = tf.image.flip_left_right(tf.squeeze(net2.layers['conv6']))
        flipped_output = tf.expand_dims(flipped_output, dim=0)
        raw_output = tf.add_n([raw_output, flipped_output])

    h, w = 473, 473
    img_shape = (480, 640)
    h, w = max(h, img_shape[0]), max(w, img_shape[1])
    # Predictions.
    raw_output_up = tf.image.resize_bilinear(raw_output, size=[h, w], align_corners=True)
    raw_output_up = tf.image.crop_to_bounding_box(raw_output_up, 0, 0, img_shape[0], img_shape[1])
    raw_output_up = tf.argmax(raw_output_up, axis=3)
    # pred = decode_labels(raw_output_up, img_shape, num_classes)

    # Init tf Session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()

    sess.run(init)
    dataroot = args.dataroot
    entries = []

    # with open(os.path.join(dataroot, 'all.txt'), 'r') as fid:
    #     entries = fid.readlines()

    # COLLECT ALL THE IMAGE TRIPLETS
    for r, d, f in os.walk(dataroot):
        for entry in f:
            if '.jpg' in entry:
                entries.append(os.path.join(r, entry))
    print('Found {} triplets in total'.format(len(entries)))


    entries = [os.path.join(dataroot, x.strip()) for x in entries]
    sess.run(iterator.initializer, feed_dict={entry_placeholder: entries})

    restore_var = tf.global_variables()

    ckpt = tf.train.get_checkpoint_state(args.checkpoints)
    if ckpt and ckpt.model_checkpoint_path:
        loader = tf.train.Saver(var_list=restore_var)
        load_step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
        load(loader, sess, ckpt.model_checkpoint_path)
    else:
        print('No checkpoint file found.')

    total = len(entries)
    counter = 0
    while True:
        try:
            print('{:6d}/{:6d}'.format(counter, total))
            probs, img, filename = sess.run([raw_output_up, img_op, filename_op])
            # make the label map consistent with ade20k_labelmap.csv: start with index 1
            tosave = probs.astype(np.uint16) + 1

            # plt.subplot(121)
            # plt.cla()
            # plt.imshow(img[0, ...])
            # plt.subplot(122)
            # plt.cla()
            # plt.imshow(tosave[0, ...])
            # plt.show()

            for i in range(tosave.shape[0]):
                output_dir = os.path.join(args.dataroot, filename[i].split('/')[-3], 'segmentation')
                if not os.path.exists(output_dir):
                    try:
                        print('making directory ' + output_dir)
                        os.mkdir(output_dir)
                    except OSError:
                        print('failed to create directory {}'.format(output_dir))
                        exit()

                basename = os.path.basename(filename[i][:-4])
                np.save(os.path.join(output_dir, basename), tosave[i, ...])

            counter += tosave.shape[0]
        except tf.errors.OutOfRangeError:
            print('finished')
            break
