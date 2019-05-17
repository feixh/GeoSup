from __future__ import division
import tensorflow as tf
import numpy as np
import os
import PIL.Image as pil
from geonet_model import GeoNetModel
from geonet_model_sigl import GeoNetModelSIGL
from parallel_dataloader import DataLoader
from termcolor import cprint

def test_depth(opt):
    ##### load testing list #####
    with open('data/kitti/test_files_%s.txt' % opt.depth_test_split, 'r') as f:
        test_files = f.readlines()
        test_files = [opt.dataset_dir + t[:-1] for t in test_files]
    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)

    ##### init #####
    input_uint8 = tf.placeholder(tf.uint8, [opt.batch_size,
                opt.img_height, opt.img_width, 3], name='raw_input')

    model = GeoNetModel(opt, input_uint8, None, None)
    fetches = { "depth": model.pred_depth[0] }

    saver = tf.train.Saver([var for var in tf.model_variables()])
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    ##### Go #####
    with tf.Session(config=config) as sess:
        saver.restore(sess, opt.init_ckpt_file)
        pred_all = []
        for t in range(0, len(test_files), opt.batch_size):
            if t % 100 == 0:
                print('processing: %d/%d' % (t, len(test_files)))
            inputs = np.zeros(
                (opt.batch_size, opt.img_height, opt.img_width, 3),
                dtype=np.uint8)

            for b in range(opt.batch_size):
                idx = t + b
                if idx >= len(test_files):
                    break
                fh = open(test_files[idx], 'r')
                raw_im = pil.open(fh)
                scaled_im = raw_im.resize((opt.img_width, opt.img_height), pil.ANTIALIAS)
                inputs[b] = np.array(scaled_im)

            pred = sess.run(fetches, feed_dict={input_uint8: inputs})
            for b in range(opt.batch_size):
                idx = t + b
                if idx >= len(test_files):
                    break
                pred_all.append(pred['depth'][b,:,:,0])

        np.save(opt.output_dir + '/' + os.path.basename(opt.init_ckpt_file), pred_all)

def test_depth_sigl(opt):
    ##### load testing list #####
    with open('data/kitti/test_files_%s.txt' % opt.depth_test_split, 'r') as f:
        test_files = f.readlines()
        test_files = [opt.dataset_dir + t[:-1] for t in test_files]
    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)

    ##### init #####
    # input_uint8 = tf.placeholder(tf.uint8, [opt.batch_size,
    #            opt.img_height, opt.img_width, 3], name='raw_input')
    # model = GeoNetModel(opt, input_uint8, None, None)

    opt.batch_size = 1
    loader = DataLoader(opt.dataset_dir,
                        1,
                        opt.img_height,
                        opt.img_width,
                        opt.num_source,
                        opt.num_scales)

    tgt_image, _, _, _, _ = loader.next_element

    # Build Model
    op_is_train = tf.placeholder(tf.bool, ())
    model = GeoNetModelSIGL(opt, tgt_image, None, None, None, None, op_is_train)
    print('pred_depth={}'.format(model.pred_depth[0]))

    # fetches = { "depth": model.pred_depth[0] }
    # TODO: next test model.pred_fo_val
    fetches = { "depth": model.pred_for_val }

    saver = tf.train.Saver([var for var in tf.model_variables()])
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    ##### Go #####
    with tf.Session(config=config) as sess:
        saver.restore(sess, opt.init_ckpt_file)
        pred_all = []
        input_all = []
        loader.initialize(sess, img_paths=test_files, is_train=False)
        step = 1
        while True:
            try:
                if step % 100 == 0:
                    print('{}/{}'.format(step, len(test_files)))
                img, pred = sess.run([tgt_image, fetches["depth"]], {op_is_train: False})
                pred_all.append(pred[0, ...])
                step += 1
            except tf.errors.OutOfRangeError:
                pred_all = np.array(pred_all)
                input_all = np.array(input_all)
                break
        basename = os.path.basename(opt.init_ckpt_file)
        np.save(os.path.join(opt.output_dir, basename), pred_all)
