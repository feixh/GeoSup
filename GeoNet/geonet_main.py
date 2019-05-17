from __future__ import division

import os
import glob
import time
import random
import pprint
import numpy as np
from copy import deepcopy
import tensorflow as tf
import tensorflow.contrib.slim as slim
from termcolor import cprint, colored

# import test functions
from geonet_test_depth import test_depth, test_depth_sigl
from geonet_test_pose import test_pose
from geonet_test_flow import test_flow

# dataloader and validator for kitti dataset
import sys
sys.path.insert(0, '..')
from validator import Validator
from parallel_dataloader import DataLoader
# for our own data
import visma_dataloader, visma_validator
# modified geonet models
import geonet_model_visma, geonet_model_sigl

flags = tf.app.flags
flags.DEFINE_string("mode",                         "",    "(train_rigid, train_flow) or (test_depth, test_pose, test_flow)")
flags.DEFINE_string("dataset_dir",                  "/home/feixh/Data/kitti_raw_formatted/",    "Dataset directory")
flags.DEFINE_string("init_ckpt_file",             None,    "Specific checkpoint file to initialize from")
flags.DEFINE_integer("batch_size",                   4,    "The size of of a sample batch")
flags.DEFINE_integer("num_threads",                 32,    "Number of threads for data loading")
flags.DEFINE_integer("img_height",                 128,    "Image height")
flags.DEFINE_integer("img_width",                  416,    "Image width")
flags.DEFINE_integer("seq_length",                   3,    "Sequence length for each example")

#adding the following flags due to API change from TF 1.1 to TF 1.7
flags.DEFINE_integer("num_source",                   2, "Number of source images")
flags.DEFINE_integer("num_scales",                   4, "Number of scale levels")
flags.DEFINE_bool("add_dispnet",                  False, "add dispnet or not")
flags.DEFINE_bool("add_flownet",                  False, "add flownet or not")
flags.DEFINE_bool("add_posenet",                  False, "add posenet or not")

##### Training Configurations #####
flags.DEFINE_string("checkpoint_dir",               "",    "Directory name to save the checkpoints")
flags.DEFINE_float("learning_rate",             0.0002,    "Learning rate for adam")
flags.DEFINE_integer("max_to_keep",                 20,    "Maximum number of checkpoints to save")
flags.DEFINE_integer("max_steps",               300000,    "Maximum number of training iterations")
flags.DEFINE_integer("save_ckpt_freq",            2500,    "Save the checkpoint model every save_ckpt_freq iterations")
flags.DEFINE_float("alpha_recon_image",           0.85,    "Alpha weight between SSIM and L1 in reconstruction loss")
flags.DEFINE_bool("restore_global_step",          True,    "Restore global step from checkpoint")

##### Configurations about DepthNet & PoseNet of GeoNet #####
flags.DEFINE_string("dispnet_encoder",      "resnet50",    "Type of encoder for dispnet, vgg or resnet50")
flags.DEFINE_boolean("scale_normalize",           True,    "Spatially normalize depth prediction")
flags.DEFINE_float("rigid_warp_weight",            1.0,    "Weight for warping by rigid flow")
flags.DEFINE_float("disp_smooth_weight",           0.5,    "Weight for disp smoothness")

##### Configurations about ResFlowNet of GeoNet (or DirFlowNetS) #####
flags.DEFINE_string("flownet_type",         "residual",    "type of flownet, residual or direct")
flags.DEFINE_float("flow_warp_weight",             1.0,    "Weight for warping by full flow")
flags.DEFINE_float("flow_smooth_weight",           0.2,    "Weight for flow smoothness")
flags.DEFINE_float("flow_consistency_weight",      0.2,    "Weight for bidirectional flow consistency")
flags.DEFINE_float("flow_consistency_alpha",       3.0,    "Alpha for flow consistency check")
flags.DEFINE_float("flow_consistency_beta",       0.05,    "Beta for flow consistency check")

##### Testing Configurations #####
flags.DEFINE_string("output_dir",                 None,    "Test result output directory")
flags.DEFINE_string("depth_test_split",        "eigen",    "KITTI depth split, eigen or stereo")
flags.DEFINE_integer("pose_test_seq",                9,    "KITTI Odometry Sequence ID to test")

##### sigl related arguments #####
flags.DEFINE_integer("sigl_scales", 1, "scale levels of sigl loss")
flags.DEFINE_float("sigl_loss_weight", 0.01, "weigth of sigl loss")
flags.DEFINE_boolean("use_sigl", False, "use sigl if true")
flags.DEFINE_integer("win_size", 8, "window size used in sigl loss")
flags.DEFINE_integer("stride", 4, "stride used in sigl loss")

##### on-the-fly validation #####
# flags.DEFINE_string("kitti_raw_dir", "/local/Data/NIPS18/kitti_raw_data/", "KITTI raw dataset for validation")
# use the following subset for testing
flags.DEFINE_string("validation_dir", "/home/feixh/Data/kitti_raw_test/", "KITTI raw dataset for validation")
flags.DEFINE_integer("validation_freq", 5000, "Frequency to run validation")
flags.DEFINE_integer("summary_freq", 100, "Frequency to generate summary")
flags.DEFINE_float("min_depth", 1e-3, "Threshold for minimum depth")
flags.DEFINE_float("max_depth", 80, "Threshold for maximum depth")
flags.DEFINE_float("best_val", 100, "Best AbsRel so far")
flags.DEFINE_integer("best_step", -1, "The step where best AbsRel is obtained")

flags.DEFINE_string("datatype", default="kitti", help="visma, void or kitti")
flags.DEFINE_boolean("use_slam_pose", False, "if set, use pose from slam")

opt = flags.FLAGS

if opt.datatype in ['visma', 'void']:
    GeoNetModelSIGL = geonet_model_visma.GeoNetModelSIGL
    DataLoader = visma_dataloader.DataLoader
    Validator = visma_validator.Validator
    if opt.datatype == 'visma':
        cprint('using VISMA setting and model', 'green')
        opt.img_height, opt.img_width = 256, 512
    else:
        cprint('using VOID setting and model', 'green')
        opt.img_height, opt.img_width = 240, 320
elif opt.datatype == 'kitti':
    cprint('using KITTI setting and model', 'green')
    GeoNetModelSIGL = geonet_model_sigl.GeoNetModelSIGL
else: raise ValueError('datatype == [kitti|visma|void]')


def save_prediction(preds, step, ckpt_dir, max_pred_to_keep=20):
    """
    Save predictions.
    Args:
        preds: NxHxWx1 depth prediction
        step: current interation number in training
        ckpt_dir: checkpoint directory to save the predcition
        max_pred_to_keep: maximum number o predictions to keep
    """
    if max_pred_to_keep <= 0: return

    pred_dir = os.path.join(ckpt_dir, 'prediction')
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)

    to_save = os.path.join(pred_dir, 'depth_prediction_{}.npy'.format(step))
    np.save(to_save, preds)
    # print('prediction {} saved'.format(to_save))

    all_npy = glob.glob(os.path.join(pred_dir, '*.npy'))
    mtime = [os.path.getmtime(x) for x in all_npy]
    all_npy = np.array(all_npy)[np.argsort(mtime)]
    if len(all_npy) > max_pred_to_keep:
        for to_remove in all_npy[:len(all_npy) - max_pred_to_keep]:
            # print('removing {}'.format(to_remove))
            os.remove(to_remove)

    # ensure the latest one is NOT removed
    assert(os.path.exists(to_save))


def validation_loop(sess, is_train_op, pred_op, step, dataloader, validator):
    """ The validation loop takes the model and validation dataloader,
    predict inverse depth and compute error & accuracy metrics.
    Args:
        sess: active session
        is_train_op: Op of training flag
        pred_op: Depth prediction Op.
        step: global step
        dataloader: validation set loader
        validator: take depth prediction and compute error & accuracy metrics
    Returns:
        a boolean: whehter or not current model is better than the previous best,
        an NxHxWx1 numpy array: inverse depth prediction
    """
    preds = []

    assert pred_op.shape.as_list() == [opt.batch_size, opt.img_height, opt.img_width]
    # pad test file list
    padded_filenames = deepcopy(validator.filenames)
    n_files = len(padded_filenames)
    if n_files % opt.batch_size is not 0:
        padded_filenames += [padded_filenames[-1]] * (opt.batch_size - n_files % opt.batch_size)
    dataloader.initialize(sess, img_paths=padded_filenames, is_train=False)

    # print('generate prediction ...')
    while True:
        try:
            pred_depth = sess.run(pred_op, {is_train_op: False})
            preds = preds + list(pred_depth)
        except tf.errors.OutOfRangeError:
            preds = np.array(preds[0:n_files])
            break
    is_better = validator.validate(preds, step=step,
        max_depth=80 if opt.datatype == 'kitti' else 5,
        mode='depth')
    return is_better, preds

def train():

    seed = 8964
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # pp = pprint.PrettyPrinter()
    # pp.pprint(flags.FLAGS.__flags)

    if not os.path.exists(opt.checkpoint_dir):
        os.makedirs(opt.checkpoint_dir)

    with tf.Graph().as_default():
        # flag to indicate training/validation phase
        is_train_op = tf.placeholder(tf.bool, ())
        # setup dataloader, model, and validator
        if opt.datatype == 'kitti':
            loader = DataLoader(opt.dataset_dir,
                                opt.batch_size,
                                opt.img_height,
                                opt.img_width,
                                opt.num_source,
                                opt.num_scales)
            tgt_image, src_image_stack, intrinsics, mask, Rcs = loader.next_element
            model = GeoNetModelSIGL(opt, tgt_image, src_image_stack,
                    intrinsics,
                    mask, Rcs, is_train_op)
            validator = Validator(val_dir=opt.validation_dir,
                    val_list='data/kitti/test_files_eigen.txt',
                    match_scale=True,
                    log_path=os.path.join(opt.checkpoint_dir, 'results.txt'))
        else:
            if opt.datatype == 'visma':
                assert (opt.img_height, opt.img_width) == (256, 512), 'image size should be 256x512 for VISMA model'
                validator = Validator(val_dir=opt.validation_dir,
                              val_list='data/visma/test.txt',
                              match_scale=True,
                              log_path=os.path.join(opt.checkpoint_dir, 'results.txt'),
                              which_camera='imx')
            elif opt.datatype == 'void':
                assert (opt.img_height, opt.img_width) == (240, 320), 'image size should be 240x320 for VOID model'
                validator = Validator(val_dir=opt.validation_dir,
                              val_list='data/void/test.txt',
                              match_scale=True,
                              log_path=os.path.join(opt.checkpoint_dir, 'results.txt'),
                              which_camera='realsense')

            loader = DataLoader(opt.dataset_dir,
                                opt.batch_size,
                                opt.img_height,
                                opt.img_width,
                                opt.num_source,
                                opt.num_scales,
                                which_camera='realsense' if opt.datatype == 'void' else 'imx')

            tgt_image, src_image_stack, tgt2src_pose_stack, intrinsics, mask, Rcs = loader.next_element
            model = GeoNetModelSIGL(opt, tgt_image, src_image_stack,
                    tgt2src_pose_stack, intrinsics,
                    mask, Rcs, is_train_op)

        # Train Op
        if opt.mode == 'train_flow' and opt.flownet_type == "residual":
            # we pretrain DepthNet & PoseNet, then finetune ResFlowNetS
            train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "flow_net")
            vars_to_restore = slim.get_variables_to_restore(include=["depth_net", "pose_net"])
        else:
            train_vars = [var for var in tf.trainable_variables()]
            vars_to_restore = slim.get_model_variables()


        loss = model.total_loss
        optim = tf.train.AdamOptimizer(opt.learning_rate) # , 0.9)
        train_op = slim.learning.create_train_op(loss, optim,
                variables_to_train=train_vars)

        # grad_op = optim.compute_gradients(loss, var_list=train_vars)
        # grad_op = [(tf.clip_by_value(grad, -MAX_GRAD, MAX_GRAD), var) for grad, var in grad_op]
        # train_op = optim.apply_gradients(grad_op)

        # Global Step
        global_step = tf.Variable(0, name='global_step', trainable=False)
        incr_global_step = tf.assign(global_step, global_step+1)
        # Parameter Count
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) \
                                        for v in train_vars])

        # Saver
        vars_to_restore = [var for var in tf.model_variables()]
        if opt.restore_global_step:
            vars_to_restore += [global_step]
        saver = tf.train.Saver(vars_to_restore, max_to_keep=opt.max_to_keep)

        saver_best = tf.train.Saver([var for var in tf.model_variables()] + \
                                [global_step],
                                max_to_keep=opt.max_to_keep*2)


        # Session
        sv = tf.train.Supervisor(logdir=opt.checkpoint_dir,
                                 save_summaries_secs=0,
                                 saver=None)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with sv.managed_session(config=config) as sess:
            # print('Trainable variables: ')
            # for var in train_vars:
            #     print(var.name)
            print("parameter_count =", sess.run(parameter_count))

            if opt.init_ckpt_file != None:
                saver.restore(sess, opt.init_ckpt_file)
                cprint('restore from checkpoint {}'.format(opt.init_ckpt_file), 'green', attrs=['bold'])


            start_time = time.time()

            # get file lists
            img_paths = loader.filenames['image']
            cam_paths = loader.filenames['camera']
            mask_paths = loader.filenames['mask']
            idx = np.arange(len(img_paths))
            np.random.shuffle(idx)
            # for step in range(1, opt.max_steps):
            # manually shuffle
            loader.initialize(sess,
                    img_paths=[img_paths[i] for i in idx],
                    cam_paths=[cam_paths[i] for i in idx],
                    mask_paths=[mask_paths[i] for i in idx], is_train=True)
            step = 1
            inner_step = 1

            gs = sess.run(global_step)
            # for step in range(1, opt.max_steps):
            while gs < opt.max_steps:
                try:
                    fetches = {
                        "train": train_op,
                        "global_step": global_step,
                        "incr_global_step": incr_global_step
                    }
                    if step % opt.summary_freq == 0:
                        fetches["loss"] = loss
                        fetches["summary"] = sv.summary_op
                        fetches["mean_depth"] = model.mean_depth
                        fetches["K"] = intrinsics
                        fetches["pose"] = tgt2src_pose_stack

                    results = sess.run(fetches, {is_train_op: True})

                    gs = results["global_step"]

                    if step % opt.summary_freq == 0:
                        sv.summary_writer.add_summary(results["summary"], gs)
                        time_per_iter = (time.time() - start_time) / 100
                        start_time = time.time()
                        print('Iteration: [%7d] | Time: %4.4f ms/step | Loss: %.3f | Mean Depth: %0.3f' \
                              % (gs, time_per_iter*1000.0, results["loss"], results["mean_depth"]))
                        # debug
                        # print('K={}\npose={}\n'.format(results['K'], results['pose']))

                    if step % opt.save_ckpt_freq == 0:
                        saver.save(sess, os.path.join(opt.checkpoint_dir, 'model'), global_step=gs)

                    is_best = False
                    if validator is not None and step % opt.validation_freq == 0:
                        is_best, preds = validation_loop(sess, is_train_op,
                            model.prediction, gs, loader, validator)

                        if is_best:
                            saver_best.save(sess, os.path.join(opt.checkpoint_dir, 'best', 'model'), gs)
                            save_prediction(preds, gs, opt.checkpoint_dir)

                        # raise Exception
                        # switch back to training set
                        loader.initialize(sess,
                                img_paths=[img_paths[i] for i in idx[opt.batch_size*inner_step:]],
                                cam_paths=[cam_paths[i] for i in idx[opt.batch_size*inner_step:]],
                                mask_paths=[mask_paths[i] for i in idx[opt.batch_size*inner_step:]],
                                is_train=True)

                    # proceed
                    step += 1
                    inner_step += 1
                except tf.errors.OutOfRangeError:
                    idx = range(len(img_paths))
                    np.random.shuffle(idx)
                    loader.initialize(sess,
                            img_paths=[img_paths[i] for i in idx],
                            cam_paths=[cam_paths[i] for i in idx],
                            mask_paths=[mask_paths[i] for i in idx], is_train=True)
                    inner_step = 1


def main(_):

    opt.num_source = opt.seq_length - 1
    opt.num_scales = 4

    opt.add_flownet = opt.mode in ['train_flow', 'test_flow']
    opt.add_dispnet = opt.add_flownet and opt.flownet_type == 'residual' \
                      or opt.mode in ['train_rigid', 'test_depth']
    opt.add_posenet = opt.add_flownet and opt.flownet_type == 'residual' \
                      or opt.mode in ['train_rigid', 'test_pose']

    if opt.mode in ['train_rigid', 'train_flow']:
        train()
    elif opt.mode == 'test_depth':
        if opt.use_sigl:
            test_depth_sigl(opt)
        else:
            test_depth(opt)
    elif opt.mode == 'test_pose':
        test_pose(opt)
    elif opt.mode == 'test_flow':
        test_flow(opt)

if __name__ == '__main__':
    tf.app.run()
