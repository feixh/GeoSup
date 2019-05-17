from __future__ import division
import os
import time
import math
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

# FIXME: only import used objects
from geonet_nets import *
from utils import *

from sigl_loss_visma import SurfaceNormalLayer, total_categories, ref_dirs_45degrees


class GeoNetModelSIGL(object):

    def __init__(self, opt,
            tgt_image, src_image_stack,
            tgt2src_pose_stack, intrinsics,
            mask, Rcs, is_train):
        """
        Args:
            tgt_image: BHW3 target image, i.e., the referece image in the middle of the 3-frame sequence.
            src_image_stack: BHW(3*N) source images stacked into last dimension.
            tgt2src_pose_stack: BN44 target to source pose, B is batch size, N is number of source frames.
            intrinsics: B33 camera calibration
            mask: BHW1 segmentation mask.
            Rcs: B33 spatial to camera rotation matrix.
            is_train: flag
        """
        self.opt = opt
        self.tgt_image = self.preprocess_image(tgt_image)
        self.src_image_stack = self.preprocess_image(src_image_stack)
        self.tgt2src_pose_stack = tgt2src_pose_stack
        self.intrinsics = intrinsics
        self.mask = mask
        self.Rcs = Rcs
        self.is_train = is_train

        self.build_model()

        if not opt.mode in ['train_rigid', 'train_flow']:
            return

        self.build_losses()
        self.collect_summaries()

    def build_model(self):
        opt = self.opt
        self.tgt_image_pyramid = self.scale_pyramid(self.tgt_image, opt.num_scales)
        self.tgt_image_tile_pyramid = [tf.tile(img, [opt.num_source, 1, 1, 1]) \
                                      for img in self.tgt_image_pyramid]
        # src images concated along batch dimension
        if self.src_image_stack != None:
            self.src_image_concat = tf.concat([self.src_image_stack[:,:,:,3*i:3*(i+1)] \
                                    for i in range(opt.num_source)], axis=0)
            self.src_image_concat_pyramid = self.scale_pyramid(self.src_image_concat, opt.num_scales)

        if opt.add_dispnet:
            self.build_dispnet()

        if opt.add_posenet and not opt.use_slam_pose:
            self.build_posenet()

        if opt.add_dispnet and opt.add_posenet:
            self.build_rigid_flow_warping()

        if opt.add_flownet:
            self.build_flownet()
            if opt.mode == 'train_flow':
                self.build_full_flow_warping()
                if opt.flow_consistency_weight > 0:
                    self.build_flow_consistency()

    def build_dispnet(self):
        opt = self.opt

        # build dispnet_inputs
        if opt.mode == 'test_depth':
            # for test_depth mode we only predict the depth of the target image
            self.dispnet_inputs = self.tgt_image
        else:
            # multiple depth predictions; tgt: disp[:bs,:,:,:] src.i: disp[bs*(i+1):bs*(i+2),:,:,:]
            self.dispnet_inputs = self.tgt_image
            for i in range(opt.num_source):
                self.dispnet_inputs = tf.concat([self.dispnet_inputs, self.src_image_stack[:,:,:,3*i:3*(i+1)]], axis=0)

        # build dispnet
        self.pred_disp = disp_net_visma(opt, self.dispnet_inputs, self.is_train)

        # NOTE: We want to directyly regress the inverse (metric) depth
        # scale normalization is no good.
        # if opt.scale_normalize:
        #     # As proposed in https://arxiv.org/abs/1712.00175, this can
        #     # bring improvement in depth estimation, but not included in our paper.
        #     self.pred_disp = [self.spatial_normalize(disp) for disp in self.pred_disp]

        # NOTE: mean depth of the scene
        self.pred_depth = [1 / d for d in self.pred_disp]
        self.prediction = self.pred_depth[0][:opt.batch_size, :, :, 0] # prediction for validation
        self.mean_depth = tf.reduce_mean(self.prediction)

    def build_posenet(self):
        opt = self.opt

        # build posenet_inputs
        self.posenet_inputs = tf.concat([self.tgt_image, self.src_image_stack], axis=3)

        # build posenet
        self.pred_poses = pose_net(opt, self.posenet_inputs, self.is_train)

    def build_rigid_flow_warping(self):
        opt = self.opt
        bs = opt.batch_size

        # build rigid flow (fwd: tgt->src, bwd: src->tgt)
        self.fwd_rigid_flow_pyramid = []
        self.bwd_rigid_flow_pyramid = []
        for s in range(opt.num_scales):
            for i in range(opt.num_source):
                if not opt.use_slam_pose:
                    fwd_rigid_flow = compute_rigid_flow(tf.squeeze(self.pred_depth[s][:bs], axis=3),
                                     self.pred_poses[:,i,:], self.intrinsics[:,s,:,:], False)
                    bwd_rigid_flow = compute_rigid_flow(tf.squeeze(self.pred_depth[s][bs*(i+1):bs*(i+2)], axis=3),
                                     self.pred_poses[:,i,:], self.intrinsics[:,s,:,:], True)
                else:
                    fwd_rigid_flow = compute_rigid_flow(tf.squeeze(self.pred_depth[s][:bs], axis=3),
                                     self.tgt2src_pose_stack[:,i, ...], self.intrinsics[:,s,:,:], False)
                    bwd_rigid_flow = compute_rigid_flow(tf.squeeze(self.pred_depth[s][bs*(i+1):bs*(i+2)], axis=3),
                                     self.tgt2src_pose_stack[:,i, ...], self.intrinsics[:,s,:,:], True)

                if not i:
                    fwd_rigid_flow_concat = fwd_rigid_flow
                    bwd_rigid_flow_concat = bwd_rigid_flow
                else:
                    fwd_rigid_flow_concat = tf.concat([fwd_rigid_flow_concat, fwd_rigid_flow], axis=0)
                    bwd_rigid_flow_concat = tf.concat([bwd_rigid_flow_concat, bwd_rigid_flow], axis=0)
            self.fwd_rigid_flow_pyramid.append(fwd_rigid_flow_concat)
            self.bwd_rigid_flow_pyramid.append(bwd_rigid_flow_concat)

        # warping by rigid flow
        self.fwd_rigid_warp_pyramid = [flow_warp(self.src_image_concat_pyramid[s], self.fwd_rigid_flow_pyramid[s]) \
                                      for s in range(opt.num_scales)]
        self.bwd_rigid_warp_pyramid = [flow_warp(self.tgt_image_tile_pyramid[s], self.bwd_rigid_flow_pyramid[s]) \
                                      for s in range(opt.num_scales)]

        # compute reconstruction error
        self.fwd_rigid_error_pyramid = [self.image_similarity(self.fwd_rigid_warp_pyramid[s], self.tgt_image_tile_pyramid[s]) \
                                       for s in range(opt.num_scales)]
        self.bwd_rigid_error_pyramid = [self.image_similarity(self.bwd_rigid_warp_pyramid[s], self.src_image_concat_pyramid[s]) \
                                       for s in range(opt.num_scales)]

    def build_flownet(self):
        raise NotImplementedError

    def build_full_flow_warping(self):
        raise NotImplementedError

    def build_flow_consistency(self):
        raise NotImplementedError

    def build_losses(self):
        opt = self.opt
        bs = opt.batch_size
        rigid_warp_loss = 0
        disp_smooth_loss = 0
        flow_warp_loss = 0
        flow_smooth_loss = 0
        flow_consistency_loss = 0

        # for sigl
        self.sigl_loss = {key:[] for key in total_categories}
        for s in range(opt.num_scales):
            # rigid_warp_loss
            if opt.mode == 'train_rigid' and opt.rigid_warp_weight > 0:
                rigid_warp_loss += opt.rigid_warp_weight*opt.num_source/2 * \
                                (tf.reduce_mean(self.fwd_rigid_error_pyramid[s]) + \
                                 tf.reduce_mean(self.bwd_rigid_error_pyramid[s]))

            # disp_smooth_loss
            if opt.mode == 'train_rigid' and opt.disp_smooth_weight > 0:
                disp_smooth_loss += opt.disp_smooth_weight/(2**s) * self.compute_smooth_loss(self.pred_disp[s],
                                tf.concat([self.tgt_image_pyramid[s], self.src_image_concat_pyramid[s]], axis=0))

            # #########################################################
            # sigl
            # #########################################################
            if opt.mode == 'train_rigid' and opt.use_sigl and opt.sigl_loss_weight > 0 and s < opt.sigl_scales:
                with tf.variable_scope('sigl_loss_scale{}'.format(s)):
                    # due to the low resolution of input image, only apply normal loss to part of the pyramid
                    # if s > 0: raise ValueError('sigl loss only support scale level 0')
                    snl = SurfaceNormalLayer(batch_size=opt.batch_size,
                            height=int(opt.img_height*0.5**s), width=int(opt.img_width*0.5**s),
                            win_size=opt.win_size, stride=opt.stride)
                    # FIXME: we actually feed depth instead of disparity
                    # print('disp.shape={}'.format(self.pred_disp[s]))
                    sigl_loss = snl.compute_sigl_loss(depth=self.pred_depth[s][:opt.batch_size],
                            mask=self.mask,
                            K=self.intrinsics[:, s, :, :],
                            Rcs=self.Rcs,
                            refs=ref_dirs_45degrees)
                    for loss_key, loss_op in sigl_loss.items():
                        self.sigl_loss[loss_key].append(loss_op)
            # #########################################################
            # end-of-sigl
            # #########################################################

        regularization_loss = tf.add_n(tf.losses.get_regularization_losses())
        self.total_loss = 0  # regularization_loss
        if opt.mode == 'train_rigid':
            self.total_loss += rigid_warp_loss + disp_smooth_loss

        # compute total sigl
        self.sigl_total = 0
        if opt.mode == 'train_rigid' and opt.use_sigl and opt.sigl_loss_weight > 0:
            for loss_key, loss_ops in self.sigl_loss.items():
                self.sigl_total += tf.add_n(
                        [l * (0.5**i) for i, l in enumerate(loss_ops)])
            self.sigl_total *= opt.sigl_loss_weight
            self.total_loss += self.sigl_total

        # keep track of losses for summary
        self.rigid_warp_loss = rigid_warp_loss
        self.disp_smooth_loss = disp_smooth_loss

    def SSIM(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = slim.avg_pool2d(x, 3, 1, 'SAME')
        mu_y = slim.avg_pool2d(y, 3, 1, 'SAME')

        sigma_x  = slim.avg_pool2d(x ** 2, 3, 1, 'SAME') - mu_x ** 2
        sigma_y  = slim.avg_pool2d(y ** 2, 3, 1, 'SAME') - mu_y ** 2
        sigma_xy = slim.avg_pool2d(x * y , 3, 1, 'SAME') - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

        SSIM = SSIM_n / SSIM_d

        return tf.clip_by_value((1 - SSIM) / 2, 0, 1)

    def image_similarity(self, x, y):
        return self.opt.alpha_recon_image * self.SSIM(x, y) + (1-self.opt.alpha_recon_image) * tf.abs(x-y)

    def L2_norm(self, x, axis=3, keep_dims=True):
        curr_offset = 1e-10
        l2_norm = tf.norm(tf.abs(x) + curr_offset, axis=axis, keep_dims=keep_dims)
        return l2_norm

    def spatial_normalize(self, disp):
        _, curr_h, curr_w, curr_c = disp.get_shape().as_list()
        disp_mean = tf.reduce_mean(disp, axis=[1,2,3], keep_dims=True)
        disp_mean = tf.tile(disp_mean, [1, curr_h, curr_w, curr_c])
        return disp/disp_mean

    def scale_pyramid(self, img, num_scales):
        if img == None:
            return None
        else:
            scaled_imgs = [img]
            _, h, w, _ = img.get_shape().as_list()
            for i in range(num_scales - 1):
                ratio = 2 ** (i + 1)
                nh = int(h / ratio)
                nw = int(w / ratio)
                scaled_imgs.append(tf.image.resize_area(img, [nh, nw]))
            return scaled_imgs

    def gradient_x(self, img):
        gx = img[:,:,:-1,:] - img[:,:,1:,:]
        return gx

    def gradient_y(self, img):
        gy = img[:,:-1,:,:] - img[:,1:,:,:]
        return gy

    def compute_smooth_loss(self, disp, img):
        disp_gradients_x = self.gradient_x(disp)
        disp_gradients_y = self.gradient_y(disp)

        image_gradients_x = self.gradient_x(img)
        image_gradients_y = self.gradient_y(img)

        weights_x = tf.exp(-tf.reduce_mean(tf.abs(image_gradients_x), 3, keep_dims=True))
        weights_y = tf.exp(-tf.reduce_mean(tf.abs(image_gradients_y), 3, keep_dims=True))

        smoothness_x = disp_gradients_x * weights_x
        smoothness_y = disp_gradients_y * weights_y

        return tf.reduce_mean(tf.abs(smoothness_x)) + tf.reduce_mean(tf.abs(smoothness_y))

    def compute_flow_smooth_loss(self, flow, img):
        smoothness = 0
        for i in range(2):
            smoothness += self.compute_smooth_loss(tf.expand_dims(flow[:,:,:,i], -1), img)
        return smoothness/2

    def preprocess_image(self, image):
        # Assuming input image is uint8
        if image == None:
            return None
        else:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            return image * 2. -1.

    def deprocess_image(self, image):
        # Assuming input image is float32
        image = (image + 1.)/2.
        return tf.image.convert_image_dtype(image, dtype=tf.uint8)

    def collect_summaries(self):
        opt = self.opt
        tf.summary.scalar("total_loss", self.total_loss)
        tf.summary.scalar("rigid_warp_loss", self.rigid_warp_loss)
        tf.summary.scalar("disp_smooth_loss", self.disp_smooth_loss)
        tf.summary.scalar("sigl_loss", self.sigl_total)
        for s in range(opt.num_scales):
            tf.summary.histogram("scale%d_depth" % s, self.pred_depth[s][:opt.batch_size])
            tf.summary.image('scale%d_disparity_image' % s,
                    1./self.pred_depth[s][:opt.batch_size],
                    max_outputs=4)
            tf.summary.image('scale%d_target_image' % s,
                    self.deprocess_image(self.tgt_image_pyramid[s]),
                    max_outputs=4)
            if opt.use_sigl and s < opt.sigl_scales:
                # for label in total_categories:
                #     tf.summary.scalar('sigl_{}_'.format(label) + str(s), self.sigl_loss[label][s])
                tf.summary.image('scale%d_mask' % s,
                                tf.image.convert_image_dtype(self.mask, tf.float32),
                                max_outputs=4)
