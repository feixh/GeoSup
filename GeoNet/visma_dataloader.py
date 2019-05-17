""" Dataloader for VISMA-depth formatted dataset.
Assume triplet of images of the following form:
    (image at time t-1, reference image at time t, image at time t+1)
and pickled dictionary object {'gwc': Nx3x4, 'Rg': Nx3x3} and calibration matrix
"""
from __future__ import division
import os
import random
import tensorflow as tf
import numpy as np
import pickle

# for testing
import matplotlib.pyplot as plt
import cv2

USE_AUGMENTATION = True
PY_MASK_TYPE = np.int16
TF_MASK_TYPE = tf.int16

# IMX camera used in the customized sensor rig at UCLA VisionLab
K500x960 = np.array([[486.405, 0, 469.199],
                [0, 535.401, 257.916],
                [0, 0, 1]], dtype=np.float32)

# Intel RealSense D435i Camera
K480x640 = np.array([[514.638, 0, 315.267],
    [0, 518.858, 247.358],
    [0, 0, 1]], dtype=np.float32)

def make_homogeneous(mat3x4):
    """ Make 3x4 pose matrix 4x4.
    """
    return np.concatenate([mat3x4, np.array([[0, 0, 0, 1]], dtype=np.float32)], axis=0)

class DataLoader(object):
    def __init__(self,
                 dataset_dir=None,
                 batch_size=None,
                 img_height=250,
                 img_width=480,
                 num_source=2,
                 num_scales=4,
                 which_camera='imx'):

        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.num_source = num_source
        self.num_scales = num_scales

        self.is_realsense = False
        which_camera = which_camera.lower()
        if which_camera == 'imx':
            self.is_realsense = False
        elif which_camera == 'realsense' or which_camera == 'rs':
            self.is_realsense = True
        else: raise ValueError('which_camera=[imx|realsense|rs]')

        seed = random.randint(0, 2**31 - 1)
        # Load the list of training files into queues
        try:
            self.filenames = self.format_filenames(self.dataset_dir, 'train')
            self.steps_per_epoch = len(self.filenames['image']) // self.batch_size
        except IOError, TypeError:
            self.filenames = None
            self.steps_per_epoch = None

        self.img_path_placeholder = tf.placeholder(tf.string, shape=[None])
        self.cam_path_placeholder = tf.placeholder(tf.string, shape=[None])
        self.mask_path_placeholder = tf.placeholder(tf.string, shape=[None])
        self.is_train = tf.placeholder(tf.bool, shape=())

        self.dataset = tf.data.Dataset.from_tensor_slices((self.img_path_placeholder,
            self.cam_path_placeholder, self.mask_path_placeholder))
        self.dataset = self.dataset \
                .map(self._load_func, num_parallel_calls=8) \
                .map(self._augment_func, num_parallel_calls=8) \
                .batch(self.batch_size) \
                .prefetch(buffer_size=128)
        self.iterator = self.dataset.make_initializable_iterator()
        self.next_element = self.iterator.get_next()
        # target image
        self.next_element[0].set_shape([self.batch_size, self.img_height, self.img_width, 3])
        # source image(s)
        self.next_element[1].set_shape([self.batch_size, self.img_height, self.img_width, 3 * self.num_source])
        # target to source poses
        self.next_element[2].set_shape([self.batch_size, self.num_source, 4, 4])
        # calibration matrix
        self.next_element[3].set_shape([self.batch_size, self.num_scales, 3, 3])
        # segmentation mask of the target image
        self.next_element[4].set_shape([self.batch_size, self.img_height, self.img_width, 1])
        # Rcs for the target image
        self.next_element[5].set_shape([self.batch_size, 3, 3])

    def initialize(self, sess, img_paths, cam_paths=None, mask_paths=None, is_train=True):
        l = self.batch_size * (len(img_paths)//self.batch_size)
        if is_train is False:
            cam_paths = ['dummy'] * l
            mask_paths = ['dummy'] * l

        feed_dict = {
                self.img_path_placeholder: img_paths[:l],
                self.cam_path_placeholder: cam_paths[:l],
                self.mask_path_placeholder: mask_paths[:l],
                self.is_train: is_train}
        sess.run(self.iterator.initializer, feed_dict)


    def _load_func(self, img_path, cam_path, mask_path):
        # Load image sequence
        def _load_image_seq(img_path):
            image_seq = tf.image.decode_jpeg(tf.read_file(img_path))
            tgt_image, src_image_stack = \
                self.unpack_image_sequence(
                    image_seq, self.img_height, self.img_width, self.num_source)
            return tgt_image, src_image_stack

        # At test time, input is single RGB image instead of triplet of images.
        def _load_image(img_path):
            # tgt_image = tf.image.decode_png(tf.read_file(img_path))
            tgt_image = tf.image.decode_jpeg(tf.read_file(img_path))
            tgt_image = tf.image.resize_images(
                    tgt_image,
                    size=[self.img_height, self.img_width],
                    method=tf.image.ResizeMethod.AREA)
            tgt_image = tf.cast(tgt_image, dtype=tf.uint8)
            src_image_stack = tf.zeros(
                    shape=[self.img_height, self.img_width, 3 * self.num_source],
                    dtype=tgt_image.dtype)
            return tgt_image, src_image_stack

        # for test & validation images, we also concatenate them, but only test/validate
        # the reference frame (middele one), so use the same loading function
        tgt_image, src_image_stack = tf.cond(self.is_train,
                lambda: _load_image_seq(img_path),
                lambda: _load_image_seq(img_path))

        # normalize to [0, 1)
        tgt_image = tf.to_float(tgt_image) / 255.0
        src_image_stack = tf.to_float(src_image_stack) / 255.0

        # pick the proper calibration matrix and adjust K when resize
        if self.is_realsense:
            K0 = K480x640
            H0, W0 = 480, 640
        else:
            K0 = K500x960
            H0, W0 = 500, 960

        K = np.stack((K0[0, :] * self.img_width / W0,
            K0[1, :] * self.img_height / H0,
            np.array([0.0, 0.0, 1.0])))
        K = tf.to_float(K)
        K.set_shape([3, 3])

        # Load segmentation masks
        def _load_mask(path):
            def py_load_mask(p):
                # NOTE: for compatibility issues with older versions of tensorflow
                # use int16 here, but the actual type of the mask is uint16
                return np.load(p).astype(PY_MASK_TYPE)
            mask = tf.py_func(py_load_mask, inp=[path], Tout=TF_MASK_TYPE)
            # NOTE: input size is hard-coded, since we use python function
            # tensorflow is not able to infer the size

            if self.is_realsense:
                MASK_HEIGHT, MASK_WIDTH = H0, W0
            else:
                MASK_HEIGHT, MASK_WIDTH = H0//2, W0//2

            mask = tf.reshape(mask, [MASK_HEIGHT, MASK_WIDTH, 1])
            tf.image.resize_images(mask,
                    [self.img_height, self.img_width],
                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            mask = tf.cast(mask, TF_MASK_TYPE)
            return mask

        mask = tf.cond(self.is_train,
                lambda: _load_mask(mask_path),
                lambda: tf.zeros(shape=[self.img_height, self.img_width]+[1], dtype=TF_MASK_TYPE))

        print('mask.shape={}'.format(mask.shape))
        print('tgt_image.shape={}'.format(tgt_image.shape))

        def _load_gravity_rotation(path):
            """
            Map formatted data file path to raw data file path.
            """
            def _py_load_pose(path):
                with open(path, 'rb') as fid:
                    datum = pickle.load(fid)
                # FIXME: not so sure, maybe need to inverse Rg first
                # But the following implementation looks OK by checking the projection
                # of gravity overlaid on image plane.
                tgt2src_poses = []
                tgt_idx = (self.num_source + 1)//2
                g_w_tgt = make_homogeneous(datum['gwc'][tgt_idx, ...])
                for src_idx in range(self.num_source + 1):
                    if src_idx != tgt_idx:
                        g_w_src = make_homogeneous(datum['gwc'][src_idx, ...])
                        tgt2src_poses.append(np.linalg.inv(g_w_src).dot(g_w_tgt))

                Rcs = datum['gwc'][tgt_idx, :3, :3].T.dot(datum['Rg'][tgt_idx, ...])
                tgt2src_poses = np.stack(tgt2src_poses, axis=0)
                return tgt2src_poses.astype(np.float32), Rcs.astype(np.float32)

            tgt2src_poses, Rcs = tf.py_func(_py_load_pose, inp=[path], Tout=[tf.float32, tf.float32])
            tgt2src_poses.set_shape([self.num_source, 4, 4])
            Rcs.set_shape([3, 3])
            return tgt2src_poses, Rcs

        # load Rcs
        tgt2src_poses, Rcs = tf.cond(self.is_train,
                            lambda: _load_gravity_rotation(cam_path),
                            lambda: (tf.eye(4, batch_shape=[self.num_source], dtype=tf.float32), tf.eye(3))
                            )
        return tgt_image, src_image_stack, tgt2src_poses, K, mask, Rcs


    def _augment_func(self, in_tgt_image, in_src_image_stack, in_tgt2src_poses, in_intrinsics, in_mask, Rcs):
        # Data augmentation
        def _real_augment_func(tgt_image, src_image_stack, tgt2src_poses, intrinsics, mask):
            # pretend we have batches ...
            tgt_image = tf.expand_dims(tgt_image, axis=0)
            src_image_stack = tf.expand_dims(src_image_stack, axis=0)
            intrinsics = tf.expand_dims(intrinsics, axis=0)
            mask = tf.expand_dims(mask, axis=0)

            image_all = tf.concat([tgt_image, src_image_stack], axis=3)
            image_all.set_shape([1, self.img_height, self.img_width, 3 * (self.num_source+1)])
            image_all, intrinsics, mask = self.data_augmentation(
                image_all, intrinsics, mask, self.img_height, self.img_width)
            tgt_image = image_all[:, :, :, :3]
            src_image_stack = image_all[:, :, :, 3:]

            intrinsics = self.get_multi_scale_intrinsics(
                intrinsics, self.num_scales)
            # strip off the batch dim
            return (tgt_image[0, ...], src_image_stack[0, ...],
                    tgt2src_poses, intrinsics[0, ...], mask[0, ...], Rcs)

        op = tf.cond(self.is_train,
                lambda: _real_augment_func(in_tgt_image, in_src_image_stack,
                    in_tgt2src_poses, in_intrinsics, in_mask),
                lambda: (in_tgt_image, in_src_image_stack,
                    in_tgt2src_poses, in_intrinsics, in_mask, Rcs))
        return op


    def make_intrinsics_matrix(self, fx, fy, cx, cy):
        # Assumes batch input
        batch_size = fx.get_shape().as_list()[0]
        zeros = tf.zeros_like(fx)
        r1 = tf.stack([fx, zeros, cx], axis=1)
        r2 = tf.stack([zeros, fy, cy], axis=1)
        r3 = tf.constant([0.,0.,1.], shape=[1, 3])
        r3 = tf.tile(r3, [batch_size, 1])
        intrinsics = tf.stack([r1, r2, r3], axis=1)
        return intrinsics

    def data_augmentation(self, im, intrinsics, mask, out_h, out_w):
        # Random scaling
        def random_scaling(im, intrinsics, mask):
            batch_size, in_h, in_w, _ = im.get_shape().as_list()
            scaling = tf.random_uniform([2], 1, 1.15)
            x_scaling = scaling[0]
            y_scaling = scaling[1]
            out_h = tf.cast(in_h * y_scaling, dtype=tf.int32)
            out_w = tf.cast(in_w * x_scaling, dtype=tf.int32)
            im = tf.image.resize_area(im, [out_h, out_w])
            mask = tf.image.resize_nearest_neighbor(mask, [out_h, out_w])
            fx = intrinsics[:,0,0] * x_scaling
            fy = intrinsics[:,1,1] * y_scaling
            cx = intrinsics[:,0,2] * x_scaling
            cy = intrinsics[:,1,2] * y_scaling
            intrinsics = self.make_intrinsics_matrix(fx, fy, cx, cy)
            return im, intrinsics , mask

        # Random cropping
        def random_cropping(im, intrinsics, mask, out_h, out_w):
            # batch_size, in_h, in_w, _ = im.get_shape().as_list()
            batch_size, in_h, in_w, _ = tf.unstack(tf.shape(im))
            offset_y = tf.random_uniform([1], 0, in_h - out_h + 1, dtype=tf.int32)[0]
            offset_x = tf.random_uniform([1], 0, in_w - out_w + 1, dtype=tf.int32)[0]
            im = tf.image.crop_to_bounding_box(
                im, offset_y, offset_x, out_h, out_w)
            mask = tf.image.crop_to_bounding_box(
                mask, offset_y, offset_x, out_h, out_w)
            fx = intrinsics[:,0,0]
            fy = intrinsics[:,1,1]
            cx = intrinsics[:,0,2] - tf.cast(offset_x, dtype=tf.float32)
            cy = intrinsics[:,1,2] - tf.cast(offset_y, dtype=tf.float32)
            intrinsics = self.make_intrinsics_matrix(fx, fy, cx, cy)
            return im, intrinsics, mask

        # Random coloring
        def random_coloring(im):
            print('random coloring got image tensor={}'.format(im))
            batch_size, in_h, in_w, in_c = im.get_shape().as_list()
            # randomly shift gamma
            random_gamma = tf.random_uniform([], 0.8, 1.2)
            im_aug = im ** random_gamma
            # randomly shift brightness
            random_brightness = tf.random_uniform([], 0.5, 1.2)
            im_aug = im_aug * random_brightness
            # randomly shift color
            random_colors = tf.random_uniform([in_c], 0.8, 1.2)
            white = tf.ones([batch_size, in_h, in_w])
            color_image = tf.stack([white * random_colors[i] for i in range(in_c)], axis=3)
            im_aug  *= color_image
            # saturate
            im_aug  = tf.clip_by_value(im_aug,  0, 1)
            return im_aug

        # im, intrinsics, mask = random_scaling(im, intrinsics, mask)
        # im, intrinsics, mask = random_cropping(im, intrinsics, mask, out_h, out_w)
        do_augment  = tf.random_uniform([], 0, 1)
        im = tf.cond(do_augment > 0.5, lambda: random_coloring(im), lambda: im)
        mask = tf.cast(mask, dtype=TF_MASK_TYPE)
        return im, intrinsics, mask

    def format_filenames(self, dataroot, split):
        with open(os.path.join(dataroot, '%s.txt' % split), 'r') as f:
            frames = f.readlines()
        frames = [x.strip() for x in frames]

        # NOTE: in VISMA, the files are put in the same directory,
        # while in VOID, the files are split into different folders.
        all_list = dict()
        all_list['image'] = [os.path.join(dataroot, x) for x in frames]
        all_list['camera'] = [os.path.join(dataroot, str.replace(x, 'rgb', 'pose')[:-4] + '.pkl') for x in frames]
        all_list['mask'] = [os.path.join(dataroot, str.replace(x, 'rgb', 'segmentation')[:-4] + '.npy') for x in frames]
        return all_list

    def unpack_image_sequence(self, image_seq, img_height, img_width, num_source):
        # Assuming the center image is the target frame
        images = tf.split(image_seq, 1+num_source, axis=1)
        # for i in range(len(images)):
        #     print(i, images[i].shape)

        # extract target image
        tgt_index = (num_source+1) // 2 # if num_source = 2
        tgt_image = tf.image.resize_images(
                images[tgt_index],
                [self.img_height, self.img_width],
                method=tf.image.ResizeMethod.AREA)
        # print('tgt shape=', tgt_image.shape)

        # stack source images
        src_images = [tf.image.resize_images(
            x,
            [self.img_height, self.img_width],
            method=tf.image.ResizeMethod.AREA) for x in images[:tgt_index] + images[tgt_index+1:]]

        src_image_stack = tf.concat(src_images, axis=-1)
        # print('src stack shape=', src_image_stack.shape)

        # redudant, but ensure the shape matches
        src_image_stack.set_shape([img_height, img_width, num_source * 3])
        tgt_image.set_shape([img_height, img_width, 3])

        # cast back to uint8 since resizing changes types to float32
        tgt_image = tf.cast(tgt_image, dtype=tf.uint8)
        src_image_stack = tf.cast(src_image_stack, dtype=tf.uint8)

        return tgt_image, src_image_stack

    def get_multi_scale_intrinsics(self, intrinsics, num_scales):
        intrinsics_mscale = []
        # Scale the intrinsics accordingly for each scale
        for s in range(num_scales):
            fx = intrinsics[:,0,0]/(2 ** s)
            fy = intrinsics[:,1,1]/(2 ** s)
            cx = intrinsics[:,0,2]/(2 ** s)
            cy = intrinsics[:,1,2]/(2 ** s)
            intrinsics_mscale.append(
                self.make_intrinsics_matrix(fx, fy, cx, cy))
        intrinsics_mscale = tf.stack(intrinsics_mscale, axis=1)
        return intrinsics_mscale

if __name__ == '__main__':
    tf.reset_default_graph()
    loader = DataLoader(dataset_dir='/local/Data/VOID',
                 batch_size=1,
                 img_height=240,
                 img_width=320,
                 num_source=2,
                 num_scales=4,
                 which_camera='realsense')

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    loader.initialize(sess,
            loader.filenames['image'],
            loader.filenames['camera'],
            loader.filenames['mask'],
            is_train=True)
    while True:
        try:
            tgt_image, src_image_stack, tgt2src_poses, K, mask, Rcs = sess.run(loader.next_element)
            proj = Rcs[0, ...].dot([0, 0, 1])
            disp = tgt_image[0, ...]
            cv2.circle(disp, (240, 120), 5, (0, 255, 0), 2)
            cv2.line(disp, (240, 120), (240+int(proj[0] * 80), 120+int(proj[1]*80)), (255, 0, 0), 2)
            plt.imshow(disp)
            plt.show()
        except tf.errors.OutOfRangeError:
            print('finished')
