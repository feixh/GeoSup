from __future__ import division
import os
import random
import tensorflow as tf
import numpy as np

# for testing
import matplotlib.pyplot as plt

from transformations import rotation_matrix, angle_between_vectors, vector_product


def rotation_from_a_to_b(a, b):
    R = rotation_matrix(angle_between_vectors(a, b), vector_product(a, b))[:3, :3]
    # assert np.linalg.norm(R.dot(a)-b) < 1e-4
    return R.astype(np.float32)


def load_gravity_rotation(filename):
    """
    Load direction of gravity in current camera frame given the path to an image.
    File structure follows default KITTI raw data organization.
    Args:
        filename: full path to the image file.
    Returns:
        3x3 rotation matrix from spatial frame to camera frame
    """
    parts = filename.split('/')
    calib_dir = '/'.join(parts[:-4])
    calib_cam_to_cam = os.path.join(calib_dir, 'calib_cam_to_cam.txt')
    calib_imu_to_velo = os.path.join(calib_dir, 'calib_imu_to_velo.txt')
    with open(calib_imu_to_velo) as fid:
        fid.readline()  # skip header
        Rvb = np.array([float(x) for x in fid.readline()[3:].strip().split(' ')],
                dtype=np.float32).reshape([3, 3])

    calib_velo_to_cam = os.path.join(calib_dir, 'calib_velo_to_cam.txt')
    with open(calib_velo_to_cam) as fid:
        fid.readline()  # skip header
        Rcv = np.array([float(x) for x in fid.readline()[3:].strip().split(' ')],
                dtype=np.float32).reshape([3, 3])

    Rcb = np.einsum('ik,kj->ij', Rcv, Rvb)  # body to camera 00
    cam_id = parts[-3][-3:]     # format: '_0X', X=0, 1, 2, 3
    tag = 'R_rect' + cam_id
    with open(calib_cam_to_cam) as fid:
        for l in fid.readlines():
            if l[:l.find(':')] == tag:
                Rc = np.array(
                        [float(x) for x in l[l.find(':')+1:].strip().split(' ')],
                        dtype=np.float32).reshape([3, 3])
                break
    Rcb = np.einsum('ik,kj->ij', Rc, Rcb) # body frame to proper camera frame

    # now let's get spatial frame to body frame alignment
    oxts_file = os.path.join('/'.join(parts[:-3]),
            'oxts', 'data', parts[-1][:-3]+'txt')
    with open(oxts_file) as fid:
        data = [float(x) for x in fid.readline().strip().split(' ')]
        a_b = np.array(data[11:11+3], dtype=np.float32) # acc in body frame
        a_s = np.array(data[14:14+3], dtype=np.float32) # acc in spatial frame
        Rbs = rotation_from_a_to_b(a=a_s, b=a_b)
    Rcs = np.einsum('ik,kj->ij', Rcb, Rbs)
    # # set gravity as a constant
    # gamma_s = np.array([0, 0, -1])
    # gamma_c = Rcs.dot(gamma_s)
    return Rcs

class DataLoader(object):
    def __init__(self,
                 dataset_dir=None,
                 batch_size=None,
                 img_height=None,
                 img_width=None,
                 num_source=None,
                 num_scales=None):
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.num_source = num_source
        self.num_scales = num_scales


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
        self.next_element[0].set_shape([self.batch_size, self.img_height, self.img_width, 3])
        self.next_element[1].set_shape([self.batch_size, self.img_height, self.img_width, 3 * self.num_source])
        self.next_element[2].set_shape([self.batch_size, self.num_scales, 3, 3])
        self.next_element[3].set_shape([self.batch_size, self.img_height, self.img_width, 1])
        self.next_element[4].set_shape([self.batch_size, 3, 3])     # Rcs only for the target image

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

        def _load_image(img_path):
            tgt_image = tf.image.decode_png(tf.read_file(img_path))
            tgt_image = tf.image.resize_images(
                    tf.expand_dims(tgt_image, axis=0),
                    size=[self.img_height, self.img_width],
                    method=tf.image.ResizeMethod.AREA)
            tgt_image = tf.cast(tgt_image[0, ...], dtype=tf.uint8)
            src_image_stack = tf.zeros(shape=[self.img_height, self.img_width, 3 * self.num_source],
                    dtype=tgt_image.dtype)
            return tgt_image, src_image_stack


        tgt_image, src_image_stack = tf.cond(self.is_train,
                lambda: _load_image_seq(img_path),
                lambda: _load_image(img_path))

        def _load_cam(cam_path):
            # Load camera intrinsics
            rec_def = []
            for i in range(9):
                rec_def.append([1.])
            raw_cam_vec = tf.decode_csv(tf.read_file(cam_path),
                                        record_defaults=rec_def)
            raw_cam_vec = tf.stack(raw_cam_vec)
            intrinsics = tf.reshape(raw_cam_vec, [3, 3])
            return intrinsics

        intrinsics = tf.cond(self.is_train,
                lambda: _load_cam(cam_path),
                lambda: tf.zeros(shape=[3, 3], dtype=tf.float32))


        # Load maskmentation masks
        def _load_mask(mask_path):
            def py_load_mask(path):
                return np.load(path).astype(np.uint8)[1, ..., np.newaxis]

            mask = tf.py_func(
                    py_load_mask,
                    inp=[mask_path],
                    Tout=tf.uint8)
            # NOTE: mask size not necessarily same as image size
            # but should be the exact size of the numpy array
            MASK_HEIGHT = 128
            MASK_WIDTH = 416
            mask.set_shape([MASK_HEIGHT, MASK_WIDTH, 1])
            if (self.img_height, self.img_width) != (MASK_HEIGHT, MASK_WIDTH):
                mask = tf.image.resize_nearest_neighbor(mask, [self.img_height, self.img_width])
            return mask

        mask = tf.cond(self.is_train,
                lambda: _load_mask(mask_path),
                lambda: tf.zeros(shape=[self.img_height, self.img_width]+[1], dtype=tf.uint8))

        print('mask.shape={}'.format(mask.shape))
        print('tgt_image.shape={}'.format(tgt_image.shape))

        def _load_gravity_rotation(img_path):
            """
            Map formatted data file path to raw data file path.
            """
            parts = img_path.split('/')
            drive, frame_id = parts[-2], parts[-1],
            drive, cam_id = drive[:-3], drive[-2:]
            date = drive[:10]
            # FIXME: hardcoded for now, root dir of raw data
            raw_data_root = '/local/Data/NIPS18/kitti_raw_data'
            raw_data_path = os.path.join(raw_data_root,
                    date, drive,
                    'image_' + cam_id, 'data',
                    frame_id[:-3] + 'png')
            return load_gravity_rotation(raw_data_path)

        # load Rcs
        Rcs = tf.cond(self.is_train,
                lambda: tf.py_func(
                    _load_gravity_rotation,
                    inp=[img_path],
                    Tout=tf.float32),
                lambda: tf.eye(3, batch_shape=[self.batch_size], dtype=tf.float32))
        return tgt_image, src_image_stack, intrinsics, mask, Rcs


    def _augment_func(self, in_tgt_image, in_src_image_stack, in_intrinsics, in_mask, Rcs):
        # Data augmentation
        def _real_augment_func(tgt_image, src_image_stack, intrinsics, mask):
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
            return tgt_image[0, ...], src_image_stack[0, ...], intrinsics[0, ...], mask[0, ...], Rcs

        op = tf.cond(self.is_train,
                lambda: _real_augment_func(in_tgt_image, in_src_image_stack, in_intrinsics, in_mask),
                lambda: (in_tgt_image, in_src_image_stack, in_intrinsics, in_mask, Rcs))
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
            im_f = tf.image.convert_image_dtype(im, tf.float32)
            # randomly shift gamma
            random_gamma = tf.random_uniform([], 0.8, 1.2)
            im_aug  = im_f  ** random_gamma
            # randomly shift brightness
            random_brightness = tf.random_uniform([], 0.5, 2.0)
            im_aug  =  im_aug * random_brightness
            # randomly shift color
            random_colors = tf.random_uniform([in_c], 0.8, 1.2)
            white = tf.ones([batch_size, in_h, in_w])
            color_image = tf.stack([white * random_colors[i] for i in range(in_c)], axis=3)
            im_aug  *= color_image
            # saturate
            im_aug  = tf.clip_by_value(im_aug,  0, 1)
            im_aug = tf.image.convert_image_dtype(im_aug, tf.uint8)
            return im_aug

        im, intrinsics, mask = random_scaling(im, intrinsics, mask)
        im, intrinsics, mask = random_cropping(im, intrinsics, mask, out_h, out_w)
        im = tf.cast(im, dtype=tf.uint8)
        do_augment  = tf.random_uniform([], 0, 1)
        im = tf.cond(do_augment > 0.5, lambda: random_coloring(im), lambda: im)
        mask = tf.cast(mask, dtype=tf.uint8)
        return im, intrinsics, mask

    def format_filenames(self, data_root, split):
        with open(data_root + '/%s.txt' % split, 'r') as f:
            frames = f.readlines()
        subfolders = [x.split(' ')[0] for x in frames]
        frame_ids = [x.split(' ')[1][:-1] for x in frames]
        image_filenames = [os.path.join(data_root, subfolders[i],
            frame_ids[i] + '.jpg') for i in range(len(frames))]
        cam_filenames = [os.path.join(data_root, subfolders[i],
            frame_ids[i] + '_cam.txt') for i in range(len(frames))]
        all_list = {}
        all_list['image'] = image_filenames
        all_list['camera'] = cam_filenames
        all_list['mask'] = [x[:-4] + '_labelnew.npy' for x in image_filenames]
        return all_list

    def unpack_image_sequence(self, image_seq, img_height, img_width, num_source):
        # Assuming the center image is the target frame
        tgt_start_idx = int(img_width * (num_source//2))
        tgt_image = tf.slice(image_seq,
                             [0, tgt_start_idx, 0],
                             [-1, img_width, -1])
        # Source frames before the target frame
        src_image_1 = tf.slice(image_seq,
                               [0, 0, 0],
                               [-1, int(img_width * (num_source//2)), -1])
        # Source frames after the target frame
        src_image_2 = tf.slice(image_seq,
                               [0, int(tgt_start_idx + img_width), 0],
                               [-1, int(img_width * (num_source//2)), -1])
        src_image_seq = tf.concat([src_image_1, src_image_2], axis=1)
        # Stack source frames along the color channels (i.e. [H, W, N*3])
        src_image_stack = tf.concat([tf.slice(src_image_seq,
                                    [0, i*img_width, 0],
                                    [-1, img_width, -1])
                                    for i in range(num_source)], axis=2)
        src_image_stack.set_shape([img_height,
                                   img_width,
                                   num_source * 3])
        tgt_image.set_shape([img_height, img_width, 3])
        return tgt_image, src_image_stack

    def batch_unpack_image_sequence(self, image_seq, img_height, img_width, num_source):
        # Assuming the center image is the target frame
        tgt_start_idx = int(img_width * (num_source//2))
        tgt_image = tf.slice(image_seq,
                             [0, 0, tgt_start_idx, 0],
                             [-1, -1, img_width, -1])
        # Source frames before the target frame
        src_image_1 = tf.slice(image_seq,
                               [0, 0, 0, 0],
                               [-1, -1, int(img_width * (num_source//2)), -1])
        # Source frames after the target frame
        src_image_2 = tf.slice(image_seq,
                               [0, 0, int(tgt_start_idx + img_width), 0],
                               [-1, -1, int(img_width * (num_source//2)), -1])
        src_image_seq = tf.concat([src_image_1, src_image_2], axis=2)
        # Stack source frames along the color channels (i.e. [B, H, W, N*3])
        src_image_stack = tf.concat([tf.slice(src_image_seq,
                                    [0, 0, i*img_width, 0],
                                    [-1, -1, img_width, -1])
                                    for i in range(num_source)], axis=3)
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
    loader = DataLoader(dataset_dir='/home/visionlab/Data/kitti_raw_formatted/',
                 batch_size=4,
                 img_height=128,
                 img_width=416,
                 num_source=2,
                 num_scales=4)

    sess = tf.Session()
    loader.initialize(sess,
            loader.filenames['image'],
            loader.filenames['camera'],
            loader.filenames['mask'],
            is_train=True)
    tgt_image, src_image_stack, K, mask, Rcs = sess.run(loader.next_element)
