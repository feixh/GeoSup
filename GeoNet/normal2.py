import tensorflow as tf
import numpy as np
from termcolor import cprint, colored

EPS = 1e-3

class SurfaceNormalLayer(object):
    def __init__(self,
                 batch_size=8,
                 height=256,
                 width=512,
                 win_size=8,
                 stride=4,
                 baseline=0.5327,
                 focal=350.0):
        """ Constructor of surface normal layer.
        Args:
            win_size: Size of the window from which normals are computed.
            stride: stride of the window from which normals are computed.
            baseline: baseline of the stereo head.
            focal: focal length of the camera.
        Comment: Ideally baseline and focal length should match the actual quantities.
            In practice, we only care about the direction of the normals, which should
            be insensitive to these quantities.
        """
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.shape = [self.batch_size, self.height, self.width]
        self.win_size = win_size
        self.stride = stride
        self.baseline = baseline
        self.focal = focal

    def _convert_disparity_to_depth(self, disparity):
        """ Convert disparity map to depth map.
        Args:
            disparity: BHWC tensor of disparity maps.
        Returns:
            op to convert disparity to depth.
        """
        assert disparity.shape.as_list() == self.shape + [1]
        disparity = tf.check_numerics(disparity,
                                      colored('invalid disparity value', 'red'))
        disparity = tf.check_numerics(
            tf.clip_by_value(disparity, 1, 1000.0),
            colored('clip disparity by value failed', 'red'))

        depth = tf.check_numerics(
            tf.reciprocal(disparity), colored('reciprocal failed', 'red'))

        depth = tf.check_numerics(
            self.focal * self.baseline * depth,
            colored('scaled by focal length and baseline failed', 'red'))
        return depth

    def _convert_invdepth_to_depth(self, invdepth):
        invdepth = tf.check_numerics(invdepth,
                                      colored('invalid invdepth value', 'red'))
        # invdepth = tf.check_numerics(
        #     tf.clip_by_value(invdepth, 1, 1000.0),
        #     colored('clip invdepth by value failed', 'red'))
        # depth range: 0.001 - 1
        # actual depth range: 0.1 - 100
        depth = tf.check_numerics(
            tf.reciprocal(invdepth+EPS), colored('reciprocal failed', 'red'))
        # depth = tf.check_numerics(
        #         100.0 * depth,
        #         colored('scaled by focal length and baseline failed', 'red'))

        return depth


    def _convert_depth_to_pointcloud(self, depth):
        """ Construct point cloud given the depth map.
        Args:
            depth: BHWC tensor of depth maps.
        Returns:
            op to construct point cloud from depth
            tensor of shape BHWC, where C=3 and contains (x, y, z) of each pixel.
        """
        assert depth.shape.as_list() == self.shape + [1]
        cy, cx = self.height * 0.5, self.width * 0.5
        Y, X = tf.meshgrid(
            tf.range(self.height, dtype=tf.float32),
            tf.range(self.width, dtype=tf.float32),
            indexing='ij')
        XY = tf.concat(
            [
                tf.expand_dims(tf.subtract(X, cx) / self.focal, axis=-1),
                tf.expand_dims(tf.subtract(Y, cy) / self.focal, axis=-1)
            ],
            axis=-1)
        XY = tf.tile(tf.expand_dims(XY, 0), [self.batch_size, 1, 1, 1]) * depth
        return tf.concat([XY, depth], axis=-1)

    def _collect_nearby_points(self, pts):
        """ Collect nearby points from a small window into the channel dimension.
        Args:
            pts: NHWC, where C=3 and contains [x,y,z] coordinates of points from a window.
        Returns:
            NHW(3K) where K is the window size (win_size*win_size) and the channel is organized in the following order:
                [x1, y1, z1, x2, y2, z2, ... xk, yk, zk]
        """
        patches = tf.extract_image_patches(
            pts,
            ksizes=[1, self.win_size, self.win_size, 1],
            strides=[1, self.stride, self.stride, 1],
            rates=[1, 1, 1, 1],
            padding='VALID')
        # decouple the last dimension of the point cloud tensor (pts)
        # such that the shape becomes N*H*W*K*3 and
        # the last dimension contains each individual 3D point
        return tf.reshape(patches, patches.shape.as_list()[0:3] + [-1, 3])

    def _compute_normal_ls(self, pts):
        """ Compute normal in a neighborhood by solving a least square problem
        Args:
            pts: NHW(3K) formed by function _collect_nearby_points.
                K = win_size^2
        Returns:
            normal: NHW3 where last dimension stores normals [x, y, z]
        """
        in_pts_shape = pts.shape.as_list()
        # center at local mean
        # pts = pts - tf.expand_dims(tf.reduce_mean(pts, axis=3), axis=3)
        # norm = tf.expand_dims(tf.sqrt(tf.reduce_sum(dx * dx, axis=-1)), axis=-1)
        # dx = dx * tf.reciprocal(norm + EPS)
        # form the RHS of linear system x'n=-b, where n is the normal (real^3)
        # and b (scalar) is the distance to origin.
        rhs = tf.stop_gradient(
            tf.ones(
                dtype=tf.float32,
                shape=in_pts_shape[0:3] + [self.win_size * self.win_size, 1]))

        # # manually solve overdetermined linear system
        # A = pts
        # AtA = tf.matmul(A, A, transpose_a=True)
        # AtA_inv = tf.matrix_inverse(AtA)
        # Atb = tf.matmul(A, rhs, transpose_a=True)
        # n = tf.matmul(AtA_inv, Atb)

        # n = tf.matrix_solve_ls(pts, rhs, fast=True, l2_regularizer=100.01)
        n = tf.matrix_solve_ls(pts, rhs, fast=False)

        n = tf.squeeze(n, axis=-1)
        # normalize by using local response normalization layer
        # bias is zero-divigion-guard, radius is 3 to cover all channels
        assert n.shape.as_list()[-1] == 3
        n = tf.nn.lrn(n, depth_radius=3, bias=1e-4)
        return n

    def _compute_covariance(self, pts):
        """ Compute covariance given a point cloud with neighborhood points collected.
        Args:
            pts: NHWK3 tensorf of per pixel local point clouds.
        Returns:
            NHW33 tensor of per pixel covaraince of local point clouds.
        """
        # center at mean
        dx = pts - tf.expand_dims(tf.reduce_mean(pts, axis=3), axis=3)
        # number of samples in the neighborhood
        K = float(pts.shape.as_list()[3])
        return tf.matmul(dx, dx, transpose_a=True) / K

    def _compute_normal_eig(self, pts):
        """ Compute normal given per pixel local point clouds by eigen decompisition.
        Args:
            pts: NHWK3 tensorf of per pixel local point clouds.
        Returns:
            NHW3 tensor of per pixel normal.
        """
        cov = self._compute_covariance(pts)
        _, v = tf.self_adjoint_eig(cov)
        # v has shape NHW33, we need to take the first slice (index 0) of the 4th dimension (the one behind W)
        # which corresponds to the least eigen value
        n = v[:, :, :, 0, :]
        n = tf.nn.lrn(n, depth_radius=3, bias=1e-4)
        return n

    def compute_normal_given_disparity_eig(self, disparity, name):
        with tf.variable_scope(name):
            op_normal = self._compute_normal_eig(
                self._collect_nearby_points(
                    self._convert_depth_to_pointcloud(
                        self._convert_disparity_to_depth(disparity))))
            return op_normal

    def compute_normal_given_disparity_ls(self, disparity, name):
        with tf.variable_scope(name):
            op_normal = self._compute_normal_ls(
                self._collect_nearby_points(
                    self._convert_depth_to_pointcloud(
                        self._convert_disparity_to_depth(disparity))))
            return op_normal

    def cholesky(self, disparity, name):
        with tf.variable_scope(name):
            cov = self._compute_covariance(
                self._collect_nearby_points(
                    self._convert_depth_to_pointcloud(
                        self._convert_disparity_to_depth(disparity))))
            chol = tf.cholesky(cov)
            return chol

    def std_ratio_given_disparity(self, disparity, scaling_factor, name):
        with tf.variable_scope(name):
            cov = self._compute_covariance(
                self._collect_nearby_points(
                    self._convert_depth_to_pointcloud(
                        self._convert_disparity_to_depth(
                            scaling_factor * disparity))))
            std = tf.sqrt(tf.matrix_diag_part(cov))
            return std * tf.reciprocal(
                tf.expand_dims(tf.reduce_sum(std, axis=-1), axis=-1) + EPS)

    def projected_std_ratio(self, disparity, scaling_factor, refs, name, is_depth=False, Rcs=None):
        """ Compute ratio of standard deviations of 1D projection of 3D points.
        Args:
            disparity: predicted disparity map
            scaling_factor: scale disparity maps to match image size if they are normalized
                to range of (0, 1)
            refs: reference directions to project the points
            name: name of the op
        Returns:
            rank-4 tensor of shape NHWR where R is the number of reference directions
        """
        assert type(refs) is np.ndarray, 'refs should be a numpy array'
        assert refs.ndim == 2 and refs.shape[1] == 3, 'refs should have shape [?, 3]'
        R = refs.shape[0]
        if Rcs is None:
            Rcs = tf.eye(3, batch_shape=[1])
        B = Rcs.shape.as_list()[0]  # batch size
        # normalize reference directions
        refs = refs / np.linalg.norm(refs, axis=1, ord=2)[:, np.newaxis]
        refs = tf.einsum('bij,nj->bni', Rcs, tf.to_float(refs)) # BN3
        with tf.variable_scope(name):
            if not is_depth:
                pts = self._collect_nearby_points(
                    self._convert_depth_to_pointcloud(
                        self._convert_disparity_to_depth(
                            scaling_factor * disparity)))
            else:
                pts = self._collect_nearby_points(
                    self._convert_depth_to_pointcloud(
                        scaling_factor * disparity))

            # take of local mean of each window
            pts = pts - tf.expand_dims(tf.reduce_mean(pts, axis=3), axis=3)
            # construct reference tensor
            N, H, W, K, _ = pts.shape.as_list()
            proj_std = []
            for i in range(R):
                # ref = tf.constant(refs[i, :], dtype=tf.float32)
                ref = refs[:, i, :]
                ref = tf.reshape(ref, [B, 1, 1, 1, 3])
                proj = tf.reduce_sum(pts * ref, axis=-1)  # NHWK
                proj = proj / K  # average over window
                std_in_ref = tf.sqrt(tf.reduce_mean(proj * proj, axis=-1)+1e-8)  # NHW
                # std_in_ref = tf.clip_by_value(std_in_ref, 1e-7, 1)
                std_in_ref = tf.expand_dims(std_in_ref, axis=-1)  # NHW1
                proj_std.append(std_in_ref)
            assert len(proj_std) == R
            # now concat the projected std
            proj_std = tf.concat(proj_std, axis=-1)  # NHWR
            # now normalize
            sum_proj_std = tf.reduce_sum(proj_std, axis=-1)  # NHW
            sum_proj_std = tf.expand_dims(sum_proj_std, axis=-1)  # NHW1
            proj_std = tf.check_numerics(
                proj_std * tf.reciprocal(sum_proj_std + EPS),
                colored('std l1 normalization failed', 'red'))
            return proj_std


def orthogonal_normal_loss(normal, ref, mask=None):
    """ Penalize normals NOT orthogonal to the reference direction.
    Args:
        normal: NHW3 tensor of normals.
        ref: reference direction, unit 3-dim vector.
        mask: specify where to impose loss.
    Returns: loss op
    Comment: If normals are orthogonal to reference, loss should be zero.
        Otherwise, it's non-negative (by construction, sqr) and should be minimized.
    """
    if mask is not None:
        assert len(mask.shape) == 4
        if mask.shape.as_list()[0:3] != normal.shape.as_list()[0:3]:
            uint8_mask = tf.cast(mask, dtype=tf.uint8)
            uint8_mask = tf.image.resize_images(
                uint8_mask,
                size=normal.shape[1:3],
                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            mask = tf.cast(uint8_mask, dtype=tf.bool)

        mask = tf.squeeze(mask, axis=-1)
        total = tf.reduce_sum(tf.cast(mask, dtype=tf.float32))
        normal = tf.boolean_mask(normal, mask)
        ref = tf.reshape(tf.constant(ref, dtype=tf.float32), [1, 3])
        sqr_cos_dist = tf.reduce_sum(tf.square(tf.multiply(normal, ref)))
        return sqr_cos_dist * tf.reciprocal(total + EPS)
    else:
        ref = tf.reshape(tf.constant(ref, dtype=tf.float32), [1, 1, 1, 3])
        return tf.reduce_mean(tf.multiply(normal, ref))


def parallel_normal_loss(normal, ref, mask=None):
    """ Penalize normals NOT parappel to the reference direction.
    Args:
        normal: NHW3 tensor of normals.
        ref: reference direction, unit 3-dim vector.
        mask: speciy where to impose loss.
    Returns: loss op
    Comment: If normals are consistent with the reference direction,
        their square dot product should be close to 1. Loss should thus be
        1 - SquaredDotProduct
    """
    return 1 - orthogonal_normal_loss(normal, ref, mask)


def horizontal_plane_loss(std_ratio, weight=None, mask=None):
    """ Horizontal plane loss is essentially the std along direction of gravity (0, 1, 0).
    Args:
        std_ratio: rank-4 tensor of shape NHWR where R is the number of directions along which
            std is computed. The 1st direction should be gravity direction (0, 1, 0).
        weight: object bounary aware weights.
        mask: where to look at.
    Returns:
        scalar tensor of average std at pixels specified by mask.
    """
    if weight is not None:
        assert len(weight.shape) == 4
        assert weight.shape[0:3] == std_ratio.shape[0:3]
        std_ratio = tf.einsum('bhwc,bhwc->bhwc', std_ratio, weight)

    if mask is not None:
        assert len(mask.shape) == 4
        assert mask.shape[0:3] == std_ratio.shape[0:3]
        mask = tf.squeeze(mask, axis=-1)
        total = tf.reduce_sum(tf.cast(mask, dtype=tf.float32))
        std_sum = tf.reduce_sum(tf.boolean_mask(std_ratio[..., 0], mask))
        return std_sum / (total + EPS)
    else:
        # gravity is along y direction, so std along y direction should be minimized
        return tf.reduce_mean(std_ratio[..., 0])


def vertical_plane_loss(std_ratio, weight=None, mask=None):
    """ Vertical plane loss is essentially the std along samples from the null space of
        gravity (0, 1, 0).
    Args:
        std_ratio: rank-4 tensor of shape NHWR where R is the number of sample directions along which
            std is computed. The 1st direction should be gravity direction (0, 1, 0) to which the rest directions are orthogonal.
        weight: object boundary aware weights.
        mask: where to look at.
    Returns:
        scalar tensor of average std at pixels specified by mask.
    """
    if weight is not None:
        assert len(weight.shape) == 4
        assert weight.shape[0:3] == std_ratio.shape[0:3]
        std_ratio = tf.einsum('bhwc,bhwc->bhwc', std_ratio, weight)

    if mask is not None:
        assert len(mask.shape) == 4
        assert mask.shape[0:3] == std_ratio.shape[0:3]
        mask = tf.squeeze(mask, axis=-1)
        total = tf.reduce_sum(tf.cast(mask, dtype=tf.float32))
        min_std_in_xz_plane = tf.reduce_min(std_ratio[..., 1:], axis=-1)  # NHW
        std_sum = tf.reduce_sum(tf.boolean_mask(min_std_in_xz_plane, mask))
        # return tf.reduce_sum(xz) * tf.reciprocal(xz.shape(0) + EPS)
        return std_sum / (total + EPS)
    else:
        # gravity is along y direction, so std along y direction should be minimized
        min_std_in_xz_plane = tf.reduce_min(std_ratio[..., 1:], axis=-1)  # NHW
        return tf.reduce_mean(min_std_in_xz_plane)
