import tensorflow as tf
import numpy as np
from termcolor import cprint, colored

EPS = 1e-8

# FOR VOID
flat_categories = ['floor']
vert_categories = ['wall', 'window', 'cabinet']

# FOR VISMA
# flat_categories = ['floor', 'table', 'desk']
# vert_categories = ['wall', 'window', 'cabinet', 'door', 'chair', 'swivelchair']
# categories consist of multiple planes, either flat or vertical
# multi_categories = ['chair', 'swivelchair']
multi_categories = []
total_categories = flat_categories + vert_categories + multi_categories
category2index = {
        'wall': 1,
        'floor': 4,
        'ceiling': 6,
        'window': 9,
        'cabinet': 11,
        'door': 15,
        'table': 16,
        'chair': 20,
        'desk': 34,
        'swivelchair': 76
    }
# weights for category-specific losses
category_weights = {
        'wall': 1,
        'floor': 1,
        'ceiling': 1,
        'window': 1,
        'cabinet': 0.5,
        'door': 1,
        'table': 0.1,
        'chair': 0.5,
        'desk': 0.1,
        'armchair': 0.5,
        'swivelchair': 0.5
}
# sample every 45 degrees
# In spatial frame, gravity is along with z-axis
ref_dirs_45degrees = np.array([[0, 0, 1],
    [1, 0, 0],
    [1, 1, 0],
    [1, -1, 0],
    [0, -1, 0]])

ref_dirs_30degrees = np.array([[0, 0, 1],
    [1, 0, 0],
    [1, 1, 0],
    [1, -1, 0],
    [0, 1, 0],
    [1, 2, 0],
    [2, 1, 0],
    [1, -2, 0],
    [2, -1, 0]])

class SurfaceNormalLayer(object):
    def __init__(self,
                 batch_size=8,
                 height=256,
                 width=512,
                 win_size=8,
                 stride=4):
        """ Constructor of surface normal layer.
        Args:
            win_size: Size of the window from which normals are computed.
            stride: stride of the window from which normals are computed.
        """
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.shape = [self.batch_size, self.height, self.width]
        self.win_size = win_size
        self.stride = stride

    def _convert_depth_to_pointcloud(self, depth, K):
        """ Construct point cloud given the depth map.
        Args:
            depth: BHW1 tensor of depth maps.
            K: B33 tensor of intrinsics
        Returns:
            op to construct point cloud from depth
            tensor of shape BHWC, where C=3 and contains (x, y, z) of each pixel.
        """
        assert depth.shape.as_list() == self.shape + [1]

        Y, X = tf.meshgrid(
            tf.range(self.height, dtype=tf.float32),
            tf.range(self.width, dtype=tf.float32),
            indexing='ij')
        XY = tf.stack([X, Y], axis=-1)
        XY = tf.tile(tf.expand_dims(XY, 0), [self.batch_size, 1, 1, 1])
        XYZ = tf.concat([XY, tf.ones(depth.shape)], axis=-1)

        Kinv = tf.linalg.inv(K)
        XYZ = tf.einsum('bij,bhwj->bhwi', Kinv, XYZ)
        # finally scale by depth
        XYZ = XYZ * depth
        return XYZ

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

    def compute_sigl_loss(self, depth, mask, K, Rcs, refs):
        """ Compute vertical & horizontal plane losses.
        Args:
            depth: BHW1 tensor, metric depth prediction.
            mask: BHW1 tensor, segmentation mask.
            K: B33 tensor of intrinsics
            Rcs: B33 spatial to camera rotation
            refs: Nx3 numpy array, reference directions
        Returns:
            dictionay of category name -> loss tensor
        """
        with tf.variable_scope('sigl_loss'):
            std_ratio = self.projected_std_ratio(depth, K=K, Rcs=Rcs, refs=refs)
            if mask.shape[1:3] != std_ratio.shape[1:3]:
                mask = tf.image.resize_images(mask,
                        size=std_ratio.shape[1:3],
                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            sigl_losses = dict()
            for label in vert_categories:
                bool_mask = tf.equal(mask, category2index[label])
                sigl_losses[label] = category_weights[label] * vertical_plane_loss(std_ratio, mask=bool_mask)

            for label in flat_categories:
                bool_mask = tf.equal(mask, category2index[label])
                sigl_losses[label] = category_weights[label] * horizontal_plane_loss(std_ratio, mask=bool_mask)

            for label in multi_categories:
                bool_mask = tf.equal(mask, category2index[label])
                sigl_losses[label] = category_weights[label] * horizontal_plane_loss(std_ratio, mask=bool_mask)

            return sigl_losses


    def projected_std_ratio(self, metric_depth, K, Rcs, refs):
        """ Compute ratio of standard deviations of 1D projection of 3D points.
        Args:
            metric_depth: rank 4 tensor of metric depth
            K: B33 tensor of intrinsics
            Rcs: B33 tensor of spatial to camera rotation matrix
            refs: reference directions to project the points
        Returns:
            rank-4 tensor of shape NHWR where R is the number of reference directions
        """
        assert type(refs) is np.ndarray, 'refs should be a numpy array'
        assert refs.ndim == 2 and refs.shape[1] == 3, 'refs should have shape [?, 3]'
        R = refs.shape[0]   # number of reference directions

        # normalize reference directions
        refs = refs / np.linalg.norm(refs, axis=1, ord=2)[:, np.newaxis]    # shape: R3
        refs = tf.to_float(refs)
        refs = tf.einsum('bij,rj->bri', Rcs, refs)  # shape: BR3
        # pts has shape BHWN3 where N is the number of 3D points in the window
        pts = self._collect_nearby_points(
            self._convert_depth_to_pointcloud(metric_depth, K))

        # take of local mean of each window
        pts = pts - tf.reduce_mean(pts, axis=3, keep_dims=True) # BHWN3
        # construct reference tensor
        B, H, W, N, THREE = pts.shape.as_list()
        _, imH, imW, _ = metric_depth.shape.as_list()
        assert THREE == 3
        proj_std = []
        for i in range(R):
            # ref = tf.constant(refs[i, :], dtype=tf.float32)
            ref = tf.reshape(refs[:, i, :], [-1, 1, 1, 1, 3])
            proj = tf.reduce_sum(pts * ref, axis=-1)  # BHWN
            proj = proj / N  # average over window
            std_in_ref = tf.sqrt(tf.reduce_mean(proj * proj, axis=-1, keep_dims=True)+EPS)  # BHW1
            proj_std.append(std_in_ref)
        assert len(proj_std) == R
        # now concat the projected std
        proj_std = tf.concat(proj_std, axis=-1)  # BHWR
        # now normalize
        sum_proj_std = tf.reduce_sum(proj_std, axis=-1, keep_dims=True)  # BHW1
        proj_std = tf.check_numerics(
            proj_std * tf.reciprocal(sum_proj_std + EPS),
            colored('std l1 normalization failed', 'red'))
        return proj_std


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
