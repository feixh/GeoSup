import numpy as np
import cv2
from kitti_eval import depth_evaluation_utils as eval_utils
import os


def log(s, filepath=None, to_console=True):
    if to_console:
        print(s)

    if filepath is not None:
        try:
            with open(filepath, "a+") as o:
                o.write(s+'\n')
        except:
            raise IOError


class Validator(object):
    def __init__(self, val_dir, val_list, match_scale=False, log_path=None, key_metric='abs_rel', use_interp=False):
        """
        Args:
            val_dist: Root directory of the validation data.
            val_list: A list of filenames used for validation.
            match_scale: If trained with monocular video, need to set true to match scale of gt and pred.
            log_path: File to log results.
        """
        val_files = eval_utils.read_text_lines(val_list)
        self.filenames = [os.path.join(val_dir, p.split(' ')[0]) for p in val_files]
        self.gt, self.calib, self.sizes, _, self.cams = eval_utils.read_file_data(val_files, val_dir)
        self.load_gt_depth(use_interp)
        print('ground truth loaded')
        self.best_value = 10000
        self.best_step = -1
        self.log_path = log_path
        self.match_scale = match_scale
        if key_metric in ['log_rms', 'abs_rel']:
            self.key_metric = key_metric
        else: raise ValueError('key_metric =[log_rms|abs_rel]')

    def load_gt_depth(self, use_interp=False):
        raw_gt_depths = []
        interp_gt_depths = []
        num_items = len(self.gt)
        for i, (calib, gt, sizes, cam_id) in enumerate(zip(self.calib, self.gt, self.sizes, self.cams)):
            # print('process gt depth {}/{}'.format(i, len(self.gt)))
            depth_interp = None
            if use_interp:
                depth, depth_interp = eval_utils.generate_depth_map(calib, gt, sizes, cam_id, True, True)
                print('interpolating depth {}/{}'.format(i, num_items))
            else:
                depth = eval_utils.generate_depth_map(calib, gt, sizes, cam_id, False, True)
            raw_gt_depths.append(depth.astype(np.float32))
            interp_gt_depths.append(depth_interp)
        self.raw_gt_depths = raw_gt_depths
        self.interp_gt_depths = interp_gt_depths

    def prepare(self, preds, min_depth=1e-3, max_depth=80, mode='disparity', verbose=False):
        """ Prepare depth prediction (convert to depth if necessary), ground truth and mask.
        Args:
            pred: input prediction
            mode: treat the pred as mode
        Returns:
            scale matched and capped prediction, masks
            as numpy arrays
        """
        assert mode in ['depth', 'invdepth', 'disparity']

        # create alias
        gt_depths = self.raw_gt_depths
        pred_depths = []
        masks = []
        # convert disparity/invdepth to depth
        num_items = len(self.gt)
        for i, (pred, calib, gt, sizes, cam_id) in enumerate(
                zip(list(preds), self.calib, self.gt, self.sizes, self.cams)):
            if verbose: print('preparing prediction {}/{}'.format(i, num_items))
            disp_pred = cv2.resize(pred, (sizes[1], sizes[0]),
                                   interpolation=cv2.INTER_LINEAR)
            if mode == 'disparity':
                # stereo case, need baseline and focal length
                disp_pred = disp_pred * disp_pred.shape[1]
                # need to convert from disparity to depth
                focal_length, baseline = eval_utils.get_focal_length_baseline(calib, cam_id)
                depth_pred = baseline * focal_length / disp_pred
            elif mode == 'invdepth':
                depth_pred = 1.0 / disp_pred
            elif mode == 'depth':
                depth_pred = disp_pred
            else: raise ValueError('unrecognized mode')
            depth_pred[np.isinf(depth_pred)] = 0
            pred_depths.append(depth_pred)

        # crop and scale matching
        for i, (gt_depth, pred_depth) in enumerate(zip(gt_depths, pred_depths)):
            # compute mask
            mask = np.logical_and(gt_depth > min_depth, gt_depth < max_depth)
            crop = self.get_crop(
                    height=gt_depth.shape[0],
                    width=gt_depth.shape[1],
                    scheme='eigen')
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1],crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)
            # Scale matching
            if self.match_scale:
                scalar = np.median(gt_depth[mask])/np.median(pred_depth[mask])
                pred_depth *= scalar
            # cap
            pred_depth[pred_depth < min_depth] = min_depth
            pred_depth[pred_depth > max_depth] = max_depth
            # save mask
            masks.append(mask)
            pred_depths[i] = pred_depth

        return pred_depths, masks


    def get_crop(self, height, width, scheme='eigen'):
        """ Get the cropping scheme.
        """
        if scheme == 'eigen':
            crop = np.array([0.40810811*height, 0.99189189*height,
                      0.03594771*width, 0.96405229*width]).astype(np.int32)
            return crop
        elif scheme == 'garg':
            raise NotImplementedError('garg crop not implemented')
        else:
            raise ValueError('scheme=[eigen|garg]')


    def validate(self, preds, step, min_depth=1e-3, max_depth=80, mode='disparity'):
        """ Compute error and accuracy metrics.
        Args:
            preds: prediction in the form of an NxHxWx1 numpy array.
            min_depth, max_depth: Min & Max depth to cap.
            mode: mode=[depth, invdepth, disparity]
        Returns:
            whether or not the current prediction is better than the previous best result.
        """
        pred_depths, masks = self.prepare(preds, min_depth, max_depth, mode)
        gt_depths = np.copy(self.raw_gt_depths)
        n_sample = gt_depths.shape[0]

        rms     = np.zeros(n_sample, np.float32)
        log_rms = np.zeros(n_sample, np.float32)
        abs_rel = np.zeros(n_sample, np.float32)
        sq_rel  = np.zeros(n_sample, np.float32)
        d1_all  = np.zeros(n_sample, np.float32)
        a1      = np.zeros(n_sample, np.float32)
        a2      = np.zeros(n_sample, np.float32)
        a3      = np.zeros(n_sample, np.float32)

        for i in range(n_sample):
            gt_depth = gt_depths[i]
            pred_depth = pred_depths[i]
            mask = masks[i]

            (abs_rel[i], sq_rel[i], rms[i], log_rms[i],
                    a1[i], a2[i], a3[i]) = eval_utils.compute_errors(
                            gt_depth[mask], pred_depth[mask])

        # pick metric to check
        if self.key_metric == 'log_rms':
            curr_value = log_rms.mean()
        elif self.key_metric == 'abs_rel':
            curr_value = abs_rel.mean()

        better_than_best = False
        if self.best_value > curr_value:
            self.best_step = step
            self.best_value = curr_value
            better_than_best = True

        log('Current(step)=%d Local Minima(step)=%d  AbsRel=%.4f  '
                % (step, self.best_step, self.best_value), self.log_path)
        log('abs_rel    sq_rel    rms    log_rms    d1_all    a1    a2    a3', self.log_path)
        log('%.4f    %.4f    %.4f    %.4f    %.4f    %.4f    %.4f    %.4f    '
                % (abs_rel.mean(), sq_rel.mean(), rms.mean(), log_rms.mean(),
                    d1_all.mean(), a1.mean(), a2.mean(), a3.mean()), self.log_path)

        return better_than_best


if __name__ == '__main__':

    validator = Validator(val_dir='/home/visionlab/Data/kitti_raw_test',
            val_list='monodepth/filenames/eigen_test_files.txt')
    dummy = np.random.random((697, 128, 416, 1))
    validator.validate(dummy, 0)
